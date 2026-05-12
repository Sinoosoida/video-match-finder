from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rich.console import Console
from tqdm import tqdm

from vmf import align, frames, hough, weights
from vmf.async_index import AsyncIndexer
from vmf.config import VIDEO_EXTS, Config
from vmf.features import FeatureExtractor, load_extractor, load_remote_extractor
from vmf.index import Store, file_sha1

console = Console(stderr=True)


@dataclass
class PairResult:
    a_id: int
    b_id: int
    a_path: str
    b_path: str
    segments: list[align.Segment]


def discover_videos(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        if r.is_file() and r.suffix.lower() in VIDEO_EXTS:
            out.append(r.resolve())
        elif r.is_dir():
            for p in sorted(r.rglob("*")):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    out.append(p.resolve())
    return sorted(set(out))


def _decode_and_submit(
    path: Path, cfg: Config, info: "frames.VideoInfo",
    indexer, video_id: int,
) -> int:
    """Stream batches from ffmpeg directly into the encoder pool.

    Does NOT buffer the whole video in RAM — each batch is submitted as soon
    as it fills up, then immediately discarded from the caller. RAM stays at
    O(encode_inflight × batch_size × 224²×3) regardless of video length.

    cropdetect runs inline; if the first pass detected real letterbox we
    discard the video from the indexer and re-decode with the crop applied.
    The 1% re-decode case costs a second disk read for that file only.
    """
    inline_crop_mode = cfg.keyframes_only and getattr(cfg, "cropdetect", True)
    legacy_crop = (
        frames.detect_crop(path)
        if (not cfg.keyframes_only and getattr(cfg, "cropdetect", True))
        else None
    )
    crops_seen: list[tuple[int, int, int, int]] = []

    def _stream_pass(crop_filter: str | None, collect_crops: bool) -> int:
        n = 0
        batch_imgs: list[np.ndarray] = []
        batch_ts: list[float] = []

        def flush() -> int:
            nonlocal batch_imgs, batch_ts
            if not batch_imgs:
                return 0
            arr = np.stack(batch_imgs, axis=0)
            ts = np.asarray(batch_ts, dtype=np.float32)
            indexer.submit_batch(video_id, arr, ts, 0)
            count = 1
            if cfg.mirror:
                arr_m = arr[:, :, ::-1, :].copy()
                indexer.submit_batch(video_id, arr_m, ts, 1)
                count += 1
            batch_imgs, batch_ts = [], []
            return count

        try:
            if cfg.keyframes_only:
                it = frames.iter_keyframes(
                    path, size=cfg.frame_size, crop=crop_filter,
                    min_std=cfg.min_frame_std, hwaccel=cfg.hwaccel,
                    crops_out=crops_seen if collect_crops else None,
                )
            else:
                it = frames.iter_frames(
                    path, fps=cfg.fps, size=cfg.frame_size, crop=crop_filter,
                    min_std=cfg.min_frame_std, hwaccel=cfg.hwaccel,
                )
            for t, img in it:
                batch_imgs.append(img)
                batch_ts.append(t)
                if len(batch_imgs) >= cfg.batch_size:
                    n += flush()
            n += flush()
        except frames.FFmpegError as e:
            console.print(f"[red]skip[/red] {path.name}: {e}")
            return -1
        return n

    n_submitted = _stream_pass(legacy_crop, inline_crop_mode)
    if n_submitted < 0:
        return -1

    if inline_crop_mode and crops_seen:
        agg = frames.aggregate_crop(crops_seen, info.width, info.height)
        if agg and frames.is_significant_crop(agg, info.width, info.height):
            crop_str = f"crop={agg[0]}:{agg[1]}:{agg[2]}:{agg[3]}"
            console.print(f"[dim]re-decode with {crop_str} → {path.name[:60]}[/dim]")
            indexer.discard(video_id)        # invalidates first-pass batches
            indexer.register(video_id)
            crops_seen.clear()
            n_submitted = _stream_pass(crop_str, False)
            if n_submitted < 0:
                return -1

    return n_submitted


def index_paths(paths: list[Path], cfg: Config, store: Store, fe: FeatureExtractor) -> int:
    """Async pipeline: serial ffmpeg, parallel encoder/network.

    A single ffmpeg runs at any given time (avoids HDD thrashing); as soon as
    one video's frames are gathered we kick off the next ffmpeg, while the
    previous video's batches keep travelling through the encoder pool. The
    encoder semaphore enforces back-pressure so RAM doesn't grow without
    bound when the network is the bottleneck.
    """
    indexer = AsyncIndexer(store, fe, cfg)
    added = 0
    pbar = tqdm(paths, desc="Indexing", unit="vid")
    for path in pbar:
        try:
            sha1 = file_sha1(path)
            if store.has_video(path, sha1):
                continue
            try:
                info = frames.probe(path)
            except frames.FFmpegError as e:
                console.print(f"[red]skip[/red] {path.name}: {e}")
                continue

            with indexer.store_lock:
                store.init_index(fe.dim)
                video_id = store.add_video(
                    path, sha1, info.duration, info.width, info.height, 0,
                )
            indexer.register(video_id)

            n_submitted = _decode_and_submit(path, cfg, info, indexer, video_id)
            if n_submitted < 0:
                indexer.discard(video_id)
                with indexer.store_lock:
                    store.mark_failed(video_id)
                continue
            indexer.mark_decoded(video_id, n_submitted)
            added += 1
        except Exception as e:
            console.print(f"[red]unexpected error[/red] {path.name}: {e}")
            continue

    indexer.wait_all()
    return added


def _detect_watermark_ids(idx_mat: np.ndarray, meta: np.ndarray, max_videos: int) -> set[int]:
    """A vector matching too many distinct videos is a logo/intro/watermark — drop it."""
    if max_videos <= 0:
        return set()
    n = idx_mat.shape[0]
    bad: set[int] = set()
    for src in range(n):
        seen: set[int] = set()
        src_vid = int(meta["video_id"][src])
        for k in range(idx_mat.shape[1]):
            tgt = int(idx_mat[src, k])
            if tgt < 0 or tgt == src:
                continue
            tv = int(meta["video_id"][tgt])
            if tv != src_vid:
                seen.add(tv)
            if len(seen) > max_videos:
                bad.add(src)
                break
    return bad


def _filter_segments(segs: list[align.Segment], cfg: Config) -> list[align.Segment]:
    out: list[align.Segment] = []
    for s in segs:
        span = max(s.a_end - s.a_start, s.b_end - s.b_start)
        if s.score < cfg.min_segment_score:
            continue
        if span < cfg.min_segment_seconds:
            continue
        if s.inliers / max(span, 1e-3) < cfg.min_inlier_density:
            continue
        out.append(s)
    return out


def _build_mutual_kset(idx_mat: np.ndarray) -> set[tuple[int, int]]:
    """Return the set of (src, tgt) frame pairs that are mutual top-K neighbors.

    A real near-duplicate pair shows up symmetrically in each other's neighbor lists.
    Spurious genre matches usually fail this — a 'popular' background frame matches
    many but isn't matched back by them at high rank.
    """
    n, k = idx_mat.shape
    mutual: set[tuple[int, int]] = set()
    neigh = [set(idx_mat[i, j] for j in range(k) if idx_mat[i, j] >= 0) for i in range(n)]
    for i in range(n):
        for tgt in neigh[i]:
            if tgt == i or tgt < 0 or tgt >= n:
                continue
            if i in neigh[tgt]:
                mutual.add((i, int(tgt)))
    return mutual


def find_pairs(cfg: Config, store: Store) -> list[PairResult]:
    if store.index is None or store.index.ntotal == 0:
        return []

    n = store.index.ntotal
    all_vecs = np.vstack([store.index.reconstruct(i) for i in range(n)]).astype(np.float32)
    sims_mat, idx_mat = store.search(all_vecs, cfg.knn + 1)
    meta = store.frame_meta

    bad_vecs = _detect_watermark_ids(idx_mat, meta, cfg.watermark_max_videos)
    if bad_vecs:
        console.print(f"[dim]Dropped {len(bad_vecs)} watermark/intro vectors.[/dim]")

    mutual = _build_mutual_kset(idx_mat) if cfg.mutual_knn else None

    pair_points: dict[tuple[int, int], list[tuple[float, float, float, int]]] = defaultdict(list)
    for src in range(n):
        if src in bad_vecs:
            continue
        src_vid = int(meta["video_id"][src])
        src_ts = float(meta["ts"][src])
        src_mir = int(meta["mirrored"][src])
        for k in range(idx_mat.shape[1]):
            tgt = int(idx_mat[src, k])
            if tgt < 0 or tgt == src or tgt in bad_vecs:
                continue
            if mutual is not None and (src, tgt) not in mutual:
                continue
            tgt_vid = int(meta["video_id"][tgt])
            if tgt_vid == src_vid:
                continue
            sim = float(sims_mat[src, k])
            if sim < cfg.min_match_sim:
                continue
            tgt_ts = float(meta["ts"][tgt])
            tgt_mir = int(meta["mirrored"][tgt])
            a, b = (src_vid, tgt_vid) if src_vid < tgt_vid else (tgt_vid, src_vid)
            ta, tb = (src_ts, tgt_ts) if src_vid < tgt_vid else (tgt_ts, src_ts)
            mirror = src_mir ^ tgt_mir
            pair_points[(a, b)].append((ta, tb, sim, mirror))

    results: list[PairResult] = []
    for (a, b), pts in pair_points.items():
        if len(pts) < cfg.min_pair_matches:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        segs = align.find_segments(
            arr[:, :2], arr[:, 2], arr[:, 3].astype(bool),
            iters=cfg.ransac_iters, tol=cfg.ransac_tol,
            min_inliers=max(6, cfg.min_pair_matches // 2),
            min_seconds=cfg.min_segment_seconds,
            slope_min=cfg.slope_min, slope_max=cfg.slope_max,
        )
        segs = _filter_segments(segs, cfg)
        if not segs:
            continue
        a_path = store.video_path(a) or "?"
        b_path = store.video_path(b) or "?"
        results.append(PairResult(a, b, a_path, b_path, segs))

    results.sort(key=lambda r: max(s.score * (s.a_end - s.a_start) for s in r.segments), reverse=True)
    return results


def _candidate_pairs_from_knn(
    cfg: Config, store: Store, sims_mat: np.ndarray, idx_mat: np.ndarray,
) -> list[tuple[int, int]]:
    """Cheap pre-filter: a pair of videos becomes a candidate only if at least
    `smooth_min_kmatches` cross-video kNN matches exist between them.
    No hardcoded scoring — just enough to skip clearly-disjoint pairs."""
    meta = store.frame_meta
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for src in range(idx_mat.shape[0]):
        sv = int(meta["video_id"][src])
        for k in range(idx_mat.shape[1]):
            tgt = int(idx_mat[src, k])
            if tgt < 0 or tgt == src:
                continue
            tv = int(meta["video_id"][tgt])
            if tv == sv:
                continue
            key = (sv, tv) if sv < tv else (tv, sv)
            counts[key] += 1
    return [pair for pair, n in counts.items() if n >= cfg.smooth_min_kmatches]


def find_pairs_smooth(cfg: Config, store: Store) -> list[PairResult]:
    """Smooth (Hough + permutation) pipeline.

    Pipeline per pair of videos:
      1. Build W[i,j] = max(0, sim)^p · idf_A·idf_B / (ρ_A·ρ_B).
      2. Importance-sampled Hough over (α, β) ⇒ peak L*, line (α*, β*).
      3. Permutation test (shuffle t_B) ⇒ p-value.
      4. Accept if p < smooth_max_pvalue.
      5. Extent extracted from raw similarity along the accepted line.
    """
    if store.index is None or store.index.ntotal == 0:
        return []

    n = store.index.ntotal
    all_vecs = np.vstack([store.index.reconstruct(i) for i in range(n)]).astype(np.float32)
    meta = store.frame_meta

    # Per-frame weights (cached on disk by exponent p)
    rho, idf = weights.ensure_weights(store, p=cfg.smooth_p, k=cfg.smooth_idf_k)

    # Pre-filter via raw kNN (just to skip clearly-disjoint pairs)
    sims_mat, idx_mat = store.search(all_vecs, cfg.knn + 1)
    candidates = _candidate_pairs_from_knn(cfg, store, sims_mat, idx_mat)
    console.print(f"[dim]{len(candidates)} candidate pair(s) to score with Hough+permutation…[/dim]")

    # Indices per video for quick slicing
    by_video: dict[int, np.ndarray] = {}
    for vid in np.unique(meta["video_id"]):
        by_video[int(vid)] = np.where(meta["video_id"] == vid)[0]

    p_exp = float(cfg.smooth_p)
    results: list[PairResult] = []

    # Cap matrix size so very long pairs don't dominate cost. We downsample by
    # taking every k-th frame; this only loses temporal resolution, not signal,
    # because the diagonal of a real duplicate is dense.
    MAX_FRAMES = cfg.smooth_max_frames

    for (a, b) in tqdm(candidates, desc="Hough+permutation", unit="pair", leave=False):
        ai, bi = by_video[a], by_video[b]
        if len(ai) < 4 or len(bi) < 4:
            continue
        if len(ai) > MAX_FRAMES:
            ai = ai[np.linspace(0, len(ai) - 1, MAX_FRAMES).astype(np.int64)]
        if len(bi) > MAX_FRAMES:
            bi = bi[np.linspace(0, len(bi) - 1, MAX_FRAMES).astype(np.int64)]
        vA, vB = all_vecs[ai], all_vecs[bi]
        tA = meta["ts"][ai].astype(np.float32)
        tB = meta["ts"][bi].astype(np.float32)

        sim_raw = np.clip(vA @ vB.T, 0.0, 1.0)
        W = sim_raw ** p_exp
        if cfg.smooth_use_idf:
            W = W * np.outer(idf[ai], idf[bi])
        if cfg.smooth_use_rho:
            W = W / (np.outer(rho[ai], rho[bi]) + 1e-6)

        rng = np.random.default_rng(((a + 1) * 0x9E3779B97F4A7C15 + b) & 0xFFFFFFFF)
        hr = hough.hough_permutation(
            tA, tB, W, tau=cfg.smooth_tau,
            n_samples=cfg.smooth_n_samples, n_perm=cfg.smooth_n_perm,
            alpha_range=(cfg.slope_min, cfg.slope_max), rng=rng,
            n_alpha_bins=20,
        )
        # z-score is the primary acceptance criterion. p-value is kept as a
        # weaker sanity gate (anything significant by z is significant by p too).
        if (hr is None
                or hr.z_score < cfg.smooth_min_zscore
                or hr.pvalue > cfg.smooth_max_pvalue):
            continue

        ext = hough.extract_extent(tA, tB, sim_raw, hr.alpha, hr.beta,
                                   tau=cfg.smooth_tau, sim_floor=cfg.smooth_min_tube_sim)
        if ext is None:
            continue
        span = max(ext["a_end"] - ext["a_start"], ext["b_end"] - ext["b_start"])
        if span < cfg.min_segment_seconds:
            continue
        # Two-stage acceptance: statistical (z) gates "is this random?",
        # semantic (mean_sim, density) gates "are the matches actually strong?".
        # Both must pass.
        if ext["mean_sim"] < cfg.min_segment_score:
            continue
        density = ext["n_supporting"] / max(span, 1.0)
        if density < cfg.min_inlier_density:
            continue

        seg = align.Segment(
            a_start=ext["a_start"], a_end=ext["a_end"],
            b_start=ext["b_start"], b_end=ext["b_end"],
            speed_ratio=hr.alpha, mirrored=False,
            inliers=ext["n_supporting"], score=ext["mean_sim"],
            pvalue=hr.pvalue, weighted_support=hr.support, z_score=hr.z_score,
        )
        results.append(PairResult(a, b, store.video_path(a) or "?",
                                  store.video_path(b) or "?", [seg]))

    # Sort by z-score (most significant first), tie-break on extent
    def sortkey(r: PairResult) -> tuple[float, float]:
        s = r.segments[0]
        return (-(s.z_score or 0.0), -(s.a_end - s.a_start))
    results.sort(key=sortkey)
    return results


def query_against_index(query: Path, cfg: Config, store: Store, fe: FeatureExtractor) -> list[PairResult]:
    if store.index is None or store.index.ntotal == 0:
        return []
    result = _encode_video(query, fe, cfg)
    if result is None:
        return []
    vectors, timestamps, mirror_flags, _info = result
    sims_mat, idx_mat = store.search(vectors, cfg.knn)
    meta = store.frame_meta

    pair_points: dict[int, list[tuple[float, float, float, int]]] = defaultdict(list)
    for src in range(len(vectors)):
        src_ts = float(timestamps[src])
        src_mir = int(mirror_flags[src])
        for k in range(idx_mat.shape[1]):
            tgt = int(idx_mat[src, k])
            if tgt < 0:
                continue
            tgt_vid = int(meta["video_id"][tgt])
            tgt_ts = float(meta["ts"][tgt])
            tgt_mir = int(meta["mirrored"][tgt])
            sim = float(sims_mat[src, k])
            mirror = src_mir ^ tgt_mir
            pair_points[tgt_vid].append((src_ts, tgt_ts, sim, mirror))

    results: list[PairResult] = []
    for vid, pts in pair_points.items():
        if len(pts) < cfg.min_pair_matches:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        segs = align.find_segments(
            arr[:, :2], arr[:, 2], arr[:, 3].astype(bool),
            iters=cfg.ransac_iters, tol=cfg.ransac_tol,
            min_inliers=max(6, cfg.min_pair_matches // 2),
            min_seconds=cfg.min_segment_seconds,
            slope_min=cfg.slope_min, slope_max=cfg.slope_max,
        )
        segs = _filter_segments(segs, cfg)
        if not segs:
            continue
        b_path = store.video_path(vid) or "?"
        results.append(PairResult(-1, vid, str(query), b_path, segs))
    results.sort(key=lambda r: max(s.score * (s.a_end - s.a_start) for s in r.segments), reverse=True)
    return results


def ensure_extractor(cfg: Config):
    """Return either a local DINOv2 extractor or a remote one, both with .name/.dim/.encode()."""
    if cfg.endpoint:
        if not cfg.api_key:
            raise ValueError("--endpoint requires --api-key")
        return load_remote_extractor(cfg.endpoint, cfg.api_key)
    return load_extractor(cfg.model, cfg.device)
