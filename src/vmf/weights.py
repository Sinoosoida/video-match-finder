"""Per-frame statistics for the smooth (Hough + permutation) pipeline.

We attach two continuous numbers to every frame in the index:

    ρ(i)   — *self-redundancy*: effective size of the cluster of near-duplicates
             this frame belongs to inside its OWN video. Isolated frame → 1.
             Frame inside a static run of K similar frames → ≈ K.

    idf(i) — *inverse document frequency*: how rare this frame's content is
             across OTHER videos. Common 'genre' frames → low idf.
             Unique frames → high idf.

Both are computed with the same softness exponent p, which controls how
"identical" two frames must look to count as the same visual content.
There are no thresholds — everything is continuous in cos similarity.

This module reads vectors from the disk-backed Store *one video at a time*
(for ρ) or in chunks (for idf), so peak RAM stays bounded regardless of
corpus size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console

console = Console(stderr=True)


def compute_self_redundancy(store, *, p: float) -> np.ndarray:
    """ρ(i) = Σ_j max(0, vec_i · vec_j)^p, summed over j in the same video as i.
    Includes self (sim=1 contributes 1), so ρ ≥ 1.

    Per-video matrices are small (≤ smooth_max_frames² × 4 bytes). When the
    store has an in-RAM cache populated, vector reads avoid disk entirely."""
    n = store.n_vectors()
    rho = np.zeros(n, dtype=np.float32)
    video_ids = store.frame_meta["video_id"]
    for vid in np.unique(video_ids):
        idxs = np.where(video_ids == vid)[0]
        if len(idxs) == 0:
            continue
        local = store.read_vectors(idxs).astype(np.float32, copy=False)
        sim = local @ local.T
        np.clip(sim, 0.0, 1.0, out=sim)
        rho[idxs] = (sim ** p).sum(axis=1)
    return rho


def compute_idf(
    store, *, p: float, n_videos: int, k: int = 50, chunk: int = 5000,
) -> np.ndarray:
    """idf(i) = log((N+1) / (df(i)+1)) where

        df(i) = Σ_{v ≠ self_v(i)} max_{j ∈ v} max(0, vec_i · vec_j)^p

    The max over the foreign video gives the *best* match this frame finds
    there — small for unique frames, large for generic ones. Approximated
    via top-k kNN through the store's exact chunked kNN (100% recall)."""
    n = store.n_vectors()
    if n == 0:
        return np.empty(0, dtype=np.float32)
    video_ids = store.frame_meta["video_id"]

    # Trigger the in-RAM cache up front so subsequent `store.search` calls
    # don't keep faulting pages from disk on every chunk.
    store._ensure_cached()

    try:
        from vmf.log import log
    except Exception:
        log = None
    import time as _t

    df = np.zeros(n, dtype=np.float32)
    n_chunks = (n + chunk - 1) // chunk
    t0 = _t.monotonic()
    for ci, start in enumerate(range(0, n, chunk)):
        end = min(start + chunk, n)
        queries = store.read_chunk(start, end)
        sims, nbrs = store.search(queries, k + 1)
        sims = np.clip(sims, 0.0, 1.0) ** p
        # Vectorised over the chunk: for each query, drop neighbours from the
        # same video as the query, then sum max-per-foreign-video.
        nbr_vids = video_ids[nbrs]      # (chunk, k+1)
        for i_local in range(end - start):
            self_vid = video_ids[start + i_local]
            mask = nbr_vids[i_local] != self_vid
            if not mask.any():
                continue
            vids = nbr_vids[i_local, mask]
            ss = sims[i_local, mask]
            uniq, inv = np.unique(vids, return_inverse=True)
            max_per_vid = np.zeros(len(uniq), dtype=np.float32)
            np.maximum.at(max_per_vid, inv, ss)
            df[start + i_local] = max_per_vid.sum()
        if log is not None:
            elapsed = _t.monotonic() - t0
            done = ci + 1
            eta = elapsed * (n_chunks - done) / done
            log.info(
                f"compute_idf: q-chunk {done}/{n_chunks} "
                f"({100 * done / n_chunks:.1f}%) "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
            )
    return np.log((n_videos + 1.0) / (df + 1.0)).astype(np.float32)


def save_weights(path: Path, rho: np.ndarray, idf: np.ndarray, *, p: float) -> None:
    np.savez(path, rho=rho.astype(np.float32), idf=idf.astype(np.float32),
             p=np.float32(p))


def load_weights(path: Path, *, p: float) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path)
    stored_p = float(data["p"]) if "p" in data.files else None
    if stored_p is not None and abs(stored_p - p) > 1e-6:
        return None
    return data["rho"], data["idf"]


def ensure_weights(
    store, *, p: float, k: int = 50, force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ρ, idf if cached at the current p; otherwise compute and save."""
    path = store.data_dir / "weights.npz"
    if not force:
        cached = load_weights(path, p=p)
        if cached is not None:
            return cached

    n = store.n_vectors()
    console.print(f"[cyan]Computing ρ, idf for {n} vectors (p={p})…[/cyan]")
    vids = store.frame_meta["video_id"]
    n_videos = int(np.unique(vids).size)

    rho = compute_self_redundancy(store, p=p)
    idf = compute_idf(store, p=p, n_videos=n_videos, k=k)

    save_weights(path, rho, idf, p=p)
    console.print(
        f"[dim]  ρ:  median={np.median(rho):.2f}  max={rho.max():.1f}\n"
        f"  idf: median={np.median(idf):.2f}  max={idf.max():.2f}[/dim]"
    )
    return rho, idf
