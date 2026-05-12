from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}


def default_data_dir() -> Path:
    base = os.environ.get("VMF_HOME") or os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    root = Path(base).expanduser()
    if root.name != "video-match-finder":
        root = root / "video-match-finder"
    return root


@dataclass
class Config:
    data_dir: Path = field(default_factory=default_data_dir)
    fps: float = 2.0                  # 2 samples/sec — 0.5s grid, good recall/cost balance
    keyframes_only: bool = False      # decode only I-frames; 30–60× less CPU on H.264
    cropdetect: bool = True           # run cropdetect pre-pass (~1–3s/video). Disable on
                                      # uniform collections without letterbox to save time.
    frame_size: int = 224
    model: str = "auto"               # auto | dinov2_vits14 | dinov2_vitb14
    batch_size: int = 64              # sweet spot on a high-RTT (~300 ms) SSH-tunnel link:
                                      # small enough to keep one request well under a second
                                      # of server-side compute, large enough to amortise the
                                      # per-request RTT floor. Empirically batch=64 × inflight=4
                                      # gave ~70 fps vs ~50 fps for batch=32 × inflight=4 on our
                                      # nas→vast setup (see bench_batch.py). Doubling again to
                                      # 128 starts losing to server-side GPU serial processing.
    # encode_inflight controls how many batches we keep in flight while ffmpeg
    # keeps decoding the next ones. With batch=64, four in-flight requests
    # already cover the RTT × bandwidth product on our SSH-tunnel link; more
    # than that piles work on a single-GPU server and hurts wall-clock.
    encode_inflight: int = 4
    knn: int = 10
    min_pair_matches: int = 20        # ↑ from 15 — at fps=2 noise clouds are denser too
    ransac_iters: int = 10_000        # exhaustive search — finds faint real lines reliably
    ransac_tol: float = 1.0           # ↓ from 1.5 — at fps=2 one sample = 0.5s, so 2 samples
    min_segment_seconds: float = 4.0  # time scale unchanged
    min_segment_score: float = 0.80
    min_match_sim: float = 0.72       # per-frame cosine floor below which neighbor is junk
    min_inlier_density: float = 0.40  # inliers/second along the matched line
    min_frame_std: float = 18.0       # drop near-uniform frames (black/fade/solid logos)
    watermark_max_videos: int = 8     # drop generic embeddings hitting too many videos
    mutual_knn: bool = True           # require frame matches to be mutual top-K neighbors
    slope_min: float = 0.5            # ↑ from 0.25 — extreme speed ratios were spurious
    slope_max: float = 2.0            # ↓ from 4.0
    mirror: bool = True
    device: str = "auto"              # auto | cpu | cuda
    hwaccel: bool = False             # opt-in: -hwaccel cuda hangs on old NVDEC (Pascal/older)

    # ---- remote embedding server ----
    endpoint: str | None = None       # e.g. http://host:8000/v1; if set, embeddings are
    api_key: str | None = None        # fetched from an OpenAI-compatible HTTP service.

    # ---- smooth pipeline (Hough + permutation) ----
    use_smooth: bool = True           # default; False = legacy RANSAC + filter cascade
    smooth_p: float = 4.0             # softness of "two frames look the same" (continuous)
    smooth_tau: float = 1.0           # Hough β-bin width in seconds
    smooth_n_samples: int = 4000      # importance-sampled Hough pairs per pair-of-videos
    smooth_n_perm: int = 50           # permutations for null distribution
    smooth_min_zscore: float = 6.0    # accept lines with z = (L* - μ_null) / σ_null ≥ this
    smooth_max_pvalue: float = 0.02   # secondary filter (1/(n_perm+1) is the floor)
    smooth_max_frames: int = 350      # cap T per side; downsample longer videos for speed
    smooth_idf_k: int = 50            # kNN width for idf estimation
    smooth_min_kmatches: int = 5      # minimum raw cross-video kNN matches to bother scoring
    # idf and ρ are computed and stored for diagnostics. On narrow corpora they over-correct
    # (ρ penalises long real duplicates, idf amplifies genre-shared frames), so by default
    # the Hough weight matrix uses sim^p only. Flip these on for wider, more diverse corpora.
    smooth_use_idf: bool = False
    smooth_use_rho: bool = False
    smooth_min_tube_sim: float = 0.70 # extent-extraction floor: cells with raw sim below this
                                      # are not counted as "supporting" the found line

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.faiss"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "videos.db"

    @property
    def frames_meta_path(self) -> Path:
        return self.data_dir / "frames.npy"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
