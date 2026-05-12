from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    a_start: float
    a_end: float
    b_start: float
    b_end: float
    speed_ratio: float       # b plays at speed_ratio × a (slope of t_b vs t_a)
    mirrored: bool
    inliers: int
    score: float             # mean cosine similarity of inlier matches (0..1)
    pvalue: float | None = None         # set by smooth pipeline; None for legacy RANSAC
    weighted_support: float | None = None  # raw Hough peak L*, for diagnostics
    z_score: float | None = None        # (L* - μ_null) / σ_null — main acceptance metric


def _ransac_line(
    points: np.ndarray, iters: int, tol: float, slope_min: float, slope_max: float,
) -> tuple[np.ndarray, float, float] | None:
    """RANSAC for t_b = a*t_a + b. points: (N, 2). Returns (inlier_mask, a, b)."""
    n = len(points)
    if n < 4:
        return None
    # Pre-sample pairs up front — vectorized RANSAC is much faster at high iters.
    rng = random.Random(0xC0FFEE)
    samples = np.empty((iters, 2), dtype=np.int64)
    for k in range(iters):
        i, j = rng.sample(range(n), 2)
        samples[k, 0], samples[k, 1] = i, j
    best_inliers: np.ndarray | None = None
    best_ab = (1.0, 0.0)
    xs, ys = points[:, 0], points[:, 1]
    for k in range(iters):
        i, j = int(samples[k, 0]), int(samples[k, 1])
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[j], ys[j]
        if abs(x2 - x1) < 1e-3:
            continue
        a = (y2 - y1) / (x2 - x1)
        if not (slope_min <= a <= slope_max):
            continue
        b = y1 - a * x1
        residuals = np.abs(ys - (a * xs + b))
        inliers = residuals <= tol
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_ab = (a, b)
    if best_inliers is None or best_inliers.sum() < 4:
        return None
    return best_inliers, best_ab[0], best_ab[1]


def find_segments(
    points: np.ndarray,
    sims: np.ndarray,
    mirror_flags: np.ndarray,
    *,
    iters: int = 200,
    tol: float = 1.5,
    min_inliers: int = 6,
    min_seconds: float = 3.0,
    max_segments: int = 4,
    slope_min: float = 0.5,
    slope_max: float = 2.0,
) -> list[Segment]:
    """Iteratively peel off matching segments from a (t_a, t_b) match cloud.

    points: (N, 2) timestamps in seconds.
    sims:   (N,)   cosine similarity per match.
    mirror_flags: (N,) bool — True if either side was the mirrored variant.
    """
    if len(points) < min_inliers:
        return []

    remaining = np.ones(len(points), dtype=bool)
    segments: list[Segment] = []

    for _ in range(max_segments):
        if remaining.sum() < min_inliers:
            break
        sub = points[remaining]
        sub_sims = sims[remaining]
        sub_mirror = mirror_flags[remaining]
        result = _ransac_line(sub, iters, tol, slope_min, slope_max)
        if result is None:
            break
        inlier_mask, a, _b = result
        if inlier_mask.sum() < min_inliers:
            break

        ax = sub[inlier_mask, 0]
        bx = sub[inlier_mask, 1]
        a_lo, a_hi = float(ax.min()), float(ax.max())
        b_lo, b_hi = float(bx.min()), float(bx.max())
        if (a_hi - a_lo) < min_seconds:
            # mark these consumed and continue searching
            indices = np.flatnonzero(remaining)[inlier_mask]
            remaining[indices] = False
            continue

        score = float(sub_sims[inlier_mask].mean())
        mirrored = bool(sub_mirror[inlier_mask].mean() > 0.5)
        segments.append(Segment(
            a_start=a_lo, a_end=a_hi, b_start=b_lo, b_end=b_hi,
            speed_ratio=float(a), mirrored=mirrored,
            inliers=int(inlier_mask.sum()), score=score,
        ))

        indices = np.flatnonzero(remaining)[inlier_mask]
        remaining[indices] = False

    segments.sort(key=lambda s: (s.score * (s.a_end - s.a_start)), reverse=True)
    return segments
