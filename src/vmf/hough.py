"""Importance-sampled Hough transform + permutation test for time alignment.

Given two videos A and B with frame timestamps t_A, t_B and a weighted match
matrix W[i, j] (typically W = sim^p · idf · idf / (ρ·ρ)), this module:

1. Samples pairs of cells (i₁,j₁) and (i₂,j₂) proportional to W (importance
   sampling). Each sampled pair votes for the unique line `t_B = α t_A + β`
   that passes through (t_A[i₁], t_B[j₁]) and (t_A[i₂], t_B[j₂]), weighted
   by W[i₁,j₁] · W[i₂,j₂].

2. Bins votes in (α, β) space and returns the peak. A real diagonal of
   matches concentrates votes; an isotropic blob scatters them.

3. Runs a permutation test: shuffle t_B labels, repeat the Hough vote with
   the SAME sampled pairs, see how often the null peak meets or beats the
   real peak. This gives a calibrated p-value without any hand-tuned
   thresholds on score or inlier count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HoughResult:
    alpha: float
    beta: float
    support: float          # raw weighted peak L*
    null_max: float         # max L* under permutation
    null_mean: float        # mean L* under permutation
    null_std: float         # std of null L*; for z-score
    z_score: float          # (L* - null_mean) / null_std — preferred decision metric
    pvalue: float           # (k+1)/(n+1) — limited by n_perm resolution
    n_perm: int


def _direct_hough(
    t_A: np.ndarray, t_B: np.ndarray, W_flat: np.ndarray,
    T_B: int, alphas: np.ndarray,
    beta_min: float, beta_step: float, n_beta: int,
) -> np.ndarray:
    """Direct (un-sampled) weighted Hough.

    For each candidate slope α, every cell (i, j) of the matrix votes
    weight W[i, j] into the bin for offset β = t_B[j] - α·t_A[i].
    Vectorised across α via a single bincount call.
    """
    n_alpha = alphas.size
    T_A = t_A.size
    # residual[k, i, j] = t_B[j] - α_k · t_A[i]
    residual = t_B[None, None, :] - alphas[:, None, None] * t_A[None, :, None]
    beta_idx = np.floor((residual - beta_min) / beta_step).astype(np.int64)
    # collapse (α, β) into one flat bin index
    alpha_offsets = (np.arange(n_alpha, dtype=np.int64) * n_beta)[:, None, None]
    flat_idx = (beta_idx + alpha_offsets).ravel()
    weights_full = np.broadcast_to(W_flat.reshape(1, T_A, T_B),
                                   (n_alpha, T_A, T_B)).ravel()
    keep = (beta_idx.ravel() >= 0) & (beta_idx.ravel() < n_beta)
    H = np.bincount(flat_idx[keep], weights=weights_full[keep],
                    minlength=n_alpha * n_beta)
    return H.reshape(n_alpha, n_beta)


def hough_permutation(
    t_A: np.ndarray, t_B: np.ndarray, W: np.ndarray,
    *,
    tau: float = 1.0,
    n_samples: int = 4000,           # kept for backward signature; unused (no sampling)
    n_perm: int = 100,
    alpha_range: tuple[float, float] = (0.5, 2.0),
    n_alpha_bins: int = 30,
    n_beta_bins: int | None = None,
    rng: np.random.Generator | None = None,
) -> HoughResult | None:
    """Direct weighted Hough on W, then a permutation test on its peak.

    Every cell of W votes (with its weight) into the (α, β) bin its position
    implies. Real signal — a diagonal of high-weight cells — concentrates
    votes in one bin. Permuting t_B labels keeps the same weight mass but
    scatters where it lands, so the null peak is much smaller. The p-value
    is the fraction of permutations whose peak meets or beats the real one.
    """
    del n_samples  # superseded by direct Hough
    if rng is None:
        rng = np.random.default_rng(0)
    if W.ndim != 2 or W.shape[0] < 4 or W.shape[1] < 4:
        return None

    T_A, T_B = W.shape
    W_flat = W.astype(np.float64).ravel()
    if W_flat.sum() <= 0:
        return None

    beta_min = float(t_B.min()) - float(t_A.max()) - tau
    beta_max = float(t_B.max()) + tau
    if n_beta_bins is None:
        span = max(beta_max - beta_min, 4 * tau)
        n_beta_bins = max(20, int(span / tau))
    beta_step = (beta_max - beta_min) / n_beta_bins
    alpha_min, alpha_max = alpha_range
    alphas = alpha_min + (np.arange(n_alpha_bins) + 0.5) * (alpha_max - alpha_min) / n_alpha_bins

    H_real = _direct_hough(t_A, t_B, W_flat, T_B, alphas, beta_min, beta_step, n_beta_bins)
    if H_real.max() == 0:
        return None

    # Collect a null distribution per (α, β) bin via permutation. Per-bin
    # normalisation matters because different bins cover different numbers of
    # cells (matrix is rectangular, line lengths vary), and their null variance
    # scales accordingly. Comparing each bin against ITS OWN null removes the
    # rectangular-diagonal geometric bias.
    H_perm = np.empty((n_perm,) + H_real.shape, dtype=np.float64)
    for k in range(n_perm):
        t_B_perm = rng.permutation(t_B)
        H_perm[k] = _direct_hough(t_A, t_B_perm, W_flat, T_B, alphas, beta_min, beta_step, n_beta_bins)

    null_bin_mean = H_perm.mean(axis=0)
    # Pooled variance — a global scale for null fluctuation, robust to
    # low-coverage bins that have near-zero per-bin std and would otherwise
    # blow up z. Equivalent to assuming homoscedastic noise across the
    # mean-subtracted histogram, which is a reasonable approximation for the
    # bin-vs-bin comparison we want.
    pooled_std = float(np.sqrt(H_perm.var(axis=0, ddof=1).mean() + 1e-12))
    Z_real = (H_real - null_bin_mean) / pooled_std
    peak_z_real = float(Z_real.max())
    a_cell, b_cell = np.unravel_index(int(np.argmax(Z_real)), Z_real.shape)
    alpha_star = float(alphas[a_cell])
    beta_star = float(beta_min + (b_cell + 0.5) * beta_step)

    # Null distribution of max-z built using the same pooled std. For random
    # matrices E[max-z] ≈ √(2 ln N_bins); our threshold is set comfortably
    # above that.
    Z_null = (H_perm - null_bin_mean[None]) / pooled_std
    null_max_z = Z_null.max(axis=(1, 2))

    pvalue = (float((null_max_z >= peak_z_real).sum()) + 1.0) / (n_perm + 1.0)
    return HoughResult(
        alpha=alpha_star, beta=beta_star,
        support=float(H_real[a_cell, b_cell]),
        null_max=float(null_max_z.max()),
        null_mean=float(null_max_z.mean()),
        null_std=float(null_max_z.std(ddof=1)) if n_perm > 1 else 0.0,
        z_score=float(peak_z_real),
        pvalue=pvalue, n_perm=n_perm,
    )


def extract_extent(
    t_A: np.ndarray, t_B: np.ndarray, sim_raw: np.ndarray,
    alpha: float, beta: float, tau: float,
    sim_floor: float = 0.5, quantile: float = 0.5,
) -> dict | None:
    """Given an accepted (α, β) line, walk along it across t_A, pick the
    best t_B per step, return the extent and mean similarity of the
    supporting frames.

    We deliberately use *raw* similarity (no ρ/idf reweighting) here:
    the weighting matters for *deciding* there's a line, but the
    geographic extent should come from where actual frame matches lie.
    """
    T_A, T_B = sim_raw.shape
    matches = []
    for i in range(T_A):
        expected = alpha * t_A[i] + beta
        if expected < t_B.min() - tau or expected > t_B.max() + tau:
            continue
        j = int(np.argmin(np.abs(t_B - expected)))
        if abs(float(t_B[j]) - expected) > tau:
            continue
        matches.append((float(t_A[i]), float(t_B[j]), float(sim_raw[i, j])))
    if len(matches) < 4:
        return None
    arr = np.asarray(matches)
    sims = arr[:, 2]
    thr = max(float(np.quantile(sims, quantile)), sim_floor)
    keep = sims >= thr
    if keep.sum() < 4:
        return None
    sub = arr[keep]
    return {
        "a_start": float(sub[:, 0].min()),
        "a_end": float(sub[:, 0].max()),
        "b_start": float(sub[:, 1].min()),
        "b_end": float(sub[:, 1].max()),
        "mean_sim": float(sims[keep].mean()),
        "n_supporting": int(keep.sum()),
    }
