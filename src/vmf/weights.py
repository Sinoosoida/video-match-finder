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
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from rich.console import Console

console = Console(stderr=True)


def compute_self_redundancy(
    vectors: np.ndarray, video_ids: np.ndarray, p: float,
) -> np.ndarray:
    """ρ(i) = Σ_j max(0, vec_i · vec_j)^p, summed over j in the same video as i.
    Includes self (sim=1 contributes 1), so ρ ≥ 1."""
    n = len(vectors)
    rho = np.zeros(n, dtype=np.float32)
    for vid in np.unique(video_ids):
        mask = video_ids == vid
        local = vectors[mask].astype(np.float32, copy=False)
        sim = local @ local.T
        np.clip(sim, 0.0, 1.0, out=sim)
        rho[mask] = (sim ** p).sum(axis=1)
    return rho


def compute_idf(
    vectors: np.ndarray, video_ids: np.ndarray, *,
    p: float, n_videos: int, k: int = 50,
) -> np.ndarray:
    """idf(i) = log((N+1) / (df(i)+1)) where

        df(i) = Σ_{v ≠ self_v(i)} max_{j ∈ v} max(0, vec_i · vec_j)^p

    The max over the foreign video gives the *best* match this frame finds
    there — small for unique frames, large for generic ones. Approximated
    via top-k kNN: contributions from rank > k are dropped because they
    would be tiny anyway.
    """
    n, d = vectors.shape
    idx = faiss.IndexFlatIP(d)
    idx.add(vectors.astype(np.float32, copy=False))
    sims, nbrs = idx.search(vectors.astype(np.float32, copy=False), k + 1)
    sims = np.clip(sims, 0.0, 1.0) ** p

    df = np.zeros(n, dtype=np.float32)
    nbr_vids = video_ids[nbrs]
    for i in range(n):
        self_vid = video_ids[i]
        mask = nbr_vids[i] != self_vid
        if not mask.any():
            continue
        vids = nbr_vids[i, mask]
        ss = sims[i, mask]
        uniq, inv = np.unique(vids, return_inverse=True)
        max_per_vid = np.zeros(len(uniq), dtype=np.float32)
        np.maximum.at(max_per_vid, inv, ss)
        df[i] = max_per_vid.sum()

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
        # exponent changed — caller must recompute
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

    assert store.index is not None
    n = store.index.ntotal
    console.print(f"[cyan]Computing ρ, idf for {n} vectors (p={p})…[/cyan]")
    vecs = np.vstack([store.index.reconstruct(i) for i in range(n)]).astype(np.float32)
    vids = store.frame_meta["video_id"]
    n_videos = int(np.unique(vids).size)

    rho = compute_self_redundancy(vecs, vids, p=p)
    idf = compute_idf(vecs, vids, p=p, n_videos=n_videos, k=k)

    save_weights(path, rho, idf, p=p)
    console.print(
        f"[dim]  ρ:  median={np.median(rho):.2f}  max={rho.max():.1f}\n"
        f"  idf: median={np.median(idf):.2f}  max={idf.max():.2f}[/dim]"
    )
    return rho, idf
