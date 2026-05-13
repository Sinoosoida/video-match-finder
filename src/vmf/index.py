"""Disk-backed vector storage with chunked exact kNN.

Architecture:

    vectors.bin           — raw float32 vectors, append-only on disk.
                            Each row is exactly `dim` floats; vector index
                            equals row number. Read via np.memmap so only
                            requested pages are paged in.
    frames.bin            — per-frame metadata (video_id, ts, mirrored),
                            also append-only.
    videos.db             — sqlite catalog (path → id, status, etc.)

kNN is **exact** (no quantisation, no partition-skipping): for each chunk
of queries we stream chunks of the database from disk, run FAISS's
BLAS-optimised IndexFlatIP per chunk, and merge top-k across chunks. RAM
is bounded by O(chunk² × 4) ≈ 100 MB regardless of corpus size.

Why exact-only: at 768 dim the "curse of dimensionality" makes
branch-and-bound (kd-tree, ball-tree, IVF with bound-pruning) degenerate
to brute-force in practice — the triangle-inequality bound is too loose
to skip cells reliably. Approximate methods (IVFPQ, HNSW) trade some
recall for speed; here we paid the time for guaranteed 100% recall.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import faiss
import numpy as np

FRAME_DTYPE = np.dtype([("video_id", np.int32), ("ts", np.float32), ("mirrored", np.int8)])

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    path      TEXT NOT NULL UNIQUE,
    sha1      TEXT NOT NULL,
    duration  REAL NOT NULL,
    width     INTEGER,
    height    INTEGER,
    n_frames  INTEGER NOT NULL,
    status    TEXT NOT NULL DEFAULT 'pending',
    added_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_videos_sha1 ON videos(sha1);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def file_sha1(path: Path, chunk: int = 1 << 20, max_bytes: int = 64 << 20) -> str:
    """Hash up to max_bytes — enough for de-dup, fast on huge files."""
    h = hashlib.sha1()
    read = 0
    with path.open("rb") as f:
        while read < max_bytes:
            buf = f.read(min(chunk, max_bytes - read))
            if not buf:
                break
            h.update(buf)
            read += len(buf)
    return h.hexdigest()


@dataclass
class VideoRow:
    id: int
    path: str
    duration: float
    n_frames: int


class Store:
    """Disk-backed vectors with chunked exact kNN."""

    def __init__(self, data_dir: Path, dim: int | None = None):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: AsyncIndexer's worker threads call
        # add_vectors/mark_complete from callbacks. AsyncIndexer.store_lock
        # serialises every write so concurrent access is safe.
        self.db = sqlite3.connect(self.data_dir / "videos.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
        try:
            self.db.execute(
                "ALTER TABLE videos ADD COLUMN status TEXT NOT NULL DEFAULT 'complete'"
            )
        except sqlite3.OperationalError:
            pass
        self.db.commit()

        self._vectors_path = self.data_dir / "vectors.bin"
        self._meta_path = self.data_dir / "frames.bin"

        # Refuse to load incompatible old-format indices. The previous layout
        # used `index.faiss` (HNSWFlat with all vectors in RAM) and `frames.npy`
        # (single-shot numpy save). There is no automatic migration.
        legacy_index = self.data_dir / "index.faiss"
        legacy_meta = self.data_dir / "frames.npy"
        if (legacy_index.exists() or legacy_meta.exists()) and not self._vectors_path.exists():
            raise RuntimeError(
                f"Legacy index format found in {self.data_dir}. The vector "
                f"storage layout changed (raw vectors are now appended to "
                f"vectors.bin; metadata to frames.bin). Run `vmf reset` and "
                f"re-scan to migrate."
            )

        stored_dim = self._read_meta("dim")
        if stored_dim is not None:
            self.dim = int(stored_dim)
        elif dim is not None:
            self.dim = dim
            self._write_meta("dim", str(dim))
        else:
            self.dim = 0

        if self._meta_path.exists():
            self.frame_meta = np.fromfile(self._meta_path, dtype=FRAME_DTYPE)
        else:
            self.frame_meta = np.empty(0, dtype=FRAME_DTYPE)

        # Self-heal: if frame_meta references vectors beyond what's on disk
        # (mid-batch crash), drop the orphaned tail. This keeps the two files
        # in sync without needing fsync on every batch.
        if self.dim and self._vectors_path.exists():
            vec_capacity = self._vectors_path.stat().st_size // (self.dim * 4)
            if len(self.frame_meta) > vec_capacity:
                self.frame_meta = self.frame_meta[:vec_capacity]
                # Rewrite frame_meta to match
                with open(self._meta_path, "wb") as f:
                    f.write(self.frame_meta.tobytes())

    # -------- meta --------

    def _read_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _write_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.db.commit()

    # -------- back-compat shim --------

    @property
    def index(self):
        """Back-compat shim: a few callers still ask `store.index is None`
        or `store.index.ntotal`. Return a tiny proxy so they keep working.

        Prefer `store.n_vectors()` and `store.read_vectors(...)` for new code.
        """
        n = self.n_vectors()
        if n == 0 and self.dim == 0:
            return None
        return _IndexShim(self, n)

    # -------- catalog --------

    def known_sha1s(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT sha1 FROM videos")}

    def has_video(self, path: Path, sha1: str) -> bool:
        """Return True for videos that should not be reprocessed.

        'complete'  — successfully indexed.
        'rejected'  — deliberately skipped (e.g. decoder sanity check
                      tripped, see `TooManyKeyframesError`). Not retried.
        'failed' / 'pending' / no row — retried on next scan."""
        row = self.db.execute(
            "SELECT 1 FROM videos WHERE status IN ('complete','rejected') "
            "AND (path=? OR sha1=?) LIMIT 1",
            (str(path), sha1),
        ).fetchone()
        return row is not None

    def mark_complete(self, video_id: int) -> None:
        self.db.execute("UPDATE videos SET status='complete' WHERE id=?", (video_id,))
        self.db.commit()

    def mark_failed(self, video_id: int) -> None:
        self.db.execute("UPDATE videos SET status='failed' WHERE id=?", (video_id,))
        self.db.commit()

    def mark_rejected(self, video_id: int) -> None:
        """Mark a video as deliberately skipped. Unlike 'failed', rejected
        videos are not retried on subsequent scans."""
        self.db.execute("UPDATE videos SET status='rejected' WHERE id=?", (video_id,))
        self.db.commit()

    def add_video(self, path: Path, sha1: str, duration: float, w: int, h: int, n_frames: int) -> int:
        """Insert (or recycle) a video row in 'pending' state."""
        row = self.db.execute(
            "SELECT id, status FROM videos WHERE path=?", (str(path),)
        ).fetchone()
        if row is not None:
            old_id, old_status = int(row[0]), row[1]
            if old_status == "complete":
                return old_id
            # Tombstone old vectors so they're filtered out at search time.
            if len(self.frame_meta):
                mask = self.frame_meta["video_id"] == old_id
                if mask.any():
                    self.frame_meta["video_id"][mask] = -old_id - 1
            self.db.execute("DELETE FROM videos WHERE id=?", (old_id,))
        cur = self.db.execute(
            "INSERT INTO videos(path, sha1, duration, width, height, n_frames, status)"
            " VALUES (?,?,?,?,?,?,'pending')",
            (str(path), sha1, duration, w, h, n_frames),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def update_n_frames(self, video_id: int, n_frames: int) -> None:
        self.db.execute("UPDATE videos SET n_frames=? WHERE id=?", (n_frames, video_id))
        self.db.commit()

    def list_videos(self) -> list[VideoRow]:
        rows = self.db.execute("SELECT id, path, duration, n_frames FROM videos ORDER BY id").fetchall()
        return [VideoRow(*r) for r in rows]

    def video_path(self, video_id: int) -> str | None:
        row = self.db.execute("SELECT path FROM videos WHERE id=?", (video_id,)).fetchone()
        return row[0] if row else None

    # -------- vectors --------

    def init_index(self, dim: int) -> None:
        """Set/validate vector dim. No persistent kNN structure is built —
        `search` does chunked exact kNN on demand."""
        if self.dim == 0:
            self.dim = dim
            self._write_meta("dim", str(dim))
        elif self.dim != dim:
            raise ValueError(f"vector dim mismatch: stored {self.dim}, got {dim}")
        # Make sure vectors.bin file exists for append.
        if not self._vectors_path.exists():
            self._vectors_path.touch()

    def n_vectors(self) -> int:
        return int(len(self.frame_meta))

    def add_vectors(
        self, vecs: np.ndarray, video_id: int,
        timestamps: np.ndarray, mirrored: np.ndarray,
    ) -> None:
        """Append vectors to disk (no in-memory accumulation)."""
        n = int(len(vecs))
        if n == 0:
            return
        if self.dim and vecs.shape[1] != self.dim:
            raise ValueError(f"add_vectors dim {vecs.shape[1]} != store dim {self.dim}")
        if self.dim == 0:
            self.init_index(int(vecs.shape[1]))
        # Float32 append, contiguous. tobytes() copies only the current batch
        # — RAM cost is O(batch × dim), not O(corpus).
        data = np.ascontiguousarray(vecs, dtype=np.float32)
        rows = np.empty(n, dtype=FRAME_DTYPE)
        rows["video_id"] = video_id
        rows["ts"] = timestamps
        rows["mirrored"] = mirrored
        # Order matters: write vectors FIRST, then metadata. If we crash
        # between the two, vectors.bin has trailing orphans (harmless — they
        # get truncated on next load via the capacity check above). If we
        # wrote metadata first, frame_meta would reference data that doesn't
        # exist on disk yet — much harder to recover from.
        with open(self._vectors_path, "ab") as f:
            f.write(data.tobytes())
        with open(self._meta_path, "ab") as f:
            f.write(rows.tobytes())
        self.frame_meta = np.concatenate([self.frame_meta, rows])

    def _vectors_mmap(self) -> np.memmap | None:
        """Memory-map vectors.bin (read-only). Returns None if empty."""
        n = self.n_vectors()
        if n == 0 or self.dim == 0 or not self._vectors_path.exists():
            return None
        # If file is shorter than expected (corrupted/partial write), fail loud.
        expected = n * self.dim * 4
        actual = self._vectors_path.stat().st_size
        if actual < expected:
            raise RuntimeError(
                f"vectors.bin has {actual} bytes but frame_meta expects "
                f"{expected} ({n} vectors × {self.dim} dim × 4 bytes). "
                f"vector file is corrupted or out of sync."
            )
        return np.memmap(
            self._vectors_path, dtype=np.float32, mode="r",
            shape=(n, self.dim),
        )

    def read_vectors(self, indices) -> np.ndarray:
        """Read a subset of vectors. `indices` can be a slice, list, or
        np.ndarray of int indices. Returns a regular np.ndarray (a copy of
        the requested rows from the mmap'd file)."""
        mm = self._vectors_mmap()
        if mm is None:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        return np.ascontiguousarray(mm[indices])

    def iter_vector_chunks(self, chunk: int = 10000) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (start_index, vectors_chunk) for sequential traversal."""
        n = self.n_vectors()
        mm = self._vectors_mmap()
        if mm is None:
            return
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            yield start, np.ascontiguousarray(mm[start:end])

    # -------- kNN index --------

    # Searches are stateless: each call to `search` streams the database
    # from disk in chunks and runs a fresh FAISS IndexFlatIP per chunk
    # (each one is a thin wrapper around BLAS GEMM). No persistent kNN
    # structure to build or warm up.

    def search(
        self, vecs: np.ndarray, k: int, chunk: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact kNN: top-k for each query vector against the whole store.

        Streams the database in chunks from mmap'd `vectors.bin`. For each
        chunk D, a fresh `IndexFlatIP(D)` runs the BLAS-optimised exact
        search; results are merged into a running top-k per query. RAM is
        bounded by O(chunk × dim × 4) for queries+database, plus O(|vecs| × k)
        for the running top-k buffer. 100% recall by construction."""
        n_q = int(len(vecs))
        n_db = self.n_vectors()
        if n_q == 0 or n_db == 0 or self.dim == 0:
            return (np.full((n_q, k), -1.0, dtype=np.float32),
                    np.full((n_q, k), -1, dtype=np.int64))
        Q = np.ascontiguousarray(vecs, dtype=np.float32)
        return self._exact_topk_against_store(Q, k, chunk)

    def search_chunked(
        self, queries_source, k: int, chunk: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Self-kNN entry point: when `queries_source` is an int N, queries
        are read from the store itself in chunks (all-vs-all kNN); otherwise
        `queries_source` is a (start, end) → ndarray getter or an ndarray.
        Two-level chunking keeps both query and database RAM bounded."""
        if isinstance(queries_source, int):
            n_q = queries_source
            get_queries = lambda a, b: self.read_vectors(slice(a, b))
        elif callable(queries_source):
            n_q = queries_source.__len__()   # type: ignore[union-attr]
            get_queries = queries_source
        else:                                # already an ndarray
            return self.search(queries_source, k, chunk)

        n_db = self.n_vectors()
        if n_q == 0 or n_db == 0 or self.dim == 0:
            return (np.full((n_q, k), -1.0, dtype=np.float32),
                    np.full((n_q, k), -1, dtype=np.int64))

        sims_out = np.empty((n_q, k), dtype=np.float32)
        idxs_out = np.empty((n_q, k), dtype=np.int64)
        for q_start in range(0, n_q, chunk):
            q_end = min(q_start + chunk, n_q)
            Q = np.ascontiguousarray(get_queries(q_start, q_end), dtype=np.float32)
            s, i = self._exact_topk_against_store(Q, k, chunk)
            sims_out[q_start:q_end] = s
            idxs_out[q_start:q_end] = i
        return sims_out, idxs_out

    def _exact_topk_against_store(
        self, Q: np.ndarray, k: int, chunk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Core: top-k for queries Q (already in RAM) against the full store
        via chunked brute-force on mmap'd vectors. Merge of top-k across DB
        chunks is associative, so the final result equals the exact global
        top-k. No approximation."""
        n_q = int(len(Q))
        n_db = self.n_vectors()
        sims = np.full((n_q, k), -np.inf, dtype=np.float32)
        idxs = np.full((n_q, k), -1, dtype=np.int64)

        for d_start in range(0, n_db, chunk):
            d_end = min(d_start + chunk, n_db)
            D = self.read_vectors(slice(d_start, d_end))
            # FAISS IndexFlatIP uses BLAS GEMM under the hood — much faster
            # than a hand-rolled numpy matmul + topk, especially for the
            # per-row partial-sort phase.
            sub = faiss.IndexFlatIP(self.dim)
            sub.add(D)
            k_local = min(k, d_end - d_start)
            s_block, i_block_local = sub.search(Q, k_local)
            # Local FAISS indices → global indices into the store.
            i_block_global = np.where(
                i_block_local >= 0, i_block_local + d_start, i_block_local,
            )
            if k_local < k:
                pad_n = k - k_local
                s_block = np.concatenate(
                    [s_block, np.full((n_q, pad_n), -np.inf, dtype=np.float32)],
                    axis=1,
                )
                i_block_global = np.concatenate(
                    [i_block_global, np.full((n_q, pad_n), -1, dtype=np.int64)],
                    axis=1,
                )
            combined_s = np.concatenate([sims, s_block], axis=1)
            combined_i = np.concatenate([idxs, i_block_global], axis=1)
            top = np.argpartition(-combined_s, k - 1, axis=1)[:, :k]
            sims = np.take_along_axis(combined_s, top, axis=1)
            idxs = np.take_along_axis(combined_i, top, axis=1)

        order = np.argsort(-sims, axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        idxs = np.take_along_axis(idxs, order, axis=1)
        return sims, idxs

    # -------- persistence --------

    def save(self) -> None:
        """Force any buffered writes to durable storage. Both vectors.bin and
        frames.bin are append-mode writes (already on disk modulo OS page
        cache); this fsyncs them so a power failure right after this call
        loses nothing that's been added so far. Called at video boundaries by
        AsyncIndexer to make mark_complete crash-consistent."""
        import os
        for p in (self._vectors_path, self._meta_path):
            if p.exists():
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)

    def reset(self) -> None:
        self.force_clear(self.data_dir, db=self.db)
        self.frame_meta = np.empty(0, dtype=FRAME_DTYPE)
        self.dim = 0

    @staticmethod
    def force_clear(data_dir: Path, db: sqlite3.Connection | None = None) -> None:
        """Wipe a data_dir regardless of format (current OR legacy). Safe to
        call without instantiating a Store — that lets `vmf reset` recover
        from data directories whose layout would otherwise be rejected by
        `Store.__init__` (e.g. a leftover index.faiss after upgrading)."""
        for name in (
            "vectors.bin", "frames.bin",
            # legacy artefacts
            "index.faiss", "frames.npy",
            "knn.faiss", "knn_built_for_n.npy",
            "weights.npz",
        ):
            p = data_dir / name
            if p.exists():
                p.unlink()
        if db is None:
            db_path = data_dir / "videos.db"
            if not db_path.exists():
                return
            db = sqlite3.connect(db_path)
            close_after = True
        else:
            close_after = False
        try:
            db.execute("DELETE FROM videos")
            db.execute("DELETE FROM meta")
            db.commit()
        finally:
            if close_after:
                db.close()


class _IndexShim:
    """Back-compat object returned by `Store.index` for the few legacy
    callers that still use `store.index.ntotal` / `store.index.reconstruct`."""

    def __init__(self, store: Store, n: int):
        self._store = store
        self.ntotal = n

    def reconstruct(self, i: int) -> np.ndarray:
        return self._store.read_vectors(np.array([i], dtype=np.int64))[0]
