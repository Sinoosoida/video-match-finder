from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

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
    added_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_videos_sha1 ON videos(sha1);

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
    """FAISS HNSW index + sqlite catalog + per-frame metadata array."""

    def __init__(self, data_dir: Path, dim: int | None = None):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the async indexer finalises videos from
        # worker threads; AsyncIndexer.store_lock serialises writes so this
        # is safe.
        self.db = sqlite3.connect(self.data_dir / "videos.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

        self._index_path = self.data_dir / "index.faiss"
        self._meta_path = self.data_dir / "frames.npy"

        stored_dim = self._read_meta("dim")
        if stored_dim is not None:
            self.dim = int(stored_dim)
        elif dim is not None:
            self.dim = dim
            self._write_meta("dim", str(dim))
        else:
            self.dim = 0

        if self._index_path.exists():
            self.index = faiss.read_index(str(self._index_path))
        elif self.dim:
            self.index = self._new_index(self.dim)
        else:
            self.index = None

        if self._meta_path.exists():
            self.frame_meta = np.load(self._meta_path, allow_pickle=False)
        else:
            self.frame_meta = np.empty(0, dtype=FRAME_DTYPE)

    @staticmethod
    def _new_index(dim: int) -> faiss.Index:
        idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 80
        idx.hnsw.efSearch = 64
        return idx

    def _read_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _write_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.db.commit()

    # -------- catalog --------

    def known_sha1s(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT sha1 FROM videos")}

    def has_video(self, path: Path, sha1: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM videos WHERE path=? OR sha1=? LIMIT 1", (str(path), sha1)
        ).fetchone()
        return row is not None

    def add_video(self, path: Path, sha1: str, duration: float, w: int, h: int, n_frames: int) -> int:
        cur = self.db.execute(
            "INSERT INTO videos(path, sha1, duration, width, height, n_frames) VALUES (?,?,?,?,?,?)",
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
        if self.index is None:
            self.dim = dim
            self.index = self._new_index(dim)
            self._write_meta("dim", str(dim))
        elif self.dim != dim:
            raise ValueError(f"index dim mismatch: stored {self.dim}, got {dim}")

    def add_vectors(self, vecs: np.ndarray, video_id: int, timestamps: np.ndarray, mirrored: np.ndarray) -> None:
        assert self.index is not None
        n = len(vecs)
        if n == 0:
            return
        self.index.add(vecs)
        rows = np.empty(n, dtype=FRAME_DTYPE)
        rows["video_id"] = video_id
        rows["ts"] = timestamps
        rows["mirrored"] = mirrored
        self.frame_meta = np.concatenate([self.frame_meta, rows])

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self._index_path))
        if len(self.frame_meta):
            np.save(self._meta_path, self.frame_meta, allow_pickle=False)

    # -------- search --------

    def search(self, vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.index is not None
        return self.index.search(vecs, k)

    def reset(self) -> None:
        for p in (self._index_path, self._meta_path):
            if p.exists():
                p.unlink()
        self.db.execute("DELETE FROM videos")
        self.db.execute("DELETE FROM meta")
        self.db.commit()
        self.index = None
        self.frame_meta = np.empty(0, dtype=FRAME_DTYPE)
        self.dim = 0
