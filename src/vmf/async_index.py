"""Cross-video async indexing.

A single ffmpeg process reads from disk at a time (so we never thrash the
HDD with parallel reads of different files), but the network/GPU pipeline
runs continuously: as soon as ffmpeg finishes a video the next ffmpeg
starts, while the previous video's batches are still in flight on the
encoder pool. Results return out of order and are stitched back into
per-video buckets keyed on video_id.

Threading:
  - `submit_batch` is called from the (single) main thread.
  - `_on_done` callbacks run on encoder-pool worker threads.
  - `_finalize` runs on whichever thread happened to receive the last
    callback for a given video; we serialise FAISS writes via store_lock.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console

console = Console(stderr=True)


@dataclass
class _VideoState:
    video_id: int
    collected: list[tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    expected: int = 0      # set once decode finishes
    received: int = 0      # incremented per callback
    decoded: bool = False
    failed: bool = False


class AsyncIndexer:
    """Cross-video async pipeline. Caller stays on one thread; we own the rest."""

    def __init__(self, store, fe, cfg):
        self.store = store
        self.fe = fe
        self.cfg = cfg
        n_inflight = max(1, getattr(cfg, "encode_inflight", 8))
        self.pool = ThreadPoolExecutor(
            max_workers=n_inflight, thread_name_prefix="vmf-encoder",
        )
        # Bounded inflight via semaphore: submit_batch blocks once we have this
        # many requests already in the air, applying back-pressure to ffmpeg.
        self.sem = threading.Semaphore(n_inflight)
        self.lock = threading.Lock()                         # for `pending` dict
        self.cond = threading.Condition(self.lock)
        self.store_lock = threading.Lock()                   # for FAISS + sqlite writes
        self.pending: dict[int, _VideoState] = {}
        self.completed = 0
        self._finalize_cb = None      # optional: caller may set to receive notifications

    def register(self, video_id: int) -> None:
        with self.lock:
            self.pending[video_id] = _VideoState(video_id=video_id)

    def submit_batch(
        self, video_id: int, arr: np.ndarray, ts: np.ndarray, mirror_flag: int,
    ) -> None:
        self.sem.acquire()
        fut = self.pool.submit(self.fe.encode, arr)
        fut.add_done_callback(lambda f: self._on_done(video_id, ts, mirror_flag, f))

    def mark_decoded(self, video_id: int, expected_batches: int) -> None:
        finalize_target: _VideoState | None = None
        with self.cond:
            st = self.pending.get(video_id)
            if st is None:
                return
            st.decoded = True
            st.expected = expected_batches
            if st.received >= st.expected:
                self.pending.pop(video_id, None)
                finalize_target = st
                self.cond.notify_all()
        if finalize_target is not None:
            self._finalize(finalize_target)

    def discard(self, video_id: int) -> None:
        """Drop a pending video without finalising — used when we decide to
        re-decode (e.g. cropdetect found letterbox after first pass)."""
        with self.cond:
            self.pending.pop(video_id, None)
            self.cond.notify_all()

    def _on_done(
        self, video_id: int, ts: np.ndarray, mirror_flag: int, fut: Future,
    ) -> None:
        vecs: np.ndarray | None
        try:
            vecs = fut.result()
        except Exception as e:
            console.print(f"[red]encode error vid={video_id}: {e}[/red]")
            vecs = None

        finalize_target: _VideoState | None = None
        with self.cond:
            st = self.pending.get(video_id)
            if st is not None:
                if vecs is not None:
                    st.collected.append((vecs, ts, mirror_flag))
                else:
                    st.failed = True
                st.received += 1
                if st.decoded and st.received >= st.expected:
                    self.pending.pop(video_id, None)
                    finalize_target = st
                    self.cond.notify_all()
        self.sem.release()
        if finalize_target is not None:
            self._finalize(finalize_target)

    def _finalize(self, st: _VideoState) -> None:
        if st.failed or not st.collected:
            self.completed += 1
            if self._finalize_cb:
                self._finalize_cb(st.video_id, success=False)
            return
        vectors = np.concatenate([c[0] for c in st.collected], axis=0)
        timestamps = np.concatenate([c[1] for c in st.collected], axis=0)
        mirror_flags = np.concatenate(
            [np.full(len(c[0]), c[2], dtype=np.int8) for c in st.collected],
            axis=0,
        )
        n_frames = int((mirror_flags == 0).sum())
        with self.store_lock:
            self.store.init_index(self.fe.dim)
            self.store.add_vectors(vectors, st.video_id, timestamps, mirror_flags)
            try:
                self.store.update_n_frames(st.video_id, n_frames)
            except AttributeError:
                pass
        self.completed += 1
        if self._finalize_cb:
            self._finalize_cb(st.video_id, success=True)

    def wait_all(self) -> None:
        with self.cond:
            while self.pending:
                self.cond.wait()
        self.pool.shutdown(wait=True)
        with self.store_lock:
            self.store.save()
