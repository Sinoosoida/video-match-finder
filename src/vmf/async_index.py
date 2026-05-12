"""Constant-memory cross-video async indexing.

A single ffmpeg process reads from disk at a time. Batches are submitted to
an encoder pool and **the resulting vectors are flushed to FAISS immediately
in the callback** — no per-video accumulation. The only per-video state is
three counters (expected / received / failed), so RAM is O(inflight × batch)
regardless of how long any individual video is.

Crash safety: each video starts as status='pending' in sqlite. Only when ALL
batches have come back successfully does it become 'complete'. On the next
scan we skip only 'complete' rows, so any half-indexed video gets retried.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from rich.console import Console

from vmf.log import log

console = Console(stderr=True)


@dataclass
class _VideoState:
    video_id: int
    expected: int = 0      # set when decode finishes (mark_decoded)
    received: int = 0      # incremented per callback
    n_frames: int = 0      # cumulative non-mirror frame count, for sqlite metadata
    decoded: bool = False
    failed: bool = False


class AsyncIndexer:
    def __init__(self, store, fe, cfg):
        self.store = store
        self.fe = fe
        self.cfg = cfg
        n_inflight = max(1, getattr(cfg, "encode_inflight", 8))
        self.pool = ThreadPoolExecutor(
            max_workers=n_inflight, thread_name_prefix="vmf-encoder",
        )
        # Back-pressure: ffmpeg blocks at submit_batch when this many requests
        # are already in flight. Keeps RAM bounded even when the network can't
        # keep up with decoding.
        self.sem = threading.Semaphore(n_inflight)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.store_lock = threading.Lock()
        self.pending: dict[int, _VideoState] = {}
        # Save FAISS periodically (after every N finalised videos), not after
        # every video, so we don't rewrite a multi-MB file per video.
        self.save_every = 25
        self._since_save = 0

        # Counters for the periodic stats reporter
        self.t_started = time.monotonic()
        self.completed = 0
        self.failed_videos = 0
        self.batches_done = 0
        self.batches_failed = 0
        self.inflight_now = 0
        self._stats_stop = threading.Event()
        self._stats_thr = threading.Thread(
            target=self._stats_loop, name="vmf-stats", daemon=True,
        )
        self._stats_thr.start()

    def _stats_loop(self) -> None:
        while not self._stats_stop.wait(10.0):
            with self.lock:
                pending = len(self.pending)
                pending_ids = sorted(self.pending.keys())[:5]
            elapsed = time.monotonic() - self.t_started
            rate = self.completed / elapsed if elapsed > 0 else 0
            log.info(
                f"stats: elapsed={elapsed:6.0f}s  done={self.completed}  failed={self.failed_videos}"
                f"  batches={self.batches_done} (errors={self.batches_failed})"
                f"  inflight={self.inflight_now}  pending={pending} {pending_ids}"
                f"  rate={rate*60:5.1f} vid/min"
            )

    # -------- main-thread API --------

    def register(self, video_id: int) -> None:
        with self.lock:
            self.pending[video_id] = _VideoState(video_id=video_id)

    def submit_batch(
        self, video_id: int, arr: np.ndarray, ts: np.ndarray, mirror_flag: int,
    ) -> None:
        wait_start = time.monotonic()
        self.sem.acquire()
        wait = time.monotonic() - wait_start
        if wait > 1.0:
            log.info(f"submit_batch vid={video_id} waited {wait:.1f}s for sem "
                     f"(inflight saturated)")
        with self.lock:
            self.inflight_now += 1
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
        """Caller decided to redo this video. Drop its state so finalize won't
        run from late-arriving callbacks of the original pass."""
        with self.cond:
            self.pending.pop(video_id, None)
            self.cond.notify_all()

    def wait_all(self) -> None:
        with self.cond:
            while self.pending:
                self.cond.wait()
        self._stats_stop.set()
        self.pool.shutdown(wait=True)
        with self.store_lock:
            self.store.save()
        log.info(
            f"wait_all done: completed={self.completed} failed={self.failed_videos} "
            f"batches={self.batches_done} batch_errors={self.batches_failed}"
        )

    # -------- worker callbacks --------

    def _on_done(
        self, video_id: int, ts: np.ndarray, mirror_flag: int, fut: Future,
    ) -> None:
        vecs: np.ndarray | None
        try:
            vecs = fut.result()
        except Exception as e:
            console.print(f"[red]encode error vid={video_id}: {e}[/red]")
            log.warning(f"encode_error vid={video_id} mirror={mirror_flag}: {e}")
            vecs = None
            with self.lock:
                self.batches_failed += 1
        else:
            with self.lock:
                self.batches_done += 1

        # IMMEDIATELY flush to FAISS — do not retain vecs in RAM.
        if vecs is not None:
            try:
                with self.store_lock:
                    self.store.init_index(self.fe.dim)
                    self.store.add_vectors(
                        vecs, video_id, ts,
                        np.full(len(vecs), mirror_flag, dtype=np.int8),
                    )
            except Exception as e:
                console.print(f"[red]index write failed vid={video_id}: {e}[/red]")
                vecs = None     # treat as failure
        n_added = int(len(ts)) if (vecs is not None and mirror_flag == 0) else 0

        finalize_target: _VideoState | None = None
        with self.cond:
            st = self.pending.get(video_id)
            if st is not None:
                st.received += 1
                st.n_frames += n_added
                if vecs is None:
                    st.failed = True
                if st.decoded and st.received >= st.expected:
                    self.pending.pop(video_id, None)
                    finalize_target = st
                    self.cond.notify_all()
            self.inflight_now = max(0, self.inflight_now - 1)
        self.sem.release()
        if finalize_target is not None:
            self._finalize(finalize_target)

    def _finalize(self, st: _VideoState) -> None:
        with self.store_lock:
            if st.failed:
                self.store.mark_failed(st.video_id)
                with self.lock:
                    self.failed_videos += 1
                log.warning(f"finalize FAILED vid={st.video_id} "
                            f"({st.received}/{st.expected} batches, n_frames={st.n_frames})")
            else:
                self.store.update_n_frames(st.video_id, st.n_frames)
                self.store.mark_complete(st.video_id)
                with self.lock:
                    self.completed += 1
                log.info(f"finalize OK vid={st.video_id} "
                         f"batches={st.expected} n_frames={st.n_frames}")
            self._since_save += 1
            if self._since_save >= self.save_every:
                self.store.save()
                self._since_save = 0
                log.info(f"FAISS saved after {self.save_every} videos")
