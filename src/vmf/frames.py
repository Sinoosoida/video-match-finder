from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class FFmpegError(RuntimeError):
    pass


class TooManyKeyframesError(FFmpegError):
    """Decoder yielded more frames than the per-video sanity threshold.

    Catches videos where `-skip_frame nokey` silently fails at the decoder
    level — notably AV1 (libdav1d, libaom), where the skip-frame hint is
    not implemented. Without this check, such files would flood the index
    with one vector per decoded frame instead of one per keyframe.
    """


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float


def probe(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", "-select_streams", "v:0", str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise FFmpegError(f"ffprobe failed for {path}: {e}") from e
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams:
        raise FFmpegError(f"no video stream in {path}")
    s = streams[0]
    fmt = info.get("format", {})
    duration = float(s.get("duration") or fmt.get("duration") or 0.0)
    num, den = (s.get("avg_frame_rate") or "0/1").split("/")
    fps = (float(num) / float(den)) if float(den) else 0.0
    return VideoInfo(duration=duration, width=int(s.get("width", 0)),
                     height=int(s.get("height", 0)), fps=fps)


_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def detect_crop(path: Path, sample_seconds: float = 30.0) -> str | None:
    """Return ffmpeg `crop=W:H:X:Y` string if non-trivial letterbox detected.

    Uses `-skip_frame nokey` so only keyframes are decoded — letterbox borders
    are structural and visible on every frame, so I-frames are enough and the
    pre-pass is ~10× cheaper than full decode.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-skip_frame", "nokey",
        "-ss", "0", "-t", str(sample_seconds),
        "-i", str(path), "-vf", "cropdetect=24:16:0", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    matches = _CROP_RE.findall(proc.stderr or "")
    if not matches:
        return None
    w, h, x, y = matches[-1]
    return f"crop={w}:{h}:{x}:{y}"


_SHOWINFO_PTS_RE = re.compile(r"pts_time:\s*([\-\d.eE+]+)")


def aggregate_crop(
    crops: list[tuple[int, int, int, int]],
    src_w: int, src_h: int,
) -> tuple[int, int, int, int] | None:
    """Robust per-video crop from a list of per-frame cropdetect outputs.

    Filters out clearly degenerate frames (where detected content is <50% of
    the source area — usually fade/intro frames where cropdetect couldn't
    find borders), then takes the median across the 4 coordinates. Median
    is resilient to outliers from noisy frames (stray bright pixels, fades,
    transient overlays).
    """
    if not crops:
        return None
    import numpy as _np
    arr = _np.asarray(crops, dtype=_np.int64)        # (N, 4): W, H, X, Y
    src_area = src_w * src_h
    if src_area <= 0:
        return None
    area = arr[:, 0] * arr[:, 1]
    keep = area >= 0.5 * src_area
    if keep.sum() < 1:                               # need at least one trustworthy frame
        return None
    arr = arr[keep]
    return (int(_np.median(arr[:, 0])), int(_np.median(arr[:, 1])),
            int(_np.median(arr[:, 2])), int(_np.median(arr[:, 3])))


def is_significant_crop(
    crop: tuple[int, int, int, int],
    src_w: int, src_h: int,
    tolerance: float = 0.97,
) -> bool:
    """True if applying this crop would meaningfully shrink the frame."""
    w, h, _x, _y = crop
    return w < tolerance * src_w or h < tolerance * src_h


# Per-video keyframe budget. Most real videos have 0.1–2 keyframes/sec;
# even dense legitimate encodings rarely exceed 3. A 5 kf/s cap is generous
# for normal content and catches the "decoder didn't filter" failure mode
# where extracted count scales with source fps (24–60+ kf/s).
MAX_KEYFRAMES_PER_SECOND = 5.0
# Absolute physical upper bound used when metadata duration is missing/zero.
# 3 hours × 60 fps — no realistic video should yield more frames than this.
ABSOLUTE_KEYFRAMES_CAP = 3 * 3600 * 60   # 648 000


def max_keyframes_for(info: VideoInfo) -> int:
    """Per-video upper bound on keyframes for the safety check in
    `iter_keyframes`. Tighter when duration is known; falls back to a
    very loose cap when it isn't."""
    if info.duration <= 0:
        return ABSOLUTE_KEYFRAMES_CAP
    # +100 floor so short clips (< 20 s) still have a workable threshold.
    soft = int(info.duration * MAX_KEYFRAMES_PER_SECOND) + 100
    return min(soft, ABSOLUTE_KEYFRAMES_CAP)


def iter_keyframes(
    path: Path,
    size: int,
    crop: str | None = None,
    *,
    min_std: float = 0.0,
    hwaccel: bool = False,
    crops_out: list[tuple[int, int, int, int]] | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Decode only keyframes (I-frames) — ~30–60× less CPU than full decode.

    Timestamps come from ffmpeg's `showinfo` filter, parsed concurrently from
    stderr in a worker thread. This avoids a separate ffprobe pass (which on
    slow HDDs doubles the disk I/O per video), so each file is read exactly
    once. The downstream algorithm (Hough + permutation) handles non-uniform
    timestamps natively.

    If `crops_out` is supplied (and `crop` is not), `cropdetect` is added to
    the filter chain at source-resolution. Per-frame `(W, H, X, Y)` tuples are
    appended; the caller can aggregate them after iteration to decide whether
    a second pass with crop pre-applied is warranted.
    """
    # `select=eq(pict_type,I)` is a safety net for codecs whose decoders
    # silently ignore `-skip_frame nokey` (AV1 via libdav1d/libaom is the
    # known case — the hint is just not implemented there). On H.264/HEVC
    # the decoder has already filtered out non-key packets so this is a
    # tautology that produces byte-identical output (verified by md5).
    # On AV1 this is what actually filters keyframes — at the cost of full
    # decoding (the decoder can't skip frames), but the resulting frame
    # count is correct.
    vf = ["select='eq(pict_type,I)'"]
    if crop:
        vf.append(crop)
    if crops_out is not None and crop is None:
        # cropdetect must run BEFORE scale (it analyses border darkness at
        # source resolution). reset_count=1 → one detection per frame, not
        # cumulative. Passthrough: frames continue unmodified to the next
        # filter, so we get the crop info "for free" along the same pass.
        vf.append("cropdetect=limit=24:round=2:reset_count=1")
    vf.append("showinfo")                      # logs `pts_time:…` per output frame to stderr
    vf.append(f"scale={size}:{size}:force_original_aspect_ratio=decrease")
    vf.append(f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info"]   # info level needed for showinfo
    if hwaccel and _hwaccel_available():
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-skip_frame", "nokey",
        "-i", str(path),
        # -fps_mode passthrough keeps frames as-is (no duplication, no drop);
        # available since ffmpeg 5.0, replaces deprecated -vsync passthrough.
        "-fps_mode", "passthrough",
        "-vf", ",".join(vf),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
    assert proc.stdout is not None and proc.stderr is not None

    # Stream timestamps off stderr concurrently while we read frames off stdout.
    pts_buf: list[float] = []
    pts_done = threading.Event()

    def _reader() -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                if b"showinfo" in raw:
                    m = _SHOWINFO_PTS_RE.search(raw.decode("utf-8", "ignore"))
                    if m:
                        try:
                            pts_buf.append(float(m.group(1)))
                        except ValueError:
                            pass
                if crops_out is not None and b"Parsed_cropdetect" in raw:
                    m = _CROP_RE.search(raw.decode("utf-8", "ignore"))
                    if m:
                        try:
                            crops_out.append((int(m.group(1)), int(m.group(2)),
                                              int(m.group(3)), int(m.group(4))))
                        except ValueError:
                            pass
        finally:
            pts_done.set()

    threading.Thread(target=_reader, daemon=True).start()

    frame_bytes = size * size * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if max_frames is not None and idx >= max_frames:
                raise TooManyKeyframesError(
                    f"{path.name}: yielded > {max_frames} keyframes — the "
                    f"decoder is probably not honouring `-skip_frame nokey` "
                    f"for this codec (typical with AV1)"
                )
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
            # showinfo line for this frame usually arrives just before the raw
            # bytes; spin briefly waiting if not.
            spin = 0
            while len(pts_buf) <= idx and not pts_done.is_set() and spin < 200:
                time.sleep(0.005)
                spin += 1
            if len(pts_buf) > idx:
                t = pts_buf[idx]
            else:
                # stderr lagged or filter quirk — fall back to monotonic index
                t = float(idx) * 2.0
            idx += 1
            if min_std > 0.0 and float(arr.std()) < min_std:
                continue
            yield t, arr
    finally:
        proc.stdout.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.returncode not in (0, None) and idx == 0:
            err = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
            raise FFmpegError(f"ffmpeg failed for {path}: {err.strip()}")


_HW_PROBED: bool | None = None


def _hwaccel_available() -> bool:
    """Probe whether ffmpeg can actually decode via NVDEC by running a tiny test.
    Some GPUs (Pascal/older) advertise CUDA but hang trying to init NVDEC."""
    global _HW_PROBED
    if _HW_PROBED is not None:
        return _HW_PROBED
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        if "cuda" not in (out.stdout or ""):
            _HW_PROBED = False
            return False
        # Actually try decoding a 0.1s synthetic clip with hwaccel; if it hangs or
        # errors, fall back to software for the whole session.
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-hwaccel", "cuda", "-f", "lavfi", "-i", "testsrc=duration=0.1:size=64x64:rate=10",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=8,
        )
        _HW_PROBED = probe.returncode == 0
    except Exception:
        _HW_PROBED = False
    return _HW_PROBED


def iter_frames(
    path: Path,
    fps: float,
    size: int,
    crop: str | None = None,
    *,
    min_std: float = 0.0,
    hwaccel: bool = False,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_seconds, HxWx3 uint8 RGB) at the given sample rate.

    Frames whose pixel standard deviation falls below `min_std` are skipped — this
    removes near-black/fade/solid intro frames that otherwise produce spurious
    identical embeddings across unrelated videos.
    """
    vf = []
    if crop:
        vf.append(crop)
    vf.append(f"fps={fps}")
    vf.append(f"scale={size}:{size}:force_original_aspect_ratio=decrease")
    vf.append(f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel and _hwaccel_available():
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i", str(path),
        "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
    assert proc.stdout is not None
    frame_bytes = size * size * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
            ts = idx / fps
            idx += 1
            if min_std > 0.0 and float(arr.std()) < min_std:
                continue
            yield ts, arr
    finally:
        proc.stdout.close()
        proc.wait(timeout=5)
        if proc.returncode not in (0, None):
            err = (proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else "")
            if idx == 0:
                raise FFmpegError(f"ffmpeg failed for {path}: {err.strip()}")
