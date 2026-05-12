from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class FFmpegError(RuntimeError):
    pass


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


def list_keyframe_pts(path: Path) -> list[float]:
    """Enumerate the timestamp (seconds) of every keyframe in the video.

    Used by `iter_keyframes` to assign real PTS to each decoded I-frame.
    Returns [] on failure; the caller should fall back to uniform sampling.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=120)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    pts: list[float] = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        flags = parts[1]
        if "K" not in flags:        # ffprobe marks keyframes with 'K' in flags
            continue
        pts_str = parts[0]
        if not pts_str or pts_str == "N/A":
            continue
        try:
            pts.append(float(pts_str))
        except ValueError:
            continue
    return pts


def iter_keyframes(
    path: Path,
    size: int,
    crop: str | None = None,
    *,
    min_std: float = 0.0,
    hwaccel: bool = False,
) -> Iterator[tuple[float, np.ndarray]]:
    """Decode only keyframes (I-frames) — ~30–60× less CPU than full decode.

    Timestamps come from `list_keyframe_pts` (ffprobe), so they're real PTS
    even though the spacing is irregular. The downstream algorithm (Hough +
    permutation) handles non-uniform t natively.
    """
    pts_list = list_keyframe_pts(path)
    if not pts_list:
        return

    vf = []
    if crop:
        vf.append(crop)
    vf.append(f"scale={size}:{size}:force_original_aspect_ratio=decrease")
    vf.append(f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel and _hwaccel_available():
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-skip_frame", "nokey",
        "-i", str(path),
        # -fps_mode passthrough is the modern name (ffmpeg ≥5.0); the legacy
        # -vsync passthrough still works in 5/6 but is removed in 7+. Either
        # way the intent is: keep decoded frames as-is, no duplication, no drop.
        "-fps_mode", "passthrough",
        "-vf", ",".join(vf),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
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
            if idx >= len(pts_list):
                break    # ffmpeg produced more frames than ffprobe enumerated; bail safely
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
            t = pts_list[idx]
            idx += 1
            if min_std > 0.0 and float(arr.std()) < min_std:
                continue
            yield t, arr
    finally:
        proc.stdout.close()
        proc.wait(timeout=5)
        if proc.returncode not in (0, None):
            err = (proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else "")
            if idx == 0:
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
