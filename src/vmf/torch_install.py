"""Detect the right PyTorch CUDA build for the current GPU/driver.

PyTorch publishes the same `torch` version under several CUDA channels:

    https://download.pytorch.org/whl/cpu      - no GPU
    https://download.pytorch.org/whl/cu118    - CUDA 11.8 (driver ≥ 450, oldest GPUs)
    https://download.pytorch.org/whl/cu121    - CUDA 12.1 (driver ≥ 525)
    https://download.pytorch.org/whl/cu124    - CUDA 12.4 (driver ≥ 525)
    https://download.pytorch.org/whl/cu126    - CUDA 12.6 (driver ≥ 525)
    https://download.pytorch.org/whl/cu128    - CUDA 12.8 (driver ≥ 555)
    https://download.pytorch.org/whl/cu130    - CUDA 13.0 (driver ≥ 580)

PyPI's default wheel is whichever channel PyTorch ships as the default,
typically the newest. Picking a wheel matching the host driver avoids the
"GPU detected but no compatible kernel" failure mode.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class TorchPlan:
    channel: str           # "cpu" | "cu118" | "cu121" | "cu124" | "cu126" | "cu128" | "cu130"
    reason: str            # human-readable explanation
    index_url: str | None  # None for cpu (default PyPI also works)
    legacy: bool = False   # True ⇒ requires old torch (≤ 2.1) for pre-Turing GPUs


# Driver-major-version → preferred CUDA channel. Older drivers can run newer
# CUDA via forward compatibility in some cases, but to minimise surprises we
# pick the channel known to ship binaries that match the driver's CUDA level.
_DRIVER_TO_CHANNEL: list[tuple[int, str]] = [
    (580, "cu130"),
    (555, "cu128"),
    (545, "cu126"),
    (525, "cu124"),
    (515, "cu121"),
    (450, "cu118"),
]


def _query_nvidia_smi() -> tuple[str, str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,compute_cap",
             "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        return None
    line = out.splitlines()[0] if out else ""
    if "," not in line:
        return None
    drv, cc = (s.strip() for s in line.split(","))
    return drv, cc


def detect_plan() -> TorchPlan:
    info = _query_nvidia_smi()
    if info is None:
        return TorchPlan(
            channel="cpu",
            reason="no nvidia-smi / no NVIDIA GPU detected",
            index_url="https://download.pytorch.org/whl/cpu",
        )
    driver, cc = info
    try:
        driver_major = int(driver.split(".")[0])
    except ValueError:
        return TorchPlan(
            channel="cpu",
            reason=f"could not parse driver version '{driver}'",
            index_url="https://download.pytorch.org/whl/cpu",
        )
    try:
        cc_major, cc_minor = (int(x) for x in cc.split(".")[:2])
    except ValueError:
        cc_major = cc_minor = 0
    cc_num = cc_major * 10 + cc_minor

    # Pre-Turing GPUs (sm < 75) are not in modern wheels. They need torch
    # ≤ 2.1 with cu118 (which still includes those kernels), or build from
    # source. We flag this so the install command refuses and tells the user.
    if cc_num and cc_num < 75:
        return TorchPlan(
            channel="cu118",
            reason=(
                f"GPU compute capability {cc} predates Turing (sm75); "
                "modern PyTorch wheels do not include kernels for it. "
                "Use torch ≤ 2.1 with cu118, or run on CPU."
            ),
            index_url="https://download.pytorch.org/whl/cu118",
            legacy=True,
        )

    for min_drv, ch in _DRIVER_TO_CHANNEL:
        if driver_major >= min_drv:
            return TorchPlan(
                channel=ch,
                reason=f"NVIDIA driver {driver} → {ch.upper()} (compute cap {cc})",
                index_url=f"https://download.pytorch.org/whl/{ch}",
            )

    return TorchPlan(
        channel="cpu",
        reason=f"NVIDIA driver {driver} is too old for any supported CUDA wheel",
        index_url="https://download.pytorch.org/whl/cpu",
    )


def install_command(plan: TorchPlan, *, force: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    if plan.index_url:
        cmd += ["--index-url", plan.index_url]
    if force:
        cmd.append("--force-reinstall")
    cmd += ["torch", "torchvision"]
    return cmd


def run_install(plan: TorchPlan, *, force: bool = False,
                extra_args: list[str] | None = None) -> int:
    cmd = install_command(plan, force=force)
    if extra_args:
        cmd += extra_args
    return subprocess.call(cmd)
