from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psutil
import torch
from rich.console import Console

console = Console(stderr=True)

# Approximate fp32 weight footprint of each variant (megabytes).
MODEL_CATALOG = [
    # name              dim   approx_mb
    ("dinov2_vits14",   384,   90),
    ("dinov2_vitb14",   768,  340),
]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class FeatureExtractor:
    name: str
    dim: int
    device: str
    model: torch.nn.Module

    @torch.inference_mode()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """frames: (B, H, W, 3) uint8 → (B, dim) float32, L2-normalized."""
        x = torch.from_numpy(frames).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - MEAN.to(self.device)) / STD.to(self.device)
        feats = self.model(x)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().numpy().astype(np.float32)


def _cuda_kernel_works() -> bool:
    """`cuda.is_available()` can be True while the GPU's compute capability is
    unsupported by the installed PyTorch wheel (common on Pascal/older). Probe
    with a tiny op and treat kernel failures as 'unusable'."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda").add_(1).cpu()
        return True
    except Exception:
        return False


def _resolve_device(pref: str) -> str:
    if pref == "cuda":
        if not _cuda_kernel_works():
            raise RuntimeError("CUDA requested but no usable GPU kernel — see `vmf doctor`.")
        return "cuda"
    if pref == "cpu":
        return "cpu"
    if _cuda_kernel_works():
        return "cuda"
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            console.print(
                f"[yellow]Note:[/yellow] GPU detected ({name}, compute capability "
                f"{cc[0]}.{cc[1]}) but the installed PyTorch wheel has no compatible kernels. "
                "Falling back to CPU. Run [cyan]vmf doctor[/cyan] for fix suggestions."
            )
        except Exception:
            console.print("[yellow]Note:[/yellow] GPU detected but unusable — falling back to CPU.")
    return "cpu"


def _available_memory_mb(device: str) -> float:
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 * 1024)
    return psutil.virtual_memory().available / (1024 * 1024)


def _pick_model(requested: str, device: str) -> tuple[str, int]:
    """Return (model_name, dim) honoring the 60% RAM/VRAM budget."""
    avail_mb = _available_memory_mb(device)
    budget_mb = avail_mb * 0.60

    if requested == "auto":
        candidates = list(reversed(MODEL_CATALOG))  # try largest first
    else:
        match = next((m for m in MODEL_CATALOG if m[0] == requested), None)
        if match is None:
            raise ValueError(f"unknown model: {requested}")
        candidates = [match] + [m for m in reversed(MODEL_CATALOG) if m[0] != requested]

    chosen = None
    for name, dim, mb in candidates:
        if mb * 2 <= budget_mb:  # weights + activations rough headroom
            chosen = (name, dim, mb)
            break

    if chosen is None:
        smallest = MODEL_CATALOG[0]
        console.print(
            f"[yellow]Warning:[/yellow] only {avail_mb:.0f} MB free on {device}; "
            f"forcing {smallest[0]} (~{smallest[2]} MB). Expect tight memory."
        )
        chosen = smallest

    if requested != "auto" and chosen[0] != requested:
        console.print(
            f"[yellow]Warning:[/yellow] {requested} would exceed 60% of available "
            f"{device.upper()} memory; falling back to {chosen[0]}."
        )
    return chosen[0], chosen[1]


def load_extractor(model: str = "auto", device: str = "auto") -> FeatureExtractor:
    dev = _resolve_device(device)
    name, dim = _pick_model(model, dev)
    console.print(f"[cyan]Loading {name} on {dev}…[/cyan]")
    net = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    net.eval().to(dev)
    return FeatureExtractor(name=name, dim=dim, device=dev, model=net)


@dataclass
class RemoteFeatureExtractor:
    """Talks to an OpenAI-compatible /v1/embeddings endpoint.

    Same `name`, `dim`, `encode()` surface as the local FeatureExtractor, so
    the rest of the pipeline doesn't know whether it's calling DINOv2 in-process
    or hitting an HTTP server (local, remote, or a 3rd-party provider).
    """
    name: str
    dim: int
    endpoint: str
    api_key: str
    device: str = "remote"

    def __post_init__(self):
        import httpx
        self._client = httpx.Client(
            base_url=self.endpoint.rstrip("/"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """frames: (B, H, W, 3) uint8 → (B, dim) float32, L2-normalised on the server."""
        import base64
        import io
        from PIL import Image

        inputs: list[str] = []
        for fr in frames:
            buf = io.BytesIO()
            Image.fromarray(fr).save(buf, format="JPEG", quality=88)
            inputs.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        resp = self._client.post(
            "/embeddings",
            json={"model": self.name, "input": inputs, "encoding_format": "float"},
        )
        resp.raise_for_status()
        payload = resp.json()
        data = sorted(payload["data"], key=lambda d: d["index"])
        out = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        if out.shape != (len(frames), self.dim):
            raise RuntimeError(
                f"server returned shape {out.shape}, expected ({len(frames)}, {self.dim})"
            )
        return out


def load_remote_extractor(endpoint: str, api_key: str) -> RemoteFeatureExtractor:
    """Probe /v1/models to learn the served model name and embedding dim."""
    import httpx
    base = endpoint.rstrip("/")
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        info = r.json()
    if not info.get("data"):
        raise RuntimeError(f"{base}/models returned no models")
    m = info["data"][0]
    name = m["id"]
    dim = m.get("dimension")
    if dim is None:
        # OpenAI-style endpoints don't include dim in /models; probe with a 1-px image
        import base64
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (224, 224)).save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{base}/embeddings",
                       headers={"Authorization": f"Bearer {api_key}"},
                       json={"model": name, "input": [b64]})
            r.raise_for_status()
            dim = len(r.json()["data"][0]["embedding"])
    console.print(f"[cyan]Connected to {base}  (model={name}, dim={dim})[/cyan]")
    return RemoteFeatureExtractor(name=name, dim=int(dim), endpoint=base, api_key=api_key)
