"""OpenAI-compatible embeddings server.

`vmf serve` loads a DINOv2 model and exposes two HTTP endpoints:

    POST /v1/embeddings   — same shape as OpenAI's text-embedding API, but the
                            `input` array carries base64-encoded images.
    GET  /v1/models       — lists the served model and its embedding dimension.

A random bearer token is generated on startup and printed to stderr along with
the exact command-line clients should use. The standardised shape means any
client that already talks to OpenAI / Cohere / HF Inference embedding endpoints
can be pointed at this server with the same code path.
"""
from __future__ import annotations

import base64
import io
import secrets
import sys

import numpy as np
from PIL import Image
from rich.console import Console

from vmf.features import FeatureExtractor, load_extractor

console = Console(stderr=True)


# ---- module-level schemas (Pydantic v2 needs them out of function scope) ----

try:
    from pydantic import BaseModel

    class EmbeddingRequest(BaseModel):
        model: str
        input: list[str]
        encoding_format: str = "float"

    class EmbeddingData(BaseModel):
        object: str = "embedding"
        index: int
        embedding: list[float]

    class EmbeddingResponse(BaseModel):
        object: str = "list"
        model: str
        data: list[EmbeddingData]
        usage: dict = {"prompt_tokens": 0, "total_tokens": 0}
except ImportError:
    EmbeddingRequest = EmbeddingData = EmbeddingResponse = None  # type: ignore


def _build_app(extractor: FeatureExtractor, api_key: str):
    try:
        from fastapi import Body, Depends, FastAPI, Header, HTTPException
    except ImportError as e:
        raise RuntimeError(
            "Server dependencies missing. Install with: "
            "pip install 'video-match-finder[serve]'"
        ) from e

    app = FastAPI(title="video-match-finder embeddings", version="0.1")

    async def _auth(authorization: str | None = Header(default=None)) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing Authorization: Bearer <token>")
        if not secrets.compare_digest(authorization[7:], api_key):
            raise HTTPException(401, "invalid api key")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "model": extractor.name, "dim": extractor.dim}

    @app.get("/v1/models", dependencies=[Depends(_auth)])
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{
                "id": extractor.name,
                "object": "model",
                "owned_by": "vmf",
                "dimension": extractor.dim,
            }],
        }

    @app.post("/v1/embeddings", response_model=EmbeddingResponse,
              dependencies=[Depends(_auth)])
    async def embeddings(req: EmbeddingRequest = Body(...)) -> EmbeddingResponse:
        if req.model != extractor.name and req.model != "auto":
            raise HTTPException(
                400, f"model '{req.model}' not loaded (server has '{extractor.name}')",
            )

        batch: list[np.ndarray] = []
        for i, b64 in enumerate(req.input):
            try:
                # strip optional data-URI prefix
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if img.size != (224, 224):
                    img = img.resize((224, 224), Image.BICUBIC)
                batch.append(np.asarray(img, dtype=np.uint8))
            except Exception as e:
                raise HTTPException(400, f"input[{i}]: bad image ({type(e).__name__}: {e})")

        arr = np.stack(batch, axis=0)
        vecs = extractor.encode(arr)
        return EmbeddingResponse(
            model=extractor.name,
            data=[EmbeddingData(index=i, embedding=v.tolist())
                  for i, v in enumerate(vecs)],
        )

    return app


def run_server(
    *, host: str = "0.0.0.0", port: int = 8000,
    model: str = "auto", device: str = "auto",
    api_key: str | None = None,
) -> None:
    """Load model, print connection info, run uvicorn until killed."""
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "Server dependencies missing. Install with: "
            "pip install 'video-match-finder[serve]'"
        ) from e

    extractor = load_extractor(model=model, device=device)
    if api_key is None:
        api_key = secrets.token_urlsafe(24)
    app = _build_app(extractor, api_key)

    # Display connection details. Printed to stderr so user can pipe stdout
    # elsewhere; this is the one user-facing message that matters.
    base_url = f"http://{host}:{port}/v1"
    public_host = host if host != "0.0.0.0" else "<server-ip>"
    public_url = f"http://{public_host}:{port}/v1"
    console.print()
    console.print(f"[green bold]vmf serve[/green bold] — embedding service ready")
    console.print(f"  Model:    [cyan]{extractor.name}[/cyan]  ({extractor.dim}-d)")
    console.print(f"  URL:      {public_url}")
    console.print(f"  API key:  [yellow]{api_key}[/yellow]")
    console.print()
    console.print("Point a client at this server:")
    console.print(
        f"  [dim]vmf scan <videos> --endpoint {public_url} --api-key {api_key}[/dim]"
    )
    console.print(
        f"  [dim]curl -H 'Authorization: Bearer {api_key}' {base_url}/models[/dim]"
    )
    console.print()
    sys.stderr.flush()

    uvicorn.run(app, host=host, port=port, log_level="warning",
                access_log=False)
