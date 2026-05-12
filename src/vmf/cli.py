from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from vmf import __version__
from vmf.config import Config
from vmf.index import Store
from vmf.pipeline import (
    discover_videos,
    ensure_extractor,
    find_pairs,
    find_pairs_smooth,
    index_paths,
    query_against_index,
)
from vmf.ui import render_pairs, to_json

app = typer.Typer(
    help="Find overlapping fragments across a video collection.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


def _make_config(
    data_dir: Optional[Path],
    fps: Optional[float],
    model: Optional[str],
    device: Optional[str],
    no_mirror: bool,
    legacy_ransac: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Config:
    cfg = Config()
    if legacy_ransac:
        cfg.use_smooth = False
    if endpoint is not None:
        cfg.endpoint = endpoint
    if api_key is not None:
        cfg.api_key = api_key
    if data_dir is not None:
        cfg.data_dir = data_dir
    if fps is not None:
        cfg.fps = fps
    if model is not None:
        cfg.model = model
    if device is not None:
        cfg.device = device
    if no_mirror:
        cfg.mirror = False
    cfg.ensure_dirs()
    return cfg


# ---------- common options ----------

DATA_DIR_OPT = typer.Option(None, "--data-dir", help="Where to store the index.")
FPS_OPT = typer.Option(None, "--fps", help="Frames per second to sample.")
MODEL_OPT = typer.Option(None, "--model", help="auto | dinov2_vits14 | dinov2_vitb14")
DEVICE_OPT = typer.Option(None, "--device", help="auto | cpu | cuda")
NO_MIRROR_OPT = typer.Option(False, "--no-mirror", help="Disable horizontal-flip indexing.")
LEGACY_OPT = typer.Option(
    False, "--legacy-ransac",
    help="Use the old RANSAC + hardcoded-thresholds pipeline instead of "
         "the default smooth (Hough + permutation) pipeline.",
)
ENDPOINT_OPT = typer.Option(
    None, "--endpoint",
    help="OpenAI-compatible embeddings URL (e.g. http://host:8000/v1). "
         "When set, frames are sent over HTTP and no local model is loaded.",
)
API_KEY_OPT = typer.Option(
    None, "--api-key",
    help="Bearer token for --endpoint. Required when --endpoint is set.",
    envvar="VMF_API_KEY",
)


@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
):
    if version:
        console.print(f"video-match-finder {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def index(
    paths: list[Path] = typer.Argument(..., help="Video files or directories."),
    data_dir: Optional[Path] = DATA_DIR_OPT,
    fps: Optional[float] = FPS_OPT,
    model: Optional[str] = MODEL_OPT,
    device: Optional[str] = DEVICE_OPT,
    no_mirror: bool = NO_MIRROR_OPT,
    endpoint: Optional[str] = ENDPOINT_OPT,
    api_key: Optional[str] = API_KEY_OPT,
) -> None:
    """Add videos to the index without searching."""
    cfg = _make_config(data_dir, fps, model, device, no_mirror,
                       endpoint=endpoint, api_key=api_key)
    videos = discover_videos(paths)
    if not videos:
        console.print("[yellow]No videos found.[/yellow]")
        raise typer.Exit(1)
    console.print(f"Found {len(videos)} video(s).")
    store = Store(cfg.data_dir)
    fe = ensure_extractor(cfg)
    added = index_paths(videos, cfg, store, fe)
    console.print(f"[green]Indexed {added} new video(s).[/green]")


@app.command()
def scan(
    paths: list[Path] = typer.Argument(..., help="Directory (or files) to scan for overlaps."),
    data_dir: Optional[Path] = DATA_DIR_OPT,
    fps: Optional[float] = FPS_OPT,
    model: Optional[str] = MODEL_OPT,
    device: Optional[str] = DEVICE_OPT,
    no_mirror: bool = NO_MIRROR_OPT,
    legacy_ransac: bool = LEGACY_OPT,
    endpoint: Optional[str] = ENDPOINT_OPT,
    api_key: Optional[str] = API_KEY_OPT,
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write JSON results to this file."),
) -> None:
    """Index a collection (if not already) and report overlapping pairs."""
    cfg = _make_config(data_dir, fps, model, device, no_mirror, legacy_ransac,
                       endpoint=endpoint, api_key=api_key)
    videos = discover_videos(paths)
    if not videos:
        console.print("[yellow]No videos found.[/yellow]")
        raise typer.Exit(1)
    console.print(f"Found {len(videos)} video(s).")
    store = Store(cfg.data_dir)
    fe = ensure_extractor(cfg)
    index_paths(videos, cfg, store, fe)
    console.print("Searching for overlapping fragments…")
    results = (find_pairs_smooth if cfg.use_smooth else find_pairs)(cfg, store)
    render_pairs(results)
    if json_out is not None:
        json_out.write_text(to_json(results), encoding="utf-8")
        console.print(f"[dim]JSON written to {json_out}[/dim]")


@app.command()
def find(
    query: Path = typer.Argument(..., help="Query video file."),
    data_dir: Optional[Path] = DATA_DIR_OPT,
    fps: Optional[float] = FPS_OPT,
    model: Optional[str] = MODEL_OPT,
    device: Optional[str] = DEVICE_OPT,
    no_mirror: bool = NO_MIRROR_OPT,
    endpoint: Optional[str] = ENDPOINT_OPT,
    api_key: Optional[str] = API_KEY_OPT,
    json_out: Optional[Path] = typer.Option(None, "--json"),
) -> None:
    """Search the existing index for matches against a single query video."""
    cfg = _make_config(data_dir, fps, model, device, no_mirror,
                       endpoint=endpoint, api_key=api_key)
    store = Store(cfg.data_dir)
    if store.index is None or store.index.ntotal == 0:
        console.print("[red]Index is empty. Run `vmf index` or `vmf scan` first.[/red]")
        raise typer.Exit(1)
    fe = ensure_extractor(cfg)
    results = query_against_index(query, cfg, store, fe)
    render_pairs(results)
    if json_out is not None:
        json_out.write_text(to_json(results), encoding="utf-8")


@app.command()
def status(data_dir: Optional[Path] = DATA_DIR_OPT) -> None:
    """Show what's currently indexed."""
    cfg = _make_config(data_dir, None, None, None, False)
    store = Store(cfg.data_dir)
    videos = store.list_videos()
    n_vec = store.index.ntotal if store.index is not None else 0
    table = Table(title=f"Index: {cfg.data_dir}")
    table.add_column("ID", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Path")
    for v in videos:
        mins = int(v.duration // 60)
        secs = int(v.duration % 60)
        table.add_row(str(v.id), str(v.n_frames), f"{mins}:{secs:02d}", v.path)
    console.print(table)
    console.print(f"[dim]{len(videos)} video(s), {n_vec} vector(s) total[/dim]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="TCP port."),
    model: Optional[str] = MODEL_OPT,
    device: Optional[str] = DEVICE_OPT,
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="Bearer token to require. If omitted, a random one is generated.",
        envvar="VMF_API_KEY",
    ),
) -> None:
    """Run an OpenAI-compatible embeddings server backed by DINOv2."""
    from vmf.serve import run_server
    run_server(
        host=host, port=port,
        model=model or "auto",
        device=device or "auto",
        api_key=api_key,
    )


@app.command()
def doctor() -> None:
    """Diagnose the environment — Python, torch, GPU, ffmpeg — and suggest fixes."""
    import shutil
    import torch as _torch
    import psutil

    console.print(f"[bold]Python[/bold]: {__import__('sys').version.split()[0]}")
    console.print(f"[bold]torch[/bold]:  {_torch.__version__}")

    ffmpeg = shutil.which("ffmpeg")
    console.print(f"[bold]ffmpeg[/bold]: {ffmpeg or '[red]not found[/red]'}")

    mem = psutil.virtual_memory()
    console.print(f"[bold]RAM[/bold]:    {mem.available/1e9:.1f} GB free of {mem.total/1e9:.1f} GB")

    if not _torch.cuda.is_available():
        console.print("[bold]GPU[/bold]:    no CUDA-capable PyTorch installed — CPU mode.")
        console.print(
            "  • If you have a modern NVIDIA GPU and a recent driver, reinstall torch "
            "from a CUDA channel, e.g.:\n"
            "    [cyan]pip install --index-url https://download.pytorch.org/whl/cu124 "
            "--force-reinstall torch torchvision[/cyan]"
        )
        return

    name = _torch.cuda.get_device_name(0)
    cc = _torch.cuda.get_device_capability(0)
    console.print(f"[bold]GPU[/bold]:    {name} (compute capability {cc[0]}.{cc[1]})")

    try:
        _torch.zeros(1, device="cuda").add_(1).cpu()
        console.print("[green]GPU kernels work — the package will use CUDA automatically.[/green]")
    except Exception as e:
        console.print(f"[red]GPU kernels fail:[/red] {type(e).__name__}: {e}")
        cc_num = cc[0] * 10 + cc[1]
        if cc_num < 70:
            console.print(
                f"  Your GPU's compute capability ({cc[0]}.{cc[1]}) is older than what "
                "recent PyTorch wheels include kernels for (≥7.5). Options:\n"
                "    1. Use CPU — recommended for this hardware, performance gain on this "
                "GPU would be modest anyway.\n"
                "    2. Install torch 2.1 with cu118 (still ships sm_61 kernels), but it "
                "requires Python ≤3.11.\n"
                "    3. Build PyTorch from source with [cyan]TORCH_CUDA_ARCH_LIST='"
                f"{cc[0]}.{cc[1]}'[/cyan]."
            )
        else:
            console.print(
                "  The installed torch CUDA build doesn't match this driver. "
                "Try a different CUDA channel (cu121 / cu124 / cu128)."
            )


@app.command()
def reset(
    data_dir: Optional[Path] = DATA_DIR_OPT,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Erase the index."""
    cfg = _make_config(data_dir, None, None, None, False)
    if not yes:
        confirm = typer.confirm(f"Delete index at {cfg.data_dir}?")
        if not confirm:
            raise typer.Exit()
    store = Store(cfg.data_dir)
    store.reset()
    console.print("[green]Index cleared.[/green]")


if __name__ == "__main__":
    app()
