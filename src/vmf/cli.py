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
    inflight: Optional[int] = None,
    batch_size: Optional[int] = None,
    keyframes_only: bool = False,
    no_cropdetect: bool = False,
) -> Config:
    cfg = Config()
    if legacy_ransac:
        cfg.use_smooth = False
    if endpoint is not None:
        cfg.endpoint = endpoint
    if api_key is not None:
        cfg.api_key = api_key
    if inflight is not None and inflight >= 1:
        cfg.encode_inflight = inflight
    if batch_size is not None and batch_size >= 1:
        cfg.batch_size = batch_size
    if keyframes_only:
        cfg.keyframes_only = True
    if no_cropdetect:
        cfg.cropdetect = False
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
INFLIGHT_OPT = typer.Option(
    None, "--inflight",
    help="Encode batches kept in flight at once. Raise for high-RTT links "
         "(rule of thumb: bandwidth × RTT / batch_size). Default: 8.",
)
BATCH_SIZE_OPT = typer.Option(
    None, "--batch-size",
    help="Frames per HTTP request. Larger batches keep the TCP window warm "
         "and amortise per-request slow-start. Useful when uploading through "
         "SSH tunnels or high-RTT networks. Default: 32. Try 128 or 256 if "
         "the channel is under-utilised.",
)
KEYFRAMES_ONLY_OPT = typer.Option(
    False, "--keyframes-only",
    help="Decode only I-frames. 30–60× less CPU than uniform fps sampling; "
         "timestamps come from ffprobe so the algorithm still gets real PTS.",
)
NO_CROPDETECT_OPT = typer.Option(
    False, "--no-cropdetect",
    help="Skip the cropdetect pre-pass. Saves 1–3 seconds per video on "
         "collections that don't have letterboxes.",
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
    inflight: Optional[int] = INFLIGHT_OPT,
    batch_size: Optional[int] = BATCH_SIZE_OPT,
    keyframes_only: bool = KEYFRAMES_ONLY_OPT,
    no_cropdetect: bool = NO_CROPDETECT_OPT,
) -> None:
    """Add videos to the index without searching."""
    cfg = _make_config(data_dir, fps, model, device, no_mirror,
                       endpoint=endpoint, api_key=api_key, inflight=inflight,
                       batch_size=batch_size,
                       keyframes_only=keyframes_only, no_cropdetect=no_cropdetect)
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
    inflight: Optional[int] = INFLIGHT_OPT,
    batch_size: Optional[int] = BATCH_SIZE_OPT,
    keyframes_only: bool = KEYFRAMES_ONLY_OPT,
    no_cropdetect: bool = NO_CROPDETECT_OPT,
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write JSON results to this file."),
) -> None:
    """Index a collection (if not already) and report overlapping pairs."""
    cfg = _make_config(data_dir, fps, model, device, no_mirror, legacy_ransac,
                       endpoint=endpoint, api_key=api_key, inflight=inflight,
                       batch_size=batch_size,
                       keyframes_only=keyframes_only, no_cropdetect=no_cropdetect)
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
    inflight: Optional[int] = INFLIGHT_OPT,
    batch_size: Optional[int] = BATCH_SIZE_OPT,
    keyframes_only: bool = KEYFRAMES_ONLY_OPT,
    no_cropdetect: bool = NO_CROPDETECT_OPT,
    json_out: Optional[Path] = typer.Option(None, "--json"),
) -> None:
    """Search the existing index for matches against a single query video."""
    cfg = _make_config(data_dir, fps, model, device, no_mirror,
                       endpoint=endpoint, api_key=api_key, inflight=inflight,
                       batch_size=batch_size,
                       keyframes_only=keyframes_only, no_cropdetect=no_cropdetect)
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
    import psutil
    from vmf.torch_install import detect_plan

    console.print(f"[bold]Python[/bold]: {__import__('sys').version.split()[0]}")
    try:
        import torch as _torch
        console.print(f"[bold]torch[/bold]:  {_torch.__version__}")
        has_torch = True
    except ImportError:
        _torch = None
        has_torch = False
        console.print("[bold]torch[/bold]:  [red]not installed[/red]")

    ffmpeg = shutil.which("ffmpeg")
    console.print(f"[bold]ffmpeg[/bold]: {ffmpeg or '[red]not found[/red]'}")

    mem = psutil.virtual_memory()
    console.print(f"[bold]RAM[/bold]:    {mem.available/1e9:.1f} GB free of {mem.total/1e9:.1f} GB")

    plan = detect_plan()
    console.print(f"[bold]GPU plan[/bold]: {plan.reason}")
    if not has_torch:
        console.print(
            "  Run [cyan]vmf install-torch[/cyan] to install the matching wheel."
        )
        return

    if not _torch.cuda.is_available():
        console.print("[bold]GPU[/bold]:    PyTorch is CPU-only (no CUDA available).")
        if plan.channel != "cpu":
            console.print(
                f"  Detected {plan.reason}. To enable GPU run "
                f"[cyan]vmf install-torch --force[/cyan] (uses {plan.channel})."
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
        if plan.legacy:
            console.print(
                f"  Compute capability {cc[0]}.{cc[1]} is too old for modern PyTorch "
                "wheels. Either use CPU or install legacy torch (≤2.1 + cu118)."
            )
        else:
            console.print(
                f"  Installed torch CUDA build doesn't match the driver. "
                f"Run [cyan]vmf install-torch --force[/cyan] to reinstall as {plan.channel}."
            )


@app.command(name="install-torch")
def install_torch(
    force: bool = typer.Option(False, "--force",
        help="Pass --force-reinstall to pip (use when torch is already installed)."),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Print the pip command without running it."),
) -> None:
    """Install torch + torchvision from the CUDA channel matching this host.

    Inspects `nvidia-smi` to pick the right `--index-url` so a fresh install
    doesn't silently grab a too-new CUDA wheel that fails at runtime.
    """
    from vmf.torch_install import detect_plan, install_command, run_install

    plan = detect_plan()
    console.print(f"[bold]Plan[/bold]: {plan.reason}")
    if plan.legacy:
        console.print(
            "[red]Legacy GPU detected.[/red] Modern PyTorch wheels do not ship "
            f"kernels for this hardware. Manual install required, e.g.:\n"
            "  [cyan]pip install torch==2.1.2 torchvision==0.16.2 "
            "--index-url https://download.pytorch.org/whl/cu118[/cyan]\n"
            "(Python ≤ 3.11 only.)"
        )
        raise typer.Exit(1)

    cmd = install_command(plan, force=force)
    console.print(f"[cyan]Running:[/cyan] {' '.join(cmd)}")
    if dry_run:
        return
    rc = run_install(plan, force=force)
    if rc != 0:
        console.print(f"[red]pip exited with code {rc}[/red]")
        raise typer.Exit(rc)
    console.print("[green]Done.[/green] Re-run [cyan]vmf doctor[/cyan] to verify.")


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
