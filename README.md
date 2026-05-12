# video-match-finder

Find overlapping fragments across a video collection — partial near-duplicate detection
for "is the same clip hiding in another video, possibly cropped, sped up, mirrored, or
recolored?". Useful for cleaning up archives where you want to keep the longest /
highest-quality copy of each piece of footage.

It samples frames, embeds them with [DINOv2](https://github.com/facebookresearch/dinov2),
indexes them with FAISS, and recovers matching segments by RANSAC-fitting a line through
the per-frame neighbour graph. Speed changes show up as the slope of that line, mirrored
copies are caught by indexing both orientations.

## Install (isolated)

The recommended path on any Linux is **pipx** — it puts the tool in its own venv so it
can't break system Python:

```bash
sudo pacman -S --needed ffmpeg python-pipx        # Manjaro / Arch
pipx install .                                    # from a checkout
pipx install video-match-finder                   # once published to PyPI
```

If you don't want pipx, use a plain venv:

```bash
python -m venv ~/.venvs/vmf
~/.venvs/vmf/bin/pip install .
ln -s ~/.venvs/vmf/bin/vmf ~/.local/bin/vmf
```

The first run will download the DINOv2 weights (~90 MB for the small variant) into
`~/.cache/torch/hub`. The index lives at `~/.local/share/video-match-finder/` by default
and can be moved with `--data-dir` or `$VMF_HOME`.

GPU is optional but ~30× faster than CPU for embedding. PyPI's default torch wheel
is CPU-only — that works on any machine. To enable GPU, reinstall torch from a
PyTorch CUDA channel matching your driver and GPU:

| Your GPU                                  | Recommended channel | Notes                                  |
|-------------------------------------------|---------------------|----------------------------------------|
| RTX 20xx and newer (compute ≥ 7.5)        | `cu124` or `cu128`  | needs driver ≥ 525 (cu124) / 545 (cu128) |
| GTX 9xx/10xx, Maxwell/Pascal (compute < 7.5) | `cu118` + torch 2.1 | only ships kernels for old CCs; requires Python ≤ 3.11 |
| No NVIDIA GPU                             | default PyPI wheel  | CPU mode                               |

Example (modern GPU):

```bash
pipx runpip video-match-finder install --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall torch torchvision
```

Not sure what you have? Run `vmf doctor` after install — it prints your GPU,
its compute capability, and the exact command to fix any mismatch.

## Quick start

```bash
# Scan a folder, build the index, and print every overlapping pair.
vmf scan ~/Videos/Archive

# Inspect what's indexed.
vmf status

# Search the existing index for matches of one new clip.
vmf find ~/Videos/maybe_a_dup.mp4

# Save machine-readable output too.
vmf scan ~/Videos/Archive --json results.json

# Throw it all away.
vmf reset
```

Output looks like:

```
A.mp4  ↔  B.mp4
┌─────────────┬─────────────┬───────┬────────┬───────┬─────────┐
│ A range     │ B range     │ Speed │ Mirror │ Score │ Inliers │
├─────────────┼─────────────┼───────┼────────┼───────┼─────────┤
│ 0:12–1:47   │ 3:04–4:14   │ ×1.25 │   —    │ 0.812 │      37 │
└─────────────┴─────────────┴───────┴────────┴───────┴─────────┘
  A: /home/u/Videos/A.mp4
  B: /home/u/Videos/B.mp4
```

`Score` is the mean cosine similarity along the matched path (0–1). `Speed` is the
playback ratio of B relative to A — `×1.25` means B is 25% faster.

## How model selection works

By default vmf checks free RAM (or VRAM on CUDA) and picks the largest DINOv2 variant
whose weights fit in **60 % of available memory**. If you ask for `dinov2_vitb14` on a
machine that can't fit it, you get a warning and an automatic fall-back to
`dinov2_vits14`. Force a choice with `--model dinov2_vits14` or `--model dinov2_vitb14`.

## Tuning

All settings have sensible defaults. The knobs that matter:

| Flag                      | Default | Effect                                                |
|---------------------------|---------|-------------------------------------------------------|
| `--fps`                   | 2       | Frame sampling rate. Higher = more recall, slower.    |
| `--model`                 | auto    | DINOv2 variant.                                       |
| `--device`                | auto    | `cuda` if available, else `cpu`.                      |
| `--no-mirror`             | off     | Skip mirrored indexing — halves time/memory.          |

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/vmf --help
```

Layout:

```
src/vmf/
├── frames.py     ffmpeg → numpy frame stream
├── features.py   DINOv2 loader with RAM-aware fallback
├── index.py      FAISS HNSW + sqlite catalog
├── align.py      RANSAC over (t_a, t_b) match clouds
├── pipeline.py   orchestration
├── cli.py        Typer entry points
└── ui.py         Rich tables / JSON output
```

The feature extractor is intentionally a single class so it can be swapped for SSCD,
CLIP, or pHash without touching the rest of the pipeline.

## License

MIT.
