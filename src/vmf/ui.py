from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from vmf.pipeline import PairResult

console = Console()


def fmt_time(s: float) -> str:
    s = max(0.0, float(s))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def render_pairs(results: list[PairResult]) -> None:
    if not results:
        console.print("[green]No overlapping fragments found.[/green]")
        return

    has_pvalue = any(s.pvalue is not None for r in results for s in r.segments)
    has_z = any(s.z_score is not None for r in results for s in r.segments)
    for r in results:
        a_name = Path(r.a_path).name
        b_name = Path(r.b_path).name
        title = f"[bold]{a_name}[/bold]  ↔  [bold]{b_name}[/bold]"
        table = Table(title=title, title_justify="left", show_lines=False)
        table.add_column("A range", style="cyan")
        table.add_column("B range", style="cyan")
        table.add_column("Speed", justify="right")
        table.add_column("Mirror", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Inliers", justify="right")
        if has_z:
            table.add_column("z", justify="right")
        if has_pvalue:
            table.add_column("p", justify="right")
        for s in r.segments:
            row = [
                f"{fmt_time(s.a_start)}–{fmt_time(s.a_end)}",
                f"{fmt_time(s.b_start)}–{fmt_time(s.b_end)}",
                f"×{s.speed_ratio:.2f}",
                "yes" if s.mirrored else "—",
                f"{s.score:.3f}",
                str(s.inliers),
            ]
            if has_z:
                row.append("—" if s.z_score is None else f"{s.z_score:.1f}")
            if has_pvalue:
                row.append("—" if s.pvalue is None else f"{s.pvalue:.3f}")
            table.add_row(*row)
        console.print(table)
        console.print(f"  [dim]A:[/dim] {r.a_path}")
        console.print(f"  [dim]B:[/dim] {r.b_path}\n")


def to_json(results: list[PairResult]) -> str:
    payload = []
    for r in results:
        payload.append({
            "a": {"id": r.a_id, "path": r.a_path},
            "b": {"id": r.b_id, "path": r.b_path},
            "segments": [
                {
                    "a_range": [s.a_start, s.a_end],
                    "b_range": [s.b_start, s.b_end],
                    "speed_ratio": s.speed_ratio,
                    "mirrored": s.mirrored,
                    "score": s.score,
                    "inliers": s.inliers,
                    "pvalue": s.pvalue,
                    "z_score": s.z_score,
                    "weighted_support": s.weighted_support,
                }
                for s in r.segments
            ],
        })
    return json.dumps(payload, indent=2, ensure_ascii=False)
