#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.v7 import default_readiness_paths, evaluate_readiness, load_json_if_exists

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a V7 deployability/readiness report.")
    defaults = default_readiness_paths(PROJECT_ROOT)
    parser.add_argument("--locked-backtest-summary", default=str(defaults["locked_backtest"]))
    parser.add_argument("--forward-summary", default=str(defaults["forward_blind"]))
    parser.add_argument("--mode", choices=("premarket", "post-open"), default="premarket")
    parser.add_argument("--freshness-ok", action="store_true")
    parser.add_argument("--live-symbols", type=int, default=0)
    parser.add_argument("--processed-symbols", type=int, default=0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    locked_summary = load_json_if_exists(Path(args.locked_backtest_summary))
    forward_summary = load_json_if_exists(Path(args.forward_summary))
    readiness = evaluate_readiness(
        locked_backtest_summary=locked_summary,
        forward_summary=forward_summary,
        target_alignment=True,
        mode=args.mode,
        freshness_ok=args.freshness_ok,
        live_symbols=args.live_symbols,
        processed_symbols=args.processed_symbols,
    )

    style = "green" if readiness.status == "READY" else ("yellow" if readiness.status == "PAPER_ONLY" else "red")
    console.print(
        Panel.fit(
            f"[bold {style}]V7 Readiness: {readiness.status}[/bold {style}]",
            border_style=style,
        )
    )

    checks_table = Table(title="Checks")
    checks_table.add_column("Check", style="cyan")
    checks_table.add_column("Pass", justify="right", style="green")
    for key, passed in readiness.checks.items():
        checks_table.add_row(key.replace("_", " "), "yes" if passed else "no")
    console.print(checks_table)

    metrics_table = Table(title="Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right", style="green")
    for key, value in readiness.metrics.items():
        metrics_table.add_row(key.replace("_", " "), "—" if value is None else str(value))
    console.print(metrics_table)

    if readiness.reasons:
        console.print(
            Panel.fit(
                "\n".join(f"- {reason}" for reason in readiness.reasons),
                title="Reasons",
                border_style=style,
            )
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(readiness.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Saved readiness JSON to[/green] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
