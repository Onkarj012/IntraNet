from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

install_rich_traceback(show_locals=False)

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    script: str
    summary: str
    example: str
    aliases: tuple[str, ...] = ()


COMMANDS = (
    CommandSpec(
        name="picks",
        script="recommend_intraday.py",
        summary="Generate morning long and short recommendations from the open-safe intraday model.",
        example="uv run intradaynet picks --model models/intraday_model_nifty500.pkl --long-count 3 --short-count 2",
    ),
    CommandSpec(
        name="live-picks",
        script="generate_live_picks.py",
        summary="Create profile-based live recommendations from the bundle workflow.",
        example="uv run intradaynet live-picks --bundle-dir runs/live_backend --output outputs/live_picks.json",
        aliases=("live",),
    ),
    CommandSpec(
        name="health",
        script="daily_health_check.py",
        summary="Run a readiness check over model files, data freshness, and trading-day status.",
        example="uv run intradaynet health --model runs/lgbm_v2 --horizon H60",
        aliases=("doctor",),
    ),
    CommandSpec(
        name="readiness",
        script="readiness_report.py",
        summary="Render the V7 deployability verdict from locked and forward summaries.",
        example="uv run intradaynet readiness --freshness-ok --mode premarket",
        aliases=("ready",),
    ),
    CommandSpec(
        name="sync-data",
        script="sync_data.py",
        summary="Refresh minute-level market data from upstream sources.",
        example="uv run intradaynet sync-data --help",
        aliases=("sync",),
    ),
    CommandSpec(
        name="train-live",
        script="train_live_backend.py",
        summary="Train or rebuild the live backend model bundle.",
        example="uv run intradaynet train-live --help",
        aliases=("train",),
    ),
    CommandSpec(
        name="train-intraday",
        script="train_intraday_model.py",
        summary="Train the open-safe intraday direction and magnitude model.",
        example="uv run intradaynet train-intraday --universe nifty100",
        aliases=("train-model",),
    ),
    CommandSpec(
        name="feature-analysis",
        script="feature_analysis.py",
        summary="Audit and score open-safe features before training.",
        example="uv run intradaynet feature-analysis --universe nifty100 --max-stocks 100",
        aliases=("features", "analyze-features"),
    ),
    CommandSpec(
        name="backtest",
        script="backtest_intraday_2025.py",
        summary="Run the intraday model backtest with hit-rate and P&L metrics.",
        example="uv run intradaynet backtest --risk balanced --capital 100000 --top-k 5",
        aliases=("evaluate",),
    ),
    CommandSpec(
        name="optinet-dataset",
        script="optinet_build_dataset.py",
        summary="Build the OptiNet v1 index-options training dataset.",
        example="uv run intradaynet optinet-dataset --index data/index.csv --options data/options.csv",
    ),
    CommandSpec(
        name="optinet-prepare-indices",
        script="prepare_optinet_indices.py",
        summary="Aggregate minute NIFTY/BANKNIFTY spot and option files into OptiNet EOD inputs.",
        example="uv run intradaynet optinet-prepare-indices --data-root data/indices",
    ),
    CommandSpec(
        name="optinet-train",
        script="train_optinet.py",
        summary="Train the OptiNet v1 LightGBM model stack.",
        example="uv run intradaynet optinet-train --dataset cache/optinet/training_dataset.parquet --profile balanced",
    ),
    CommandSpec(
        name="optinet-picks",
        script="recommend_optinet.py",
        summary="Generate OptiNet index option recommendations.",
        example="uv run intradaynet optinet-picks --model results/models/optinet/optinet_balanced.pkl --index data/index.csv --options data/options.csv",
    ),
    CommandSpec(
        name="optinet-backtest",
        script="backtest_optinet.py",
        summary="Run the OptiNet daily options backtester.",
        example="uv run intradaynet optinet-backtest --model results/models/optinet/optinet_balanced.pkl --index data/index.csv --options data/options.csv",
    ),
    CommandSpec(
        name="optinet-evaluate",
        script="evaluate_optinet.py",
        summary="Run OptiNet train, walk-forward, blind backtest, confidence diagnostics, and readiness gates.",
        example="uv run intradaynet optinet-evaluate --index optinet_data/index/index_spot_daily.csv --options optinet_data/options/options_eod_2021.csv --train-start 2021-01-01 --train-end 2025-12-31 --blind-start 2026-01-01 --blind-end 2026-04-30",
    ),
    CommandSpec(
        name="equity-evaluate",
        script="evaluate_equity.py",
        summary="Wrap an existing IntradayNet equity backtest summary in the shared readiness gate.",
        example="uv run intradaynet equity-evaluate --model models/intraday_model_nifty500.pkl --summary results/backtests/backtest_results/summary.json --start 2026-01-01 --end 2026-03-31",
    ),
    CommandSpec(
        name="paper-ledger",
        script="paper_ledger.py",
        summary="Create a paper-trading ledger from gated model recommendations.",
        example="uv run intradaynet paper-ledger --system optinet --model results/models/optinet/optinet_balanced.pkl --index data/index.csv --options data/options.csv",
    ),
    CommandSpec(
        name="reconcile-paper",
        script="reconcile_paper.py",
        summary="Reconcile paper-trading ledger rows against next-day option-chain outcomes.",
        example="uv run intradaynet reconcile-paper --ledger outputs/paper/optinet_paper_ledger.csv --index data/index.csv --options data/options.csv",
    ),
    CommandSpec(
        name="equity-paper-ledger",
        script="equity_paper_ledger.py",
        summary="Create an IntradayNet equity paper-trading ledger from recommendation JSON.",
        example="uv run intradaynet equity-paper-ledger --recommendations recommendations/morning_picks.json --output outputs/paper/equity_paper_ledger.csv",
    ),
    CommandSpec(
        name="reconcile-equity-paper",
        script="reconcile_equity_paper.py",
        summary="Reconcile equity paper-trading ledger rows against minute bars.",
        example="uv run intradaynet reconcile-equity-paper --ledger outputs/paper/equity_paper_ledger.csv --data-dir nifty500",
    ),
    CommandSpec(
        name="system-status",
        script="trading_system_status.py",
        summary="Create one governed readiness snapshot for equity and options systems.",
        example="uv run intradaynet system-status --output outputs/system/trading_system_status.json",
    ),
)


def _command_index() -> dict[str, CommandSpec]:
    index: dict[str, CommandSpec] = {}
    for command in COMMANDS:
        index[command.name] = command
        for alias in command.aliases:
            index[alias] = command
    return index


def _format_forwarded_args(args: Sequence[str]) -> str:
    return " ".join(args) if args else "[dim]No extra arguments forwarded[/dim]"


def _render_home() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]IntradayNet CLI[/bold cyan]\n"
            "[dim]Rich-powered command center for the project workflows.[/dim]\n"
            "[dim]Run through uv for a consistent environment: [bold]uv run intradaynet <command>[/bold][/dim]",
            border_style="cyan",
        )
    )

    table = Table(title="Available Commands", header_style="bold cyan")
    table.add_column("Command", style="bold")
    table.add_column("Aliases", style="dim")
    table.add_column("What it does")
    table.add_column("Example", style="green")

    for command in COMMANDS:
        table.add_row(
            command.name,
            ", ".join(command.aliases) if command.aliases else "—",
            command.summary,
            command.example,
        )

    console.print(table)
    console.print("[dim]Tip: use `uv run intradaynet help <command>` for a focused command view.[/dim]")


def _render_command_help(command: CommandSpec) -> None:
    aliases = ", ".join(command.aliases) if command.aliases else "none"
    console.print(
        Panel.fit(
            f"[bold]{command.name}[/bold]\n"
            f"{command.summary}\n\n"
            f"[bold]Aliases:[/bold] {aliases}\n"
            f"[bold]Example:[/bold] {command.example}\n"
            f"[bold]Options:[/bold] forwarded directly to [dim]{command.script}[/dim]",
            border_style="green",
        )
    )


def _resolve_script(command: CommandSpec) -> Path:
    return SCRIPTS_DIR / command.script


def _run_script(command: CommandSpec, forwarded_args: Sequence[str]) -> int:
    script_path = _resolve_script(command)
    if not script_path.exists():
        console.print(
            Panel.fit(
                f"[bold red]Missing workflow script[/bold red]\n[dim]{script_path}[/dim]",
                border_style="red",
            )
        )
        return 1

    console.print(
        Panel.fit(
            f"[bold cyan]Running[/bold cyan] {command.name}\n"
            f"[bold]Script:[/bold] [dim]{script_path.name}[/dim]\n"
            f"[bold]Args:[/bold] {_format_forwarded_args(forwarded_args)}",
            border_style="cyan",
        )
    )

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), *forwarded_args],
            cwd=PROJECT_ROOT,
            check=False,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130

    if completed.returncode != 0:
        console.print(
            f"[red]Workflow exited with status {completed.returncode}[/red]"
        )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    command_index = _command_index()

    if not args or args[0] in {"-h", "--help", "list", "commands"}:
        _render_home()
        return 0

    if args[0] in {"help"}:
        if len(args) == 1:
            _render_home()
            return 0
        command = command_index.get(args[1])
        if command is None:
            console.print(f"[red]Unknown command:[/red] {args[1]}")
            _render_home()
            return 1
        _render_command_help(command)
        return 0

    if args[0] in {"-V", "--version", "version"}:
        console.print("intradaynet 0.1.0")
        return 0

    command = command_index.get(args[0])
    if command is None:
        console.print(f"[red]Unknown command:[/red] {args[0]}")
        console.print()
        _render_home()
        return 1

    if any(flag in args[1:] for flag in ("-h", "--help")):
        _render_command_help(command)

    return _run_script(command, args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
