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
