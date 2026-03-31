#!/usr/bin/env python3
"""
Threshold Sweep — find the optimal (dir_threshold, min_confidence) for your capital.

Runs the backtest_pro engine across multiple threshold/confidence combinations
in a single run, producing a ranked comparison table.

Usage:
    python scripts/threshold_sweep.py --model runs/intraday/resnls/best_model.pt
    python scripts/threshold_sweep.py --model runs/intraday/resnls/best_model.pt --capital 50000 --zero-cost
    python scripts/threshold_sweep.py --model runs/intraday/resnls/best_model.pt --max-price 500
"""

import argparse
import subprocess
import sys
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# Sweep grid
DEFAULT_THRESHOLDS = [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
DEFAULT_CONFIDENCES = [0.50, 0.55, 0.60, 0.65]


def parse_args():
    parser = argparse.ArgumentParser(description="Threshold Sweep for IntradayNet")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--features-cache", type=str, default="features_cache")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--horizon", type=str, default="H60")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--max-price", type=float, default=0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--zero-cost", action="store_true",
                        help="Skip transaction costs (pure signal analysis)")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--output", type=str, default="threshold_sweep_results.csv")
    parser.add_argument("--thresholds", type=str, default="",
                        help="Comma-separated thresholds (default: 0.52,0.55,0.58,0.60,0.65,0.70)")
    parser.add_argument("--confidences", type=str, default="",
                        help="Comma-separated confidences (default: 0.50,0.55,0.60,0.65)")
    return parser.parse_args()


def run_single_backtest(model_path, config, features_cache, capital, horizon,
                        dir_threshold, min_confidence, max_stocks, max_price,
                        max_positions, zero_cost, long_only, output_dir):
    """Run a single backtest via subprocess and return the summary dict."""
    cmd = [
        sys.executable, "scripts/backtest_pro.py",
        "--model", model_path,
        "--config", config,
        "--features-cache", features_cache,
        "--capital", str(capital),
        "--horizon", horizon,
        "--dir-threshold", str(dir_threshold),
        "--min-confidence", str(min_confidence),
        "--max-positions", str(max_positions),
        "--output-dir", output_dir,
    ]

    if max_stocks > 0:
        cmd.extend(["--max-stocks", str(max_stocks)])
    if max_price > 0:
        cmd.extend(["--max-price", str(max_price)])
    if zero_cost:
        cmd.append("--zero-cost")
    if long_only:
        cmd.append("--long-only")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        summary_path = Path(output_dir) / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")

    return None


def main():
    args = parse_args()

    # Parse custom grids
    thresholds = ([float(x) for x in args.thresholds.split(",")]
                  if args.thresholds else DEFAULT_THRESHOLDS)
    confidences = ([float(x) for x in args.confidences.split(",")]
                   if args.confidences else DEFAULT_CONFIDENCES)

    total_combos = len(thresholds) * len(confidences)

    console.print(Panel.fit(
        f"[bold cyan]IntradayNet — Threshold Sweep[/bold cyan]\n"
        f"[dim]{len(thresholds)} thresholds × {len(confidences)} confidences = "
        f"{total_combos} combinations\n"
        f"Capital: ₹{args.capital:,.0f} | Horizon: {args.horizon} | "
        f"{'ZERO-COST' if args.zero_cost else 'WITH COSTS'} | "
        f"{'LONG-ONLY' if args.long_only else 'LONG+SHORT'}[/dim]",
        border_style="cyan",
    ))

    results = []
    sweep_dir = PROJECT_ROOT / ".tmp" / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Sweeping...", total=total_combos)

        for thresh in thresholds:
            for conf in confidences:
                progress.update(task, description=f"T={thresh:.2f} C={conf:.2f}")

                out_dir = str(sweep_dir / f"t{thresh:.2f}_c{conf:.2f}")
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                summary = run_single_backtest(
                    model_path=args.model,
                    config=args.config,
                    features_cache=args.features_cache,
                    capital=args.capital,
                    horizon=args.horizon,
                    dir_threshold=thresh,
                    min_confidence=conf,
                    max_stocks=args.max_stocks,
                    max_price=args.max_price,
                    max_positions=args.max_positions,
                    zero_cost=args.zero_cost,
                    long_only=args.long_only,
                    output_dir=out_dir,
                )

                if summary:
                    results.append({
                        "threshold": thresh,
                        "confidence": conf,
                        "total_trades": summary.get("total_trades", 0),
                        "win_rate": summary.get("win_rate", 0),
                        "total_pnl": summary.get("total_pnl", 0),
                        "total_return_pct": summary.get("total_return_pct", 0),
                        "profit_factor": summary.get("profit_factor", 0),
                        "sharpe_ratio": summary.get("sharpe_ratio", 0),
                        "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
                    })
                else:
                    results.append({
                        "threshold": thresh,
                        "confidence": conf,
                        "total_trades": 0,
                        "win_rate": 0,
                        "total_pnl": 0,
                        "total_return_pct": 0,
                        "profit_factor": 0,
                        "sharpe_ratio": 0,
                        "max_drawdown_pct": 0,
                    })

                progress.update(task, advance=1)

    # ── Results Table ──
    # Rank by total PnL
    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    table = Table(title="Threshold Sweep Results (ranked by P&L)")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Threshold", style="cyan", justify="right")
    table.add_column("Confidence", style="cyan", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", style="green", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Profit Factor", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", style="red", justify="right")

    for i, r in enumerate(results):
        pnl_style = "green" if r["total_pnl"] > 0 else "red"
        pf_style = "green" if r["profit_factor"] > 1 else "red"

        table.add_row(
            str(i + 1),
            f"{r['threshold']:.2f}",
            f"{r['confidence']:.2f}",
            str(r["total_trades"]),
            f"{r['win_rate']:.1%}" if r["total_trades"] > 0 else "N/A",
            f"[{pnl_style}]₹{r['total_pnl']:,.0f}[/{pnl_style}]",
            f"[{pnl_style}]{r['total_return_pct']:.1f}%[/{pnl_style}]",
            f"[{pf_style}]{r['profit_factor']:.2f}[/{pf_style}]",
            f"{r['sharpe_ratio']:.2f}",
            f"{r['max_drawdown_pct']:.1f}%",
        )

    console.print()
    console.print(table)

    # ── Save CSV ──
    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[bold green]✓ Results saved to {output_path}[/bold green]")

    # ── Best combination recommendation ──
    if results and results[0]["total_trades"] > 0:
        best = results[0]
        console.print(
            f"\n[bold]Recommended:[/bold] threshold=[cyan]{best['threshold']:.2f}[/cyan], "
            f"confidence=[cyan]{best['confidence']:.2f}[/cyan] — "
            f"[green]{best['total_return_pct']:.1f}%[/green] return, "
            f"{best['total_trades']} trades, "
            f"{best['win_rate']:.1%} win rate\n"
        )


if __name__ == "__main__":
    main()
