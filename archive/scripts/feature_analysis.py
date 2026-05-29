#!/usr/bin/env python3
"""
Analyze causal open-safe daily features across a universe and save a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.open_safe_daily_features import (
    build_daily_training_frame,
    classify_feature_family,
)
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder
from intradaynet.run_logging import command_string, start_run_logging
from intradaynet.universe import get_universe

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze open-safe feature quality")
    parser.add_argument("--universe", default="nifty100")
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--target-pct", type=float, default=0.01)
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--market-cache", default="market_data_cache")
    parser.add_argument("--sentiment-csv", default="data/sentiment/combined_sentiment_2015_2025.csv")
    parser.add_argument("--output", default="reports/feature_analysis_open_safe_v2.json")
    return parser.parse_args()


def _safe_corr(series: pd.Series, target: pd.Series) -> float:
    if series.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
        return 0.0
    corr = series.corr(target, method="spearman")
    return 0.0 if pd.isna(corr) else float(corr)


def main():
    args = parse_args()
    run_name = f"feature_analysis_{args.universe}"
    with start_run_logging(project_root=PROJECT_ROOT, log_group="analysis", run_name=run_name) as run_logger:
        global console
        console = Console()

        console.print(
            Panel.fit(
                "[bold cyan]IntradayNet Feature Analysis[/bold cyan]\n"
                f"[dim]Universe: {args.universe} | Max stocks: {args.max_stocks}[/dim]",
                border_style="cyan",
            )
        )
        console.print(f"[dim]Command:[/dim] {command_string()}")
        console.print(f"[dim]Run log:[/dim] {run_logger.log_path}")

        symbols = get_universe(args.universe)
        if args.max_stocks > 0:
            symbols = symbols[: args.max_stocks]

        market_builder = MarketFeatureBuilder(cache_dir=args.market_cache)
        market_builder.download(start="2021-01-01", end="2024-12-31")
        sentiment_builder = SentimentFeatureBuilder(args.sentiment_csv, market_builder=market_builder)
        sentiment_builder._load()

        data_dir = Path(args.data_dir)
        feature_frames: list[pd.DataFrame] = []
        target_frames: list[pd.DataFrame] = []
        processed = 0
        skipped = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing symbols", total=len(symbols))
            for symbol in symbols:
                progress.update(task, description=f"Analyzing {symbol}")
                csv_path = data_dir / f"{symbol}_minute.csv"
                if not csv_path.exists():
                    skipped += 1
                    progress.advance(task)
                    continue

                try:
                    minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
                    minute_df.columns = minute_df.columns.str.lower()
                    minute_df = minute_df[(minute_df.index >= "2021-01-01") & (minute_df.index <= "2024-12-31")]
                    features, targets = build_daily_training_frame(
                        minute_df,
                        symbol,
                        market_builder,
                        sentiment_builder,
                        args.target_pct,
                    )
                    if features is None or targets is None or features.empty:
                        skipped += 1
                    else:
                        feature_frames.append(features.assign(symbol=symbol))
                        target_frames.append(targets.assign(symbol=symbol))
                        processed += 1
                except Exception:
                    skipped += 1

                progress.advance(task)

        if not feature_frames:
            raise SystemExit("No feature frames produced")

        X = pd.concat(feature_frames)
        y = pd.concat(target_frames).reindex(X.index)
        feature_cols = [col for col in X.columns if col != "symbol"]

        rows = []
        for feature in feature_cols:
            series = X[feature]
            coverage = float(series.notna().mean())
            std = float(series.std(ddof=0)) if len(series) else 0.0
            rows.append(
                {
                    "feature": feature,
                    "family": classify_feature_family(feature),
                    "coverage": coverage,
                    "std": std,
                    "corr_long": _safe_corr(series, y["long_viable"]),
                    "corr_short": _safe_corr(series, y["short_viable"]),
                    "corr_up_mag": _safe_corr(series, y["max_up"]),
                    "corr_down_mag": _safe_corr(series, y["max_down"]),
                }
            )

        report_df = pd.DataFrame(rows)
        report_df["score"] = (
            report_df["corr_long"].abs()
            + report_df["corr_short"].abs()
            + report_df["corr_up_mag"].abs()
            + report_df["corr_down_mag"].abs()
        )
        family_summary = (
            report_df.groupby("family")
            .agg(
                feature_count=("feature", "count"),
                avg_score=("score", "mean"),
                avg_coverage=("coverage", "mean"),
                avg_std=("std", "mean"),
            )
            .reset_index()
            .sort_values("avg_score", ascending=False)
        )
        top_features = report_df.sort_values("score", ascending=False).head(20)

        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "open_safe_v2",
            "universe": args.universe,
            "max_stocks": args.max_stocks,
            "processed_symbols": processed,
            "skipped_symbols": skipped,
            "feature_count": len(feature_cols),
            "rows": len(X),
            "families": family_summary.to_dict(orient="records"),
            "top_features": top_features.to_dict(orient="records"),
            "run_log": str(run_logger.log_path),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        summary = Table(title="Feature Analysis Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right", style="green")
        summary.add_row("Processed symbols", str(processed))
        summary.add_row("Skipped symbols", str(skipped))
        summary.add_row("Rows", f"{len(X):,}")
        summary.add_row("Features", str(len(feature_cols)))
        summary.add_row("Output", str(output_path))

        families = Table(title="Top Feature Families")
        families.add_column("Family", style="cyan")
        families.add_column("Count", justify="right", style="green")
        families.add_column("Avg score", justify="right", style="green")
        families.add_column("Coverage", justify="right", style="green")
        for _, row in family_summary.head(6).iterrows():
            families.add_row(
                row["family"],
                str(int(row["feature_count"])),
                f"{row['avg_score']:.4f}",
                f"{row['avg_coverage']:.1%}",
            )

        top = Table(title="Top Features")
        top.add_column("Feature", style="bold")
        top.add_column("Family", style="cyan")
        top.add_column("Score", justify="right", style="green")
        top.add_column("Long", justify="right")
        top.add_column("Short", justify="right")
        for _, row in top_features.head(10).iterrows():
            top.add_row(
                row["feature"],
                row["family"],
                f"{row['score']:.4f}",
                f"{row['corr_long']:.3f}",
                f"{row['corr_short']:.3f}",
            )

        console.print()
        console.print(summary)
        console.print()
        console.print(families)
        console.print()
        console.print(top)
        console.print()
        console.print(
            Panel.fit(
                f"[bold green]Feature report saved[/bold green]\n"
                f"[dim]{output_path}[/dim]\n"
                f"[bold]Run log:[/bold] [dim]{run_logger.log_path}[/dim]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
