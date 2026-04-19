#!/usr/bin/env python3
"""
Complete Intraday Movement Prediction System.

Predicts intraday price movement from open:
- Will stock move +X% above open during the day? (LONG signal)
- Will stock move -X% below open during the day? (SHORT signal)

Uses gap as a feature, not the primary target.

Targets:
1. long_viable: Can we make +target% profit from open? (binary)
2. short_viable: Can we make -target% profit from open? (binary)
3. max_up_move: How much does price go up from open? (regression)
4. max_down_move: How much does price go down from open? (regression)

Usage:
    python scripts/train_intraday_model.py --target-pct 0.01
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_absolute_error
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder
from intradaynet.open_safe_daily_features import build_daily_training_frame
from intradaynet.run_logging import command_string, start_run_logging

console = Console()
def train_models(X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series, 
                 y_up_mag: pd.Series, y_down_mag: pd.Series) -> tuple[Dict, Dict, dict]:
    """Train 4 models: long/short binary + magnitude regression."""
    
    unique_dates = pd.Index(sorted(pd.to_datetime(X.index).unique()))
    split_idx = max(1, int(len(unique_dates) * 0.8))
    if split_idx >= len(unique_dates):
        split_idx = len(unique_dates) - 1

    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]
    train_mask = pd.to_datetime(X.index).isin(train_dates)
    val_mask = pd.to_datetime(X.index).isin(val_dates)

    train_X, val_X = X.loc[train_mask], X.loc[val_mask]
    y_long_train, y_long_val = y_long.loc[train_mask], y_long.loc[val_mask]
    y_short_train, y_short_val = y_short.loc[train_mask], y_short.loc[val_mask]
    y_up_train, y_up_val = y_up_mag.loc[train_mask], y_up_mag.loc[val_mask]
    y_down_train, y_down_val = y_down_mag.loc[train_mask], y_down_mag.loc[val_mask]
    
    models = {}
    metrics = {}
    split_info = {
        "train_start": str(pd.Timestamp(train_dates.min()).date()),
        "train_end": str(pd.Timestamp(train_dates.max()).date()),
        "val_start": str(pd.Timestamp(val_dates.min()).date()),
        "val_end": str(pd.Timestamp(val_dates.max()).date()),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "early_stopping_rounds": 50,
    }
    
    # Model 1: LONG viable
    console.print("\n[bold]Training LONG viability model[/bold]")
    m1 = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=1000,
        max_depth=5, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        force_col_wise=True, verbosity=-1,
    )
    m1.fit(
        train_X,
        y_long_train,
        eval_set=[(val_X, y_long_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    pred1 = m1.predict_proba(val_X)[:, 1]
    models["long"] = m1
    metrics["long_auc"] = roc_auc_score(y_long_val, pred1)
    metrics["long_best_iteration"] = int(m1.best_iteration_ or m1.n_estimators)
    console.print(f"  [green]Validation AUC:[/green] {metrics['long_auc']:.4f}")
    
    # Model 2: SHORT viable
    console.print("\n[bold]Training SHORT viability model[/bold]")
    m2 = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=1000,
        max_depth=5, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        force_col_wise=True, verbosity=-1,
    )
    m2.fit(
        train_X,
        y_short_train,
        eval_set=[(val_X, y_short_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    pred2 = m2.predict_proba(val_X)[:, 1]
    models["short"] = m2
    metrics["short_auc"] = roc_auc_score(y_short_val, pred2)
    metrics["short_best_iteration"] = int(m2.best_iteration_ or m2.n_estimators)
    console.print(f"  [green]Validation AUC:[/green] {metrics['short_auc']:.4f}")
    
    # Model 3: Magnitude UP
    console.print("\n[bold]Training UP magnitude model[/bold]")
    m3 = lgb.LGBMRegressor(
        objective="regression", metric="mae", n_estimators=1000,
        max_depth=5, num_leaves=31, learning_rate=0.05,
        force_col_wise=True, verbosity=-1,
    )
    m3.fit(
        train_X,
        y_up_train,
        eval_set=[(val_X, y_up_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    pred3 = m3.predict(val_X)
    models["up_mag"] = m3
    metrics["up_mag_mae"] = mean_absolute_error(y_up_val, pred3)
    metrics["up_mag_best_iteration"] = int(m3.best_iteration_ or m3.n_estimators)
    console.print(f"  [green]Validation MAE:[/green] {metrics['up_mag_mae']:.4f}")
    
    # Model 4: Magnitude DOWN
    console.print("\n[bold]Training DOWN magnitude model[/bold]")
    m4 = lgb.LGBMRegressor(
        objective="regression", metric="mae", n_estimators=1000,
        max_depth=5, num_leaves=31, learning_rate=0.05,
        force_col_wise=True, verbosity=-1,
    )
    m4.fit(
        train_X,
        y_down_train,
        eval_set=[(val_X, y_down_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    pred4 = m4.predict(val_X)
    models["down_mag"] = m4
    metrics["down_mag_mae"] = mean_absolute_error(y_down_val, pred4)
    metrics["down_mag_best_iteration"] = int(m4.best_iteration_ or m4.n_estimators)
    console.print(f"  [green]Validation MAE:[/green] {metrics['down_mag_mae']:.4f}")
    
    return models, metrics, split_info


def summarize_top_features(model, feature_cols: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ranked = sorted(
        zip(feature_cols, importances),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(name, int(score)) for name, score in ranked[:top_n]]


def potential_leakage_features(feature_cols: list[str]) -> list[tuple[str, str]]:
    suspicious_patterns = {
        "close_vs_vwap": "Only safe if shifted to previous session values.",
        "close_vs_day_high": "Only safe if shifted to previous session values.",
        "close_vs_day_low": "Only safe if shifted to previous session values.",
        "volume": "Only safe if using previous-session volume.",
        "vol_momentum": "Only safe if derived from previous-session volume.",
        "vwap": "Only safe if shifted to previous-session VWAP.",
    }
    return [(name, reason) for name, reason in suspicious_patterns.items() if name in feature_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pct", type=float, default=0.01, help="Target movement %% (e.g., 0.01 = 1%%)")
    parser.add_argument("--universe", type=str, default="nifty100")
    parser.add_argument("--output", type=str, default="models/intraday_model.pkl")
    
    args = parser.parse_args()

    run_name = f"train_intraday_model_{args.universe}"
    with start_run_logging(project_root=PROJECT_ROOT, log_group="training", run_name=run_name) as run_logger:
        global console
        console = Console()

        console.print(
            Panel.fit(
                "[bold cyan]Intraday Movement Prediction Model[/bold cyan]\n"
                f"[dim]Target: +/-{args.target_pct*100:.1f}% from open | Universe: {args.universe}[/dim]",
                border_style="cyan",
            )
        )
        console.print(f"[dim]Command:[/dim] {command_string()}")
        console.print(f"[dim]Run log:[/dim] {run_logger.log_path}")
        
        # Load data builders
        console.print("\n[bold]Loading market and sentiment data...[/bold]")
        market_builder = MarketFeatureBuilder()
        market_builder.download(start="2021-01-01", end="2024-12-31")
        
        sentiment_builder = SentimentFeatureBuilder(
            "sentiment/combined_sentiment_2015_2025.csv",
            market_builder=market_builder
        )
        sentiment_builder._load()
        console.print("[green]Data loaded[/green]")
        
        # Load stocks
        symbols = get_universe(args.universe)
        data_dir = Path("nifty500")
        
        console.print(f"\n[bold]Processing {len(symbols)} stocks (2021-2024)...[/bold]")
        
        all_features = []
        all_targets = []
        processed = 0
        skipped = 0
        failed = 0
        per_symbol_rows: list[tuple[str, str, str]] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Scanning symbols", total=len(symbols))
            for symbol in symbols:
                progress.update(task, description=f"Scanning {symbol}")
                csv_path = data_dir / f"{symbol}_minute.csv"
                if not csv_path.exists():
                    skipped += 1
                    per_symbol_rows.append((symbol, "Missing file", "skip"))
                    progress.advance(task)
                    continue
                
                try:
                    minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
                    minute_df.columns = minute_df.columns.str.lower()
                    minute_df = minute_df[(minute_df.index >= "2021-01-01") & (minute_df.index <= "2024-12-31")]
                    
                    if len(minute_df) < 1000:
                        skipped += 1
                        per_symbol_rows.append((symbol, "Too little data", "skip"))
                        progress.advance(task)
                        continue
                    
                    features, targets = build_daily_training_frame(
                        minute_df, symbol, market_builder, sentiment_builder, args.target_pct
                    )
                    
                    if features is not None and len(features) > 100:
                        features["symbol"] = symbol
                        all_features.append(features)
                        all_targets.append(targets)
                        processed += 1
                        per_symbol_rows.append((symbol, f"{len(features)} days", "ok"))
                    else:
                        skipped += 1
                        per_symbol_rows.append((symbol, "Insufficient features", "skip"))
                        
                except Exception as e:
                    failed += 1
                    per_symbol_rows.append((symbol, str(e)[:50], "fail"))

                progress.advance(task)
        
        if not all_features:
            console.print("\n[bold red]No data loaded.[/bold red]")
            console.print(f"[bold yellow]Run log saved to[/bold yellow] {run_logger.log_path}")
            return
        
        # Combine
        X = pd.concat(all_features)
        y = pd.concat(all_targets)

        stock_table = Table(title="Stock Processing Summary")
        stock_table.add_column("Metric", style="cyan")
        stock_table.add_column("Value", justify="right", style="green")
        stock_table.add_row("Processed", str(processed))
        stock_table.add_row("Skipped", str(skipped))
        stock_table.add_row("Failed", str(failed))
        stock_table.add_row("Samples", f"{len(X):,}")
        stock_table.add_row("Symbols", str(X["symbol"].nunique()))
        stock_table.add_row("Date start", str(X.index.min().date()))
        stock_table.add_row("Date end", str(X.index.max().date()))
        stock_table.add_row("Long viable rate", f"{y['long_viable'].mean():.1%}")
        stock_table.add_row("Short viable rate", f"{y['short_viable'].mean():.1%}")
        console.print()
        console.print(stock_table)
        
        # Train models
        feature_cols = [c for c in X.columns if c != "symbol"]
        leakage_flags = potential_leakage_features(feature_cols)
        if leakage_flags:
            leakage_table = Table(title="Potential Leakage Signals")
            leakage_table.add_column("Feature", style="yellow")
            leakage_table.add_column("Constraint", style="red")
            for feature_name, reason in leakage_flags:
                leakage_table.add_row(feature_name, reason)
            console.print()
            console.print(leakage_table)

        models, metrics, split_info = train_models(
            X[feature_cols],
            y["long_viable"],
            y["short_viable"],
            y["max_up"],
            y["max_down"]
        )

        split_table = Table(title="Temporal Split Summary")
        split_table.add_column("Metric", style="cyan")
        split_table.add_column("Value", justify="right", style="green")
        for key, value in split_info.items():
            split_table.add_row(key.replace("_", " ").title(), str(value))
        console.print()
        console.print(split_table)
        
        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump({
                "models": models,
                "features": feature_cols,
                "metrics": metrics,
                "target_pct": args.target_pct,
                "n_samples": len(X),
                "n_symbols": X["symbol"].nunique(),
                "split_info": split_info,
                "log_path": str(run_logger.log_path),
            }, f)

        metrics_table = Table(title="Validation Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")
        for key, value in metrics.items():
            metrics_table.add_row(key, f"{value:.4f}")

        importance_table = Table(title="Top Feature Importances")
        importance_table.add_column("Model", style="cyan")
        importance_table.add_column("Feature", style="bold")
        importance_table.add_column("Importance", justify="right", style="green")
        for model_name, model in models.items():
            for feature_name, importance in summarize_top_features(model, feature_cols, top_n=5):
                importance_table.add_row(model_name, feature_name, str(importance))

        failure_rows = [row for row in per_symbol_rows if row[2] != "ok"][:10]
        if failure_rows:
            issue_table = Table(title="First Non-OK Symbols")
            issue_table.add_column("Symbol", style="cyan")
            issue_table.add_column("Status", style="yellow")
            issue_table.add_column("Details", style="dim")
            for symbol, detail, status in failure_rows:
                issue_table.add_row(symbol, status, detail)
            console.print()
            console.print(issue_table)

        console.print()
        console.print(metrics_table)
        console.print()
        console.print(importance_table)
        console.print()
        console.print(
            Panel.fit(
                f"[bold green]Model saved[/bold green]\n[dim]{output_path}[/dim]\n"
                f"[bold]Run log:[/bold] [dim]{run_logger.log_path}[/dim]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
