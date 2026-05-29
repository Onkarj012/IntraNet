#!/usr/bin/env python3
"""
Daily Health Check — run before morning_picks.py to verify everything is working.

Checks:
1. Model files exist and are readable
2. Model metrics look reasonable
3. Data files are fresh (not stale)
4. Feature counts match
5. NSE market calendar (is today a trading day?)

Usage:
    python scripts/daily_health_check.py --model runs/lgbm_v2/
    python scripts/daily_health_check.py --model runs/lgbm_v2/ --horizon H60
"""

import argparse
import json
import lightgbm as lgb
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

HORIZONS = ["H15", "H30", "H60"]


def check_model_files(model_dir: Path, horizon: str) -> tuple[bool, str]:
    dir_path = model_dir / f"dir_{horizon}.lgb"
    mag_path = model_dir / f"mag_{horizon}.lgb"

    if not dir_path.exists():
        return False, f"Missing direction model: {dir_path.name}"
    if not mag_path.exists():
        return False, f"Missing magnitude model: {mag_path.name}"

    try:
        dir_model = lgb.Booster(model_file=str(dir_path))
        mag_model = lgb.Booster(model_file=str(mag_path))
        n_trees_dir = dir_model.num_trees()
        n_trees_mag = mag_model.num_trees()
        n_features_dir = dir_model.num_feature()
        n_features_mag = mag_model.num_feature()

        status = (
            f"Direction: {n_trees_dir} trees, {n_features_dir} features | "
            f"Magnitude: {n_trees_mag} trees, {n_features_mag} features"
        )
        return True, status
    except Exception as e:
        return False, f"Error loading models: {e}"


def check_metrics(model_dir: Path, horizon: str) -> tuple[bool, dict]:
    metrics_path = model_dir / f"metrics_{horizon}.json"
    if not metrics_path.exists():
        return False, {"error": "No metrics file found"}

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except Exception as e:
        return False, {"error": f"Failed to parse metrics: {e}"}

    summary = metrics.get("summary", {})
    auc = summary.get("avg_auc", 0)
    prec60 = summary.get("avg_precision_at_60", 0)
    mae = summary.get("avg_mae", 0)

    ok = auc >= 0.50 and prec60 >= 0.45
    return ok, {
        "avg_auc": round(auc, 4),
        "avg_precision_at_60": round(prec60, 4),
        "avg_mae": round(mae, 6),
        "n_folds": metrics.get("n_folds", "?"),
        "n_samples": metrics.get("n_samples", "?"),
    }


def check_data_freshness(data_dir: Path) -> tuple[bool, str]:
    stock_files = list(data_dir.glob("*_minute.csv"))
    if not stock_files:
        return False, f"No stock files in {data_dir}"

    latest = max(f.stat().st_mtime for f in stock_files)
    latest_dt = datetime.fromtimestamp(latest)
    age_hours = (datetime.now() - latest_dt).total_seconds() / 3600

    if age_hours > 48:
        status = f"STALE — {age_hours:.0f} hours old"
        ok = False
    elif age_hours > 24:
        status = f"OLD — {age_hours:.1f} hours old"
        ok = True
    else:
        status = f"FRESH — {age_hours:.1f} hours old"
        ok = True

    return ok, f"{status} | {len(stock_files)} stocks"


def check_feature_consistency(model_dir: Path, data_path: Path, horizon: str) -> tuple[bool, str]:
    if not data_path.exists():
        return True, f"Data file not found (OK if not yet prebatched): {data_path}"

    try:
        data = np.load(data_path, allow_pickle=True)
        n_features = data["X"].shape[1]
        feature_names = data["feature_names"].tolist()

        dir_model = lgb.Booster(model_file=str(model_dir / f"dir_{horizon}.lgb"))
        model_features = dir_model.num_feature()

        if n_features != model_features:
            return False, f"MISMATCH: npz has {n_features} features, model expects {model_features}"
        return True, f"Consistent: {n_features} features"
    except Exception as e:
        return True, f"Could not verify: {e}"


def check_market_calendar() -> tuple[bool, str]:
    today = datetime.now().date()
    weekday = today.weekday()

    if weekday >= 5:
        return False, f"Today is {today.strftime('%A')} — market is closed"
    return True, f"Today is {today.strftime('%A, %Y-%m-%d')} — trading day"


def check_recent_predictions(model_dir: Path, horizon: str) -> tuple[bool, str]:
    recent_dir = model_dir / "recent_predictions"
    if not recent_dir.exists():
        return True, "No recent prediction cache (OK for first run)"

    files = sorted(recent_dir.glob("*.json"))
    if not files:
        return True, "No recent predictions cached"

    latest = files[-1]
    age_hours = (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).total_seconds() / 3600
    if age_hours > 36:
        return False, f"Last predictions were {age_hours:.0f}h ago — stale"
    return True, f"Last predictions {age_hours:.1f}h ago"


def parse_args():
    parser = argparse.ArgumentParser(description="Daily health check for IntradayNet")
    parser.add_argument("--model", default="runs/lgbm_v2",
                        help="Path to model directory")
    parser.add_argument("--data", default="data/nifty500",
                        help="Path to stock data directory")
    parser.add_argument("--prebatched", default="prebatched_v2/lgbm_dataset.npz",
                        help="Path to prebatched data")
    parser.add_argument("--horizon", default="H60", choices=["H15", "H30", "H60"])
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model)
    data_dir = Path(args.data)
    prebatched_path = Path(args.prebatched)

    console.print(Panel.fit(
        "[bold cyan]Daily Health Check[/bold cyan]",
        subtitle=f"Model: {model_dir} | Horizon: {args.horizon} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        border_style="cyan",
    ))
    console.print()

    checks = []

    table = Table(title="Check Results", show_header=True, header_style="bold")
    table.add_column("Check", style="cyan", width=30)
    table.add_column("Status", width=10)
    table.add_column("Details", style="dim")

    check_name = "Model Files"
    ok, detail = check_model_files(model_dir, args.horizon)
    checks.append(ok)
    table.add_row(check_name, "[green]OK[/green]" if ok else "[red]FAIL[/red]", detail)

    check_name = "Model Metrics"
    ok, detail_dict = check_metrics(model_dir, args.horizon)
    checks.append(ok)
    if "error" in detail_dict:
        detail = detail_dict["error"]
    else:
        auc = detail_dict.get("avg_auc", 0)
        prec = detail_dict.get("avg_precision_at_60", 0)
        detail = f"AUC={auc:.4f} | Prec@0.60={prec:.4f}"
        if auc < 0.52:
            detail += " [yellow](below random?)[/yellow]"
    table.add_row(check_name, "[green]OK[/green]" if ok else "[red]WARN[/red]", detail)

    check_name = "Data Freshness"
    ok, detail = check_data_freshness(data_dir)
    checks.append(ok)
    table.add_row(
        check_name,
        "[green]OK[/green]" if ok else "[red]STALE[/red]",
        detail
    )

    check_name = "Feature Consistency"
    ok, detail = check_feature_consistency(model_dir, prebatched_path, args.horizon)
    checks.append(ok)
    table.add_row(
        check_name,
        "[green]OK[/green]" if ok else "[red]MISMATCH[/red]",
        detail
    )

    check_name = "Market Calendar"
    ok, detail = check_market_calendar()
    checks.append(ok)
    table.add_row(
        check_name,
        "[green]OPEN[/green]" if ok else "[yellow]CLOSED[/yellow]",
        detail
    )

    check_name = "Recent Predictions"
    ok, detail = check_recent_predictions(model_dir, args.horizon)
    checks.append(ok)
    table.add_row(
        check_name,
        "[green]OK[/green]" if ok else "[yellow]STALE[/yellow]",
        detail
    )

    console.print(table)
    console.print()

    n_pass = sum(checks)
    n_total = len(checks)
    passed = all(checks)

    if passed:
        console.print("[bold green]ALL CHECKS PASSED — ready for morning picks[/bold green]")
    elif n_pass >= n_total - 1:
        console.print(f"[bold yellow]WARNINGS FOUND ({n_total - n_pass}/{n_total}) — review before trading[/bold yellow]")
    else:
        console.print(f"[bold red]ISSUES FOUND ({n_total - n_pass}/{n_total}) — fix before trading[/bold red]")

    console.print()


if __name__ == "__main__":
    main()
