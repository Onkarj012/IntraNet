#!/usr/bin/env python
"""V5 v1 thorough multi-day paper-trading test.

Runs the full pipeline (broker → online features → vol kill → gate → caps →
stop-loss → force-close → reconcile) across many trading days and compares
two variants of the EARLIEST_TRADE_TIME setting:

  variant A: 09:30 (no early-trade guard, baseline)
  variant B: 10:30 (recommended fix from paper-trading evidence)

Saves separate ledgers per variant so we can compare apples-to-apples.

Usage:
    python scripts/v5_thorough_test.py
        --start 2024-01-01 --end 2024-12-31
        [--variants 09:30,10:30]
        [--symbol NIFTY]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd

# Setting V5_EARLIEST_TRADE BEFORE importing runtime_config matters.
# We delay the imports until after env-var setup inside `run_variant`.

DATA_ROOT = REPO_ROOT / "data" / "option_data"
LEDGER_DIR = REPO_ROOT / "ledger"
RESULTS_DIR = REPO_ROOT / "results" / "optinet_v5"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_eligible_days(symbol: str, start: date, end: date) -> list[date]:
    """Return all trading days in [start, end] for which:
    - All three CSVs (spot/fut/options) exist
    - dte_to_next_thursday in {3, 6}  (bucket 2 or 3)
    """
    sym_lower = symbol.lower()
    opt_root = DATA_ROOT / f"{sym_lower}_data" / f"{sym_lower}_options"
    out = []
    for f in sorted(opt_root.rglob(f"{sym_lower}_options_*.csv")):
        m = re.match(rf"{sym_lower}_options_(\d\d)_(\d\d)_(\d\d\d\d)\.csv", f.name)
        if not m:
            continue
        d = date(int(m[3]), int(m[2]), int(m[1]))
        if not (start <= d <= end):
            continue
        spot = (DATA_ROOT / f"{sym_lower}_data" / f"{sym_lower}_spot"
                / f"{d.year}" / f"{d.month}"
                / f"{sym_lower}_spot{d.day:02d}_{d.month:02d}_{d.year}.csv")
        fut = (DATA_ROOT / f"{sym_lower}_data" / f"{sym_lower}_fut"
                / f"{d.year}" / f"{d.month}"
                / f"{sym_lower}_fut_{d.day:02d}_{d.month:02d}_{d.year}.csv")
        if not (spot.exists() and fut.exists()):
            continue
        # next Thursday
        delta = (3 - d.weekday()) % 7
        if delta == 0:
            delta = 7
        if delta in (3, 6) or delta in (4, 5, 7):
            # bucket 2 = dte 3-4; bucket 3 = dte 5-7
            out.append(d)
    return out


def run_variant(*, variant_label: str, earliest_trade: str,
                 days: list[date], symbol: str,
                 ledger_path: Path) -> dict:
    """Run the full pipeline across `days` with the given EARLIEST_TRADE_TIME.

    Importantly: env var V5_EARLIEST_TRADE must be set BEFORE the runtime
    modules are imported, because `EARLIEST_TRADE_TIME` is captured at import.
    To ensure each variant uses fresh imports, we use importlib.reload.
    """
    import os, importlib
    os.environ["V5_EARLIEST_TRADE"] = earliest_trade

    # Reload all runtime modules so the new env var is picked up
    for mod_name in [
        "optinet.v5_runtime.runtime_config",
        "optinet.v5_runtime.online_features",
        "optinet.v5_runtime.broker",
        "optinet.v5_runtime.ledger",
    ]:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
    if "v5_minute_decision" in sys.modules:
        importlib.reload(sys.modules["v5_minute_decision"])
    if "v5_force_close" in sys.modules:
        importlib.reload(sys.modules["v5_force_close"])
    if "v5_eod_reconcile" in sys.modules:
        importlib.reload(sys.modules["v5_eod_reconcile"])
    if "v5_e2e_test" in sys.modules:
        importlib.reload(sys.modules["v5_e2e_test"])

    from optinet.v5_runtime import ledger as ld
    from optinet.v5_runtime.runtime_config import (
        EARLIEST_TRADE_TIME, FLAGS_DIR,
    )
    from v5_e2e_test import run_day  # type: ignore

    # Reset state
    if ld.LEDGER_PATH.exists():
        ld.LEDGER_PATH.unlink()
    for f in FLAGS_DIR.glob("*.flag"):
        f.unlink()

    print(f"\n{'='*78}")
    print(f"  Variant {variant_label}: EARLIEST_TRADE_TIME={EARLIEST_TRADE_TIME}")
    print(f"  {len(days)} days, symbol={symbol}")
    print(f"{'='*78}")
    t0 = time.time()
    for i, d in enumerate(days, 1):
        try:
            run_day(sim_date=d)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:3d}/{len(days)}] {d} FAILED: {exc}")
            continue
        if i % 5 == 0 or i == len(days):
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(days)}] {d}  ({elapsed:.0f}s elapsed)")

    # Snapshot the ledger
    df = ld.load_ledger()
    df.to_parquet(ledger_path, index=False)
    print(f"\n  saved → {ledger_path}  ({len(df)} trades)")
    return {"variant": variant_label, "n_trades": len(df),
            "ledger_path": str(ledger_path)}


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0}
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_ts"]).dt.date
    days = df["entry_date"].nunique()
    realized = df["realized_pnl_inr"].astype(float).dropna()
    win = (realized > 0).mean() if len(realized) else float("nan")
    daily = (df.groupby("entry_date")["realized_pnl_inr"]
              .agg(total="sum", n="count").reset_index())
    cum = realized.cumsum()
    dd = (cum - cum.cummax()).min() if len(cum) else float("nan")
    sharpe = (float(daily["total"].mean() / daily["total"].std() * np.sqrt(252))
              if len(daily) > 1 and daily["total"].std() > 0 else float("nan"))
    gross_win = realized[realized > 0].sum()
    gross_loss = abs(realized[realized < 0].sum())
    pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return {
        "n_trades": int(len(df)),
        "n_days": int(days),
        "trades_per_day": round(len(df) / days, 2),
        "win_rate": float(win),
        "stop_rate": float(df["was_stopped"].astype(bool).mean()),
        "mean_pnl_inr": float(realized.mean()),
        "median_pnl_inr": float(realized.median()),
        "total_pnl_inr": float(realized.sum()),
        "best_trade_inr": float(realized.max()),
        "worst_trade_inr": float(realized.min()),
        "sharpe_daily_ann": sharpe,
        "profit_factor": pf,
        "max_drawdown_inr": float(dd),
        "best_day_inr": float(daily["total"].max()),
        "worst_day_inr": float(daily["total"].min()),
    }


def entry_time_distribution(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    return (pd.to_datetime(df["entry_ts"]).dt.strftime("%H:%M")
              .value_counts().sort_index())


def monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["month"] = pd.to_datetime(df["entry_ts"]).dt.to_period("M").astype(str)
    realized = df["realized_pnl_inr"].astype(float)
    g = df.groupby("month").agg(
        n=("trade_id", "count"),
        total_pnl=("realized_pnl_inr", lambda s: float(s.astype(float).sum())),
        win_rate=("realized_pnl_inr", lambda s: float((s.astype(float) > 0).mean())),
        n_stops=("was_stopped", lambda s: int(s.astype(bool).sum())),
    )
    return g.round(2)


def print_compare(a: dict, b: dict, label_a: str, label_b: str) -> None:
    print(f"\n  {'metric':<22} | {label_a:>12} | {label_b:>12} | {'delta':>10}")
    print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    rows = [
        ("n_trades", "{:>12d}", "{:>12d}", "{:>+10d}"),
        ("n_days", "{:>12d}", "{:>12d}", "{:>+10d}"),
        ("trades_per_day", "{:>12.2f}", "{:>12.2f}", "{:>+10.2f}"),
        ("win_rate", "{:>12.1%}", "{:>12.1%}", "{:>+10.1%}"),
        ("stop_rate", "{:>12.1%}", "{:>12.1%}", "{:>+10.1%}"),
        ("mean_pnl_inr", "{:>12.0f}", "{:>12.0f}", "{:>+10.0f}"),
        ("total_pnl_inr", "{:>12.0f}", "{:>12.0f}", "{:>+10.0f}"),
        ("best_trade_inr", "{:>12.0f}", "{:>12.0f}", "{:>+10.0f}"),
        ("worst_trade_inr", "{:>12.0f}", "{:>12.0f}", "{:>+10.0f}"),
        ("sharpe_daily_ann", "{:>12.2f}", "{:>12.2f}", "{:>+10.2f}"),
        ("profit_factor", "{:>12.2f}", "{:>12.2f}", "{:>+10.2f}"),
        ("max_drawdown_inr", "{:>12.0f}", "{:>12.0f}", "{:>+10.0f}"),
        ("best_day_inr", "{:>12.0f}", "{:>12.0f}", "{:>+10.0f}"),
        ("worst_day_inr", "{:>12.0f}", "{:>12.0f}", "{:>+10.0f}"),
    ]
    for key, fa, fb, fd in rows:
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            print(f"  {key:<22} | {'n/a':>12} | {'n/a':>12} | {'n/a':>10}")
            continue
        try:
            delta = vb - va
            print(f"  {key:<22} | {fa.format(va)} | {fb.format(vb)} | {fd.format(delta)}")
        except Exception:
            print(f"  {key:<22} | {va} | {vb} | (delta n/a)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--variants", default="09:30,10:30",
                    help="Comma-sep list of HH:MM EARLIEST_TRADE_TIME values")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit to first N eligible days (0 = no limit)")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"Discovering eligible {args.symbol} days in [{start}, {end}]…")
    days = find_eligible_days(args.symbol, start, end)
    if args.limit > 0:
        days = days[:args.limit]
    print(f"  found {len(days)} eligible days")
    if not days:
        return 1

    variants = [v.strip() for v in args.variants.split(",")]
    summaries = {}

    for v in variants:
        label = f"v{v.replace(':','')}"
        ledger_path = LEDGER_DIR / f"v5_paper_ledger_{label}.parquet"
        run_variant(variant_label=v, earliest_trade=v,
                     days=days, symbol=args.symbol,
                     ledger_path=ledger_path)
        df = pd.read_parquet(ledger_path)
        summaries[v] = {
            "ledger": df,
            "metrics": metrics(df),
            "entry_times": entry_time_distribution(df),
            "monthly": monthly_breakdown(df),
        }

    # Print headline comparison
    print(f"\n{'='*78}")
    print(f"  THOROUGH MULTI-DAY COMPARISON")
    print(f"{'='*78}")
    if len(variants) >= 2:
        a, b = variants[0], variants[1]
        print_compare(summaries[a]["metrics"], summaries[b]["metrics"],
                       f"v{a}", f"v{b}")

    for v in variants:
        s = summaries[v]
        print(f"\n--- variant {v} entry-time distribution ---")
        if not s["entry_times"].empty:
            for k, n in s["entry_times"].items():
                bar = "#" * min(int(n), 40)
                print(f"  {k}  {n:>3d}  {bar}")
        print(f"\n--- variant {v} monthly ---")
        if not s["monthly"].empty:
            print(s["monthly"].to_string())

    # Save consolidated summary
    out = {v: {**s["metrics"]} for v, s in summaries.items()}
    out["meta"] = {"start": str(start), "end": str(end),
                    "symbol": args.symbol, "n_days_tested": len(days)}
    summary_path = RESULTS_DIR / "thorough_test_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
