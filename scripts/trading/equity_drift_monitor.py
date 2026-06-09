#!/usr/bin/env python
"""Drift monitor — rolling win-rate, Sharpe, and P&L alerts.

Exit codes: 0=ok, 2=soft alert, 3=hard halt active.

Usage:
    python scripts/trading/equity_drift_monitor.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LEDGER = ROOT / "results/equity/intraday_paper_ledger.csv"
HALT   = ROOT / "results/equity/INTRADAY_HALTED"
STARTING_CAP = 100_000


def main() -> int:
    if HALT.exists():
        print(f"  *** HARD HALT ACTIVE: {HALT.read_text()} ***")
        return 3

    if not LEDGER.exists():
        print("  No paper ledger yet. Run equity_intraday_paper.py after first trading day.")
        return 0

    df = pd.read_csv(LEDGER, parse_dates=["date"])
    if df.empty:
        print("  Ledger is empty."); return 0

    n = len(df)
    window_trade = 20   # rolling over last 20 trades
    window_day   = 10   # rolling over last 10 trading days

    # Overall stats
    total_pnl = df["pnl_inr"].sum()
    cap = float(df["capital_after"].iloc[-1]) if "capital_after" in df.columns else STARTING_CAP + total_pnl
    eq = pd.Series([STARTING_CAP] + list(df.groupby("date")["pnl_inr"].sum().cumsum() + STARTING_CAP))
    max_dd = float((eq / eq.cummax() - 1).min())

    # Rolling metrics
    last_n = df.tail(window_trade)
    roll_win = (last_n["pnl_pct"] > 0).mean()
    roll_sharpe = last_n["pnl_pct"].mean() / last_n["pnl_pct"].std() * np.sqrt(252) \
        if last_n["pnl_pct"].std() > 0 else 0

    last_days = df[df["date"] >= df["date"].max() - pd.Timedelta(days=window_day*2)]
    roll_10d_pnl = last_days.groupby("date")["pnl_inr"].sum().sum()
    roll_10d_pct = roll_10d_pnl / cap

    # Thresholds
    alerts = []
    if roll_win < 0.40:     alerts.append(f"rolling {window_trade}-trade win rate {roll_win:.0%} < 40%")
    if roll_sharpe < 0.50:  alerts.append(f"rolling {window_trade}-trade Sharpe {roll_sharpe:.2f} < 0.5")
    if roll_10d_pct < -0.03: alerts.append(f"rolling 10-day P&L {roll_10d_pct:.1%} < -3%")

    print(f"\n  ══ DRIFT MONITOR ══════════════════════════════════")
    print(f"  Trades: {n}  |  Capital: ₹{cap:,.0f}  |  Total P&L: ₹{total_pnl:+,.0f}")
    print(f"  Max drawdown: {max_dd:.1%}")
    print(f"  Rolling {window_trade}-trade win rate : {roll_win:.1%}  "
          f"{'⚠ ALERT' if roll_win<0.40 else '✓'}")
    print(f"  Rolling {window_trade}-trade Sharpe   : {roll_sharpe:.2f}  "
          f"{'⚠ ALERT' if roll_sharpe<0.50 else '✓'}")
    print(f"  Rolling 10-day P&L        : ₹{roll_10d_pnl:+,.0f} ({roll_10d_pct:+.1%})  "
          f"{'⚠ ALERT' if roll_10d_pct<-0.03 else '✓'}")

    # Per-regime breakdown
    if "regime" in df.columns:
        by_regime = df.groupby("regime").agg(
            n=("pnl_inr","count"), win=("pnl_pct",lambda x:(x>0).mean()),
            pnl=("pnl_inr","sum"))
        print(f"\n  By regime:")
        for r, row in by_regime.iterrows():
            print(f"    {r:<25} n={int(row.n):>4}  win={row.win:.0%}  P&L=₹{row.pnl:+,.0f}")

    if alerts:
        print(f"\n  ⚠ SOFT ALERTS:")
        for a in alerts: print(f"    • {a}")
        print(f"  ══════════════════════════════════════════════════")
        return 2

    print(f"  Status: ✓ All monitors green")
    print(f"  ══════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    sys.exit(main())
