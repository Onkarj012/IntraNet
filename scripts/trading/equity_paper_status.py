#!/usr/bin/env python
"""Equity momentum paper-trading status dashboard.

Reads the live forward ledger and reports:
  - All-time aggregate metrics
  - Rolling 10-rebalance (≈ 3-month) metrics
  - Per-year breakdown
  - Halt condition checks

Usage:
  python scripts/trading/equity_paper_status.py
  python scripts/trading/equity_paper_status.py --write-halt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

from equity import momentum as M

LIVE_LEDGER = _root / "results/equity/live_ledger.csv"
KILL_SWITCH = _root / "results/equity/EQUITY_PAPER_HALTED"
HARD_DD_HALT = -0.30
SOFT_SHARPE  =  0.3
PPY = 252 / M.REBALANCE_DAYS


def _metrics(r: pd.Series) -> dict:
    if r.empty or r.std() == 0:
        return {"n": len(r), "ret": 0.0, "sharpe": 0.0, "win": 0.0, "dd": 0.0}
    sh = r.mean() / r.std() * np.sqrt(PPY)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"n": len(r), "ret": float((1+r).prod()-1),
            "sharpe": sh, "win": float((r > 0).mean()), "dd": dd}


def _fmt(m: dict, label: str) -> str:
    return (f"  {label:<28} n={m['n']:>4}  ret={m['ret']*100:>+7.1f}%  "
            f"Sharpe={m['sharpe']:>+5.2f}  win={m['win']*100:>4.0f}%  DD={m['dd']*100:>6.1f}%")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", default=str(LIVE_LEDGER))
    p.add_argument("--write-halt", action="store_true")
    args = p.parse_args()

    ledger_path = Path(args.ledger)
    box = "=" * 80

    print(f"\n{box}")
    print(f"|  Equity Momentum Paper Trading Status{' '*41}|")
    print(f"{box}")

    if not ledger_path.exists():
        print(f"\n  No live ledger found at {ledger_path}")
        print("  Run equity_paper_trade.py first to initialise.")
        print(f"\n{box}\n")
        return 0

    led = pd.read_csv(ledger_path, parse_dates=["rebalance_date", "exit_date"])
    if led.empty:
        print("\n  Ledger exists but has no rows yet.")
        print(f"\n{box}\n")
        return 0

    print(f"\n  ledger: {ledger_path}")
    print(f"          {len(led)} rebalances  "
          f"{led['rebalance_date'].iloc[0].date()} → {led['exit_date'].iloc[-1].date()}")

    r_all = led["net_ret"]
    eq    = led["equity"]

    # All-time
    print(f"\n  -- ALL-TIME --")
    print(_fmt(_metrics(r_all), "All rebalances"))

    # Rolling 10-rebal
    if len(led) >= 5:
        r10 = r_all.iloc[-10:]
        print(f"\n  -- ROLLING LAST 10 REBALANCES (~3 months) --")
        print(_fmt(_metrics(r10), "Last 10 rebalances"))

    # Per-year
    print(f"\n  -- PER-YEAR --")
    print(f"  {'Year':<6} {'Ret%':>8} {'Bench%':>8} {'Excess%':>9} {'Sharpe':>7} {'DD%':>7} {'Invested%':>10}")
    led["year"] = led["rebalance_date"].dt.year
    for y, sub in led.groupby("year"):
        r = sub["net_ret"]
        sh = r.mean()/r.std()*np.sqrt(PPY) if r.std()>0 else 0
        s = (1+r).prod()-1
        b = sub["bench_equity"].iloc[-1]/sub["bench_equity"].iloc[0]-1
        dd = (sub["equity"]/sub["equity"].cummax()-1).min()
        inv = (sub["state"]=="invested").mean()
        print(f"  {y:<6} {s*100:>8.1f} {b*100:>8.1f} {(s-b)*100:>+9.1f} "
              f"{sh:>7.2f} {dd*100:>7.1f} {inv*100:>9.0f}%")

    # Open position MTM
    last_w_json = led["holdings"].iloc[-1]
    import json
    last_w = json.loads(last_w_json) if last_w_json else {}
    if last_w:
        print(f"\n  -- OPEN POSITION (last rebalance {led['rebalance_date'].iloc[-1].date()}) --")
        for s, wt in sorted(last_w.items(), key=lambda x: -x[1])[:8]:
            print(f"    {s:<14} {wt*100:.1f}%")
        if len(last_w) > 8:
            print(f"    ... +{len(last_w)-8} more")

    # Halt checks
    print(f"\n  -- HALT CHECKS --")
    soft_triggered = False
    rc = 0

    dd_all = float((eq / eq.cummax() - 1).min())
    if dd_all <= HARD_DD_HALT:
        print(f"  *** HARD HALT: cumulative DD {dd_all*100:.1f}% <= {HARD_DD_HALT*100:.0f}% ***")
        if args.write_halt:
            KILL_SWITCH.parent.mkdir(parents=True, exist_ok=True)
            KILL_SWITCH.write_text(f"Hard halt: DD {dd_all*100:.1f}%\n")
            print(f"  Kill-switch written: {KILL_SWITCH}")
        rc = 3
    else:
        print(f"  Cumulative DD: {dd_all*100:.1f}%  (hard halt at {HARD_DD_HALT*100:.0f}%)  OK")

    if len(led) >= 10:
        r10 = r_all.iloc[-10:]
        sh10 = r10.mean()/r10.std()*np.sqrt(PPY) if r10.std()>0 else 0
        if sh10 < SOFT_SHARPE:
            print(f"  WARNING SOFT HALT: rolling 10-rebal Sharpe {sh10:.2f} < {SOFT_SHARPE}")
            soft_triggered = True
        else:
            print(f"  Rolling 10-rebal Sharpe: {sh10:.2f}  OK")

    if rc == 0 and soft_triggered:
        rc = 2

    print(f"\n{box}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
