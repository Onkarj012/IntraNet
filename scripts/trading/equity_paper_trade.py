#!/usr/bin/env python
"""Equity momentum paper-trade forward ledger appender.

Idempotent — safe to run multiple times on the same date.
Logic:
  1. Load the adjusted panel (update it first via equity_eod_panel.py --update).
  2. Determine the last rebalance date from the live ledger (or use --start).
  3. If today is a rebalance day (every REBALANCE_DAYS trading days from start),
     compute new holdings and append a row to the live ledger.
  4. Mark the open (last) position to market using the latest panel close.

Live ledger schema (one row per completed rebalance period):
  rebalance_date, exit_date, state, n_holdings, turnover,
  gross_ret, cost, net_ret, equity, bench_equity, holdings (JSON)

Usage:
  python scripts/trading/equity_paper_trade.py
  python scripts/trading/equity_paper_trade.py --date 2026-05-29
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

from equity import momentum as M
from equity.costs import DEFAULT_COSTS

COST = DEFAULT_COSTS.estimate_round_trip_fraction(1000.0)
LIVE_LEDGER = _root / "results/equity/live_ledger.csv"
PANEL_PATH  = _root / "cache/v8/daily_panel_nifty500_adj.parquet"
KILL_SWITCH = _root / "results/equity/EQUITY_PAPER_HALTED"

# Halt thresholds
HARD_DD_HALT   = -0.30   # -30% equity drawdown → hard halt
SOFT_SHARPE    =  0.3    # rolling 10-rebal Sharpe below this → soft alert


def _load_ledger() -> pd.DataFrame:
    if LIVE_LEDGER.exists():
        return pd.read_csv(LIVE_LEDGER, parse_dates=["rebalance_date", "exit_date"])
    cols = ["rebalance_date", "exit_date", "state", "n_holdings", "turnover",
            "gross_ret", "cost", "net_ret", "equity", "bench_equity", "holdings"]
    return pd.DataFrame(columns=cols)


def _save_ledger(df: pd.DataFrame) -> None:
    LIVE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LIVE_LEDGER, index=False)


def _bench_ret(close: pd.DataFrame, d1, d2) -> float:
    cols = close.columns[close.loc[d1].notna() & close.loc[d2].notna()]
    return float((close.loc[d2, cols] / close.loc[d1, cols] - 1).mean())


def run(as_of: str | None = None) -> int:
    if KILL_SWITCH.exists():
        print(f"  EQUITY HARD HALT active ({KILL_SWITCH}). Remove file to resume.")
        return 3

    close, vol = M.load_panel(PANEL_PATH)
    score  = M.momentum_score(close)
    v63    = M.vol63(close)
    adv_p  = M.adv(close, vol)

    # trading dates available in panel
    tdates = close.index
    today  = tdates[-1] if as_of is None else tdates[tdates <= pd.Timestamp(as_of)][-1]

    led = _load_ledger()

    # Determine rebalance grid anchor and last known state
    anchor = pd.Timestamp("2026-06-02")  # first live rebalance (first Mon after deploy)
    if not led.empty:
        anchor = pd.Timestamp(led["rebalance_date"].iloc[0])

    # Build rebalance grid from anchor up to today
    grid = [d for d in tdates if d >= anchor]
    grid = [grid[i] for i in range(0, len(grid), M.REBALANCE_DAYS)]

    # Find rebalance dates we haven't yet closed (need an exit date in panel)
    done_dates = set(pd.to_datetime(led["rebalance_date"]).dt.normalize()) if not led.empty else set()

    appended = 0
    prev_holdings: dict[str, float] = {}
    if not led.empty:
        prev_holdings = json.loads(led["holdings"].iloc[-1])

    equity      = float(led["equity"].iloc[-1])      if not led.empty else 1.0
    bench_eq    = float(led["bench_equity"].iloc[-1]) if not led.empty else 1.0

    for i, rebal_dt in enumerate(grid[:-1]):
        exit_dt = grid[i + 1]
        if rebal_dt in done_dates:
            continue
        if exit_dt > today:
            break  # exit price not yet available

        w, state = M.select(rebal_dt, close, score, v63, adv_p)
        fwd = close.loc[exit_dt] / close.loc[rebal_dt] - 1.0
        gross = float(sum(wt * float(fwd.get(s, 0.0)) for s, wt in w.items()))
        turnover = 0.5 * sum(abs(w.get(s, 0) - prev_holdings.get(s, 0))
                             for s in set(w) | set(prev_holdings))
        cost = turnover * COST
        equity   *= (1 + gross - cost)
        bench_eq *= (1 + _bench_ret(close, rebal_dt, exit_dt))

        row = {
            "rebalance_date": rebal_dt.date(),
            "exit_date":      exit_dt.date(),
            "state":          state,
            "n_holdings":     len(w),
            "turnover":       round(turnover, 4),
            "gross_ret":      round(gross, 5),
            "cost":           round(cost, 5),
            "net_ret":        round(gross - cost, 5),
            "equity":         round(equity, 4),
            "bench_equity":   round(bench_eq, 4),
            "holdings":       json.dumps({s: round(wt, 4) for s, wt in w.items()}),
        }
        led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
        prev_holdings = w
        appended += 1

    if appended:
        _save_ledger(led)
        print(f"  Appended {appended} rebalance(s) to live ledger → {LIVE_LEDGER}")
    else:
        print(f"  Live ledger up to date (no new completed rebalances as of {today.date()})")

    # Mark-to-market: show open position value
    if not led.empty:
        last_rebal = pd.Timestamp(led["rebalance_date"].iloc[-1])
        last_w = json.loads(led["holdings"].iloc[-1])
        if last_w and last_rebal in close.index and today in close.index:
            mtm = sum(wt * float((close.loc[today, s] / close.loc[last_rebal, s]) - 1)
                      for s, wt in last_w.items() if s in close.columns)
            print(f"  Open position MTM since {last_rebal.date()}: {mtm*100:+.2f}%")

    # Halt checks
    if not led.empty:
        eq = led["equity"]
        dd = float((eq / eq.cummax() - 1).min())
        if dd <= HARD_DD_HALT:
            KILL_SWITCH.parent.mkdir(parents=True, exist_ok=True)
            KILL_SWITCH.write_text(f"Hard halt triggered: equity drawdown {dd*100:.1f}%\n")
            print(f"  *** HARD HALT: drawdown {dd*100:.1f}% <= {HARD_DD_HALT*100:.0f}% — kill-switch written ***")
            return 3

        if len(led) >= 10:
            r = led["net_ret"].iloc[-10:]
            ppy = 252 / M.REBALANCE_DAYS
            sh = r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0
            if sh < SOFT_SHARPE:
                print(f"  WARNING SOFT HALT: rolling 10-rebal Sharpe {sh:.2f} < {SOFT_SHARPE}")
                return 2

    return 0


def main() -> int:
    global PANEL_PATH
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="as-of date YYYY-MM-DD (default: latest in panel)")
    p.add_argument("--panel", default=str(PANEL_PATH))
    args = p.parse_args()
    PANEL_PATH = Path(args.panel)
    rc = run(args.date)
    return rc


if __name__ == "__main__":
    sys.exit(main())
