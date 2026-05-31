#!/usr/bin/env python
"""Equity momentum paper-trade ledger.

Walks the rebalance grid on the adjusted panel and records, for every
rebalance: state, holdings+weights, turnover, gross/net period return, and
compounding equity — using the shared momentum library so it reconciles with
the backtest. Deterministic recompute = idempotent (extends as data updates).
The trailing (open) position is marked-to-market to the latest price.

  python scripts/trading/equity_paper_run.py
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


def run(panel_path, *, start, rebalance_days, top_n, min_adv, use_trend):
    close, vol = M.load_panel(panel_path)
    score, v63, adv_panel = M.momentum_score(close), M.vol63(close), M.adv(close, vol)
    dates = close.index[close.index >= pd.Timestamp(start)]
    rebal = list(dates[::rebalance_days])

    equity, bench = 1.0, 1.0
    prev: dict[str, float] = {}
    rows = []
    for i in range(len(rebal) - 1):
        dt, nxt = rebal[i], rebal[i + 1]
        w, state = M.select(dt, close, score, v63, adv_panel,
                            top_n=top_n, min_adv=min_adv, use_trend=use_trend)
        fwd = close.loc[nxt] / close.loc[dt] - 1.0
        gross = float(sum(wt * float(fwd.get(s, 0.0)) for s, wt in w.items()))
        turnover = 0.5 * sum(abs(w.get(s, 0) - prev.get(s, 0)) for s in set(w) | set(prev))
        cost = turnover * COST
        equity *= (1 + gross - cost)
        bench *= (1 + float(fwd.reindex(close.columns[close.loc[dt].notna() & close.loc[nxt].notna()]).mean()))
        prev = w
        rows.append({
            "rebalance_date": dt.date(), "exit_date": nxt.date(), "state": state,
            "n_holdings": len(w), "turnover": round(turnover, 4),
            "gross_ret": round(gross, 5), "cost": round(cost, 5),
            "net_ret": round(gross - cost, 5), "equity": round(equity, 4),
            "bench_equity": round(bench, 4),
            "holdings": json.dumps({s: round(wt, 4) for s, wt in w.items()}),
        })
    return pd.DataFrame(rows), rebalance_days


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default=str(_root / "cache/v8/daily_panel_nifty500_adj.parquet"))
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--rebalance-days", type=int, default=M.REBALANCE_DAYS)
    p.add_argument("--top-n", type=int, default=M.TOP_N)
    p.add_argument("--min-adv", type=float, default=M.MIN_ADV)
    p.add_argument("--no-trend", action="store_true")
    p.add_argument("--out", default=str(_root / "results/equity/paper_ledger.csv"))
    args = p.parse_args()

    led, k = run(args.panel, start=args.start, rebalance_days=args.rebalance_days,
                 top_n=args.top_n, min_adv=args.min_adv, use_trend=not args.no_trend)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(out, index=False)

    r = led["net_ret"]
    ppy = 252.0 / k
    eq = led["equity"]
    sharpe = r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0
    yrs = (pd.Timestamp(led["exit_date"].iloc[-1]) - pd.Timestamp(led["rebalance_date"].iloc[0])).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    invested = (led["state"] == "invested").mean()
    print(f"\nPaper ledger: {len(led)} rebalances {led['rebalance_date'].iloc[0]} → {led['exit_date'].iloc[-1]}")
    print(f"  CAGR {cagr*100:.1f}%  Sharpe {sharpe:.2f}  maxDD {dd*100:.1f}%  "
          f"final {eq.iloc[-1]:.2f}x (bench {led['bench_equity'].iloc[-1]:.2f}x)")
    print(f"  invested {invested*100:.0f}% of rebalances, avg turnover {led['turnover'].mean()*100:.0f}%")
    print(f"  last holdings ({led['rebalance_date'].iloc[-1]}): "
          f"{list(json.loads(led['holdings'].iloc[-1]).keys())[:8]}...")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
