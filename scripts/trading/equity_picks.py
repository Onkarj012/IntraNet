#!/usr/bin/env python
"""Daily equity momentum picks generator.

Loads the corporate-action-adjusted daily panel, computes the momentum factor
as-of the decision date, applies the liquidity + market-trend gates, and emits
the top-N inverse-vol-weighted holdings. Uses the shared src/equity/momentum.py
so output is identical to the validated backtest.

  python scripts/trading/equity_picks.py                 # latest date
  python scripts/trading/equity_picks.py --as-of 2025-06-30
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

from equity import momentum as M


def generate(panel_path, as_of=None, *, top_n=M.TOP_N, min_adv=M.MIN_ADV,
             min_price=M.MIN_PRICE, weight=M.WEIGHT, use_trend=True):
    close, vol = M.load_panel(panel_path)
    score = M.momentum_score(close)
    v63 = M.vol63(close)
    adv_panel = M.adv(close, vol)
    if as_of is None:
        date = close.index[-1]
    else:
        prior = close.index[close.index <= pd.Timestamp(as_of)]
        if len(prior) == 0:
            raise ValueError(f"as_of {as_of} is before panel start {close.index[0].date()}")
        date = prior[-1]

    weights, state = M.select(date, close, score, v63, adv_panel,
                              top_n=top_n, min_price=min_price, min_adv=min_adv,
                              weight=weight, use_trend=use_trend)
    picks = [
        {"symbol": s, "weight": round(w, 4),
         "score": round(float(score.loc[date, s]), 3),
         "price": round(float(close.loc[date, s]), 2),
         "adv_cr": round(float(adv_panel.loc[date, s]) / 1e7, 2)}
        for s, w in sorted(weights.items(), key=lambda kv: -kv[1])
    ]
    return {
        "as_of": str(date.date()),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": state,
        "config": {"top_n": top_n, "min_adv": min_adv, "weight": weight,
                   "trend_filter": use_trend, "rebalance_days": M.REBALANCE_DAYS},
        "n_picks": len(picks),
        "picks": picks,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default=str(_root / "cache/v8/daily_panel_nifty500_adj.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-n", type=int, default=M.TOP_N)
    p.add_argument("--min-adv", type=float, default=M.MIN_ADV)
    p.add_argument("--no-trend", action="store_true")
    p.add_argument("--out-dir", default=str(_root / "results/equity/picks"))
    args = p.parse_args()

    res = generate(args.panel, args.as_of, top_n=args.top_n, min_adv=args.min_adv,
                   use_trend=not args.no_trend)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"picks_{res['as_of']}.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"\nEquity momentum picks — as of {res['as_of']}  [state: {res['state']}]")
    print(f"config: top{res['config']['top_n']} invvol, trend_filter={res['config']['trend_filter']}, "
          f"min_adv ₹{res['config']['min_adv']/1e7:.0f}cr, rebalance every {res['config']['rebalance_days']}d")
    if not res["picks"]:
        print("  >> RISK-OFF / no eligible universe — hold cash.")
    else:
        print(f"  {'#':>2} {'symbol':<14}{'weight':>8}{'score':>8}{'price':>10}{'ADV(cr)':>9}")
        for i, p_ in enumerate(res["picks"], 1):
            print(f"  {i:>2} {p_['symbol']:<14}{p_['weight']*100:>7.1f}%{p_['score']:>8.2f}"
                  f"{p_['price']:>10.1f}{p_['adv_cr']:>9.1f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
