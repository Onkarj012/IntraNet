#!/usr/bin/env python
"""Durable long-only momentum portfolio backtest.

Pipeline (all point-in-time):
  factors  : multi-horizon momentum, optionally risk-adjusted (mom / vol)
  universe : price + liquidity (ADV) filter so picks are live-tradeable
  weights  : equal or inverse-volatility
  overlay  : market trend gate (cash < 200d MA) + volatility target (de-risk only)
  costs    : turnover * round-trip cost each rebalance

Reports CAGR / Sharpe / maxDD vs an equal-weight benchmark, with an
out-of-sample train/test split to guard against parameter overfitting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

from equity.universe import get_universe
from equity.v8.data_pipeline import load_minute_data
from equity.costs import DEFAULT_COSTS
from equity import momentum as M

COST = DEFAULT_COSTS.estimate_round_trip_fraction(1000.0)  # ~0.18% round trip per fully-rotated name


def build_daily_panel(universe, data_dir, cache):
    if cache.exists():
        store = pd.read_parquet(cache)
        return store.xs("close", axis=1, level=0), store.xs("volume", axis=1, level=0)
    closes, vols = {}, {}
    syms = get_universe(universe)
    for i, s in enumerate(syms):
        df = load_minute_data(s, data_dir, min_bars=1, max_gap_pct=1.0)
        if df.empty:
            continue
        d = df.resample("D").agg(close=("close", "last"), volume=("volume", "sum")).dropna()
        if len(d) < 260:
            continue
        closes[s], vols[s] = d["close"], d["volume"]
        if (i + 1) % 50 == 0:
            print(f"  panel: {i + 1}/{len(syms)} ({len(closes)} kept)")
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).reindex_like(close)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.concat({"close": close, "volume": vol}, axis=1).to_parquet(cache)
    return close, vol


def _z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def score_panel(close, signal):
    if signal == "riskadj":
        v = close.pct_change().rolling(63, min_periods=21).std()
        rets = {h: (close.shift(s) / close.shift(h) - 1) / v for h, s in [(21, 0), (63, 0), (126, 0), (252, 21)]}
        return sum(_z(r) for r in rets.values()) / len(rets)
    return M.momentum_score(close)


def backtest(close, vol, score, *, top_n, k, min_price, min_adv, weight,
             trend_filter, vol_target, start):
    dates = close.index[close.index >= pd.Timestamp(start)]
    rebal = list(dates[::k])
    vol63 = close.pct_change().rolling(63, min_periods=21).std()
    adv = (close * vol).rolling(21, min_periods=10).mean()
    ew_ret = close.pct_change().clip(-0.5, 0.5).mean(axis=1).fillna(0.0)
    ew_idx = (1 + ew_ret).cumprod()
    ma200 = ew_idx.rolling(200, min_periods=100).mean()
    mkt_vol = ew_ret.rolling(21, min_periods=10).std() * np.sqrt(252)

    equity, bench = 1.0, 1.0
    prev_eff: dict[str, float] = {}
    rows = []
    for i in range(len(rebal) - 1):
        dt, nxt = rebal[i], rebal[i + 1]
        fwd = close.loc[nxt] / close.loc[dt] - 1.0
        exposure = 1.0
        if trend_filter and not (ew_idx.loc[dt] >= ma200.loc[dt]):
            exposure = 0.0
        if vol_target > 0 and exposure > 0 and mkt_vol.loc[dt] > 0:
            exposure = min(1.0, vol_target / float(mkt_vol.loc[dt]))

        eff: dict[str, float] = {}
        port_ret = 0.0
        if exposure > 0:
            s = score.loc[dt].dropna()
            elig = s[(close.loc[dt].reindex(s.index) >= min_price) &
                     (adv.loc[dt].reindex(s.index) >= min_adv)].dropna()
            elig = elig[fwd.reindex(elig.index).notna()]
            if len(elig) >= top_n:
                picks = list(elig.nlargest(top_n).index)
                if weight == "invvol":
                    iv = (1.0 / vol63.loc[dt, picks].replace(0, np.nan)).fillna(0.0)
                    w = (iv / iv.sum()) if iv.sum() > 0 else pd.Series(1.0 / top_n, index=picks)
                else:
                    w = pd.Series(1.0 / top_n, index=picks)
                port_ret = float((w * fwd.reindex(picks)).sum())
                eff = {sym: exposure * float(w[sym]) for sym in picks}
        turnover = 0.5 * sum(abs(eff.get(s, 0) - prev_eff.get(s, 0))
                             for s in set(eff) | set(prev_eff))
        equity *= (1 + exposure * port_ret - turnover * COST)
        prev_eff = eff
        bench *= (1 + float(fwd.reindex(score.loc[dt].dropna().index).mean()))
        rows.append({"date": nxt, "equity": equity, "bench": bench,
                     "exposure": exposure, "turnover": turnover})
    return pd.DataFrame(rows).set_index("date")


def stats(curve, col, k):
    eq = curve[col]
    rets = eq.pct_change().dropna()
    ppy = 252.0 / k
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    return {
        "CAGR": eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else float("nan"),
        "Sharpe": rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else 0.0,
        "maxDD": float((eq / eq.cummax() - 1).min()),
    }


def _report(curve, k, label, train_end):
    full, b = stats(curve, "equity", k), stats(curve, "bench", k)
    print(f"\n{label}")
    print(f"  {'':10}{'CAGR':>9}{'Sharpe':>9}{'maxDD':>9}")
    print(f"  {'strategy':10}{full['CAGR']*100:>8.1f}%{full['Sharpe']:>9.2f}{full['maxDD']*100:>8.1f}%")
    print(f"  {'benchmark':10}{b['CAGR']*100:>8.1f}%{b['Sharpe']:>9.2f}{b['maxDD']*100:>8.1f}%")
    print(f"  avg exposure {curve['exposure'].mean()*100:.0f}%  avg turnover {curve['turnover'].mean()*100:.0f}%")
    if train_end:
        te = pd.Timestamp(train_end)
        tr, ts = curve[curve.index <= te], curve[curve.index > te]
        for nm, c in [("TRAIN " + str(te.date()), tr), ("TEST (OOS)", ts)]:
            if len(c) > 3:
                c = c.assign(equity=c["equity"] / c["equity"].iloc[0], bench=c["bench"] / c["bench"].iloc[0])
                s, bb = stats(c, "equity", k), stats(c, "bench", k)
                print(f"  {nm:18} strat CAGR {s['CAGR']*100:6.1f}% Sharpe {s['Sharpe']:.2f} maxDD {s['maxDD']*100:6.1f}%"
                      f"  | bench CAGR {bb['CAGR']*100:6.1f}% Sharpe {bb['Sharpe']:.2f}")
    print("  per-year (strat vs bench):", end=" ")
    for y in sorted(curve.index.year.unique()):
        sub = curve[curve.index.year == y]
        base = curve[curve.index.year < y]
        e0 = base["equity"].iloc[-1] if not base.empty else 1.0
        b0 = base["bench"].iloc[-1] if not base.empty else 1.0
        print(f"{y}:{(sub['equity'].iloc[-1]/e0-1)*100:+.0f}/{(sub['bench'].iloc[-1]/b0-1)*100:+.0f}", end="  ")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="nifty500")
    p.add_argument("--data-dir", default="data/nifty500")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--rebalance-days", type=int, default=10)
    p.add_argument("--signal", choices=["mom", "riskadj"], default="mom")
    p.add_argument("--weight", choices=["equal", "invvol"], default="equal")
    p.add_argument("--min-price", type=float, default=20.0)
    p.add_argument("--min-adv", type=float, default=5e7, help="min 21d avg daily traded value (₹)")
    p.add_argument("--trend-filter", action="store_true")
    p.add_argument("--vol-target", type=float, default=0.0, help="annualized vol target (0=off, de-risk only)")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--cache", default=None)
    args = p.parse_args()

    cache = Path(args.cache) if args.cache else _root / "cache/v8" / f"daily_panel_{args.universe}.parquet"
    print(f"Loading daily panel ({args.universe})...")
    close, vol = build_daily_panel(args.universe, args.data_dir, cache)
    print(f"Panel: {close.shape[1]} symbols, {close.index.min().date()} → {close.index.max().date()}")

    score = score_panel(close, args.signal)
    curve = backtest(close, vol, score, top_n=args.top_n, k=args.rebalance_days,
                     min_price=args.min_price, min_adv=args.min_adv, weight=args.weight,
                     trend_filter=args.trend_filter, vol_target=args.vol_target, start=args.start)
    label = (f"signal={args.signal} weight={args.weight} top{args.top_n}/{args.rebalance_days}d "
             f"adv>=₹{args.min_adv/1e7:.0f}cr trend={args.trend_filter} voltgt={args.vol_target}")
    _report(curve, args.rebalance_days, label, args.train_end)
    print("NOTE: current-constituent universe → survivorship inflates absolute levels; "
          "the durable read is strategy-vs-benchmark and OOS stability.")


if __name__ == "__main__":
    main()
