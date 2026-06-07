#!/usr/bin/env python
"""
Exit-strategy sweep — find the best real-life intraday exit + sizing config.

Strategy: pre-open recommendation → buy at open → exit at close (intraday square-off)
with a protective stop. The edge is the open→close drift; the question is the
optimal stop width, position sizing, and number of picks.

Collects all candidate trades ONCE (with intraday OHLC + dual-head probs), then
applies each exit/sizing config analytically. Reports meaningful metrics:
  % profitable, profit factor, Sharpe, max DD, total P&L on ₹1L.

Grid:
  stop_mult ∈ {none, 1.0, 1.5, 2.0, 2.5} × ATR
  sizing    ∈ {equal, confidence-weighted}
  top_n     ∈ {3, 5, 8, 12}
  prob_thr  ∈ {0.45, 0.48, 0.50}   (dir-head probability threshold)

Usage:
    python scripts/research/exit_strategy_sweep.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

COST_PCT = 0.0018
CAPITAL  = 100_000


def collect_candidates(start: str, end: str) -> pd.DataFrame:
    """Score every momentum candidate each day; return tall trade-candidate frame."""
    from equity.momentum import load_ohlcv, load_panel, momentum_score, vol63, adv, select
    from equity.v8.signal_models import MetaEnsemble, SignalModel
    from equity.v8.regime_detector import DEFAULT_REGIME_WEIGHTS

    spec = importlib.util.spec_from_file_location(
        "train_intraday", ROOT / "scripts/trading/train_intraday.py")
    _ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(_ti)
    _build = _ti._build_all_features_vectorised

    panel_path = ROOT / "cache/v8/daily_panel_nifty500_adj.parquet"
    ohlcv = load_ohlcv(panel_path)
    close_p, vol_p = load_panel(panel_path)
    symbols = list(close_p.columns)
    model_dir = ROOT / "models/v8_intraday"

    def _head(suffix):
        models = {}
        for n in ["momentum","reversal","breakout","sentiment","macro"]:
            p = model_dir / f"{n}_{suffix}.pkl"
            if p.exists(): models[n] = SignalModel.load(p)
        w = model_dir / f"regime_weights_{suffix}.npy"
        return MetaEnsemble(models=models,
                            regime_weights=np.load(w) if w.exists() else DEFAULT_REGIME_WEIGHTS) if models else None

    barrier_ens, dir_ens = _head("barrier"), _head("dir")

    print("  Building features...")
    tall = _build(ohlcv, symbols, str(ROOT / "market_data_cache"))
    tall["date"] = pd.to_datetime(tall["date"])

    # ATR panel (T-1)
    c_, h_, lo_ = ohlcv["close"], ohlcv.get("high"), ohlcv.get("low")
    prev = c_.shift(1)
    tr = pd.DataFrame(np.maximum(np.maximum((h_-lo_).values,(h_-prev).abs().values),(lo_-prev).abs().values),
                      index=c_.index, columns=c_.columns)
    atr_panel = (tr.ewm(span=14, min_periods=7).mean() / c_.replace(0,np.nan)).shift(1).fillna(0.02)

    score_df, v63_df, adv_df = momentum_score(close_p), vol63(close_p), adv(close_p, vol_p)
    days = [d for d in close_p.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    o_p, h_p, lo_p, c_p = ohlcv.get("open",c_), ohlcv.get("high",c_), ohlcv.get("low",c_), c_
    rows = []
    for date in days:
        try:
            w, state = select(date, close_p, score_df, v63_df, adv_df, top_n=20)
        except Exception:
            continue
        if state != "invested" or not w:
            continue
        cands = list(w.keys())
        dX = tall[tall["date"]==date].set_index("symbol").reindex(cands).dropna(how="all")
        if dX.empty: continue
        fcols = [c for c in dX.columns if c not in
                 {"date","symbol","long_label","dir_label","open_price","long_target","long_stop","atr_pct"}]
        X = dX[fcols].fillna(0).replace([np.inf,-np.inf],0)
        bp,_ = barrier_ens.predict(X)
        dp,_ = dir_ens.predict(X) if dir_ens else (bp, None)
        blend = 0.6*bp + 0.4*dp
        for i, sym in enumerate(X.index):
            try:
                entry=float(o_p.loc[date,sym]); high=float(h_p.loc[date,sym])
                low=float(lo_p.loc[date,sym]); close=float(c_p.loc[date,sym])
            except Exception: continue
            if entry<=0 or np.isnan(entry): continue
            atr=float(atr_panel.loc[date,sym]) if sym in atr_panel.columns else 0.02
            rows.append({"date":date.date(),"symbol":sym,"blend_p":float(blend[i]),
                         "dir_p":float(dp[i]),"barrier_p":float(bp[i]),
                         "atr":max(min(atr,0.10),0.005),
                         "entry":entry,"high":high,"low":low,"close":close,
                         "oc_ret":(close-entry)/entry})
    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} candidate trades over {len(days)} days")
    return df


def apply_strategy(cand: pd.DataFrame, *, stop_mult, sizing, top_n, prob_thr,
                   rank_col="blend_p") -> dict:
    """Apply one exit+sizing config to the candidate set, return metrics."""
    trades = []
    cap = CAPITAL
    eq = [CAPITAL]
    for date, grp in cand.groupby("date"):
        g = grp[grp[rank_col] > prob_thr].sort_values(rank_col, ascending=False).head(top_n)
        if g.empty:
            continue
        # Sizing weights
        if sizing == "conf":
            wts = (g[rank_col] - prob_thr).clip(lower=0.001)
            wts = wts / wts.sum()
        else:
            wts = pd.Series(1.0/len(g), index=g.index)

        day_pnl = 0.0
        for idx, r in g.iterrows():
            alloc = cap * float(wts.loc[idx])
            if stop_mult is None:
                pnl_pct = r.oc_ret - COST_PCT
            else:
                stop_lvl = r.entry * (1 - stop_mult * r.atr)
                if r.low <= stop_lvl:
                    pnl_pct = -stop_mult * r.atr - COST_PCT
                else:
                    pnl_pct = r.oc_ret - COST_PCT
            day_pnl += alloc * pnl_pct
            trades.append(pnl_pct)
        cap += day_pnl
        eq.append(cap)

    if not trades:
        return {}
    t = np.array(trades)
    eq = pd.Series(eq)
    total_pnl = cap - CAPITAL
    pf = t[t>0].sum() / abs(t[t<0].sum()) if (t<0).any() else np.inf
    max_dd = float((eq/eq.cummax()-1).min())
    # Sharpe on per-trade returns annualised (~252*avg_trades/day)
    sharpe = t.mean()/t.std()*np.sqrt(252) if t.std()>0 else 0
    return {
        "stop_mult": stop_mult if stop_mult else "none",
        "sizing": sizing, "top_n": top_n, "prob_thr": prob_thr,
        "n_trades": len(t),
        "pct_profitable": round((t>0).mean(), 4),
        "avg_pnl_pct": round(t.mean(), 5),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 4),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl/CAPITAL, 4),
    }


def main():
    print("\n=== EXIT STRATEGY SWEEP: Jan→May 2026 ===")
    cand = collect_candidates("2026-01-01", "2026-05-31")
    if cand.empty:
        print("No candidates."); return
    cand.to_csv(ROOT / "results/equity/sweep_candidates.csv", index=False)

    grid = []
    for stop_mult in [None, 1.0, 1.5, 2.0, 2.5]:
        for sizing in ["equal", "conf"]:
            for top_n in [3, 5, 8, 12]:
                for prob_thr in [0.42, 0.44, 0.45, 0.46]:
                    m = apply_strategy(cand, stop_mult=stop_mult, sizing=sizing,
                                       top_n=top_n, prob_thr=prob_thr,
                                       rank_col="dir_p")
                    if m: grid.append(m)

    res = pd.DataFrame(grid)
    res.to_csv(ROOT / "results/equity/exit_sweep_results.csv", index=False)

    # Rank by Sharpe then total P&L, require >=30 trades and positive PF
    valid = res[(res.n_trades >= 30) & (res.total_pnl > 0)].copy()
    valid = valid.sort_values(["sharpe","total_pnl"], ascending=False)

    print(f"\n=== TOP 12 CONFIGS (by Sharpe, min 30 trades, positive) ===")
    cols = ["stop_mult","sizing","top_n","prob_thr","n_trades",
            "pct_profitable","profit_factor","sharpe","max_dd","total_pnl","return_pct"]
    print(valid[cols].head(12).to_string(index=False))

    if not valid.empty:
        best = valid.iloc[0]
        print(f"\n=== BEST CONFIG ===")
        print(f"  Stop: {best.stop_mult}×ATR | Sizing: {best.sizing} | "
              f"top_n: {best.top_n} | prob_thr: {best.prob_thr}")
        print(f"  % Profitable: {best.pct_profitable:.1%} | PF: {best.profit_factor} | "
              f"Sharpe: {best.sharpe}")
        print(f"  Max DD: {best.max_dd:.1%} | Total P&L: ₹{best.total_pnl:+,.0f} "
              f"({best.return_pct:+.1%})")

    # Also show the highest-return config regardless of Sharpe
    by_pnl = res[res.n_trades>=30].sort_values("total_pnl", ascending=False)
    print(f"\n=== HIGHEST RETURN (min 30 trades) ===")
    print(by_pnl[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
