#!/usr/bin/env python
"""
Backtest the PRODUCTION intraday strategy — Jan→May 2026, ₹1L capital.

Strategy (best config from exit_strategy_sweep.py):
  - Pre-open: score momentum candidates with dual-head ensemble
  - Rank by dir_p (direction head); keep dir_p ≥ 0.44
  - Equal-weight top-12 (max 4 per sector)
  - Buy at open, protective stop at 2.0×ATR, square off at close (intraday)

Reports MEANINGFUL metrics: % profitable, profit factor, Sharpe, max DD, P&L.

Usage:
    python scripts/research/backtest_intraday_jan_may_2026.py
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

K_STOP   = 2.0      # protective stop = 2.0×ATR
DIR_THR  = 0.44     # direction-head threshold
TOP_N    = 12       # equal-weight names per day
MAX_PER_SECTOR = 4
COST_PCT = 0.0018
CAPITAL  = 100_000


def run(start: str, end: str, top_n: int, dir_thr: float, k_stop: float,
        trailing_stop: bool = False, slippage: float = 0.001) -> pd.DataFrame:
    from equity.momentum import load_ohlcv, load_panel, momentum_score, vol63, adv, select
    from equity.v8.signal_models import MetaEnsemble, SignalModel
    from equity.v8.regime_detector import DEFAULT_REGIME_WEIGHTS
    from equity.universe import get_symbol_metadata

    spec = importlib.util.spec_from_file_location(
        "train_intraday", ROOT / "scripts/trading/train_intraday.py")
    _ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(_ti)
    _build = _ti._build_all_features_vectorised

    print(f"\n  PRODUCTION BACKTEST  {start} → {end}  capital=₹{CAPITAL:,}")
    print(f"  Config: rank dir_p≥{dir_thr}, equal-weight top-{top_n}, "
          f"stop {k_stop}×ATR, trailing={trailing_stop}, slippage={slippage:.3%}")

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
    meta = json.loads((model_dir/"train_meta.json").read_text())
    print(f"  Model AUC → barrier={meta.get('barrier_auc')}  dir={meta.get('dir_auc')}")

    print("  Building features...")
    tall = _build(ohlcv, symbols, str(ROOT / "market_data_cache"))
    tall["date"] = pd.to_datetime(tall["date"])

    # ATR panel (T-1)
    c_, h_, lo_ = ohlcv["close"], ohlcv.get("high"), ohlcv.get("low")
    prev = c_.shift(1)
    tr = pd.DataFrame(np.maximum(np.maximum((h_-lo_).values,(h_-prev).abs().values),(lo_-prev).abs().values),
                      index=c_.index, columns=c_.columns)
    atr_panel = (tr.ewm(span=14, min_periods=7).mean()/c_.replace(0,np.nan)).shift(1).fillna(0.02)

    # Load regime history (Phase B)
    regime_hist_path = ROOT / "models/v8_intraday/regime_history.parquet"
    exposure_path    = ROOT / "models/v8_intraday/exposure_by_regime.json"
    regime_hist = pd.read_parquet(regime_hist_path)["regime"] if regime_hist_path.exists() else None
    exposure_map = json.loads(exposure_path.read_text()) if exposure_path.exists() else {}
    CIRCUIT_BREAKER = -0.015   # halt next day if today P&L < -1.5% of capital

    score_df, v63_df, adv_df = momentum_score(close_p), vol63(close_p), adv(close_p, vol_p)
    days = [d for d in close_p.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    o_p, h_p, lo_p, c_p = ohlcv.get("open",c_), ohlcv.get("high",c_), ohlcv.get("low",c_), c_

    sector_cache = {}
    def sector_of(sym):
        if sym not in sector_cache:
            sector_cache[sym] = get_symbol_metadata(sym).get("industry","other") or "other"
        return sector_cache[sym]

    trades = []
    cap = CAPITAL
    eq_by_day = []
    halted_next = False   # circuit breaker flag

    for date in days:
        # ── Circuit breaker: skip day after big loss ──
        if halted_next:
            halted_next = False
            eq_by_day.append(cap)
            continue

        # ── Regime exposure scaling (Phase B) ──
        regime = "strong_trend_up"
        if regime_hist is not None and date in regime_hist.index:
            regime = regime_hist.loc[date]
        exposure = exposure_map.get(regime, 1.0)
        if exposure == 0.0:
            eq_by_day.append(cap); continue   # sit out: high_vol_crisis

        try:
            w, state = select(date, close_p, score_df, v63_df, adv_df, top_n=20)
        except Exception:
            continue
        if state != "invested" or not w:
            continue
        cands = list(w.keys())
        dX = tall[tall["date"]==date].set_index("symbol").reindex(cands).dropna(how="all")
        if dX.empty: continue
        fcols=[c for c in dX.columns if c not in
               {"date","symbol","long_label","dir_label","open_price","long_target","long_stop","atr_pct"}]
        X = dX[fcols].fillna(0).replace([np.inf,-np.inf],0)

        b,_ = barrier_ens.predict(X)
        d,_ = dir_ens.predict(X) if dir_ens else (b,None)

        # rank by dir_p, threshold, sector-diversify to top_n
        ranked = sorted([(sym, float(d[i])) for i,sym in enumerate(X.index) if float(d[i])>dir_thr],
                        key=lambda x:-x[1])
        sel, sec_count = [], {}
        for sym, dp in ranked:
            s = sector_of(sym)
            if sec_count.get(s,0) >= MAX_PER_SECTOR: continue
            sel.append((sym, dp)); sec_count[s]=sec_count.get(s,0)+1
            if len(sel) >= top_n: break
        if not sel: continue

        # Scale deployed capital by regime exposure
        deployed = cap * exposure
        alloc = deployed / len(sel)
        day_pnl = 0.0
        for sym, dp in sel:
            try:
                entry=float(o_p.loc[date,sym]); high=float(h_p.loc[date,sym])
                low=float(lo_p.loc[date,sym]); close=float(c_p.loc[date,sym])
            except Exception: continue
            if entry<=0 or np.isnan(entry): continue
            atr=max(min(float(atr_panel.loc[date,sym]),0.10),0.005)
            effective_entry = entry * (1 + slippage)
            stop_lvl = effective_entry*(1 - k_stop*atr)
            if low <= stop_lvl:
                pnl_pct = -k_stop*atr - slippage - COST_PCT; outcome="STOP"
            elif trailing_stop and high >= effective_entry*(1 + atr):
                # trailing stop triggered: lock in at entry + 0.7×ATR
                exit_price = effective_entry*(1 + 0.7*atr)
                pnl_pct = (exit_price - effective_entry)/effective_entry - COST_PCT
                outcome = "TRAIL_WIN"
            else:
                pnl_pct = (close - effective_entry)/effective_entry - COST_PCT
                outcome = "WIN" if pnl_pct>0 else "LOSS"
            day_pnl += alloc*pnl_pct
            trades.append({"date":date.date(),"symbol":sym,"dir_p":round(dp,3),
                           "regime":regime,"exposure":exposure,
                           "atr_pct":round(atr,4),"entry":round(effective_entry,2),
                           "stop":round(stop_lvl,2),"close":round(close,2),
                           "outcome":outcome,"pnl_pct":round(pnl_pct,5),
                           "pnl_inr":round(alloc*pnl_pct,2),"profitable":pnl_pct>0})
        cap += day_pnl
        eq_by_day.append(cap)

        # Circuit breaker check
        if day_pnl / cap < CIRCUIT_BREAKER:
            halted_next = True

    if not trades:
        print("  No trades."); return pd.DataFrame()

    df = pd.DataFrame(trades)
    t = df.pnl_pct.values
    total_pnl = df.pnl_inr.sum()
    pct_prof = df.profitable.mean()
    pf = df.loc[df.pnl_inr>0,"pnl_inr"].sum()/abs(df.loc[df.pnl_inr<0,"pnl_inr"].sum()) if (df.pnl_inr<0).any() else np.inf
    daily = df.groupby("date")["pnl_inr"].sum()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    eq = pd.Series([CAPITAL]+eq_by_day)
    max_dd = float((eq/eq.cummax()-1).min())
    n_stop = (df.outcome=="STOP").sum()
    n_trail = (df.outcome=="TRAIL_WIN").sum()

    print(f"\n{'='*60}")
    print(f"  PRODUCTION STRATEGY RESULTS  {start} → {end}")
    print(f"{'='*60}")
    print(f"  Total trades        : {len(df)}")
    print(f"  Trading days        : {len(days)}  ({len(df)/len(days):.1f} trades/day)")
    print(f"  % PROFITABLE        : {pct_prof:.1%}   ← real win rate")
    print(f"  Stopped out         : {n_stop} ({n_stop/len(df):.1%})")
    if trailing_stop:
        print(f"  Trailing-stop exits : {n_trail} ({n_trail/len(df):.1%})")
    print(f"  Avg P&L/trade       : ₹{df.pnl_inr.mean():+.0f}")
    print(f"  Profit factor       : {pf:.2f}")
    print(f"  Sharpe (annual)     : {sharpe:.2f}")
    print(f"  Max drawdown        : {max_dd:.1%}")
    print(f"  Total P&L           : ₹{total_pnl:+,.0f}  ({total_pnl/CAPITAL:+.1%})")
    print(f"  Final capital       : ₹{CAPITAL+total_pnl:,.0f}")
    print(f"{'='*60}")

    df["month"] = pd.to_datetime(df.date).dt.to_period("M")
    m = df.groupby("month").agg(trades=("pnl_inr","count"),
                                 prof=("profitable","mean"), pnl=("pnl_inr","sum"))
    print("\n  Monthly:")
    print(f"  {'Month':<10}{'Trades':>8}{'%Prof':>8}{'P&L (₹)':>12}")
    for mo,r in m.iterrows():
        print(f"  {str(mo):<10}{int(r.trades):>8}{r.prof:>8.0%}{r.pnl:>+12,.0f}")
    print(f"\n  Positive months: {(m.pnl>0).sum()}/{len(m)}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-06-01")
    p.add_argument("--end",   default="2026-05-31")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--dir-thr", type=float, default=DIR_THR)
    p.add_argument("--k-stop", type=float, default=K_STOP)
    p.add_argument("--trailing-stop", action="store_true", default=False)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--out", default=str(ROOT/"results/equity/backtest_production_v2.csv"))
    args = p.parse_args()

    # Always run base config (no trailing, 0.1% slippage)
    df_base = run(args.start, args.end, args.top_n, args.dir_thr, args.k_stop,
                  trailing_stop=False, slippage=args.slippage)

    if args.trailing_stop:
        print("\n" + "─"*60)
        df_trail = run(args.start, args.end, args.top_n, args.dir_thr, args.k_stop,
                       trailing_stop=True, slippage=args.slippage)
        # Side-by-side comparison
        print(f"\n{'='*60}")
        print(f"  COMPARISON: Base vs Trailing Stop")
        print(f"{'='*60}")
        def _stats(df):
            if df.empty: return {}
            daily = df.groupby("date")["pnl_inr"].sum()
            sh = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
            return {"pnl": df.pnl_inr.sum(), "win": df.profitable.mean(),
                    "sharpe": sh, "pf": df.loc[df.pnl_inr>0,"pnl_inr"].sum()/abs(df.loc[df.pnl_inr<0,"pnl_inr"].sum()) if (df.pnl_inr<0).any() else np.inf}
        b, t = _stats(df_base), _stats(df_trail)
        print(f"  {'Metric':<22}{'Base':>12}{'Trailing':>12}")
        print(f"  {'Total P&L (₹)':<22}{b['pnl']:>+12,.0f}{t['pnl']:>+12,.0f}")
        print(f"  {'Win rate':<22}{b['win']:>12.1%}{t['win']:>12.1%}")
        print(f"  {'Sharpe':<22}{b['sharpe']:>12.2f}{t['sharpe']:>12.2f}")
        print(f"  {'Profit factor':<22}{b['pf']:>12.2f}{t['pf']:>12.2f}")
        print(f"{'='*60}")
        df_out = df_trail
    else:
        df_out = df_base

    if not df_out.empty:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.out, index=False)
        print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
