#!/usr/bin/env python
"""
Rolling walk-forward validation — quarterly folds, frozen regime map.

For each fold: train on expanding window, test on next quarter.
Regime map (exposure_by_regime.json) is frozen — NOT retuned per fold.

Usage:
    python scripts/research/walk_forward_intraday.py
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

K_STOP = 2.0; DIR_THR = 0.44; TOP_N = 12; MAX_SECTOR = 4; COST = 0.0018; CAP = 100_000

# Quarterly folds: (train_start, train_end, test_start, test_end)
FOLDS = [
    ("2021-01-01", "2023-12-31", "2024-01-01", "2024-03-31"),
    ("2021-01-01", "2024-03-31", "2024-04-01", "2024-06-30"),
    ("2021-01-01", "2024-06-30", "2024-07-01", "2024-09-30"),
    ("2021-01-01", "2024-09-30", "2024-10-01", "2024-12-31"),
    ("2021-01-01", "2024-12-31", "2025-01-01", "2025-03-31"),
    ("2021-01-01", "2025-03-31", "2025-04-01", "2025-06-30"),
    ("2021-01-01", "2025-06-30", "2025-07-01", "2025-09-30"),
    ("2021-01-01", "2025-09-30", "2025-10-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31", "2026-01-01", "2026-05-31"),
]


def load_infra():
    spec = importlib.util.spec_from_file_location("ti", ROOT / "scripts/trading/train_intraday.py")
    ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
    from equity.momentum import load_ohlcv, load_panel, momentum_score, vol63, adv, select
    from equity.universe import get_symbol_metadata
    ohlcv = load_ohlcv(ROOT / "cache/v8/daily_panel_nifty500_adj.parquet")
    close_p, vol_p = load_panel(ROOT / "cache/v8/daily_panel_nifty500_adj.parquet")
    symbols = list(close_p.columns)
    print("  Building full feature matrix (once)...")
    tall = ti._build_all_features_vectorised(ohlcv, symbols, str(ROOT / "market_data_cache"))
    tall["date"] = pd.to_datetime(tall["date"])
    labels = ti.label_barriers(ohlcv)
    labels["date"] = pd.to_datetime(labels["date"])
    # ATR panel
    c_, h_, lo_ = ohlcv["close"], ohlcv.get("high"), ohlcv.get("low")
    tr = pd.DataFrame(np.maximum(np.maximum((h_-lo_).values,(h_-c_.shift(1)).abs().values),
                                 (lo_-c_.shift(1)).abs().values), index=c_.index, columns=c_.columns)
    atr_panel = (tr.ewm(span=14,min_periods=7).mean()/c_.replace(0,np.nan)).shift(1).fillna(0.02)
    # Frozen regime map
    regime_hist = pd.read_parquet(ROOT/"models/v8_intraday/regime_history.parquet")["regime"]
    exposure_map = json.loads((ROOT/"models/v8_intraday/exposure_by_regime.json").read_text())
    score_df, v63_df, adv_df = momentum_score(close_p), vol63(close_p), adv(close_p, vol_p)
    sec_cache = {}
    def sector(s):
        if s not in sec_cache:
            sec_cache[s]=get_symbol_metadata(s).get("industry","other") or "other"
        return sec_cache[s]
    return dict(ti=ti, ohlcv=ohlcv, close_p=close_p, tall=tall, labels=labels,
                atr_panel=atr_panel, regime_hist=regime_hist, exposure_map=exposure_map,
                score_df=score_df, v63_df=v63_df, adv_df=adv_df, symbols=symbols,
                sector=sector, select=select)


def run_fold(inf, fold_idx, tr_start, tr_end, te_start, te_end):
    ti = inf["ti"]
    tall, labels = inf["tall"], inf["labels"]
    atr_panel, regime_hist, exposure_map = inf["atr_panel"], inf["regime_hist"], inf["exposure_map"]
    ohlcv, close_p = inf["ohlcv"], inf["close_p"]
    score_df, v63_df, adv_df, select = inf["score_df"], inf["v63_df"], inf["adv_df"], inf["select"]
    sector = inf["sector"]

    print(f"\n  Fold {fold_idx+1}: train {tr_start}→{tr_end}  test {te_start}→{te_end}")

    # Build train dataset (slice from precomputed tall)
    tr_mask = (tall["date"] >= tr_start) & (tall["date"] <= tr_end)
    te_mask = (tall["date"] >= te_start) & (tall["date"] <= te_end)
    feat_cols = [c for c in tall.columns if c not in
                 {"date","symbol","long_label","dir_label","oc_ret","mag_label",
                  "open_price","long_target","long_stop","atr_pct"}]

    ds = tall[tr_mask].merge(labels[["symbol","date","long_label","dir_label","oc_ret","mag_label",
                                     "open_price","long_target","long_stop","atr_pct"]],
                              on=["symbol","date"], how="inner")
    if len(ds) < 1000:
        print("    skip: insufficient training data"); return None

    X_tr = ds[feat_cols].fillna(0).replace([np.inf,-np.inf],0)
    # Quick single-model training (not full 5-specialist for speed)
    from equity.v8.signal_models import SignalModel
    from equity.v8.config import SignalModelConfig
    cfg = SignalModelConfig(name="dir", lgb_n_estimators=500, lgb_early_stopping_rounds=50,
                            lgb_min_child_samples=30, lgb_num_leaves=31, lgb_metric="auc")
    dir_model = SignalModel(name="dir", config=cfg)
    # Use 90% for train, 10% for val within fold
    n_val = max(int(len(X_tr)*0.1), 100)
    dir_model.fit(X_tr.iloc[:-n_val], ds["dir_label"].values[:-n_val],
                  X_tr.iloc[-n_val:],  ds["dir_label"].values[-n_val:], verbose=False)
    dir_model.calibrate(X_tr.iloc[-n_val:], ds["dir_label"].values[-n_val:])

    # Validate AUC on a held-out slice
    val_probs = dir_model.predict_proba(X_tr.iloc[-n_val:])
    fold_auc = roc_auc_score(ds["dir_label"].values[-n_val:], val_probs)

    # Simulate test period
    days = [d for d in close_p.index if pd.Timestamp(te_start) <= d <= pd.Timestamp(te_end)]
    o_p = ohlcv.get("open",close_p); h_p = ohlcv.get("high",close_p)
    lo_p = ohlcv.get("low",close_p); c_p = close_p

    trades=[]; cap=CAP; halted=False
    for date in days:
        if halted: halted=False; continue
        regime = regime_hist.get(date, "strong_trend_up") if date in regime_hist.index else "strong_trend_up"
        exposure = exposure_map.get(str(regime), 1.0)
        if exposure == 0: continue
        try:
            w,state = select(date, close_p, score_df, v63_df, adv_df, top_n=20)
        except Exception: continue
        if state!="invested" or not w: continue
        cands = list(w.keys())
        dX = tall[tall["date"]==date].set_index("symbol").reindex(cands).dropna(how="all")
        if dX.empty: continue
        X = dX[feat_cols].fillna(0).replace([np.inf,-np.inf],0)
        d_probs = dir_model.predict_proba(X)
        ranked = sorted([(sym,float(d_probs[i])) for i,sym in enumerate(X.index)
                         if float(d_probs[i])>DIR_THR], key=lambda x:-x[1])
        sel,sc=[],{}
        for sym,dp in ranked:
            s=sector(sym)
            if sc.get(s,0)>=MAX_SECTOR: continue
            sel.append((sym,dp)); sc[s]=sc.get(s,0)+1
            if len(sel)>=TOP_N: break
        if not sel: continue
        deployed = cap*exposure; alloc = deployed/len(sel); day_pnl=0
        for sym,dp in sel:
            try:
                en=float(o_p.loc[date,sym]); hi=float(h_p.loc[date,sym])
                lo=float(lo_p.loc[date,sym]); cl=float(c_p.loc[date,sym])
            except Exception: continue
            if en<=0 or np.isnan(en): continue
            atr=max(min(float(atr_panel.loc[date,sym]),0.10),0.005)
            if lo<=en*(1-K_STOP*atr): pnl=-K_STOP*atr-COST; outcome="STOP"
            else: pnl=(cl-en)/en-COST; outcome="WIN" if pnl>0 else "LOSS"
            day_pnl+=alloc*pnl
            trades.append({"date":date.date(),"pnl_pct":pnl,"pnl_inr":alloc*pnl,"profitable":pnl>0})
        cap+=day_pnl
        if day_pnl/cap<-0.015: halted=True

    if not trades: return None
    df=pd.DataFrame(trades)
    total=df.pnl_inr.sum(); pf_val=df.loc[df.pnl_inr>0,"pnl_inr"].sum()/abs(df.loc[df.pnl_inr<0,"pnl_inr"].sum()) if (df.pnl_inr<0).any() else 99
    daily=df.groupby("date")["pnl_inr"].sum()
    sh=daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    eq=pd.Series([CAP]+list(df.groupby("date")["pnl_inr"].sum().cumsum()+CAP))
    dd=float((eq/eq.cummax()-1).min())
    print(f"    AUC={fold_auc:.4f}  trades={len(df)}  %prof={df.profitable.mean():.1%}  "
          f"PF={pf_val:.2f}  Sharpe={sh:.2f}  DD={dd:.1%}  P&L=₹{total:+,.0f}")
    return {"fold":fold_idx+1,"train_end":tr_end,"test":f"{te_start}→{te_end}",
            "auc":round(fold_auc,4),"trades":len(df),"pct_prof":round(df.profitable.mean(),4),
            "pf":round(pf_val,3),"sharpe":round(sh,2),"max_dd":round(dd,4),
            "pnl":round(total,0),"ret":round(total/CAP,4)}


def main():
    print("\n=== ROLLING WALK-FORWARD VALIDATION ===")
    inf = load_infra()
    results = []
    for i,(ts,te,vs,ve) in enumerate(FOLDS):
        r = run_fold(inf, i, ts, te, vs, ve)
        if r: results.append(r)

    if not results: print("No results."); return
    df=pd.DataFrame(results)
    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(df[["fold","test","auc","trades","pct_prof","pf","sharpe","max_dd","pnl"]].to_string(index=False))
    print(f"\nAggregate:")
    print(f"  Mean AUC:       {df.auc.mean():.4f}")
    print(f"  Mean Sharpe:    {df.sharpe.mean():.2f}")
    print(f"  Mean max DD:    {df.max_dd.mean():.1%}")
    print(f"  Total P&L:      ₹{df.pnl.sum():+,.0f}")
    print(f"  Positive folds: {(df.pnl>0).sum()}/{len(df)}")
    (ROOT/"results/equity/walk_forward_results.json").write_text(
        json.dumps({"folds":results,"summary":{
            "mean_auc":round(float(df.auc.mean()),4),
            "mean_sharpe":round(float(df.sharpe.mean()),2),
            "mean_max_dd":round(float(df.max_dd.mean()),4),
            "total_pnl":round(float(df.pnl.sum()),0),
            "positive_folds":int((df.pnl>0).sum()),
            "total_folds":len(df),
        }}, indent=2))
    print(f"\nSaved → results/equity/walk_forward_results.json")


if __name__=="__main__":
    main()
