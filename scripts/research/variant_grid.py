#!/usr/bin/env python3
"""Variant C grid search — macro regime protection on the 19-month forward walk.

Builds on the Phase-3 forward-walk pipeline.  Tests a small grid of
regime / selectivity filters on top of Variant A's hard filters.  The goal
is NOT to maximize backtest PnL — it's to find a config that:
  (a) keeps trade count high enough to be measurable in live paper data
      (≥ 50 trades / month average → ≥ 30 trades in any 30-day window)
  (b) materially improves Sharpe and drawdown vs Variant A
  (c) is robust across the worst months (Feb 2025, Feb 2026)

Filters in the grid:
  ret_5d_cut    : 5-day NIFTY return floor    (None, -1.5%, -3.0%)
  ret_20d_cut   : 20-day NIFTY return floor   (None, -5%, -10%)
  vix_state     : VIX rising AND > 75th pct   (False, True)
  intraday_halt : daily PnL halt              (-15000, -6000)
  signal_pct    : score percentile            (0.85, 0.90, 0.95)

Output: results/router_v0/variant_c_grid.parquet
"""
from __future__ import annotations

import json
import sys
from datetime import time as dtime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine.features import FUTURES_FEATURES, add_regime

PROXY_FEATURES = PROJECT_ROOT / "cache/router_v0/futures_features_proxy.parquet"
MODEL_PATH     = PROJECT_ROOT / "models/router_v0/futures/final_long.lgb"
NIFTY_DAILY    = PROJECT_ROOT / "data/indices/nifty_daily.csv"
VIX_DAY        = PROJECT_ROOT / "data/nifty_intraday/INDIA VIX_day.csv"
OUT_DIR        = PROJECT_ROOT / "results/router_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Variant A backtest config (do NOT change) ────────────────────────────────
TARGET_PCT  = 0.0040
STOP_PCT    = 0.0030
HORIZON     = 60
LOT         = 50
COSTS_INR   = 105.0
STOP_FLOOR  = -3000.0
MAX_TRADES  = 3
HARD_CUTOFF = dtime(14, 55)
SKIP_START  = dtime(11, 0)
SKIP_END    = dtime(12, 0)
ENTRY_MIN   = 30
HIGH_CONF_PCT = 0.95
SKIP_REGIMES = {"compression"}

# Grid axes
GRID_RET_5D    = [None, -0.015, -0.030]
GRID_RET_20D   = [None, -0.05, -0.10]
GRID_VIX_STATE = [False, True]            # True = skip when VIX rising AND high
GRID_INTRADAY  = [-15000.0, -6000.0]
GRID_SIGNAL    = [0.85, 0.90, 0.95]


# ─────────────────────────────────────────────────────────────────────────────

def load_macro_signals() -> pd.DataFrame:
    """Daily NIFTY return + VIX state, all *prior-day* (no look-ahead)."""
    nd = pd.read_csv(NIFTY_DAILY)
    nd["date"] = pd.to_datetime(nd["date"], utc=True).dt.tz_convert(None).dt.normalize()
    nd = nd.sort_values("date").reset_index(drop=True)
    nd["ret_5d"]  = nd["close"].pct_change(5)
    nd["ret_20d"] = nd["close"].pct_change(20)

    vix = pd.read_csv(VIX_DAY)
    vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
    vix = vix.sort_values("date").reset_index(drop=True)
    vix["vix_5d_ma"] = vix["close"].rolling(5, min_periods=2).mean()
    vix["vix_60d_q75"] = vix["close"].rolling(60, min_periods=20).quantile(0.75)
    vix["vix_rising"]  = vix["close"] > vix["vix_5d_ma"]
    vix["vix_high"]    = vix["close"] > vix["vix_60d_q75"]
    vix["vix_state_block"] = (vix["vix_rising"] & vix["vix_high"]).astype(int)
    vix = vix.rename(columns={"close": "vix_close"})

    df = pd.merge(
        nd[["date", "ret_5d", "ret_20d"]],
        vix[["date", "vix_close", "vix_state_block"]],
        on="date", how="outer",
    ).sort_values("date")
    # Forward-fill weekend gaps before shifting
    df = df.ffill()
    # Shift by 1 day so each row's signals are "as of yesterday's close"
    for c in ["ret_5d", "ret_20d", "vix_close", "vix_state_block"]:
        df[c] = df[c].shift(1)
    df = df.rename(columns={
        "ret_5d": "ret_5d_prev",
        "ret_20d": "ret_20d_prev",
        "vix_close": "vix_prev_close",
        "vix_state_block": "vix_state_block_prev",
    })
    return df


def _simulate_long(entry_idx, day_bars, entry_px, size_mult, costs):
    target_px = entry_px * (1 + TARGET_PCT)
    stop_px   = entry_px * (1 - STOP_PCT)
    j_end = min(len(day_bars), entry_idx + 1 + HORIZON)
    walk = day_bars.iloc[entry_idx + 1:j_end]
    exit_px = None
    exit_reason = None
    for _, r in walk.iterrows():
        c = r["fut_close"]
        if c >= target_px: exit_px, exit_reason = target_px, "TARGET"; break
        if c <= stop_px:   exit_px, exit_reason = stop_px,   "STOP";   break
    if exit_px is None:
        exit_px = float(walk["fut_close"].iloc[-1]) if not walk.empty else entry_px
        exit_reason = "TIME" if not walk.empty else "NO_BARS"
    gross = (exit_px - entry_px) * LOT * size_mult
    net = gross - costs * size_mult
    if net < STOP_FLOOR:
        net = STOP_FLOOR
        exit_reason = "STOP_FLOOR"
    return net, exit_reason


def metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "n_days": 0, "win_rate": 0.0,
                 "total_pnl_inr": 0.0, "sharpe": 0.0,
                 "profit_factor": 0.0, "max_dd_inr": 0.0,
                 "trades_per_30d": 0.0, "neg_months": 0}
    p = trades_df["net_pnl_inr"]
    daily = trades_df.groupby("trade_date")["net_pnl_inr"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
               if len(daily) > 1 and daily.std() > 0 else 0.0)
    cum = p.cumsum()
    gw = p[p > 0].sum()
    gl = abs(p[p < 0].sum())
    pf = float(gw / gl) if gl > 0 else float("inf")
    span_days = (trades_df["trade_date"].max() - trades_df["trade_date"].min()).days
    span_days = max(span_days, 1)
    by_month = trades_df.copy()
    by_month["m"] = pd.to_datetime(by_month["trade_date"]).dt.to_period("M")
    monthly = by_month.groupby("m")["net_pnl_inr"].sum()
    return {
        "n_trades": int(len(trades_df)),
        "n_days": int(trades_df["trade_date"].nunique()),
        "win_rate": float((p > 0).mean()),
        "total_pnl_inr": float(p.sum()),
        "sharpe": float(sharpe),
        "profit_factor": pf,
        "max_dd_inr": float((cum - cum.cummax()).min()),
        "trades_per_30d": float(30.0 * len(trades_df) / span_days),
        "neg_months": int((monthly < 0).sum()),
        "n_months": int(len(monthly)),
    }


def score_proxy_features(features: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
    """Pre-score, pre-filter to eligible minutes, and join macro signals."""
    df = features.copy()
    t = df["datetime"].dt.time
    mod = df["minute_of_day"] - 9 * 60 - 15
    df = df[
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~df["regime"].isin(SKIP_REGIMES)
    ].copy()
    df["long_score"] = model.predict(df[FUTURES_FEATURES])

    macro = load_macro_signals()
    macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
    df["trade_date_norm"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df = df.merge(macro, left_on="trade_date_norm", right_on="date", how="left")
    df = df.sort_values("datetime")
    for c in ["ret_5d_prev", "ret_20d_prev", "vix_prev_close", "vix_state_block_prev"]:
        df[c] = df[c].ffill()

    return df


def run_one_config(scored: pd.DataFrame, full_features: pd.DataFrame,
                     ret_5d_cut, ret_20d_cut, vix_state, intraday_halt, signal_pct):
    """Run backtest with one config; return trades dataframe.

    `scored`        — pre-filtered (eligible-for-entry) rows with scores + macro joined
    `full_features` — FULL feature table; used to build fut_cache for simulation walk
    """
    df = scored.copy()

    if ret_5d_cut is not None:
        df = df[df["ret_5d_prev"] > ret_5d_cut]
    if ret_20d_cut is not None:
        df = df[df["ret_20d_prev"] > ret_20d_cut]
    if vix_state:
        df = df[df["vix_state_block_prev"] != 1]

    if df.empty:
        return pd.DataFrame()

    # Per-day percentile thresholds AFTER macro filtering
    p_sig = df.groupby("trade_date")["long_score"].transform(
        lambda s: s.quantile(signal_pct))
    p_high = df.groupby("trade_date")["long_score"].transform(
        lambda s: s.quantile(HIGH_CONF_PCT))
    df["take_long"] = df["long_score"] >= p_sig
    df["size_mult"] = np.where(df["long_score"] >= p_high, 1.5, 1.0)

    cands = df[df["take_long"]].sort_values(
        ["trade_date", "datetime"]).reset_index(drop=True)

    # Build fut_cache from the FULL features (not the filtered one) so that
    # simulation can walk through minutes we wouldn't enter at (e.g. 11:00-11:59)
    cache = {}
    for d, sub in full_features.groupby("trade_date"):
        bars = sub[["datetime", "f_close"]].rename(
            columns={"f_close": "fut_close"}).sort_values(
            "datetime").reset_index(drop=True)
        cache[d.date()] = bars

    rows = []
    daily_pnl, daily_count = {}, {}
    for _, c in cands.iterrows():
        td = c["trade_date"]
        if daily_count.get(td, 0) >= MAX_TRADES:
            continue
        if daily_pnl.get(td, 0.0) <= intraday_halt:
            continue
        bars = cache.get(td.date())
        if bars is None:                              continue
        idx_arr = bars.index[bars["datetime"] == c["datetime"]]
        if len(idx_arr) == 0:                         continue
        i = int(idx_arr[0])
        entry_px = float(bars["fut_close"].iat[i])
        net, reason = _simulate_long(i, bars, entry_px, float(c["size_mult"]), COSTS_INR)
        rows.append({
            "trade_date": td,
            "datetime": c["datetime"],
            "regime": str(c["regime"]),
            "long_score": float(c["long_score"]),
            "entry_px": entry_px,
            "size_mult": float(c["size_mult"]),
            "net_pnl_inr": net,
            "exit_reason": reason,
        })
        daily_pnl[td]   = daily_pnl.get(td, 0.0) + net
        daily_count[td] = daily_count.get(td, 0) + 1

    return pd.DataFrame(rows)


def main() -> int:
    print("╔" + "═" * 88 + "╗")
    print("║  Variant C grid search (macro regime + selectivity)".ljust(89) + "║")
    print("╚" + "═" * 88 + "╝")

    # Load + score once
    feats = pd.read_parquet(PROXY_FEATURES)
    feats["datetime"]   = pd.to_datetime(feats["datetime"])
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    print(f"  loaded {len(feats):,} feature rows  ({feats['trade_date'].nunique()} days)")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    print("  scoring + joining macro signals …")
    scored = score_proxy_features(feats, model)
    print(f"  scored rows after pre-filter: {len(scored):,}")

    # Variant A baseline (current live config)
    print("\n  Variant A baseline (no regime filter, signal_pct=0.85, halt=-15k):")
    base_trades = run_one_config(scored, feats, None, None, False, -15000.0, 0.85)
    base_m = metrics(base_trades)
    print(f"    n={base_m['n_trades']:>4d}  win={base_m['win_rate']*100:>5.1f}%  "
          f"PnL=₹{base_m['total_pnl_inr']:>+11,.0f}  Sharpe={base_m['sharpe']:>+5.2f}  "
          f"DD=₹{base_m['max_dd_inr']:>+10,.0f}  neg_months={base_m['neg_months']}")

    # Grid
    rows = []
    print(f"\n  running grid: {len(GRID_RET_5D)} × {len(GRID_RET_20D)} × "
          f"{len(GRID_VIX_STATE)} × {len(GRID_INTRADAY)} × {len(GRID_SIGNAL)} = "
          f"{len(GRID_RET_5D)*len(GRID_RET_20D)*len(GRID_VIX_STATE)*len(GRID_INTRADAY)*len(GRID_SIGNAL)} configs")
    n_done = 0
    for r5 in GRID_RET_5D:
        for r20 in GRID_RET_20D:
            for vs in GRID_VIX_STATE:
                for hl in GRID_INTRADAY:
                    for sp in GRID_SIGNAL:
                        trades = run_one_config(scored, feats, r5, r20, vs, hl, sp)
                        m = metrics(trades)
                        rows.append({
                            "ret_5d_cut": r5,
                            "ret_20d_cut": r20,
                            "vix_state_block": vs,
                            "intraday_halt": hl,
                            "signal_pct": sp,
                            **m,
                        })
                        n_done += 1
                        if n_done % 20 == 0:
                            print(f"    {n_done} configs done")

    grid_df = pd.DataFrame(rows)
    grid_df.to_parquet(OUT_DIR / "variant_c_grid.parquet", index=False)
    grid_df.to_csv(OUT_DIR / "variant_c_grid.csv", index=False)

    # Score each config:
    #   - require trades_per_30d >= 30 (measurable in any 30-day window)
    #   - require neg_months <= 6 (no worse than Variant A's 6 negative months)
    #   - rank by Sharpe DESC, drawdown DESC (less negative is better)
    eligible = grid_df[
        (grid_df["trades_per_30d"] >= 30) &
        (grid_df["neg_months"] <= 6) &
        (grid_df["total_pnl_inr"] > 0)
    ].copy()
    eligible = eligible.sort_values(
        ["sharpe", "max_dd_inr"], ascending=[False, False]).reset_index(drop=True)

    print(f"\n  Eligible configs (trades_per_30d≥30, neg_months≤6, PnL>0): "
          f"{len(eligible)} of {len(grid_df)}")

    print(f"\n  Top 10 by Sharpe:")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    cols = ["ret_5d_cut", "ret_20d_cut", "vix_state_block", "intraday_halt",
             "signal_pct", "n_trades", "trades_per_30d", "win_rate",
             "total_pnl_inr", "sharpe", "max_dd_inr", "neg_months"]
    if not eligible.empty:
        print(eligible.head(10)[cols].to_string(index=False,
            formatters={
                "win_rate": lambda x: f"{100*x:.1f}%",
                "trades_per_30d": lambda x: f"{x:.1f}",
                "total_pnl_inr": lambda x: f"₹{x:+,.0f}",
                "sharpe": lambda x: f"{x:+.2f}",
                "max_dd_inr": lambda x: f"₹{x:+,.0f}",
            }))

    # Recommend
    if not eligible.empty:
        best = eligible.iloc[0]
        recommended = {
            "ret_5d_cut": float(best["ret_5d_cut"]) if pd.notna(best["ret_5d_cut"]) else None,
            "ret_20d_cut": float(best["ret_20d_cut"]) if pd.notna(best["ret_20d_cut"]) else None,
            "vix_state_block": bool(best["vix_state_block"]),
            "intraday_halt": float(best["intraday_halt"]),
            "signal_pct": float(best["signal_pct"]),
        }
        print(f"\n  RECOMMENDED Variant C config:")
        for k, v in recommended.items():
            print(f"    {k}: {v}")
        print(f"  Expected (in-sample on the 19m forward-walk):")
        print(f"    n_trades={int(best['n_trades'])}  trades/30d={best['trades_per_30d']:.1f}  "
              f"win={100*best['win_rate']:.1f}%  PnL=₹{best['total_pnl_inr']:+,.0f}")
        print(f"    Sharpe={best['sharpe']:+.2f}  DD=₹{best['max_dd_inr']:+,.0f}  "
              f"neg_months={int(best['neg_months'])}")

        (OUT_DIR / "variant_c_config.json").write_text(json.dumps({
            "recommended": recommended,
            "expected_metrics": {k: float(best[k]) for k in [
                "n_trades", "trades_per_30d", "win_rate", "total_pnl_inr",
                "sharpe", "max_dd_inr"]},
            "expected_neg_months": int(best["neg_months"]),
            "variant_a_baseline": {k: float(base_m[k]) if not isinstance(base_m[k], int) else base_m[k]
                                    for k in base_m if k not in ("n_months",)},
        }, indent=2, default=str))
        print(f"\n  → {OUT_DIR}/variant_c_config.json")

    print(f"  → {OUT_DIR}/variant_c_grid.parquet (full grid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
