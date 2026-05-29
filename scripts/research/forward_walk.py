#!/usr/bin/env python3
"""End-to-end Phases 0-3 driver:

  Phase 0: Build proxy feature table from NIFTY index minute, Nov 2024 - May 2026.
  Phase 1: Q4-2024 regime hypothesis test on existing trade ledger.
  Phase 2: Add regime guards to 2024 backtest, grid-search VIX + 5d-return cutoffs.
  Phase 3: 19-month forward-walk Nov 2024 - May 2026 with chosen guards.

The NIFTY-future minute vendor data ends Sep/Oct 2024.  For the forward window
we use NIFTY-INDEX minute as the price proxy (basis < 0.3 %, intraday returns
are nearly identical).  Top-of-model features (realized_vol_30m, minute_of_day,
or_dist_high, atr_15m, day_of_week) are price/time only, so the OI-zeroing has
negligible impact (OI features carry 0.7 % of model gain).
"""
from __future__ import annotations

import json
import sys
from datetime import time as dtime, date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine.features import (
    FUTURES_FEATURES, add_regime, compute_features,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
NIFTY_MIN_PATH   = PROJECT_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"
VIX_DAY_PATH     = PROJECT_ROOT / "data/nifty_intraday/INDIA VIX_day.csv"
NIFTY_DAILY_PATH = PROJECT_ROOT / "data/indices/nifty_daily.csv"
EXISTING_TRADES  = PROJECT_ROOT / "results/router_v0/futures_long_2024_trades.parquet"
EXISTING_FEATURES = PROJECT_ROOT / "cache/router_v0/futures_features.parquet"
MODEL_PATH       = PROJECT_ROOT / "models/router_v0/futures/final_long.lgb"
OUT_DIR          = PROJECT_ROOT / "results/router_v0"
PROXY_FEAT_PATH  = PROJECT_ROOT / "cache/router_v0/futures_features_proxy.parquet"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROXY_FEAT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Backtest config (matches existing backtest_futures_long.py) ──────────────
TARGET_PCT  = 0.0040
STOP_PCT    = 0.0030
HORIZON     = 60
LOT         = 50
COSTS_INR   = 105.0
STOP_FLOOR  = -3000.0
DAILY_HALT  = -15000.0
MAX_TRADES  = 3
HARD_CUTOFF = dtime(14, 55)
SKIP_START  = dtime(11, 0)
SKIP_END    = dtime(12, 0)
ENTRY_MIN   = 30
SIGNAL_PCT  = 0.85
HIGH_CONF_PCT = 0.95
SKIP_REGIMES = {"compression"}

# Forward window
FWD_START = pd.Timestamp("2024-11-01")
FWD_END   = pd.Timestamp("2026-05-31")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Build proxy feature table
# ═════════════════════════════════════════════════════════════════════════════

def build_proxy_features() -> pd.DataFrame:
    """Build minute features from NIFTY index, Nov-2024 onward.

    Uses NIFTY index OHLC for all f_* columns (futures basis is small).
    Sets f_oi = f_vol = 0 (these features carry 0.7 % of model gain).
    """
    print("\n" + "═" * 90)
    print("  PHASE 0: Build proxy feature table (NIFTY index, Nov-2024 → May-2026)")
    print("═" * 90)

    raw = pd.read_csv(NIFTY_MIN_PATH)
    raw["datetime"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["datetime"] >= FWD_START) & (raw["datetime"] <= FWD_END)].copy()

    # Map index OHLC to all the schema columns the feature builder expects
    raw = raw.rename(columns={
        "open": "f_open", "high": "f_high", "low": "f_low",
        "close": "f_close", "volume": "f_vol",
    })
    # The index CSV has volume=0 for every minute.  Replace with a constant 1
    # so that cumulative VWAP becomes a simple cumulative-mean of price (a
    # reasonable proxy when actual volume is unavailable).
    raw["f_vol"] = 1.0
    raw["f_oi"] = 0.0
    raw["s_close"] = raw["f_close"]   # spot ≈ futures for proxy
    raw["trade_date"] = raw["datetime"].dt.normalize()
    raw = raw.sort_values("datetime").reset_index(drop=True)

    print(f"  raw bars Nov-2024 → May-2026: {len(raw):,}")
    print(f"  date range: {raw['datetime'].min()} → {raw['datetime'].max()}")
    print(f"  unique trading days: {raw['trade_date'].nunique()}")

    # Compute features per day (compute_features expects a single day)
    # OI-dependent features (oi_chg_*, vol_oi_ratio, vol_zscore) come back NaN
    # because f_oi=0 in proxy data; fill with 0 so dropna doesn't kill rows.
    OI_FILL_COLS = ["oi_chg_1m", "oi_chg_5m", "oi_chg_30m",
                     "vol_oi_ratio", "vol_zscore"]

    frames = []
    bad = 0
    for d, day_df in raw.groupby("trade_date"):
        if len(day_df) < 60:           # need at least an hour for warmup
            bad += 1
            continue
        try:
            feats = compute_features(day_df, d.date())
            if not feats.empty:
                for col in OI_FILL_COLS:
                    if col in feats.columns:
                        feats[col] = feats[col].fillna(0.0)
                frames.append(feats)
        except Exception as exc:
            print(f"  skip {d.date()}: {exc}")
            bad += 1

    full = pd.concat(frames, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["datetime"])
    full["trade_date"] = pd.to_datetime(full["trade_date"])
    pre_dropna = len(full)

    # Diagnostic: per-column NaN counts on FUTURES_FEATURES
    nan_counts = full[FUTURES_FEATURES].isna().sum().sort_values(ascending=False)
    bad_cols = nan_counts[nan_counts > 0]
    if len(bad_cols):
        print(f"  features with NaN ({len(bad_cols)} cols):")
        for c, n in bad_cols.head(15).items():
            print(f"    {c:<28s}  {n:>7,d} ({100*n/pre_dropna:.1f}%)")

    full = full.dropna(subset=FUTURES_FEATURES)
    print(f"  dropna effect: {pre_dropna:,} → {len(full):,} rows "
          f"({pre_dropna - len(full):,} dropped)")
    full = add_regime(full)

    full.to_parquet(PROXY_FEAT_PATH, index=False)
    print(f"\n  feature rows: {len(full):,}")
    print(f"  unique days: {full['trade_date'].nunique()}")
    print(f"  skipped days: {bad}")
    print(f"  → {PROXY_FEAT_PATH}")
    print(f"\n  Regime distribution:")
    print(full["regime"].value_counts().to_string())
    return full


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Q4 2024 hypothesis test
# ═════════════════════════════════════════════════════════════════════════════

def load_vix_daily() -> pd.DataFrame:
    vix = pd.read_csv(VIX_DAY_PATH)
    vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
    vix = vix.rename(columns={"close": "vix_close",
                                "high": "vix_high", "low": "vix_low",
                                "open": "vix_open"})
    return vix[["date", "vix_open", "vix_high", "vix_low", "vix_close"]]


def load_nifty_daily() -> pd.DataFrame:
    nd = pd.read_csv(NIFTY_DAILY_PATH)
    nd["date"] = pd.to_datetime(nd["date"], utc=True).dt.tz_convert(None).dt.normalize()
    nd = nd.sort_values("date").reset_index(drop=True)
    nd["nifty_close"] = nd["close"]
    nd["ret_1d"]  = nd["close"].pct_change(1)
    nd["ret_3d"]  = nd["close"].pct_change(3)
    nd["ret_5d"]  = nd["close"].pct_change(5)
    nd["ret_10d"] = nd["close"].pct_change(10)
    return nd[["date", "nifty_close", "ret_1d", "ret_3d", "ret_5d", "ret_10d"]]


def phase1_hypothesis_test() -> dict:
    print("\n" + "═" * 90)
    print("  PHASE 1: Q4-2024 regime hypothesis test")
    print("═" * 90)

    trades = pd.read_parquet(EXISTING_TRADES)
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()
    print(f"  loaded {len(trades)} trades from 2024 ledger")

    vix = load_vix_daily()
    nd  = load_nifty_daily()

    # Use prior-day's VIX close (no look-ahead)
    vix_prior = vix.copy()
    vix_prior["date"] = vix_prior["date"] + pd.Timedelta(days=1)
    vix_prior = vix_prior.rename(columns={"vix_close": "vix_prev_close"})

    # 5d return is computed up to & including prior day
    nd_prior = nd.copy()
    nd_prior["date"] = nd_prior["date"] + pd.Timedelta(days=1)
    nd_prior = nd_prior.rename(columns={
        "ret_1d": "ret_1d_prev", "ret_3d": "ret_3d_prev",
        "ret_5d": "ret_5d_prev", "ret_10d": "ret_10d_prev",
    })

    # Merge prior-day signals onto each trade
    df = trades.merge(vix_prior[["date", "vix_prev_close"]],
                       left_on="trade_date", right_on="date", how="left")
    df = df.merge(nd_prior[["date", "ret_1d_prev", "ret_3d_prev",
                              "ret_5d_prev", "ret_10d_prev"]],
                   left_on="trade_date", right_on="date", how="left",
                   suffixes=("", "_nd"))

    # Forward-fill VIX (weekend / holiday gaps)
    df = df.sort_values("trade_date")
    df["vix_prev_close"] = df["vix_prev_close"].ffill()
    df["ret_5d_prev"]    = df["ret_5d_prev"].ffill()
    df["ret_3d_prev"]    = df["ret_3d_prev"].ffill()

    df["win"] = (df["net_pnl_inr"] > 0).astype(int)
    df["quarter"] = df["trade_date"].dt.to_period("Q").astype(str)
    df["month"]   = df["trade_date"].dt.to_period("M").astype(str)

    rows = []
    def block(name, sub):
        if len(sub) == 0:
            return
        rows.append({
            "bin": name,
            "n_trades": len(sub),
            "win_rate": float((sub["net_pnl_inr"] > 0).mean()),
            "mean_pnl": float(sub["net_pnl_inr"].mean()),
            "total_pnl": float(sub["net_pnl_inr"].sum()),
        })

    print("\n  ── PnL × Quarter ──")
    for q in sorted(df["quarter"].unique()):
        block(f"quarter={q}", df[df["quarter"] == q])

    print("\n  ── PnL × VIX bucket (prior day's close) ──")
    bins = [(0, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 100)]
    for lo, hi in bins:
        sub = df[(df["vix_prev_close"] >= lo) & (df["vix_prev_close"] < hi)]
        block(f"VIX [{lo},{hi})", sub)

    print("\n  ── PnL × 5-day NIFTY return bucket (prior 5d, ending t-1) ──")
    rbins = [(-0.10, -0.04), (-0.04, -0.02), (-0.02, -0.01),
             (-0.01,  0.00), (0.00,  0.01), (0.01,  0.02),
             (0.02,  0.04),  (0.04,  0.10)]
    for lo, hi in rbins:
        sub = df[(df["ret_5d_prev"] >= lo) & (df["ret_5d_prev"] < hi)]
        block(f"5d-ret [{lo:+.2f},{hi:+.2f})", sub)

    print("\n  ── October 2024 zoom ──")
    for m in ["2024-09", "2024-10", "2024-11", "2024-12"]:
        block(f"month={m}", df[df["month"] == m])

    # Print compactly
    print("\n  Summary table (sorted by total_pnl ascending):")
    summary = pd.DataFrame(rows).sort_values("total_pnl")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # Count Oct trades skipped under various filters
    oct = df[df["month"] == "2024-10"]
    print(f"\n  October 2024 had {len(oct)} trades, total PnL ₹{oct['net_pnl_inr'].sum():,.0f}, "
          f"win {oct['win'].mean()*100:.1f}%")
    if len(oct):
        print(f"    median VIX on Oct trade days: {oct['vix_prev_close'].median():.2f}")
        print(f"    median 5d-ret on Oct trade days: {oct['ret_5d_prev'].median()*100:+.2f}%")

    out = {
        "n_total_trades": int(len(df)),
        "buckets": rows,
    }
    (OUT_DIR / "phase1_hypothesis_buckets.json").write_text(
        json.dumps(out, indent=2, default=str))

    return out


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Add regime guards & tune thresholds on 2024
# ═════════════════════════════════════════════════════════════════════════════

def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "n_days": 0, "win_rate": 0.0,
                 "total_pnl_inr": 0.0, "mean_pnl_inr": 0.0,
                 "sharpe_daily_ann": 0.0, "profit_factor": 0.0,
                 "max_drawdown_inr": 0.0}
    p = trades["net_pnl_inr"]
    daily = trades.groupby("trade_date")["net_pnl_inr"].sum()
    n_days = int(trades["trade_date"].nunique())
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
               if len(daily) > 1 and daily.std() > 0 else 0.0)
    cum = p.cumsum()
    dd = float((cum - cum.cummax()).min())
    gw = p[p > 0].sum()
    gl = abs(p[p < 0].sum())
    pf = float(gw / gl) if gl > 0 else float("inf")
    return {
        "n_trades": int(len(trades)),
        "n_days": n_days,
        "win_rate": float((p > 0).mean()),
        "mean_pnl_inr": float(p.mean()),
        "total_pnl_inr": float(p.sum()),
        "sharpe_daily_ann": float(sharpe),
        "profit_factor": pf,
        "max_drawdown_inr": float(dd),
    }


def phase2_tune_guards() -> dict:
    print("\n" + "═" * 90)
    print("  PHASE 2: Grid-search regime guards on 2024 trade ledger")
    print("═" * 90)

    trades = pd.read_parquet(EXISTING_TRADES)
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()

    vix = load_vix_daily()
    nd  = load_nifty_daily()

    vix_prior = vix.copy()
    vix_prior["date"] = vix_prior["date"] + pd.Timedelta(days=1)
    nd_prior = nd.copy()
    nd_prior["date"] = nd_prior["date"] + pd.Timedelta(days=1)
    nd_prior = nd_prior.rename(columns={"ret_5d": "ret_5d_prev",
                                          "ret_3d": "ret_3d_prev"})

    df = trades.merge(vix_prior[["date", "vix_close"]].rename(
        columns={"vix_close": "vix_prev_close"}),
                       left_on="trade_date", right_on="date", how="left")
    df = df.merge(nd_prior[["date", "ret_5d_prev", "ret_3d_prev"]],
                   left_on="trade_date", right_on="date", how="left",
                   suffixes=("", "_nd"))
    df = df.sort_values("trade_date")
    df["vix_prev_close"] = df["vix_prev_close"].ffill()
    df["ret_5d_prev"]    = df["ret_5d_prev"].ffill()
    df["ret_3d_prev"]    = df["ret_3d_prev"].ffill()

    base = metrics(df)
    print(f"\n  baseline (no guards) → "
          f"n={base['n_trades']:>4d}  win={base['win_rate']*100:>5.1f}%  "
          f"PnL=₹{base['total_pnl_inr']:>+11,.0f}  "
          f"Sharpe={base['sharpe_daily_ann']:>+5.2f}  "
          f"DD=₹{base['max_drawdown_inr']:>+10,.0f}  PF={base['profit_factor']:.2f}")

    vix_cuts = [None, 14, 15, 16, 17, 18, 19, 20, 22]
    ret_cuts = [None, -0.005, -0.01, -0.015, -0.02, -0.025, -0.03, -0.04]

    rows = []
    print("\n  Grid search (VIX ceiling × 5d-return floor):")
    print(f"  {'VIX≤':>5s} {'5d≥':>7s}  {'n':>5s} {'win%':>6s} {'totPnL':>11s} {'Sharpe':>7s}  {'DD':>11s} {'PF':>6s}")
    for vc in vix_cuts:
        for rc in ret_cuts:
            sub = df.copy()
            if vc is not None:
                sub = sub[sub["vix_prev_close"] < vc]
            if rc is not None:
                sub = sub[sub["ret_5d_prev"] > rc]
            m = metrics(sub)
            row = {"vix_cut": vc, "ret_cut": rc, **m}
            rows.append(row)
            vc_s = "─" if vc is None else f"{vc:.0f}"
            rc_s = "─" if rc is None else f"{rc*100:+.1f}%"
            print(f"  {vc_s:>5s} {rc_s:>7s}  {m['n_trades']:>5d} "
                  f"{m['win_rate']*100:>5.1f}% ₹{m['total_pnl_inr']:>+10,.0f} "
                  f"{m['sharpe_daily_ann']:>+6.2f}  ₹{m['max_drawdown_inr']:>+10,.0f} "
                  f"{m['profit_factor']:>5.2f}")

    grid = pd.DataFrame(rows)
    grid.to_parquet(OUT_DIR / "phase2_guard_grid.parquet", index=False)

    # Recommend: best Sharpe with min n_trades >= 200 (keep statistical power)
    eligible = grid[grid["n_trades"] >= 200].copy()
    if eligible.empty:
        eligible = grid[grid["n_trades"] >= 100].copy()
    best = eligible.sort_values("sharpe_daily_ann", ascending=False).iloc[0]
    print(f"\n  RECOMMENDED guards (best Sharpe with n≥200 trades):")
    print(f"    vix_cut={best['vix_cut']}, ret_cut={best['ret_cut']}")
    print(f"    n={best['n_trades']:.0f}, win={best['win_rate']*100:.1f}%, "
          f"PnL=₹{best['total_pnl_inr']:+,.0f}, Sharpe={best['sharpe_daily_ann']:+.2f}")

    return {
        "baseline": base,
        "best_guard": {
            "vix_cut": float(best["vix_cut"]) if pd.notna(best["vix_cut"]) else None,
            "ret_cut": float(best["ret_cut"]) if pd.notna(best["ret_cut"]) else None,
            **{k: float(best[k]) for k in [
                "n_trades", "win_rate", "total_pnl_inr",
                "sharpe_daily_ann", "max_drawdown_inr", "profit_factor"]}
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Forward-walk Nov-2024 → May-2026
# ═════════════════════════════════════════════════════════════════════════════

def simulate_long_trade(entry_idx: int, day_bars: pd.DataFrame,
                         entry_px: float, size_mult: float) -> dict:
    target_px = entry_px * (1 + TARGET_PCT)
    stop_px   = entry_px * (1 - STOP_PCT)
    j_end = min(len(day_bars), entry_idx + 1 + HORIZON)
    walk = day_bars.iloc[entry_idx+1:j_end]

    exit_px = None
    exit_reason = None
    for _, r in walk.iterrows():
        c = r["fut_close"]
        if c >= target_px:
            exit_px, exit_reason = target_px, "TARGET"
            break
        if c <= stop_px:
            exit_px, exit_reason = stop_px, "STOP"
            break
    if exit_px is None:
        if walk.empty:
            exit_px = entry_px
            exit_reason = "NO_BARS"
        else:
            exit_px = float(walk["fut_close"].iloc[-1])
            exit_reason = "TIME"

    gross = (exit_px - entry_px) * LOT * size_mult
    net   = gross - COSTS_INR * size_mult
    was_stopped = exit_reason == "STOP"
    if net < STOP_FLOOR:
        net = STOP_FLOOR
        was_stopped = True
    return {
        "exit_px": exit_px,
        "exit_reason": exit_reason,
        "net_pnl_inr": net,
        "was_stopped": was_stopped,
        "size_mult": size_mult,
    }


def build_fut_cache_from_proxy(features: pd.DataFrame) -> dict:
    """fut_cache[date] = DataFrame with datetime + fut_close columns."""
    cache = {}
    for d, sub in features.groupby("trade_date"):
        bars = sub[["datetime", "f_close"]].rename(
            columns={"f_close": "fut_close"}).sort_values(
            "datetime").reset_index(drop=True)
        cache[d.date()] = bars
    return cache


def run_forward_backtest(features: pd.DataFrame, model: lgb.Booster,
                           vix: pd.DataFrame, nd: pd.DataFrame,
                           vix_cut: float | None,
                           ret_cut: float | None) -> pd.DataFrame:
    """Run LONG-only backtest with all hard filters + regime guards."""

    # Apply hard filters first (same as production)
    df = features.copy()
    t = df["datetime"].dt.time
    mod = df["minute_of_day"] - 9 * 60 - 15
    df = df[
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~df["regime"].isin(SKIP_REGIMES)
    ].copy()

    # Score
    df["long_score"] = model.predict(df[FUTURES_FEATURES])

    # Per-day percentile thresholds
    p85 = df.groupby("trade_date")["long_score"].transform(
        lambda s: s.quantile(SIGNAL_PCT))
    p95 = df.groupby("trade_date")["long_score"].transform(
        lambda s: s.quantile(HIGH_CONF_PCT))
    df["take_long"] = df["long_score"] >= p85
    df["size_mult"] = np.where(df["long_score"] >= p95, 1.5, 1.0)

    # Apply regime guards (day-level filter using prior-day's VIX & 5d-ret)
    vix_prior = vix.copy()
    vix_prior["date"] = vix_prior["date"] + pd.Timedelta(days=1)
    vix_prior = vix_prior.rename(columns={"vix_close": "vix_prev_close"})

    nd_prior = nd.copy()
    nd_prior["date"] = nd_prior["date"] + pd.Timedelta(days=1)
    nd_prior = nd_prior.rename(columns={"ret_5d": "ret_5d_prev"})

    df = df.merge(vix_prior[["date", "vix_prev_close"]],
                   left_on="trade_date", right_on="date", how="left")
    df = df.drop(columns=["date"], errors="ignore")
    df = df.merge(nd_prior[["date", "ret_5d_prev"]],
                   left_on="trade_date", right_on="date", how="left",
                   suffixes=("", "_nd"))
    df = df.drop(columns=["date"], errors="ignore")

    df = df.sort_values("datetime")
    df["vix_prev_close"] = df["vix_prev_close"].ffill()
    df["ret_5d_prev"]    = df["ret_5d_prev"].ffill()

    if vix_cut is not None:
        df = df[df["vix_prev_close"] < vix_cut].copy()
    if ret_cut is not None:
        df = df[df["ret_5d_prev"] > ret_cut].copy()

    candidates = df[df["take_long"]].sort_values(
        ["trade_date", "datetime"]).reset_index(drop=True)

    fut_cache = build_fut_cache_from_proxy(features)

    results = []
    daily_pnl = {}
    daily_count = {}

    for _, row in candidates.iterrows():
        td = row["trade_date"]
        if daily_count.get(td, 0) >= MAX_TRADES:
            continue
        if daily_pnl.get(td, 0.0) <= DAILY_HALT:
            continue
        day_bars = fut_cache.get(td.date())
        if day_bars is None:
            continue
        idx_arr = day_bars.index[day_bars["datetime"] == row["datetime"]]
        if len(idx_arr) == 0:
            continue
        i = int(idx_arr[0])
        entry_px = float(day_bars["fut_close"].iat[i])
        sim = simulate_long_trade(i, day_bars, entry_px, float(row["size_mult"]))
        results.append({
            "trade_date": td,
            "datetime": row["datetime"],
            "regime": row["regime"],
            "long_score": float(row["long_score"]),
            "entry_px": entry_px,
            "vix_prev_close": float(row["vix_prev_close"])
                if pd.notna(row["vix_prev_close"]) else np.nan,
            "ret_5d_prev": float(row["ret_5d_prev"])
                if pd.notna(row["ret_5d_prev"]) else np.nan,
            **sim,
        })
        daily_pnl[td]   = daily_pnl.get(td, 0.0) + sim["net_pnl_inr"]
        daily_count[td] = daily_count.get(td, 0) + 1

    return pd.DataFrame(results)


def phase3_forward_walk(features: pd.DataFrame,
                          best_guard: dict) -> pd.DataFrame:
    print("\n" + "═" * 90)
    print("  PHASE 3: 19-month forward-walk Nov-2024 → May-2026")
    print("═" * 90)

    model = lgb.Booster(model_file=str(MODEL_PATH))
    vix = load_vix_daily()
    nd  = load_nifty_daily()

    vix_cut = best_guard.get("vix_cut")
    ret_cut = best_guard.get("ret_cut")
    print(f"\n  Using guards: vix_cut={vix_cut}, ret_cut={ret_cut}")

    print("\n  ── Variant A: HARD FILTERS ONLY (no regime guards) ──")
    trades_a = run_forward_backtest(features, model, vix, nd, None, None)
    print_full(trades_a)

    print("\n  ── Variant B: HARD FILTERS + RECOMMENDED GUARDS ──")
    trades_b = run_forward_backtest(features, model, vix, nd, vix_cut, ret_cut)
    print_full(trades_b)

    # Save
    trades_a.to_parquet(OUT_DIR / "phase3_fwd_no_guard.parquet", index=False)
    trades_b.to_parquet(OUT_DIR / "phase3_fwd_with_guard.parquet", index=False)

    return trades_a, trades_b


def print_full(trades: pd.DataFrame):
    if trades.empty:
        print("    NO TRADES")
        return
    m = metrics(trades)
    print(f"    OVERALL  n={m['n_trades']:>4d}  days={m['n_days']:>3d}  "
          f"win={m['win_rate']*100:>5.1f}%  PnL=₹{m['total_pnl_inr']:>+11,.0f}  "
          f"Sharpe={m['sharpe_daily_ann']:>+5.2f}  PF={m['profit_factor']:>4.2f}  "
          f"DD=₹{m['max_drawdown_inr']:>+10,.0f}")

    trades = trades.copy()
    trades["month"] = pd.to_datetime(trades["datetime"]).dt.to_period("M").astype(str)
    print("    by month:")
    for mo, sub in trades.groupby("month"):
        mm = metrics(sub)
        print(f"      {mo}  n={mm['n_trades']:>3d}  days={mm['n_days']:>2d}  "
              f"win={mm['win_rate']*100:>5.1f}%  "
              f"PnL=₹{mm['total_pnl_inr']:>+9,.0f}  "
              f"Sharpe={mm['sharpe_daily_ann']:>+5.2f}")

    print("    by regime:")
    for r, sub in trades.groupby("regime"):
        rm = metrics(sub)
        print(f"      {r:<14s}  n={rm['n_trades']:>3d}  "
              f"win={rm['win_rate']*100:>5.1f}%  "
              f"PnL=₹{rm['total_pnl_inr']:>+9,.0f}  PF={rm['profit_factor']:>4.2f}")

    print("    by exit reason:")
    for r, sub in trades.groupby("exit_reason"):
        em = metrics(sub)
        print(f"      {r:<14s}  n={em['n_trades']:>3d}  "
              f"win={em['win_rate']*100:>5.1f}%  "
              f"mean=₹{em['mean_pnl_inr']:>+6,.0f}  "
              f"total=₹{em['total_pnl_inr']:>+9,.0f}")


# ═════════════════════════════════════════════════════════════════════════════
# Main driver
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("╔" + "═" * 88 + "╗")
    print("║  NIFTY long-only futures engine — end-to-end forward validation".ljust(89) + "║")
    print("║  Phases 0-3, with proxy data for the post-Oct-2024 window".ljust(89) + "║")
    print("╚" + "═" * 88 + "╝")

    # Phase 0
    if PROXY_FEAT_PATH.exists():
        print(f"\n[skip Phase 0]  using cached {PROXY_FEAT_PATH}")
        features = pd.read_parquet(PROXY_FEAT_PATH)
        features["datetime"]   = pd.to_datetime(features["datetime"])
        features["trade_date"] = pd.to_datetime(features["trade_date"])
    else:
        features = build_proxy_features()

    # Phase 1
    p1 = phase1_hypothesis_test()

    # Phase 2
    p2 = phase2_tune_guards()

    # Phase 3
    trades_a, trades_b = phase3_forward_walk(features, p2["best_guard"])

    # Save consolidated summary
    summary = {
        "phase1_buckets_n": len(p1["buckets"]),
        "phase2_baseline": p2["baseline"],
        "phase2_best_guard": p2["best_guard"],
        "phase3_no_guard": metrics(trades_a),
        "phase3_with_guard": metrics(trades_b),
    }
    (OUT_DIR / "forward_walk_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n→ {OUT_DIR}/forward_walk_summary.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
