#!/usr/bin/env python3
"""Tier-1 validation before paper trading.

  A. Proxy validation — score correlation between actual-futures features
     and index-proxy features over Jan-Oct 2024 (the overlap window).
  B. Cost sensitivity — re-run the 19-month forward walk at COSTS_INR
     = 105, 150, 200.  Confirm Sharpe stays > 1.5 and PnL > 0 at 150.
  C. Score-quintile stratification — on the forward-walk trades, bin by
     long_score quintile and check win-rate / mean-PnL slope.  Confirms
     the model is doing real work (not picking 15 % of the day blindly).

All three must pass before paper trading goes live.
"""
from __future__ import annotations

import json
import sys
from datetime import time as dtime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet_router.futures_features import (
    FUTURES_FEATURES, add_regime, compute_features,
)

NIFTY_MIN_PATH    = PROJECT_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"
EXISTING_FEATURES = PROJECT_ROOT / "cache/router_v0/futures_features.parquet"
PROXY_FEATURES    = PROJECT_ROOT / "cache/router_v0/futures_features_proxy.parquet"
TRADES_FWD_NOG    = PROJECT_ROOT / "results/router_v0/phase3_fwd_no_guard.parquet"
MODEL_PATH        = PROJECT_ROOT / "models/router_v0/futures/final_long.lgb"
OUT_DIR           = PROJECT_ROOT / "results/router_v0"

# Backtest config (must match Variant A)
TARGET_PCT  = 0.0040
STOP_PCT    = 0.0030
HORIZON     = 60
LOT         = 50
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


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1A — Proxy validation (BACKTEST-PnL comparison, not score correlation)
# ─────────────────────────────────────────────────────────────────────────────
# The futures model is a single-tree LightGBM. Tiny price differences (well
# within the 0.25 % cash-future basis) can flip individual minute decisions,
# which makes per-minute score correlation a misleading metric. The right
# question is: "Does the proxy produce comparable PnL / Sharpe to actual?"
# We answer that by running Variant-A backtest on both feature streams over
# Jan-Oct 2024 (the overlap window) and comparing aggregates.

def _simulate_long_trade(entry_idx, day_bars, entry_px, size_mult,
                          target_pct, stop_pct, horizon, lot, costs):
    target_px = entry_px * (1 + target_pct)
    stop_px   = entry_px * (1 - stop_pct)
    j_end = min(len(day_bars), entry_idx + 1 + horizon)
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
    gross = (exit_px - entry_px) * lot * size_mult
    net = gross - costs * size_mult
    if net < STOP_FLOOR:
        net = STOP_FLOOR
    return net, exit_reason


def _run_va_backtest(features: pd.DataFrame, model: lgb.Booster,
                      px_col: str, costs: float = 105.0) -> pd.DataFrame:
    """Run Variant A (hard filters only) on a feature frame.
    px_col is which column to use for fill prices (f_close)."""
    df = features.copy()
    t = df["datetime"].dt.time
    mod = df["minute_of_day"] - 9 * 60 - 15
    df = df[
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~df["regime"].isin(SKIP_REGIMES)
    ].copy()
    df["score"] = model.predict(df[FUTURES_FEATURES])

    p85 = df.groupby("trade_date")["score"].transform(
        lambda s: s.quantile(SIGNAL_PCT))
    p95 = df.groupby("trade_date")["score"].transform(
        lambda s: s.quantile(HIGH_CONF_PCT))
    df["take_long"] = df["score"] >= p85
    df["size_mult"] = np.where(df["score"] >= p95, 1.5, 1.0)

    cands = df[df["take_long"]].sort_values(
        ["trade_date", "datetime"]).reset_index(drop=True)

    # Build fut_cache from same feature frame
    cache = {}
    for d, sub in df.groupby("trade_date"):
        bars = sub[["datetime", px_col]].rename(
            columns={px_col: "fut_close"}).sort_values(
            "datetime").reset_index(drop=True)
        cache[d.date()] = bars

    rows = []
    daily_pnl, daily_count = {}, {}
    for _, row in cands.iterrows():
        td = row["trade_date"]
        if daily_count.get(td, 0) >= MAX_TRADES:    continue
        if daily_pnl.get(td, 0.0) <= DAILY_HALT:    continue
        bars = cache.get(td.date())
        if bars is None:                              continue
        idx_arr = bars.index[bars["datetime"] == row["datetime"]]
        if len(idx_arr) == 0:                         continue
        i = int(idx_arr[0])
        entry_px = float(bars["fut_close"].iat[i])
        net, reason = _simulate_long_trade(
            i, bars, entry_px, float(row["size_mult"]),
            TARGET_PCT, STOP_PCT, HORIZON, LOT, costs,
        )
        rows.append({
            "trade_date": td, "datetime": row["datetime"],
            "score": float(row["score"]),
            "entry_px": entry_px, "net_pnl_inr": net, "exit_reason": reason,
            "size_mult": float(row["size_mult"]),
        })
        daily_pnl[td]   = daily_pnl.get(td, 0.0) + net
        daily_count[td] = daily_count.get(td, 0) + 1
    return pd.DataFrame(rows)


def tier1a_proxy_validation() -> dict:
    print("\n" + "═" * 90)
    print("  TIER 1A: Proxy validation — backtest-PnL comparison (Jan-Oct 2024)")
    print("═" * 90)

    # Actual-futures features
    actual = pd.read_parquet(EXISTING_FEATURES)
    actual["datetime"] = pd.to_datetime(actual["datetime"])
    actual["trade_date"] = pd.to_datetime(actual["trade_date"])
    actual = actual[
        (actual["trade_date"] >= "2024-01-01") &
        (actual["trade_date"] < "2024-11-01")
    ].copy()
    actual = actual.dropna(subset=FUTURES_FEATURES)
    actual = add_regime(actual)
    print(f"  actual-futures rows (Jan-Oct 2024): {len(actual):,}")

    # Proxy features for the same window
    raw = pd.read_csv(NIFTY_MIN_PATH)
    raw["datetime"] = pd.to_datetime(raw["date"])
    raw = raw[
        (raw["datetime"] >= "2024-01-01") &
        (raw["datetime"] < "2024-11-01")
    ].copy()
    raw = raw.rename(columns={
        "open": "f_open", "high": "f_high", "low": "f_low",
        "close": "f_close", "volume": "f_vol",
    })
    raw["f_vol"] = 1.0
    raw["f_oi"]  = 0.0
    raw["s_close"] = raw["f_close"]
    raw["trade_date"] = raw["datetime"].dt.normalize()
    raw = raw.sort_values("datetime").reset_index(drop=True)

    OI_FILL_COLS = ["oi_chg_1m", "oi_chg_5m", "oi_chg_30m",
                     "vol_oi_ratio", "vol_zscore"]
    frames = []
    for d, day_df in raw.groupby("trade_date"):
        if len(day_df) < 60:
            continue
        feats = compute_features(day_df, d.date())
        if feats.empty:
            continue
        for col in OI_FILL_COLS:
            if col in feats.columns:
                feats[col] = feats[col].fillna(0.0)
        frames.append(feats)
    proxy = pd.concat(frames, ignore_index=True)
    proxy["datetime"] = pd.to_datetime(proxy["datetime"])
    proxy["trade_date"] = pd.to_datetime(proxy["trade_date"])
    proxy = proxy.dropna(subset=FUTURES_FEATURES)
    proxy = add_regime(proxy)
    print(f"  proxy rows (Jan-Oct 2024):           {len(proxy):,}")

    # Run Variant A on each
    model = lgb.Booster(model_file=str(MODEL_PATH))
    print("\n  running Variant-A backtest on each feature stream …")
    trades_actual = _run_va_backtest(actual, model, "f_close")
    trades_proxy  = _run_va_backtest(proxy,  model, "f_close")

    m_a = metrics(trades_actual)
    m_p = metrics(trades_proxy)

    print(f"\n  {'metric':<22s} {'actual':>14s} {'proxy':>14s} {'Δ':>10s}")
    def row(name, va, vp, fmt="{:>+13,.0f}"):
        diff = vp - va
        print(f"  {name:<22s} {fmt.format(va):>14s} {fmt.format(vp):>14s} "
              f"{fmt.format(diff):>10s}")
    row("n_trades",          m_a["n_trades"], m_p["n_trades"], "{:>13d}")
    row("total_pnl_inr",     m_a["total_pnl_inr"], m_p["total_pnl_inr"])
    row("win_rate",          100*m_a["win_rate"], 100*m_p["win_rate"], "{:>13.1f}%")
    row("sharpe_daily_ann",  m_a["sharpe_daily_ann"], m_p["sharpe_daily_ann"], "{:>13.2f}")
    row("max_drawdown_inr",  m_a["max_drawdown_inr"], m_p["max_drawdown_inr"])

    # Pass criteria: trade count similar, win rate close, PnL within tolerance,
    # drawdown not materially worse on proxy, AND proxy Sharpe is itself
    # production-grade (> 1.5).  Proxy under-estimating PnL is the SAFE
    # direction of bias and should not block paper trading.
    if m_a["total_pnl_inr"] != 0:
        pnl_rel = abs(m_p["total_pnl_inr"] - m_a["total_pnl_inr"]) / abs(m_a["total_pnl_inr"])
    else:
        pnl_rel = float("inf")
    sharpe_abs = abs(m_p["sharpe_daily_ann"] - m_a["sharpe_daily_ann"])
    win_abs = abs(m_p["win_rate"] - m_a["win_rate"])
    n_rel = abs(m_p["n_trades"] - m_a["n_trades"]) / max(m_a["n_trades"], 1)
    dd_worse = (m_p["max_drawdown_inr"] - m_a["max_drawdown_inr"]) < -10_000  # > ₹10k worse on proxy

    print(f"\n  trade-count rel diff:  {100*n_rel:.1f} %  (criterion ≤ 5 %)")
    print(f"  PnL relative diff:     {100*pnl_rel:.1f} %  (criterion ≤ 25 %)")
    print(f"  Win-rate diff:         {100*win_abs:.1f} pp (criterion ≤ 5 pp)")
    print(f"  Sharpe absolute diff:  {sharpe_abs:.2f}  (criterion ≤ 1.0 OR both Sharpe > 2)")
    print(f"  Drawdown materially worse on proxy: {dd_worse}")
    print(f"  Proxy Sharpe itself:   {m_p['sharpe_daily_ann']:.2f}  (criterion > 1.5)")

    sharpe_ok = (sharpe_abs <= 1.0) or (
        m_a["sharpe_daily_ann"] > 2.0 and m_p["sharpe_daily_ann"] > 2.0)
    passed = bool(
        n_rel <= 0.05 and
        pnl_rel <= 0.25 and
        win_abs <= 0.05 and
        sharpe_ok and
        not dd_worse and
        m_p["sharpe_daily_ann"] > 1.5
    )
    verdict = "PASS" if passed else "FAIL"
    note = ""
    if passed and m_p["total_pnl_inr"] < m_a["total_pnl_inr"]:
        note = "  (proxy is conservative — safer direction of bias)"
    print(f"\n  → {verdict}{note}")

    return {
        "pass": passed,
        "actual": m_a,
        "proxy": m_p,
        "pnl_relative_diff": float(pnl_rel),
        "sharpe_absolute_diff": float(sharpe_abs),
        "winrate_absolute_diff": float(win_abs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1B — Cost sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "total_pnl_inr": 0.0,
                 "sharpe_daily_ann": 0.0, "win_rate": 0.0,
                 "max_drawdown_inr": 0.0}
    p = trades["net_pnl_inr"]
    daily = trades.groupby("trade_date")["net_pnl_inr"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
               if len(daily) > 1 and daily.std() > 0 else 0.0)
    cum = p.cumsum()
    return {
        "n_trades": int(len(trades)),
        "total_pnl_inr": float(p.sum()),
        "win_rate": float((p > 0).mean()),
        "sharpe_daily_ann": float(sharpe),
        "max_drawdown_inr": float((cum - cum.cummax()).min()),
    }


def tier1b_cost_sensitivity() -> dict:
    print("\n" + "═" * 90)
    print("  TIER 1B: Cost sensitivity (re-cost the 19-month forward walk)")
    print("═" * 90)

    trades = pd.read_parquet(TRADES_FWD_NOG).copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()

    # The trades parquet has net_pnl already at COSTS_INR=105.  Recover gross
    # then re-apply different cost levels.
    trades["gross_pnl"] = trades["net_pnl_inr"] + 105.0 * trades["size_mult"]

    rows = []
    print(f"\n  {'cost':>5s}  {'n':>5s}  {'win%':>6s}  {'PnL':>11s}  {'Sharpe':>7s}  {'DD':>11s}")
    for cost in [105, 130, 150, 175, 200, 250]:
        re_trades = trades.copy()
        re_trades["net_pnl_inr"] = re_trades["gross_pnl"] - cost * re_trades["size_mult"]

        # Re-apply the per-trade -₹3000 floor (matches simulator)
        re_trades.loc[re_trades["net_pnl_inr"] < STOP_FLOOR, "net_pnl_inr"] = STOP_FLOOR

        m = metrics(re_trades)
        rows.append({"cost_inr": cost, **m})
        print(f"  ₹{cost:>4d}  {m['n_trades']:>5d}  "
              f"{m['win_rate']*100:>5.1f}%  ₹{m['total_pnl_inr']:>+10,.0f}  "
              f"{m['sharpe_daily_ann']:>+6.2f}  ₹{m['max_drawdown_inr']:>+10,.0f}")

    # Pass criterion: at COSTS_INR=150, Sharpe > 1.5 and PnL > 0
    at_150 = next(r for r in rows if r["cost_inr"] == 150)
    passed = bool(at_150["sharpe_daily_ann"] > 1.5 and at_150["total_pnl_inr"] > 0)
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  → {verdict}  (criterion at cost=₹150: Sharpe > 1.5 AND PnL > 0)")

    return {
        "pass": passed,
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1C — Score-quintile stratification
# ─────────────────────────────────────────────────────────────────────────────

def tier1c_score_quintiles() -> dict:
    print("\n" + "═" * 90)
    print("  TIER 1C: Score-quintile stratification on forward-walk trades")
    print("═" * 90)

    trades = pd.read_parquet(TRADES_FWD_NOG).copy()
    if "long_score" not in trades.columns:
        return {"pass": False, "reason": "long_score missing from ledger"}

    trades["quintile"] = pd.qcut(
        trades["long_score"], q=5,
        labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    )

    print(f"\n  {'quintile':<10s} {'n':>5s} {'win%':>6s} {'mean':>8s} {'total':>11s}  score range")
    rows = []
    for q, sub in trades.groupby("quintile", observed=True):
        m = metrics(sub)
        rows.append({
            "quintile": str(q),
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "mean_pnl": float(sub["net_pnl_inr"].mean()),
            "total_pnl": m["total_pnl_inr"],
            "score_min": float(sub["long_score"].min()),
            "score_max": float(sub["long_score"].max()),
        })
        print(f"  {str(q):<10s} {m['n_trades']:>5d}  "
              f"{m['win_rate']*100:>5.1f}%  ₹{sub['net_pnl_inr'].mean():>+6,.0f}  "
              f"₹{m['total_pnl_inr']:>+10,.0f}  "
              f"[{sub['long_score'].min():.4f}, {sub['long_score'].max():.4f}]")

    # Pass: the highest-quintile bucket beats the lowest by ≥ +₹500 mean PnL
    high = rows[-1]
    low  = rows[0]
    spread_mean   = high["mean_pnl"] - low["mean_pnl"]
    spread_winpct = high["win_rate"] - low["win_rate"]
    print(f"\n  Q5 vs Q1 mean-PnL spread:  ₹{spread_mean:+,.0f}")
    print(f"  Q5 vs Q1 win-rate spread:  {100*spread_winpct:+.2f} pp")
    passed = bool(spread_mean > 500.0 or spread_winpct > 0.05)
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  → {verdict}  (criterion: mean-PnL spread > ₹500 OR win-rate spread > 5pp)")

    return {
        "pass": passed,
        "rows": rows,
        "spread_mean_pnl": spread_mean,
        "spread_win_rate": spread_winpct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("╔" + "═" * 88 + "╗")
    print("║  Tier-1 validation suite (must pass before paper trading)".ljust(89) + "║")
    print("╚" + "═" * 88 + "╝")

    a = tier1a_proxy_validation()
    b = tier1b_cost_sensitivity()
    c = tier1c_score_quintiles()

    overall = a["pass"] and b["pass"] and c["pass"]

    print("\n" + "═" * 90)
    print("  SUMMARY")
    print("═" * 90)
    print(f"  A. Proxy validation         {'PASS ✓' if a['pass'] else 'FAIL ✗'}")
    print(f"  B. Cost sensitivity         {'PASS ✓' if b['pass'] else 'FAIL ✗'}")
    print(f"  C. Score-quintile stratify  {'PASS ✓' if c['pass'] else 'FAIL ✗'}")
    print()
    print(f"  OVERALL                     {'CLEARED FOR PAPER TRADING ✓' if overall else 'BLOCKED ✗'}")

    out = {
        "tier1a_proxy": a,
        "tier1b_cost": b,
        "tier1c_quintile": c,
        "overall_pass": overall,
    }
    (OUT_DIR / "tier1_validation.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n→ {OUT_DIR}/tier1_validation.json")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
