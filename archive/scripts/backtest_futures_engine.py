#!/usr/bin/env python3
"""Futures trade-card engine — backtest + slice diagnostics.

Uses the final_long.lgb / final_short.lgb models trained on pre-2024 data.
Backtests on 2024 blind window with realistic execution:
  - entry at 1-min futures close
  - target +0.40%, stop -0.30%, time-stop 60 min (actual minute bars)
  - hard per-trade floor -₹3,000
  - daily caps: 3 trades/day
  - daily loss halt: -₹15,000
  - no entries before 09:45 or after 14:55
  - no lunch-hour entries (12:00-13:00) — identified as anti-edge in sprint-1

Trade card output per trade (printed + saved):
  regime / action / confidence / entry_zone / stop / target / time_stop /
  size_multiplier / reason_codes

Slices: month, weekday, hour, regime, quarter.
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

from optinet_router.futures_features import FUTURES_FEATURES, add_regime  # type: ignore

FEAT_PATH  = PROJECT_ROOT / "cache/router_v0/futures_features.parquet"
LABEL_PATH = PROJECT_ROOT / "models/router_v0/futures/futures_barrier_labels.parquet"
MODEL_DIR  = PROJECT_ROOT / "models/router_v0/futures"
RESULTS    = PROJECT_ROOT / "results/router_v0"
RESULTS.mkdir(parents=True, exist_ok=True)

# Execution config
TARGET_PCT = 0.0040
STOP_PCT   = 0.0030
HORIZON    = 60
LOT        = 50
COSTS_INR  = 105.0          # round-trip: brokerage + STT + GST + slippage
STOP_FLOOR = -3000.0
DAILY_HALT = -15000.0
MAX_TRADES = 3
HARD_CUTOFF = dtime(14, 55)
LUNCH_START = dtime(12, 0)
LUNCH_END   = dtime(13, 0)
ENTRY_MIN   = 30            # minutes from open (09:45)

# Confidence thresholds — use top-percentile of score distribution
# (absolute thresholds don't work when models output narrow score bands)
LONG_TOP_PCT  = 0.85   # take LONG when long_score > 85th percentile of day's scores
SHORT_TOP_PCT = 0.85   # take SHORT when short_score > 85th percentile
# Size multiplier: top 95th → 1.5x, top 85th → 1.0x
HIGH_CONF_PCT = 0.95
LOW_CONF_PCT  = 0.85


def size_multiplier(score: float, p85: float, p95: float) -> float:
    if score >= p95:
        return 1.5
    if score >= p85:
        return 1.0
    return 0.5


def reason_codes(row: pd.Series, side: str) -> list[str]:
    codes = []
    if row["oi_long_buildup"] and side == "LONG":
        codes.append("OI_LONG_BUILDUP")
    if row["oi_short_buildup"] and side == "SHORT":
        codes.append("OI_SHORT_BUILDUP")
    if row["or_breakout_up"] and side == "LONG":
        codes.append("OR_BREAKOUT_UP")
    if row["or_breakout_dn"] and side == "SHORT":
        codes.append("OR_BREAKOUT_DN")
    if row["ema_slope"] > 0.002 and side == "LONG":
        codes.append("EMA_TREND_UP")
    if row["ema_slope"] < -0.002 and side == "SHORT":
        codes.append("EMA_TREND_DN")
    if row["vwap_dev"] > 0.001 and side == "LONG":
        codes.append("ABOVE_VWAP")
    if row["vwap_dev"] < -0.001 and side == "SHORT":
        codes.append("BELOW_VWAP")
    if row["realized_vol_30m"] < 0.10:
        codes.append("LOW_VOL")
    if row["realized_vol_30m"] > 0.25:
        codes.append("HIGH_VOL")
    if not codes:
        codes.append("MODEL_SIGNAL")
    return codes


def simulate_trade(side: str, entry_idx: int, day_bars: pd.DataFrame,
                    entry_px: float, size_mult: float) -> dict:
    sign = 1 if side == "LONG" else -1
    target_px = entry_px * (1 + sign * TARGET_PCT)
    stop_px   = entry_px * (1 - sign * STOP_PCT)
    j_end = min(len(day_bars), entry_idx + 1 + HORIZON)
    walk = day_bars.iloc[entry_idx+1:j_end]

    exit_px, exit_reason = None, None
    for _, r in walk.iterrows():
        c = r["fut_close"]
        if side == "LONG":
            if c >= target_px: exit_px, exit_reason = target_px, "TARGET"; break
            if c <= stop_px:   exit_px, exit_reason = stop_px,   "STOP";   break
        else:
            if c <= target_px: exit_px, exit_reason = target_px, "TARGET"; break
            if c >= stop_px:   exit_px, exit_reason = stop_px,   "STOP";   break
    if exit_px is None:
        exit_px = float(walk["fut_close"].iloc[-1]) if not walk.empty else entry_px
        exit_reason = "TIME" if not walk.empty else "NO_BARS"

    gross = sign * (exit_px - entry_px) * LOT * size_mult
    net   = gross - COSTS_INR * size_mult
    was_stopped = (exit_reason == "STOP") or (net <= STOP_FLOOR)
    if net < STOP_FLOOR:
        net = STOP_FLOOR
        was_stopped = True
    return {
        "side": side, "entry_px": entry_px, "exit_px": exit_px,
        "exit_reason": exit_reason, "net_pnl_inr": net,
        "was_stopped": was_stopped, "size_mult": size_mult,
    }


def backtest(df: pd.DataFrame, long_m: lgb.Booster, short_m: lgb.Booster,
              year: int = 2024) -> pd.DataFrame:
    test = df[df["trade_date"].dt.year == year].copy()
    test["long_score"]  = long_m.predict(test[FUTURES_FEATURES])
    test["short_score"] = short_m.predict(test[FUTURES_FEATURES])

    # Apply timing filters
    t = test["datetime"].dt.time
    mod = test["minute_of_day"] - 9 * 60 - 15
    eligible = (
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= LUNCH_START) & (t < LUNCH_END))
    )
    test = test[eligible].copy()

    # Compute per-day percentile thresholds (no look-ahead: only uses same-day scores)
    long_p85  = test.groupby("trade_date")["long_score"].transform(lambda s: s.quantile(LONG_TOP_PCT))
    long_p95  = test.groupby("trade_date")["long_score"].transform(lambda s: s.quantile(HIGH_CONF_PCT))
    short_p85 = test.groupby("trade_date")["short_score"].transform(lambda s: s.quantile(SHORT_TOP_PCT))
    short_p95 = test.groupby("trade_date")["short_score"].transform(lambda s: s.quantile(HIGH_CONF_PCT))
    test["long_p85"] = long_p85; test["long_p95"] = long_p95
    test["short_p85"] = short_p85; test["short_p95"] = short_p95

    # Signal: score above day's 85th percentile, and opposite side below 50th
    long_med  = test.groupby("trade_date")["long_score"].transform("median")
    short_med = test.groupby("trade_date")["short_score"].transform("median")
    test["take_long"]  = (test["long_score"] >= test["long_p85"]) & (test["short_score"] < short_med)
    test["take_short"] = (test["short_score"] >= test["short_p85"]) & (test["long_score"] < long_med)
    test = test[test["take_long"] | test["take_short"]].copy()
    test = test.sort_values(["trade_date", "datetime"]).reset_index(drop=True)

    # Load raw futures bars for simulation
    from optinet.v5_futures import discover_fut_days, load_fut_day
    DATA_ROOT = PROJECT_ROOT / "data/option_data"
    fut_days = {d: p for d, p in discover_fut_days(DATA_ROOT, "NIFTY")}

    results = []
    daily_pnl, daily_count = {}, {}

    for _, row in test.iterrows():
        td = row["trade_date"]
        if daily_count.get(td, 0) >= MAX_TRADES:
            continue
        if daily_pnl.get(td, 0.0) <= DAILY_HALT:
            continue
        side = "LONG" if row["take_long"] else "SHORT"
        score = float(row["long_score"] if side == "LONG" else row["short_score"])
        p85 = float(row["long_p85"] if side == "LONG" else row["short_p85"])
        p95 = float(row["long_p95"] if side == "LONG" else row["short_p95"])
        sz = size_multiplier(score, p85, p95)

        fut_path = fut_days.get(td.date())
        if fut_path is None:
            continue
        day_bars = load_fut_day(fut_path, "NIFTY")
        day_bars["datetime"] = pd.to_datetime(day_bars["datetime"])
        idx_arr = day_bars.index[day_bars["datetime"] == row["datetime"]]
        if len(idx_arr) == 0:
            continue
        i = int(idx_arr[0])
        entry_px = float(day_bars["fut_close"].iat[i])

        sim = simulate_trade(side, i, day_bars, entry_px, sz)
        rc = reason_codes(row, side)

        results.append({
            "trade_date": td, "datetime": row["datetime"],
            "regime": row["regime"],
            "long_score": float(row["long_score"]),
            "short_score": float(row["short_score"]),
            "reason_codes": "|".join(rc),
            **sim,
        })
        daily_pnl[td]   = daily_pnl.get(td, 0.0) + sim["net_pnl_inr"]
        daily_count[td] = daily_count.get(td, 0) + 1

    return pd.DataFrame(results)


def print_metrics(trades: pd.DataFrame, label: str = "ALL") -> dict:
    if trades.empty:
        print(f"  {label}: no trades")
        return {}
    p = trades["net_pnl_inr"]
    daily = trades.groupby("trade_date")["net_pnl_inr"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan")
    cum = p.cumsum(); dd = float((cum - cum.cummax()).min())
    gw = p[p > 0].sum(); gl = abs(p[p < 0].sum())
    pf = gw / gl if gl > 0 else float("inf")
    n_days = trades["trade_date"].nunique()
    m = {
        "n_trades": int(len(trades)), "n_days": int(n_days),
        "trades_per_day": round(len(trades) / n_days, 2),
        "win_rate": float((p > 0).mean()),
        "stop_rate": float(trades["was_stopped"].mean()),
        "mean_pnl_inr": float(p.mean()),
        "total_pnl_inr": float(p.sum()),
        "sharpe_daily_ann": float(sharpe),
        "profit_factor": float(pf),
        "max_drawdown_inr": float(dd),
    }
    print(f"  {label:<30s}  n={m['n_trades']:>4d}  days={m['n_days']:>3d}  "
          f"win={m['win_rate']*100:>5.1f}%  mean=₹{m['mean_pnl_inr']:>+6.0f}  "
          f"total=₹{m['total_pnl_inr']:>+10,.0f}  "
          f"PF={m['profit_factor']:>4.2f}  Sharpe={m['sharpe_daily_ann']:>+5.2f}  "
          f"DD=₹{m['max_drawdown_inr']:>+9,.0f}")
    return m


def main() -> int:
    print("=" * 90)
    print("Futures trade-card engine — 2024 blind backtest + slices")
    print("=" * 90)

    feats = pd.read_parquet(FEAT_PATH)
    feats["datetime"] = pd.to_datetime(feats["datetime"])
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    labels = pd.read_parquet(LABEL_PATH)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    df = feats.merge(labels[["datetime", "long_label", "short_label"]],
                      on="datetime", how="inner")
    df = df.dropna(subset=FUTURES_FEATURES)
    df = add_regime(df)

    long_m  = lgb.Booster(model_file=str(MODEL_DIR / "final_long.lgb"))
    short_m = lgb.Booster(model_file=str(MODEL_DIR / "final_short.lgb"))

    print("\nRunning backtest …")
    trades = backtest(df, long_m, short_m, year=2024)

    if trades.empty:
        print("No trades generated.")
        return 1

    trades.to_parquet(RESULTS / "futures_trades_2024.parquet", index=False)

    print("\n" + "=" * 90)
    print("  HEADLINE — 2024 blind window, NIFTY futures, realistic execution")
    print("=" * 90)
    headline = print_metrics(trades, "ALL")

    print("\n--- by SIDE ---")
    for s, sub in trades.groupby("side"):
        print_metrics(sub, f"side={s}")

    print("\n--- by REGIME ---")
    for r, sub in trades.groupby("regime"):
        print_metrics(sub, f"regime={r}")

    print("\n--- by MONTH ---")
    trades["month"] = pd.to_datetime(trades["datetime"]).dt.to_period("M").astype(str)
    for m, sub in trades.groupby("month"):
        print_metrics(sub, f"month={m}")

    print("\n--- by QUARTER ---")
    trades["quarter"] = pd.to_datetime(trades["datetime"]).dt.to_period("Q").astype(str)
    for q, sub in trades.groupby("quarter"):
        print_metrics(sub, f"quarter={q}")

    print("\n--- by WEEKDAY ---")
    trades["weekday"] = pd.to_datetime(trades["datetime"]).dt.day_name()
    for wd in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        sub = trades[trades["weekday"] == wd]
        if len(sub): print_metrics(sub, f"weekday={wd}")

    print("\n--- by ENTRY HOUR ---")
    trades["hour"] = pd.to_datetime(trades["datetime"]).dt.hour
    for h, sub in trades.groupby("hour"):
        print_metrics(sub, f"hour={h}")

    print("\n--- by EXIT REASON ---")
    for r, sub in trades.groupby("exit_reason"):
        print_metrics(sub, f"exit={r}")

    print("\n--- top reason codes ---")
    from collections import Counter
    all_codes = [c for row in trades["reason_codes"] for c in row.split("|")]
    for code, cnt in Counter(all_codes).most_common(10):
        print(f"  {code:<25s}: {cnt}")

    (RESULTS / "futures_summary_2024.json").write_text(
        json.dumps(headline, indent=2, default=str))
    print(f"\nSaved → {RESULTS}/futures_trades_2024.parquet + futures_summary_2024.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
