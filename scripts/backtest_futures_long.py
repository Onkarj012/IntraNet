#!/usr/bin/env python3
"""NIFTY long-only futures trade-card engine — backtest + walk-forward.

Hard filters (v1 rules):
  - LONG only (SHORT disabled)
  - Skip 11:00-11:59 (mid-morning consolidation, anti-edge)
  - Skip compression regime (not enough move to cover costs)
  - No entries before 09:45 (lag feature warmup)
  - No entries after 14:55

Trade card per entry:
  regime / action / confidence / entry_zone / stop / target / time_stop /
  size_multiplier / reason_codes

Walk-forward: for each quarter in 2021-2024, train on all prior data,
simulate trades on that quarter, aggregate results.

Reports:
  - Headline 2024 blind window
  - Quarter-by-quarter walk-forward
  - By weekday, hour, regime, month
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

from optinet_router.futures_features import FUTURES_FEATURES, add_regime
from optinet.v5_futures import discover_fut_days, load_fut_day

FEAT_PATH  = PROJECT_ROOT / "cache/router_v0/futures_features.parquet"
LABEL_PATH = PROJECT_ROOT / "models/router_v0/futures/futures_barrier_labels.parquet"
MODEL_DIR  = PROJECT_ROOT / "models/router_v0/futures"
RESULTS    = PROJECT_ROOT / "results/router_v0"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Execution config ──────────────────────────────────────────────────────────
TARGET_PCT  = 0.0040
STOP_PCT    = 0.0030
HORIZON     = 60
LOT         = 50
COSTS_INR   = 105.0       # round-trip: brokerage + STT + GST + slippage
STOP_FLOOR  = -3000.0
DAILY_HALT  = -15000.0
MAX_TRADES  = 3
HARD_CUTOFF = dtime(14, 55)
SKIP_START  = dtime(11, 0)
SKIP_END    = dtime(12, 0)
ENTRY_MIN   = 30          # minutes from open (09:45)

# ── Hard filters ──────────────────────────────────────────────────────────────
SKIP_REGIMES = {"compression"}   # never trade in these regimes
LONG_ONLY    = True              # SHORT disabled for v1

# ── Threshold: top-N% of day's long scores ────────────────────────────────────
SIGNAL_PCT   = 0.85   # take LONG when score > 85th pct of day's eligible scores
HIGH_CONF_PCT = 0.95  # size 1.5x when score > 95th pct

# ── LightGBM training params ─────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 1.0,
    "is_unbalance": True, "n_jobs": -1, "verbosity": -1,
}

WF_FOLDS = [
    ("2021-Q1", "2021-01-01", "2021-04-01"),
    ("2021-Q2", "2021-04-01", "2021-07-01"),
    ("2021-Q3", "2021-07-01", "2021-10-01"),
    ("2021-Q4", "2021-10-01", "2022-01-01"),
    ("2022-Q1", "2022-01-01", "2022-04-01"),
    ("2022-Q2", "2022-04-01", "2022-07-01"),
    ("2022-Q3", "2022-07-01", "2022-10-01"),
    ("2022-Q4", "2022-10-01", "2023-01-01"),
    ("2023-Q1", "2023-01-01", "2023-04-01"),
    ("2023-Q2", "2023-04-01", "2023-07-01"),
    ("2023-Q3", "2023-07-01", "2023-10-01"),
    ("2023-Q4", "2023-10-01", "2024-01-01"),
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    feats = pd.read_parquet(FEAT_PATH)
    feats["datetime"] = pd.to_datetime(feats["datetime"])
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    labels = pd.read_parquet(LABEL_PATH)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    df = feats.merge(labels[["datetime", "long_label"]], on="datetime", how="inner")
    df = df.dropna(subset=FUTURES_FEATURES + ["long_label"])
    df = add_regime(df)
    # Pre-filter: only eligible minutes (saves time in backtest loops)
    t = df["datetime"].dt.time
    mod = df["minute_of_day"] - 9 * 60 - 15
    df = df[
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~df["regime"].isin(SKIP_REGIMES)
    ].copy()
    return df


# ── Model training ────────────────────────────────────────────────────────────

def train_long_model(train_df: pd.DataFrame) -> lgb.Booster:
    split = int(len(train_df) * 0.8)
    X_tr = train_df[FUTURES_FEATURES].iloc[:split]
    y_tr = train_df["long_label"].iloc[:split]
    X_va = train_df[FUTURES_FEATURES].iloc[split:]
    y_va = train_df["long_label"].iloc[split:]
    return lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )


# ── Trade simulation ──────────────────────────────────────────────────────────

def _load_fut_bars_cache(data_root: Path) -> dict:
    """Pre-load all futures bars into a dict keyed by date."""
    cache = {}
    for d, p in discover_fut_days(data_root, "NIFTY"):
        try:
            bars = load_fut_day(p, "NIFTY")
            bars["datetime"] = pd.to_datetime(bars["datetime"])
            cache[d] = bars.reset_index(drop=True)
        except Exception:
            pass
    return cache


def simulate_trade(side: str, entry_idx: int, day_bars: pd.DataFrame,
                    entry_px: float, size_mult: float) -> dict:
    sign = 1
    target_px = entry_px * (1 + TARGET_PCT)
    stop_px   = entry_px * (1 - STOP_PCT)
    j_end = min(len(day_bars), entry_idx + 1 + HORIZON)
    walk = day_bars.iloc[entry_idx+1:j_end]

    exit_px, exit_reason = None, None
    for _, r in walk.iterrows():
        c = r["fut_close"]
        if c >= target_px: exit_px, exit_reason = target_px, "TARGET"; break
        if c <= stop_px:   exit_px, exit_reason = stop_px,   "STOP";   break
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
        "exit_px": exit_px, "exit_reason": exit_reason,
        "net_pnl_inr": net, "was_stopped": was_stopped, "size_mult": size_mult,
    }


def _reason_codes(row: pd.Series) -> str:
    codes = []
    if row.get("oi_long_buildup", 0): codes.append("OI_LONG_BUILDUP")
    if row.get("or_breakout_up", 0):  codes.append("OR_BREAKOUT_UP")
    if row.get("ema_slope", 0) > 0.002: codes.append("EMA_TREND_UP")
    if row.get("vwap_dev", 0) > 0.001:  codes.append("ABOVE_VWAP")
    if row.get("regime") == "expansion": codes.append("EXPANSION_REGIME")
    if row.get("regime") == "trend_up":  codes.append("TREND_UP_REGIME")
    if row.get("realized_vol_30m", 0) < 0.10: codes.append("LOW_VOL")
    return "|".join(codes) if codes else "MODEL_SIGNAL"


def run_backtest(df: pd.DataFrame, model: lgb.Booster,
                  fut_cache: dict) -> pd.DataFrame:
    """Run the LONG-only trade-card backtest on df using model."""
    df = df.copy()
    df["long_score"] = model.predict(df[FUTURES_FEATURES])

    # Per-day percentile thresholds (no look-ahead within a day)
    p85 = df.groupby("trade_date")["long_score"].transform(
        lambda s: s.quantile(SIGNAL_PCT))
    p95 = df.groupby("trade_date")["long_score"].transform(
        lambda s: s.quantile(HIGH_CONF_PCT))
    df["take_long"] = df["long_score"] >= p85
    df["size_mult"] = np.where(df["long_score"] >= p95, 1.5, 1.0)

    candidates = df[df["take_long"]].sort_values(
        ["trade_date", "datetime"]).reset_index(drop=True)

    results = []
    daily_pnl, daily_count = {}, {}

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
        sim = simulate_trade("LONG", i, day_bars, entry_px, float(row["size_mult"]))

        results.append({
            "trade_date": td,
            "datetime": row["datetime"],
            "regime": row["regime"],
            "long_score": float(row["long_score"]),
            "entry_px": entry_px,
            "reason_codes": _reason_codes(row),
            **sim,
        })
        daily_pnl[td]   = daily_pnl.get(td, 0.0) + sim["net_pnl_inr"]
        daily_count[td] = daily_count.get(td, 0) + 1

    return pd.DataFrame(results)


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    p = trades["net_pnl_inr"]
    daily = trades.groupby("trade_date")["net_pnl_inr"].sum()
    n_days = int(trades["trade_date"].nunique())
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
               if len(daily) > 1 and daily.std() > 0 else float("nan"))
    cum = p.cumsum(); dd = float((cum - cum.cummax()).min())
    gw = p[p > 0].sum(); gl = abs(p[p < 0].sum())
    pf = float(gw / gl) if gl > 0 else float("inf")
    return {
        "n_trades": int(len(trades)), "n_days": n_days,
        "trades_per_day": round(len(trades) / n_days, 2),
        "win_rate": float((p > 0).mean()),
        "stop_rate": float(trades["was_stopped"].mean()),
        "mean_pnl_inr": float(p.mean()),
        "total_pnl_inr": float(p.sum()),
        "best_trade_inr": float(p.max()),
        "worst_trade_inr": float(p.min()),
        "best_day_inr": float(daily.max()),
        "worst_day_inr": float(daily.min()),
        "sharpe_daily_ann": float(sharpe),
        "profit_factor": pf,
        "max_drawdown_inr": float(dd),
    }


def print_row(label: str, m: dict) -> None:
    if not m.get("n_trades"):
        print(f"  {label:<30s}  (no trades)")
        return
    print(f"  {label:<30s}  n={m['n_trades']:>4d}  days={m['n_days']:>3d}  "
          f"win={m['win_rate']*100:>5.1f}%  mean=₹{m['mean_pnl_inr']:>+6.0f}  "
          f"total=₹{m['total_pnl_inr']:>+10,.0f}  "
          f"PF={m['profit_factor']:>4.2f}  Sharpe={m['sharpe_daily_ann']:>+5.2f}  "
          f"DD=₹{m['max_drawdown_inr']:>+9,.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 90)
    print("NIFTY long-only futures trade-card engine")
    print(f"  Filters: LONG only | skip 11:00-11:59 | skip compression regime")
    print(f"  Target +{TARGET_PCT*100:.2f}%  Stop -{STOP_PCT*100:.2f}%  Horizon {HORIZON}min")
    print("=" * 90)

    df = load_dataset()
    print(f"\nDataset after hard filters: {len(df):,} rows, "
          f"{df['trade_date'].nunique()} days")
    print(f"Regime distribution:\n{df['regime'].value_counts().to_string()}")

    DATA_ROOT = PROJECT_ROOT / "data/option_data"
    print("\nPre-loading futures bars …")
    fut_cache = _load_fut_bars_cache(DATA_ROOT)
    print(f"  loaded {len(fut_cache)} days")

    # ── Walk-forward backtest ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  WALK-FORWARD BACKTEST (quarterly folds)")
    print("=" * 90)

    wf_trades_all = []
    wf_results = []

    for fold_name, test_start, test_end in WF_FOLDS:
        train = df[df["trade_date"] < test_start]
        test  = df[(df["trade_date"] >= test_start) & (df["trade_date"] < test_end)]
        if len(train) < 5000 or len(test) < 200:
            continue
        model = train_long_model(train)
        fold_trades = run_backtest(test, model, fut_cache)
        m = metrics(fold_trades)
        m["fold"] = fold_name
        wf_results.append(m)
        if not fold_trades.empty:
            fold_trades["fold"] = fold_name
            wf_trades_all.append(fold_trades)
        print_row(fold_name, m)

    wf_trades = pd.concat(wf_trades_all, ignore_index=True) if wf_trades_all else pd.DataFrame()

    print("\n--- Walk-forward aggregate ---")
    wf_agg = metrics(wf_trades)
    print_row("WF AGGREGATE", wf_agg)

    # ── 2024 blind window (final model) ──────────────────────────────────────
    print("\n" + "=" * 90)
    print("  2024 BLIND WINDOW (final model trained on pre-2024)")
    print("=" * 90)

    final_model = lgb.Booster(model_file=str(MODEL_DIR / "final_long.lgb"))
    test_2024 = df[df["trade_date"].dt.year == 2024]
    trades_2024 = run_backtest(test_2024, final_model, fut_cache)
    m2024 = metrics(trades_2024)
    print_row("2024 BLIND", m2024)

    if not trades_2024.empty:
        print("\n--- by REGIME ---")
        for r, sub in trades_2024.groupby("regime"):
            print_row(f"regime={r}", metrics(sub))

        print("\n--- by QUARTER ---")
        trades_2024["quarter"] = pd.to_datetime(
            trades_2024["datetime"]).dt.to_period("Q").astype(str)
        for q, sub in trades_2024.groupby("quarter"):
            print_row(f"quarter={q}", metrics(sub))

        print("\n--- by WEEKDAY ---")
        trades_2024["weekday"] = pd.to_datetime(trades_2024["datetime"]).dt.day_name()
        for wd in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
            sub = trades_2024[trades_2024["weekday"] == wd]
            if len(sub): print_row(f"weekday={wd}", metrics(sub))

        print("\n--- by ENTRY HOUR ---")
        trades_2024["hour"] = pd.to_datetime(trades_2024["datetime"]).dt.hour
        for h, sub in trades_2024.groupby("hour"):
            print_row(f"hour={h}", metrics(sub))

        print("\n--- by MONTH ---")
        trades_2024["month"] = pd.to_datetime(
            trades_2024["datetime"]).dt.to_period("M").astype(str)
        for m, sub in trades_2024.groupby("month"):
            print_row(f"month={m}", metrics(sub))

        print("\n--- by EXIT REASON ---")
        for r, sub in trades_2024.groupby("exit_reason"):
            print_row(f"exit={r}", metrics(sub))

    # ── Save ──────────────────────────────────────────────────────────────────
    if not wf_trades.empty:
        wf_trades.to_parquet(RESULTS / "futures_long_wf_trades.parquet", index=False)
    if not trades_2024.empty:
        trades_2024.to_parquet(RESULTS / "futures_long_2024_trades.parquet", index=False)

    summary = {
        "wf_aggregate": wf_agg,
        "blind_2024": m2024,
        "wf_folds": wf_results,
        "config": {
            "long_only": True,
            "skip_11_12": True,
            "skip_compression": True,
            "target_pct": TARGET_PCT,
            "stop_pct": STOP_PCT,
            "horizon_min": HORIZON,
            "signal_pct": SIGNAL_PCT,
        },
    }
    (RESULTS / "futures_long_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → {RESULTS}/futures_long_*.parquet + futures_long_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
