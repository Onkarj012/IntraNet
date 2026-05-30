#!/usr/bin/env python3
"""Phase-2: Retrain NIFTY futures LONG classifier with proper boosting (>1 tree).

Protocol (avoids invalidating the existing forward-walk validation):
  Train   : 2020-01-01 → 2024-09-30  (same data as final_long.lgb)
  Val     : 2024-Q4 + 2025-H1        (model selection / early stopping)
  Blind   : 2025-H2 + 2026-H1        (never touched during training)

The single-tree final_long.lgb had best_iter=4 (essentially a 2-feature rule).
This script uses a proper regularised LightGBM with early stopping on the
validation set to find the right depth without overfitting.

Outputs:
  models/router_v0/futures/final_long_v2.lgb
  results/router_v0/futures_v2_training_summary.json
  results/router_v0/futures_v2_blind_trades.parquet
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
from scripts.research.backtest_futures import (
    COSTS_INR, DAILY_HALT, ENTRY_MIN, HARD_CUTOFF, HIGH_CONF_PCT,
    HORIZON, INTRADAY_CUM_HALT, LOT, MAX_TRADES, SIGNAL_PCT,
    SKIP_END, SKIP_REGIMES, SKIP_START, STOP_FLOOR, STOP_PCT,
    TARGET_PCT, VIX_SPIKE_PCT, VIX_PATH,
    _load_fut_bars_cache, load_vix_prior, metrics, print_row,
    run_backtest, simulate_trade,
)

FEAT_PATH  = PROJECT_ROOT / "cache/router_v0/futures_features.parquet"
LABEL_PATH = PROJECT_ROOT / "models/router_v0/futures/futures_barrier_labels.parquet"
PROXY_PATH = PROJECT_ROOT / "cache/router_v0/futures_features_proxy.parquet"
MODEL_DIR  = PROJECT_ROOT / "models/router_v0/futures"
RESULTS    = PROJECT_ROOT / "results/router_v0"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Training windows ──────────────────────────────────────────────────────────
TRAIN_END  = "2024-10-01"   # exclusive — same as final_long.lgb
VAL_START  = "2024-10-01"   # 2024-Q4 + 2025-H1
VAL_END    = "2025-07-01"
BLIND_START = "2025-07-01"  # 2025-H2 + 2026-H1
BLIND_END   = "2026-07-01"

# ── LightGBM v2 params (proper boosting, regularised) ────────────────────────
LGBM_PARAMS_V2 = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,          # shallower than v1's 63 — less overfit
    "min_data_in_leaf": 300,   # stricter than v1's 200
    "feature_fraction": 0.75,
    "bagging_fraction": 0.80,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "is_unbalance": True,
    "n_jobs": -1,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING  = 50


def load_dataset() -> pd.DataFrame:
    feats = pd.read_parquet(FEAT_PATH)
    feats["datetime"] = pd.to_datetime(feats["datetime"])
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    labels = pd.read_parquet(LABEL_PATH)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    df = feats.merge(labels[["datetime", "long_label"]], on="datetime", how="inner")
    df = df.dropna(subset=FUTURES_FEATURES + ["long_label"])
    df = add_regime(df)
    t = df["datetime"].dt.time
    mod = df["minute_of_day"] - 9 * 60 - 15
    df = df[
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~df["regime"].isin(SKIP_REGIMES)
    ].copy()
    return df


def load_proxy() -> pd.DataFrame:
    """Load proxy features (Nov 2024 → present) for blind window."""
    if not PROXY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PROXY_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.dropna(subset=FUTURES_FEATURES)
    df = add_regime(df)
    t = df["datetime"].dt.time
    mod = df["minute_of_day"] - 9 * 60 - 15
    df = df[
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~df["regime"].isin(SKIP_REGIMES)
    ].copy()
    # Proxy has no long_label — add dummy for backtest compatibility
    if "long_label" not in df.columns:
        df["long_label"] = 0
    return df


def train_v2(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    X_tr = train_df[FUTURES_FEATURES]
    y_tr = train_df["long_label"]
    X_va = val_df[FUTURES_FEATURES]
    y_va = val_df["long_label"]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(100),
    ]
    model = lgb.train(
        LGBM_PARAMS_V2,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    # AUC on val
    from sklearn.metrics import roc_auc_score
    val_pred = model.predict(X_va)
    val_auc = float(roc_auc_score(y_va, val_pred))
    train_pred = model.predict(X_tr)
    train_auc = float(roc_auc_score(y_tr, train_pred))

    summary = {
        "best_iteration": int(model.best_iteration),
        "num_trees": int(model.num_trees()),
        "train_auc": round(train_auc, 4),
        "val_auc": round(val_auc, 4),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
    }
    return model, summary


def main() -> int:
    print("=" * 80)
    print("Phase-2: NIFTY futures v2 — proper boosting retrain")
    print(f"  Train: 2020-01-01 → {TRAIN_END}")
    print(f"  Val  : {VAL_START} → {VAL_END}")
    print(f"  Blind: {BLIND_START} → {BLIND_END}")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_dataset()
    print(f"\nFull dataset: {len(df):,} rows, {df['trade_date'].nunique()} days")

    train_df = df[df["trade_date"] < TRAIN_END].copy()
    val_df   = df[(df["trade_date"] >= VAL_START) & (df["trade_date"] < VAL_END)].copy()
    print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")

    if len(train_df) < 10_000 or len(val_df) < 1_000:
        print("  ERROR: insufficient data for training")
        return 1

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining v2 model …")
    model_v2, train_summary = train_v2(train_df, val_df)
    print(f"\n  best_iteration : {train_summary['best_iteration']}")
    print(f"  num_trees      : {train_summary['num_trees']}")
    print(f"  train AUC      : {train_summary['train_auc']:.4f}")
    print(f"  val   AUC      : {train_summary['val_auc']:.4f}")

    # Save model
    model_path = MODEL_DIR / "final_long_v2.lgb"
    model_v2.save_model(str(model_path))
    print(f"\n  Saved → {model_path}")

    # ── Backtest on val window (sanity check) ─────────────────────────────────
    print("\n" + "=" * 80)
    print("  VAL WINDOW BACKTEST (2024-Q4 + 2025-H1)")
    print("=" * 80)
    DATA_ROOT = PROJECT_ROOT / "data/option_data"
    fut_cache = _load_fut_bars_cache(DATA_ROOT)
    vix_prior = load_vix_prior(VIX_PATH)

    val_trades = run_backtest(val_df, model_v2, fut_cache, vix_prior)
    m_val = metrics(val_trades)
    print_row("VAL (v2)", m_val)

    # Compare v1 on same window
    model_v1 = lgb.Booster(model_file=str(MODEL_DIR / "final_long.lgb"))
    val_trades_v1 = run_backtest(val_df, model_v1, fut_cache, vix_prior)
    m_val_v1 = metrics(val_trades_v1)
    print_row("VAL (v1)", m_val_v1)

    # ── Blind window backtest ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BLIND WINDOW BACKTEST (2025-H2 + 2026-H1)")
    print("=" * 80)

    # Combine actual features (if available) + proxy
    blind_actual = df[(df["trade_date"] >= BLIND_START) & (df["trade_date"] < BLIND_END)].copy()
    proxy_df = load_proxy()
    if not proxy_df.empty:
        proxy_blind = proxy_df[
            (proxy_df["trade_date"] >= BLIND_START) &
            (proxy_df["trade_date"] < BLIND_END)
        ].copy()
        # Prefer actual over proxy for overlapping dates
        actual_dates = set(blind_actual["trade_date"].dt.date.unique()) if not blind_actual.empty else set()
        proxy_blind = proxy_blind[~proxy_blind["trade_date"].dt.date.isin(actual_dates)]
        blind_df = pd.concat([blind_actual, proxy_blind], ignore_index=True).sort_values(
            ["trade_date", "datetime"]).reset_index(drop=True)
    else:
        blind_df = blind_actual

    if blind_df.empty:
        print("  No blind data available yet — run after 2025-H2 data is loaded")
    else:
        print(f"  Blind rows: {len(blind_df):,}  days: {blind_df['trade_date'].nunique()}")
        blind_trades_v2 = run_backtest(blind_df, model_v2, fut_cache, vix_prior)
        blind_trades_v1 = run_backtest(blind_df, model_v1, fut_cache, vix_prior)
        m_blind_v2 = metrics(blind_trades_v2)
        m_blind_v1 = metrics(blind_trades_v1)
        print_row("BLIND v2", m_blind_v2)
        print_row("BLIND v1", m_blind_v1)

        if not blind_trades_v2.empty:
            blind_trades_v2.to_parquet(RESULTS / "futures_v2_blind_trades.parquet", index=False)

        train_summary["blind_v2"] = m_blind_v2
        train_summary["blind_v1"] = m_blind_v1

    train_summary["val_v2"] = m_val
    train_summary["val_v1"] = m_val_v1

    # ── Feature importance ────────────────────────────────────────────────────
    print("\n  Top-10 features (v2 gain):")
    imp = model_v2.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FUTURES_FEATURES, imp), key=lambda x: -x[1])
    for name, score in feat_imp[:10]:
        print(f"    {name:<30s}  {score:>10.1f}")

    # ── Save summary ──────────────────────────────────────────────────────────
    out_path = RESULTS / "futures_v2_training_summary.json"
    out_path.write_text(json.dumps(train_summary, indent=2, default=str))
    print(f"\n  Saved → {out_path}")

    # ── Decision ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  DECISION GATE")
    print("=" * 80)
    val_sharpe_v2 = m_val.get("sharpe_daily_ann", 0.0)
    val_sharpe_v1 = m_val_v1.get("sharpe_daily_ann", 0.0)
    if val_sharpe_v2 > val_sharpe_v1 and val_sharpe_v2 > 1.0:
        print(f"  ✅ PROMOTE v2: val Sharpe {val_sharpe_v2:.2f} > v1 {val_sharpe_v1:.2f}")
        print(f"     Use final_long_v2.lgb for paper trading")
    else:
        print(f"  ⚠️  KEEP v1: v2 val Sharpe {val_sharpe_v2:.2f} did not beat v1 {val_sharpe_v1:.2f}")
        print(f"     final_long.lgb remains the production model")

    return 0


if __name__ == "__main__":
    sys.exit(main())
