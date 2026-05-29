#!/usr/bin/env python3
"""Futures engine — train regime + direction models with walk-forward validation.

Walk-forward folds (quarterly, 2021-2024):
  Train on all data before fold start, validate on fold quarter.
  Reports per-fold AUC and aggregated metrics.

Regime labels (unsupervised, rule-based):
  trend_up   : ema9 > ema21 AND realized_vol_30m < 75th pct AND ret_30m > 0
  trend_dn   : ema9 < ema21 AND realized_vol_30m < 75th pct AND ret_30m < 0
  expansion  : realized_vol_30m > 90th pct
  compression: realized_vol_30m < 25th pct AND |ret_30m| < 0.001
  range      : everything else

Direction labels (barrier, same as sprint-1):
  long_label  = 1 if +0.40% hit before -0.30% within 60 min
  short_label = 1 if -0.40% hit before +0.30% within 60 min

Saves:
  models/router_v0/futures/regime_stats.json
  models/router_v0/futures/wf_long_fold{N}.lgb  (one per fold)
  models/router_v0/futures/wf_short_fold{N}.lgb
  models/router_v0/futures/wf_summary.json
  models/router_v0/futures/final_long.lgb   (trained on all data < 2024)
  models/router_v0/futures/final_short.lgb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEAT_PATH  = PROJECT_ROOT / "cache/router_v0/futures_features.parquet"
LABEL_PATH = PROJECT_ROOT / "models/router_v0/futures/futures_barrier_labels.parquet"
OUT_DIR    = PROJECT_ROOT / "models/router_v0/futures"
RESULTS    = PROJECT_ROOT / "results/router_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET_PCT = 0.0040
STOP_PCT   = 0.0030
HORIZON    = 60
ENTRY_MIN  = 30   # minutes from open (09:45)

from optinet_router.futures_features import FUTURES_FEATURES, add_regime
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 1.0,
    "is_unbalance": True, "n_jobs": -1, "verbosity": -1,
}

# Walk-forward folds: (fold_name, test_start, test_end)
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("Loading features …")
    feats = pd.read_parquet(FEAT_PATH)
    feats["datetime"] = pd.to_datetime(feats["datetime"])
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])

    print("Loading barrier labels …")
    labels = pd.read_parquet(LABEL_PATH)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])

    df = feats.merge(
        labels[["datetime", "long_label", "short_label"]],
        on="datetime", how="inner"
    )
    df = df.dropna(subset=FUTURES_FEATURES + ["long_label", "short_label"])
    # Restrict to post-open-range minutes
    df["mod"] = df["minute_of_day"] - 9 * 60 - 15
    df = df[df["mod"] >= ENTRY_MIN].copy()
    df = add_regime(df)
    print(f"Dataset: {len(df):,} rows, {df['trade_date'].nunique()} days")
    print(f"Regime distribution:\n{df['regime'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def train_fold(train_df: pd.DataFrame, label_col: str) -> lgb.Booster:
    X = train_df[FUTURES_FEATURES]
    y = train_df[label_col]
    # Use last 20% of train as internal val for early stopping
    split = int(len(train_df) * 0.8)
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]
    booster = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return booster


def walk_forward(df: pd.DataFrame) -> dict:
    results = []
    for fold_name, test_start, test_end in WF_FOLDS:
        train = df[df["trade_date"] < test_start]
        test = df[(df["trade_date"] >= test_start) & (df["trade_date"] < test_end)]
        if len(train) < 5000 or len(test) < 500:
            print(f"  {fold_name}: skip (train={len(train)}, test={len(test)})")
            continue

        fold_res = {"fold": fold_name, "n_train": len(train), "n_test": len(test)}
        for side in ("long", "short"):
            label = f"{side}_label"
            booster = train_fold(train, label)
            preds = booster.predict(test[FUTURES_FEATURES])
            auc = roc_auc_score(test[label], preds)
            fold_res[f"{side}_auc"] = float(auc)
            fold_res[f"{side}_pos_rate"] = float(test[label].mean())
            fold_res[f"{side}_best_iter"] = int(booster.best_iteration)
            booster.save_model(str(OUT_DIR / f"wf_{side}_{fold_name.replace('-','_')}.lgb"))

        results.append(fold_res)
        print(f"  {fold_name}: long_auc={fold_res['long_auc']:.4f}  "
              f"short_auc={fold_res['short_auc']:.4f}  "
              f"n_test={len(test):,}")

    return {"folds": results}


# ---------------------------------------------------------------------------
# Final models (train on all data before 2024)
# ---------------------------------------------------------------------------

def train_final(df: pd.DataFrame) -> tuple[lgb.Booster, lgb.Booster]:
    train = df[df["trade_date"] < "2024-01-01"]
    print(f"\nFinal models: training on {len(train):,} rows (pre-2024) …")
    long_m = train_fold(train, "long_label")
    short_m = train_fold(train, "short_label")
    long_m.save_model(str(OUT_DIR / "final_long.lgb"))
    short_m.save_model(str(OUT_DIR / "final_short.lgb"))
    print(f"  long best_iter={long_m.best_iteration}  "
          f"short best_iter={short_m.best_iteration}")
    return long_m, short_m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("Futures engine — walk-forward + regime + final models")
    print("=" * 72)

    df = load_data()

    # Regime stats
    regime_stats = df.groupby("regime").agg(
        n=("long_label", "count"),
        long_pos=("long_label", "mean"),
        short_pos=("short_label", "mean"),
    ).round(4).to_dict()
    (OUT_DIR / "regime_stats.json").write_text(
        json.dumps(regime_stats, indent=2, default=str))
    print(f"\nRegime stats saved.")

    print("\n--- Walk-forward validation ---")
    wf = walk_forward(df)

    # Summary
    folds = wf["folds"]
    long_aucs = [f["long_auc"] for f in folds]
    short_aucs = [f["short_auc"] for f in folds]
    wf["summary"] = {
        "n_folds": len(folds),
        "long_auc_mean": float(np.mean(long_aucs)),
        "long_auc_std": float(np.std(long_aucs)),
        "long_auc_min": float(np.min(long_aucs)),
        "short_auc_mean": float(np.mean(short_aucs)),
        "short_auc_std": float(np.std(short_aucs)),
        "short_auc_min": float(np.min(short_aucs)),
        "folds_long_above_55": int(sum(a > 0.55 for a in long_aucs)),
        "folds_short_above_55": int(sum(a > 0.55 for a in short_aucs)),
    }
    print(f"\nWalk-forward summary:")
    for k, v in wf["summary"].items():
        print(f"  {k:<30s}: {v}")

    (OUT_DIR / "wf_summary.json").write_text(json.dumps(wf, indent=2, default=str))

    # Final models
    long_m, short_m = train_final(df)

    # Feature importance
    imp = pd.DataFrame({
        "feature": long_m.feature_name(),
        "gain_long": long_m.feature_importance("gain"),
        "gain_short": short_m.feature_importance("gain"),
    }).sort_values("gain_long", ascending=False).head(12)
    print("\nTop features (long side):")
    print(imp[["feature", "gain_long", "gain_short"]].to_string(index=False))

    print(f"\nAll outputs → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
