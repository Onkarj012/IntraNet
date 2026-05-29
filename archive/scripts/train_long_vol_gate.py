#!/usr/bin/env python3
"""Sprint #2 — train a long-vol-specific binary gate.

Same architecture as gate_v5.py, but the LABEL is specific to long-vol
profitability. For each minute we ask:

    "Did LONG_STRADDLE_60M (strategy_id=11) come out profitable?"

The gate predicts that. This is the fair test of whether the same
microstructure feature set has signal for long-vol entries — independent
of the V5 gate, which was trained on short-vol-favorable minutes.

Trained on 2020-2022, validated 2023, tested 2024 (same splits as V5).
Saves to models/router_v0/long_vol/gate_dte{N}.lgb.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS_DIR = PROJECT_ROOT / "cache/optinet_v5/strategy_labels"
CHAIN_DIR  = PROJECT_ROOT / "cache/optinet_v4/chain_features"
FUT_DIR    = PROJECT_ROOT / "cache/optinet_v5/futures_features"
OUT_DIR    = PROJECT_ROOT / "models/router_v0/long_vol"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAIN_FEATURES = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
]
FUT_FEATURES = [
    "fut_basis", "fut_basis_change_30m",
    "fut_oi_change_1m", "fut_oi_change_5m", "fut_oi_change_30m",
    "fut_volume_oi_ratio", "fut_session_position",
    "fut_oi_x_long_buildup", "fut_oi_x_short_buildup",
    "fut_oi_x_short_cover", "fut_oi_x_long_unwind",
]
SIM_CONTEXT = ["minute_of_day", "hour_of_day", "dte"]


def train_one_bucket(*, dte_bucket: int, target_strategy_id: int = 11,
                      gate_margin: float = 0.0) -> dict:
    print(f"\n{'='*70}")
    print(f"  Long-vol gate: dte_bucket={dte_bucket}  "
          f"target_strategy={target_strategy_id}  margin={gate_margin}")
    print(f"{'='*70}")

    # Load labels filtered to this dte_bucket and the target long-vol strategy
    files = sorted(LABELS_DIR.rglob("data.parquet"))
    chunks = []
    for f in files:
        df = pd.read_parquet(f)
        sub = df[(df["dte_bucket"] == dte_bucket)
                  & (df["strategy_id"] == target_strategy_id)
                  & (df["valid"])]
        chunks.append(sub)
    labels = pd.concat(chunks, ignore_index=True)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    print(f"  rows for strategy {target_strategy_id} in dte={dte_bucket}: {len(labels):,}")

    # Label = 1 if pnl_per_premium > gate_margin
    labels["gate_label"] = (labels["pnl_per_premium"] > gate_margin).astype(int)
    pos = labels["gate_label"].mean()
    print(f"  positive rate: {pos:.4f} ({int(labels['gate_label'].sum()):,} of {len(labels):,})")

    # Load context features
    chain = pd.concat(
        [pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))],
        ignore_index=True)
    chain["datetime"] = pd.to_datetime(chain["datetime"])
    fut = pd.concat(
        [pd.read_parquet(f) for f in sorted(FUT_DIR.rglob("data.parquet"))],
        ignore_index=True)
    fut["datetime"] = pd.to_datetime(fut["datetime"])
    feats = (chain[["index", "datetime"] + CHAIN_FEATURES]
              .merge(fut[["index", "datetime"] + FUT_FEATURES],
                      on=["index", "datetime"], how="inner"))

    # Drop columns that exist in both labels and feats to avoid _x/_y suffixes
    overlap_cols = [c for c in CHAIN_FEATURES + FUT_FEATURES + ["atm_iv", "spot",
                     "atm_strike", "minute_of_day", "hour_of_day", "dte"]
                    if c in labels.columns]
    if overlap_cols:
        labels = labels.drop(columns=overlap_cols)

    df = labels.merge(feats, on=["index", "datetime"], how="inner")
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["hour_of_day"] = df["datetime"].dt.hour
    df["dte"] = dte_bucket
    df = df.dropna(subset=CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT).copy()
    print(f"  after dropna: {len(df):,}")

    # Splits
    train = df[df["trade_date"] < "2023-01-01"]
    val = df[(df["trade_date"] >= "2023-01-01") & (df["trade_date"] < "2024-01-01")]
    test = df[df["trade_date"] >= "2024-01-01"]
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        return {"error": "empty split"}

    # Train
    feat_cols = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "is_unbalance": True, "n_jobs": -1, "verbosity": -1,
    }
    print("  training …")
    booster = lgb.train(
        params, lgb.Dataset(train[feat_cols], label=train["gate_label"]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(val[feat_cols], label=val["gate_label"])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    print(f"  best_iter: {booster.best_iteration}")

    val_auc = roc_auc_score(val["gate_label"], booster.predict(val[feat_cols]))
    test_auc = roc_auc_score(test["gate_label"], booster.predict(test[feat_cols]))
    print(f"  VAL AUC: {val_auc:.4f}   TEST AUC: {test_auc:.4f}")

    out = OUT_DIR / f"gate_dte{dte_bucket}.lgb"
    booster.save_model(str(out))
    print(f"  saved → {out}")

    return {
        "dte_bucket": dte_bucket,
        "target_strategy": target_strategy_id,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "pos_rate": float(pos),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "best_iter": int(booster.best_iteration),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_strategy_id", type=int, default=11,
                    help="Long-vol strategy: 10=STRADDLE_30M, 11=STRADDLE_60M, 12=STRANGLE_60M")
    args = ap.parse_args()

    summary = {}
    for b in (2, 3):
        summary[f"dte{b}"] = train_one_bucket(
            dte_bucket=b, target_strategy_id=args.target_strategy_id)

    out = OUT_DIR / "long_vol_gate_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
