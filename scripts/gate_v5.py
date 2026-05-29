#!/usr/bin/env python3
"""V5 Phase 2 — Binary trade-vs-no-trade gate.

Two-stage architecture:
  Stage 1 (gate): predict "will the best active strategy at this minute beat
                  NO_TRADE by at least gate_margin (default 2%)?"
                  Binary classifier; uses ONLY context features (no strategy
                  descriptors).
  Stage 2 (ranker): existing dte-bucket LambdaRank model picks the strategy.
                    Only invoked when gate predicts trade=1.

Compare gated vs ungated:
  - Total trades, win rate, mean ppp, mean ₹ PnL on 2024 blind window
  - Lift over: random pick, baseline SS_EOD, ungated ranker
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS_DIR = PROJECT_ROOT / "cache/optinet_v5/strategy_labels"
CHAIN_DIR  = PROJECT_ROOT / "cache/optinet_v4/chain_features"
FUT_DIR    = PROJECT_ROOT / "cache/optinet_v5/futures_features"
OUT_DIR    = PROJECT_ROOT / "results/optinet_v5"
MODEL_DIR  = PROJECT_ROOT / "models/optinet_v5"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dte_bucket", type=int, default=3)
    parser.add_argument("--gate_margin", type=float, default=0.02,
                        help="Minimum ppp the best active strategy must beat NO_TRADE by, "
                             "for the gate label to be 1")
    args = parser.parse_args()
    DTE_BUCKET = args.dte_bucket
    GATE_MARGIN = args.gate_margin

    print(f"=== V5 binary gate — dte_bucket={DTE_BUCKET}  gate_margin={GATE_MARGIN} ===")

    # ─── Load strategy labels and aggregate to one row per (index, datetime) ─
    files = sorted(LABELS_DIR.rglob("data.parquet"))
    chunks = [pd.read_parquet(f).query(f"dte_bucket == {DTE_BUCKET}") for f in files]
    labels = pd.concat(chunks, ignore_index=True)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    print(f"Labels: {len(labels):,} rows ({len(labels)//13} decision-points)")

    # For each (index, datetime) compute:
    #   - best_active_ppp: max ppp among non-NO_TRADE strategies
    #   - best_active_id : argmax
    valid = labels[labels["valid"]]
    active = valid[valid["strategy_id"] != 0]

    grp = active.groupby(["index", "datetime"], sort=False)
    best_active = grp.agg(
        best_active_ppp=("pnl_per_premium", "max"),
        best_active_id=("pnl_per_premium",
                        lambda s: int(active.loc[s.idxmax(), "strategy_id"])
                        if not s.empty else -1),
        best_active_pnl=("net_pnl",
                         lambda s: float(active.loc[s.idxmax(), "net_pnl"])
                         if not s.empty else 0.0),
    ).reset_index()
    print(f"Best-active per minute: {len(best_active):,} groups")

    # Gate label: 1 if best_active_ppp > GATE_MARGIN, else 0
    best_active["gate_label"] = (best_active["best_active_ppp"] > GATE_MARGIN).astype(int)
    best_active["trade_date"] = best_active["datetime"].dt.normalize()
    pos_rate = best_active["gate_label"].mean()
    print(f"Gate positive rate: {pos_rate:.4f}  ({best_active['gate_label'].sum():,} of {len(best_active):,})")

    # ─── Load context features (chain + futures) ──────────────────────────────
    chain = pd.concat([pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))],
                       ignore_index=True)
    chain["datetime"] = pd.to_datetime(chain["datetime"])
    chain = chain[["index", "datetime"] + CHAIN_FEATURES]

    fut = pd.concat([pd.read_parquet(f) for f in sorted(FUT_DIR.rglob("data.parquet"))],
                     ignore_index=True)
    fut["datetime"] = pd.to_datetime(fut["datetime"])
    fut = fut[["index", "datetime"] + FUT_FEATURES]

    feats = chain.merge(fut, on=["index", "datetime"], how="inner")
    print(f"Context features: {len(feats):,} rows")

    df = best_active.merge(feats, on=["index", "datetime"], how="inner")

    # Add the simulator context columns
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["hour_of_day"]   = df["datetime"].dt.hour
    df["dte"] = DTE_BUCKET  # constant within bucket

    df = df.dropna(subset=CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT).copy()
    print(f"After dropna: {len(df):,}")

    # ─── Splits ──────────────────────────────────────────────────────────────
    train = df[df["trade_date"] < "2023-01-01"]
    val   = df[(df["trade_date"] >= "2023-01-01") & (df["trade_date"] < "2024-01-01")]
    test  = df[df["trade_date"] >= "2024-01-01"]
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"train pos_rate={train['gate_label'].mean():.4f}  "
          f"val pos_rate={val['gate_label'].mean():.4f}  "
          f"test pos_rate={test['gate_label'].mean():.4f}")

    # ─── Train binary gate ───────────────────────────────────────────────────
    feat_cols = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT
    X_tr = train[feat_cols]; y_tr = train["gate_label"]
    X_va = val[feat_cols];   y_va = val["gate_label"]
    X_te = test[feat_cols];  y_te = test["gate_label"]

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "is_unbalance": True,
        "n_jobs": -1, "verbosity": -1,
    }
    print("\nTraining gate …")
    booster = lgb.train(
        params, lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
    )
    print(f"best_iter={booster.best_iteration}")

    # ─── Evaluate gate ────────────────────────────────────────────────────────
    p_va = booster.predict(X_va)
    p_te = booster.predict(X_te)
    val_auc  = roc_auc_score(y_va, p_va)
    test_auc = roc_auc_score(y_te, p_te)
    print(f"\n  VAL AUC: {val_auc:.4f}    TEST AUC: {test_auc:.4f}")

    # Threshold sweep on test: precision/recall/PnL impact
    print("\n=== Gate threshold sweep on TEST ===")
    test = test.copy()
    test["gate_score"] = p_te

    sweep_rows = []
    n_test_days = test["trade_date"].nunique()
    baseline_ss_eod_ppp = -0.0353  # from dte=3 desc smoke test
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        sel = test[test["gate_score"] >= thr]
        n_sel = len(sel)
        if n_sel == 0:
            sweep_rows.append({"threshold": thr, "n": 0})
            continue
        precision = sel["gate_label"].mean()
        recall = sel["gate_label"].sum() / max(test["gate_label"].sum(), 1)
        # Approx gated ranker PnL: assume we take best_active_ppp when gate=1
        # (this is the oracle upper bound for the gated system)
        oracle_mean_ppp = sel["best_active_ppp"].mean()
        oracle_mean_pnl = sel["best_active_pnl"].mean()
        oracle_hit_rate = (sel["best_active_ppp"] > 0).mean()
        sweep_rows.append({
            "threshold": thr,
            "n": int(n_sel),
            "trades_per_day": round(n_sel / n_test_days, 2),
            "gate_precision": round(float(precision), 4),
            "gate_recall": round(float(recall), 4),
            "oracle_mean_ppp": round(float(oracle_mean_ppp), 4),
            "oracle_mean_pnl_inr": round(float(oracle_mean_pnl), 0),
            "oracle_hit_rate": round(float(oracle_hit_rate), 4),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False))

    print(f"\nBaseline SS_EOD ppp (test): {baseline_ss_eod_ppp:.4f}")
    print(f"Best-active oracle ppp on ALL test rows: {test['best_active_ppp'].mean():.4f}")
    print(f"Gate-1 baseline (no model): {test['gate_label'].mean():.4f} pos rate")

    # ─── Feature importance ───────────────────────────────────────────────────
    fi = pd.DataFrame({"feature": feat_cols,
                        "gain": booster.feature_importance("gain")})
    fi = fi.sort_values("gain", ascending=False).head(15)
    print(f"\nTop 15 features:")
    print(fi.to_string(index=False))

    # Save
    booster.save_model(str(MODEL_DIR / f"gate_dte{DTE_BUCKET}.lgb"))
    summary = {
        "dte_bucket": DTE_BUCKET, "gate_margin": GATE_MARGIN,
        "val_auc": float(val_auc), "test_auc": float(test_auc),
        "best_iteration": int(booster.best_iteration),
        "test_threshold_sweep": sweep_df.to_dict(orient="records"),
        "feature_importance_top15": fi.to_dict(orient="records"),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "train_pos_rate": float(train["gate_label"].mean()),
        "val_pos_rate": float(val["gate_label"].mean()),
        "test_pos_rate": float(test["gate_label"].mean()),
    }
    (OUT_DIR / f"phase2_gate_dte{DTE_BUCKET}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → {MODEL_DIR/f'gate_dte{DTE_BUCKET}.lgb'}")


if __name__ == "__main__":
    main()
