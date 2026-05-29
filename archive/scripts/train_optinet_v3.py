#!/usr/bin/env python3
"""OptiNet v3 Phase 1: train and evaluate intraday spot direction model.

Trains a 4-LGBM stack on the intraday decision-point dataset and reports
per-decision-time AUC + simulated daily signal counts for both 1H and EOD horizons.

Usage:
    python scripts/train_optinet_v3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.intraday_v3 import build_intraday_dataset

DATA_OUT = PROJECT_ROOT / "cache/optinet_v3/intraday_dataset.parquet"
RESULTS_OUT = PROJECT_ROOT / "results/optinet_v3"
MODEL_OUT = PROJECT_ROOT / "models/optinet_v3"

NON_FEATURE = {
    "index", "trade_date", "decision_time",
    "decision_close", "session_open", "session_high", "session_low",
    "ret_1h", "ret_eod",
    "label_long_1h", "label_short_1h", "label_long_eod", "label_short_eod",
}


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot decision_time + index, drop non-feature columns."""
    df = df.copy()
    dt_dummies = pd.get_dummies(df["decision_time"], prefix="dt")
    # LightGBM rejects ':' in feature names — sanitize
    dt_dummies.columns = [c.replace(":", "") for c in dt_dummies.columns]
    df = pd.concat([df, dt_dummies], axis=1)
    df = pd.concat([df, pd.get_dummies(df["index"], prefix="idx")], axis=1)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    return df, feat_cols


def _train_one(X_tr, y_tr, X_te, y_te) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    if n_pos < 3 or n_neg < 3:
        return {"auc": np.nan, "n_pos_train": n_pos, "n_neg_train": n_neg, "model": None}

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,           # smaller leaves for tiny dataset
        min_data_in_leaf=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_tr, y_tr)

    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    return {"auc": auc, "n_pos_train": n_pos, "n_neg_train": n_neg, "model": model}


def main():
    print("=== v3 Phase 1: intraday spot direction model ===\n")
    if not DATA_OUT.exists():
        print(f"Building dataset → {DATA_OUT}")
        ds = build_intraday_dataset()
        DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(DATA_OUT, index=False)
    else:
        ds = pd.read_parquet(DATA_OUT)

    print(f"Dataset: {len(ds)} rows × {len(ds.columns)} cols  ({ds['trade_date'].nunique()} days)")

    df, feat_cols = _build_features(ds)
    print(f"Features used: {len(feat_cols)}\n")

    # Time-based split: train on first 70% of days, test on last 30%
    days = sorted(df["trade_date"].unique())
    split_idx = int(len(days) * 0.7)
    train_days = days[:split_idx]
    test_days = days[split_idx:]
    train = df[df["trade_date"].isin(train_days)].copy()
    test = df[df["trade_date"].isin(test_days)].copy()
    print(f"Train: {len(train)} rows ({len(train_days)} days)")
    print(f"Test : {len(test)} rows  ({len(test_days)} days)\n")

    X_tr = train[feat_cols].fillna(0.0)
    X_te = test[feat_cols].fillna(0.0)

    results = {}
    models = {}
    for label_col in ["label_long_1h", "label_short_1h", "label_long_eod", "label_short_eod"]:
        y_tr = train[label_col].astype(int).to_numpy()
        y_te = test[label_col].astype(int).to_numpy()
        res = _train_one(X_tr, y_tr, X_te, y_te)
        results[label_col] = {k: v for k, v in res.items() if k != "model"}
        models[label_col] = res["model"]
        print(f"  {label_col:25s}  train_pos={res['n_pos_train']:3d}  AUC={res['auc']:.3f}")

    print()
    # Per-decision-time AUC for the long_1h model on test set
    print("=== Per-decision-time AUC on test (label_long_1h) ===")
    if models["label_long_1h"] is not None:
        from sklearn.metrics import roc_auc_score
        test_with_score = test.copy()
        test_with_score["score"] = models["label_long_1h"].predict_proba(X_te)[:, 1]
        for dt in sorted(test_with_score["decision_time"].unique()):
            cell = test_with_score[test_with_score["decision_time"] == dt]
            y = cell["label_long_1h"].astype(int).to_numpy()
            if y.sum() in (0, len(y)):
                print(f"  {dt}: skip (no positive variance)  n={len(cell)}")
                continue
            auc = roc_auc_score(y, cell["score"])
            print(f"  {dt}: AUC={auc:.3f}  pos={int(y.sum())}/{len(cell)}")

    # Signal generation on test set: how many high-confidence picks per day?
    print("\n=== Daily signal count simulation on test set ===")
    if models["label_long_1h"] is not None and models["label_short_1h"] is not None:
        test_sig = test.copy()
        test_sig["long_p"] = models["label_long_1h"].predict_proba(X_te)[:, 1]
        test_sig["short_p"] = models["label_short_1h"].predict_proba(X_te)[:, 1]
        for thr in [0.30, 0.40, 0.50, 0.60]:
            picks = test_sig[(test_sig["long_p"] >= thr) | (test_sig["short_p"] >= thr)]
            picks_per_day = picks.groupby("trade_date").size()
            avg_per_day = picks_per_day.mean() if not picks_per_day.empty else 0
            total_picks = len(picks)
            # win rate: did the picked direction match the actual sign of ret_1h?
            picks = picks.copy()
            picks["picked_long"] = picks["long_p"] >= picks["short_p"]
            wins = ((picks["picked_long"] & (picks["ret_1h"] > 0)) |
                    (~picks["picked_long"] & (picks["ret_1h"] < 0))).sum()
            wr = wins / max(len(picks), 1)
            print(f"  threshold={thr:.2f}: {total_picks} picks  "
                  f"avg/day={avg_per_day:.1f}  win_rate={wr:.2%}")

    # Save artifacts
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(ds)),
        "days": int(ds["trade_date"].nunique()),
        "date_range": [str(ds["trade_date"].min().date()), str(ds["trade_date"].max().date())],
        "feature_count": len(feat_cols),
        "train_days": len(train_days),
        "test_days": len(test_days),
        "metrics": results,
    }
    (RESULTS_OUT / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to {RESULTS_OUT/'phase1_summary.json'}")

    # Save the long_1h and short_1h models for downstream use
    import pickle
    for label_col, model in models.items():
        if model is None:
            continue
        out = MODEL_OUT / f"{label_col}.pkl"
        with out.open("wb") as f:
            pickle.dump({"model": model, "feature_columns": feat_cols}, f)
    print(f"Models saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
