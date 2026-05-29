#!/usr/bin/env python3
"""V4-B: Forecast next-30-min realized volatility from chain features.

Target: realized vol of spot returns over the FOLLOWING 30 minutes
        (annualized, computed at minute t from minutes t+1..t+30).

Inputs: V4-A chain features at minute t (ATM IV, skew, PCR, OI, IV-RV spread, etc.)
        plus 5-min and 15-min lags of chain features and realized vol.

Splits: train 2020-2022, val 2023, test 2024 (chronological).

Baselines:
  - Persistence: rv_30m_forward ≈ rv_30m (trailing)
  - ATM IV    : rv_30m_forward ≈ atm_iv * sqrt(30/(252*375))  (no, atm_iv is already annualized)
                actually rv_30m_forward ≈ atm_iv (both annualized)

Reports MAE, RMSE, R², correlation, baseline comparison, feature importance,
and per-vol-bucket diagnostics (does the model improve in low/medium/high regimes?).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHAIN_DIR = PROJECT_ROOT / "cache/optinet_v4/chain_features"
OUT_DIR = PROJECT_ROOT / "results/optinet_v4"
MODEL_DIR = PROJECT_ROOT / "models/optinet_v4"

TRADING_DAYS = 252
TRADING_MINUTES = 375


def _annualize(per_minute_std: pd.Series) -> pd.Series:
    return per_minute_std * np.sqrt(TRADING_DAYS * TRADING_MINUTES)


def add_target_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    """For each (index, trade_date) compute forward 30-min RV and feature lags.

    All operations done as groupby().transform(...) so columns stay flat
    (no MultiIndex side effects from .apply()).
    """
    df = df.sort_values(["index", "datetime"]).reset_index(drop=True)
    df["trade_date"] = df["datetime"].dt.normalize()
    grp = df.groupby(["index", "trade_date"], sort=False)

    # 1-minute log return (within-day; first bar of each day = NaN)
    df["log_ret_1m"] = np.log(df["spot"] / grp["spot"].shift(1))

    # Forward 30-min realized vol = std of returns from t+1 to t+30, annualized.
    # Implementation: reverse the series within each day, take trailing rolling
    # std, reverse back. To do that cleanly with transform, define a helper.
    def _fwd_rv(s: pd.Series) -> pd.Series:
        rev = s.iloc[::-1]
        rev_std = rev.rolling(window=30, min_periods=20).std().iloc[::-1]
        return rev_std

    df["realized_vol_30m_forward"] = (
        df.groupby(["index", "trade_date"], sort=False)["log_ret_1m"]
          .transform(_fwd_rv) * np.sqrt(TRADING_DAYS * TRADING_MINUTES)
    )

    # Within-day lags
    lag_cols = ["atm_iv", "skew_slope", "pcr_oi", "pcr_vol",
                "iv_rv_spread", "realized_vol_30m", "max_oi_total_dist_pct"]
    for col in lag_cols:
        df[f"{col}_lag5"] = (
            df.groupby(["index", "trade_date"], sort=False)[col].shift(5)
        )
        df[f"{col}_lag15"] = (
            df.groupby(["index", "trade_date"], sort=False)[col].shift(15)
        )

    # Time-of-day features
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minutes_to_close"] = ((15 * 60 + 30) - df["minute_of_day"]).clip(lower=0)
    return df


def build_dataset() -> pd.DataFrame:
    files = sorted(CHAIN_DIR.rglob("data.parquet"))
    print(f"Loading {len(files)} partitions …")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  {len(df):,} rows loaded")
    df = add_target_and_lags(df)
    return df


FEATURE_COLS = [
    # Current-minute chain features
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol",
    "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years",
    "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
    # Time of day
    "minute_of_day", "minutes_to_close",
    # Lags
    "atm_iv_lag5", "atm_iv_lag15",
    "skew_slope_lag5", "skew_slope_lag15",
    "pcr_oi_lag5", "pcr_oi_lag15",
    "pcr_vol_lag5", "pcr_vol_lag15",
    "iv_rv_spread_lag5", "iv_rv_spread_lag15",
    "realized_vol_30m_lag5", "realized_vol_30m_lag15",
    "max_oi_total_dist_pct_lag5", "max_oi_total_dist_pct_lag15",
]
TARGET = "realized_vol_30m_forward"


def split_by_date(df: pd.DataFrame):
    train = df[df["trade_date"] < "2023-01-01"]
    val = df[(df["trade_date"] >= "2023-01-01") & (df["trade_date"] < "2024-01-01")]
    test = df[df["trade_date"] >= "2024-01-01"]
    return train, val, test


def report(label: str, y_true, y_pred) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = r2_score(yt, yp)
    corr = float(np.corrcoef(yt, yp)[0, 1])
    return {"label": label, "n": int(len(yt)),
            "mae": float(mae), "rmse": rmse, "r2": float(r2), "corr": corr}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building dataset ===")
    t0 = time.time()
    df = build_dataset()
    print(f"  Built in {time.time()-t0:.0f}s, {len(df):,} rows")

    # Drop rows without target / required features
    needed = FEATURE_COLS + [TARGET, "index", "trade_date"]
    df = df.dropna(subset=needed).copy()
    print(f"  After drop NA: {len(df):,} rows")

    train, val, test = split_by_date(df)
    print(f"  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    print(f"  train days: {train['trade_date'].min().date()} → {train['trade_date'].max().date()}")
    print(f"  val   days: {val['trade_date'].min().date()} → {val['trade_date'].max().date()}")
    print(f"  test  days: {test['trade_date'].min().date()} → {test['trade_date'].max().date()}")

    # ---- Baselines ----
    print("\n=== Baselines (test set) ===")
    baselines = {}
    bl_persistence = report("persistence",
                             test[TARGET].to_numpy(),
                             test["realized_vol_30m"].to_numpy())
    bl_atm_iv = report("ATM IV", test[TARGET].to_numpy(), test["atm_iv"].to_numpy())
    baselines["persistence"] = bl_persistence
    baselines["atm_iv"] = bl_atm_iv
    for b in [bl_persistence, bl_atm_iv]:
        print(f"  {b['label']:>14s}: MAE={b['mae']:.4f}  RMSE={b['rmse']:.4f}  R²={b['r2']:.4f}  corr={b['corr']:.4f}")

    # ---- Train LightGBM ----
    print("\n=== Training LightGBM ===")
    X_tr = train[FEATURE_COLS]
    y_tr = train[TARGET].to_numpy()
    X_va = val[FEATURE_COLS]
    y_va = val[TARGET].to_numpy()
    X_te = test[FEATURE_COLS]
    y_te = test[TARGET].to_numpy()

    params = {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "n_jobs": -1,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    t0 = time.time()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    print(f"  Trained in {time.time()-t0:.0f}s,  best iter={booster.best_iteration}")

    # ---- Eval on val + test ----
    pred_val = booster.predict(X_va, num_iteration=booster.best_iteration)
    pred_test = booster.predict(X_te, num_iteration=booster.best_iteration)
    rep_val = report("LGBM val", y_va, pred_val)
    rep_test = report("LGBM test", y_te, pred_test)
    print(f"\n  {rep_val['label']:>14s}: MAE={rep_val['mae']:.4f}  RMSE={rep_val['rmse']:.4f}  R²={rep_val['r2']:.4f}  corr={rep_val['corr']:.4f}")
    print(f"  {rep_test['label']:>14s}: MAE={rep_test['mae']:.4f}  RMSE={rep_test['rmse']:.4f}  R²={rep_test['r2']:.4f}  corr={rep_test['corr']:.4f}")

    # ---- Per-bucket diagnostics on test ----
    print("\n=== Per-vol-bucket diagnostics (test) ===")
    test_eval = test.copy()
    test_eval["pred"] = pred_test
    qs = test_eval[TARGET].quantile([0.0, 0.33, 0.67, 1.0]).to_numpy()
    test_eval["bucket"] = pd.cut(test_eval[TARGET], bins=qs,
                                   labels=["low", "mid", "high"], include_lowest=True)
    for bucket in ["low", "mid", "high"]:
        sub = test_eval[test_eval["bucket"] == bucket]
        if sub.empty:
            continue
        bucket_mae = mean_absolute_error(sub[TARGET], sub["pred"])
        persist_mae = mean_absolute_error(sub[TARGET], sub["realized_vol_30m"])
        print(f"  {bucket:>4s} ({len(sub):>6d} rows): "
              f"target mean={sub[TARGET].mean():.3f}  "
              f"LGBM MAE={bucket_mae:.4f}  persist MAE={persist_mae:.4f}  "
              f"lift={persist_mae - bucket_mae:+.4f}")

    # ---- Per-index ----
    print("\n=== Per-index test metrics ===")
    per_idx = {}
    for idx in sorted(test_eval["index"].unique()):
        sub = test_eval[test_eval["index"] == idx]
        rep = report(f"LGBM {idx}", sub[TARGET].to_numpy(), sub["pred"].to_numpy())
        per_idx[idx] = rep
        print(f"  {rep['label']}: MAE={rep['mae']:.4f}  R²={rep['r2']:.4f}  corr={rep['corr']:.4f}")

    # ---- Feature importance ----
    print("\n=== Top 15 features (gain) ===")
    fi = pd.DataFrame({"feature": FEATURE_COLS,
                       "gain": booster.feature_importance("gain"),
                       "split": booster.feature_importance("split")})
    fi = fi.sort_values("gain", ascending=False)
    print(fi.head(15).to_string(index=False))

    # ---- Save ----
    booster.save_model(str(MODEL_DIR / "rv_30m_forward.lgb"))
    summary = {
        "n_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "split_sizes": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "baselines_test": baselines,
        "lgbm_val": rep_val,
        "lgbm_test": rep_test,
        "per_index_test": per_idx,
        "feature_importance_top15": fi.head(15).to_dict(orient="records"),
        "best_iteration": int(booster.best_iteration),
    }
    (OUT_DIR / "v4b_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved model → {MODEL_DIR/'rv_30m_forward.lgb'}")
    print(f"Saved summary → {OUT_DIR/'v4b_summary.json'}")


if __name__ == "__main__":
    main()
