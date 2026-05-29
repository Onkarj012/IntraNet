#!/usr/bin/env python3
"""V4-D (corrected): Trade-quality meta-label with causal labelling.

The label is defined purely from REALIZED outcomes (future data), but the
MODEL is trained only on features available at time t. The key discipline:
features must not include fwd_rv, fwd_ret_30m, or any forward-looking quantity.

Label definition (from realized outcomes, used only for training targets):
  sell-options (2): short ATM straddle held 30 min was profitable after costs
                    i.e. theta_captured > cost_threshold
  buy-options  (1): long ATM straddle held 30 min was profitable after costs
  no-trade     (0): neither

Theta captured = entry_straddle_px - exit_straddle_px (BS re-priced at t+30)
Cost threshold = brokerage + slippage (₹40 + 0.5% × straddle × lot)

Features: ONLY trailing chain features (no forward-looking quantities).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHAIN_DIR   = PROJECT_ROOT / "cache/optinet_v4/chain_features"
REGIME_FILE = PROJECT_ROOT / "results/optinet_v4/v4c_test_regimes.parquet"
OUT_DIR     = PROJECT_ROOT / "results/optinet_v4"
MODEL_DIR   = PROJECT_ROOT / "models/optinet_v4"

LOT_SIZE = {"NIFTY": 50, "BANKNIFTY": 15}
BROKERAGE = 40.0
SLIPPAGE_PCT = 0.005
RISK_FREE = 0.065
TRADING_DAYS = 252
TRADING_MINUTES = 375

LABEL_NAMES = {0: "no-trade", 1: "buy-options", 2: "sell-options"}


def bs_straddle(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(call + put)


def assign_causal_label(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row using realized future data (for training targets only)."""
    df = df.sort_values(["index", "datetime"]).reset_index(drop=True)
    df["trade_date"] = df["datetime"].dt.normalize()
    grp = df.groupby(["index", "trade_date"], sort=False)

    # Forward 30-min spot return (future — used only for labelling)
    df["fwd_ret_30m"] = grp["spot"].transform(lambda s: s.shift(-30) / s - 1.0)

    # BS re-price straddle at t+30
    dt_30m = 30.0 / (TRADING_DAYS * TRADING_MINUTES)
    labels = np.zeros(len(df), dtype=np.int8)

    for i, row in df.iterrows():
        if pd.isna(row["fwd_ret_30m"]) or row["T_years"] <= dt_30m:
            continue
        lot = LOT_SIZE.get(row["index"], 50)
        S0, K = row["spot"], row["atm_strike"]
        T0, iv = row["T_years"], row["atm_iv"]
        entry_px = row["atm_straddle_premium"] * 2.0
        if entry_px <= 0:
            continue
        S1 = S0 * (1.0 + row["fwd_ret_30m"])
        T1 = max(T0 - dt_30m, 1e-6)
        exit_px = bs_straddle(S1, K, T1, RISK_FREE, iv)
        cost = BROKERAGE + SLIPPAGE_PCT * (entry_px + exit_px) * lot
        theta = (entry_px - exit_px) * lot
        gamma_loss = (exit_px - entry_px) * lot
        if theta > cost:
            labels[i] = 2   # sell-options profitable
        elif gamma_loss > cost:
            labels[i] = 1   # buy-options profitable

    df["trade_quality"] = labels
    return df


# Features available at time t (NO forward-looking quantities)
FEATURE_COLS = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
    "minute_of_day", "minutes_to_close",
    "atm_iv_lag5", "atm_iv_lag15",
    "skew_slope_lag5", "skew_slope_lag15",
    "pcr_oi_lag5", "pcr_oi_lag15",
    "realized_vol_30m_lag5", "realized_vol_30m_lag15",
    "iv_rv_spread_lag5", "iv_rv_spread_lag15",
]
REGIME_FEATURES = [f"regime_prob_{i}" for i in range(5)]
TARGET = "trade_quality"


def add_lags_and_time(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["index", "trade_date"], sort=False)
    for col in ["atm_iv", "skew_slope", "pcr_oi", "realized_vol_30m", "iv_rv_spread"]:
        df[f"{col}_lag5"]  = grp[col].shift(5)
        df[f"{col}_lag15"] = grp[col].shift(15)
    df["minute_of_day"]    = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minutes_to_close"] = ((15 * 60 + 30) - df["minute_of_day"]).clip(lower=0)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chain features …")
    df = pd.concat([pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))],
                   ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  {len(df):,} rows")

    print("Assigning causal labels (BS re-pricing, this takes ~5 min) …")
    # Vectorized version to avoid row-by-row loop
    df = df.sort_values(["index", "datetime"]).reset_index(drop=True)
    df["trade_date"] = df["datetime"].dt.normalize()
    grp = df.groupby(["index", "trade_date"], sort=False)
    df["fwd_ret_30m"] = grp["spot"].transform(lambda s: s.shift(-30) / s - 1.0)

    dt_30m = 30.0 / (TRADING_DAYS * TRADING_MINUTES)
    S0 = df["spot"].to_numpy()
    K  = df["atm_strike"].to_numpy()
    T0 = df["T_years"].to_numpy()
    iv = df["atm_iv"].to_numpy()
    entry_px = df["atm_straddle_premium"].to_numpy() * 2.0
    fwd = df["fwd_ret_30m"].to_numpy()
    idx_col = df["index"].to_numpy()

    S1 = S0 * (1.0 + np.where(np.isnan(fwd), 0.0, fwd))
    T1 = np.maximum(T0 - dt_30m, 1e-6)

    # Vectorized BS straddle at exit
    sqrtT1 = np.sqrt(T1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(S1, 1e-9) / np.maximum(K, 1e-9))
              + (RISK_FREE + 0.5 * iv**2) * T1) / (iv * sqrtT1)
    d2 = d1 - iv * sqrtT1
    call_exit = S1 * norm.cdf(d1) - K * np.exp(-RISK_FREE * T1) * norm.cdf(d2)
    put_exit  = K * np.exp(-RISK_FREE * T1) * norm.cdf(-d2) - S1 * norm.cdf(-d1)
    exit_px = call_exit + put_exit

    lots = np.array([LOT_SIZE.get(i, 50) for i in idx_col])
    cost = BROKERAGE + SLIPPAGE_PCT * (entry_px + exit_px) * lots
    theta = (entry_px - exit_px) * lots
    gamma = (exit_px - entry_px) * lots

    valid = (~np.isnan(fwd)) & (T0 > dt_30m) & (entry_px > 0) & (iv > 0)
    labels = np.zeros(len(df), dtype=np.int8)
    labels = np.where(valid & (theta > cost), 2, labels)
    labels = np.where(valid & (gamma > cost) & (labels == 0), 1, labels)
    df[TARGET] = labels

    df = add_lags_and_time(df)

    # Merge regime probabilities (test window only — avoids leakage)
    for i in range(5):
        df[f"regime_prob_{i}"] = 0.0
    if REGIME_FILE.exists():
        reg = pd.read_parquet(REGIME_FILE)[["index", "datetime"] + REGIME_FEATURES]
        reg["datetime"] = pd.to_datetime(reg["datetime"])
        df = df.merge(reg, on=["index", "datetime"], how="left", suffixes=("", "_reg"))
        for i in range(5):
            col, col_reg = f"regime_prob_{i}", f"regime_prob_{i}_reg"
            if col_reg in df.columns:
                df[col] = df[col_reg].fillna(df[col])
                df.drop(columns=[col_reg], inplace=True)

    ALL_FEATURES = FEATURE_COLS + REGIME_FEATURES
    df = df.dropna(subset=ALL_FEATURES + [TARGET]).copy()
    print(f"  After dropna: {len(df):,} rows")

    print("\nLabel distribution:")
    for k, name in LABEL_NAMES.items():
        n = (df[TARGET] == k).sum()
        print(f"  {k} {name:>14s}: {n:>7,}  ({n/len(df):.1%})")

    train = df[df["trade_date"] < "2023-01-01"]
    val   = df[(df["trade_date"] >= "2023-01-01") & (df["trade_date"] < "2024-01-01")]
    test  = df[df["trade_date"] >= "2024-01-01"]
    print(f"\ntrain {len(train):,}  val {len(val):,}  test {len(test):,}")

    X_tr, y_tr = train[ALL_FEATURES], train[TARGET].astype(int)
    X_va, y_va = val[ALL_FEATURES],   val[TARGET].astype(int)
    X_te, y_te = test[ALL_FEATURES],  test[TARGET].astype(int)

    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "class_weight": "balanced", "n_jobs": -1, "verbosity": -1,
    }
    print("\nTraining …")
    booster = lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
    )

    pred_proba_te = booster.predict(X_te)
    pred_te = pred_proba_te.argmax(axis=1)

    print("\n=== Test classification report ===")
    print(classification_report(y_te, pred_te,
                                 target_names=[LABEL_NAMES[i] for i in range(3)]))

    # Save predictions for V4-E
    test = test.copy()
    for i in range(3):
        test[f"tq_prob_{i}"] = pred_proba_te[:, i]
    test["tq_pred"] = pred_te
    test["tq_true"] = y_te.to_numpy()

    save_cols = (["index", "datetime", "trade_date", "spot", "atm_strike",
                  "atm_iv", "atm_straddle_premium", "T_years",
                  "realized_vol_30m", "fwd_ret_30m", TARGET, "tq_pred"]
                 + [f"tq_prob_{i}" for i in range(3)])
    test[save_cols].to_parquet(OUT_DIR / "v4d_test_predictions.parquet", index=False)

    fi = pd.DataFrame({"feature": ALL_FEATURES,
                        "gain": booster.feature_importance("gain")})
    fi = fi.sort_values("gain", ascending=False)
    print("\n=== Top 10 features ===")
    print(fi.head(10).to_string(index=False))

    booster.save_model(str(MODEL_DIR / "trade_quality.lgb"))
    summary = {
        "label_distribution": {LABEL_NAMES[k]: int((df[TARGET]==k).sum()) for k in range(3)},
        "test_accuracy": float((pred_te == y_te.to_numpy()).mean()),
        "feature_importance_top10": fi.head(10).to_dict(orient="records"),
        "best_iteration": int(booster.best_iteration),
    }
    (OUT_DIR / "v4d_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {MODEL_DIR/'trade_quality.lgb'}")
    print(f"Saved → {OUT_DIR/'v4d_test_predictions.parquet'}")


if __name__ == "__main__":
    main()
