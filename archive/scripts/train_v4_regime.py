#!/usr/bin/env python3
"""V4-C: 5-class intraday regime classifier.

Regimes (labelled from realized data, not predicted):
  0 = range        : low RV, IV-RV spread high, spot oscillating
  1 = trending-up  : positive 30-min return, low vol
  2 = trending-down: negative 30-min return, low vol
  3 = vol-expansion: RV rising fast, large spot move
  4 = vol-crush    : RV falling fast, IV-RV spread compressing

Labels are assigned per-minute using the NEXT 30-min window (forward-looking),
so the classifier learns to predict regime from current chain features.

Walk-forward: train on 2020-2022, val 2023, test 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHAIN_DIR = PROJECT_ROOT / "cache/optinet_v4/chain_features"
OUT_DIR   = PROJECT_ROOT / "results/optinet_v4"
MODEL_DIR = PROJECT_ROOT / "models/optinet_v4"

TRADING_DAYS = 252
TRADING_MINUTES = 375
REGIME_NAMES = {0: "range", 1: "trend-up", 2: "trend-down", 3: "vol-expansion", 4: "vol-crush"}


# ── label assignment ──────────────────────────────────────────────────────────

def assign_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Assign forward-looking 30-min regime label per row."""
    df = df.sort_values(["index", "datetime"]).reset_index(drop=True)
    df["trade_date"] = df["datetime"].dt.normalize()

    grp = df.groupby(["index", "trade_date"], sort=False)

    # Forward 30-min spot return
    df["fwd_ret_30m"] = grp["spot"].transform(
        lambda s: (s.shift(-30) / s - 1.0)
    )

    # Forward 30-min realized vol (annualized)
    log_ret = np.log(df["spot"] / grp["spot"].shift(1))
    df["log_ret_1m"] = log_ret

    def _fwd_rv(s):
        rev = s.iloc[::-1]
        return (rev.rolling(30, min_periods=20).std().iloc[::-1]
                * np.sqrt(TRADING_DAYS * TRADING_MINUTES))

    df["fwd_rv"] = df.groupby(["index", "trade_date"], sort=False)["log_ret_1m"].transform(_fwd_rv)

    # Trailing 30-min RV (already in chain features as realized_vol_30m)
    trail_rv = df["realized_vol_30m"]

    # RV change rate
    df["rv_change"] = df["fwd_rv"] - trail_rv

    # Thresholds (percentile-based, computed on train window 2020-2022)
    # Hard-coded from empirical quantiles to avoid leakage
    RV_HIGH   = 0.18   # ~75th pct of annualized RV
    RV_LOW    = 0.07   # ~25th pct
    RET_TREND = 0.003  # 0.3% 30-min move = trending

    fwd_rv   = df["fwd_rv"]
    fwd_ret  = df["fwd_ret_30m"]
    rv_chg   = df["rv_change"]

    # Priority order: vol-expansion > vol-crush > trend-up > trend-down > range
    regime = np.full(len(df), 0, dtype=np.int8)  # default: range
    regime = np.where(fwd_ret > RET_TREND,  1, regime)   # trend-up
    regime = np.where(fwd_ret < -RET_TREND, 2, regime)   # trend-down
    regime = np.where(rv_chg > 0.05,        3, regime)   # vol-expansion
    regime = np.where(rv_chg < -0.05,       4, regime)   # vol-crush

    df["regime"] = regime
    return df


# ── features ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol",
    "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years",
    "atm_straddle_premium", "realized_vol_30m", "iv_rv_spread",
    "minute_of_day", "minutes_to_close",
    # lags (recomputed here)
    "atm_iv_lag5", "atm_iv_lag15",
    "skew_slope_lag5", "skew_slope_lag15",
    "pcr_oi_lag5", "pcr_oi_lag15",
    "realized_vol_30m_lag5", "realized_vol_30m_lag15",
    "iv_rv_spread_lag5", "iv_rv_spread_lag15",
]


def add_lags_and_time(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["index", "trade_date"], sort=False)
    for col in ["atm_iv", "skew_slope", "pcr_oi", "realized_vol_30m", "iv_rv_spread"]:
        df[f"{col}_lag5"]  = grp[col].shift(5)
        df[f"{col}_lag15"] = grp[col].shift(15)
    df["minute_of_day"]    = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minutes_to_close"] = ((15 * 60 + 30) - df["minute_of_day"]).clip(lower=0)
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chain features …")
    df = pd.concat([pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))],
                   ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  {len(df):,} rows")

    print("Assigning regime labels …")
    df = assign_regime(df)
    df = add_lags_and_time(df)
    df = df.dropna(subset=FEATURE_COLS + ["regime"]).copy()
    print(f"  After dropna: {len(df):,} rows")

    # Regime distribution
    print("\nRegime distribution:")
    for k, name in REGIME_NAMES.items():
        n = (df["regime"] == k).sum()
        print(f"  {k} {name:>14s}: {n:>7,}  ({n/len(df):.1%})")

    # Splits
    train = df[df["trade_date"] < "2023-01-01"]
    val   = df[(df["trade_date"] >= "2023-01-01") & (df["trade_date"] < "2024-01-01")]
    test  = df[df["trade_date"] >= "2024-01-01"]
    print(f"\ntrain {len(train):,}  val {len(val):,}  test {len(test):,}")

    X_tr, y_tr = train[FEATURE_COLS], train["regime"].astype(int)
    X_va, y_va = val[FEATURE_COLS],   val["regime"].astype(int)
    X_te, y_te = test[FEATURE_COLS],  test["regime"].astype(int)

    params = {
        "objective": "multiclass", "num_class": 5, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "class_weight": "balanced",
        "n_jobs": -1, "verbosity": -1,
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
                                 target_names=[REGIME_NAMES[i] for i in range(5)]))

    print("=== Confusion matrix ===")
    cm = confusion_matrix(y_te, pred_te)
    print(pd.DataFrame(cm,
                        index=[f"true_{REGIME_NAMES[i]}" for i in range(5)],
                        columns=[f"pred_{REGIME_NAMES[i]}" for i in range(5)]).to_string())

    # Feature importance
    fi = pd.DataFrame({"feature": FEATURE_COLS,
                        "gain": booster.feature_importance("gain")})
    fi = fi.sort_values("gain", ascending=False)
    print("\n=== Top 10 features ===")
    print(fi.head(10).to_string(index=False))

    # Save model + regime probabilities on test set for downstream use
    test = test.copy()
    for i in range(5):
        test[f"regime_prob_{i}"] = pred_proba_te[:, i]
    test["regime_pred"] = pred_te
    test[["index", "datetime", "trade_date", "regime", "regime_pred"]
         + [f"regime_prob_{i}" for i in range(5)]].to_parquet(
        OUT_DIR / "v4c_test_regimes.parquet", index=False)

    booster.save_model(str(MODEL_DIR / "regime_classifier.lgb"))

    summary = {
        "regime_distribution": {REGIME_NAMES[k]: int((df["regime"]==k).sum()) for k in range(5)},
        "test_accuracy": float((pred_te == y_te.to_numpy()).mean()),
        "feature_importance_top10": fi.head(10).to_dict(orient="records"),
        "best_iteration": int(booster.best_iteration),
    }
    (OUT_DIR / "v4c_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {MODEL_DIR/'regime_classifier.lgb'}")
    print(f"Saved → {OUT_DIR/'v4c_test_regimes.parquet'}")


if __name__ == "__main__":
    main()
