#!/usr/bin/env python3
"""V5 Phase 0 — Diagnose the V4-B signal inversion.

Question: V4-B's lowest rv_iv_ratio quintile (Q1, "vol cheap, sell premium")
earned ₹8,686/trade vs Q5's ₹12,073. Why?

Hypotheses:
  H1 — Selection bias by time-of-day: Q1 clusters late in the day when premium
       has already decayed → less theta available
  H2 — Pre-event compression: Q1 occurs in artificially calm pre-event windows
       that then break violently (or, on expiry day, gamma blowups)
  H3 — Feature lag dominance: realized_vol_30m is V4-B's dominant feature →
       model is essentially predicting RV(t+30) ≈ RV(t-30), which fails on
       regime transitions

Tests:
  Slice 1 — Hour-of-day × quintile (H1)
  Slice 2 — Days-to-expiry × quintile, is_expiry_day flag (H2)
  Slice 3 — V4-C regime label × quintile
  Slice 4 — Index × quintile, then cross with hour and dte
  Slice 5 — Feature ablation: retrain V4-B without realized_vol_* features (H3)

Realistic cost model used throughout (5-7% round-trip on ATM straddles).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHAIN_DIR   = PROJECT_ROOT / "cache/optinet_v4/chain_features"
REGIME_FILE = PROJECT_ROOT / "results/optinet_v4/v4c_test_regimes.parquet"
VOL_MODEL   = PROJECT_ROOT / "models/optinet_v4/rv_30m_forward.lgb"
OUT_DIR     = PROJECT_ROOT / "results/optinet_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOT_SIZE = {"NIFTY": 50, "BANKNIFTY": 15}
RISK_FREE = 0.065
TRADING_DAYS = 252
TRADING_MINUTES = 375

# Realistic cost model
BROK_PER_LEG = 40.0       # ₹/leg
N_LEGS = 4                # entry CE + entry PE + exit CE + exit PE
SLIPPAGE_PCT = 0.015      # 1.5% per leg of premium
STT_RATE = 0.000625       # 0.0625% on premium SELL side
EXCH_RATE = 0.000019      # exchange fee ~0.0019%
SEBI_RATE = 0.000001      # SEBI ~0.0001%
STAMP_RATE = 0.00003      # 0.003% on entry
GST_RATE = 0.18           # GST on brokerage + transaction charges

VOL_FEATURES = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
    "minute_of_day", "minutes_to_close",
    "atm_iv_lag5", "atm_iv_lag15",
    "skew_slope_lag5", "skew_slope_lag15",
    "pcr_oi_lag5", "pcr_oi_lag15",
    "pcr_vol_lag5", "pcr_vol_lag15",
    "iv_rv_spread_lag5", "iv_rv_spread_lag15",
    "realized_vol_30m_lag5", "realized_vol_30m_lag15",
    "max_oi_total_dist_pct_lag5", "max_oi_total_dist_pct_lag15",
]
# H3 ablation: drop trailing-realized-vol features
VOL_FEATURES_NO_TRAIL_RV = [c for c in VOL_FEATURES
                              if not c.startswith("realized_vol_30m")
                              and c != "iv_rv_spread"
                              and not c.startswith("iv_rv_spread")]


# ─── Pricing ──────────────────────────────────────────────────────────────────


def bs_straddle_vec(S, K, T, r, sigma):
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(S, 1e-9) / np.maximum(K, 1e-9))
              + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where((T > 0) & (sigma > 0), call + put, 0.0)


def realistic_cost(entry_px_per_share, exit_px_per_share, lot):
    """Total round-trip cost in INR for short straddle. entry_px / exit_px are
    per-share combined call+put premiums. lot is shares per lot.

    Components:
      - 4 legs brokerage
      - Slippage on entry (receive less) and exit (pay more)
      - STT on entry sell premium
      - Exchange + SEBI on entry+exit premium
      - Stamp on entry
      - GST on brokerage + exchange charges
    """
    notional_entry = entry_px_per_share * lot
    notional_exit  = exit_px_per_share  * lot

    brokerage = BROK_PER_LEG * N_LEGS
    slippage_entry = SLIPPAGE_PCT * notional_entry  # receive 1.5% less
    slippage_exit  = SLIPPAGE_PCT * notional_exit   # pay 1.5% more
    stt = STT_RATE * notional_entry  # only on the sell premium
    exch = EXCH_RATE * (notional_entry + notional_exit)
    sebi = SEBI_RATE * (notional_entry + notional_exit)
    stamp = STAMP_RATE * notional_entry
    gst = GST_RATE * (brokerage + exch + sebi)
    return brokerage + slippage_entry + slippage_exit + stt + exch + sebi + stamp + gst


# ─── Data loading ─────────────────────────────────────────────────────────────


def add_lags_and_time(df: pd.DataFrame) -> pd.DataFrame:
    df["trade_date"] = df["datetime"].dt.normalize()
    grp = df.groupby(["index", "trade_date"], sort=False)
    for col in ["atm_iv", "skew_slope", "pcr_oi", "pcr_vol",
                "realized_vol_30m", "iv_rv_spread", "max_oi_total_dist_pct"]:
        df[f"{col}_lag5"]  = grp[col].shift(5)
        df[f"{col}_lag15"] = grp[col].shift(15)
    df["minute_of_day"]    = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minutes_to_close"] = ((15 * 60 + 30) - df["minute_of_day"]).clip(lower=0)
    return df


def load_chain(years: list[int]) -> pd.DataFrame:
    files = [f for f in sorted(CHAIN_DIR.rglob("data.parquet"))
             if any(f"year={y}" in str(f) for y in years)]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["index", "datetime"]).reset_index(drop=True)


def fwd_target_and_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Add fwd_ret_30m, fwd_rv_30m, and the realistic-cost short-straddle PnL."""
    df["trade_date"] = df["datetime"].dt.normalize()
    grp = df.groupby(["index", "trade_date"], sort=False)
    df["fwd_ret_30m"] = grp["spot"].transform(lambda s: s.shift(-30) / s - 1.0)
    df["log_ret_1m"] = np.log(df["spot"] / grp["spot"].shift(1))

    def _fwd_rv(s):
        rev = s.iloc[::-1]
        return (rev.rolling(30, min_periods=20).std().iloc[::-1]
                * np.sqrt(TRADING_DAYS * TRADING_MINUTES))

    df["fwd_rv_30m"] = grp["log_ret_1m"].transform(_fwd_rv)

    # Days to expiry
    df["expiry_dt"] = pd.to_datetime(df["expiry"])
    df["days_to_expiry"] = (df["expiry_dt"] - df["trade_date"]).dt.days
    df["is_expiry_day"] = (df["days_to_expiry"] == 0).astype(int)
    df["hour"] = df["datetime"].dt.hour

    # Short-straddle PnL with realistic costs
    dt_30m = 30.0 / (TRADING_DAYS * TRADING_MINUTES)
    S0 = df["spot"].to_numpy()
    K  = df["atm_strike"].to_numpy()
    T0 = df["T_years"].to_numpy()
    iv = df["atm_iv"].to_numpy()
    entry_px = df["atm_straddle_premium"].to_numpy() * 2.0
    fwd = df["fwd_ret_30m"].to_numpy()

    S1 = S0 * (1.0 + np.where(np.isnan(fwd), 0.0, fwd))
    T1 = np.maximum(T0 - dt_30m, 1e-6)
    exit_px = bs_straddle_vec(S1, K, T1, RISK_FREE, iv)

    lots = np.array([LOT_SIZE.get(i, 50) for i in df["index"]])
    gross = (entry_px - exit_px) * lots

    # Vectorized realistic cost
    notional_entry = entry_px * lots
    notional_exit  = exit_px * lots
    brokerage = BROK_PER_LEG * N_LEGS
    slippage = SLIPPAGE_PCT * (notional_entry + notional_exit)
    stt = STT_RATE * notional_entry
    exch = EXCH_RATE * (notional_entry + notional_exit)
    sebi = SEBI_RATE * (notional_entry + notional_exit)
    stamp = STAMP_RATE * notional_entry
    gst = GST_RATE * (brokerage + exch + sebi)
    cost = brokerage + slippage + stt + exch + sebi + stamp + gst

    df["entry_px"] = entry_px
    df["exit_px"] = exit_px
    df["gross_pnl"] = gross
    df["cost"] = cost
    df["net_pnl"] = gross - cost
    return df


# ─── Quintile analysis ────────────────────────────────────────────────────────


def quintile_table(df: pd.DataFrame, label_col: str = "quintile") -> pd.DataFrame:
    rows = []
    for q in sorted(df[label_col].dropna().unique()):
        sub = df[df[label_col] == q]
        if sub.empty:
            continue
        rows.append({
            "quintile": q,
            "n": len(sub),
            "mean_pnl": float(sub["net_pnl"].mean()),
            "median_pnl": float(sub["net_pnl"].median()),
            "win_rate": float((sub["net_pnl"] > 0).mean()),
            "p5_pnl": float(np.percentile(sub["net_pnl"], 5)),
            "p95_pnl": float(np.percentile(sub["net_pnl"], 95)),
            "mean_entry_px": float(sub["entry_px"].mean()),
            "mean_exit_px": float(sub["exit_px"].mean()),
            "mean_atm_iv": float(sub["atm_iv"].mean()),
            "mean_pred_rv": float(sub["pred_rv"].mean()),
            "mean_rv_iv_ratio": float(sub["rv_iv_ratio"].mean()),
            "mean_minute_of_day": float(sub["minute_of_day"].mean()),
            "mean_dte": float(sub["days_to_expiry"].mean()),
        })
    return pd.DataFrame(rows)


def quintile_by_slice(df: pd.DataFrame, slice_col: str, slice_label: str) -> pd.DataFrame:
    """For each value of slice_col, compute Q1-Q5 PnL stats. Returns long table."""
    rows = []
    for sv in sorted(df[slice_col].dropna().unique()):
        sub = df[df[slice_col] == sv]
        if len(sub) < 50:
            continue
        # Compute quintiles WITHIN this slice (so each slice is fair)
        edges = sub["rv_iv_ratio"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy().copy()
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        sub = sub.copy()
        sub["q"] = pd.cut(sub["rv_iv_ratio"], bins=edges,
                            labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            qsub = sub[sub["q"] == q]
            if qsub.empty:
                continue
            rows.append({
                slice_label: sv,
                "quintile": q,
                "n": len(qsub),
                "mean_pnl": float(qsub["net_pnl"].mean()),
                "median_pnl": float(qsub["net_pnl"].median()),
                "win_rate": float((qsub["net_pnl"] > 0).mean()),
                "mean_entry_px": float(qsub["entry_px"].mean()),
                "mean_atm_iv": float(qsub["atm_iv"].mean()),
                "mean_rv_iv_ratio": float(qsub["rv_iv_ratio"].mean()),
            })
    return pd.DataFrame(rows)


def q1_minus_q5_summary(slice_table: pd.DataFrame, slice_label: str) -> pd.DataFrame:
    """Compute Q1 - Q5 PnL difference per slice. Negative = inversion."""
    pivot = slice_table.pivot(index=slice_label, columns="quintile", values="mean_pnl")
    pivot["Q1_minus_Q5"] = pivot.get("Q1", 0) - pivot.get("Q5", 0)
    n_pivot = slice_table.pivot(index=slice_label, columns="quintile", values="n")
    pivot["n_Q1"] = n_pivot.get("Q1", 0)
    pivot["n_Q5"] = n_pivot.get("Q5", 0)
    return pivot.reset_index()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("Phase 0 — V4-B inversion diagnostic")
    print("=" * 80)

    print("\n[1/8] Loading 2024 chain features …")
    df = load_chain([2024])
    print(f"   {len(df):,} rows")
    df = add_lags_and_time(df)

    print("[2/8] Computing forward target + realistic-cost PnL …")
    df = fwd_target_and_pnl(df)

    print("[3/8] Loading V4-B vol model and predicting on 2024 …")
    vol_model = lgb.Booster(model_file=str(VOL_MODEL))
    valid = df[VOL_FEATURES].notna().all(axis=1)
    df["pred_rv"] = np.nan
    df.loc[valid, "pred_rv"] = vol_model.predict(df.loc[valid, VOL_FEATURES])
    df["rv_iv_ratio"] = df["pred_rv"] / df["atm_iv"].replace(0, np.nan)

    print("[4/8] Loading V4-C regime predictions …")
    reg = pd.read_parquet(REGIME_FILE)[["index", "datetime", "regime_pred"]]
    reg["datetime"] = pd.to_datetime(reg["datetime"])
    df = df.merge(reg, on=["index", "datetime"], how="left")
    df["regime_pred"] = df["regime_pred"].fillna(-1).astype(int)

    # Filter to valid trades (all features present, fwd return present, T > 30m left)
    dt_30m = 30.0 / (TRADING_DAYS * TRADING_MINUTES)
    base_valid = (
        df["pred_rv"].notna() &
        df["fwd_ret_30m"].notna() &
        df["T_years"].gt(dt_30m) &
        df["atm_straddle_premium"].gt(0) &
        df["atm_iv"].gt(0)
    )
    valid_df = df[base_valid].copy()
    print(f"   Valid rows for diagnostic: {len(valid_df):,}")

    # Global quintile assignment (matches the original V4 inversion observation)
    edges = valid_df["rv_iv_ratio"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy().copy()
    edges[0] -= 1e-9; edges[-1] += 1e-9
    valid_df["q_global"] = pd.cut(valid_df["rv_iv_ratio"], bins=edges,
                                     labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    print("\n[5/8] Baseline reproduction of V4-E inversion …")
    baseline = quintile_table(valid_df, "q_global")
    print(baseline.to_string(index=False))

    # ─── H1: Hour-of-day ──────────────────────────────────────────────────────
    print("\n[6a/8] H1 test — Q1-Q5 PnL by hour-of-day …")
    h1 = quintile_by_slice(valid_df, "hour", "hour")
    h1_summary = q1_minus_q5_summary(h1, "hour")
    print(h1_summary.round(0).to_string(index=False))

    # ─── H2: Days to expiry + expiry day ──────────────────────────────────────
    print("\n[6b/8] H2 test — Q1-Q5 PnL by days_to_expiry …")
    h2_dte = quintile_by_slice(valid_df, "days_to_expiry", "days_to_expiry")
    h2_dte_summary = q1_minus_q5_summary(h2_dte, "days_to_expiry")
    print(h2_dte_summary.round(0).to_string(index=False))

    print("\n[6c/8] H2 test — Q1-Q5 PnL by is_expiry_day …")
    h2_exp = quintile_by_slice(valid_df, "is_expiry_day", "is_expiry_day")
    h2_exp_summary = q1_minus_q5_summary(h2_exp, "is_expiry_day")
    print(h2_exp_summary.round(0).to_string(index=False))

    # ─── Slice 3: Regime ──────────────────────────────────────────────────────
    print("\n[6d/8] Q1-Q5 PnL by V4-C regime …")
    regime_names = {-1: "unknown", 0: "range", 1: "trend-up", 2: "trend-down",
                     3: "vol-expansion", 4: "vol-crush"}
    valid_df["regime_name"] = valid_df["regime_pred"].map(regime_names)
    h3_reg = quintile_by_slice(valid_df, "regime_name", "regime_name")
    h3_reg_summary = q1_minus_q5_summary(h3_reg, "regime_name")
    print(h3_reg_summary.round(0).to_string(index=False))

    # ─── Slice 4: Index ───────────────────────────────────────────────────────
    print("\n[6e/8] Q1-Q5 PnL by index …")
    idx_slice = quintile_by_slice(valid_df, "index", "index")
    idx_summary = q1_minus_q5_summary(idx_slice, "index")
    print(idx_summary.round(0).to_string(index=False))

    # ─── Slice 5: Cross-tab hour × dte ────────────────────────────────────────
    print("\n[6f/8] Q1 - Q5 PnL diff: hour × days_to_expiry heatmap …")
    cross = []
    for h in sorted(valid_df["hour"].dropna().unique()):
        for dte in sorted(valid_df["days_to_expiry"].dropna().unique()):
            sub = valid_df[(valid_df["hour"] == h) & (valid_df["days_to_expiry"] == dte)]
            if len(sub) < 100:
                continue
            edges = sub["rv_iv_ratio"].quantile([0, 0.2, 0.8, 1.0]).to_numpy().copy()
            edges[0] -= 1e-9; edges[-1] += 1e-9
            sub = sub.copy()
            sub["q_local"] = pd.cut(sub["rv_iv_ratio"], bins=edges,
                                       labels=["Q1", "Mid", "Q5"])
            q1_pnl = sub[sub["q_local"] == "Q1"]["net_pnl"].mean()
            q5_pnl = sub[sub["q_local"] == "Q5"]["net_pnl"].mean()
            cross.append({"hour": int(h), "dte": int(dte),
                          "q1_pnl": float(q1_pnl), "q5_pnl": float(q5_pnl),
                          "q1_minus_q5": float(q1_pnl - q5_pnl),
                          "n_q1": int((sub["q_local"] == "Q1").sum()),
                          "n_q5": int((sub["q_local"] == "Q5").sum())})
    cross_df = pd.DataFrame(cross)
    cross_pivot = cross_df.pivot(index="hour", columns="dte", values="q1_minus_q5")
    print("Q1 minus Q5 mean PnL (negative = inversion):")
    print(cross_pivot.round(0).to_string())

    # ─── H3: Feature ablation ─────────────────────────────────────────────────
    print("\n[7/8] H3 test — Re-train V4-B WITHOUT trailing-RV features …")
    print(f"   Features dropped: {set(VOL_FEATURES) - set(VOL_FEATURES_NO_TRAIL_RV)}")

    # Train on 2020-2022, val 2023, predict on 2024 (same as V4-B but no trail-RV)
    train_df = load_chain([2020, 2021, 2022])
    train_df = add_lags_and_time(train_df)
    train_df = fwd_target_and_pnl(train_df)
    val_df = load_chain([2023])
    val_df = add_lags_and_time(val_df)
    val_df = fwd_target_and_pnl(val_df)

    train_valid = train_df.dropna(subset=VOL_FEATURES_NO_TRAIL_RV + ["fwd_rv_30m"])
    val_valid = val_df.dropna(subset=VOL_FEATURES_NO_TRAIL_RV + ["fwd_rv_30m"])
    print(f"   Train: {len(train_valid):,}  Val: {len(val_valid):,}")

    params = {
        "objective": "regression", "metric": "l1",
        "learning_rate": 0.04, "num_leaves": 63, "min_data_in_leaf": 50,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "n_jobs": -1, "verbosity": -1,
    }
    t0 = time.time()
    booster_ablation = lgb.train(
        params,
        lgb.Dataset(train_valid[VOL_FEATURES_NO_TRAIL_RV],
                    label=train_valid["fwd_rv_30m"]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(val_valid[VOL_FEATURES_NO_TRAIL_RV],
                                label=val_valid["fwd_rv_30m"])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    print(f"   Trained in {time.time() - t0:.0f}s, best_iter={booster_ablation.best_iteration}")

    # Predict on 2024 valid_df
    abl_valid = valid_df.dropna(subset=VOL_FEATURES_NO_TRAIL_RV).copy()
    abl_valid["pred_rv_ablation"] = booster_ablation.predict(
        abl_valid[VOL_FEATURES_NO_TRAIL_RV])
    abl_valid["rv_iv_ratio_ablation"] = abl_valid["pred_rv_ablation"] / abl_valid["atm_iv"]
    edges_a = abl_valid["rv_iv_ratio_ablation"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy().copy()
    edges_a[0] -= 1e-9; edges_a[-1] += 1e-9
    abl_valid["q_ablation"] = pd.cut(abl_valid["rv_iv_ratio_ablation"], bins=edges_a,
                                         labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    rv_test = abl_valid["fwd_rv_30m"].dropna().to_numpy()
    pred_test = abl_valid.loc[abl_valid["fwd_rv_30m"].notna(), "pred_rv_ablation"].to_numpy()
    if len(rv_test) > 0:
        from sklearn.metrics import r2_score, mean_absolute_error
        print(f"   Ablation 2024 test: MAE={mean_absolute_error(rv_test, pred_test):.4f}  "
              f"R²={r2_score(rv_test, pred_test):.4f}  "
              f"corr={np.corrcoef(rv_test, pred_test)[0,1]:.4f}")

    abl_quintiles = quintile_table(abl_valid, "q_ablation")
    print("\nAblation quintile structure (no trailing-RV features):")
    print(abl_quintiles.to_string(index=False))

    # Save everything
    print("\n[8/8] Saving outputs …")
    summary = {
        "baseline_quintiles": baseline.to_dict(orient="records"),
        "h1_hour_summary": h1_summary.to_dict(orient="records"),
        "h2_dte_summary": h2_dte_summary.to_dict(orient="records"),
        "h2_expiry_day_summary": h2_exp_summary.to_dict(orient="records"),
        "regime_summary": h3_reg_summary.to_dict(orient="records"),
        "index_summary": idx_summary.to_dict(orient="records"),
        "cross_hour_dte_q1_minus_q5": cross_df.to_dict(orient="records"),
        "h3_ablation_quintiles": abl_quintiles.to_dict(orient="records"),
    }
    (OUT_DIR / "phase0_diagnostic.json").write_text(
        json.dumps(summary, indent=2, default=str))

    baseline.to_parquet(OUT_DIR / "phase0_baseline_quintiles.parquet", index=False)
    h1.to_parquet(OUT_DIR / "phase0_by_hour.parquet", index=False)
    h2_dte.to_parquet(OUT_DIR / "phase0_by_dte.parquet", index=False)
    h2_exp.to_parquet(OUT_DIR / "phase0_by_expiry_day.parquet", index=False)
    h3_reg.to_parquet(OUT_DIR / "phase0_by_regime.parquet", index=False)
    idx_slice.to_parquet(OUT_DIR / "phase0_by_index.parquet", index=False)
    cross_df.to_parquet(OUT_DIR / "phase0_cross_hour_dte.parquet", index=False)
    abl_quintiles.to_parquet(OUT_DIR / "phase0_ablation_quintiles.parquet", index=False)

    print(f"   Saved 8 parquets + summary JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
