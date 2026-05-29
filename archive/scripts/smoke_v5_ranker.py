#!/usr/bin/env python3
"""V5 Phase 1.5 — LambdaRank smoke test on dte=0 bucket.

Train a LambdaRank model that ranks the 13 candidate strategies for each
(index, datetime) on dte=0 (expiry day). Evaluate:
  - NDCG@1 (diagnostic)
  - Top-1 pnl_per_premium on val + test (primary go/no-go)
  - Top-1 hit rate
  - Lift over SHORT_STRADDLE_EOD baseline (best static strategy on dte=0)
  - Lift over random pick

Splits:
  Train: 2020-2022
  Val  : 2023
  Test : 2024 (Jan-Oct)

Sample weights:
  Real prices       → 1.0
  Synthetic prices  → 0.3 (down-weight per V5 spec)

Features:
  V4-A chain + V5 futures + time + dte + interaction
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS_DIR = PROJECT_ROOT / "cache/optinet_v5/strategy_labels"
CHAIN_DIR  = PROJECT_ROOT / "cache/optinet_v4/chain_features"
FUT_DIR    = PROJECT_ROOT / "cache/optinet_v5/futures_features"
OUT_DIR    = PROJECT_ROOT / "results/optinet_v5"
MODEL_DIR  = PROJECT_ROOT / "models/optinet_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Feature columns ──────────────────────────────────────────────────────────

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
SIM_CONTEXT_FEATURES = [
    "minute_of_day", "hour_of_day",
    "dte",  # within dte_bucket=0 it's always 0, but kept for shape
]
STRATEGY_FEATURE = "strategy_id"   # the candidate identity, important for ranking

# Strategy descriptors: engineered features that describe each candidate's structure
# without revealing identity. These let LambdaRank reason about strategies via type.
STRATEGY_DESCRIPTORS = {
    0:  {"sd_horizon_min": 0,   "sd_n_legs": 0, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 1, "sd_atm_dist_pct": 0.0},
    1:  {"sd_horizon_min": 30,  "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    2:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    3:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.01},
    4:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.01},
    5:  {"sd_horizon_min": 360, "sd_n_legs": 4, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.05},
    6:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    7:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    8:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    9:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    10: {"sd_horizon_min": 30,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    11: {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    12: {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.01},
}
STRATEGY_DESCRIPTOR_FEATURES = [
    "sd_horizon_min", "sd_n_legs", "sd_short_vol", "sd_long_vol",
    "sd_directional", "sd_defined_risk", "sd_is_eod", "sd_is_no_trade",
    "sd_atm_dist_pct",
]


def load_dte_bucket(dte_bucket: int) -> pd.DataFrame:
    files = sorted(LABELS_DIR.rglob("data.parquet"))
    print(f"Loading {len(files)} partitions, filtering dte_bucket={dte_bucket} …")
    chunks = []
    for f in files:
        df = pd.read_parquet(f)
        chunks.append(df[df["dte_bucket"] == dte_bucket])
    out = pd.concat(chunks, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out


def load_features() -> pd.DataFrame:
    """Load all V4-A chain features + V5 futures features, union all years."""
    chain = pd.concat(
        [pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))],
        ignore_index=True)
    chain["datetime"] = pd.to_datetime(chain["datetime"])
    chain = chain[["index", "datetime"] + CHAIN_FEATURES]
    fut = pd.concat(
        [pd.read_parquet(f) for f in sorted(FUT_DIR.rglob("data.parquet"))],
        ignore_index=True)
    fut["datetime"] = pd.to_datetime(fut["datetime"])
    fut = fut[["index", "datetime"] + FUT_FEATURES]
    return chain.merge(fut, on=["index", "datetime"], how="inner")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dte_bucket", type=int, default=0,
                        help="Which dte bucket to train on: 0, 1, 2, or 3")
    parser.add_argument("--no_strategy_id", action="store_true",
                        help="Drop strategy_id feature to force learning from chain context")
    parser.add_argument("--use_descriptors", action="store_true",
                        help="Add 8 strategy descriptor features (type/horizon/risk profile)")
    parser.add_argument("--candidate_strategies", type=int, nargs="+", default=None,
                        help="If set, restrict training to these strategy_ids (constrained ranker)")
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--num_boost_round", type=int, default=500)
    args = parser.parse_args()

    DTE_BUCKET = args.dte_bucket
    USE_STRATEGY_ID = not args.no_strategy_id
    USE_DESCRIPTORS = args.use_descriptors
    CANDIDATES = args.candidate_strategies
    suffix = (f"dte{DTE_BUCKET}"
              + ("" if USE_STRATEGY_ID else "_nosid")
              + ("_desc" if USE_DESCRIPTORS else "")
              + (f"_constrained{len(CANDIDATES)}" if CANDIDATES else ""))

    print("=" * 80)
    print(f"V5 LambdaRank smoke test — dte_bucket={DTE_BUCKET}  "
          f"strategy_id={USE_STRATEGY_ID}  descriptors={USE_DESCRIPTORS}")
    if CANDIDATES:
        print(f"  CONSTRAINED to strategies: {CANDIDATES}")
    print("=" * 80)

    t0 = time.time()
    labels = load_dte_bucket(DTE_BUCKET)
    print(f"  labels: {len(labels):,} rows  "
          f"({labels['datetime'].min()} → {labels['datetime'].max()})")

    feats = load_features()
    print(f"  features: {len(feats):,} rows")

    # Drop columns from labels that overlap with feature columns (avoid suffix conflicts)
    labels_cols_to_drop = ["atm_iv", "atm_strike", "spot"]
    labels = labels.drop(columns=[c for c in labels_cols_to_drop if c in labels.columns])

    df = labels.merge(feats, on=["index", "datetime"], how="inner")
    print(f"  joined : {len(df):,} rows  ({(time.time()-t0):.1f}s)")

    # Drop invalid strategy rows
    df = df[df["valid"]].copy()
    print(f"  valid  : {len(df):,} rows")

    # Drop rows missing essential features
    needed = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT_FEATURES
    df = df.dropna(subset=needed).copy()
    print(f"  after dropna : {len(df):,} rows")

    # Sample weights: real=1.0, synthetic=0.3
    df["weight"] = np.where(df["synthetic_price_used"], 0.3, 1.0)

    # Optional: restrict candidate set (constrained ranker experiment)
    if CANDIDATES is not None:
        before = len(df)
        df = df[df["strategy_id"].isin(CANDIDATES)].copy()
        print(f"  candidate filter applied: {before:,} → {len(df):,} rows "
              f"(strategies: {sorted(CANDIDATES)})")

    # Optionally inject strategy descriptor features (per-candidate but not identity)
    if USE_DESCRIPTORS:
        desc_df = pd.DataFrame.from_dict(STRATEGY_DESCRIPTORS, orient="index").reset_index()
        desc_df = desc_df.rename(columns={"index": "strategy_id"})
        df = df.merge(desc_df, on="strategy_id", how="left")
        print(f"  injected {len(STRATEGY_DESCRIPTOR_FEATURES)} strategy descriptor features")

    # Splits
    train = df[df["trade_date"] < "2023-01-01"].copy()
    val   = df[(df["trade_date"] >= "2023-01-01") & (df["trade_date"] < "2024-01-01")].copy()
    test  = df[df["trade_date"] >= "2024-01-01"].copy()
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        n_groups = sub.groupby(["index", "datetime"]).ngroups
        print(f"  {name:>5s}: {len(sub):>9,} rows  {n_groups:>6,} groups  "
              f"({sub['trade_date'].min().date()}→{sub['trade_date'].max().date()})")

    # ─── Prepare LightGBM ranking ────────────────────────────────────────────
    # IMPORTANT: rows must be sorted by (index, datetime) so groups are contiguous
    train = train.sort_values(["index", "datetime", "strategy_id"]).reset_index(drop=True)
    val   = val.sort_values(["index", "datetime", "strategy_id"]).reset_index(drop=True)
    test  = test.sort_values(["index", "datetime", "strategy_id"]).reset_index(drop=True)

    # group sizes
    train_groups = train.groupby(["index", "datetime"], sort=False).size().to_numpy()
    val_groups   = val.groupby(["index", "datetime"], sort=False).size().to_numpy()
    test_groups  = test.groupby(["index", "datetime"], sort=False).size().to_numpy()

    # Relevance for LambdaRank: bucket pnl_per_premium into integer bins for NDCG.
    # Map ppp → relevance score (0..15) by quantile binning within group.
    # Simpler: rank within group, use rank as relevance.
    def assign_relevance(group: pd.DataFrame) -> pd.Series:
        return group["pnl_per_premium"].rank(method="dense").astype(int)

    train["relevance"] = train.groupby(["index", "datetime"], sort=False).apply(
        lambda g: g["pnl_per_premium"].rank(method="dense").astype(int)
    ).reset_index(level=[0, 1], drop=True)
    val["relevance"] = val.groupby(["index", "datetime"], sort=False).apply(
        lambda g: g["pnl_per_premium"].rank(method="dense").astype(int)
    ).reset_index(level=[0, 1], drop=True)
    test["relevance"] = test.groupby(["index", "datetime"], sort=False).apply(
        lambda g: g["pnl_per_premium"].rank(method="dense").astype(int)
    ).reset_index(level=[0, 1], drop=True)

    # Feature matrix: optionally include strategy_id and/or descriptors
    feat_cols = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT_FEATURES
    if USE_STRATEGY_ID:
        feat_cols = feat_cols + [STRATEGY_FEATURE]
    if USE_DESCRIPTORS:
        feat_cols = feat_cols + STRATEGY_DESCRIPTOR_FEATURES
    print(f"\n  feature columns ({len(feat_cols)}):")
    print("    " + ", ".join(feat_cols))

    X_tr = train[feat_cols].to_numpy()
    y_tr = train["relevance"].to_numpy()
    w_tr = train["weight"].to_numpy()

    X_va = val[feat_cols].to_numpy()
    y_va = val["relevance"].to_numpy()
    w_va = val["weight"].to_numpy()

    X_te = test[feat_cols].to_numpy()
    y_te = test["relevance"].to_numpy()

    # Training
    print("\nTraining LambdaRank …")
    train_ds = lgb.Dataset(X_tr, label=y_tr, group=train_groups, weight=w_tr,
                            feature_name=feat_cols)
    val_ds   = lgb.Dataset(X_va, label=y_va, group=val_groups, weight=w_va,
                            reference=train_ds)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "n_jobs": -1,
        "verbosity": -1,
    }
    t0 = time.time()
    booster = lgb.train(
        params, train_ds, num_boost_round=args.num_boost_round,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    print(f"Trained in {time.time()-t0:.0f}s  best_iter={booster.best_iteration}")

    # ─── Inference ────────────────────────────────────────────────────────────
    print("\nPredicting on val + test …")
    val_scores  = booster.predict(X_va, num_iteration=booster.best_iteration)
    test_scores = booster.predict(X_te, num_iteration=booster.best_iteration)
    val["score"]  = val_scores
    test["score"] = test_scores

    # ─── Top-1 selection per group ────────────────────────────────────────────
    def evaluate_split(sub: pd.DataFrame, name: str) -> dict:
        # Rank within group by score, take top-1
        sub = sub.copy()
        sub["rank_in_group"] = sub.groupby(["index", "datetime"], sort=False)["score"].rank(
            method="first", ascending=False)
        top1 = sub[sub["rank_in_group"] == 1].copy()

        n_groups = sub.groupby(["index", "datetime"]).ngroups
        n_days = sub["trade_date"].nunique()

        # Baselines
        random_pick_ppp = sub.groupby(["index", "datetime"], sort=False)["pnl_per_premium"].mean().mean()
        baseline_ss_eod = sub[sub["strategy_name"] == "SHORT_STRADDLE_EOD"]["pnl_per_premium"].mean()
        baseline_no_trade = 0.0

        # Top-1 stats
        top1_mean_ppp = top1["pnl_per_premium"].mean()
        top1_median_ppp = top1["pnl_per_premium"].median()
        top1_hit_rate = (top1["pnl_per_premium"] > 0).mean()
        top1_pnl_inr = top1["net_pnl"].mean()

        # Strategy distribution of top-1 picks
        strat_dist = top1["strategy_name"].value_counts(normalize=True).round(4) * 100

        # NDCG@1 manual computation: relevance of top-1 pick / max relevance in group
        max_rel_per_group = sub.groupby(["index", "datetime"], sort=False)["relevance"].max()
        top1_rel = top1.set_index(["index", "datetime"])["relevance"]
        ndcg1 = (top1_rel / max_rel_per_group.replace(0, 1)).mean()

        print(f"\n  === {name} ===")
        print(f"  groups (decision points): {n_groups:,}  days: {n_days}")
        print(f"  trades/day: {n_groups/n_days:.2f}")
        print(f"  Top-1 mean ppp     : {top1_mean_ppp:+.4f}  ({top1_mean_ppp*100:+.2f}%)")
        print(f"  Top-1 median ppp   : {top1_median_ppp:+.4f}")
        print(f"  Top-1 hit rate     : {top1_hit_rate:.2%}")
        print(f"  Top-1 mean PnL ₹   : {top1_pnl_inr:+.0f}")
        print(f"  Random pick ppp    : {random_pick_ppp:+.4f}")
        print(f"  Baseline SS_EOD ppp: {baseline_ss_eod:+.4f}")
        print(f"  Baseline NO_TRADE  : {baseline_no_trade:+.4f}")
        print(f"  Lift over random   : {(top1_mean_ppp - random_pick_ppp):+.4f}")
        print(f"  Lift over SS_EOD   : {(top1_mean_ppp - baseline_ss_eod):+.4f}")
        print(f"  NDCG@1 (top1_rel / max_rel): {ndcg1:.4f}")
        print(f"\n  Top-1 strategy distribution (% of picks):")
        print("    " + strat_dist.to_string().replace("\n", "\n    "))

        return {
            "name": name,
            "n_groups": int(n_groups), "n_days": int(n_days),
            "trades_per_day": round(n_groups / n_days, 2),
            "top1_mean_ppp": float(top1_mean_ppp),
            "top1_median_ppp": float(top1_median_ppp),
            "top1_hit_rate": float(top1_hit_rate),
            "top1_mean_pnl_inr": float(top1_pnl_inr),
            "random_pick_ppp": float(random_pick_ppp),
            "baseline_ss_eod": float(baseline_ss_eod),
            "lift_over_random": float(top1_mean_ppp - random_pick_ppp),
            "lift_over_ss_eod": float(top1_mean_ppp - baseline_ss_eod),
            "ndcg1": float(ndcg1),
            "strategy_dist_pct": strat_dist.to_dict(),
        }

    val_metrics = evaluate_split(val, "VAL (2023)")
    test_metrics = evaluate_split(test, "TEST (2024)")

    # ─── Threshold-based selectivity check (calibrate on val) ────────────────
    # Margin = top1_score - top2_score; only keep groups where margin > threshold
    print("\n=== Selectivity sweep on TEST (top1_score margin over top2) ===")
    test_grp = test.groupby(["index", "datetime"], sort=False)
    test["rank_in_group"] = test_grp["score"].rank(method="first", ascending=False)
    top1_t = test[test["rank_in_group"] == 1].copy().rename(columns={"score": "top1_score"})
    top2_t = test[test["rank_in_group"] == 2][["index", "datetime", "score"]].rename(
        columns={"score": "top2_score"})
    selected = top1_t.merge(top2_t, on=["index", "datetime"], how="left")
    selected["margin"] = selected["top1_score"] - selected["top2_score"]
    n_days_test = test["trade_date"].nunique()

    sweep_rows = []
    for thr in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0]:
        sub = selected[selected["margin"] >= thr]
        if sub.empty:
            sweep_rows.append({"threshold": thr, "n": 0})
            continue
        sweep_rows.append({
            "threshold": thr,
            "n": int(len(sub)),
            "trades_per_day": round(len(sub) / n_days_test, 2),
            "mean_ppp": round(float(sub["pnl_per_premium"].mean()), 4),
            "hit_rate": round(float((sub["pnl_per_premium"] > 0).mean()), 4),
            "mean_pnl_inr": round(float(sub["net_pnl"].mean()), 0),
        })
    print(pd.DataFrame(sweep_rows).to_string(index=False))

    # ─── Save ────────────────────────────────────────────────────────────────
    booster.save_model(str(MODEL_DIR / f"ranker_{suffix}.lgb"))
    summary = {
        "dte_bucket": DTE_BUCKET,
        "use_strategy_id": USE_STRATEGY_ID,
        "feature_columns": feat_cols,
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
        "n_test_rows": int(len(test)),
        "best_iteration": int(booster.best_iteration),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_threshold_sweep": sweep_rows,
        "feature_importance_top15": pd.DataFrame({
            "feature": feat_cols,
            "gain": booster.feature_importance("gain"),
        }).sort_values("gain", ascending=False).head(15).to_dict(orient="records"),
    }
    (OUT_DIR / f"phase1_smoke_{suffix}.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"\nFeature importance top 10:")
    fi = pd.DataFrame({"feature": feat_cols, "gain": booster.feature_importance("gain")})
    fi = fi.sort_values("gain", ascending=False).head(10)
    print(fi.to_string(index=False))

    print(f"\nSaved → {MODEL_DIR/f'ranker_{suffix}.lgb'}")
    print(f"Saved → {OUT_DIR/f'phase1_smoke_{suffix}.json'}")


if __name__ == "__main__":
    main()
