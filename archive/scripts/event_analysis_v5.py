#!/usr/bin/env python3
"""V5 Phase 2 — Event-day investigation.

For dte=3 ranker on 2024 test: identify days with extreme negative top-1 PnL
and cross-reference with India market events:
  - RBI MPC dates 2024
  - Union Budget 1-Feb-2024
  - 2024 Lok Sabha election results 4-Jun-2024
  - US FOMC dates 2024
  - India quarterly results announcements (rough proxy)

Goal: confirm or refute that fat-tail losses cluster on event days.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS_DIR = PROJECT_ROOT / "cache/optinet_v5/strategy_labels"
CHAIN_DIR  = PROJECT_ROOT / "cache/optinet_v4/chain_features"
FUT_DIR    = PROJECT_ROOT / "cache/optinet_v5/futures_features"
RANKER     = PROJECT_ROOT / "models/optinet_v5/ranker_dte3_nosid_desc.lgb"
OUT_DIR    = PROJECT_ROOT / "results/optinet_v5"

# India 2024 known events
INDIA_EVENTS_2024 = {
    date(2024, 2, 1):  "Union Budget 2024-25",
    date(2024, 2, 8):  "RBI MPC",
    date(2024, 4, 5):  "RBI MPC",
    date(2024, 6, 4):  "Lok Sabha results",
    date(2024, 6, 7):  "RBI MPC",
    date(2024, 7, 23): "Union Budget 2024 (full)",
    date(2024, 8, 8):  "RBI MPC",
    date(2024, 10, 9): "RBI MPC",
}
# US FOMC 2024 (impact on India morning)
US_FOMC_2024 = [
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
    date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
]
# Add T-1 day (markets often pre-position on event-eve)
ALL_EVENTS = {}
for d, name in INDIA_EVENTS_2024.items():
    ALL_EVENTS[d] = name
for d in US_FOMC_2024:
    ALL_EVENTS[d] = "US FOMC"


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
DESCRIPTORS = ["sd_horizon_min", "sd_n_legs", "sd_short_vol", "sd_long_vol",
                "sd_directional", "sd_defined_risk", "sd_is_eod", "sd_is_no_trade"]


def main():
    print("=== Event-day analysis on dte=3, 2024 test ===")
    # Load 2024 only
    files = [f for f in sorted(LABELS_DIR.rglob("data.parquet"))
             if "year=2024" in str(f)]
    labels = pd.concat([pd.read_parquet(f).query("dte_bucket == 3") for f in files],
                        ignore_index=True)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    print(f"  labels (dte=3, 2024): {len(labels):,}")

    chain = pd.concat([pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))
                        if "year=2024" in str(f)], ignore_index=True)
    chain["datetime"] = pd.to_datetime(chain["datetime"])
    chain = chain[["index", "datetime"] + CHAIN_FEATURES]

    fut = pd.concat([pd.read_parquet(f) for f in sorted(FUT_DIR.rglob("data.parquet"))
                      if "year=2024" in str(f)], ignore_index=True)
    fut["datetime"] = pd.to_datetime(fut["datetime"])
    fut = fut[["index", "datetime"] + FUT_FEATURES]

    df = labels.drop(columns=["atm_iv", "atm_strike", "spot"], errors="ignore").merge(
        chain.merge(fut, on=["index", "datetime"], how="inner"),
        on=["index", "datetime"], how="inner")
    df = df[df["valid"]].copy()

    # Add simulator context + descriptors
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["hour_of_day"]   = df["datetime"].dt.hour
    df["dte"] = 5  # dte_bucket=3 contains dte 5,6,7. Use 5 as median (matches training)

    desc = pd.DataFrame.from_dict({
        0:  {"sd_horizon_min": 0,   "sd_n_legs": 0, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 1},
        1:  {"sd_horizon_min": 30,  "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0},
        2:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 1, "sd_is_no_trade": 0},
        3:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0},
        4:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 1, "sd_is_no_trade": 0},
        5:  {"sd_horizon_min": 360, "sd_n_legs": 4, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0},
        6:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 0, "sd_is_no_trade": 0},
        7:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0},
        8:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 0, "sd_is_no_trade": 0},
        9:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0},
        10: {"sd_horizon_min": 30,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0},
        11: {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0},
        12: {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0},
    }, orient="index").reset_index().rename(columns={"index": "strategy_id"})
    df = df.merge(desc, on="strategy_id", how="left")
    df = df.dropna(subset=CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT + DESCRIPTORS).copy()

    # Run ranker
    ranker = lgb.Booster(model_file=str(RANKER))
    feat_cols = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT + DESCRIPTORS
    df["score"] = ranker.predict(df[feat_cols])
    df = df.sort_values(["index", "datetime", "score"], ascending=[True, True, False])
    top1 = df.groupby(["index", "datetime"], sort=False).head(1).copy()
    print(f"  top-1 picks on dte=3 2024: {len(top1):,}")

    # Daily PnL
    daily = top1.groupby("trade_date").agg(
        n=("strategy_id", "size"),
        mean_ppp=("pnl_per_premium", "mean"),
        sum_pnl=("net_pnl", "sum"),
        mean_pnl=("net_pnl", "mean"),
        worst_pnl=("net_pnl", "min"),
    ).round(2)
    daily["is_event_day"] = daily.index.map(
        lambda d: ALL_EVENTS.get(d.date(), None))
    daily["event_t_minus_1"] = daily.index.map(
        lambda d: any(
            ((d.date() + pd.Timedelta(days=1).to_pytimedelta()) == ev_d)
            for ev_d in ALL_EVENTS.keys()))

    print("\n=== 10 worst trading days (by sum_pnl) ===")
    worst = daily.sort_values("sum_pnl").head(10)
    print(worst.to_string())

    print("\n=== 10 best trading days (by sum_pnl) ===")
    best = daily.sort_values("sum_pnl", ascending=False).head(10)
    print(best.to_string())

    print("\n=== Event days vs non-event days ===")
    event_mask = daily["is_event_day"].notna()
    print(f"Event days  : n={event_mask.sum()}  "
          f"mean_pnl_per_trade={daily[event_mask]['mean_pnl'].mean():.2f}  "
          f"mean_sum_pnl_per_day={daily[event_mask]['sum_pnl'].mean():.0f}")
    print(f"Non-event   : n={(~event_mask).sum()}  "
          f"mean_pnl_per_trade={daily[~event_mask]['mean_pnl'].mean():.2f}  "
          f"mean_sum_pnl_per_day={daily[~event_mask]['sum_pnl'].mean():.0f}")

    print("\n=== Per-month 2024 PnL ===")
    daily["month"] = daily.index.to_period("M")
    monthly = daily.groupby("month")["sum_pnl"].agg(["sum", "mean", "count"]).round(0)
    print(monthly.to_string())

    # Save
    daily.to_parquet(OUT_DIR / "phase2_event_analysis_daily.parquet")
    summary = {
        "n_event_days": int(event_mask.sum()),
        "n_non_event_days": int((~event_mask).sum()),
        "event_mean_pnl_per_trade": float(daily[event_mask]["mean_pnl"].mean()) if event_mask.any() else 0.0,
        "non_event_mean_pnl_per_trade": float(daily[~event_mask]["mean_pnl"].mean()),
        "event_mean_sum_pnl_per_day": float(daily[event_mask]["sum_pnl"].mean()) if event_mask.any() else 0.0,
        "non_event_mean_sum_pnl_per_day": float(daily[~event_mask]["sum_pnl"].mean()),
        "worst_10_days": worst.reset_index().to_dict(orient="records"),
        "best_10_days": best.reset_index().to_dict(orient="records"),
        "monthly_pnl": monthly.reset_index().astype(str).to_dict(orient="records"),
    }
    (OUT_DIR / "phase2_event_analysis.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → {OUT_DIR/'phase2_event_analysis.json'}")


if __name__ == "__main__":
    main()
