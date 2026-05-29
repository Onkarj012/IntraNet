#!/usr/bin/env python3
"""V5 Phase 1.3 + 1.4 — Simulator sanity checks + dte-stratified validation.

Sanity checks:
  S1: NO_TRADE has zero PnL everywhere
  S2: SHORT_STRADDLE_30M aggregate ppp matches realistic theta-capture range
  S3: synthetic_price_used fraction by strategy
  S4: per-strategy global aggregate stats (mean ppp, win rate, n)

Dte-stratified validation (the key V5 corrective):
  V1: SHORT_STRADDLE_30M ppp by dte bucket — should NOT show V4-E inversion
  V2: Mean ppp of best-strategy-per-minute, by dte bucket
  V3: Distribution of best_strategy_id by dte bucket
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABELS_DIR = PROJECT_ROOT / "cache/optinet_v5/strategy_labels"
OUT_DIR = PROJECT_ROOT / "results/optinet_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all():
    files = sorted(LABELS_DIR.rglob("data.parquet"))
    print(f"Loading {len(files)} partitions …")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    print(f"  {len(df):,} rows  ({len(df)/1e6:.2f}M)")
    return df


def main():
    print("=" * 80)
    print("V5 simulator sanity checks + dte-stratified validation")
    print("=" * 80)

    df = load_all()
    print(f"\nDate range: {df['trade_date'].min().date()} → {df['trade_date'].max().date()}")
    print(f"Indices: {sorted(df['index'].unique())}")
    print(f"Strategies: {sorted(df['strategy_id'].unique())}")
    print(f"Dte distribution: {df['dte_bucket'].value_counts().sort_index().to_dict()}")

    # ─── S1: NO_TRADE ────────────────────────────────────────────────────────
    print("\n[S1] NO_TRADE rows always zero …")
    nt = df[df["strategy_id"] == 0]
    print(f"  n={len(nt):,}")
    print(f"  net_pnl: min={nt['net_pnl'].min()}  max={nt['net_pnl'].max()}  "
          f"all_zero={(nt['net_pnl'] == 0).all()}")
    print(f"  ppp:     min={nt['pnl_per_premium'].min()}  max={nt['pnl_per_premium'].max()}  "
          f"all_zero={(nt['pnl_per_premium'] == 0).all()}")
    assert (nt["net_pnl"] == 0).all(), "NO_TRADE has nonzero PnL"
    print("  PASS")

    # ─── S3: synthetic fraction per strategy ─────────────────────────────────
    print("\n[S3] Synthetic price usage by strategy …")
    syn = df[df["valid"]].groupby("strategy_name")["synthetic_price_used"].mean()
    print(syn.sort_values(ascending=False).round(4).to_string())

    # ─── S4: per-strategy global aggregate ────────────────────────────────────
    print("\n[S4] Per-strategy global aggregate (valid only, real prices only) …")
    real_df = df[df["valid"] & ~df["synthetic_price_used"]]
    agg = real_df.groupby("strategy_name").agg(
        n=("strategy_id", "size"),
        avg_entry_inr=("entry_premium_inr", "mean"),
        avg_net_pnl=("net_pnl", "mean"),
        avg_ppp=("pnl_per_premium", "mean"),
        median_ppp=("pnl_per_premium", "median"),
        std_ppp=("pnl_per_premium", "std"),
        win_rate=("net_pnl", lambda s: (s > 0).mean()),
    ).round(4)
    agg = agg.sort_values("avg_ppp", ascending=False)
    print(agg.to_string())

    # ─── V1: SHORT_STRADDLE_30M ppp by dte bucket (THE CRITICAL CHECK) ───────
    print("\n[V1] SHORT_STRADDLE_30M ppp by dte_bucket — INVERSION CHECK")
    print("  (V4-E showed Q1<Q5 by ₹3,387 in absolute PnL; if we did this right,")
    print("   ppp by dte_bucket should NOT show such confounded patterns.)")
    ss30 = real_df[real_df["strategy_name"] == "SHORT_STRADDLE_30M"]
    by_dte = ss30.groupby("dte_bucket").agg(
        n=("strategy_id", "size"),
        avg_entry_inr=("entry_premium_inr", "mean"),
        avg_net_pnl=("net_pnl", "mean"),
        avg_ppp=("pnl_per_premium", "mean"),
        median_ppp=("pnl_per_premium", "median"),
        win_rate=("net_pnl", lambda s: (s > 0).mean()),
    ).round(4)
    print(by_dte.to_string())

    # Same but for SHORT_STRADDLE_EOD (longer hold, more theta)
    print("\n  SHORT_STRADDLE_EOD by dte_bucket:")
    ss_eod = real_df[real_df["strategy_name"] == "SHORT_STRADDLE_EOD"]
    by_dte_eod = ss_eod.groupby("dte_bucket").agg(
        n=("strategy_id", "size"),
        avg_entry_inr=("entry_premium_inr", "mean"),
        avg_net_pnl=("net_pnl", "mean"),
        avg_ppp=("pnl_per_premium", "mean"),
        win_rate=("net_pnl", lambda s: (s > 0).mean()),
    ).round(4)
    print(by_dte_eod.to_string())

    # ─── V2: Best-strategy-per-minute by dte bucket ──────────────────────────
    print("\n[V2] Best strategy per minute by dte_bucket (oracle upper bound)…")
    valid_only = df[df["valid"]].copy()
    # For each (datetime, index), find the max ppp strategy
    best = valid_only.loc[
        valid_only.groupby(["index", "datetime"])["pnl_per_premium"].idxmax()
    ]
    by_dte_best = best.groupby("dte_bucket").agg(
        n=("strategy_id", "size"),
        avg_best_ppp=("pnl_per_premium", "mean"),
        median_best_ppp=("pnl_per_premium", "median"),
        avg_best_pnl=("net_pnl", "mean"),
        positive_frac=("pnl_per_premium", lambda s: (s > 0).mean()),
    ).round(4)
    print(by_dte_best.to_string())

    # ─── V3: Distribution of best_strategy_id by dte bucket ──────────────────
    print("\n[V3] Distribution of best-strategy-per-minute by dte_bucket (% of minutes):")
    dist = (best.groupby(["dte_bucket", "strategy_name"]).size()
              .unstack(fill_value=0))
    dist_pct = dist.div(dist.sum(axis=1), axis=0).round(3) * 100
    print(dist_pct.to_string())

    # ─── V4: Per-strategy ppp by dte_bucket (full table) ─────────────────────
    print("\n[V4] Per-strategy mean ppp by dte_bucket:")
    pivot = real_df.pivot_table(values="pnl_per_premium",
                                  index="strategy_name", columns="dte_bucket",
                                  aggfunc="mean").round(4)
    pivot["ALL"] = real_df.groupby("strategy_name")["pnl_per_premium"].mean().round(4)
    print(pivot.sort_values("ALL", ascending=False).to_string())

    # ─── V5: Per-(dte, regime) — load V4-C regime predictions for 2024 ───────
    # (skipping for now; V4-C only has 2024 predictions)

    # Save outputs
    summary = {
        "total_rows": int(len(df)),
        "valid_rows": int(df["valid"].sum()),
        "synthetic_overall_pct": float(df["synthetic_price_used"].mean()),
        "no_trade_zero": bool((nt["net_pnl"] == 0).all()),
        "agg_per_strategy": agg.reset_index().to_dict(orient="records"),
        "ss30m_by_dte": by_dte.reset_index().to_dict(orient="records"),
        "ss_eod_by_dte": by_dte_eod.reset_index().to_dict(orient="records"),
        "best_per_dte": by_dte_best.reset_index().to_dict(orient="records"),
        "best_strategy_dist_per_dte_pct": dist_pct.reset_index().to_dict(orient="records"),
        "ppp_pivot_strategy_x_dte": pivot.reset_index().to_dict(orient="records"),
    }
    (OUT_DIR / "phase1_sanity.json").write_text(json.dumps(summary, indent=2, default=str))
    agg.to_parquet(OUT_DIR / "phase1_per_strategy_agg.parquet")
    by_dte.to_parquet(OUT_DIR / "phase1_ss30m_by_dte.parquet")
    by_dte_best.to_parquet(OUT_DIR / "phase1_best_per_dte.parquet")
    pivot.to_parquet(OUT_DIR / "phase1_strategy_x_dte_ppp.parquet")
    print(f"\nSaved → {OUT_DIR/'phase1_sanity.json'} + 4 parquets")


if __name__ == "__main__":
    main()
