#!/usr/bin/env python3
"""OptiNet v2.1 A/B evaluation harness.

Trains the 4-LGBM stack under different feature/filter configurations and
records per-config blind metrics so we can isolate the contribution of each
component (sentiment, GDELT, regime feature, regime filter, calibration).

Output: results/optinet/ab_history.csv  (one row per config × run timestamp)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from index_options.backtester import backtest_daily
from index_options.data import align_spot_to_chain
from index_options.features import build_training_frame
from index_options.labels import build_labels, merge_features_labels
from index_options.models import train_model_stack
from index_options.parquet_loader import load_index_lake, load_options_lake


# config name → (build_kwargs, apply_regime_filter)
CONFIGS: dict[str, tuple[dict, bool]] = {
    "baseline":          (dict(include_sentiment_cache=False, include_gdelt=False, include_regime=False), False),
    "+sentiment":        (dict(include_sentiment_cache=True,  include_gdelt=False, include_regime=False), False),
    "+gdelt":            (dict(include_sentiment_cache=False, include_gdelt=True,  include_regime=False), False),
    "+regime_feat":      (dict(include_sentiment_cache=False, include_gdelt=False, include_regime=True),  False),
    "+regime_filter":    (dict(include_sentiment_cache=False, include_gdelt=False, include_regime=True),  True),
    "full":              (dict(include_sentiment_cache=True,  include_gdelt=True,  include_regime=True),  True),
}


def _run_config(name: str, build_kwargs: dict, apply_regime_filter: bool,
                index_bars: pd.DataFrame, chain_spot: pd.DataFrame,
                profile: str, train_end_year: int, blind_year: int,
                min_confidence: float) -> dict:
    t0 = time.time()
    features = build_training_frame(index_bars, chain_spot, **build_kwargs)
    labels   = build_labels(index_bars)
    dataset  = merge_features_labels(features, labels)
    dataset["date"] = pd.to_datetime(dataset["date"])

    train = dataset[dataset["date"].dt.year <= train_end_year].copy()
    blind = dataset[dataset["date"].dt.year == blind_year].copy()
    blind_chain = chain_spot[pd.to_datetime(chain_spot["date"]).dt.year == blind_year].copy()

    bundle = train_model_stack(train, profile=profile)
    trades, summary = backtest_daily(
        bundle, blind, blind_chain,
        profile=profile, min_confidence=min_confidence,
        apply_regime_filter=apply_regime_filter,
    )
    dur = time.time() - t0
    return {
        "config": name,
        "rows": int(len(dataset)),
        "feature_cols": int(len(bundle.feature_columns)),
        "long_auc": round(bundle.metrics.get("long_auc", 0.0), 4),
        "short_auc": round(bundle.metrics.get("short_auc", 0.0), 4),
        "long_brier_cal": round(bundle.metrics.get("long_brier_cal", 0.0), 4),
        "short_brier_cal": round(bundle.metrics.get("short_brier_cal", 0.0), 4),
        "blind_trades": int(summary.get("trades", 0)),
        "blind_win_rate": round(summary.get("win_rate", 0.0), 4),
        "blind_stop_rate": round(summary.get("stop_exit_rate", 0.0), 4),
        "blind_target_rate": round(summary.get("target_exit_rate", 0.0), 4),
        "blind_net_pnl": round(summary.get("net_pnl", 0.0), 2),
        "blind_sharpe": round(summary.get("sharpe", 0.0), 4),
        "blind_max_drawdown": round(summary.get("max_drawdown", 0.0), 2),
        "duration_seconds": round(dur, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OptiNet v2.1 A/B evaluation harness.")
    parser.add_argument("--profile", default="balanced",
                        choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--train-end-year", type=int, default=2025)
    parser.add_argument("--blind-year", type=int, default=2026)
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    parser.add_argument("--output", default="results/optinet/ab_history.csv")
    args = parser.parse_args()

    print("Loading parquet lake…")
    option_chain = load_options_lake()
    index_bars   = load_index_lake()
    chain_spot   = align_spot_to_chain(option_chain, index_bars)

    rows = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for name in args.configs:
        if name not in CONFIGS:
            print(f"[skip] unknown config: {name}")
            continue
        build_kwargs, regime_filter = CONFIGS[name]
        print(f"\n── {name} ──")
        print(f"  build_kwargs={build_kwargs}  regime_filter={regime_filter}")
        try:
            result = _run_config(name, build_kwargs, regime_filter,
                                 index_bars, chain_spot, args.profile,
                                 args.train_end_year, args.blind_year,
                                 args.min_confidence)
            result["timestamp"] = timestamp
            result["profile"] = args.profile
            rows.append(result)
            print(f"  trades={result['blind_trades']}  "
                  f"win={result['blind_win_rate']:.0%}  "
                  f"stop={result['blind_stop_rate']:.0%}  "
                  f"sharpe={result['blind_sharpe']}  "
                  f"net_pnl={result['blind_net_pnl']}")
        except Exception as e:
            print(f"  [error] {e}")
            rows.append({"config": name, "timestamp": timestamp,
                         "profile": args.profile, "error": str(e)})

    # Append to history CSV
    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if out.exists():
        existing = pd.read_csv(out)
        history = pd.concat([existing, new_df], ignore_index=True)
    else:
        history = new_df
    history.to_csv(out, index=False)

    print(f"\n── Summary (this run) ──")
    if not new_df.empty and "blind_trades" in new_df.columns:
        cols = ["config", "blind_trades", "blind_win_rate", "blind_stop_rate",
                "blind_sharpe", "blind_net_pnl", "long_auc"]
        print(new_df[cols].to_string(index=False))
    print(f"\nAppended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
