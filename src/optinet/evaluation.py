from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from intradaynet.robustness import (
    PromotionGateConfig,
    RiskPolicy,
    confidence_bucket_diagnostics,
    evaluate_promotion_gates,
    json_ready,
    summarize_trade_frame,
    write_json,
)
from optinet.backtester import backtest_daily
from optinet.models import train_model_stack
from optinet.recommender import build_dataset


def _date_filter(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"], format="mixed", errors="coerce")
    return frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _chain_filter(chain: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(chain["date"], format="mixed", errors="coerce")
    return chain[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def evaluate_optinet(
    *,
    index_paths,
    option_paths,
    profile: str,
    train_start: str,
    train_end: str,
    blind_start: str,
    blind_end: str,
    output_dir: str | Path,
    model_output: str | Path | None = None,
    dataset_path: str | Path | None = None,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    dataset, option_chain = build_dataset(index_paths, option_paths)
    dataset["date"] = pd.to_datetime(dataset["date"], format="mixed", errors="coerce")
    option_chain["date"] = pd.to_datetime(option_chain["date"], format="mixed", errors="coerce")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if dataset_path:
        dataset_out = Path(dataset_path)
    else:
        dataset_out = output / "dataset.csv"
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_out, index=False)

    train_frame = _date_filter(dataset, train_start, train_end)
    blind_frame = _date_filter(dataset, blind_start, blind_end)
    blind_chain = _chain_filter(option_chain, blind_start, blind_end)
    bundle = train_model_stack(train_frame, profile=profile)

    model_path = Path(model_output) if model_output else output / f"optinet_{profile}.pkl"
    bundle.save(model_path)

    blind_trades, blind_summary = backtest_daily(
        bundle,
        blind_frame,
        blind_chain,
        profile=profile,
        min_confidence=min_confidence,
    )
    blind_trades.to_csv(output / "blind_trades.csv", index=False)

    year_results: dict[str, dict[str, float]] = {}
    for year in sorted(pd.to_datetime(dataset["date"]).dt.year.dropna().unique()):
        year = int(year)
        frame = dataset[pd.to_datetime(dataset["date"]).dt.year == year].copy()
        chain = option_chain[pd.to_datetime(option_chain["date"]).dt.year == year].copy()
        trades, summary = backtest_daily(bundle, frame, chain, profile=profile, min_confidence=min_confidence)
        year_results[str(year)] = summary
        trades.to_csv(output / f"trades_{year}.csv", index=False)

    walk_forward = run_walk_forward(
        dataset,
        option_chain,
        profile=profile,
        min_confidence=min_confidence,
    )

    confidence = confidence_bucket_diagnostics(blind_trades)
    gate = PromotionGateConfig(min_blind_trades=50)
    readiness = evaluate_promotion_gates(blind_summary, confidence, gate)
    registry = {
        "system": "optinet",
        "profile": profile,
        "model_path": str(model_path),
        "dataset_path": str(dataset_out),
        "feature_schema_version": "optinet_v1",
        "feature_column_count": len(bundle.feature_columns),
        "feature_columns": bundle.feature_columns,
        "data_start": str(dataset["date"].min().date()) if not dataset.empty else None,
        "data_end": str(dataset["date"].max().date()) if not dataset.empty else None,
        "train_start": train_start,
        "train_end": train_end,
        "blind_start": blind_start,
        "blind_end": blind_end,
        "training_metrics": bundle.metrics,
        "blind_metrics": blind_summary,
        "readiness_status": readiness.status,
        "readiness_reasons": readiness.reasons,
        "risk_policy": asdict(RiskPolicy()),
    }

    write_json(output / "summary.json", {
        "train_rows": len(train_frame),
        "blind_rows": len(blind_frame),
        "model": str(model_path),
        "dataset": str(dataset_out),
        "training_metrics": bundle.metrics,
        "blind_summary": blind_summary,
    })
    write_json(output / "year_by_year.json", year_results)
    write_json(output / "walk_forward.json", walk_forward)
    write_json(output / "confidence_buckets.json", confidence)
    write_json(output / "readiness.json", asdict(readiness))
    write_json(output / "model_registry.json", registry)

    return json_ready({
        "summary": {
            "model": str(model_path),
            "dataset": str(dataset_out),
            "readiness": readiness.status,
            "reasons": readiness.reasons,
            "blind_summary": blind_summary,
        },
        "artifacts": {
            "summary": str(output / "summary.json"),
            "walk_forward": str(output / "walk_forward.json"),
            "confidence_buckets": str(output / "confidence_buckets.json"),
            "readiness": str(output / "readiness.json"),
            "model_registry": str(output / "model_registry.json"),
        },
    })


def run_walk_forward(
    dataset: pd.DataFrame,
    option_chain: pd.DataFrame,
    *,
    profile: str,
    min_confidence: float,
    retrain_freq: str = "annual",
) -> dict[str, Any]:
    """Walk-forward evaluation with configurable retraining frequency.

    retrain_freq:
        'annual'    — retrain at year boundary, test on whole next year
        'quarterly' — retrain at quarter boundary, test on next quarter
                       (much more responsive to regime shifts like 2024Q1)
    """
    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"], format="mixed", errors="coerce")
    option_chain = option_chain.copy()
    option_chain["date"] = pd.to_datetime(option_chain["date"], format="mixed", errors="coerce")
    results: dict[str, Any] = {}

    if retrain_freq == "annual":
        years = sorted(int(year) for year in dataset["date"].dt.year.dropna().unique())
        for test_year in years[2:]:
            train = dataset[dataset["date"].dt.year < test_year].copy()
            test = dataset[dataset["date"].dt.year == test_year].copy()
            if len(train) < 30 or test.empty:
                continue
            chain = option_chain[option_chain["date"].dt.year == test_year].copy()
            bundle = train_model_stack(train, profile=profile)
            trades, summary = backtest_daily(bundle, test, chain, profile=profile, min_confidence=min_confidence)
            results[f"through_{test_year - 1}_to_{test_year}"] = {
                "train_rows": len(train),
                "test_rows": len(test),
                "long_brier_raw": bundle.metrics.get("long_brier_raw"),
                "long_brier_cal": bundle.metrics.get("long_brier_cal"),
                "short_brier_raw": bundle.metrics.get("short_brier_raw"),
                "short_brier_cal": bundle.metrics.get("short_brier_cal"),
                "long_auc": bundle.metrics.get("long_auc"),
                "short_auc": bundle.metrics.get("short_auc"),
                **summarize_trade_frame(trades),
                "summary": summary,
            }

    elif retrain_freq == "quarterly":
        # Iterate over (year, quarter) pairs starting from the third quarter of the dataset
        # (we need at least 2 prior quarters of training data to start)
        dataset["quarter"] = dataset["date"].dt.to_period("Q")
        quarters = sorted(dataset["quarter"].dropna().unique())
        # Skip first 4 quarters as pure-train priming, then walk forward
        for test_q in quarters[4:]:
            train = dataset[dataset["quarter"] < test_q].copy()
            test = dataset[dataset["quarter"] == test_q].copy()
            if len(train) < 30 or test.empty:
                continue
            chain = option_chain[
                option_chain["date"].dt.to_period("Q") == test_q
            ].copy()
            bundle = train_model_stack(train, profile=profile)
            trades, summary = backtest_daily(bundle, test, chain, profile=profile, min_confidence=min_confidence)
            results[f"q_{test_q}"] = {
                "train_rows": len(train),
                "test_rows": len(test),
                "long_auc": bundle.metrics.get("long_auc"),
                "short_auc": bundle.metrics.get("short_auc"),
                "long_brier_cal": bundle.metrics.get("long_brier_cal"),
                "short_brier_cal": bundle.metrics.get("short_brier_cal"),
                **summarize_trade_frame(trades),
                "summary": summary,
            }

    else:
        raise ValueError(f"Unknown retrain_freq: {retrain_freq!r}")

    return results
