from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from index_options.backtester import backtest_daily
from index_options.data import align_spot_to_chain, load_index_bars, load_option_chain
from index_options.features import build_training_frame
from index_options.labels import build_labels, merge_features_labels
from index_options.models import OptiNetModelBundle, score_frame, train_model_stack
from index_options.readiness import evaluate_readiness
from index_options.translator import translate_ranked_signals


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_dataset(index_paths, option_paths, *, market_builder=None, sentiment_builder=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_bars = load_index_bars(index_paths)
    option_chain = align_spot_to_chain(load_option_chain(option_paths), index_bars)
    features = build_training_frame(index_bars, option_chain, market_builder=market_builder, sentiment_builder=sentiment_builder)
    labels = build_labels(index_bars)
    return merge_features_labels(features, labels), option_chain


def train_from_files(index_paths, option_paths, output_path: str | Path, *, profile: str = "balanced") -> OptiNetModelBundle:
    dataset, _ = build_dataset(index_paths, option_paths)
    bundle = train_model_stack(dataset, profile=profile)
    bundle.save(output_path)
    return bundle


def recommend_latest(
    bundle: OptiNetModelBundle,
    features: pd.DataFrame,
    option_chain: pd.DataFrame,
    *,
    profile: str = "balanced",
    top_k: int = 4,
    min_confidence: float = 0.55,
) -> dict[str, object]:
    if features.empty:
        return {"status": "NOT_READY", "picks": [], "readiness": {"warnings": ["No features available"]}}
    latest_date = pd.to_datetime(features["date"]).max().normalize()
    latest_features = features[pd.to_datetime(features["date"]).dt.normalize() == latest_date].copy()
    scores = score_frame(bundle, latest_features)
    scores = scores[scores["confidence"] >= min_confidence]
    trades = translate_ranked_signals(scores, option_chain, latest_features, profile=profile, top_k=top_k)
    readiness = evaluate_readiness(features, option_chain)
    return {
        "status": readiness.status,
        "as_of": str(latest_date.date()),
        "profile": profile,
        "picks": [trade.as_dict() for trade in trades],
        "readiness": {
            "checks": readiness.checks,
            "warnings": readiness.warnings,
            "coverage": readiness.coverage,
        },
    }


def write_json_report(payload: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def backtest_from_files(
    model_path: str | Path,
    index_paths,
    option_paths,
    *,
    profile: str = "balanced",
    output_dir: str | Path = "results/optinet/backtest",
) -> dict[str, object]:
    dataset, option_chain = build_dataset(index_paths, option_paths)
    bundle = OptiNetModelBundle.load(model_path)
    trades, summary = backtest_daily(bundle, dataset, option_chain, profile=profile)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "trades.csv", index=False)
    write_json_report(summary, output / "summary.json")
    return {"trades_csv": str(output / "trades.csv"), "summary": summary}
