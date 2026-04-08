#!/usr/bin/env python3
"""
Generate profile-based pre-market recommendations from the live model bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.costs import DEFAULT_COSTS
from intradaynet.feature_contract import FEATURE_NAMES
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.per_bar_features import compute_per_bar_features
from intradaynet.features.sentiment_features import SentimentFeatureBuilder
from intradaynet.features.session_features import compute_session_features
from intradaynet.flatten import flatten_window_for_lgbm
from intradaynet.model_bundle import load_bundle
from intradaynet.recommendation import build_candidate, build_recommendation_payload
from intradaynet.regime import get_regime_from_market_data


SEQ_LENGTH = 120
HORIZON_WEIGHTS = {"H15": 0.2, "H30": 0.3, "H60": 0.5}
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate live pre-market picks")
    parser.add_argument("--bundle-dir", default="runs/live_backend")
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--source", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--sentiment-csv", default="sentiment/combined_sentiment_2015_2025.csv")
    parser.add_argument("--market-cache", default="market_data_cache")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-stocks", type=int, default=0)
    return parser.parse_args()


def load_stock_data(symbol: str, source: str, data_dir: Path) -> pd.DataFrame | None:
    if source == "parquet":
        path = data_dir / f"{symbol}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
    else:
        path = data_dir / f"{symbol}_minute.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        date_col = "date" if "date" in df.columns else "datetime"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    df.columns = df.columns.str.lower()
    return df.sort_index()


def apply_calibrator(calibrator, raw_probs: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw_probs
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    return calibrator.predict(raw_probs)


def infer_symbol(symbol: str, bundle_models, sentiment_builder, data_dir: Path, source: str):
    df = load_stock_data(symbol, source, data_dir)
    if df is None or len(df) < SEQ_LENGTH:
        return None

    per_bar = compute_per_bar_features(df)
    session = compute_session_features(df)
    sentiment = sentiment_builder.get_features(symbol, session.index)
    if len(session) == 0:
        return None

    session_idx = session.index[-1]
    flat, runtime_names = flatten_window_for_lgbm(
        per_bar.iloc[-SEQ_LENGTH:].values.astype(np.float32),
        session.iloc[-1].values.astype(np.float32),
        sentiment.loc[session_idx].values.astype(np.float32),
        None,
    )
    if runtime_names != list(FEATURE_NAMES):
        raise ValueError("Runtime feature names do not match shared contract")

    X = flat.reshape(1, -1)
    latest_close = float(df["close"].iloc[-1])
    turnover = df["close"].values * df["volume"].values
    avg_daily_traded_value = float(np.mean(turnover[-375:])) if len(turnover) >= 375 else float(np.mean(turnover))
    median_minute_turnover = float(np.median(turnover[-375:])) if len(turnover) >= 375 else float(np.median(turnover))

    per_horizon = {}
    for horizon, model_group in bundle_models.items():
        raw_prob = model_group["dir"].predict(X)
        prob = float(apply_calibrator(model_group["calibrator"], raw_prob)[0])
        gross = float(model_group["ret"].predict(X)[0])
        edge = float(model_group["edge"].predict(X)[0])
        per_horizon[horizon] = {
            "probability": prob,
            "gross_return": gross,
            "net_edge": edge,
            "confidence": abs(prob - 0.5) * 2.0,
        }

    return {
        "symbol": symbol,
        "session_date": str(session_idx.date()),
        "entry_reference": latest_close,
        "avg_daily_traded_value": avg_daily_traded_value,
        "median_minute_turnover": median_minute_turnover,
        "per_horizon": per_horizon,
    }


def aggregate_candidate(symbol_result, market_regime: str):
    per_horizon = symbol_result["per_horizon"]
    total_weight = sum(HORIZON_WEIGHTS.get(h, 0.0) for h in per_horizon)
    if total_weight <= 0:
        return []

    combined_prob = sum(per_horizon[h]["probability"] * HORIZON_WEIGHTS[h] for h in per_horizon) / total_weight
    combined_gross = sum(per_horizon[h]["gross_return"] * HORIZON_WEIGHTS[h] for h in per_horizon) / total_weight
    combined_edge = sum(per_horizon[h]["net_edge"] * HORIZON_WEIGHTS[h] for h in per_horizon) / total_weight
    combined_conf = sum(per_horizon[h]["confidence"] * HORIZON_WEIGHTS[h] for h in per_horizon) / total_weight

    side = "LONG" if combined_prob >= 0.5 else "SHORT"
    side_adjusted_gross = combined_gross if side == "LONG" else -combined_gross
    side_adjusted_edge = combined_edge if side == "LONG" else -combined_edge
    cost_fraction = DEFAULT_COSTS.estimate_round_trip_fraction(symbol_result["entry_reference"])
    strongest_horizon = max(per_horizon.items(), key=lambda item: abs(item[1]["net_edge"]))[0]

    drivers = [f"{h}:edge={values['net_edge']:.4f}" for h, values in per_horizon.items()]
    candidate = build_candidate(
        symbol=symbol_result["symbol"],
        side=side,
        horizon=strongest_horizon,
        entry_reference=symbol_result["entry_reference"],
        expected_gross_return=max(side_adjusted_gross, 0.0),
        expected_net_edge=max(side_adjusted_edge, 0.0),
        confidence=combined_conf,
        probability=combined_prob,
        avg_daily_traded_value=symbol_result["avg_daily_traded_value"],
        median_minute_turnover=symbol_result["median_minute_turnover"],
        regime=market_regime,
        sector="UNKNOWN",
        driver_flags=drivers,
        cost_fraction=cost_fraction,
    )
    return [candidate]


def main():
    args = parse_args()
    manifest, models = load_bundle(args.bundle_dir)

    data_dir = Path(args.data_dir)
    pattern = "*.parquet" if args.source == "parquet" else "*_minute.csv"
    symbols = sorted(p.stem.replace("_minute", "") for p in data_dir.glob(pattern))
    if args.max_stocks > 0:
        symbols = symbols[: args.max_stocks]

    market_builder = MarketFeatureBuilder(cache_dir=args.market_cache)
    sentiment_builder = SentimentFeatureBuilder(args.sentiment_csv, market_builder=market_builder)

    console.print(
        Panel.fit(
            "[bold cyan]IntradayNet Live Picks[/bold cyan]\n"
            f"[dim]Bundle: {manifest.bundle_name} | Symbols: {len(symbols)}[/dim]",
            border_style="cyan",
        )
    )

    market_regime, should_trade, regime_reason = get_regime_from_market_data(
        nifty50_path=str(Path(args.market_cache) / "nifty50.csv"),
        india_vix_path=str(Path(args.market_cache) / "india_vix.csv"),
    )

    candidates = []
    session_date = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scoring symbols", total=len(symbols))
        for symbol in symbols:
            progress.update(task, description=f"Scoring {symbol}")
            result = infer_symbol(symbol, models, sentiment_builder, data_dir, args.source)
            if result is not None:
                session_date = result["session_date"]
                candidates.extend(aggregate_candidate(result, market_regime.value))
            progress.advance(task)

    payload = build_recommendation_payload(
        trade_date=session_date or pd.Timestamp.now().strftime("%Y-%m-%d"),
        market_regime=market_regime.value,
        market_summary={
            "should_trade": should_trade,
            "reason": regime_reason,
            "bundle_name": manifest.bundle_name,
            "bundle_version": manifest.bundle_version,
        },
        candidates=candidates if should_trade else [],
    )

    summary = Table(title="Recommendation Summary")
    summary.add_column("Profile", style="cyan")
    summary.add_column("Long", justify="right", style="green")
    summary.add_column("Short", justify="right", style="red")
    for profile, books in payload["profiles"].items():
        summary.add_row(profile, str(len(books["long"])), str(len(books["short"])))

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Trade Date:[/bold] {payload['trade_date']}\n"
            f"[bold]Market Regime:[/bold] {payload['market_regime']}\n"
            f"[dim]{payload['market_summary'].get('reason', '')}[/dim]",
            border_style="green" if payload["market_summary"].get("should_trade", False) else "yellow",
        )
    )
    console.print(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"\n[bold green]Saved picks to[/bold green] [dim]{output_path}[/dim]")
    else:
        console.print_json(json.dumps(payload))


if __name__ == "__main__":
    main()
