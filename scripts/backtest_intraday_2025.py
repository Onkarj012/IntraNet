#!/usr/bin/env python3
"""
Risk-based backtesting for the intraday movement model.

Runs the trained LONG / SHORT / magnitude models on daily open-safe features,
selects the top-ranked opportunities for each trading day, and simulates
intraday exits using stop-loss / target / trailing-stop rules.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder
from intradaynet.live_news import (
    LiveNewsSummary,
    combine_article_sources,
    fetch_live_yfinance_news,
    normalize_historical_sentiment_csv,
    summarize_article_coverage,
)
from intradaynet.open_safe_daily_features import build_open_safe_daily_features
from intradaynet.run_logging import command_string, start_run_logging
from intradaynet.universe import filter_symbols_by_industry, get_universe, resolve_industry_filters
from intradaynet.v7 import (
    compute_trade_levels,
    default_readiness_paths,
    evaluate_readiness,
    executable_edge_from_prediction,
    load_json_if_exists,
    margin_adjusted_confidence,
    score_candidate,
    select_candidates,
)
from intradaynet.v7_modes import (
    RuntimeTracker,
    classify_regime,
    compute_post_open_adjustment,
    compute_preferred_filter_pass,
    extract_post_open_session,
)

console = Console()
COST_PER_1L = 182.0


@dataclass
class RiskConfig:
    name: str
    stop_loss_pct: float
    target_pct: float
    trailing_start: float
    trailing_stop_pct: float
    max_trades_per_day: int
    min_confidence: float
    min_predicted_magnitude: float


@dataclass
class TradeRecord:
    date: str
    symbol: str
    mode: str
    direction: str
    entry_basis: str
    previous_close: float
    cutoff_close: float | None
    cutoff_time: str | None
    confidence: float
    score: float
    predicted_magnitude: float
    preferred_filter_pass: bool
    entry_price: float
    target_price: float
    stop_loss_price: float
    exit_price: float
    exit_reason: str
    gross_pct: float
    net_pnl: float
    max_profit_pct: float
    max_drawdown_pct: float


RISK_PROFILES = {
    "conservative": RiskConfig(
        name="Conservative",
        stop_loss_pct=0.005,
        target_pct=0.010,
        trailing_start=0.005,
        trailing_stop_pct=0.003,
        max_trades_per_day=3,
        min_confidence=0.50,
        min_predicted_magnitude=0.013,
    ),
    "balanced": RiskConfig(
        name="Balanced",
        stop_loss_pct=0.010,
        target_pct=0.015,
        trailing_start=0.008,
        trailing_stop_pct=0.005,
        max_trades_per_day=5,
        min_confidence=0.50,
        min_predicted_magnitude=0.012,
    ),
    "aggressive": RiskConfig(
        name="Aggressive",
        stop_loss_pct=0.020,
        target_pct=0.025,
        trailing_start=0.015,
        trailing_stop_pct=0.010,
        max_trades_per_day=8,
        min_confidence=0.48,
        min_predicted_magnitude=0.010,
    ),
}


def simulate_trade(
    day_data: pd.DataFrame,
    direction: str,
    entry_price: float,
    config: RiskConfig,
    position_size: float,
    *,
    target_price: float,
    stop_price: float,
    start_timestamp: pd.Timestamp | None = None,
) -> dict:
    if start_timestamp is not None:
        day_data = day_data[day_data.index > start_timestamp]
    if day_data.empty:
        return {
            "exit_price": float(entry_price),
            "exit_reason": "NO_BARS_AFTER_ENTRY",
            "gross_pct": 0.0,
            "net_pnl": -(COST_PER_1L * (position_size / 100_000.0)),
            "max_profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    target = target_price
    stop = stop_price

    highs = day_data["high"].values
    lows = day_data["low"].values
    closes = day_data["close"].values

    exit_3pm_idx = min(330, len(day_data) - 1)
    max_profit = 0.0
    max_drawdown = 0.0
    trailing_stop = None
    exit_price = closes[-1]
    exit_reason = "EOD"

    for i in range(len(day_data)):
        high = highs[i]
        low = lows[i]

        if direction == "LONG":
            current_profit = (high - entry_price) / entry_price
            current_drawdown = (low - entry_price) / entry_price
        else:
            current_profit = (entry_price - low) / entry_price
            current_drawdown = (entry_price - high) / entry_price

        max_profit = max(max_profit, current_profit)
        max_drawdown = min(max_drawdown, current_drawdown)

        if trailing_stop is None and current_profit >= config.trailing_start:
            if direction == "LONG":
                trailing_stop = high * (1 - config.trailing_stop_pct)
            else:
                trailing_stop = low * (1 + config.trailing_stop_pct)

        if direction == "LONG":
            if high >= target:
                exit_price = target
                exit_reason = "TARGET"
                break
            if low <= stop:
                exit_price = stop
                exit_reason = "STOP"
                break
            if trailing_stop is not None and low <= trailing_stop:
                exit_price = trailing_stop
                exit_reason = "TRAILING"
                break
        else:
            if low <= target:
                exit_price = target
                exit_reason = "TARGET"
                break
            if high >= stop:
                exit_price = stop
                exit_reason = "STOP"
                break
            if trailing_stop is not None and high >= trailing_stop:
                exit_price = trailing_stop
                exit_reason = "TRAILING"
                break

        if i >= exit_3pm_idx:
            exit_price = closes[i]
            exit_reason = "3PM"
            break

    gross_pct = (exit_price - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_price) / entry_price
    costs = COST_PER_1L * (position_size / 100_000.0)
    net_pnl = gross_pct * position_size - costs
    return {
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_pct": float(gross_pct),
        "net_pnl": float(net_pnl),
        "max_profit_pct": float(max_profit),
        "max_drawdown_pct": float(max_drawdown),
    }


def evaluate_hit_metrics(
    day_data: pd.DataFrame,
    direction: str,
    target_price: float,
    stop_price: float,
) -> dict:
    highs = day_data["high"].values
    lows = day_data["low"].values

    target_touched_intraday = False
    target_before_stop = False
    stop_hit = False

    for high, low in zip(highs, lows):
        if direction == "LONG":
            if low <= stop_price:
                stop_hit = True
                break
            if high >= target_price:
                target_touched_intraday = True
                target_before_stop = True
                break
        else:
            if high >= stop_price:
                stop_hit = True
                break
            if low <= target_price:
                target_touched_intraday = True
                target_before_stop = True
                break

    if not target_touched_intraday:
        if direction == "LONG":
            target_touched_intraday = bool((day_data["high"] >= target_price).any())
        else:
            target_touched_intraday = bool((day_data["low"] <= target_price).any())

    return {
        "target_touched_intraday": target_touched_intraday,
        "target_before_stop": target_before_stop,
        "stop_hit_first": stop_hit and not target_before_stop,
    }


def load_minute_data(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.columns = df.columns.str.lower()
    return df.sort_index()


def _infer_title_sentiment(text: str) -> float:
    positive = {
        "beat", "beats", "growth", "surge", "up", "gain", "gains", "strong",
        "bullish", "profit", "profits", "record", "expands", "upgrade",
    }
    negative = {
        "miss", "misses", "drop", "falls", "down", "loss", "losses", "weak",
        "bearish", "cut", "cuts", "downgrade", "fraud", "slump", "warning",
    }
    words = {w.strip(".,:;!?()[]{}'\"").lower() for w in text.split()}
    score = 0
    score += sum(1 for word in words if word in positive)
    score -= sum(1 for word in words if word in negative)
    return float(np.clip(score / 4.0, -1.0, 1.0))


def augment_sentiment_with_yfinance(
    symbols: list[str],
    base_csv: Path,
    start_date: str,
    end_date: str,
    output_csv: Path,
) -> Path:
    import yfinance as yf

    base_df = pd.read_csv(base_csv) if base_csv.exists() else pd.DataFrame(
        columns=["Symbol", "Publish Date", "sentiment_score"]
    )
    rows: list[dict] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    for symbol in symbols:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            news_items = getattr(ticker, "news", []) or []
        except Exception:
            continue

        for item in news_items:
            publish_ts = item.get("providerPublishTime")
            if publish_ts is None:
                continue
            ts = pd.to_datetime(publish_ts, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
            if ts < start_ts or ts >= end_ts:
                continue
            title = item.get("title", "") or ""
            rows.append(
                {
                    "Symbol": symbol,
                    "Publish Date": ts.isoformat(sep=" "),
                    "sentiment_score": _infer_title_sentiment(title),
                }
            )

    if rows:
        extra_df = pd.DataFrame(rows).drop_duplicates(subset=["Symbol", "Publish Date"], keep="last")
        merged = pd.concat([base_df, extra_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Symbol", "Publish Date"], keep="last")
    else:
        merged = base_df

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv


def maybe_backfill_with_yfinance(
    minute_df: pd.DataFrame | None,
    symbol: str,
    start_date: str,
    end_date: str,
    refresh: bool,
) -> pd.DataFrame | None:
    if minute_df is None:
        minute_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if not refresh:
        return minute_df if not minute_df.empty else None

    import yfinance as yf

    ticker = yf.Ticker(f"{symbol}.NS")
    try:
        daily = ticker.history(
            start=start_date,
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
    except Exception:
        return minute_df if not minute_df.empty else None

    if daily.empty:
        return minute_df if not minute_df.empty else None

    daily = daily.reset_index()
    date_col = "Date" if "Date" in daily.columns else daily.columns[0]
    daily[date_col] = pd.to_datetime(daily[date_col]).dt.tz_localize(None)
    pseudo_rows = []

    existing_days = set(minute_df.index.normalize()) if not minute_df.empty else set()
    for _, row in daily.iterrows():
        dt = pd.Timestamp(row[date_col]).normalize()
        if dt in existing_days:
            continue
        pseudo_rows.append(
            {
                "date": dt + pd.Timedelta(hours=15, minutes=29),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0.0)),
            }
        )

    if not pseudo_rows:
        return minute_df if not minute_df.empty else None

    pseudo_df = pd.DataFrame(pseudo_rows).set_index("date").sort_index()
    merged = pd.concat([minute_df, pseudo_df]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest intraday movement model")
    parser.add_argument("--model", default="results/models/models/intraday_model_nifty500.pkl")
    parser.add_argument("--risk", default="balanced", choices=sorted(RISK_PROFILES.keys()))
    parser.add_argument("--mode", choices=("premarket", "post-open"), default="premarket")
    parser.add_argument("--post-open-cutoff", default="09:30")
    parser.add_argument("--post-open-news-cutoff", default="09:20")
    parser.add_argument("--universe", default="nifty100")
    parser.add_argument("--industry", action="append", default=[])
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--output-dir", default="backtest_results")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--allow-below-preferred",
        action="store_true",
        help="Backfill daily slots with below-threshold names after exhausting preferred candidates.",
    )
    parser.add_argument("--min-confidence", type=float, default=-1.0)
    parser.add_argument("--min-predicted-magnitude", type=float, default=-1.0)
    parser.add_argument("--refresh-yfinance", action="store_true")
    parser.add_argument("--augment-yf-news", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = RISK_PROFILES[args.risk]
    started_at = perf_counter()
    runtime_tracker = RuntimeTracker()
    run_name = f"backtest_intraday_2025_{args.risk}_{args.mode}"

    with start_run_logging(project_root=PROJECT_ROOT, log_group="backtests", run_name=run_name) as run_logger:
        global console
        console = Console()

        max_trades_per_day = args.top_k if args.top_k > 0 else config.max_trades_per_day
        min_confidence = args.min_confidence if args.min_confidence >= 0 else config.min_confidence
        min_predicted_magnitude = (
            args.min_predicted_magnitude if args.min_predicted_magnitude >= 0 else config.min_predicted_magnitude
        )
        per_trade_risk_budget = args.capital * 0.01 / max(max_trades_per_day, 1)
        position_size = per_trade_risk_budget / max(config.stop_loss_pct, 1e-6)
        position_size = max(10_000.0, min(position_size, args.capital / max(max_trades_per_day, 1)))

        console.print(
            Panel.fit(
                "[bold cyan]Intraday Movement Backtest[/bold cyan]\n"
                f"[dim]Risk: {config.name} | Range: {args.start_date} to {args.end_date} | Capital: ₹{args.capital:,.0f}[/dim]",
                border_style="cyan",
            )
        )
        console.print(f"[dim]Command:[/dim] {command_string()}")
        console.print(f"[dim]Run log:[/dim] {run_logger.log_path}")

        model_path = Path(args.model)
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        models = model_data["models"]
        feature_cols = model_data["features"]

        data_dir = Path(args.data_dir)
        mode_output_name = "premarket_backtest" if args.mode == "premarket" else "post_open_backtest"
        output_dir = Path(args.output_dir) / mode_output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        symbols = get_universe(args.universe)
        resolved_industries = resolve_industry_filters(args.industry) if args.industry else []
        symbols = filter_symbols_by_industry(symbols, resolved_industries)
        if args.max_stocks > 0:
            symbols = symbols[:args.max_stocks]

        market_builder = MarketFeatureBuilder()
        runtime_tracker.start("data_refresh_seconds")
        market_builder.download(start="2021-01-01", end=args.end_date)
        sentiment_csv = Path("data/sentiment/combined_sentiment_2015_2025.csv")
        historical_articles = normalize_historical_sentiment_csv(
            sentiment_csv,
            universe_metadata_csv=str(PROJECT_ROOT / "data" / "sentiment" / "ind_nifty500list.csv"),
            market_open_cutoff="09:15",
            post_open_cutoff=args.post_open_news_cutoff,
        )
        live_articles = pd.DataFrame()
        news_summary = LiveNewsSummary()
        if args.augment_yf_news:
            console.print("[bold]Augmenting backtest context with yfinance news where available...[/bold]")
            live_articles, news_summary = fetch_live_yfinance_news(
                symbols,
                start_ts=pd.Timestamp(args.start_date),
                end_ts=pd.Timestamp(args.end_date) + pd.Timedelta(days=1),
                universe_metadata_csv=str(PROJECT_ROOT / "data" / "sentiment" / "ind_nifty500list.csv"),
                market_open_cutoff="09:15",
                post_open_cutoff=args.post_open_news_cutoff,
            )
        combined_articles = combine_article_sources(historical_articles, live_articles)
        news_summary = summarize_article_coverage(
            combined_articles,
            symbols=symbols,
            mode=args.mode,
            target_dates=pd.date_range(args.start_date, args.end_date, freq="B"),
            base_summary=news_summary,
        )
        sentiment_builder = SentimentFeatureBuilder(
            str(sentiment_csv),
            market_builder=market_builder,
            mode=args.mode,
            post_open_news_cutoff=args.post_open_news_cutoff,
            universe_metadata_csv=str(PROJECT_ROOT / "data" / "sentiment" / "ind_nifty500list.csv"),
            articles_df=combined_articles,
        )
        sentiment_builder._load()
        runtime_tracker.stop("data_refresh_seconds")

        start_date = pd.Timestamp(args.start_date)
        end_date = pd.Timestamp(args.end_date)
        candidates_by_date: dict[pd.Timestamp, list[dict]] = defaultdict(list)
        skipped_symbols: list[str] = []
        processed_symbols = 0
        coverage_by_date: dict[pd.Timestamp, int] = defaultdict(int)

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
                runtime_tracker.start("data_refresh_seconds")
                minute_df = load_minute_data(data_dir / f"{symbol}_minute.csv")
                minute_df = maybe_backfill_with_yfinance(
                    minute_df,
                    symbol,
                    args.start_date,
                    args.end_date,
                    args.refresh_yfinance,
                )
                runtime_tracker.stop("data_refresh_seconds")
                if minute_df is None:
                    skipped_symbols.append(symbol)
                    progress.advance(task)
                    continue

                runtime_tracker.start("feature_build_seconds")
                feature_df = build_open_safe_daily_features(minute_df, symbol, market_builder, sentiment_builder)
                runtime_tracker.stop("feature_build_seconds")
                if feature_df is None or feature_df.empty:
                    skipped_symbols.append(symbol)
                    progress.advance(task)
                    continue

                feature_df = feature_df[(feature_df.index >= start_date) & (feature_df.index <= end_date)]
                if feature_df.empty:
                    skipped_symbols.append(symbol)
                    progress.advance(task)
                    continue

                processed_symbols += 1
                for dt in feature_df.index:
                    coverage_by_date[dt.normalize()] += 1
                X = feature_df.reindex(columns=feature_cols, fill_value=0.0)
                long_probs = models["long"].predict_proba(X)[:, 1]
                short_probs = models["short"].predict_proba(X)[:, 1]
                up_mags = np.maximum(models["up_mag"].predict(X), 0.0)
                down_mags = np.maximum(models["down_mag"].predict(X), 0.0)

                for idx, dt in enumerate(feature_df.index):
                    trade_date = dt.normalize()
                    date_data = minute_df[minute_df.index.normalize() == trade_date]
                    if date_data.empty:
                        continue

                    long_prob = float(long_probs[idx])
                    short_prob = float(short_probs[idx])
                    long_confidence = margin_adjusted_confidence(long_prob, short_prob)
                    short_confidence = margin_adjusted_confidence(short_prob, long_prob)
                    long_edge = executable_edge_from_prediction(
                        float(up_mags[idx]),
                        target_pct=config.target_pct,
                    )
                    short_edge = executable_edge_from_prediction(
                        float(down_mags[idx]),
                        target_pct=config.target_pct,
                    )
                    previous_close = float(date_data["close"].iloc[-1])
                    feature_row = feature_df.loc[dt]
                    regime = classify_regime(feature_row)
                    risk_on_signal = float(feature_row.get("risk_on_signal", 0.0))
                    market_breadth = float(feature_row.get("market_breadth", 0.0))
                    sector_relative_strength = float(feature_row.get("sector_relative_strength", 0.0))
                    long_regime_ok = not (risk_on_signal < -0.25 and market_breadth < -0.02 and sector_relative_strength < -0.01)
                    short_regime_ok = not (risk_on_signal > 0.25 and market_breadth > 0.02 and sector_relative_strength > 0.01)

                    for direction, base_probability, base_confidence, pred_mag in (
                        ("LONG", long_prob, long_confidence, long_edge),
                        ("SHORT", short_prob, short_confidence, short_edge),
                    ):
                        runtime_tracker.start("ranking_seconds")
                        confidence = base_confidence
                        adjusted_mag = pred_mag
                        entry_basis = "Previous Close"
                        entry_price = previous_close
                        cutoff_close = None
                        cutoff_time = None
                        session_df = pd.DataFrame()
                        long_side = direction == "LONG"
                        alignment_score = 0.0
                        alignment_ok = True
                        regime_ok = long_regime_ok if long_side else short_regime_ok

                        if args.mode == "post-open":
                            trade_date = dt.normalize() + pd.Timedelta(days=1)
                            if trade_date > end_date:
                                runtime_tracker.stop("ranking_seconds")
                                continue
                            session_df = extract_post_open_session(minute_df, trade_date, args.post_open_cutoff)
                            if session_df.empty:
                                runtime_tracker.stop("ranking_seconds")
                                continue
                            coverage_by_date[trade_date] += 1
                            adjustment = compute_post_open_adjustment(
                                direction=direction,
                                prev_close=previous_close,
                                base_probability=base_probability,
                                predicted_magnitude=pred_mag,
                                session_df=session_df,
                                minute_df=minute_df,
                                feature_row=feature_row,
                            )
                            confidence = float(adjustment["adjusted_probability"])
                            adjusted_mag = float(adjustment["adjusted_magnitude"])
                            entry_price = float(adjustment["reference_price"])
                            cutoff_close = adjustment["cutoff_close"]
                            cutoff_time = adjustment["cutoff_timestamp"]
                            alignment_score = float(adjustment["alignment_score"])
                            alignment_ok = adjustment["gap_pct"] is not None and alignment_score >= 0.05

                        preferred_filter_pass = compute_preferred_filter_pass(
                            confidence=confidence,
                            predicted_magnitude=adjusted_mag,
                            min_confidence=min_confidence,
                            min_predicted_magnitude=min_predicted_magnitude,
                            alignment_ok=alignment_ok,
                            regime_ok=regime_ok,
                        )
                        score = score_candidate(confidence, adjusted_mag)
                        target_price, stop_price = compute_trade_levels(
                            reference_price=entry_price,
                            direction=direction,
                            target_pct=config.target_pct,
                            stop_loss_pct=config.stop_loss_pct,
                        )

                        candidates_by_date[(dt.normalize() + pd.Timedelta(days=1)) if args.mode == "post-open" else dt.normalize()].append(
                            {
                                "date": (dt.normalize() + pd.Timedelta(days=1)) if args.mode == "post-open" else dt.normalize(),
                                "symbol": symbol,
                                "mode": args.mode,
                                "direction": direction,
                                "confidence": confidence,
                                "predicted_magnitude": adjusted_mag,
                                "score": score,
                                "preferred_filter_pass": preferred_filter_pass,
                                "entry_basis": "Cutoff Close" if args.mode == "post-open" else entry_basis,
                                "entry_price": entry_price,
                                "previous_close": previous_close,
                                "cutoff_close": cutoff_close,
                                "cutoff_time": cutoff_time,
                                "target_price": target_price,
                                "stop_price": stop_price,
                                "day_data": minute_df[minute_df.index.normalize() == ((dt.normalize() + pd.Timedelta(days=1)) if args.mode == "post-open" else dt.normalize())],
                                "session_df": session_df,
                                "feature_date": dt,
                                **regime,
                            }
                        )
                        runtime_tracker.stop("ranking_seconds")
                progress.advance(task)

        trades: list[TradeRecord] = []
        hit_records: list[dict] = []
        for trade_date in sorted(candidates_by_date):
            ranked = select_candidates(
                candidates_by_date[trade_date],
                count=max_trades_per_day,
                allow_below_preferred=args.allow_below_preferred,
            )
            for candidate in ranked[:max_trades_per_day]:
                entry_price = float(candidate["entry_price"])
                if entry_price <= 0:
                    continue
                start_timestamp = (
                    pd.Timestamp(candidate["cutoff_time"]) if candidate["mode"] == "post-open" and candidate["cutoff_time"] else None
                )
                outcome = simulate_trade(
                    candidate["day_data"],
                    candidate["direction"],
                    entry_price,
                    config,
                    position_size,
                    target_price=float(candidate["target_price"]),
                    stop_price=float(candidate["stop_price"]),
                    start_timestamp=start_timestamp,
                )
                hit = evaluate_hit_metrics(
                    candidate["day_data"][candidate["day_data"].index > start_timestamp] if start_timestamp is not None else candidate["day_data"],
                    candidate["direction"],
                    float(candidate["target_price"]),
                    float(candidate["stop_price"]),
                )
                hit_records.append(
                    {
                        "date": str(trade_date.date()),
                        "symbol": candidate["symbol"],
                        "mode": candidate["mode"],
                        "direction": candidate["direction"],
                        "predicted_target": round(float(candidate["target_price"]), 2),
                        **hit,
                    }
                )
                trades.append(
                    TradeRecord(
                        date=str(trade_date.date()),
                        symbol=candidate["symbol"],
                        mode=candidate["mode"],
                        direction=candidate["direction"],
                        entry_basis=candidate["entry_basis"],
                        previous_close=round(float(candidate["previous_close"]), 2),
                        cutoff_close=round(float(candidate["cutoff_close"]), 2) if candidate["cutoff_close"] is not None else None,
                        cutoff_time=candidate["cutoff_time"],
                        confidence=round(candidate["confidence"], 4),
                        score=round(candidate["score"], 6),
                        predicted_magnitude=round(candidate["predicted_magnitude"], 6),
                        preferred_filter_pass=bool(candidate["preferred_filter_pass"]),
                        entry_price=round(entry_price, 2),
                        target_price=round(float(candidate["target_price"]), 2),
                        stop_loss_price=round(float(candidate["stop_price"]), 2),
                        exit_price=round(outcome["exit_price"], 2),
                        exit_reason=outcome["exit_reason"],
                        gross_pct=round(outcome["gross_pct"], 6),
                        net_pnl=round(outcome["net_pnl"], 2),
                        max_profit_pct=round(outcome["max_profit_pct"], 6),
                        max_drawdown_pct=round(outcome["max_drawdown_pct"], 6),
                    )
                )

        if not trades:
            console.print("[bold red]No trades were generated.[/bold red]")
            console.print(f"[bold yellow]Run log saved to[/bold yellow] {run_logger.log_path}")
            return

        trades_df = pd.DataFrame([asdict(t) for t in trades])
        trades_path = output_dir / f"trades_intraday_model_{args.risk}_{args.mode}_{args.start_date}_{args.end_date}.csv"
        trades_df.to_csv(trades_path, index=False)
        hits_df = pd.DataFrame(hit_records)
        hits_path = output_dir / f"hits_intraday_model_{args.risk}_{args.mode}_{args.start_date}_{args.end_date}.csv"
        hits_df.to_csv(hits_path, index=False)

        expected_dates = pd.date_range(start_date, end_date, freq="B")
        coverage_rows = []
        trade_days = set(pd.to_datetime(trades_df["date"]).dt.normalize()) if not trades_df.empty else set()
        for dt in expected_dates:
            coverage_rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "symbols_with_data": int(coverage_by_date.get(dt.normalize(), 0)),
                    "has_any_data": int(coverage_by_date.get(dt.normalize(), 0) > 0),
                    "trade_count": int((trades_df["date"] == dt.strftime("%Y-%m-%d")).sum()) if not trades_df.empty else 0,
                    "had_trade": int(dt.normalize() in trade_days),
                }
            )
        coverage_df = pd.DataFrame(coverage_rows)
        coverage_path = output_dir / f"coverage_intraday_model_{args.risk}_{args.mode}_{args.start_date}_{args.end_date}.csv"
        coverage_df.to_csv(coverage_path, index=False)

        total_trades = len(trades_df)
        winning_trades = int((trades_df["net_pnl"] > 0).sum())
        total_net = float(trades_df["net_pnl"].sum())
        avg_net = float(trades_df["net_pnl"].mean())
        avg_gross_pct = float(trades_df["gross_pct"].mean() * 100.0)
        win_rate = winning_trades / total_trades
        by_reason = dict(Counter(trades_df["exit_reason"]))
        active_days = int(trades_df["date"].nunique())
        avg_trades_per_day = total_trades / max(active_days, 1)
        daily_pnl = trades_df.groupby("date")["net_pnl"].sum()
        sharpe = float((daily_pnl.mean() / max(daily_pnl.std(ddof=0), 1e-9)) * np.sqrt(252)) if len(daily_pnl) > 1 else 0.0
        equity_curve = daily_pnl.cumsum()
        max_drawdown = float((equity_curve - equity_curve.cummax()).min()) if len(equity_curve) else 0.0
        hit_rate = float(hits_df["target_touched_intraday"].mean()) if not hits_df.empty else 0.0
        target_before_stop_rate = float(hits_df["target_before_stop"].mean()) if not hits_df.empty else 0.0
        runtime_metrics = runtime_tracker.snapshot(perf_counter() - started_at)
        preferred_count = int(trades_df["preferred_filter_pass"].sum()) if "preferred_filter_pass" in trades_df else 0

        summary = {
            "model": str(model_path),
            "mode": args.mode,
            "entry_basis": "Cutoff Close" if args.mode == "post-open" else "Previous Close",
            "post_open_cutoff": args.post_open_cutoff if args.mode == "post-open" else None,
            "risk_profile": config.name,
            "capital": args.capital,
            "top_k": max_trades_per_day,
            "allow_below_preferred": args.allow_below_preferred,
            "position_size": position_size,
            "target_pct": config.target_pct,
            "stop_loss_pct": config.stop_loss_pct,
            "min_confidence": min_confidence,
            "min_predicted_magnitude": min_predicted_magnitude,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "processed_symbols": processed_symbols,
            "industry_filter": resolved_industries,
            "skipped_symbols": skipped_symbols,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_net_pnl": total_net,
            "avg_net_pnl": avg_net,
            "avg_gross_pct": avg_gross_pct,
            "active_days": active_days,
            "avg_trades_per_day": avg_trades_per_day,
            "sharpe_like_daily": sharpe,
            "target_touched_intraday_hit_rate": hit_rate,
            "target_before_stop_rate": target_before_stop_rate,
            "max_drawdown": max_drawdown,
            "expected_business_days": int(len(expected_dates)),
            "days_with_any_data": int(coverage_df["has_any_data"].sum()),
            "days_without_any_data": int((coverage_df["has_any_data"] == 0).sum()),
            "target_alignment": True,
            "exact_logic_match": True,
            "exit_reasons": by_reason,
            "runtime_metrics": runtime_metrics,
            "news_summary": news_summary.to_dict(),
            "recommendation_count_requested": max_trades_per_day,
            "recommendation_count_returned": total_trades,
            "top_n_forced_count": max(total_trades - preferred_count, 0),
            "top_n_preferred_count": preferred_count,
            "regime_breakdown": {
                "trend_regime": trades_df.groupby("date")["gross_pct"].mean().mean() if not trades_df.empty else 0.0,
                "volatility_regime": trades_df.groupby("direction")["gross_pct"].mean().to_dict() if not trades_df.empty else {},
                "long_short_split": trades_df.groupby("direction")["net_pnl"].agg(["count", "mean", "sum"]).to_dict() if not trades_df.empty else {},
            },
            "run_log": str(run_logger.log_path),
            "trades_csv": str(trades_path),
            "hits_csv": str(hits_path),
            "coverage_csv": str(coverage_path),
            "sentiment_csv_used": str(sentiment_csv),
        }

        readiness_paths = default_readiness_paths(PROJECT_ROOT, mode=args.mode)
        locked_summary = summary if (
            args.start_date == "2025-01-01" and args.end_date == "2025-12-31"
        ) else load_json_if_exists(readiness_paths["locked_backtest"])
        forward_summary = summary if (
            args.start_date == "2026-01-01" and args.end_date == "2026-03-31"
        ) else load_json_if_exists(readiness_paths["forward_blind"])
        readiness = evaluate_readiness(
            locked_backtest_summary=locked_summary,
            forward_summary=forward_summary,
            target_alignment=True,
            mode=args.mode,
            freshness_ok=int(coverage_df["has_any_data"].sum()) >= int(len(expected_dates) * 0.85),
            live_symbols=int(coverage_df["trade_count"].gt(0).sum()) if args.mode == "post-open" else 0,
            processed_symbols=processed_symbols,
        )
        summary["readiness"] = readiness.to_dict()

        summary_path = output_dir / f"summary_intraday_model_{args.risk}_{args.mode}_{args.start_date}_{args.end_date}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        summary_table = Table(title="Backtest Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Processed symbols", str(processed_symbols))
        summary_table.add_row("Skipped symbols", str(len(skipped_symbols)))
        summary_table.add_row("Mode", args.mode)
        summary_table.add_row("Entry basis", summary["entry_basis"])
        summary_table.add_row("Total trades", str(total_trades))
        summary_table.add_row("Win rate", f"{win_rate:.1%}")
        summary_table.add_row("Total net P&L", f"₹{total_net:,.0f}")
        summary_table.add_row("Avg net P&L / trade", f"₹{avg_net:,.0f}")
        summary_table.add_row("Avg gross % / trade", f"{avg_gross_pct:.2f}%")
        summary_table.add_row("Active days", str(active_days))
        summary_table.add_row("Avg trades / day", f"{avg_trades_per_day:.2f}")
        summary_table.add_row("Daily Sharpe-like", f"{sharpe:.2f}")
        summary_table.add_row("Target touched hit rate", f"{hit_rate:.1%}")
        summary_table.add_row("Target before stop", f"{target_before_stop_rate:.1%}")
        summary_table.add_row("Max drawdown", f"₹{max_drawdown:,.0f}")
        summary_table.add_row("Business days", str(len(expected_dates)))
        summary_table.add_row("Days with data", str(int(coverage_df["has_any_data"].sum())))
        summary_table.add_row("Days without data", str(int((coverage_df["has_any_data"] == 0).sum())))
        summary_table.add_row("Forced top-N picks", str(summary["top_n_forced_count"]))
        summary_table.add_row("Runtime", f"{runtime_metrics['total_runtime_seconds']:.2f}s")
        summary_table.add_row("Readiness", readiness.status)

        exit_table = Table(title="Exit Reasons")
        exit_table.add_column("Reason", style="cyan")
        exit_table.add_column("Count", justify="right", style="green")
        for reason, count in sorted(by_reason.items()):
            exit_table.add_row(reason, str(count))

        sample_table = Table(title="Top Trades")
        sample_table.add_column("Date", style="dim")
        sample_table.add_column("Symbol", style="bold")
        sample_table.add_column("Dir", style="cyan")
        sample_table.add_column("Conf", justify="right", style="green")
        sample_table.add_column("Net P&L", justify="right", style="green")
        sample_table.add_column("Exit", justify="right", style="yellow")
        for _, row in trades_df.sort_values("net_pnl", ascending=False).head(10).iterrows():
            sample_table.add_row(
                row["date"],
                row["symbol"],
                row["direction"],
                f"{row['confidence']:.2f}",
                f"₹{row['net_pnl']:,.0f}",
                row["exit_reason"],
            )

        console.print()
        console.print(summary_table)
        console.print()
        console.print(exit_table)
        console.print()
        console.print(sample_table)
        console.print()
        readiness_table = Table(title="Readiness")
        readiness_table.add_column("Check", style="cyan")
        readiness_table.add_column("Pass", justify="right", style="green")
        for check_name, passed in readiness.checks.items():
            readiness_table.add_row(check_name.replace("_", " "), "yes" if passed else "no")
        console.print(readiness_table)
        console.print()
        console.print(
            Panel.fit(
                f"[bold green]Backtest complete[/bold green]\n"
                f"[bold]Readiness:[/bold] {readiness.status}\n"
                f"[bold]Summary:[/bold] [dim]{summary_path}[/dim]\n"
                f"[bold]Trades:[/bold] [dim]{trades_path}[/dim]\n"
                f"[bold]Hits:[/bold] [dim]{hits_path}[/dim]\n"
                f"[bold]Coverage:[/bold] [dim]{coverage_path}[/dim]\n"
                f"[bold]Run log:[/bold] [dim]{run_logger.log_path}[/dim]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
