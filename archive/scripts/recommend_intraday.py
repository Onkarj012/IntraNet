#!/usr/bin/env python3
"""
Generate morning long/short recommendations from the open-safe intraday model.

This CLI is designed for pre-market personal use. It scores the latest
available daily open-safe feature row for each symbol, ranks long and short
candidates separately, and saves a timestamped report.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

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
    DEFAULT_STRATEGY_PROFILES,
    compute_trade_levels,
    default_readiness_paths,
    evaluate_readiness,
    executable_edge_from_prediction,
    feature_staleness_bdays,
    load_json_if_exists,
    margin_adjusted_confidence,
    score_candidate,
    select_candidates,
)
from intradaynet.v7_modes import (
    RuntimeTracker,
    compute_post_open_adjustment,
    compute_preferred_filter_pass,
    extract_post_open_session,
    session_cutoff_timestamp,
)

console = Console()


@dataclass
class Recommendation:
    symbol: str
    direction: str
    data_through_date: str
    picks_for_date: str
    mode: str
    entry_basis: str
    entry_price: float
    reference_price: float
    previous_close: float
    cutoff_close: float | None
    cutoff_timestamp: str | None
    session_open: float | None
    live_price: float | None
    gap_pct: float | None
    move_from_open_pct: float | None
    opening_range_pct: float | None
    confidence: float
    side_probability: float
    predicted_magnitude: float
    score: float
    target_price: float
    stop_loss_price: float
    target_pct: float
    stop_loss_pct: float
    preferred_filter_pass: bool


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
    score = sum(1 for word in words if word in positive) - sum(1 for word in words if word in negative)
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
            rows.append(
                {
                    "Symbol": symbol,
                    "Publish Date": ts.isoformat(sep=" "),
                    "sentiment_score": _infer_title_sentiment(item.get("title", "") or ""),
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


def maybe_refresh_target_session_with_yfinance(
    minute_df: pd.DataFrame | None,
    symbol: str,
    target_date: pd.Timestamp,
    refresh: bool,
) -> pd.DataFrame | None:
    if minute_df is None:
        minute_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if not refresh:
        return minute_df if not minute_df.empty else None

    import yfinance as yf

    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        intraday = ticker.history(
            start=target_date.strftime("%Y-%m-%d"),
            end=(target_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1m",
            auto_adjust=True,
            prepost=False,
        )
    except Exception:
        return minute_df if not minute_df.empty else None

    if intraday.empty:
        return minute_df if not minute_df.empty else None

    intraday = intraday.reset_index()
    date_col = "Datetime" if "Datetime" in intraday.columns else intraday.columns[0]
    intraday[date_col] = pd.to_datetime(intraday[date_col]).dt.tz_localize(None)
    intraday = intraday.rename(
        columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    intraday = intraday[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()
    merged = pd.concat([minute_df, intraday]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def load_minute_data(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.columns = df.columns.str.lower()
    return df.sort_index()


def next_business_day(date_value: pd.Timestamp) -> pd.Timestamp:
    dt = date_value.normalize() + pd.Timedelta(days=1)
    while dt.weekday() >= 5:
        dt += pd.Timedelta(days=1)
    return dt


def default_target_date() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    dt = now.normalize()
    while dt.weekday() >= 5:
        dt += pd.Timedelta(days=1)
    return dt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate morning picks from the open-safe intraday model")
    parser.add_argument("--model", default="results/models/models/intraday_model_nifty500.pkl")
    parser.add_argument("--universe", default="nifty500")
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--target-date", default="", help="Trading date to generate picks for (YYYY-MM-DD). Defaults to today/next business day.")
    parser.add_argument("--mode", choices=("premarket", "post-open"), default="premarket")
    parser.add_argument("--post-open-cutoff", default="09:30", help="Use bars up to this market time in post-open mode.")
    parser.add_argument(
        "--post-open-news-cutoff",
        default="09:20",
        help="Latest news timestamp to include for post-open runs (HH:MM).",
    )
    parser.add_argument(
        "--post-open-min-alignment",
        type=float,
        default=0.05,
        help="Minimum same-day alignment score required in post-open mode.",
    )
    parser.add_argument("--risk-profile", choices=sorted(DEFAULT_STRATEGY_PROFILES.keys()), default="balanced")
    parser.add_argument("--long-count", type=int, default=3, help="Number of long recommendations to return.")
    parser.add_argument("--short-count", type=int, default=2, help="Number of short recommendations to return.")
    parser.add_argument("--per-side", type=int, default=-1, help="Override both long and short counts with the same number.")
    parser.add_argument("--max-stocks", type=int, default=0, help="Limit number of symbols to score (0 = all).")
    parser.add_argument(
        "--industry",
        action="append",
        default=[],
        help="Filter the recommendation universe to one or more CSV industry values. Repeat or use comma-separated values.",
    )
    parser.add_argument(
        "--allow-below-preferred",
        action="store_true",
        help="Backfill requested slots with below-threshold names after exhausting preferred picks.",
    )
    parser.add_argument(
        "--max-feature-staleness-bdays",
        type=int,
        default=0,
        help="Maximum business-day lag allowed between the latest symbol feature row and the expected pre-pick feature date.",
    )
    parser.add_argument("--min-confidence", type=float, default=-1.0)
    parser.add_argument("--min-predicted-magnitude", type=float, default=-1.0)
    parser.add_argument("--target-pct", type=float, default=-1.0, help="Executable target as a fraction. Defaults to the risk profile target.")
    parser.add_argument("--stop-loss-pct", type=float, default=-1.0, help="Stop-loss as a fraction. Defaults to the risk profile stop.")
    parser.add_argument(
        "--refresh-yfinance",
        dest="refresh_yfinance",
        action="store_true",
        default=True,
        help="Backfill missing latest daily bars from yfinance. Enabled by default.",
    )
    parser.add_argument(
        "--no-refresh-yfinance",
        dest="refresh_yfinance",
        action="store_false",
        help="Disable yfinance price backfill and use only local market data.",
    )
    parser.add_argument(
        "--disable-live-news",
        action="store_true",
        help="Disable live yfinance news and rely on historical/fallback sentiment only.",
    )
    parser.add_argument(
        "--augment-yf-news",
        dest="disable_live_news",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-augment-yf-news",
        dest="disable_live_news",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--live-news-required",
        action="store_true",
        help="Fail closed when live news coverage is too low for the filtered universe.",
    )
    parser.add_argument("--save-json", type=str, nargs="?", const="default", default="default")
    parser.add_argument("--save-csv", type=str, nargs="?", const="default", default="default")
    parser.add_argument(
        "--save-cache",
        type=str,
        nargs="?",
        const="default",
        default="",
        help="Cache all per-symbol raw predictions as JSON for fast post-open reuse.",
    )
    parser.add_argument(
        "--from-cache",
        type=str,
        default="",
        help="Load pre-scored predictions from a cache file instead of running full inference.",
    )
    return parser.parse_args()


def _resolve_output_path(raw_value: str, *, suffix: str, stem_prefix: str, target_date: str) -> Path | None:
    if not raw_value or raw_value.lower() == "none":
        return None
    if raw_value == "default":
        out_dir = PROJECT_ROOT / "recommendations"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return out_dir / f"{stem_prefix}_{target_date}_{timestamp}.{suffix}"
    path = Path(raw_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_pick(
    *,
    symbol: str,
    direction: str,
    feature_date: pd.Timestamp,
    picks_for_date: pd.Timestamp,
    mode: str,
    entry_basis: str,
    reference_price: float,
    previous_close: float,
    cutoff_close: float | None,
    cutoff_timestamp: str | None,
    session_open: float | None,
    live_price: float | None,
    gap_pct: float | None,
    move_from_open_pct: float | None,
    opening_range_pct: float | None,
    confidence: float,
    side_probability: float,
    predicted_magnitude: float,
    target_pct: float,
    stop_loss_pct: float,
    preferred_filter_pass: bool,
) -> Recommendation:
    target_price, stop_loss_price = compute_trade_levels(
        reference_price=reference_price,
        direction=direction,
        target_pct=target_pct,
        stop_loss_pct=stop_loss_pct,
    )
    return Recommendation(
        symbol=symbol,
        direction=direction,
        data_through_date=feature_date.strftime("%Y-%m-%d"),
        picks_for_date=picks_for_date.strftime("%Y-%m-%d"),
        mode=mode,
        entry_basis=entry_basis,
        entry_price=round(reference_price, 2),
        reference_price=round(reference_price, 2),
        previous_close=round(previous_close, 2),
        cutoff_close=round(cutoff_close, 2) if cutoff_close is not None else None,
        cutoff_timestamp=cutoff_timestamp,
        session_open=round(session_open, 2) if session_open is not None else None,
        live_price=round(live_price, 2) if live_price is not None else None,
        gap_pct=round(gap_pct, 6) if gap_pct is not None else None,
        move_from_open_pct=round(move_from_open_pct, 6) if move_from_open_pct is not None else None,
        opening_range_pct=round(opening_range_pct, 6) if opening_range_pct is not None else None,
        confidence=round(confidence, 4),
        side_probability=round(side_probability, 4),
        predicted_magnitude=round(predicted_magnitude, 6),
        score=round(score_candidate(confidence, predicted_magnitude), 6),
        target_price=round(target_price, 2),
        stop_loss_price=round(stop_loss_price, 2),
        target_pct=round(target_pct, 4),
        stop_loss_pct=round(stop_loss_pct, 4),
        preferred_filter_pass=preferred_filter_pass,
    )


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _render_pick_table(title: str, picks: list[Recommendation], color: str, mode: str) -> None:
    table = Table(title=title, header_style=f"bold {color}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="bold")
    table.add_column("Prev Close", justify="right")
    if mode == "post-open":
        table.add_column("Cutoff Close", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Target", justify="right", style="green")
    table.add_column("Stop", justify="right", style="red")
    table.add_column("Pref", justify="center")
    table.add_column("Conf", justify="right")
    table.add_column("Score", justify="right", style="cyan")
    for idx, pick in enumerate(picks, start=1):
        row = [
            str(idx),
            pick.symbol,
            f"₹{pick.previous_close:,.2f}",
        ]
        if mode == "post-open":
            row.append(f"₹{(pick.cutoff_close if pick.cutoff_close is not None else pick.reference_price):,.2f}")
        row.extend(
            [
                f"₹{pick.entry_price:,.2f}",
                f"₹{pick.target_price:,.2f}",
                f"₹{pick.stop_loss_price:,.2f}",
                "yes" if pick.preferred_filter_pass else "no",
                f"{pick.confidence:.2f}",
                f"{pick.score:.4f}",
            ]
        )
        table.add_row(*row)
    console.print(table)


def main() -> int:
    global console
    args = parse_args()
    started_at = perf_counter()
    runtime_tracker = RuntimeTracker()
    if args.per_side > 0:
        args.long_count = args.per_side
        args.short_count = args.per_side
    if args.long_count < 0 or args.short_count < 0:
        console.print("[red]long-count and short-count must be non-negative.[/red]")
        return 1
    strategy = DEFAULT_STRATEGY_PROFILES[args.risk_profile]
    target_pct = args.target_pct if args.target_pct > 0 else strategy.target_pct
    stop_loss_pct = args.stop_loss_pct if args.stop_loss_pct > 0 else strategy.stop_loss_pct
    min_confidence = args.min_confidence if args.min_confidence >= 0 else strategy.min_confidence
    min_predicted_magnitude = (
        args.min_predicted_magnitude if args.min_predicted_magnitude >= 0 else strategy.min_predicted_magnitude
    )

    picks_for_date = pd.Timestamp(args.target_date) if args.target_date else default_target_date()
    backfill_start = (picks_for_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    backfill_end = (picks_for_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    market_data_end = picks_for_date.strftime("%Y-%m-%d") if args.mode == "post-open" else backfill_end
    run_name = f"recommend_intraday_{args.universe}_{picks_for_date.strftime('%Y%m%d')}"

    with start_run_logging(project_root=PROJECT_ROOT, log_group="recommendations", run_name=run_name) as run_logger:
        console = Console()
        console.print(
            Panel.fit(
                "[bold cyan]IntradayNet Morning Recommender[/bold cyan]\n"
                f"[dim]Universe: {args.universe} | Picks for: {picks_for_date.strftime('%Y-%m-%d')} | "
                f"Longs: {args.long_count} | Shorts: {args.short_count} | Mode: {args.mode}[/dim]\n"
                f"[dim]Profile: {strategy.name} | Target: {target_pct*100:.2f}% | Stop: {stop_loss_pct*100:.2f}%[/dim]",
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

        symbols = get_universe(args.universe)
        resolved_industries = resolve_industry_filters(args.industry) if args.industry else []
        symbols = filter_symbols_by_industry(symbols, resolved_industries)
        if args.max_stocks > 0:
            symbols = symbols[:args.max_stocks]
        if not symbols:
            console.print("[red]No symbols matched the selected universe / industry filters.[/red]")
            return 1

        market_builder = MarketFeatureBuilder()
        runtime_tracker.start("data_refresh_seconds")
        market_builder.download(start="2021-01-01", end=market_data_end)

        sentiment_csv = Path("data/sentiment/combined_sentiment_2015_2025.csv")
        historical_articles = normalize_historical_sentiment_csv(
            sentiment_csv,
            universe_metadata_csv=str(PROJECT_ROOT / "data" / "sentiment" / "ind_nifty500list.csv"),
            market_open_cutoff="09:15",
            post_open_cutoff=args.post_open_news_cutoff,
        )
        live_articles = pd.DataFrame()
        news_summary = LiveNewsSummary()
        if not args.disable_live_news:
            console.print("[bold]Fetching live yfinance news and merging it with historical fallback...[/bold]")
            live_end_ts = picks_for_date + pd.Timedelta(hours=9, minutes=15)
            if args.mode == "post-open":
                today_now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
                cutoff_today = pd.Timestamp(f"{picks_for_date.strftime('%Y-%m-%d')} {args.post_open_news_cutoff}")
                live_end_ts = min(today_now, cutoff_today)
            live_articles, news_summary = fetch_live_yfinance_news(
                symbols,
                start_ts=pd.Timestamp(backfill_start),
                end_ts=live_end_ts,
                universe_metadata_csv=str(PROJECT_ROOT / "data" / "sentiment" / "ind_nifty500list.csv"),
                market_open_cutoff="09:15",
                post_open_cutoff=args.post_open_news_cutoff,
            )
        combined_articles = combine_article_sources(historical_articles, live_articles)
        news_summary = summarize_article_coverage(
            combined_articles,
            symbols=symbols,
            mode=args.mode,
            target_dates=[picks_for_date],
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
        sector_index_coverage = 0

        data_dir = Path(args.data_dir)
        long_candidates: dict[str, Recommendation] = {}
        short_candidates: dict[str, Recommendation] = {}
        processed_symbols = 0
        skipped_symbols: list[str] = []
        stale_symbols: list[str] = []
        latest_data_counts: dict[pd.Timestamp, int] = {}
        post_open_symbols_with_live_data = 0
        post_open_partial_symbols = 0

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
                    backfill_start,
                    backfill_end,
                    args.refresh_yfinance,
                )
                if args.mode == "post-open":
                    minute_df = maybe_refresh_target_session_with_yfinance(
                        minute_df,
                        symbol,
                        picks_for_date,
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

                feature_df = feature_df[feature_df.index < picks_for_date]
                if feature_df.empty:
                    skipped_symbols.append(symbol)
                    progress.advance(task)
                    continue

                last_row = feature_df.iloc[[-1]]
                feature_date = last_row.index[-1]
                if float(last_row.iloc[0].get("sector_index_prev_return", 0.0)) != 0.0:
                    sector_index_coverage += 1
                feature_age_bdays = feature_staleness_bdays(feature_date, picks_for_date)
                if feature_age_bdays > args.max_feature_staleness_bdays:
                    stale_symbols.append(symbol)
                    progress.advance(task)
                    continue
                latest_data_counts[feature_date.normalize()] = latest_data_counts.get(feature_date.normalize(), 0) + 1
                minute_hist = minute_df[minute_df.index.normalize() == feature_date.normalize()]
                if minute_hist.empty:
                    skipped_symbols.append(symbol)
                    progress.advance(task)
                    continue

                last_close = float(minute_hist["close"].iloc[-1])
                processed_symbols += 1
                X = last_row.reindex(columns=feature_cols, fill_value=0.0)
                long_prob = float(models["long"].predict_proba(X)[:, 1][0])
                short_prob = float(models["short"].predict_proba(X)[:, 1][0])
                up_mag = executable_edge_from_prediction(
                    float(max(models["up_mag"].predict(X)[0], 0.0)),
                    target_pct=target_pct,
                )
                down_mag = executable_edge_from_prediction(
                    float(max(models["down_mag"].predict(X)[0], 0.0)),
                    target_pct=target_pct,
                )
                long_reference_price = last_close
                short_reference_price = last_close
                long_confidence = margin_adjusted_confidence(long_prob, short_prob)
                short_confidence = margin_adjusted_confidence(short_prob, long_prob)
                long_mag = up_mag
                short_mag = down_mag
                long_alignment_score = 0.0
                short_alignment_score = 0.0
                long_gap_pct = None
                short_gap_pct = None
                long_move_pct = None
                short_move_pct = None
                long_range_pct = None
                short_range_pct = None
                long_cutoff_close = None
                short_cutoff_close = None
                long_cutoff_timestamp = None
                short_cutoff_timestamp = None
                long_session_open = None
                short_session_open = None
                long_live_price = None
                short_live_price = None
                row_context = last_row.iloc[0]
                risk_on_signal = float(row_context.get("risk_on_signal", 0.0))
                market_breadth = float(row_context.get("market_breadth", 0.0))
                sector_relative_strength = float(row_context.get("sector_relative_strength", 0.0))

                if args.mode == "post-open":
                    session_df = extract_post_open_session(minute_df, picks_for_date, args.post_open_cutoff)
                    if not session_df.empty:
                        post_open_symbols_with_live_data += 1
                        if session_df.index.max().strftime("%H:%M") < args.post_open_cutoff:
                            post_open_partial_symbols += 1

                        long_adjustment = compute_post_open_adjustment(
                            direction="LONG",
                            prev_close=last_close,
                            base_probability=long_prob,
                            predicted_magnitude=up_mag,
                            session_df=session_df,
                            minute_df=minute_df,
                            feature_row=row_context,
                        )
                        short_adjustment = compute_post_open_adjustment(
                            direction="SHORT",
                            prev_close=last_close,
                            base_probability=short_prob,
                            predicted_magnitude=down_mag,
                            session_df=session_df,
                            minute_df=minute_df,
                            feature_row=row_context,
                        )
                        long_confidence = long_adjustment["adjusted_probability"]
                        short_confidence = short_adjustment["adjusted_probability"]
                        long_mag = long_adjustment["adjusted_magnitude"]
                        short_mag = short_adjustment["adjusted_magnitude"]
                        long_alignment_score = float(long_adjustment["alignment_score"])
                        short_alignment_score = float(short_adjustment["alignment_score"])
                        long_reference_price = float(long_adjustment["reference_price"])
                        short_reference_price = float(short_adjustment["reference_price"])
                        long_gap_pct = long_adjustment["gap_pct"]
                        short_gap_pct = short_adjustment["gap_pct"]
                        long_move_pct = long_adjustment["move_from_open_pct"]
                        short_move_pct = short_adjustment["move_from_open_pct"]
                        long_range_pct = long_adjustment["opening_range_pct"]
                        short_range_pct = short_adjustment["opening_range_pct"]
                        long_cutoff_close = long_adjustment["cutoff_close"]
                        short_cutoff_close = short_adjustment["cutoff_close"]
                        long_cutoff_timestamp = long_adjustment["cutoff_timestamp"]
                        short_cutoff_timestamp = short_adjustment["cutoff_timestamp"]
                        long_session_open = long_adjustment["session_open"]
                        short_session_open = short_adjustment["session_open"]
                        long_live_price = long_adjustment["live_price"]
                        short_live_price = short_adjustment["live_price"]
                    else:
                        progress.advance(task)
                        continue

                long_alignment_ok = True
                short_alignment_ok = True
                long_regime_ok = not (risk_on_signal < -0.25 and market_breadth < -0.02 and sector_relative_strength < -0.01)
                short_regime_ok = not (risk_on_signal > 0.25 and market_breadth > 0.02 and sector_relative_strength > 0.01)
                if args.mode == "post-open":
                    long_alignment_ok = long_gap_pct is not None and long_alignment_score >= args.post_open_min_alignment
                    short_alignment_ok = short_gap_pct is not None and short_alignment_score >= args.post_open_min_alignment

                long_passes_preferred_filters = compute_preferred_filter_pass(
                    confidence=long_confidence,
                    predicted_magnitude=long_mag,
                    min_confidence=min_confidence,
                    min_predicted_magnitude=min_predicted_magnitude,
                    alignment_ok=long_alignment_ok,
                    regime_ok=long_regime_ok,
                )
                short_passes_preferred_filters = compute_preferred_filter_pass(
                    confidence=short_confidence,
                    predicted_magnitude=short_mag,
                    min_confidence=min_confidence,
                    min_predicted_magnitude=min_predicted_magnitude,
                    alignment_ok=short_alignment_ok,
                    regime_ok=short_regime_ok,
                )

                runtime_tracker.start("ranking_seconds")
                long_pick = _build_pick(
                    symbol=symbol,
                    direction="LONG",
                    feature_date=feature_date,
                    picks_for_date=picks_for_date,
                    mode=args.mode,
                    entry_basis="Cutoff Close" if args.mode == "post-open" else "Previous Close",
                    reference_price=long_reference_price,
                    previous_close=last_close,
                    cutoff_close=long_cutoff_close,
                    cutoff_timestamp=long_cutoff_timestamp,
                    session_open=long_session_open,
                    live_price=long_live_price,
                    gap_pct=long_gap_pct,
                    move_from_open_pct=long_move_pct,
                    opening_range_pct=long_range_pct,
                    confidence=long_confidence,
                    side_probability=long_prob,
                    predicted_magnitude=long_mag,
                    target_pct=target_pct,
                    stop_loss_pct=stop_loss_pct,
                    preferred_filter_pass=long_passes_preferred_filters,
                )
                long_candidates[symbol] = long_pick

                short_pick = _build_pick(
                    symbol=symbol,
                    direction="SHORT",
                    feature_date=feature_date,
                    picks_for_date=picks_for_date,
                    mode=args.mode,
                    entry_basis="Cutoff Close" if args.mode == "post-open" else "Previous Close",
                    reference_price=short_reference_price,
                    previous_close=last_close,
                    cutoff_close=short_cutoff_close,
                    cutoff_timestamp=short_cutoff_timestamp,
                    session_open=short_session_open,
                    live_price=short_live_price,
                    gap_pct=short_gap_pct,
                    move_from_open_pct=short_move_pct,
                    opening_range_pct=short_range_pct,
                    confidence=short_confidence,
                    side_probability=short_prob,
                    predicted_magnitude=short_mag,
                    target_pct=target_pct,
                    stop_loss_pct=stop_loss_pct,
                    preferred_filter_pass=short_passes_preferred_filters,
                )
                short_candidates[symbol] = short_pick
                runtime_tracker.stop("ranking_seconds")
                progress.advance(task)

        # Prevent contradictory picks by keeping each symbol on its stronger side.
        overlap = set(long_candidates) & set(short_candidates)
        for symbol in overlap:
            if long_candidates[symbol].score >= short_candidates[symbol].score:
                short_candidates.pop(symbol, None)
            else:
                long_candidates.pop(symbol, None)

        top_longs = select_candidates(
            list(long_candidates.values()),
            count=args.long_count,
            allow_below_preferred=args.allow_below_preferred,
        )
        top_shorts = select_candidates(
            list(short_candidates.values()),
            count=args.short_count,
            allow_below_preferred=args.allow_below_preferred,
        )
        all_picks = top_longs + top_shorts
        latest_data_date = max(latest_data_counts) if latest_data_counts else None
        symbols_on_latest_date = latest_data_counts.get(latest_data_date, 0) if latest_data_date is not None else 0
        expected_data_date = (picks_for_date - pd.offsets.BDay(1)).normalize()
        freshness_ok = latest_data_date is not None and latest_data_date.normalize() >= expected_data_date

        summary_table = Table(title="Morning Recommendation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Processed symbols", str(processed_symbols))
        summary_table.add_row("Industry filter", ", ".join(resolved_industries) if resolved_industries else "All")
        summary_table.add_row("Skipped symbols", str(len(skipped_symbols)))
        summary_table.add_row("Stale symbols filtered", str(len(stale_symbols)))
        summary_table.add_row("Live article count kept", str(news_summary.live_article_count_kept))
        summary_table.add_row("Historical article rows used", str(news_summary.historical_article_count_used))
        summary_table.add_row("Stock news coverage", str(news_summary.stock_news_coverage))
        summary_table.add_row("Industry news coverage", str(news_summary.industry_news_coverage))
        summary_table.add_row("Live source failures", str(news_summary.live_source_failure_count))
        summary_table.add_row("Fallback-used symbols", str(news_summary.fallback_used_count))
        summary_table.add_row("Sector index coverage", str(sector_index_coverage))
        summary_table.add_row(
            "Above preferred longs",
            str(sum(1 for pick in long_candidates.values() if pick.preferred_filter_pass)),
        )
        summary_table.add_row(
            "Above preferred shorts",
            str(sum(1 for pick in short_candidates.values() if pick.preferred_filter_pass)),
        )
        summary_table.add_row("Ranked longs", str(len(long_candidates)))
        summary_table.add_row("Ranked shorts", str(len(short_candidates)))
        summary_table.add_row("Returned longs", str(len(top_longs)))
        summary_table.add_row("Returned shorts", str(len(top_shorts)))
        summary_table.add_row("Allow below preferred", "yes" if args.allow_below_preferred else "no")
        summary_table.add_row("Strategy target", f"{target_pct*100:.2f}%")
        summary_table.add_row("Strategy stop", f"{stop_loss_pct*100:.2f}%")
        summary_table.add_row(
            "Latest completed data date",
            latest_data_date.strftime("%Y-%m-%d") if latest_data_date is not None else "N/A",
        )
        summary_table.add_row(
            "Symbols on latest date",
            str(symbols_on_latest_date),
        )
        summary_table.add_row("Recommendation mode", args.mode)
        summary_table.add_row("Same-day gap aware", "yes" if args.mode == "post-open" else "no")
        if args.mode == "post-open":
            summary_table.add_row("Post-open cutoff", args.post_open_cutoff)
            summary_table.add_row("Live symbols", str(post_open_symbols_with_live_data))
            summary_table.add_row("Partial live symbols", str(post_open_partial_symbols))
            cutoff_times = sorted({pick.cutoff_timestamp for pick in all_picks if pick.cutoff_timestamp})
            summary_table.add_row("Latest cutoff seen", cutoff_times[-1] if cutoff_times else "N/A")
        summary_table.add_row("Freshness OK", "yes" if freshness_ok else "no")
        runtime_metrics = runtime_tracker.snapshot(perf_counter() - started_at)
        summary_table.add_row("Runtime", f"{runtime_metrics['total_runtime_seconds']:.2f}s")
        console.print()
        console.print(summary_table)
        console.print()
        readiness_paths = default_readiness_paths(PROJECT_ROOT, mode=args.mode)
        readiness = evaluate_readiness(
            locked_backtest_summary=load_json_if_exists(readiness_paths["locked_backtest"]),
            forward_summary=load_json_if_exists(readiness_paths["forward_blind"]),
            target_alignment=True,
            mode=args.mode,
            freshness_ok=freshness_ok,
            live_symbols=post_open_symbols_with_live_data,
            processed_symbols=processed_symbols,
        )
        readiness_reason_text = "\n".join(f"- {reason}" for reason in readiness.reasons[:4]) or "- All readiness checks passed."
        readiness_style = "green" if readiness.status == "READY" else ("yellow" if readiness.status in {"PAPER_ONLY", "SMALL_LIVE"} else "red")
        console.print(
            Panel.fit(
                f"[bold {readiness_style}]Readiness: {readiness.status}[/bold {readiness_style}]\n{readiness_reason_text}",
                border_style=readiness_style,
            )
        )
        console.print()
        if args.allow_below_preferred and any(not pick.preferred_filter_pass for pick in all_picks):
            console.print(
                Panel.fit(
                    "[bold yellow]Ranking mode[/bold yellow]\n"
                    "The CLI backfilled your requested slots with below-threshold names after exhausting the preferred set.\n"
                    "Use the [bold]Pref[/bold] column plus [bold]Conf[/bold] to decide whether you want to act on them.",
                    border_style="yellow",
                )
            )
            console.print()
        if not freshness_ok:
            console.print("[bold red]Fresh data coverage is not sufficient for honest recommendations.[/bold red]")
            return 1
        if args.live_news_required:
            minimum_symbols = max(1, int(len(symbols) * 0.10))
            if news_summary.symbols_with_live_news < minimum_symbols:
                console.print("[bold red]Live news coverage was below the required threshold, so the run is failing closed.[/bold red]")
                return 1
        if args.mode == "post-open" and post_open_symbols_with_live_data == 0:
            console.print("[bold red]No live post-open bars were available, so V7 is failing closed instead of inventing picks.[/bold red]")
            return 1
        if args.mode == "post-open":
            console.print(
                Panel.fit(
                    "[bold green]Gap-aware mode[/bold green]\n"
                    "These picks use the previous-session model as the base signal, then rerank it with today's real gap, early range, relative volume, VWAP displacement, and market confirmation up to the cutoff time.",
                    border_style="green",
                )
            )
            if post_open_symbols_with_live_data == 0:
                console.print(
                    Panel.fit(
                        "[bold yellow]No live bars detected[/bold yellow]\n"
                        "No target-date intraday bars were available up to the requested cutoff.\n"
                        "Keep the default yfinance refresh enabled when you run this after 9:15 so the system can fetch today's opening session.",
                        border_style="yellow",
                    )
                )
        else:
            console.print(
                Panel.fit(
                    "[bold yellow]Freshness note[/bold yellow]\n"
                    "These pre-market picks use the latest completed session plus refreshed macro/news inputs.\n"
                    "They do [bold]not[/bold] include the same-day opening gap yet. For true gap-aware picks, use [bold]--mode post-open[/bold] after the first live bars are available.",
                    border_style="yellow",
                )
            )
        console.print()

        if top_longs:
            _render_pick_table(f"Top {len(top_longs)} Long Picks", top_longs, "green", args.mode)
            console.print()
        else:
            console.print("[yellow]No long picks could be ranked for the requested day.[/yellow]\n")

        if top_shorts:
            _render_pick_table(f"Top {len(top_shorts)} Short Picks", top_shorts, "red", args.mode)
            console.print()
        else:
            console.print("[yellow]No short picks could be ranked for the requested day.[/yellow]\n")

        if not all_picks:
            console.print("[bold red]No recommendations could be generated for the requested day.[/bold red]")
            return 1

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "picks_for_date": picks_for_date.strftime("%Y-%m-%d"),
            "model": str(model_path),
            "target_version": model_data.get("target_version"),
            "feature_version": model_data.get("feature_version"),
            "universe": args.universe,
            "industry_filter": resolved_industries,
            "data_dir": str(data_dir),
            "risk_profile": args.risk_profile,
            "allow_below_preferred": args.allow_below_preferred,
            "max_feature_staleness_bdays": args.max_feature_staleness_bdays,
            "min_confidence": min_confidence,
            "min_predicted_magnitude": min_predicted_magnitude,
            "target_pct": target_pct,
            "stop_loss_pct": stop_loss_pct,
            "processed_symbols": processed_symbols,
            "skipped_symbols": skipped_symbols,
            "stale_symbols_filtered": stale_symbols,
            "latest_completed_data_date": latest_data_date.strftime("%Y-%m-%d") if latest_data_date is not None else None,
            "symbols_on_latest_date": symbols_on_latest_date,
            "freshness_ok": freshness_ok,
            "recommendation_mode": args.mode,
            "same_day_gap_aware": args.mode == "post-open",
            "mode": args.mode,
            "entry_basis": "Cutoff Close" if args.mode == "post-open" else "Previous Close",
            "post_open_cutoff": args.post_open_cutoff if args.mode == "post-open" else None,
            "post_open_min_alignment": args.post_open_min_alignment if args.mode == "post-open" else None,
            "post_open_symbols_with_live_data": post_open_symbols_with_live_data if args.mode == "post-open" else None,
            "post_open_partial_symbols": post_open_partial_symbols if args.mode == "post-open" else None,
            "runtime_metrics": runtime_metrics,
            "news_summary": news_summary.to_dict() | {"sector_index_coverage": sector_index_coverage},
            "counts": {
                "requested_longs": args.long_count,
                "requested_shorts": args.short_count,
                "above_preferred_longs": sum(1 for pick in long_candidates.values() if pick.preferred_filter_pass),
                "above_preferred_shorts": sum(1 for pick in short_candidates.values() if pick.preferred_filter_pass),
                "ranked_longs": len(long_candidates),
                "ranked_shorts": len(short_candidates),
                "returned_longs": len(top_longs),
                "returned_shorts": len(top_shorts),
                "forced_vs_preferred_returned": int(sum(1 for pick in all_picks if not pick.preferred_filter_pass)),
            },
            "readiness": readiness.to_dict(),
            "long_picks": [asdict(pick) for pick in top_longs],
            "short_picks": [asdict(pick) for pick in top_shorts],
            "run_log": str(run_logger.log_path),
        }

        json_path = _resolve_output_path(
            args.save_json,
            suffix="json",
            stem_prefix="morning_picks",
            target_date=picks_for_date.strftime("%Y%m%d"),
        )
        csv_path = _resolve_output_path(
            args.save_csv,
            suffix="csv",
            stem_prefix="morning_picks",
            target_date=picks_for_date.strftime("%Y%m%d"),
        )

        if json_path is not None:
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if csv_path is not None:
            pd.DataFrame([asdict(pick) for pick in all_picks]).to_csv(csv_path, index=False)

        console.print(
            Panel.fit(
                f"[bold green]Morning picks ready[/bold green]\n"
                f"[bold]Picks for:[/bold] {picks_for_date.strftime('%Y-%m-%d')}\n"
                f"[bold]Longs:[/bold] {len(top_longs)} | [bold]Shorts:[/bold] {len(top_shorts)}\n"
                f"[bold]JSON:[/bold] [dim]{json_path if json_path else 'disabled'}[/dim]\n"
                f"[bold]CSV:[/bold] [dim]{csv_path if csv_path else 'disabled'}[/dim]\n"
                f"[bold]Run log:[/bold] [dim]{run_logger.log_path}[/dim]",
                border_style="green",
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
