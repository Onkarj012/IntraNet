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
from intradaynet.open_safe_daily_features import build_open_safe_daily_features
from intradaynet.run_logging import command_string, start_run_logging
from intradaynet.universe import get_universe

console = Console()


@dataclass
class Recommendation:
    symbol: str
    direction: str
    data_through_date: str
    picks_for_date: str
    mode: str
    reference_price: float
    previous_close: float
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
    stop_loss_pct: float


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


def extract_post_open_session(
    minute_df: pd.DataFrame,
    target_date: pd.Timestamp,
    cutoff_time: str,
) -> pd.DataFrame:
    session = minute_df[minute_df.index.normalize() == target_date.normalize()].copy()
    if session.empty:
        return session
    session = session.between_time("09:15", cutoff_time)
    return session.sort_index()


def compute_post_open_adjustment(
    *,
    direction: str,
    prev_close: float,
    base_probability: float,
    predicted_magnitude: float,
    session_df: pd.DataFrame,
) -> dict:
    if session_df.empty or prev_close <= 0:
        return {
            "aligned": False,
            "alignment_score": -1.0,
            "adjusted_probability": base_probability,
            "adjusted_magnitude": predicted_magnitude,
            "reference_price": prev_close,
            "session_open": None,
            "live_price": None,
            "gap_pct": None,
            "move_from_open_pct": None,
            "opening_range_pct": None,
        }

    session_open = float(session_df["open"].iloc[0])
    live_price = float(session_df["close"].iloc[-1])
    session_high = float(session_df["high"].max())
    session_low = float(session_df["low"].min())
    gap_pct = (session_open - prev_close) / prev_close
    move_from_open_pct = (live_price - session_open) / max(session_open, 1e-9)
    opening_range_pct = (session_high - session_low) / max(session_open, 1e-9)
    range_width = max(session_high - session_low, 1e-9)

    scale = max(predicted_magnitude, 0.005)
    if direction == "LONG":
        gap_component = np.clip(gap_pct / scale, -1.0, 1.0)
        move_component = np.clip(move_from_open_pct / scale, -1.0, 1.0)
        location_component = np.clip(((live_price - session_low) / range_width) * 2.0 - 1.0, -1.0, 1.0)
    else:
        gap_component = np.clip((-gap_pct) / scale, -1.0, 1.0)
        move_component = np.clip((-move_from_open_pct) / scale, -1.0, 1.0)
        location_component = np.clip(((session_high - live_price) / range_width) * 2.0 - 1.0, -1.0, 1.0)

    alignment_score = float((0.30 * gap_component) + (0.50 * move_component) + (0.20 * location_component))
    adjusted_probability = float(np.clip(base_probability + (0.12 * alignment_score), 0.0, 0.999))
    adjusted_magnitude = float(max(predicted_magnitude * (1.0 + (0.25 * alignment_score)), 0.0))

    return {
        "aligned": True,
        "alignment_score": alignment_score,
        "adjusted_probability": adjusted_probability,
        "adjusted_magnitude": adjusted_magnitude,
        "reference_price": live_price,
        "session_open": session_open,
        "live_price": live_price,
        "gap_pct": gap_pct,
        "move_from_open_pct": move_from_open_pct,
        "opening_range_pct": opening_range_pct,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate morning picks from the open-safe intraday model")
    parser.add_argument("--model", default="models/intraday_model_nifty500.pkl")
    parser.add_argument("--universe", default="nifty500")
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--target-date", default="", help="Trading date to generate picks for (YYYY-MM-DD). Defaults to today/next business day.")
    parser.add_argument("--mode", choices=("premarket", "post-open"), default="premarket")
    parser.add_argument("--post-open-cutoff", default="09:30", help="Use bars up to this market time in post-open mode.")
    parser.add_argument(
        "--post-open-min-alignment",
        type=float,
        default=0.05,
        help="Minimum same-day alignment score required in post-open mode.",
    )
    parser.add_argument("--long-count", type=int, default=3, help="Number of long recommendations to return.")
    parser.add_argument("--short-count", type=int, default=2, help="Number of short recommendations to return.")
    parser.add_argument("--per-side", type=int, default=-1, help="Override both long and short counts with the same number.")
    parser.add_argument("--max-stocks", type=int, default=0, help="Limit number of symbols to score (0 = all).")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-predicted-magnitude", type=float, default=0.01)
    parser.add_argument("--stop-loss-pct", type=float, default=0.01, help="Stop-loss as a fraction, e.g. 0.01 = 1%%.")
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
        "--augment-yf-news",
        dest="augment_yf_news",
        action="store_true",
        default=True,
        help="Augment sentiment data with yfinance news where available. Enabled by default.",
    )
    parser.add_argument(
        "--no-augment-yf-news",
        dest="augment_yf_news",
        action="store_false",
        help="Disable yfinance news augmentation and use only local sentiment data.",
    )
    parser.add_argument("--save-json", type=str, nargs="?", const="default", default="default")
    parser.add_argument("--save-csv", type=str, nargs="?", const="default", default="default")
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
    reference_price: float,
    previous_close: float,
    session_open: float | None,
    live_price: float | None,
    gap_pct: float | None,
    move_from_open_pct: float | None,
    opening_range_pct: float | None,
    confidence: float,
    side_probability: float,
    predicted_magnitude: float,
    stop_loss_pct: float,
) -> Recommendation:
    if direction == "LONG":
        target_price = reference_price * (1 + predicted_magnitude)
        stop_loss_price = reference_price * (1 - stop_loss_pct)
    else:
        target_price = reference_price * (1 - predicted_magnitude)
        stop_loss_price = reference_price * (1 + stop_loss_pct)
    return Recommendation(
        symbol=symbol,
        direction=direction,
        data_through_date=feature_date.strftime("%Y-%m-%d"),
        picks_for_date=picks_for_date.strftime("%Y-%m-%d"),
        mode=mode,
        reference_price=round(reference_price, 2),
        previous_close=round(previous_close, 2),
        session_open=round(session_open, 2) if session_open is not None else None,
        live_price=round(live_price, 2) if live_price is not None else None,
        gap_pct=round(gap_pct, 6) if gap_pct is not None else None,
        move_from_open_pct=round(move_from_open_pct, 6) if move_from_open_pct is not None else None,
        opening_range_pct=round(opening_range_pct, 6) if opening_range_pct is not None else None,
        confidence=round(confidence, 4),
        side_probability=round(side_probability, 4),
        predicted_magnitude=round(predicted_magnitude, 6),
        score=round(confidence * max(predicted_magnitude, 1e-6), 6),
        target_price=round(target_price, 2),
        stop_loss_price=round(stop_loss_price, 2),
        stop_loss_pct=round(stop_loss_pct, 4),
    )


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _render_pick_table(title: str, picks: list[Recommendation], color: str, mode: str) -> None:
    table = Table(title=title, header_style=f"bold {color}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="bold")
    table.add_column("Data Through", style="dim")
    table.add_column("Ref Price", justify="right")
    table.add_column("Target", justify="right", style="green")
    table.add_column("Stop", justify="right", style="red")
    if mode == "post-open":
        table.add_column("Gap", justify="right")
        table.add_column("Move", justify="right")
        table.add_column("Range", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Prob", justify="right")
    table.add_column("Pred %", justify="right", style="cyan")
    table.add_column("Score", justify="right", style="cyan")
    for idx, pick in enumerate(picks, start=1):
        row = [
            str(idx),
            pick.symbol,
            pick.data_through_date,
            f"₹{pick.reference_price:,.2f}",
            f"₹{pick.target_price:,.2f}",
            f"₹{pick.stop_loss_price:,.2f}",
        ]
        if mode == "post-open":
            row.extend(
                [
                    _fmt_pct(pick.gap_pct),
                    _fmt_pct(pick.move_from_open_pct),
                    _fmt_pct(pick.opening_range_pct),
                ]
            )
        row.extend(
            [
                f"{pick.confidence:.2f}",
                f"{pick.side_probability:.2f}",
                f"{pick.predicted_magnitude * 100:.2f}%",
                f"{pick.score:.4f}",
            ]
        )
        table.add_row(*row)
    console.print(table)


def main() -> int:
    global console
    args = parse_args()
    if args.per_side > 0:
        args.long_count = args.per_side
        args.short_count = args.per_side
    if args.long_count < 0 or args.short_count < 0:
        console.print("[red]long-count and short-count must be non-negative.[/red]")
        return 1

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
                f"Longs: {args.long_count} | Shorts: {args.short_count} | Mode: {args.mode}[/dim]",
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
        if args.max_stocks > 0:
            symbols = symbols[:args.max_stocks]

        market_builder = MarketFeatureBuilder()
        market_builder.download(start="2021-01-01", end=market_data_end)

        sentiment_csv = Path("sentiment/combined_sentiment_2015_2025.csv")
        if args.augment_yf_news:
            console.print("[bold]Augmenting sentiment with yfinance news where available...[/bold]")
            augmented_csv = PROJECT_ROOT / "sentiment" / f"combined_sentiment_augmented_for_{picks_for_date.strftime('%Y-%m-%d')}.csv"
            sentiment_csv = augment_sentiment_with_yfinance(
                symbols,
                sentiment_csv,
                backfill_start,
                market_data_end,
                augmented_csv,
            )
        sentiment_builder = SentimentFeatureBuilder(str(sentiment_csv), market_builder=market_builder)
        sentiment_builder._load()

        data_dir = Path(args.data_dir)
        long_candidates: dict[str, Recommendation] = {}
        short_candidates: dict[str, Recommendation] = {}
        processed_symbols = 0
        skipped_symbols: list[str] = []
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
                if minute_df is None:
                    skipped_symbols.append(symbol)
                    progress.advance(task)
                    continue

                feature_df = build_open_safe_daily_features(minute_df, symbol, market_builder, sentiment_builder)
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
                up_mag = float(max(models["up_mag"].predict(X)[0], 0.0))
                down_mag = float(max(models["down_mag"].predict(X)[0], 0.0))
                long_reference_price = last_close
                short_reference_price = last_close
                long_confidence = long_prob
                short_confidence = short_prob
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
                long_session_open = None
                short_session_open = None
                long_live_price = None
                short_live_price = None

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
                        )
                        short_adjustment = compute_post_open_adjustment(
                            direction="SHORT",
                            prev_close=last_close,
                            base_probability=short_prob,
                            predicted_magnitude=down_mag,
                            session_df=session_df,
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
                        long_session_open = long_adjustment["session_open"]
                        short_session_open = short_adjustment["session_open"]
                        long_live_price = long_adjustment["live_price"]
                        short_live_price = short_adjustment["live_price"]
                    else:
                        long_confidence = 0.0
                        short_confidence = 0.0

                long_alignment_ok = True
                short_alignment_ok = True
                if args.mode == "post-open":
                    long_alignment_ok = long_gap_pct is not None and long_alignment_score >= args.post_open_min_alignment
                    short_alignment_ok = short_gap_pct is not None and short_alignment_score >= args.post_open_min_alignment

                if long_confidence >= args.min_confidence and long_mag >= args.min_predicted_magnitude and long_alignment_ok:
                    long_pick = _build_pick(
                        symbol=symbol,
                        direction="LONG",
                        feature_date=feature_date,
                        picks_for_date=picks_for_date,
                        mode=args.mode,
                        reference_price=long_reference_price,
                        previous_close=last_close,
                        session_open=long_session_open,
                        live_price=long_live_price,
                        gap_pct=long_gap_pct,
                        move_from_open_pct=long_move_pct,
                        opening_range_pct=long_range_pct,
                        confidence=long_confidence,
                        side_probability=long_prob,
                        predicted_magnitude=long_mag,
                        stop_loss_pct=args.stop_loss_pct,
                    )
                    long_candidates[symbol] = long_pick

                if short_confidence >= args.min_confidence and short_mag >= args.min_predicted_magnitude and short_alignment_ok:
                    short_pick = _build_pick(
                        symbol=symbol,
                        direction="SHORT",
                        feature_date=feature_date,
                        picks_for_date=picks_for_date,
                        mode=args.mode,
                        reference_price=short_reference_price,
                        previous_close=last_close,
                        session_open=short_session_open,
                        live_price=short_live_price,
                        gap_pct=short_gap_pct,
                        move_from_open_pct=short_move_pct,
                        opening_range_pct=short_range_pct,
                        confidence=short_confidence,
                        side_probability=short_prob,
                        predicted_magnitude=short_mag,
                        stop_loss_pct=args.stop_loss_pct,
                    )
                    short_candidates[symbol] = short_pick
                progress.advance(task)

        # Prevent contradictory picks by keeping each symbol on its stronger side.
        overlap = set(long_candidates) & set(short_candidates)
        for symbol in overlap:
            if long_candidates[symbol].score >= short_candidates[symbol].score:
                short_candidates.pop(symbol, None)
            else:
                long_candidates.pop(symbol, None)

        top_longs = sorted(long_candidates.values(), key=lambda item: item.score, reverse=True)[: args.long_count]
        top_shorts = sorted(short_candidates.values(), key=lambda item: item.score, reverse=True)[: args.short_count]
        all_picks = top_longs + top_shorts
        latest_data_date = max(latest_data_counts) if latest_data_counts else None
        symbols_on_latest_date = latest_data_counts.get(latest_data_date, 0) if latest_data_date is not None else 0

        summary_table = Table(title="Morning Recommendation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Processed symbols", str(processed_symbols))
        summary_table.add_row("Skipped symbols", str(len(skipped_symbols)))
        summary_table.add_row("Qualified longs", str(len(long_candidates)))
        summary_table.add_row("Qualified shorts", str(len(short_candidates)))
        summary_table.add_row("Returned longs", str(len(top_longs)))
        summary_table.add_row("Returned shorts", str(len(top_shorts)))
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
        console.print()
        console.print(summary_table)
        console.print()
        if args.mode == "post-open":
            console.print(
                Panel.fit(
                    "[bold green]Gap-aware mode[/bold green]\n"
                    "These picks use the previous-session model as the base signal, then rerank it with today's real gap and live opening-session bars up to the cutoff time.",
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
            console.print("[yellow]No long picks passed the current filters.[/yellow]\n")

        if top_shorts:
            _render_pick_table(f"Top {len(top_shorts)} Short Picks", top_shorts, "red", args.mode)
            console.print()
        else:
            console.print("[yellow]No short picks passed the current filters.[/yellow]\n")

        if not all_picks:
            console.print("[bold red]No recommendations generated with the current thresholds.[/bold red]")
            return 1

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "picks_for_date": picks_for_date.strftime("%Y-%m-%d"),
            "model": str(model_path),
            "universe": args.universe,
            "data_dir": str(data_dir),
            "min_confidence": args.min_confidence,
            "min_predicted_magnitude": args.min_predicted_magnitude,
            "stop_loss_pct": args.stop_loss_pct,
            "processed_symbols": processed_symbols,
            "skipped_symbols": skipped_symbols,
            "latest_completed_data_date": latest_data_date.strftime("%Y-%m-%d") if latest_data_date is not None else None,
            "symbols_on_latest_date": symbols_on_latest_date,
            "recommendation_mode": args.mode,
            "same_day_gap_aware": args.mode == "post-open",
            "post_open_cutoff": args.post_open_cutoff if args.mode == "post-open" else None,
            "post_open_min_alignment": args.post_open_min_alignment if args.mode == "post-open" else None,
            "post_open_symbols_with_live_data": post_open_symbols_with_live_data if args.mode == "post-open" else None,
            "post_open_partial_symbols": post_open_partial_symbols if args.mode == "post-open" else None,
            "counts": {
                "requested_longs": args.long_count,
                "requested_shorts": args.short_count,
                "returned_longs": len(top_longs),
                "returned_shorts": len(top_shorts),
            },
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
