#!/usr/bin/env python3
"""
Fast post-open recommendation pipeline.

Two-phase design:
  P1 (premarket): Score all stocks → cache to JSON  (2-3 min, no time pressure)
  P2 (post-open): Load cache + live bars → adjust → rank  (<10 seconds)

Usage:
  # Generate cache before market opens
  python scripts/post_open_picks.py --mode premarket --universe nifty500

  # Fast post-open picks after 9:15 AM
  python scripts/post_open_picks.py --mode post-open --universe nifty500

  # One-shot: auto-detect mode based on time of day
  python scripts/post_open_picks.py --universe nifty500
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

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
from intradaynet.universe import get_universe
from intradaynet.v7 import (
    DEFAULT_STRATEGY_PROFILES,
    compute_trade_levels,
    executable_edge_from_prediction,
    feature_staleness_bdays,
    margin_adjusted_confidence,
    score_candidate,
    select_candidates,
)
from intradaynet.v7_modes import (
    RuntimeTracker,
    compute_post_open_adjustment,
    compute_preferred_filter_pass,
    extract_post_open_session,
)

console = Console()

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(picks_for_date: str, universe: str) -> Path:
    return CACHE_DIR / f"premarket_cache_{universe}_{picks_for_date}.json"


def cache_exists(picks_for_date: str, universe: str) -> bool:
    return _cache_path(picks_for_date, universe).exists()


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


def load_minute_data(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.columns = df.columns.str.lower()
    return df.sort_index()


def load_minute_data_tail(csv_path: Path, target_date_str: str, before_date: str) -> pd.DataFrame | None:
    """Fast load: only read rows from target_date + last 2 prior days for session adjustment."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.columns = df.columns.str.lower()
    df = df.sort_index()
    mask = (df.index >= before_date) & (df.index <= f"{target_date_str} 23:59:59")
    return df[mask].copy() if not df[mask].empty else None


def default_target_date() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    dt = now.normalize()
    while dt.weekday() >= 5:
        dt += pd.Timedelta(days=1)
    return dt


def maybe_backfill_with_yfinance(
    minute_df: pd.DataFrame | None, symbol: str, start_date: str, end_date: str, refresh: bool,
) -> pd.DataFrame | None:
    if minute_df is None:
        minute_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not refresh:
        return minute_df if not minute_df.empty else None

    import yfinance as yf
    ticker = yf.Ticker(f"{symbol}.NS")
    try:
        daily = ticker.history(
            start=start_date, end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=True,
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
        pseudo_rows.append({
            "date": dt + pd.Timedelta(hours=15, minutes=29),
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0.0)),
        })
    if not pseudo_rows:
        return minute_df if not minute_df.empty else None
    pseudo_df = pd.DataFrame(pseudo_rows).set_index("date").sort_index()
    merged = pd.concat([minute_df, pseudo_df]).sort_index()
    return merged[~merged.index.duplicated(keep="last")]


def build_premarket_cache(
    symbols: list[str],
    data_dir: Path,
    model_path: Path,
    picks_for_date: pd.Timestamp,
    strategy_profile: str,
    universe: str,
) -> Path:
    """Score all stocks with the model and cache predictions to disk."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    models = model_data["models"]
    feature_cols = model_data["features"]
    target_pct = model_data.get("target_pct", 0.015)

    market_builder = MarketFeatureBuilder()
    backfill_end = (picks_for_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    market_builder.download(start="2021-01-01", end=backfill_end)

    sentiment_csv = Path("data/sentiment/combined_sentiment_2015_2025.csv")
    historical_articles = normalize_historical_sentiment_csv(
        sentiment_csv,
        universe_metadata_csv=str(PROJECT_ROOT / "data" / "sentiment" / "ind_nifty500list.csv"),
        market_open_cutoff="09:15",
    )
    combined_articles = combine_article_sources(historical_articles, pd.DataFrame())
    sentiment_builder = SentimentFeatureBuilder(
        str(sentiment_csv),
        market_builder=market_builder,
        articles_df=combined_articles,
    )
    sentiment_builder._load()

    entries = []
    strategy = DEFAULT_STRATEGY_PROFILES[strategy_profile]
    strategy_target_pct = strategy.target_pct if target_pct <= 0 else target_pct
    backfill_start = (picks_for_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    backfill_end_str = (picks_for_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    console.print(f"\n[bold]Building premarket cache for {len(symbols)} symbols...[/bold]")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Scoring symbols", total=len(symbols))
        for symbol in symbols:
            progress.update(task, description=f"Scoring {symbol}")
            csv_path = data_dir / f"{symbol}_minute.csv"
            minute_df = load_minute_data(csv_path)
            minute_df = maybe_backfill_with_yfinance(
                minute_df, symbol, backfill_start, backfill_end_str, True,
            )
            if minute_df is None:
                progress.advance(task)
                continue

            feature_df = build_open_safe_daily_features(minute_df, symbol, market_builder, sentiment_builder)
            if feature_df is None or feature_df.empty:
                progress.advance(task)
                continue

            feature_df = feature_df[feature_df.index < picks_for_date]
            if feature_df.empty:
                progress.advance(task)
                continue

            last_row = feature_df.iloc[[-1]]
            feature_date = last_row.index[-1]
            minute_hist = minute_df[minute_df.index.normalize() == feature_date.normalize()]
            if minute_hist.empty:
                progress.advance(task)
                continue

            last_close = float(minute_hist["close"].iloc[-1])
            X = last_row.reindex(columns=feature_cols, fill_value=0.0)

            long_prob = float(models["long"].predict_proba(X)[:, 1][0])
            short_prob = float(models["short"].predict_proba(X)[:, 1][0])
            up_mag = float(max(models["up_mag"].predict(X)[0], 0.0))
            down_mag = float(max(models["down_mag"].predict(X)[0], 0.0))

            row_context = last_row.iloc[0]
            entries.append({
                "symbol": symbol,
                "long_prob": long_prob,
                "short_prob": short_prob,
                "up_mag": up_mag,
                "down_mag": down_mag,
                "prev_close": last_close,
                "feature_date": feature_date.strftime("%Y-%m-%d"),
                "market_breadth": float(row_context.get("market_breadth", 0.0)),
                "risk_on_signal": float(row_context.get("risk_on_signal", 0.0)),
                "sector_relative_strength": float(row_context.get("sector_relative_strength", 0.0)),
            })
            progress.advance(task)

    cache_path = _cache_path(picks_for_date.strftime("%Y%m%d"), universe)
    payload = {
        "created_at": datetime.now().isoformat(),
        "picks_for_date": picks_for_date.strftime("%Y-%m-%d"),
        "universe": universe,
        "strategy": strategy_profile,
        "model": str(model_path),
        "target_pct": target_pct,
        "count": len(entries),
        "predictions": entries,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[green]Premarket cache saved:[/green] {cache_path} ({len(entries)} symbols)")
    return cache_path


def apply_post_open_adjustments(
    cache_path: Path,
    picks_for_date: pd.Timestamp,
    strategy_profile: str,
    long_count: int,
    short_count: int,
    post_open_cutoff: str,
    post_open_min_alignment: float,
    allow_below_preferred: bool,
    refresh_yfinance: bool,
    data_dir: Path,
) -> dict[str, Any]:
    """Load cached premarket predictions, adjust with live bars, rank and return picks."""
    cache_data = json.loads(cache_path.read_text())
    entries = cache_data["predictions"]
    target_pct = cache_data.get("target_pct", 0.015)

    strategy = DEFAULT_STRATEGY_PROFILES[strategy_profile]
    strategy_target_pct = strategy.target_pct if target_pct <= 0 else target_pct
    stop_loss_pct = strategy.stop_loss_pct
    min_confidence = strategy.min_confidence
    min_predicted_magnitude = strategy.min_predicted_magnitude

    console.print(f"\n[bold]Fetching today's intraday bars for {len(entries)} symbols...[/bold]")
    started = perf_counter()

    long_candidates = {}
    short_candidates = {}
    live_count = 0
    partial_count = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Adjusting predictions", total=len(entries))
        target_date_str = picks_for_date.strftime("%Y-%m-%d")
        prior_date = (picks_for_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

        for entry in entries:
            symbol = entry["symbol"]
            progress.update(task, description=f"Adjusting {symbol}")

            csv_path = data_dir / f"{symbol}_minute.csv"
            minute_df = load_minute_data_tail(csv_path, target_date_str, prior_date)
            if minute_df is None:
                continue

            if refresh_yfinance:
                import yfinance as yf
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    intraday = ticker.history(
                        start=picks_for_date.strftime("%Y-%m-%d"),
                        end=(picks_for_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval="1m", auto_adjust=True, prepost=False,
                    )
                    if not intraday.empty:
                        intraday = intraday.reset_index()
                        date_col = "Datetime" if "Datetime" in intraday.columns else intraday.columns[0]
                        intraday[date_col] = pd.to_datetime(intraday[date_col]).dt.tz_localize(None)
                        intraday = intraday.rename(columns={
                            date_col: "date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume",
                        })
                        intraday = intraday[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()
                        merged = pd.concat([minute_df, intraday]).sort_index()
                        minute_df = merged[~merged.index.duplicated(keep="last")]
                except Exception:
                    pass

            session_df = extract_post_open_session(minute_df, picks_for_date, post_open_cutoff)
            if session_df.empty:
                progress.advance(task)
                continue

            live_count += 1
            if session_df.index.max().strftime("%H:%M") < post_open_cutoff:
                partial_count += 1

            prev_close = entry["prev_close"]
            feature_row = pd.Series({
                "market_breadth": entry["market_breadth"],
                "risk_on_signal": entry["risk_on_signal"],
                "sector_relative_strength": entry["sector_relative_strength"],
            })

            # Long adjustment
            long_adjustment = compute_post_open_adjustment(
                direction="LONG", prev_close=prev_close,
                base_probability=entry["long_prob"],
                predicted_magnitude=entry["up_mag"],
                session_df=session_df, minute_df=minute_df,
                feature_row=feature_row,
            )

            # Short adjustment
            short_adjustment = compute_post_open_adjustment(
                direction="SHORT", prev_close=prev_close,
                base_probability=entry["short_prob"],
                predicted_magnitude=entry["down_mag"],
                session_df=session_df, minute_df=minute_df,
                feature_row=feature_row,
            )

            long_conf = long_adjustment["adjusted_probability"]
            short_conf = short_adjustment["adjusted_probability"]
            long_mag = long_adjustment["adjusted_magnitude"]
            short_mag = short_adjustment["adjusted_magnitude"]
            long_align = float(long_adjustment["alignment_score"])
            short_align = float(short_adjustment["alignment_score"])
            long_ref = float(long_adjustment["reference_price"])
            short_ref = float(short_adjustment["reference_price"])

            long_alignment_ok = long_align >= post_open_min_alignment
            short_alignment_ok = short_align >= post_open_min_alignment

            risk_on = entry["risk_on_signal"]
            breadth = entry["market_breadth"]
            sector = entry["sector_relative_strength"]
            long_regime_ok = not (risk_on < -0.25 and breadth < -0.02 and sector < -0.01)
            short_regime_ok = not (risk_on > 0.25 and breadth > 0.02 and sector > 0.01)

            long_passes = compute_preferred_filter_pass(
                confidence=long_conf, predicted_magnitude=long_mag,
                min_confidence=min_confidence, min_predicted_magnitude=min_predicted_magnitude,
                alignment_ok=long_alignment_ok, regime_ok=long_regime_ok,
            )
            short_passes = compute_preferred_filter_pass(
                confidence=short_conf, predicted_magnitude=short_mag,
                min_confidence=min_confidence, min_predicted_magnitude=min_predicted_magnitude,
                alignment_ok=short_alignment_ok, regime_ok=short_regime_ok,
            )

            # Long pick
            long_target, long_stop = compute_trade_levels(
                reference_price=long_ref, direction="LONG",
                target_pct=strategy_target_pct, stop_loss_pct=stop_loss_pct,
            )
            long_candidates[symbol] = {
                "symbol": symbol, "direction": "LONG",
                "entry_price": round(long_ref, 2), "prev_close": round(prev_close, 2),
                "target_price": round(long_target, 2), "stop_loss_price": round(long_stop, 2),
                "confidence": round(long_conf, 4), "predicted_magnitude": round(long_mag, 6),
                "score": round(score_candidate(long_conf, long_mag), 6),
                "preferred_filter_pass": long_passes,
                "cutoff_close": round(long_adjustment["cutoff_close"], 2) if long_adjustment.get("cutoff_close") else None,
                "gap_pct": long_adjustment.get("gap_pct"),
                "alignment_score": long_align,
            }

            # Short pick
            short_target, short_stop = compute_trade_levels(
                reference_price=short_ref, direction="SHORT",
                target_pct=strategy_target_pct, stop_loss_pct=stop_loss_pct,
            )
            short_candidates[symbol] = {
                "symbol": symbol, "direction": "SHORT",
                "entry_price": round(short_ref, 2), "prev_close": round(prev_close, 2),
                "target_price": round(short_target, 2), "stop_loss_price": round(short_stop, 2),
                "confidence": round(short_conf, 4), "predicted_magnitude": round(short_mag, 6),
                "score": round(score_candidate(short_conf, short_mag), 6),
                "preferred_filter_pass": short_passes,
                "cutoff_close": round(short_adjustment["cutoff_close"], 2) if short_adjustment.get("cutoff_close") else None,
                "gap_pct": short_adjustment.get("gap_pct"),
                "alignment_score": short_align,
            }

            progress.advance(task)

    # Remove contradictory picks
    overlap = set(long_candidates) & set(short_candidates)
    for sym in overlap:
        if long_candidates[sym]["score"] >= short_candidates[sym]["score"]:
            short_candidates.pop(sym, None)
        else:
            long_candidates.pop(sym, None)

    top_longs = select_candidates(
        list(long_candidates.values()), count=long_count, allow_below_preferred=allow_below_preferred,
    )
    top_shorts = select_candidates(
        list(short_candidates.values()), count=short_count, allow_below_preferred=allow_below_preferred,
    )

    runtime = perf_counter() - started
    console.print(f"[green]Post-open adjustments complete in {runtime:.2f}s[/green]")
    console.print(f"[dim]Live symbols: {live_count} | Ranks: {len(long_candidates)}L / {len(short_candidates)}S[/dim]")

    return {
        "longs": top_longs,
        "shorts": top_shorts,
        "runtime_seconds": round(runtime, 2),
        "live_symbols": live_count,
        "partial_symbols": partial_count,
        "ranked_longs": len(long_candidates),
        "ranked_shorts": len(short_candidates),
        "cutoff_time": post_open_cutoff,
    }


def _render_table(title: str, picks: list[dict], color: str) -> None:
    table = Table(title=title, header_style=f"bold {color}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="bold")
    table.add_column("Prev Close", justify="right")
    table.add_column("Cutoff Close", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Target", justify="right", style="green")
    table.add_column("Stop", justify="right", style="red")
    table.add_column("Pref", justify="center")
    table.add_column("Conf", justify="right")
    table.add_column("Score", justify="right", style="cyan")
    for idx, pick in enumerate(picks, start=1):
        table.add_row(
            str(idx),
            pick["symbol"],
            f"rs.{pick['prev_close']:,.2f}",
            f"rs.{pick.get('cutoff_close', pick['entry_price']):,.2f}" if pick.get("cutoff_close") else "—",
            f"rs.{pick['entry_price']:,.2f}",
            f"rs.{pick['target_price']:,.2f}",
            f"rs.{pick['stop_loss_price']:,.2f}",
            "yes" if pick["preferred_filter_pass"] else "no",
            f"{pick['confidence']:.2f}",
            f"{pick['score']:.4f}",
        )
    console.print(table)


def parse_args():
    parser = argparse.ArgumentParser(description="Fast post-open recommendation picks")
    parser.add_argument("--model", default="models/intraday_model_nifty500.pkl")
    parser.add_argument("--universe", default="nifty500")
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--target-date", default="")
    parser.add_argument("--mode", choices=("premarket", "post-open", "auto"), default="auto")
    parser.add_argument("--post-open-cutoff", default="09:30")
    parser.add_argument("--post-open-min-alignment", type=float, default=0.05)
    parser.add_argument("--risk-profile", choices=sorted(DEFAULT_STRATEGY_PROFILES.keys()), default="balanced")
    parser.add_argument("--long-count", type=int, default=3)
    parser.add_argument("--short-count", type=int, default=2)
    parser.add_argument("--allow-below-preferred", action="store_true")
    parser.add_argument("--no-refresh-yfinance", dest="refresh_yfinance", action="store_false", default=True)
    parser.add_argument("--save-json", type=str, nargs="?", const="default", default="default")
    parser.add_argument("--save-csv", type=str, nargs="?", const="default", default="default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    picks_for_date = pd.Timestamp(args.target_date) if args.target_date else default_target_date()
    date_str = picks_for_date.strftime("%Y%m%d")

    run_name = f"post_open_picks_{args.universe}_{date_str}"
    with start_run_logging(project_root=PROJECT_ROOT, log_group="recommendations", run_name=run_name) as run_logger:
        global console
        console = Console()

        console.print(Panel.fit(
            "[bold cyan]Post-Open Fast Picks Pipeline[/bold cyan]\n"
            f"[dim]Universe: {args.universe} | Date: {picks_for_date.strftime('%Y-%m-%d')} | "
            f"Longs: {args.long_count} | Shorts: {args.short_count}[/dim]\n"
            f"[dim]Mode: {args.mode} | Profile: {args.risk_profile}[/dim]",
            border_style="cyan",
        ))
        console.print(f"[dim]Command:[/dim] {command_string()}")
        console.print(f"[dim]Run log:[/dim] {run_logger.log_path}")

        data_dir = Path(args.data_dir)
        symbols = get_universe(args.universe)
        model_path = Path(args.model)

        cache_path = _cache_path(date_str, args.universe)

        # Determine mode
        mode = args.mode
        if mode == "auto":
            now_kolkata = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
            market_open = now_kolkata.replace(hour=9, minute=15, second=0, microsecond=0)
            mode = "post-open" if now_kolkata >= market_open else "premarket"
            console.print(f"[dim]Auto-detected mode: [bold]{mode}[/bold][/dim]")

        if mode == "premarket":
            if cache_path.exists():
                console.print(f"[yellow]Cache already exists: {cache_path}[/yellow]")
                console.print("[dim]Use --mode post-open to generate picks from cache.[/dim]")
                return 0
            build_premarket_cache(
                symbols, data_dir, model_path, picks_for_date,
                args.risk_profile, args.universe,
            )
            console.print(Panel.fit(
                "[bold green]Cache ready[/bold green]\n"
                f"Run again with [bold]--mode post-open[/bold] after 9:15 AM for live picks.",
                border_style="green",
            ))
            return 0

        if not cache_path.exists():
            console.print("[yellow]No premarket cache found. Building it now...[/yellow]")
            cache_path = build_premarket_cache(
                symbols, data_dir, model_path, picks_for_date,
                args.risk_profile, args.universe,
            )

        result = apply_post_open_adjustments(
            cache_path=cache_path,
            picks_for_date=picks_for_date,
            strategy_profile=args.risk_profile,
            long_count=args.long_count,
            short_count=args.short_count,
            post_open_cutoff=args.post_open_cutoff,
            post_open_min_alignment=args.post_open_min_alignment,
            allow_below_preferred=args.allow_below_preferred,
            refresh_yfinance=args.refresh_yfinance,
            data_dir=data_dir,
        )

        # Display results
        longs = result["longs"]
        shorts = result["shorts"]

        summary_table = Table(title="Post-Open Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Live symbols", str(result["live_symbols"]))
        summary_table.add_row("Partial symbols", str(result["partial_symbols"]))
        summary_table.add_row("Cutoff time", result["cutoff_time"])
        summary_table.add_row("Ranked longs", str(result["ranked_longs"]))
        summary_table.add_row("Ranked shorts", str(result["ranked_shorts"]))
        summary_table.add_row("Returned longs", str(len(longs)))
        summary_table.add_row("Returned shorts", str(len(shorts)))
        summary_table.add_row("Runtime", f"{result['runtime_seconds']:.2f}s")
        summary_table.add_row("Above pref longs", str(sum(1 for p in longs if p["preferred_filter_pass"])))
        summary_table.add_row("Above pref shorts", str(sum(1 for p in shorts if p["preferred_filter_pass"])))
        console.print()
        console.print(summary_table)
        console.print()

        if longs:
            _render_table(f"Top {len(longs)} Long Picks", longs, "green")
            console.print()
        if shorts:
            _render_table(f"Top {len(shorts)} Short Picks", shorts, "red")
            console.print()

        # Save outputs
        payload = {
            "generated_at": datetime.now().isoformat(),
            "picks_for_date": picks_for_date.strftime("%Y-%m-%d"),
            "model": str(model_path),
            "universe": args.universe,
            "mode": "post-open",
            "risk_profile": args.risk_profile,
            "cache_path": str(cache_path),
            "long_picks": longs,
            "short_picks": shorts,
            "summary": result,
            "run_log": str(run_logger.log_path),
        }

        json_path = _resolve_output_path(args.save_json, suffix="json", stem_prefix="post_open_picks", target_date=date_str)
        csv_path = _resolve_output_path(args.save_csv, suffix="csv", stem_prefix="post_open_picks", target_date=date_str)
        if json_path:
            json_path.write_text(json.dumps(payload, indent=2, default=str))
            console.print(f"[dim]JSON saved: {json_path}[/dim]")
        if csv_path:
            all_picks = longs + shorts
            pd.DataFrame(all_picks).to_csv(csv_path, index=False)
            console.print(f"[dim]CSV saved: {csv_path}[/dim]")

        console.print(Panel.fit(
            "[bold green]Post-open picks ready[/bold green]\n"
            f"[bold]Picks for:[/bold] {picks_for_date.strftime('%Y-%m-%d')}\n"
            f"[bold]Longs:[/bold] {len(longs)} | [bold]Shorts:[/bold] {len(shorts)}\n"
            f"[bold]Runtime:[/bold] {result['runtime_seconds']:.2f}s",
            border_style="green",
        ))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
