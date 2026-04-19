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
from intradaynet.open_safe_daily_features import build_open_safe_daily_features
from intradaynet.run_logging import command_string, start_run_logging
from intradaynet.universe import get_universe

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
    direction: str
    confidence: float
    score: float
    predicted_magnitude: float
    entry_price: float
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
        min_confidence=0.65,
        min_predicted_magnitude=0.010,
    ),
    "balanced": RiskConfig(
        name="Balanced",
        stop_loss_pct=0.010,
        target_pct=0.015,
        trailing_start=0.008,
        trailing_stop_pct=0.005,
        max_trades_per_day=5,
        min_confidence=0.65,
        min_predicted_magnitude=0.010,
    ),
    "aggressive": RiskConfig(
        name="Aggressive",
        stop_loss_pct=0.020,
        target_pct=0.025,
        trailing_start=0.015,
        trailing_stop_pct=0.010,
        max_trades_per_day=8,
        min_confidence=0.55,
        min_predicted_magnitude=0.008,
    ),
}


def simulate_trade(day_data: pd.DataFrame, direction: str, entry_price: float, config: RiskConfig, position_size: float) -> dict:
    if direction == "LONG":
        target = entry_price * (1 + config.target_pct)
        stop = entry_price * (1 - config.stop_loss_pct)
    else:
        target = entry_price * (1 - config.target_pct)
        stop = entry_price * (1 + config.stop_loss_pct)

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
    parser.add_argument("--model", default="models/intraday_model.pkl")
    parser.add_argument("--risk", default="balanced", choices=sorted(RISK_PROFILES.keys()))
    parser.add_argument("--universe", default="nifty100")
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--output-dir", default="backtest_results")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=-1.0)
    parser.add_argument("--min-predicted-magnitude", type=float, default=-1.0)
    parser.add_argument("--refresh-yfinance", action="store_true")
    parser.add_argument("--augment-yf-news", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = RISK_PROFILES[args.risk]
    run_name = f"backtest_intraday_2025_{args.risk}"

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
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        symbols = get_universe(args.universe)
        if args.max_stocks > 0:
            symbols = symbols[:args.max_stocks]

        market_builder = MarketFeatureBuilder()
        market_builder.download(start="2021-01-01", end=args.end_date)
        sentiment_csv = Path("sentiment/combined_sentiment_2015_2025.csv")
        if args.augment_yf_news:
            console.print("[bold]Augmenting sentiment with yfinance news where available...[/bold]")
            augmented_csv = PROJECT_ROOT / "sentiment" / f"combined_sentiment_augmented_{args.start_date}_{args.end_date}.csv"
            sentiment_csv = augment_sentiment_with_yfinance(
                symbols,
                sentiment_csv,
                args.start_date,
                args.end_date,
                augmented_csv,
            )
        sentiment_builder = SentimentFeatureBuilder(
            str(sentiment_csv),
            market_builder=market_builder,
        )
        sentiment_builder._load()

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
                minute_df = load_minute_data(data_dir / f"{symbol}_minute.csv")
                minute_df = maybe_backfill_with_yfinance(
                    minute_df,
                    symbol,
                    args.start_date,
                    args.end_date,
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
                    date_data = minute_df[minute_df.index.normalize() == dt.normalize()]
                    if date_data.empty:
                        continue

                    long_prob = float(long_probs[idx])
                    short_prob = float(short_probs[idx])
                    if long_prob >= short_prob:
                        direction = "LONG"
                        confidence = long_prob
                        pred_mag = float(up_mags[idx])
                    else:
                        direction = "SHORT"
                        confidence = short_prob
                        pred_mag = float(down_mags[idx])

                    if confidence < min_confidence or pred_mag < min_predicted_magnitude:
                        continue

                    score = confidence * max(pred_mag, 1e-6)
                    candidates_by_date[dt.normalize()].append(
                        {
                            "date": dt.normalize(),
                            "symbol": symbol,
                            "direction": direction,
                            "confidence": confidence,
                            "predicted_magnitude": pred_mag,
                            "score": score,
                            "day_data": date_data,
                            "feature_date": dt,
                        }
                    )
                progress.advance(task)

        trades: list[TradeRecord] = []
        hit_records: list[dict] = []
        for trade_date in sorted(candidates_by_date):
            ranked = sorted(candidates_by_date[trade_date], key=lambda item: item["score"], reverse=True)
            for candidate in ranked[:max_trades_per_day]:
                entry_price = float(candidate["day_data"]["open"].iloc[0])
                if entry_price <= 0:
                    continue
                outcome = simulate_trade(candidate["day_data"], candidate["direction"], entry_price, config, position_size)
                predicted_target = (
                    entry_price * (1 + candidate["predicted_magnitude"])
                    if candidate["direction"] == "LONG"
                    else entry_price * (1 - candidate["predicted_magnitude"])
                )
                stop_price = (
                    entry_price * (1 - config.stop_loss_pct)
                    if candidate["direction"] == "LONG"
                    else entry_price * (1 + config.stop_loss_pct)
                )
                hit = evaluate_hit_metrics(
                    candidate["day_data"],
                    candidate["direction"],
                    predicted_target,
                    stop_price,
                )
                hit_records.append(
                    {
                        "date": str(trade_date.date()),
                        "symbol": candidate["symbol"],
                        "direction": candidate["direction"],
                        "predicted_target": round(predicted_target, 2),
                        **hit,
                    }
                )
                trades.append(
                    TradeRecord(
                        date=str(trade_date.date()),
                        symbol=candidate["symbol"],
                        direction=candidate["direction"],
                        confidence=round(candidate["confidence"], 4),
                        score=round(candidate["score"], 6),
                        predicted_magnitude=round(candidate["predicted_magnitude"], 6),
                        entry_price=round(entry_price, 2),
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
        trades_path = output_dir / f"trades_intraday_model_{args.risk}_{args.start_date}_{args.end_date}.csv"
        trades_df.to_csv(trades_path, index=False)
        hits_df = pd.DataFrame(hit_records)
        hits_path = output_dir / f"hits_intraday_model_{args.risk}_{args.start_date}_{args.end_date}.csv"
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
        coverage_path = output_dir / f"coverage_intraday_model_{args.risk}_{args.start_date}_{args.end_date}.csv"
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

        summary = {
            "model": str(model_path),
            "risk_profile": config.name,
            "capital": args.capital,
            "top_k": max_trades_per_day,
            "position_size": position_size,
            "min_confidence": min_confidence,
            "min_predicted_magnitude": min_predicted_magnitude,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "processed_symbols": processed_symbols,
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
            "exit_reasons": by_reason,
            "run_log": str(run_logger.log_path),
            "trades_csv": str(trades_path),
            "hits_csv": str(hits_path),
            "coverage_csv": str(coverage_path),
            "sentiment_csv_used": str(sentiment_csv),
        }

        summary_path = output_dir / f"summary_intraday_model_{args.risk}_{args.start_date}_{args.end_date}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        summary_table = Table(title="Backtest Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Processed symbols", str(processed_symbols))
        summary_table.add_row("Skipped symbols", str(len(skipped_symbols)))
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
        console.print(
            Panel.fit(
                f"[bold green]Backtest complete[/bold green]\n"
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
