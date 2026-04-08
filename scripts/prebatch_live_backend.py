#!/usr/bin/env python3
"""
Prebatch training data for the live LightGBM backend.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.config import load_config
from intradaynet.costs import DEFAULT_COSTS, estimate_liquidity_penalty
from intradaynet.feature_contract import FEATURE_NAMES, flatten_intraday_batch
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.per_bar_features import compute_per_bar_features
from intradaynet.features.sentiment_features import SentimentFeatureBuilder
from intradaynet.features.session_features import compute_session_features
from intradaynet.targets import HORIZONS, TargetConfig
from intradaynet.recommendation import liquidity_score


SEQ_LENGTH = 120
SAMPLE_INTERVAL = 15
MIN_SESSION_BARS = 150
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Prebatch live backend data")
    parser.add_argument("--config", default="configs/intraday_config.yaml")
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--source", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--sentiment-csv", default="sentiment/combined_sentiment_2015_2025.csv")
    parser.add_argument("--market-cache", default="market_data_cache")
    parser.add_argument("--output-dir", default="prebatched_live")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--splits", default="train,val,test",
                        help="Comma-separated splits to build")
    parser.add_argument("--resume", action="store_true",
                        help="Skip split files that already exist")
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


def build_split(
    symbols: list[str],
    *,
    split_name: str,
    split_start: str,
    split_end: str,
    source: str,
    data_dir: Path,
    sentiment_builder: SentimentFeatureBuilder,
):
    all_windows = []
    all_session = []
    all_sentiment = []
    all_meta: dict[str, list] = {
        "symbol": [],
        "session_date": [],
        "entry_price": [],
        "avg_daily_traded_value": [],
        "median_minute_turnover": [],
        "liquidity_score": [],
    }
    all_targets: dict[str, list[np.ndarray]] = {}

    start_date = pd.Timestamp(split_start).date()
    end_date = pd.Timestamp(split_end).date()
    target_cfg = TargetConfig(horizons={h: HORIZONS[h] for h in ("H15", "H30", "H60")})

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"{split_name}: scanning symbols", total=len(symbols))

        for symbol in symbols:
            progress.update(task, description=f"{split_name}: {symbol}")
            df = load_stock_data(symbol, source, data_dir)
            if df is None or len(df) < 500:
                progress.advance(task)
                continue

            per_bar = compute_per_bar_features(df)
            session = compute_session_features(df)
            sentiment = sentiment_builder.get_features(symbol, session.index)

            dates = per_bar.index.date
            unique_dates = np.unique(dates)

            for session_date in unique_dates:
                if session_date < start_date or session_date > end_date:
                    continue

                mask = dates == session_date
                idx = np.where(mask)[0]
                n_bars = len(idx)
                if n_bars < MIN_SESSION_BARS:
                    continue

                offsets = np.arange(SEQ_LENGTH, n_bars - max(target_cfg.horizons.values()), SAMPLE_INTERVAL)
                if len(offsets) == 0:
                    continue

                session_idx = pd.Timestamp(session_date)
                session_vec = session.loc[session_idx].values.astype(np.float32)
                sentiment_vec = sentiment.loc[session_idx].values.astype(np.float32)

                session_close = df["close"].values[idx]
                session_volume = df["volume"].values[idx]
                turnover = session_close * session_volume
                avg_daily_traded_value = float(np.mean(turnover))
                median_minute_turnover = float(np.median(turnover))
                liq_score = liquidity_score(avg_daily_traded_value, median_minute_turnover)
                liquidity_penalty = estimate_liquidity_penalty(
                    avg_daily_traded_value=avg_daily_traded_value,
                    median_minute_turnover=median_minute_turnover,
                )

                for offset in offsets:
                    global_idx = idx[offset]
                    start_idx = global_idx - SEQ_LENGTH
                    end_idx = global_idx

                    all_windows.append(per_bar.iloc[start_idx:end_idx].values.astype(np.float32))
                    all_session.append(session_vec)
                    all_sentiment.append(sentiment_vec)
                    all_meta["symbol"].append(symbol)
                    all_meta["session_date"].append(str(session_date))
                    all_meta["entry_price"].append(float(df["close"].iloc[global_idx]))
                    all_meta["avg_daily_traded_value"].append(avg_daily_traded_value)
                    all_meta["median_minute_turnover"].append(median_minute_turnover)
                    all_meta["liquidity_score"].append(liq_score)

                    for horizon in ("H15", "H30", "H60"):
                        future_offset = offset + HORIZONS[horizon]
                        if future_offset >= n_bars:
                            all_targets.setdefault(f"dir_{horizon}", []).append(np.nan)
                            all_targets.setdefault(f"gross_{horizon}", []).append(0.0)
                            all_targets.setdefault(f"edge_{horizon}", []).append(0.0)
                            all_targets.setdefault(f"valid_{horizon}", []).append(False)
                            continue

                        anchor = float(session_close[offset])
                        future_close = float(session_close[future_offset])
                        raw_return = (future_close - anchor) / max(anchor, 1e-8)
                        gross_return = float(np.clip(raw_return, -target_cfg.magnitude_clip, target_cfg.magnitude_clip))

                        round_trip_cost = DEFAULT_COSTS.estimate_round_trip_fraction(
                            entry_price=anchor,
                            position_value=target_cfg.position_value,
                        )
                        total_penalty = max(
                            round_trip_cost + liquidity_penalty,
                            target_cfg.cost_adjustment + target_cfg.liquidity_penalty_floor,
                        )
                        net_return = raw_return - total_penalty if raw_return > 0 else raw_return + total_penalty
                        net_return = abs(net_return) * np.sign(raw_return) if raw_return != 0 else 0.0
                        edge_return = float(np.clip(net_return, -target_cfg.magnitude_clip, target_cfg.magnitude_clip))

                        if net_return > target_cfg.move_threshold:
                            direction = 1.0
                            valid = True
                        elif net_return < -target_cfg.move_threshold:
                            direction = 0.0
                            valid = True
                        else:
                            direction = np.nan
                            valid = False

                        all_targets.setdefault(f"dir_{horizon}", []).append(direction)
                        all_targets.setdefault(f"gross_{horizon}", []).append(gross_return)
                        all_targets.setdefault(f"edge_{horizon}", []).append(edge_return)
                        all_targets.setdefault(f"valid_{horizon}", []).append(valid)

            progress.advance(task)

    if not all_windows:
        return None

    windows = np.stack(all_windows).astype(np.float32)
    session_batch = np.stack(all_session).astype(np.float32)
    sentiment_batch = np.stack(all_sentiment).astype(np.float32)
    flat = flatten_intraday_batch(windows, session_batch, sentiment_batch)

    output = {
        "X_flat": flat,
        "feature_names": np.array(FEATURE_NAMES, dtype=object),
        "session_features": session_batch,
        "sentiment_features": sentiment_batch,
    }
    for key, values in all_targets.items():
        output[key] = np.asarray(values)
    for key, values in all_meta.items():
        output[key] = np.asarray(values)

    console.print(
        f"[green]{split_name}[/green]: {flat.shape[0]:,} samples x {flat.shape[1]} features"
    )
    return output


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = "*.parquet" if args.source == "parquet" else "*_minute.csv"
    symbols = sorted(
        p.stem.replace("_minute", "")
        for p in data_dir.glob(ext)
        if not p.name.startswith(".")
    )
    if args.max_stocks > 0:
        symbols = symbols[: args.max_stocks]

    market_builder = MarketFeatureBuilder(cache_dir=args.market_cache)
    sentiment_builder = SentimentFeatureBuilder(args.sentiment_csv, market_builder=market_builder)

    splits = {
        "train": (cfg.splits.train_start, cfg.splits.train_end),
        "val": (cfg.splits.val_start, cfg.splits.val_end),
        "test": (cfg.splits.test_start, cfg.splits.test_end),
    }
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    console.print(
        Panel.fit(
            "[bold cyan]IntradayNet Live Backend Prebatcher[/bold cyan]\n"
            f"[dim]Source: {data_dir} ({args.source}) | Stocks: {len(symbols)}[/dim]",
            border_style="cyan",
        )
    )

    started = time.time()
    summary = Table(title="Prebatch Summary")
    summary.add_column("Split", style="cyan")
    summary.add_column("Samples", justify="right", style="green")
    summary.add_column("Features", justify="right", style="dim")

    for split_name, (split_start, split_end) in splits.items():
        if split_name not in requested_splits:
            continue
        split_path = output_dir / f"{split_name}.npz"
        if args.resume and split_path.exists():
            loaded = np.load(split_path, allow_pickle=True)
            summary.add_row(split_name, f"{loaded['X_flat'].shape[0]:,}", str(loaded["X_flat"].shape[1]))
            console.print(f"[yellow]Skipping {split_name}[/yellow] [dim](already exists)[/dim]")
            continue
        split = build_split(
            symbols,
            split_name=split_name,
            split_start=split_start,
            split_end=split_end,
            source=args.source,
            data_dir=data_dir,
            sentiment_builder=sentiment_builder,
        )
        if split is not None:
            np.savez_compressed(split_path, **split)
            summary.add_row(
                split_name,
                f"{split['X_flat'].shape[0]:,}",
                str(split["X_flat"].shape[1]),
            )
        else:
            summary.add_row(split_name, "0", "0")

    console.print()
    console.print(summary)
    console.print(
        f"\n[bold green]Done in {time.time() - started:.1f}s[/bold green]\n"
        f"[dim]Output: {output_dir}[/dim]"
    )


if __name__ == "__main__":
    main()
