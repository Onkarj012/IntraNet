#!/usr/bin/env python3
"""
Pre-compute features for all stocks and save to disk.

This avoids recomputing features every epoch during training.
Run once before training:

    python scripts/precompute_features.py --config configs/intraday_config.yaml
    python scripts/precompute_features.py --config configs/intraday_config.yaml --subset-stocks RELIANCE,TCS
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.config import load_config
from intradaynet.features.per_bar_features import compute_per_bar_features, PER_BAR_FEATURE_NAMES
from intradaynet.features.session_features import compute_session_features, SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SentimentFeatureBuilder, SENTIMENT_FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute features")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--subset-stocks", type=str, default="")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="features_cache")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minute_dir = Path(cfg.data.minute_data_dir)

    # Symbol selection
    if args.subset_stocks:
        symbols = [s.strip() for s in args.subset_stocks.split(",")]
    elif args.max_stocks > 0:
        all_csvs = sorted(minute_dir.glob("*_minute.csv"))
        symbols = [f.stem.replace("_minute", "") for f in all_csvs[:args.max_stocks]]
    else:
        symbols = sorted([
            f.stem.replace("_minute", "")
            for f in minute_dir.glob("*_minute.csv")
        ])

    logger.info(f"Pre-computing features for {len(symbols)} stocks → {output_dir}/")

    # Market feature builder — download macro data once
    from intradaynet.features.market_features import MarketFeatureBuilder
    market_builder = MarketFeatureBuilder(cache_dir=str(output_dir / ".market_cache"))
    logger.info("Downloading/updating market macro data...")
    market_builder.download(start=str(cfg.splits.train_start)[:10])

    # Sentiment builder (with market features)
    sentiment_builder = SentimentFeatureBuilder(cfg.data.sentiment_csv,
                                                market_builder=market_builder)

    success = 0
    failed = 0
    t_total = time.time()

    for i, symbol in enumerate(symbols):
        csv_path = minute_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            logger.warning(f"[{i+1}/{len(symbols)}] {symbol}: CSV not found, skipping")
            failed += 1
            continue

        t0 = time.time()
        try:
            # Load raw data
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            df.columns = df.columns.str.lower()

            # Filter to market hours
            df["_time"] = df.index.strftime("%H:%M")
            df = df[(df["_time"] >= cfg.data.market_open) & (df["_time"] < cfg.data.market_close)]
            df = df.drop(columns=["_time"])

            if len(df) < 100:
                logger.warning(f"[{i+1}/{len(symbols)}] {symbol}: too few bars ({len(df)}), skipping")
                failed += 1
                continue

            # Compute per-bar features (25 cols)
            per_bar = compute_per_bar_features(df)

            # Compute session features (20 cols)
            session = compute_session_features(df)

            # Compute sentiment + market features (24 cols)
            sent = sentiment_builder.get_features(symbol, session.index)

            # Save: per_bar as (N, 25) float32, session/sentiment indexed by date
            out_path = output_dir / f"{symbol}.npz"
            np.savez_compressed(
                out_path,
                # Per-bar data
                per_bar_features=per_bar[PER_BAR_FEATURE_NAMES].values.astype(np.float32),
                per_bar_timestamps=per_bar.index.astype(np.int64),  # datetime64 as int
                per_bar_dates=(per_bar.index.date.astype(str)),
                # OHLCV for target computation
                close=df["close"].values.astype(np.float32),
                # Session features
                session_features=session[SESSION_FEATURE_NAMES].values.astype(np.float32),
                session_dates=session.index.astype(str),
                # Sentiment
                sentiment_features=sent[SENTIMENT_FEATURE_NAMES].values.astype(np.float32),
            )

            elapsed = time.time() - t0
            logger.info(f"[{i+1}/{len(symbols)}] {symbol}: {len(df)} bars, {len(session)} sessions → {out_path.name} ({elapsed:.1f}s)")
            success += 1

        except Exception as e:
            logger.error(f"[{i+1}/{len(symbols)}] {symbol}: FAILED — {e}")
            failed += 1
            continue

    total_time = time.time() - t_total
    logger.info(f"\nDone: {success} succeeded, {failed} failed in {total_time:.0f}s")
    logger.info(f"Features cached in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
