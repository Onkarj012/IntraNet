"""
Per-stock sentiment features via yfinance news.

Unlike the existing market-level sentiment in features/sentiment_features.py,
this module computes sentiment features for EACH individual stock using
yfinance news API.

Features computed per stock per day:
- sentiment_score_1d: Average sentiment of articles in the past 1 day
- sentiment_score_3d: Average sentiment of articles in the past 3 days
- sentiment_momentum_5d: Change in sentiment over 5 days
- article_count_1d: Number of articles in the past 1 day
- article_count_3d: Number of articles in the past 3 days
- sentiment_volatility_5d: Std of daily sentiment over 5 days
- headline_sentiment_bias: Average of headline sentiment scores
- news_to_price_ratio: Article count normalized by recent price move
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from intradaynet.live_news import (
    LiveNewsSummary,
    combine_article_sources,
    fetch_live_yfinance_news,
    infer_title_sentiment,
    normalize_historical_sentiment_csv,
)


DEFAULT_TZ = "Asia/Kolkata"


def compute_per_stock_sentiment_features(
    symbol: str,
    target_date: pd.Timestamp,
    *,
    historical_df: pd.DataFrame | None = None,
    news_lookback_days: int = 3,
    min_articles_for_score: int = 2,
    fetch_live: bool = False,
) -> dict[str, float]:
    """
    Compute sentiment features for one stock on one target date.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE").
    target_date : pd.Timestamp
        Target trading date (features use data up to previous day).
    historical_df : pd.DataFrame, optional
        Pre-loaded historical sentiment data.
    news_lookback_days : int
        Number of days of news history to aggregate.
    min_articles_for_score : int
        Minimum articles required for a valid sentiment score.
    fetch_live : bool
        If True, also fetch live news via yfinance.

    Returns
    -------
    dict[str, float]
        Dictionary of per-stock sentiment features.
    """
    features = {
        "sentiment_score_1d": np.nan,
        "sentiment_score_3d": np.nan,
        "sentiment_momentum_5d": np.nan,
        "article_count_1d": 0,
        "article_count_3d": 0,
        "sentiment_volatility_5d": np.nan,
        "headline_sentiment_bias": np.nan,
        "news_to_price_ratio": np.nan,
    }

    if historical_df is None or historical_df.empty:
        return features

    symbol_articles = historical_df[historical_df["symbol"].str.upper() == symbol.upper()].copy()
    if symbol_articles.empty:
        return features

    if "timestamp" not in symbol_articles.columns:
        return features

    symbol_articles["date"] = pd.to_datetime(symbol_articles["timestamp"]).dt.normalize()
    symbol_articles = symbol_articles.sort_values("date")

    lookback = pd.Timestamp(target_date) - pd.Timedelta(days=news_lookback_days)
    lookback_5d = pd.Timestamp(target_date) - pd.Timedelta(days=5)

    recent = symbol_articles[symbol_articles["date"] >= lookback]
    recent_5d = symbol_articles[symbol_articles["date"] >= lookback_5d]

    # 1-day sentiment
    one_day = recent[recent["date"] >= pd.Timestamp(target_date) - pd.Timedelta(days=1)]
    if len(one_day) >= min_articles_for_score:
        features["sentiment_score_1d"] = float(one_day["score"].mean())
        features["article_count_1d"] = len(one_day)

    # 3-day sentiment
    if len(recent) >= min_articles_for_score:
        features["sentiment_score_3d"] = float(recent["score"].mean())
        features["article_count_3d"] = len(recent)

    # 5-day sentiment momentum
    daily_sentiment = recent_5d.groupby("date")["score"].mean()
    if len(daily_sentiment) >= 3:
        features["sentiment_momentum_5d"] = float(daily_sentiment.iloc[-1] - daily_sentiment.iloc[0])
        features["sentiment_volatility_5d"] = float(daily_sentiment.std())

    # Headline sentiment bias
    if "headline" in symbol_articles.columns:
        headlines = recent["headline"].dropna().unique()
        if len(headlines) > 0:
            headline_scores = [infer_title_sentiment(h) for h in headlines]
            features["headline_sentiment_bias"] = float(np.mean(headline_scores))

    return features


def build_historical_sentiment_cache(
    csv_paths: list[str | Path],
    *,
    output_path: str | Path | None = None,
    universe_metadata_csv: str | None = None,
) -> pd.DataFrame:
    """
    Load and merge multiple historical sentiment CSV files into one cache.

    Parameters
    ----------
    csv_paths : list
        Paths to sentiment CSV files.
    output_path : str | Path, optional
        If provided, save merged DataFrame to parquet.
    universe_metadata_csv : str, optional
        Path to universe metadata for industry mapping.

    Returns
    -------
    pd.DataFrame
        Combined historical sentiment data.
    """
    frames = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue
        try:
            df = normalize_historical_sentiment_csv(
                csv_path, universe_metadata_csv=universe_metadata_csv,
            )
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "headline", "source", "score"])

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    combined = combined.sort_values(["timestamp", "symbol"])
    combined = combined.drop_duplicates(subset=["symbol", "timestamp", "headline", "source"], keep="last")
    combined = combined.reset_index(drop=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)

    return combined


def load_sentiment_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load pre-built sentiment cache."""
    cache_path = Path(cache_path)
    if cache_path.suffix == ".parquet":
        return pd.read_parquet(cache_path)
    elif cache_path.suffix == ".csv":
        return pd.read_csv(cache_path, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(cache_path, nrows=0).columns else False)
    else:
        return pd.DataFrame()


def compute_sentiment_features_batch(
    symbols: list[str],
    target_date: pd.Timestamp,
    *,
    historical_df: pd.DataFrame,
    news_lookback_days: int = 3,
    min_articles_for_score: int = 2,
) -> pd.DataFrame:
    """
    Compute per-stock sentiment features for all symbols on one date.

    Returns a DataFrame with one row per symbol.
    """
    rows = []
    for symbol in symbols:
        features = compute_per_stock_sentiment_features(
            symbol, target_date,
            historical_df=historical_df,
            news_lookback_days=news_lookback_days,
            min_articles_for_score=min_articles_for_score,
        )
        features["symbol"] = symbol
        rows.append(features)

    df = pd.DataFrame(rows)
    if "symbol" in df.columns:
        df = df.set_index("symbol")
    return df


def fetch_live_sentiment(
    symbols: list[str],
    target_date: pd.Timestamp,
    *,
    historical_df: pd.DataFrame | None = None,
    universe_metadata_csv: str | None = None,
    max_symbols: int = 0,
) -> tuple[pd.DataFrame, LiveNewsSummary]:
    """
    Fetch live per-stock sentiment via yfinance.

    Parameters
    ----------
    symbols : list
        Stock symbols to fetch news for.
    target_date : pd.Timestamp
        Target date for news window.
    historical_df : pd.DataFrame, optional
        Historical sentiment data to combine with.
    universe_metadata_csv : str, optional
        Path to universe metadata.
    max_symbols : int
        Max symbols to fetch (0 = all).

    Returns
    -------
    tuple
        (combined articles DataFrame, summary stats).
    """
    symbols_to_fetch = symbols[:max_symbols] if max_symbols > 0 else symbols

    start_ts = target_date - pd.Timedelta(days=7)
    end_ts = target_date + pd.Timedelta(days=1)

    live_articles, summary = fetch_live_yfinance_news(
        symbols_to_fetch,
        start_ts=start_ts,
        end_ts=end_ts,
        universe_metadata_csv=universe_metadata_csv,
    )

    combined = combine_article_sources(historical_df, live_articles)

    return combined, summary


def get_all_sentiment_csv_paths(data_dir: str | Path = "data/sentiment") -> list[Path]:
    """Discover all sentiment CSV files in the data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))
