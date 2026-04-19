"""
Sentiment feature builder for IntradayNet.

Loads daily sentiment data and computes sentiment-derived features
for each stock-date. Market-level features (VIX, NIFTY return, etc.)
are filled from MarketFeatureBuilder when available.

Expanded sentiment and market context features for both the daily and
live-backend pipelines.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import time

logger = logging.getLogger("intradaynet.features.sentiment_features")

# sentiment + market features total
SENTIMENT_FEATURE_NAMES = [
    # ── Stock-level news sentiment (1-8) ──
    "premarket_sentiment",
    "premarket_sentiment_count",
    "premarket_sentiment_max",
    "premarket_sentiment_std",
    "sentiment_5d_avg",
    "sentiment_momentum",
    "sentiment_spike",
    "sentiment_price_div",
    "news_volume_shock",
    "sentiment_surprise",
    "sentiment_macro_agreement",
    "sentiment_confidence",
    # ── India market features (9-14) ──
    "nifty_intraday_return",
    "sector_intraday_return",
    "vix_level",
    "vix_change",
    "market_breadth",
    "global_cue",
    # ── Global macro features (15-24) ──
    "crude_oil_return",
    "crude_oil_5d_change",
    "gold_return",
    "usdinr_change",
    "us_10y_yield_change",
    "dxy_change",
    "asia_sentiment",
    "dow_overnight_return",
    "nasdaq_overnight_return",
    "global_volatility_regime",
    "india_vix_percentile",
    "nifty_5d_return",
    "sp500_overnight_return",
    "commodity_pressure",
    "dollar_yield_pressure",
    "risk_on_signal",
]

# Original 14 feature names (for backward compatibility)
SENTIMENT_FEATURE_NAMES_V1 = SENTIMENT_FEATURE_NAMES[:14]


class SentimentFeatureBuilder:
    """
    Builds daily sentiment features per stock from CSV data.

    Usage:
        builder = SentimentFeatureBuilder("path/to/sentiment.csv")
        features = builder.get_features("RELIANCE", dates)

    With market features:
        builder = SentimentFeatureBuilder("path/to/sentiment.csv",
                                         market_builder=market_builder)
    """

    def __init__(self, csv_path: str, market_builder=None, market_open_time: str = "09:15"):
        self.csv_path = Path(csv_path)
        self._data = None
        self._loaded = False
        self.market_builder = market_builder
        hour, minute = market_open_time.split(":")
        self.market_open_cutoff = time(int(hour), int(minute))

    def _load(self):
        """Lazy-load sentiment CSV."""
        if self._loaded:
            return

        if not self.csv_path.exists():
            logger.warning(f"Sentiment CSV not found: {self.csv_path}")
            self._data = pd.DataFrame()
            self._loaded = True
            return

        try:
            df = pd.read_csv(self.csv_path, parse_dates=["Publish Date"])
            df = df.rename(columns={
                "Symbol": "symbol",
                "Publish Date": "timestamp",
                "sentiment_score": "score",
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"].notna()].copy()
            df["date"] = df["timestamp"].dt.date
            self._data = df
            logger.info(f"Loaded sentiment data: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load sentiment CSV: {e}")
            self._data = pd.DataFrame()

        self._loaded = True

    def get_features(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Get 24 sentiment + market features for a stock across given dates.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            dates: DatetimeIndex of dates to get features for

        Returns:
            DataFrame with 24 features, indexed by date.
        """
        self._load()

        features = pd.DataFrame(0.0, index=dates, columns=SENTIMENT_FEATURE_NAMES)

        if self._data is not None and not self._data.empty:
            # Filter for this symbol
            sym_data = self._data[self._data["symbol"] == symbol].copy()

            if not sym_data.empty:
                # Use only articles published before market open for same-day features.
                sym_data = sym_data[
                    sym_data["timestamp"].dt.time <= self.market_open_cutoff
                ].copy()

            if not sym_data.empty:
                # Aggregate daily sentiment scores
                daily_sent = sym_data.groupby("date").agg(
                    mean_score=("score", "mean"),
                    count=("score", "count"),
                    max_score=("score", "max"),
                    std_score=("score", "std"),
                ).fillna(0)
                daily_sent.index = pd.to_datetime(daily_sent.index)

                # Align with requested dates
                daily_sent = daily_sent.reindex(dates).fillna(0)

                # ── Features 1-4: Pre-market sentiment ──
                features["premarket_sentiment"] = daily_sent["mean_score"].clip(-1, 1)
                features["premarket_sentiment_count"] = (daily_sent["count"] / 10.0).clip(0, 5)
                features["premarket_sentiment_max"] = daily_sent["max_score"].clip(-1, 1)
                features["premarket_sentiment_std"] = daily_sent["std_score"].clip(0, 1)

                # ── Features 5-11: Rolling sentiment ──
                features["sentiment_5d_avg"] = (
                    daily_sent["mean_score"].rolling(5, min_periods=1).mean().clip(-1, 1)
                )
                sent_21d = daily_sent["mean_score"].rolling(21, min_periods=1).mean()
                features["sentiment_momentum"] = (features["sentiment_5d_avg"] - sent_21d).clip(-1, 1)

                sent_5d_std = daily_sent["mean_score"].rolling(5, min_periods=1).std().replace(0, 1)
                features["sentiment_spike"] = (
                    (daily_sent["mean_score"] - features["sentiment_5d_avg"]) / sent_5d_std
                ).clip(-3, 3)

                count_mean = daily_sent["count"].rolling(10, min_periods=1).mean().replace(0, 1)
                features["news_volume_shock"] = (
                    daily_sent["count"] / count_mean - 1
                ).clip(-2, 5)
                features["sentiment_surprise"] = (
                    daily_sent["mean_score"] - features["sentiment_5d_avg"]
                ).clip(-1, 1)
                features["sentiment_confidence"] = (
                    np.log1p(daily_sent["count"]) / (1.0 + daily_sent["std_score"].fillna(0))
                ).clip(0, 5) / 5.0

        # ── Feature 8: Sentiment-price divergence (stubbed) ──
        features["sentiment_price_div"] = 0.0
        features["sentiment_macro_agreement"] = 0.0

        # ── Market-level features ──
        if self.market_builder is not None:
            try:
                # Fill features 9-14 (India-specific)
                india_feats = self.market_builder.get_india_market_features(dates)
                for key, values in india_feats.items():
                    if key in features.columns:
                        features[key] = values

                # Fill features 15-24 (global macro)
                from intradaynet.features.market_features import MARKET_FEATURE_NAMES
                market_feats = self.market_builder.get_features(dates)
                for col in MARKET_FEATURE_NAMES:
                    if col in features.columns and col in market_feats.columns:
                        features[col] = market_feats[col]

                if "global_cue" in features.columns:
                    macro_sign = np.sign(features["global_cue"]).replace(0, 1)
                else:
                    macro_sign = np.sign(features["risk_on_signal"]).replace(0, 1)
                sent_sign = np.sign(features["premarket_sentiment"]).replace(0, 0)
                features["sentiment_macro_agreement"] = (sent_sign * macro_sign).clip(-1, 1)

            except Exception as e:
                logger.warning(f"Failed to compute market features: {e}")

        return features.fillna(0.0)
