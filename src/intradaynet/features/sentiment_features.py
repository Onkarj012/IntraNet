"""
Sentiment and news-context feature builder for IntradayNet.

Supports:
- historical CSV sentiment for training/backtests
- live normalized articles for recommendation runs
- mode-specific article cutoffs for premarket and post-open workflows
- industry-aware aggregates built from the same symbol article pool
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from intradaynet.live_news import normalize_historical_sentiment_csv
from intradaynet.universe import get_symbol_metadata

logger = logging.getLogger("intradaynet.features.sentiment_features")

SENTIMENT_FEATURE_NAMES = [
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
    "nifty_intraday_return",
    "sector_intraday_return",
    "vix_level",
    "vix_change",
    "market_breadth",
    "global_cue",
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
    "industry_premarket_sentiment",
    "industry_premarket_sentiment_count",
    "industry_sentiment_5d_avg",
    "industry_sentiment_momentum",
    "industry_news_volume_shock",
    "industry_sentiment_surprise",
    "industry_sentiment_stock_divergence",
    "sector_index_prev_return",
    "sector_index_5d_return",
    "sector_index_volatility",
    "stock_vs_sector_1d",
    "stock_vs_sector_5d",
    "industry_relative_strength_rank",
    "sector_breadth_proxy",
    "secondary_sector_confirmation",
]

SENTIMENT_FEATURE_NAMES_V1 = SENTIMENT_FEATURE_NAMES[:14]


class SentimentFeatureBuilder:
    def __init__(
        self,
        csv_path: str,
        market_builder=None,
        market_open_time: str = "09:15",
        *,
        mode: str = "premarket",
        post_open_news_cutoff: str = "09:20",
        universe_metadata_csv: str | None = None,
        articles_df: pd.DataFrame | None = None,
    ):
        self.csv_path = Path(csv_path)
        self._data: pd.DataFrame | None = None
        self._loaded = False
        self.market_builder = market_builder
        self.mode = mode
        self.market_open_time = market_open_time
        self.post_open_news_cutoff = post_open_news_cutoff
        self.universe_metadata_csv = universe_metadata_csv
        self._supplied_articles = articles_df.copy() if articles_df is not None else None

    def _load(self):
        if self._loaded:
            return

        if self._supplied_articles is not None:
            df = self._supplied_articles.copy()
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df[df["timestamp"].notna()].copy()
            self._data = df
            self._loaded = True
            return

        try:
            df = normalize_historical_sentiment_csv(
                self.csv_path,
                universe_metadata_csv=self.universe_metadata_csv,
                market_open_cutoff=self.market_open_time,
                post_open_cutoff=self.post_open_news_cutoff,
            )
            self._data = df
            logger.info("Loaded normalized sentiment rows: %s", len(df))
        except Exception as exc:
            logger.warning("Failed to load sentiment CSV: %s", exc)
            self._data = pd.DataFrame()
        self._loaded = True

    def get_features(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        self._load()
        features = pd.DataFrame(0.0, index=dates, columns=SENTIMENT_FEATURE_NAMES)

        symbol = symbol.upper()
        metadata = get_symbol_metadata(symbol, self.universe_metadata_csv)
        industry = metadata.get("industry", "")
        trade_date_col = "premarket_trade_date" if self.mode == "premarket" else "post_open_trade_date"

        if self._data is not None and not self._data.empty:
            symbol_data = self._data[self._data["symbol"].str.upper() == symbol].copy()
            if not symbol_data.empty:
                daily_sent = self._aggregate_daily(symbol_data, trade_date_col, dates)
                self._fill_stock_sentiment(features, daily_sent)

            if industry:
                industry_data = self._data[self._data["industry"] == industry].copy()
                if not industry_data.empty:
                    industry_daily = self._aggregate_daily(industry_data, trade_date_col, dates)
                    self._fill_industry_sentiment(features, industry_daily)

        features["industry_sentiment_stock_divergence"] = (
            features["premarket_sentiment"] - features["industry_premarket_sentiment"]
        ).clip(-1, 1)
        features["sentiment_price_div"] = 0.0
        features["sentiment_macro_agreement"] = 0.0

        if self.market_builder is not None:
            try:
                india_feats = self.market_builder.get_india_market_features(dates, symbol=symbol, industry=industry)
                for key, values in india_feats.items():
                    if key in features.columns:
                        features[key] = values

                from intradaynet.features.market_features import MARKET_FEATURE_NAMES

                market_feats = self.market_builder.get_features(dates)
                for col in MARKET_FEATURE_NAMES:
                    if col in features.columns and col in market_feats.columns:
                        features[col] = market_feats[col]

                macro_sign = np.sign(features.get("global_cue", pd.Series(0.0, index=dates))).replace(0, 1)
                sent_sign = np.sign(features["premarket_sentiment"]).replace(0, 0)
                features["sentiment_macro_agreement"] = (sent_sign * macro_sign).clip(-1, 1)
            except Exception as exc:
                logger.warning("Failed to compute market features: %s", exc)

        return features.fillna(0.0)

    def _aggregate_daily(self, data: pd.DataFrame, trade_date_col: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        grouped = data.groupby(trade_date_col).agg(
            mean_score=("score", "mean"),
            count=("score", "count"),
            max_score=("score", "max"),
            std_score=("score", "std"),
        )
        grouped.index = pd.to_datetime(grouped.index)
        grouped = grouped.reindex(dates).fillna(0.0)
        return grouped

    def _fill_stock_sentiment(self, features: pd.DataFrame, daily_sent: pd.DataFrame) -> None:
        features["premarket_sentiment"] = daily_sent["mean_score"].clip(-1, 1)
        features["premarket_sentiment_count"] = (daily_sent["count"] / 10.0).clip(0, 5)
        features["premarket_sentiment_max"] = daily_sent["max_score"].clip(-1, 1)
        features["premarket_sentiment_std"] = daily_sent["std_score"].clip(0, 1)
        features["sentiment_5d_avg"] = daily_sent["mean_score"].rolling(5, min_periods=1).mean().clip(-1, 1)
        sent_21d = daily_sent["mean_score"].rolling(21, min_periods=1).mean()
        features["sentiment_momentum"] = (features["sentiment_5d_avg"] - sent_21d).clip(-1, 1)
        sent_5d_std = daily_sent["mean_score"].rolling(5, min_periods=1).std().replace(0, 1)
        features["sentiment_spike"] = (
            (daily_sent["mean_score"] - features["sentiment_5d_avg"]) / sent_5d_std
        ).clip(-3, 3)
        count_mean = daily_sent["count"].rolling(10, min_periods=1).mean().replace(0, 1)
        features["news_volume_shock"] = (daily_sent["count"] / count_mean - 1).clip(-2, 5)
        features["sentiment_surprise"] = (daily_sent["mean_score"] - features["sentiment_5d_avg"]).clip(-1, 1)
        features["sentiment_confidence"] = (
            np.log1p(daily_sent["count"]) / (1.0 + daily_sent["std_score"].fillna(0))
        ).clip(0, 5) / 5.0

    def _fill_industry_sentiment(self, features: pd.DataFrame, industry_daily: pd.DataFrame) -> None:
        features["industry_premarket_sentiment"] = industry_daily["mean_score"].clip(-1, 1)
        features["industry_premarket_sentiment_count"] = (industry_daily["count"] / 20.0).clip(0, 5)
        features["industry_sentiment_5d_avg"] = (
            industry_daily["mean_score"].rolling(5, min_periods=1).mean().clip(-1, 1)
        )
        industry_21d = industry_daily["mean_score"].rolling(21, min_periods=1).mean()
        features["industry_sentiment_momentum"] = (
            features["industry_sentiment_5d_avg"] - industry_21d
        ).clip(-1, 1)
        count_mean = industry_daily["count"].rolling(10, min_periods=1).mean().replace(0, 1)
        features["industry_news_volume_shock"] = (industry_daily["count"] / count_mean - 1).clip(-2, 5)
        features["industry_sentiment_surprise"] = (
            industry_daily["mean_score"] - features["industry_sentiment_5d_avg"]
        ).clip(-1, 1)
