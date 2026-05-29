from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from equity.feature_contract import (
    DAILY_FEATURE_NAMES,
    get_feature_registry,
)
from equity.features.market_features import MarketFeatureBuilder
from equity.features.sentiment_features import (
    SENTIMENT_FEATURE_NAMES,
    SentimentFeatureBuilder,
)
from equity.v7 import FEATURE_VERSION, TARGET_VERSION, compute_directional_targets
from equity.universe import get_symbol_metadata


DAILY_FEATURE_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "price_action": (
        "prev_day_",
        "overnight_",
        "prev_gap_",
        "price_momentum_",
        "close_vs_",
        "distance_to_",
        "fib_",
        "swing_",
        "range_",
    ),
    "volume": ("volume", "vol_", "vwap", "avg_", "prev_volume"),
    "sentiment": (
        "premarket_",
        "sentiment_",
        "news_",
    ),
    "macro": (
        "crude_",
        "gold_",
        "usdinr_",
        "us_10y_",
        "dxy_",
        "asia_",
        "dow_",
        "nasdaq_",
        "sp500_",
        "global_",
        "india_",
        "nifty_",
        "risk_",
        "commodity_",
        "dollar_",
        "market_",
        "vix_",
    ),
    "cross_sectional": (
        "sector_relative_",
        "breadth_",
        "volatility_normalized_",
    ),
}


def build_open_safe_daily_features(
    minute_df: pd.DataFrame,
    symbol: str,
    market_builder: MarketFeatureBuilder,
    sentiment_builder: SentimentFeatureBuilder,
) -> pd.DataFrame | None:
    daily = minute_df.resample("D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    if len(daily) < 30:
        return None

    features = pd.DataFrame(index=daily.index)
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]
    open_ = daily["open"]
    daily_return = close.pct_change()

    features["prev_day_return"] = daily_return.shift(1)
    features["prev_day_volatility"] = daily_return.rolling(21, min_periods=5).std().shift(1)

    daily_range = (high - low) / open_.replace(0, np.nan)
    features["prev_day_range"] = daily_range.shift(1)
    features["prev_day_atr"] = daily_range.rolling(14, min_periods=5).mean().shift(1)

    features["overnight_gap"] = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)
    features["prev_gap_size"] = features["overnight_gap"].shift(1).abs()
    features["prev_gap_direction"] = np.sign(features["overnight_gap"].shift(1)).fillna(0.0)

    work_df = minute_df.copy()
    work_df["date_only"] = work_df.index.normalize()
    vol_by_date = work_df.groupby("date_only")["volume"].sum()
    prev_day_volume = vol_by_date.reindex(features.index).shift(1)
    features["prev_volume"] = prev_day_volume
    features["volume"] = prev_day_volume
    features["vol_momentum"] = (
        prev_day_volume / prev_day_volume.rolling(20, min_periods=5).mean() - 1
    )
    features["volume_zscore"] = (
        (prev_day_volume - prev_day_volume.rolling(20, min_periods=5).mean())
        / prev_day_volume.rolling(20, min_periods=5).std().replace(0, np.nan)
    )

    work_df["tp"] = (work_df["high"] + work_df["low"] + work_df["close"]) / 3.0
    work_df["tpv"] = work_df["tp"] * work_df["volume"]
    vwap_daily = work_df.groupby("date_only").apply(
        lambda x: x["tpv"].sum() / x["volume"].sum() if x["volume"].sum() > 0 else x["close"].iloc[-1],
        include_groups=False,
    )
    vwap_daily.index = pd.to_datetime(vwap_daily.index)
    daily_close_vs_vwap = close / vwap_daily.reindex(features.index) - 1
    features["vwap"] = vwap_daily.reindex(features.index).shift(1)
    features["close_vs_vwap"] = daily_close_vs_vwap.shift(1)

    features["price_momentum_5d"] = close.pct_change(5).shift(1)
    features["price_momentum_10d"] = close.pct_change(10).shift(1)
    features["price_momentum_20d"] = close.pct_change(20).shift(1)

    features["close_vs_day_high"] = (close / high - 1).shift(1)
    features["close_vs_day_low"] = (close / low - 1).shift(1)
    features["range_expansion_5d"] = (
        daily_range.shift(1) / daily_range.rolling(5, min_periods=3).mean().shift(1).replace(0, np.nan)
    )

    _append_fibonacci_features(features, daily)

    market_feats = market_builder.get_features(features.index).shift(1)
    india_feats = {
        key: series.shift(1)
        for key, series in market_builder.get_india_market_features(features.index).items()
    }
    for col in market_feats.columns:
        features[col] = market_feats[col]
    for key, series in india_feats.items():
        features[key] = series

    sentiment = sentiment_builder.get_features(symbol, features.index)
    for col in SENTIMENT_FEATURE_NAMES:
        features[col] = sentiment[col]

    symbol_metadata = get_symbol_metadata(symbol, getattr(sentiment_builder, "universe_metadata_csv", None))
    sector_context = market_builder.get_sector_context(features.index, industry=symbol_metadata.get("industry", ""))
    for key, series in sector_context.items():
        if key in features.columns:
            features[key] = series

    features["sector_relative_strength"] = (
        features["price_momentum_5d"] - features["sector_intraday_return"]
    ).clip(-0.25, 0.25)
    features["stock_vs_sector_1d"] = (
        features["prev_day_return"] - features["sector_index_prev_return"]
    ).clip(-0.25, 0.25)
    features["stock_vs_sector_5d"] = (
        features["price_momentum_5d"] - features["sector_index_5d_return"]
    ).clip(-0.5, 0.5)
    features["breadth_momentum_confirmation"] = (
        np.sign(features["price_momentum_5d"]).fillna(0.0) * features["market_breadth"]
    ).clip(-1.0, 1.0)
    features["volatility_normalized_gap"] = (
        features["overnight_gap"] / features["prev_day_atr"].replace(0, np.nan)
    ).clip(-5.0, 5.0)
    features["volatility_normalized_momentum_5d"] = (
        features["price_momentum_5d"] / features["prev_day_volatility"].replace(0, np.nan)
    ).clip(-5.0, 5.0)
    features["feature_version_code"] = 7.0

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()
    if features.empty:
        return features

    registry = get_feature_registry()
    missing = registry.validate_daily_frame(list(features.columns))
    if missing:
        import logging
        logger = logging.getLogger("intradaynet.open_safe_daily")
        logger.warning(
            "Daily features diverge from canonical contract. Missing: %s. "
            "Columns present but not in contract will be dropped.",
            missing[:20],
        )
        canonical_cols = [c for c in DAILY_FEATURE_NAMES if c in features.columns]
        features = features[canonical_cols]

    ordered_cols = [c for c in DAILY_FEATURE_NAMES if c in features.columns]
    extra_cols = [c for c in features.columns if c not in DAILY_FEATURE_NAMES]
    features = features[ordered_cols + extra_cols]
    features.attrs["feature_schema_version"] = "daily_v2"
    return features


def compute_intraday_targets(
    daily_df: pd.DataFrame,
    target_pct: float = 0.015,
    min_tradable_move_pct: float = 0.0075,
    cost_buffer_pct: float = 0.0018,
    ambiguity_band_pct: float = 0.0025,
) -> pd.DataFrame:
    targets = compute_directional_targets(
        daily_df,
        target_pct=target_pct,
        min_tradable_move_pct=min_tradable_move_pct,
        cost_buffer_pct=cost_buffer_pct,
        ambiguity_band_pct=ambiguity_band_pct,
    )
    targets["target_version"] = TARGET_VERSION
    return targets


def build_daily_training_frame(
    minute_df: pd.DataFrame,
    symbol: str,
    market_builder: MarketFeatureBuilder,
    sentiment_builder: SentimentFeatureBuilder,
    target_pct: float = 0.01,
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    daily = minute_df.resample("D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    if len(daily) < 30:
        return None, None

    features = build_open_safe_daily_features(minute_df, symbol, market_builder, sentiment_builder)
    if features is None or features.empty:
        return None, None

    targets = compute_intraday_targets(daily, target_pct)
    valid_idx = features.index.intersection(targets.dropna().index)
    feature_frame = features.loc[valid_idx].copy()
    feature_frame.attrs["feature_version"] = FEATURE_VERSION
    targets_frame = targets.loc[valid_idx].copy()
    targets_frame.attrs["target_version"] = TARGET_VERSION
    return feature_frame, targets_frame


def classify_feature_family(feature_name: str) -> str:
    for family, prefixes in DAILY_FEATURE_FAMILY_PREFIXES.items():
        if any(feature_name.startswith(prefix) for prefix in prefixes):
            return family
    return "other"


def _append_fibonacci_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]
    lookback_map = {
        "5d": 5,
        "20d": 20,
    }
    ratio_map = {
        "236": 0.236,
        "382": 0.382,
        "500": 0.500,
        "618": 0.618,
        "786": 0.786,
    }

    prior_close = close.shift(1)
    for label, lookback in lookback_map.items():
        swing_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
        swing_low = low.rolling(lookback, min_periods=lookback).min().shift(1)
        swing_range = (swing_high - swing_low).replace(0, np.nan)
        features[f"swing_range_{label}"] = (swing_range / prior_close.replace(0, np.nan)).clip(0, 1)

        confluence = pd.Series(0.0, index=features.index)
        nearest_dist = pd.Series(np.nan, index=features.index)
        prior_pos = ((prior_close - swing_low) / swing_range).clip(0, 1)
        features[f"prior_swing_position_{label}"] = prior_pos

        for ratio_label, ratio in ratio_map.items():
            fib_level = swing_low + ratio * swing_range
            dist = ((prior_close - fib_level) / prior_close.replace(0, np.nan)).clip(-0.25, 0.25)
            features[f"fib_{ratio_label}_dist_{label}"] = dist
            confluence = confluence + (dist.abs() <= 0.005).astype(float)
            nearest_dist = pd.concat([nearest_dist.abs(), dist.abs()], axis=1).min(axis=1)

        features[f"fib_confluence_{label}"] = confluence / len(ratio_map)
        features[f"distance_to_nearest_fib_{label}"] = nearest_dist.fillna(0.25).clip(0, 0.25)
