"""
Shared feature contract for ALL IntradayNet model pipelines.

This module is the single source of truth for:
- flattened intraday feature names (LightGBM backend)
- daily feature names (open-safe premarket model)
- feature order, versioning, and schema validation
- training/inference feature generation

Pipeline mapping:
    ┌─ LightGBM backend (live) ── uses FEATURE_NAMES (flattened intraday)
    │   • input: 120-bar × 69 raw features per minute
    │   • output: ~669 flattened features with rolling window stats
    │   • used by: model_bundle.py, recommend_live.py, train_live_backend.py
    │
    └─ Open-safe premarket model ── uses DAILY_FEATURE_NAMES
        • input: daily OHLCV + market + sentiment aggregates
        • output: ~85 daily features
        • used by: open_safe_daily_features.py, recommend_intraday.py

All training, validation, inference, and backtesting code should go through
this contract so we never silently drift into a schema mismatch again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from intradaynet.features.per_bar_features import PER_BAR_FEATURE_NAMES
from intradaynet.features.session_features import SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES


FEATURE_SCHEMA_VERSION = "live_v2"
DAILY_FEATURE_SCHEMA_VERSION = "daily_v2"
FLAT_WINDOWS = (5, 15, 30, 60, 120)
FLAT_STATS = ("mean", "std", "min", "max")

# ── Daily-level feature names (open-safe premarket model) ──

DAILY_PRICE_ACTION_FEATURES = [
    "prev_day_return",
    "prev_day_volatility",
    "prev_day_range",
    "prev_day_atr",
    "overnight_gap",
    "prev_gap_size",
    "prev_gap_direction",
    "price_momentum_5d",
    "price_momentum_10d",
    "price_momentum_20d",
    "close_vs_day_high",
    "close_vs_day_low",
    "range_expansion_5d",
    "close_vs_vwap",
    "vwap",
    "sector_relative_strength",
    "stock_vs_sector_1d",
    "stock_vs_sector_5d",
    "breadth_momentum_confirmation",
    "volatility_normalized_gap",
    "volatility_normalized_momentum_5d",
]

DAILY_VOLUME_FEATURES = [
    "prev_volume",
    "volume",
    "vol_momentum",
    "volume_zscore",
]

DAILY_FIB_FEATURES = [
    "swing_range_5d",
    "swing_range_20d",
    "prior_swing_position_5d",
    "prior_swing_position_20d",
]
for lookback in ("5d", "20d"):
    for fib_ratio in ("236", "382", "500", "618", "786"):
        DAILY_FIB_FEATURES.append(f"fib_{fib_ratio}_dist_{lookback}")
    DAILY_FIB_FEATURES.append(f"fib_confluence_{lookback}")
    DAILY_FIB_FEATURES.append(f"distance_to_nearest_fib_{lookback}")

DAILY_MARKET_FEATURES = [
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

DAILY_INDIA_FEATURES = [
    "nifty_intraday_return",
    "sector_intraday_return",
    "vix_level",
    "vix_change",
    "market_breadth",
    "global_cue",
    "sector_index_prev_return",
    "sector_index_5d_return",
    "sector_index_volatility",
    "industry_relative_strength_rank",
    "sector_breadth_proxy",
    "secondary_sector_confirmation",
]

DAILY_META_FEATURES = [
    "feature_version_code",
]

_DAILY_FEATURE_NAMES_RAW: list[str] = (
    DAILY_PRICE_ACTION_FEATURES
    + DAILY_VOLUME_FEATURES
    + DAILY_FIB_FEATURES
    + DAILY_MARKET_FEATURES
    + DAILY_INDIA_FEATURES
    + list(SENTIMENT_FEATURE_NAMES)
    + DAILY_META_FEATURES
)

DAILY_FEATURE_NAMES: list[str] = list(dict.fromkeys(_DAILY_FEATURE_NAMES_RAW))


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    feature_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


@dataclass
class FeatureRegistry:
    """
    Canonical registry of all feature names across both pipelines.

    Usage:
        registry = get_feature_registry()
        assert model_input.shape[1] == registry.intraday.feature_count
    """
    intraday: FeatureSchema = field(default_factory=lambda: FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        feature_names=tuple(FEATURE_NAMES),
    ))
    daily: FeatureSchema = field(default_factory=lambda: FeatureSchema(
        version=DAILY_FEATURE_SCHEMA_VERSION,
        feature_names=tuple(DAILY_FEATURE_NAMES),
    ))

    def validate_daily_frame(self, columns: list[str]) -> list[str]:
        """Check daily DataFrame columns against the contract. Returns missing features."""
        expected = set(self.daily.feature_names)
        actual = set(columns)
        missing = expected - actual
        return sorted(missing)

    def validate_intraday_vector(self, length: int) -> bool:
        """Check that a flat feature vector has the expected length."""
        return length == self.intraday.feature_count


def get_feature_registry() -> FeatureRegistry:
    return FeatureRegistry()


def build_feature_names() -> list[str]:
    names: list[str] = []

    for feature_name in PER_BAR_FEATURE_NAMES:
        for window in FLAT_WINDOWS:
            for stat in FLAT_STATS:
                names.append(f"{feature_name}_{stat}_{window}")
        names.append(f"{feature_name}_last")
        names.append(f"{feature_name}_first")
        names.append(f"{feature_name}_diff_last_first")
        names.append(f"{feature_name}_diff_5v30")
        names.append(f"{feature_name}_diff_15v60")

    names.extend(f"session::{name}" for name in SESSION_FEATURE_NAMES)
    names.extend(f"sentiment::{name}" for name in SENTIMENT_FEATURE_NAMES)
    return names


FEATURE_NAMES = build_feature_names()
FEATURE_SCHEMA = FeatureSchema(
    version=FEATURE_SCHEMA_VERSION,
    feature_names=tuple(FEATURE_NAMES),
)


def _safe_stat(segment: np.ndarray, stat: str) -> float:
    if stat == "mean":
        return float(np.nanmean(segment))
    if stat == "std":
        return float(np.nanstd(segment))
    if stat == "min":
        return float(np.nanmin(segment))
    if stat == "max":
        return float(np.nanmax(segment))
    raise ValueError(f"Unsupported stat: {stat}")


def flatten_intraday_window(
    window: np.ndarray,
    session: Sequence[float],
    sentiment: Sequence[float],
) -> np.ndarray:
    """
    Flatten a single (L, F) window into the shared backend feature vector.

    Used by: LightGBM backend (live recommendation pipeline)
    """
    if window.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {window.shape}")
    if window.shape[1] != len(PER_BAR_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(PER_BAR_FEATURE_NAMES)} per-bar features, got {window.shape[1]}"
        )

    session_arr = np.asarray(session, dtype=np.float32)
    sentiment_arr = np.asarray(sentiment, dtype=np.float32)

    if session_arr.shape[0] != len(SESSION_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(SESSION_FEATURE_NAMES)} session features, got {session_arr.shape[0]}"
        )
    if sentiment_arr.shape[0] != len(SENTIMENT_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(SENTIMENT_FEATURE_NAMES)} sentiment features, got {sentiment_arr.shape[0]}"
        )

    length = window.shape[0]
    values: list[float] = []

    for col_idx in range(window.shape[1]):
        column = window[:, col_idx]

        for flat_window in FLAT_WINDOWS:
            segment = column[-flat_window:] if flat_window <= length else column
            for stat in FLAT_STATS:
                values.append(_safe_stat(segment, stat))

        values.append(float(column[-1]))
        values.append(float(column[0]))
        values.append(float(column[-1] - column[0]))

        last_5 = column[-5:] if length >= 5 else column
        last_30 = column[-30:] if length >= 30 else column
        last_15 = column[-15:] if length >= 15 else column
        last_60 = column[-60:] if length >= 60 else column

        values.append(float(np.nanmean(last_5) - np.nanmean(last_30)))
        values.append(float(np.nanmean(last_15) - np.nanmean(last_60)))

    values.extend(float(v) for v in session_arr)
    values.extend(float(v) for v in sentiment_arr)

    flat = np.asarray(values, dtype=np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0)

    if flat.shape[0] != FEATURE_SCHEMA.feature_count:
        raise RuntimeError(
            f"Feature contract mismatch: built {flat.shape[0]} features, "
            f"expected {FEATURE_SCHEMA.feature_count}"
        )
    return flat


def flatten_intraday_batch(
    windows: np.ndarray,
    session_batch: np.ndarray,
    sentiment_batch: np.ndarray,
) -> np.ndarray:
    """
    Flatten a batch of windows using the exact same feature order as inference.

    Used by: LightGBM backend (live recommendation pipeline)
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D windows array, got {windows.shape}")
    if windows.shape[0] != len(session_batch) or windows.shape[0] != len(sentiment_batch):
        raise ValueError("Batch size mismatch between windows, session, and sentiment")

    batch = [
        flatten_intraday_window(windows[i], session_batch[i], sentiment_batch[i])
        for i in range(windows.shape[0])
    ]
    return np.stack(batch, axis=0).astype(np.float32)
