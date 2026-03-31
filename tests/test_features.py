"""
Tests for feature computation modules.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intradaynet.features.per_bar_features import (
    compute_per_bar_features,
    PER_BAR_FEATURE_NAMES,
)
from intradaynet.features.session_features import (
    compute_session_features,
    SESSION_FEATURE_NAMES,
)
from intradaynet.features.sentiment_features import (
    SentimentFeatureBuilder,
    SENTIMENT_FEATURE_NAMES,
)


def _make_minute_df(n_days=3, bars_per_day=375, base_price=100.0):
    """Create a synthetic minute-bar DataFrame for testing."""
    dates = []
    for d in range(n_days):
        day = pd.Timestamp(f"2024-01-{d+2:02d}")
        for bar in range(bars_per_day):
            hour = 9 + (bar + 15) // 60  # start from 9:15
            minute = (bar + 15) % 60
            if hour >= 16:
                break
            ts = day.replace(hour=hour, minute=minute)
            dates.append(ts)

    n = len(dates)
    np.random.seed(42)

    prices = base_price + np.cumsum(np.random.randn(n) * 0.1)
    prices = np.maximum(prices, 1.0)

    df = pd.DataFrame({
        "open": prices + np.random.randn(n) * 0.05,
        "high": prices + abs(np.random.randn(n) * 0.1),
        "low": prices - abs(np.random.randn(n) * 0.1),
        "close": prices,
        "volume": np.random.randint(1000, 100000, n).astype(float),
    }, index=pd.DatetimeIndex(dates))

    # Ensure high >= open, close and low <= open, close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


class TestPerBarFeatures:
    def test_feature_count(self):
        df = _make_minute_df()
        features = compute_per_bar_features(df)
        assert len(features.columns) == 25, \
            f"Expected 25 features, got {len(features.columns)}"

    def test_feature_names(self):
        df = _make_minute_df()
        features = compute_per_bar_features(df)
        assert list(features.columns) == PER_BAR_FEATURE_NAMES

    def test_no_nans(self):
        df = _make_minute_df()
        features = compute_per_bar_features(df)
        nan_counts = features.isna().sum()
        assert nan_counts.sum() == 0, f"NaN features: {nan_counts[nan_counts > 0]}"

    def test_no_infinities(self):
        df = _make_minute_df()
        features = compute_per_bar_features(df)
        inf_counts = np.isinf(features.values).sum()
        assert inf_counts == 0, "Infinite values found in features"

    def test_rsi_range(self):
        df = _make_minute_df()
        features = compute_per_bar_features(df)
        rsi = features["rsi_14"]
        assert rsi.min() >= 0 and rsi.max() <= 1, \
            f"RSI should be in [0,1], got [{rsi.min():.3f}, {rsi.max():.3f}]"

    def test_time_normalized_range(self):
        df = _make_minute_df()
        features = compute_per_bar_features(df)
        tn = features["time_normalized"]
        assert tn.min() >= 0 and tn.max() <= 1, \
            f"time_normalized should be [0,1], got [{tn.min():.3f}, {tn.max():.3f}]"

    def test_same_length_as_input(self):
        df = _make_minute_df()
        features = compute_per_bar_features(df)
        assert len(features) == len(df)


class TestSessionFeatures:
    def test_feature_count(self):
        df = _make_minute_df(n_days=10)
        features = compute_session_features(df)
        assert len(features.columns) == 20, \
            f"Expected 20 features, got {len(features.columns)}"

    def test_feature_names(self):
        df = _make_minute_df(n_days=10)
        features = compute_session_features(df)
        assert list(features.columns) == SESSION_FEATURE_NAMES

    def test_no_nans(self):
        df = _make_minute_df(n_days=10)
        features = compute_session_features(df)
        nan_counts = features.isna().sum()
        assert nan_counts.sum() == 0, f"NaN features: {nan_counts[nan_counts > 0]}"

    def test_one_row_per_day(self):
        n_days = 5
        df = _make_minute_df(n_days=n_days)
        features = compute_session_features(df)
        assert len(features) == n_days


class TestSentimentFeatures:
    def test_feature_count_no_csv(self):
        builder = SentimentFeatureBuilder("/nonexistent/path.csv")
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        features = builder.get_features("RELIANCE", dates)
        assert len(features.columns) == 14

    def test_all_zeros_when_no_data(self):
        builder = SentimentFeatureBuilder("/nonexistent/path.csv")
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        features = builder.get_features("RELIANCE", dates)
        assert (features.values == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
