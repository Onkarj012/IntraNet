"""
Tests for feature computation modules.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from equity.features.per_bar_features import (
    compute_per_bar_features,
    PER_BAR_FEATURE_NAMES,
)
from equity.features.session_features import (
    compute_session_features,
    SESSION_FEATURE_NAMES,
)
from equity.features.sentiment_features import (
    SentimentFeatureBuilder,
    SENTIMENT_FEATURE_NAMES,
)
from equity.features.market_features import MarketFeatureBuilder
from equity.feature_contract import (
    DAILY_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    get_feature_registry,
)
from equity.live_news import normalize_historical_sentiment_csv
from equity.universe import (
    filter_symbols_by_industry,
    get_symbol_to_industry_map,
    get_universe_metadata,
    resolve_industry_filters,
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
        assert len(features.columns) == len(SESSION_FEATURE_NAMES), \
            f"Expected {len(SESSION_FEATURE_NAMES)} features, got {len(features.columns)}"

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
        assert len(features.columns) == len(SENTIMENT_FEATURE_NAMES)

    def test_all_zeros_when_no_data(self):
        builder = SentimentFeatureBuilder("/nonexistent/path.csv")
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        features = builder.get_features("RELIANCE", dates)
        assert (features.values == 0).all()

    def test_universe_metadata_loader(self):
        metadata = get_universe_metadata()
        assert len(metadata) == 500
        assert "symbol" in metadata.columns
        assert "industry" in metadata.columns
        assert get_symbol_to_industry_map()["TCS"] == "Information Technology"

    def test_industry_filter_resolution(self):
        resolved = resolve_industry_filters(["information technology, healthcare"])
        assert "Information Technology" in resolved
        assert "Healthcare" in resolved
        filtered = filter_symbols_by_industry(["TCS", "SUNPHARMA", "RELIANCE"], ["Information Technology"])
        assert filtered == ["TCS"]

    def test_historical_news_cutoff_trade_dates(self, tmp_path: Path):
        csv_path = tmp_path / "sentiment.csv"
        pd.DataFrame(
            [
                {"Symbol": "TCS", "Publish Date": "2026-04-23 22:00:00", "sentiment_score": 0.8},
                {"Symbol": "TCS", "Publish Date": "2026-04-24 09:18:00", "sentiment_score": 0.3},
                {"Symbol": "TCS", "Publish Date": "2026-04-24 09:25:00", "sentiment_score": -0.2},
            ]
        ).to_csv(csv_path, index=False)
        normalized = normalize_historical_sentiment_csv(csv_path, post_open_cutoff="09:20")
        premarket_dates = normalized["premarket_trade_date"].dt.strftime("%Y-%m-%d").tolist()
        post_open_dates = normalized["post_open_trade_date"].dt.strftime("%Y-%m-%d").tolist()
        assert premarket_dates == ["2026-04-24", "2026-04-27", "2026-04-27"]
        assert post_open_dates == ["2026-04-24", "2026-04-24", "2026-04-27"]

    def test_industry_aggregation_and_sector_context(self, tmp_path: Path):
        csv_path = tmp_path / "sentiment.csv"
        pd.DataFrame(
            [
                {"Symbol": "TCS", "Publish Date": "2026-04-23 21:30:00", "sentiment_score": 0.8},
                {"Symbol": "INFY", "Publish Date": "2026-04-23 22:00:00", "sentiment_score": 0.4},
                {"Symbol": "SUNPHARMA", "Publish Date": "2026-04-23 22:10:00", "sentiment_score": -0.5},
            ]
        ).to_csv(csv_path, index=False)
        builder = SentimentFeatureBuilder(str(csv_path), mode="premarket")
        dates = pd.DatetimeIndex([pd.Timestamp("2026-04-24")])
        features = builder.get_features("TCS", dates)
        assert features.iloc[0]["premarket_sentiment"] > 0
        assert features.iloc[0]["industry_premarket_sentiment"] > 0
        assert features.iloc[0]["industry_premarket_sentiment_count"] > 0

        market_builder = MarketFeatureBuilder()
        sector_context = market_builder.get_sector_context(dates, industry="Information Technology")
        assert "sector_index_prev_return" in sector_context
        assert "secondary_sector_confirmation" in sector_context


class TestFeatureContract:
    def test_intraday_feature_count(self):
        assert FEATURE_SCHEMA.feature_count == len(FEATURE_NAMES)
        assert FEATURE_SCHEMA.feature_count > 600

    def test_daily_feature_names_are_defined(self):
        assert len(DAILY_FEATURE_NAMES) > 50
        assert "prev_day_return" in DAILY_FEATURE_NAMES
        assert "overnight_gap" in DAILY_FEATURE_NAMES
        assert "price_momentum_5d" in DAILY_FEATURE_NAMES

    def test_feature_registry_has_both_pipelines(self):
        registry = get_feature_registry()
        assert registry.intraday.feature_count > 0
        assert registry.daily.feature_count > 0
        assert registry.intraday.feature_count != registry.daily.feature_count

    def test_registry_validate_daily_frame(self):
        registry = get_feature_registry()
        columns = ["prev_day_return", "overnight_gap", "volume"]
        missing = registry.validate_daily_frame(columns)
        assert len(missing) > 0
        assert "prev_day_return" not in missing

    def test_registry_validate_daily_frame_all_present(self):
        registry = get_feature_registry()
        all_cols = list(registry.daily.feature_names)
        missing = registry.validate_daily_frame(all_cols)
        assert missing == []

    def test_registry_validate_intraday_vector(self):
        registry = get_feature_registry()
        assert registry.validate_intraday_vector(registry.intraday.feature_count)
        assert not registry.validate_intraday_vector(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
