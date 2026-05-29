from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity.v7 import (
    compute_daily_targets_from_minute,
    compute_directional_targets,
    compute_horizon_targets,
    compute_horizon_targets_batched,
    compute_trade_levels,
    evaluate_readiness,
    executable_edge_from_prediction,
    extract_sessions,
    feature_staleness_bdays,
    margin_adjusted_confidence,
    select_candidates,
)


@dataclass
class DummyCandidate:
    score: float
    preferred_filter_pass: bool


def test_directional_targets_are_mutually_exclusive():
    daily = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [102.5, 100.8, 101.1],
            "low": [99.2, 96.5, 98.9],
            "close": [101.8, 97.2, 100.2],
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    )

    targets = compute_directional_targets(daily, target_pct=0.015)

    assert list(targets["trade_label"]) == ["LONG", "SHORT", "NO_TRADE"]
    assert ((targets["long_target"] + targets["short_target"] + targets["no_trade_target"]) == 1).all()


def test_executable_edge_and_trade_levels_use_strategy_bounds():
    edge = executable_edge_from_prediction(0.04, target_pct=0.015)
    target_price, stop_price = compute_trade_levels(
        reference_price=100.0,
        direction="LONG",
        target_pct=0.015,
        stop_loss_pct=0.01,
    )

    assert round(edge, 6) == 0.015
    assert round(target_price, 2) == 101.5
    assert round(stop_price, 2) == 99.0


def test_margin_adjusted_confidence_rewards_clear_side():
    strong = margin_adjusted_confidence(0.78, 0.22)
    weak = margin_adjusted_confidence(0.78, 0.70)

    assert strong > weak


def test_feature_staleness_counts_business_days_behind_expected_pick_date():
    assert feature_staleness_bdays(pd.Timestamp("2026-04-21"), pd.Timestamp("2026-04-22")) == 0
    assert feature_staleness_bdays(pd.Timestamp("2026-04-20"), pd.Timestamp("2026-04-22")) == 1


def test_select_candidates_prefers_preferred_before_fallbacks():
    candidates = [
        DummyCandidate(score=0.9, preferred_filter_pass=False),
        DummyCandidate(score=0.7, preferred_filter_pass=True),
        DummyCandidate(score=0.6, preferred_filter_pass=True),
    ]

    strict = select_candidates(candidates, count=2, allow_below_preferred=False)
    fallback = select_candidates(candidates, count=3, allow_below_preferred=True)

    assert [candidate.score for candidate in strict] == [0.7, 0.6]
    assert [candidate.score for candidate in fallback] == [0.7, 0.6, 0.9]


def test_readiness_ready_when_locked_forward_and_freshness_pass():
    readiness = evaluate_readiness(
        locked_backtest_summary={
            "mode": "premarket",
            "exact_logic_match": True,
            "runtime_metrics": {"total_runtime_seconds": 42.0},
            "total_net_pnl": 44834.64,
            "target_touched_intraday_hit_rate": 0.384,
            "target_before_stop_rate": 0.261,
            "win_rate": 0.641,
        },
        forward_summary={
            "mode": "premarket",
            "runtime_metrics": {"total_runtime_seconds": 39.0},
            "total_net_pnl": 6056.13,
            "target_touched_intraday_hit_rate": 0.333,
            "target_before_stop_rate": 0.18,
            "win_rate": 0.61,
        },
        target_alignment=True,
        mode="premarket",
        freshness_ok=True,
    )

    assert readiness.status == "READY"


def test_readiness_post_open_needs_runtime_and_mode_parity():
    readiness = evaluate_readiness(
        locked_backtest_summary={
            "mode": "post-open",
            "exact_logic_match": True,
            "runtime_metrics": {"total_runtime_seconds": 130.0},
            "total_net_pnl": 44834.64,
            "target_touched_intraday_hit_rate": 0.384,
            "target_before_stop_rate": 0.261,
            "win_rate": 0.641,
        },
        forward_summary={
            "mode": "post-open",
            "runtime_metrics": {"total_runtime_seconds": 125.0},
            "total_net_pnl": 6056.13,
            "target_touched_intraday_hit_rate": 0.333,
            "target_before_stop_rate": 0.18,
            "win_rate": 0.61,
        },
        target_alignment=True,
        mode="post-open",
        freshness_ok=True,
        live_symbols=25,
        processed_symbols=50,
    )

    assert readiness.status == "SMALL_LIVE"


def test_horizon_targets_point_in_time_no_lookahead():
    """Horizon targets at bar t must only use data from [t, t+H], not full session."""
    minutes = pd.date_range("2025-01-01 09:15", periods=30, freq="1min")
    prices = [100.0 + i * 0.1 for i in range(30)]
    prices[25] = 105.0
    minute_df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
        },
        index=minutes,
    )

    targets = compute_horizon_targets(minute_df, horizon_bars=5, target_pct=0.02)

    assert len(targets) == 30
    assert set(targets["trade_label"].unique()).issubset({"LONG", "SHORT", "NO_TRADE"})
    label_counts = targets["trade_label"].value_counts().to_dict()
    assert label_counts.get("LONG", 0) + label_counts.get("SHORT", 0) > 0

    final_5_close = minute_df["close"].iloc[-5:].values
    final_5_future_high = targets["long_executable_move"].iloc[-5:].values + 0.0018
    assert all(f <= 0.02 for f in final_5_future_high)


def test_horizon_targets_batched_produces_all_horizons():
    """Batched horizon target function produces correct keys."""
    minutes = pd.date_range("2025-01-01 09:15", periods=60, freq="1min")
    prices = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.05, 60))
    prices = np.maximum(prices, 80.0)
    minute_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
        },
        index=minutes,
    )

    results = compute_horizon_targets_batched(
        minute_df,
        horizons={"H15": 15, "H30": 30},
    )

    assert set(results.keys()) == {"H15", "H30"}
    for name, df in results.items():
        assert len(df) == 60
        assert "trade_label" in df.columns
        assert "trade_side_code" in df.columns


def test_horizon_targets_labels_are_mutually_exclusive():
    """Each bar gets exactly one of LONG, SHORT, or NO_TRADE."""
    minutes = pd.date_range("2025-01-01 09:15", periods=50, freq="1min")
    rng = np.random.default_rng(42)
    prices = 100.0 + rng.normal(0, 0.1, 50).cumsum()
    prices = np.maximum(prices, 80.0)
    minute_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + abs(rng.normal(0.2, 0.05, 50)),
            "low": prices - abs(rng.normal(0.2, 0.05, 50)),
            "close": prices,
        },
        index=minutes,
    )

    targets = compute_horizon_targets(minute_df, horizon_bars=10)

    assert ((targets["long_target"] + targets["short_target"] + targets["no_trade_target"]) == 1).all()
    for label in targets["trade_label"]:
        assert label in ("LONG", "SHORT", "NO_TRADE")


def test_extract_sessions_splits_multiday():
    """extract_sessions should split a multi-day DF into per-day sessions."""
    minutes = pd.date_range("2025-01-01 09:15", periods=30, freq="1min").union(
        pd.date_range("2025-01-02 09:15", periods=30, freq="1min")
    )
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        },
        index=minutes,
    )

    sessions = extract_sessions(df)
    assert len(sessions) == 2
    assert sessions[0].index[0].date() == pd.Timestamp("2025-01-01").date()
    assert sessions[1].index[0].date() == pd.Timestamp("2025-01-02").date()


def test_compute_daily_targets_from_minute():
    """Daily targets derived from minute data should match the shape of daily data."""
    rng = np.random.default_rng(42)
    minutes = pd.date_range("2025-01-01 09:15", periods=375, freq="1min").union(
        pd.date_range("2025-01-02 09:15", periods=375, freq="1min")
    )
    prices = 100.0 + rng.normal(0, 0.15, len(minutes)).cumsum()
    prices = np.maximum(prices, 80.0)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1000,
        },
        index=minutes,
    )

    daily = compute_daily_targets_from_minute(df)

    assert len(daily) == 2
    assert "trade_label" in daily.columns
    assert "long_executable_move" in daily.columns
    assert set(daily["trade_label"].unique()).issubset({"LONG", "SHORT", "NO_TRADE"})
