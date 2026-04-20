from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intradaynet.v7 import (
    compute_directional_targets,
    compute_trade_levels,
    evaluate_readiness,
    executable_edge_from_prediction,
    margin_adjusted_confidence,
)


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


def test_readiness_ready_when_locked_forward_and_freshness_pass():
    readiness = evaluate_readiness(
        locked_backtest_summary={
            "total_net_pnl": 44834.64,
            "target_touched_intraday_hit_rate": 0.384,
            "target_before_stop_rate": 0.261,
            "win_rate": 0.641,
        },
        forward_summary={
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
