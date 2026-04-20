from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_VERSION = "v7_directional_executable_v1"
FEATURE_VERSION = "open_safe_daily_v7"
CALIBRATION_VERSION = "margin_adjusted_v1"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    stop_loss_pct: float
    target_pct: float
    trailing_start: float
    trailing_stop_pct: float
    min_confidence: float
    min_predicted_magnitude: float


@dataclass(frozen=True)
class ReadinessAssessment:
    status: str
    reasons: tuple[str, ...]
    checks: dict[str, bool]
    metrics: dict[str, float | int | str | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "checks": self.checks,
            "metrics": self.metrics,
        }


DEFAULT_STRATEGY_PROFILES: dict[str, StrategyConfig] = {
    "conservative": StrategyConfig(
        name="Conservative",
        stop_loss_pct=0.005,
        target_pct=0.010,
        trailing_start=0.005,
        trailing_stop_pct=0.003,
        min_confidence=0.67,
        min_predicted_magnitude=0.008,
    ),
    "balanced": StrategyConfig(
        name="Balanced",
        stop_loss_pct=0.010,
        target_pct=0.015,
        trailing_start=0.008,
        trailing_stop_pct=0.005,
        min_confidence=0.65,
        min_predicted_magnitude=0.010,
    ),
    "aggressive": StrategyConfig(
        name="Aggressive",
        stop_loss_pct=0.020,
        target_pct=0.025,
        trailing_start=0.015,
        trailing_stop_pct=0.010,
        min_confidence=0.58,
        min_predicted_magnitude=0.012,
    ),
}


def compute_directional_targets(
    daily_df: pd.DataFrame,
    *,
    target_pct: float = 0.015,
    min_tradable_move_pct: float = 0.0075,
    cost_buffer_pct: float = 0.0018,
    ambiguity_band_pct: float = 0.0025,
) -> pd.DataFrame:
    targets = pd.DataFrame(index=daily_df.index)
    open_prices = daily_df["open"].replace(0, np.nan)
    targets["max_up"] = ((daily_df["high"] - daily_df["open"]) / open_prices).clip(lower=0)
    targets["max_down"] = ((daily_df["open"] - daily_df["low"]) / open_prices).clip(lower=0)
    targets["gap"] = (daily_df["open"] - daily_df["close"].shift(1)) / daily_df["close"].shift(1).replace(0, np.nan)
    targets["close_return"] = (daily_df["close"] - daily_df["open"]) / open_prices

    long_move = (targets["max_up"] - cost_buffer_pct).clip(lower=0, upper=target_pct)
    short_move = (targets["max_down"] - cost_buffer_pct).clip(lower=0, upper=target_pct)
    edge_gap = (long_move - short_move).abs()
    best_move = np.maximum(long_move, short_move)

    trade_label = pd.Series("NO_TRADE", index=daily_df.index, dtype="object")
    long_mask = (long_move >= min_tradable_move_pct) & (long_move > short_move + ambiguity_band_pct)
    short_mask = (short_move >= min_tradable_move_pct) & (short_move > long_move + ambiguity_band_pct)
    trade_label.loc[long_mask] = "LONG"
    trade_label.loc[short_mask] = "SHORT"

    targets["long_executable_move"] = long_move
    targets["short_executable_move"] = short_move
    targets["trade_edge"] = best_move
    targets["edge_gap"] = edge_gap
    targets["trade_label"] = trade_label
    targets["trade_side_code"] = trade_label.map({"LONG": 1, "SHORT": -1, "NO_TRADE": 0}).astype(int)
    targets["long_target"] = (trade_label == "LONG").astype(int)
    targets["short_target"] = (trade_label == "SHORT").astype(int)
    targets["no_trade_target"] = (trade_label == "NO_TRADE").astype(int)
    return targets


def margin_adjusted_confidence(primary_probability: float, secondary_probability: float) -> float:
    margin = max(primary_probability - secondary_probability, 0.0)
    score = (0.75 * primary_probability) + (0.25 * margin)
    return float(np.clip(score, 0.0, 0.999))


def executable_edge_from_prediction(
    predicted_magnitude: float,
    *,
    target_pct: float,
    cost_buffer_pct: float = 0.0018,
) -> float:
    return float(np.clip(predicted_magnitude - cost_buffer_pct, 0.0, target_pct))


def score_candidate(confidence: float, executable_edge: float) -> float:
    return float(confidence * max(executable_edge, 1e-6))


def compute_trade_levels(
    *,
    reference_price: float,
    direction: str,
    target_pct: float,
    stop_loss_pct: float,
) -> tuple[float, float]:
    if direction == "LONG":
        return (
            reference_price * (1 + target_pct),
            reference_price * (1 - stop_loss_pct),
        )
    return (
        reference_price * (1 - target_pct),
        reference_price * (1 + stop_loss_pct),
    )


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def default_readiness_paths(project_root: Path) -> dict[str, Path]:
    return {
        "locked_backtest": project_root
        / "results"
        / "backtests"
        / "backtest_results_nifty500_universe"
        / "summary_intraday_model_balanced_2025-01-01_2025-12-31.json",
        "forward_blind": project_root
        / "results"
        / "backtests"
        / "backtest_results_q1_2026_refreshed"
        / "summary_intraday_model_balanced_2026-01-01_2026-03-31.json",
    }


def evaluate_readiness(
    *,
    locked_backtest_summary: dict[str, Any] | None,
    forward_summary: dict[str, Any] | None,
    target_alignment: bool,
    mode: str,
    freshness_ok: bool,
    live_symbols: int = 0,
    processed_symbols: int = 0,
) -> ReadinessAssessment:
    reasons: list[str] = []
    checks = {
        "target_alignment": bool(target_alignment),
        "freshness_ok": bool(freshness_ok),
        "locked_positive": False,
        "forward_positive": False,
        "hit_rate_ok": False,
        "target_before_stop_ok": False,
        "live_data_ok": mode != "post-open",
    }
    metrics: dict[str, float | int | str | None] = {
        "locked_net_pnl": None,
        "forward_net_pnl": None,
        "locked_win_rate": None,
        "forward_win_rate": None,
        "locked_hit_rate": None,
        "forward_hit_rate": None,
        "locked_target_before_stop": None,
        "forward_target_before_stop": None,
        "live_symbols": int(live_symbols),
        "processed_symbols": int(processed_symbols),
        "mode": mode,
    }

    if not target_alignment:
        reasons.append("Displayed targets are not aligned with execution logic.")

    if not freshness_ok:
        reasons.append("Latest usable market data is stale for the requested trading date.")

    if locked_backtest_summary:
        locked_net = float(locked_backtest_summary.get("total_net_pnl", 0.0))
        locked_hit_rate = float(locked_backtest_summary.get("target_touched_intraday_hit_rate", 0.0))
        locked_tbs = float(locked_backtest_summary.get("target_before_stop_rate", 0.0))
        locked_win_rate = float(locked_backtest_summary.get("win_rate", 0.0))
        metrics.update(
            {
                "locked_net_pnl": locked_net,
                "locked_win_rate": locked_win_rate,
                "locked_hit_rate": locked_hit_rate,
                "locked_target_before_stop": locked_tbs,
            }
        )
        checks["locked_positive"] = locked_net > 0
        checks["hit_rate_ok"] = locked_hit_rate >= 0.30
        checks["target_before_stop_ok"] = locked_tbs >= 0.18
        if locked_net <= 0:
            reasons.append("Locked backtest is not positive after costs.")
        if locked_hit_rate < 0.30:
            reasons.append("Locked backtest hit rate is below the 30% floor.")
        if locked_tbs < 0.18:
            reasons.append("Locked backtest target-before-stop rate is below the 18% floor.")
    else:
        reasons.append("Locked backtest summary is missing.")

    if forward_summary:
        forward_net = float(forward_summary.get("total_net_pnl", 0.0))
        forward_hit_rate = float(forward_summary.get("target_touched_intraday_hit_rate", 0.0))
        forward_tbs = float(forward_summary.get("target_before_stop_rate", 0.0))
        forward_win_rate = float(forward_summary.get("win_rate", 0.0))
        metrics.update(
            {
                "forward_net_pnl": forward_net,
                "forward_win_rate": forward_win_rate,
                "forward_hit_rate": forward_hit_rate,
                "forward_target_before_stop": forward_tbs,
            }
        )
        checks["forward_positive"] = forward_net > 0
        if forward_net <= 0:
            reasons.append("Forward blind test is not positive after costs.")
    else:
        reasons.append("Forward blind test summary is missing.")

    if mode == "post-open":
        live_ok = live_symbols > 0 and live_symbols >= max(5, int(processed_symbols * 0.20))
        checks["live_data_ok"] = live_ok
        if not live_ok:
            reasons.append("Post-open mode does not have enough live symbols to trust the run.")

    if all(checks.values()):
        status = "READY"
    elif checks["target_alignment"] and checks["freshness_ok"] and checks["locked_positive"]:
        status = "PAPER_ONLY"
    else:
        status = "NOT_READY"

    return ReadinessAssessment(
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        checks=checks,
        metrics=metrics,
    )


def strategy_config_to_dict(config: StrategyConfig) -> dict[str, Any]:
    return asdict(config)
