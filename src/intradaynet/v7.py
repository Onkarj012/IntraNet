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
        min_confidence=0.50,
        min_predicted_magnitude=0.013,
    ),
    "balanced": StrategyConfig(
        name="Balanced",
        stop_loss_pct=0.010,
        target_pct=0.015,
        trailing_start=0.008,
        trailing_stop_pct=0.005,
        min_confidence=0.50,
        min_predicted_magnitude=0.012,
    ),
    "aggressive": StrategyConfig(
        name="Aggressive",
        stop_loss_pct=0.020,
        target_pct=0.025,
        trailing_start=0.015,
        trailing_stop_pct=0.010,
        min_confidence=0.48,
        min_predicted_magnitude=0.010,
    ),
}


def compute_horizon_targets(
    minute_df: pd.DataFrame,
    horizon_bars: int,
    *,
    target_pct: float = 0.015,
    min_tradable_move_pct: float = 0.0075,
    cost_buffer_pct: float = 0.0018,
    ambiguity_band_pct: float = 0.0025,
) -> pd.DataFrame:
    """
    Point-in-time horizon-specific target labelling.

    For each minute bar at time t, looks ahead `horizon_bars` bars and labels
    LONG / SHORT / NO_TRADE based on whether the stock reaches the target
    within [t, t + horizon_bars].

    This is the P0-correct target function that fixes the daily-label flaw.
    Each bar gets its own horizon-specific label — no look-ahead beyond the
    horizon window.

    Returns a DataFrame indexed identically to minute_df with columns:
        long_executable_move, short_executable_move, trade_edge, edge_gap,
        trade_label, trade_side_code, long_target, short_target, no_trade_target
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")

    close = minute_df["close"].values.astype(np.float64)
    high = minute_df["high"].values.astype(np.float64)
    low = minute_df["low"].values.astype(np.float64)
    n = len(minute_df)

    targets = pd.DataFrame(index=minute_df.index)

    future_high = np.full(n, np.nan, dtype=np.float64)
    future_low = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        end = min(i + horizon_bars, n)
        future_high[i] = np.max(high[i:end])
        future_low[i] = np.min(low[i:end])

    long_move = np.clip((future_high - close) / np.maximum(close, 1e-9) - cost_buffer_pct, 0, target_pct)
    short_move = np.clip((close - future_low) / np.maximum(close, 1e-9) - cost_buffer_pct, 0, target_pct)
    edge_gap = np.abs(long_move - short_move)
    best_move = np.maximum(long_move, short_move)

    trade_label = np.full(n, "NO_TRADE", dtype=object)
    long_mask = (long_move >= min_tradable_move_pct) & (long_move > short_move + ambiguity_band_pct)
    short_mask = (short_move >= min_tradable_move_pct) & (short_move > long_move + ambiguity_band_pct)
    trade_label[long_mask] = "LONG"
    trade_label[short_mask] = "SHORT"

    label_series = pd.Series(trade_label, index=minute_df.index)
    targets["long_executable_move"] = long_move
    targets["short_executable_move"] = short_move
    targets["trade_edge"] = best_move
    targets["edge_gap"] = edge_gap
    targets["trade_label"] = label_series
    targets["trade_side_code"] = label_series.map({"LONG": 1, "SHORT": -1, "NO_TRADE": 0}).astype("Int64")
    targets["long_target"] = (label_series == "LONG").astype("Int64")
    targets["short_target"] = (label_series == "SHORT").astype("Int64")
    targets["no_trade_target"] = (label_series == "NO_TRADE").astype("Int64")
    return targets


def compute_horizon_targets_batched(
    minute_df: pd.DataFrame,
    horizons: dict[str, int] | None = None,
    *,
    target_pct: float = 0.015,
    min_tradable_move_pct: float = 0.0075,
    cost_buffer_pct: float = 0.0018,
    ambiguity_band_pct: float = 0.0025,
) -> dict[str, pd.DataFrame]:
    """
    Compute horizon-specific targets for multiple horizons in a single pass.

    horizons: dict mapping horizon name → number of bars
              default: {"H15": 15, "H30": 30, "H60": 60, "H375": 375}
    """
    if horizons is None:
        horizons = {"H15": 15, "H30": 30, "H60": 60, "H375": 375}

    close = minute_df["close"].values.astype(np.float64)
    high = minute_df["high"].values.astype(np.float64)
    low = minute_df["low"].values.astype(np.float64)
    n = len(minute_df)

    max_horizon = max(horizons.values())
    rolling_high = pd.Series(high, index=minute_df.index).rolling(max_horizon, min_periods=1).max().values
    rolling_low = pd.Series(low, index=minute_df.index).rolling(max_horizon, min_periods=1).min().values

    results: dict[str, pd.DataFrame] = {}
    for name, horizon_bars in horizons.items():
        targets = pd.DataFrame(index=minute_df.index)

        future_high = np.full(n, np.nan, dtype=np.float64)
        future_low = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            end = min(i + horizon_bars, n)
            future_high[i] = np.max(high[i:end])
            future_low[i] = np.min(low[i:end])

        long_move = np.clip((future_high - close) / np.maximum(close, 1e-9) - cost_buffer_pct, 0, target_pct)
        short_move = np.clip((close - future_low) / np.maximum(close, 1e-9) - cost_buffer_pct, 0, target_pct)
        edge_gap_val = np.abs(long_move - short_move)
        best_move = np.maximum(long_move, short_move)

        trade_label = np.full(n, "NO_TRADE", dtype=object)
        long_mask = (long_move >= min_tradable_move_pct) & (long_move > short_move + ambiguity_band_pct)
        short_mask = (short_move >= min_tradable_move_pct) & (short_move > long_move + ambiguity_band_pct)
        trade_label[long_mask] = "LONG"
        trade_label[short_mask] = "SHORT"

        label_series = pd.Series(trade_label, index=minute_df.index)
        targets["long_executable_move"] = long_move
        targets["short_executable_move"] = short_move
        targets["trade_edge"] = best_move
        targets["edge_gap"] = edge_gap_val
        targets["trade_label"] = label_series
        targets["trade_side_code"] = label_series.map({"LONG": 1, "SHORT": -1, "NO_TRADE": 0}).astype("Int64")
        targets["long_target"] = (label_series == "LONG").astype("Int64")
        targets["short_target"] = (label_series == "SHORT").astype("Int64")
        targets["no_trade_target"] = (label_series == "NO_TRADE").astype("Int64")
        results[name] = targets

    return results


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


def expected_feature_date(picks_for_date: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(picks_for_date).normalize() - pd.offsets.BDay(1)).normalize()


def feature_staleness_bdays(
    feature_date: pd.Timestamp,
    picks_for_date: pd.Timestamp,
) -> int:
    feature_dt = pd.Timestamp(feature_date).normalize()
    expected_dt = expected_feature_date(pd.Timestamp(picks_for_date))
    if feature_dt >= expected_dt:
        return 0
    return max(len(pd.bdate_range(feature_dt, expected_dt)) - 1, 0)


def _candidate_attr(candidate: Any, field: str) -> Any:
    if isinstance(candidate, dict):
        return candidate[field]
    return getattr(candidate, field)


def select_candidates(
    candidates: list[Any],
    *,
    count: int,
    allow_below_preferred: bool,
) -> list[Any]:
    ranked_preferred = sorted(
        [candidate for candidate in candidates if _candidate_attr(candidate, "preferred_filter_pass")],
        key=lambda candidate: _candidate_attr(candidate, "score"),
        reverse=True,
    )
    if not allow_below_preferred or len(ranked_preferred) >= count:
        return ranked_preferred[:count]

    ranked_fallback = sorted(
        [candidate for candidate in candidates if not _candidate_attr(candidate, "preferred_filter_pass")],
        key=lambda candidate: _candidate_attr(candidate, "score"),
        reverse=True,
    )
    return (ranked_preferred + ranked_fallback)[:count]


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


def default_readiness_paths(project_root: Path, mode: str = "premarket") -> dict[str, Path]:
    mode_dir = "premarket_backtest" if mode == "premarket" else "post_open_backtest"
    return {
        "locked_backtest": project_root
        / "results"
        / "backtests"
        / mode_dir
        / "summary_intraday_model_balanced_premarket_2025-01-01_2025-12-31.json"
        if mode == "premarket"
        else project_root
        / "results"
        / "backtests"
        / mode_dir
        / "summary_intraday_model_balanced_post-open_2025-01-01_2025-12-31.json",
        "forward_blind": project_root
        / "results"
        / "backtests"
        / mode_dir
        / "summary_intraday_model_balanced_premarket_2026-01-01_2026-03-31.json"
        if mode == "premarket"
        else project_root
        / "results"
        / "backtests"
        / mode_dir
        / "summary_intraday_model_balanced_post-open_2026-01-01_2026-03-31.json",
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
        "mode_backtested": False,
        "locked_positive": False,
        "forward_positive": False,
        "hit_rate_ok": False,
        "target_before_stop_ok": False,
        "live_data_ok": mode != "post-open",
        "runtime_ok": mode != "post-open",
        "logic_match_ok": False,
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
        "locked_mode": None,
        "forward_mode": None,
        "locked_runtime_seconds": None,
        "forward_runtime_seconds": None,
        "decision": None,
    }

    if not target_alignment:
        reasons.append("Displayed targets are not aligned with execution logic.")

    if not freshness_ok:
        reasons.append("Latest usable market data is stale for the requested trading date.")

    if locked_backtest_summary:
        locked_mode = locked_backtest_summary.get("mode")
        metrics["locked_mode"] = locked_mode
        checks["mode_backtested"] = locked_mode == mode
        checks["logic_match_ok"] = bool(locked_backtest_summary.get("exact_logic_match", False))
        metrics["locked_runtime_seconds"] = locked_backtest_summary.get("runtime_metrics", {}).get("total_runtime_seconds")
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
        if locked_mode != mode:
            reasons.append("Locked backtest was not run in the same mode being certified.")
        if not checks["logic_match_ok"]:
            reasons.append("Locked backtest is not marked as exact-logic parity with live recommendations.")
    else:
        reasons.append("Locked backtest summary is missing.")

    if forward_summary:
        metrics["forward_mode"] = forward_summary.get("mode")
        metrics["forward_runtime_seconds"] = forward_summary.get("runtime_metrics", {}).get("total_runtime_seconds")
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
        if forward_summary.get("mode") != mode:
            reasons.append("Forward blind test was not run in the same mode being certified.")
    else:
        reasons.append("Forward blind test summary is missing.")

    if mode == "post-open":
        live_ok = live_symbols > 0 and live_symbols >= max(5, int(processed_symbols * 0.20))
        checks["live_data_ok"] = live_ok
        runtime_candidates = [
            value
            for value in (metrics.get("locked_runtime_seconds"), metrics.get("forward_runtime_seconds"))
            if value is not None
        ]
        checks["runtime_ok"] = bool(runtime_candidates) and float(np.median(runtime_candidates)) < 120.0
        if not live_ok:
            reasons.append("Post-open mode does not have enough live symbols to trust the run.")
        if not checks["runtime_ok"]:
            reasons.append("Post-open runtime is still above the 2 minute personal-live threshold.")
    else:
        checks["runtime_ok"] = True

    if all(checks.values()):
        status = "READY"
        metrics["decision"] = "ready"
    elif (
        checks["target_alignment"]
        and checks["freshness_ok"]
        and checks["mode_backtested"]
        and checks["locked_positive"]
        and checks["logic_match_ok"]
    ):
        status = "SMALL_LIVE"
        metrics["decision"] = "small-live"
    elif checks["target_alignment"] and checks["freshness_ok"] and checks["locked_positive"]:
        status = "PAPER_ONLY"
        metrics["decision"] = "paper-only"
    else:
        status = "NOT_READY"
        metrics["decision"] = "not-ready"

    return ReadinessAssessment(
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        checks=checks,
        metrics=metrics,
    )


def strategy_config_to_dict(config: StrategyConfig) -> dict[str, Any]:
    return asdict(config)


def extract_sessions(minute_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split a multi-day minute DataFrame into per-session DataFrames.

    Each session runs from market open (09:15) to market close (15:30).
    """
    if minute_df.empty:
        return []
    dates = pd.Series(minute_df.index.normalize()).unique()
    sessions = []
    for date in dates:
        session = minute_df[minute_df.index.normalize() == date].copy()
        if len(session) > 0:
            sessions.append(session.sort_index())
    return sessions


def compute_daily_targets_from_minute(
    minute_df: pd.DataFrame,
    *,
    target_pct: float = 0.015,
    min_tradable_move_pct: float = 0.0075,
    cost_buffer_pct: float = 0.0018,
    ambiguity_band_pct: float = 0.0025,
) -> pd.DataFrame:
    """
    Compute H375 (full-session / daily) targets for each session in the data.

    Uses compute_horizon_targets per session with horizon_bars = session length.
    Returns a per-session summary as daily-level target labels.
    """
    sessions = extract_sessions(minute_df)
    daily_rows = []
    for session in sessions:
        if len(session) < 5:
            continue
        date = session.index[0].normalize()
        horizon_targets = compute_horizon_targets(
            session,
            horizon_bars=len(session),
            target_pct=target_pct,
            min_tradable_move_pct=min_tradable_move_pct,
            cost_buffer_pct=cost_buffer_pct,
            ambiguity_band_pct=ambiguity_band_pct,
        )
        row = {
            "date": date,
                "open": float(session["open"].iloc[0]),
                "high": float(session["high"].max()),
                "low": float(session["low"].min()),
                "close": float(session["close"].iloc[-1]),
                "volume": float(session["volume"].sum()) if "volume" in session.columns else 0.0,
                "max_up": float((session["high"].max() - session["open"].iloc[0]) / max(session["open"].iloc[0], 1e-9)),
                "max_down": float((session["open"].iloc[0] - session["low"].min()) / max(session["open"].iloc[0], 1e-9)),
                "trade_label": str(horizon_targets["trade_label"].iloc[0]),
                "trade_side_code": int(horizon_targets["trade_side_code"].iloc[0] or 0),
                "long_executable_move": float(horizon_targets["long_executable_move"].iloc[0]),
                "short_executable_move": float(horizon_targets["short_executable_move"].iloc[0]),
                "long_target": int(horizon_targets["long_target"].iloc[0] or 0),
                "short_target": int(horizon_targets["short_target"].iloc[0] or 0),
                "no_trade_target": int(horizon_targets["no_trade_target"].iloc[0] or 0),
            }
        daily_rows.append(row)

    if not daily_rows:
        return pd.DataFrame()
    result = pd.DataFrame(daily_rows)
    result = result.set_index("date")
    result.index = pd.to_datetime(result.index)
    return result
