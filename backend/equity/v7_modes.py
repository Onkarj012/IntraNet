from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


class RuntimeTracker:
    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.metrics: dict[str, float] = {
            "data_refresh_seconds": 0.0,
            "feature_build_seconds": 0.0,
            "ranking_seconds": 0.0,
        }

    def start(self, key: str) -> None:
        self._starts[key] = perf_counter()

    def stop(self, key: str) -> None:
        started = self._starts.pop(key, None)
        if started is None:
            return
        self.metrics[key] = self.metrics.get(key, 0.0) + max(perf_counter() - started, 0.0)

    def snapshot(self, total_runtime_seconds: float) -> dict[str, float]:
        return {
            "total_runtime_seconds": round(total_runtime_seconds, 4),
            "data_refresh_seconds": round(self.metrics.get("data_refresh_seconds", 0.0), 4),
            "feature_build_seconds": round(self.metrics.get("feature_build_seconds", 0.0), 4),
            "ranking_seconds": round(self.metrics.get("ranking_seconds", 0.0), 4),
        }


def extract_post_open_session(
    minute_df: pd.DataFrame,
    target_date: pd.Timestamp,
    cutoff_time: str,
) -> pd.DataFrame:
    session = minute_df[minute_df.index.normalize() == target_date.normalize()].copy()
    if session.empty:
        return session
    session = session.between_time("09:15", cutoff_time)
    return session.sort_index()


def session_cutoff_timestamp(session_df: pd.DataFrame) -> pd.Timestamp | None:
    if session_df.empty:
        return None
    return pd.Timestamp(session_df.index.max())


def target_price_date(session_df: pd.DataFrame) -> pd.Timestamp:
    return session_df.index[0].normalize()


def compute_post_open_adjustment(
    *,
    direction: str,
    prev_close: float,
    base_probability: float,
    predicted_magnitude: float,
    session_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    feature_row: pd.Series,
) -> dict[str, Any]:
    if session_df.empty or prev_close <= 0:
        return {
            "aligned": False,
            "alignment_score": -1.0,
            "adjusted_probability": base_probability,
            "adjusted_magnitude": predicted_magnitude,
            "reference_price": prev_close,
            "cutoff_close": None,
            "session_open": None,
            "live_price": None,
            "gap_pct": None,
            "move_from_open_pct": None,
            "opening_range_pct": None,
            "early_relative_volume": None,
            "vwap_displacement_pct": None,
            "cutoff_timestamp": None,
        }

    session_open = float(session_df["open"].iloc[0])
    live_price = float(session_df["close"].iloc[-1])
    session_high = float(session_df["high"].max())
    session_low = float(session_df["low"].min())
    gap_pct = (session_open - prev_close) / prev_close
    move_from_open_pct = (live_price - session_open) / max(session_open, 1e-9)
    opening_range_pct = (session_high - session_low) / max(session_open, 1e-9)
    range_width = max(session_high - session_low, 1e-9)
    session_volume = float(session_df["volume"].sum()) if "volume" in session_df.columns else 0.0
    session_vwap = float(
        ((session_df["close"] * session_df["volume"]).sum() / max(session_df["volume"].sum(), 1e-9))
        if "volume" in session_df.columns
        else session_df["close"].mean()
    )
    vwap_displacement_pct = (live_price - session_vwap) / max(session_vwap, 1e-9)
    cutoff_timestamp = session_cutoff_timestamp(session_df)
    recent_days = sorted(d for d in minute_df.index.normalize().unique() if d < target_price_date(session_df))
    historical_opening_volumes: list[float] = []
    for prior_day in recent_days[-10:]:
        prior_session = minute_df[minute_df.index.normalize() == prior_day].between_time(
            "09:15",
            session_df.index.max().strftime("%H:%M"),
        )
        if not prior_session.empty and "volume" in prior_session.columns:
            historical_opening_volumes.append(float(prior_session["volume"].sum()))
    avg_opening_volume = float(np.mean(historical_opening_volumes)) if historical_opening_volumes else 0.0
    early_relative_volume = session_volume / max(avg_opening_volume, 1e-9) if avg_opening_volume > 0 else 1.0
    market_confirmation = float(
        np.clip(
            0.4 * feature_row.get("market_breadth", 0.0)
            + 0.3 * feature_row.get("risk_on_signal", 0.0)
            + 0.3 * feature_row.get("sector_relative_strength", 0.0),
            -1.0,
            1.0,
        )
    )

    scale = max(predicted_magnitude, 0.005)
    if direction == "LONG":
        gap_component = np.clip(gap_pct / scale, -1.0, 1.0)
        move_component = np.clip(move_from_open_pct / scale, -1.0, 1.0)
        location_component = np.clip(((live_price - session_low) / range_width) * 2.0 - 1.0, -1.0, 1.0)
        vwap_component = np.clip(vwap_displacement_pct / scale, -1.0, 1.0)
        confirmation_component = market_confirmation
    else:
        gap_component = np.clip((-gap_pct) / scale, -1.0, 1.0)
        move_component = np.clip((-move_from_open_pct) / scale, -1.0, 1.0)
        location_component = np.clip(((session_high - live_price) / range_width) * 2.0 - 1.0, -1.0, 1.0)
        vwap_component = np.clip((-vwap_displacement_pct) / scale, -1.0, 1.0)
        confirmation_component = -market_confirmation

    relative_volume_component = np.clip((early_relative_volume - 1.0) / 1.5, -1.0, 1.0)
    alignment_score = float(
        (0.22 * gap_component)
        + (0.28 * move_component)
        + (0.16 * location_component)
        + (0.14 * relative_volume_component)
        + (0.10 * vwap_component)
        + (0.10 * confirmation_component)
    )
    adjusted_probability = float(np.clip(base_probability + (0.10 * alignment_score), 0.0, 0.999))
    adjusted_magnitude = float(max(predicted_magnitude * (1.0 + (0.20 * alignment_score)), 0.0))

    return {
        "aligned": True,
        "alignment_score": alignment_score,
        "adjusted_probability": adjusted_probability,
        "adjusted_magnitude": adjusted_magnitude,
        "reference_price": live_price,
        "cutoff_close": live_price,
        "session_open": session_open,
        "live_price": live_price,
        "gap_pct": gap_pct,
        "move_from_open_pct": move_from_open_pct,
        "opening_range_pct": opening_range_pct,
        "early_relative_volume": early_relative_volume,
        "vwap_displacement_pct": vwap_displacement_pct,
        "cutoff_timestamp": cutoff_timestamp.isoformat() if cutoff_timestamp is not None else None,
    }


def compute_preferred_filter_pass(
    *,
    confidence: float,
    predicted_magnitude: float,
    min_confidence: float,
    min_predicted_magnitude: float,
    alignment_ok: bool,
    regime_ok: bool,
) -> bool:
    return bool(
        confidence >= min_confidence
        and predicted_magnitude >= min_predicted_magnitude
        and alignment_ok
        and regime_ok
    )


def classify_regime(feature_row: pd.Series) -> dict[str, str]:
    price_momentum = float(feature_row.get("price_momentum_5d", 0.0))
    volatility = float(feature_row.get("prev_day_volatility", 0.0))
    direction = "trending" if abs(price_momentum) >= 0.02 else "choppy"
    volatility_bucket = "higher_volatility" if volatility >= 0.02 else "calmer"
    return {
        "trend_regime": direction,
        "volatility_regime": volatility_bucket,
    }

