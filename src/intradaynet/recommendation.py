"""
Profile-based recommendation engine for the live LightGBM backend.

Probability calibration is integrated: if a calibrator is available in the
model bundle, calibrated probabilities are used for confidence scoring.
Uncalibrated raw probabilities are demoted to a diagnostic field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RiskProfile:
    name: str
    max_total: int
    max_per_side: int
    min_confidence: float
    min_expected_net_edge: float
    min_liquidity_score: float
    reward_cost_floor: float
    regime_floor: float


RISK_PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        name="conservative",
        max_total=3,
        max_per_side=3,
        min_confidence=0.68,
        min_expected_net_edge=0.0020,
        min_liquidity_score=0.70,
        reward_cost_floor=2.0,
        regime_floor=0.65,
    ),
    "balanced": RiskProfile(
        name="balanced",
        max_total=5,
        max_per_side=5,
        min_confidence=0.65,
        min_expected_net_edge=0.0012,
        min_liquidity_score=0.50,
        reward_cost_floor=1.5,
        regime_floor=0.50,
    ),
    "aggressive": RiskProfile(
        name="aggressive",
        max_total=8,
        max_per_side=8,
        min_confidence=0.60,
        min_expected_net_edge=0.0008,
        min_liquidity_score=0.25,
        reward_cost_floor=1.05,
        regime_floor=0.25,
    ),
}


def probability_strength(probability: float) -> float:
    return abs(probability - 0.5) * 2.0


def liquidity_score(
    avg_daily_traded_value: float,
    median_minute_turnover: float,
) -> float:
    adv_component = min(avg_daily_traded_value / 25_000_000.0, 1.0)
    minute_component = min(median_minute_turnover / 300_000.0, 1.0)
    return float(np.clip((adv_component * 0.65) + (minute_component * 0.35), 0.0, 1.0))


def regime_alignment_score(regime: str, side: str) -> float:
    side = side.upper()
    if regime == "extreme":
        return 0.0
    if regime in {"calm_bull", "volatile_bull"}:
        return 1.0 if side == "LONG" else 0.55
    if regime in {"calm_bear", "volatile_bear"}:
        return 1.0 if side == "SHORT" else 0.55
    return 0.7


def build_candidate(
    *,
    symbol: str,
    side: str,
    horizon: str,
    entry_reference: float,
    expected_gross_return: float,
    expected_net_edge: float,
    confidence: float,
    probability: float,
    avg_daily_traded_value: float,
    median_minute_turnover: float,
    regime: str,
    sector: str = "UNKNOWN",
    driver_flags: list[str] | None = None,
    cost_fraction: float = 0.0,
    stop_loss_pct: float = 0.008,
) -> dict[str, Any]:
    liq_score = liquidity_score(avg_daily_traded_value, median_minute_turnover)
    reg_score = regime_alignment_score(regime, side)
    prob_strength = probability_strength(probability)
    reward_cost = abs(expected_gross_return) / max(cost_fraction, 1e-6) if cost_fraction > 0 else 99.0

    open_gap_penalty = 0.15 if expected_net_edge < 0.001 else 0.0
    score = (
        expected_net_edge * 1000.0
        + prob_strength * 0.9
        + liq_score * 0.8
        + reg_score * 0.6
        - open_gap_penalty
    )

    if side == "LONG":
        target = entry_reference * (1 + max(expected_gross_return, 0.0))
        stop_loss = entry_reference * (1 - stop_loss_pct)
    else:
        target = entry_reference * (1 - max(abs(expected_gross_return), 0.0))
        stop_loss = entry_reference * (1 + stop_loss_pct)

    return {
        "symbol": symbol,
        "side": side,
        "horizon": horizon,
        "entry_reference": entry_reference,
        "expected_gross_return": expected_gross_return,
        "expected_net_edge": expected_net_edge,
        "confidence": confidence,
        "probability": probability,
        "prob_strength": prob_strength,
        "liquidity_score": liq_score,
        "avg_daily_traded_value": avg_daily_traded_value,
        "median_minute_turnover": median_minute_turnover,
        "regime": regime,
        "regime_alignment": reg_score,
        "reward_cost_ratio": reward_cost,
        "sector": sector,
        "target": target,
        "stop_loss": stop_loss,
        "driver_flags": driver_flags or [],
        "score": score,
    }


def filter_for_profile(
    candidates: list[dict[str, Any]],
    profile: RiskProfile,
) -> dict[str, list[dict[str, Any]]]:
    output = {"picks": [], "long": [], "short": []}
    grouped = {"LONG": [], "SHORT": []}

    for candidate in candidates:
        if candidate["confidence"] < profile.min_confidence:
            continue
        if candidate["expected_net_edge"] < profile.min_expected_net_edge:
            continue
        if candidate["liquidity_score"] < profile.min_liquidity_score:
            continue
        if candidate["reward_cost_ratio"] < profile.reward_cost_floor:
            continue
        if candidate["regime_alignment"] < profile.regime_floor:
            continue
        grouped[candidate["side"]].append(candidate)

    selected: list[dict[str, Any]] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    ranked_all = sorted(
        grouped["LONG"] + grouped["SHORT"],
        key=lambda item: item["score"],
        reverse=True,
    )

    for item in ranked_all:
        side = item["side"]
        if side_counts[side] >= profile.max_per_side:
            continue
        selected.append(item)
        side_counts[side] += 1
        if len(selected) >= profile.max_total:
            break

    for rank, item in enumerate(selected, start=1):
        enriched = dict(item)
        enriched["profile"] = profile.name
        enriched["rank"] = rank
        output["picks"].append(enriched)
        if item["side"] == "LONG":
            output["long"].append(enriched)
        else:
            output["short"].append(enriched)

    return output


def build_recommendation_payload(
    *,
    trade_date: str,
    market_regime: str,
    market_summary: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "trade_date": trade_date,
        "generation_timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "market_regime": market_regime,
        "market_summary": market_summary,
        "profiles": {},
    }

    for profile_name, profile in RISK_PROFILES.items():
        payload["profiles"][profile_name] = filter_for_profile(candidates, profile)

    return payload


def serialize_profile_config() -> dict[str, Any]:
    return {name: asdict(profile) for name, profile in RISK_PROFILES.items()}


def calibrate_confidence(
    raw_confidence: float,
    raw_probability: float,
    *,
    calibrator=None,
    probability_strength_score: float | None = None,
) -> dict[str, float]:
    """
    Apply calibration to raw model outputs and return calibrated confidence.

    If a calibrator is provided, raw_probability is run through it.
    Otherwise, a heuristic blend using margin_adjusted_confidence is applied
    as fallback.

    Returns a dict with:
        'calibrated_confidence': The calibrated (or heuristic) confidence score.
        'raw_confidence': Original confidence from model.
        'raw_probability': Original raw probability.
        'calibration_applied': True if calibrator was used.
    """
    from intradaynet.v7 import margin_adjusted_confidence

    if calibrator is not None:
        try:
            from intradaynet.calibrator import apply_calibration

            cal_probs = apply_calibration(
                calibrator,
                np.array([[raw_probability, 1.0 - raw_probability]]),
            )
            cal_conf = float(cal_probs[0, 0])
            return {
                "calibrated_confidence": max(0.0, min(1.0, cal_conf)),
                "raw_confidence": raw_confidence,
                "raw_probability": raw_probability,
                "calibration_applied": True,
            }
        except Exception:
            pass

    if probability_strength_score is None:
        probability_strength_score = probability_strength(raw_probability)
    heuristic = margin_adjusted_confidence(raw_probability, 1.0 - raw_probability)
    return {
        "calibrated_confidence": max(0.0, min(1.0, heuristic)),
        "raw_confidence": raw_confidence,
        "raw_probability": raw_probability,
        "calibration_applied": False,
    }
