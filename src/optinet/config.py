from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfileSpec:
    name: str
    target_pct: float
    stop_pct: float
    cost_buffer_pct: float
    min_delta: float
    max_delta: float
    prefer_monthly: bool = False
    allow_expiry_day: bool = True


@dataclass(frozen=True)
class IndexSpec:
    name: str
    aliases: tuple[str, ...]
    lot_size: int
    strike_step: int


PROFILE_SPECS: dict[str, RiskProfileSpec] = {
    "conservative": RiskProfileSpec(
        name="conservative",
        target_pct=0.005,
        stop_pct=0.003,
        cost_buffer_pct=0.001,
        min_delta=0.55,
        max_delta=0.65,
        prefer_monthly=True,
        allow_expiry_day=False,
    ),
    "balanced": RiskProfileSpec(
        name="balanced",
        target_pct=0.010,
        stop_pct=0.006,
        cost_buffer_pct=0.001,
        min_delta=0.40,
        max_delta=0.55,
    ),
    "aggressive": RiskProfileSpec(
        name="aggressive",
        target_pct=0.015,
        stop_pct=0.010,
        cost_buffer_pct=0.001,
        min_delta=0.25,
        max_delta=0.40,
    ),
}

INDEX_SPECS: dict[str, IndexSpec] = {
    "NIFTY": IndexSpec("NIFTY", ("NIFTY", "NIFTY 50", "NIFTY50", "NIFTY-I"), 50, 50),
    "BANKNIFTY": IndexSpec("BANKNIFTY", ("BANKNIFTY", "BANK NIFTY", "NIFTY BANK", "BANKNIFTY-I"), 15, 100),
}


def canonical_index(value: str) -> str:
    normalized = str(value).upper().replace("_", " ").replace("-", " ").strip()
    squashed = normalized.replace(" ", "")
    for key, spec in INDEX_SPECS.items():
        if normalized == key or squashed == key:
            return key
        if any(normalized == alias.upper() or squashed == alias.upper().replace(" ", "") for alias in spec.aliases):
            return key
    return squashed
