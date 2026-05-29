"""OptiNet Router — Trade Card schema.

The Trade Card is the canonical OUTPUT of the router. It is a structured
decision packet — not a prediction — that a trader (or downstream execution
system) can act on directly.

Design principles:
1. Self-contained. Anyone reading a trade card knows exactly what to do.
2. Explicit no-trade. Most market states should produce NO_TRADE; that is a
   first-class output, not a fallback.
3. JSON round-trippable for logging, replay, and reconciliation.
4. Minimum required fields per spec — extra fields belong in feature_snapshot.

Schema sections (per the next-phase plan):
- MarketState   — regime, vol condition, time horizon
- Recommendation — instrument, strategy, expiry, strike/zone, size, entry/exit
- Risk          — expected edge, tail-risk, do-not-trade flag
- Meta          — timestamp, model_version, snapshot of features used
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    TREND = "trend"
    RANGE = "range"
    EXPANSION = "expansion"     # volatility expanding
    COMPRESSION = "compression"  # volatility compressing
    EVENT_RISK = "event_risk"   # known event window (results, RBI, etc.)
    UNKNOWN = "unknown"


class VolCondition(str, Enum):
    COMPRESSED = "compressed"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


class Instrument(str, Enum):
    NIFTY_FUT = "NIFTY_FUT"
    BANKNIFTY_FUT = "BANKNIFTY_FUT"
    NIFTY_OPT = "NIFTY_OPT"
    BANKNIFTY_OPT = "BANKNIFTY_OPT"


class StrategyFamily(str, Enum):
    """Top-level family. The actual recommendation is one of these or NO_TRADE."""
    FUTURES_LONG = "FUT_LONG"
    FUTURES_SHORT = "FUT_SHORT"
    CALL_DEBIT_SPREAD = "CALL_DEBIT_SPREAD"
    PUT_DEBIT_SPREAD = "PUT_DEBIT_SPREAD"
    LONG_STRADDLE = "LONG_STRADDLE"
    LONG_STRANGLE = "LONG_STRANGLE"
    IRON_CONDOR = "IRON_CONDOR"
    NO_TRADE = "NO_TRADE"


class LegSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


# ---------------------------------------------------------------------------
# Dataclasses — the schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeLeg:
    """One leg of a trade. Futures legs have strike=None and opt_type=None."""
    side: LegSide
    instrument_kind: str          # "futures" or "option"
    expiry: date
    strike: Optional[int]         # None for futures
    opt_type: Optional[OptionType]  # None for futures
    lots: int                     # ≥ 1


@dataclass(frozen=True)
class MarketState:
    regime: Regime
    regime_confidence: float          # [0, 1]
    vol_condition: VolCondition
    time_horizon_minutes: int         # how long the regime is expected to hold
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (0.0 <= self.regime_confidence <= 1.0):
            raise ValueError(f"regime_confidence out of [0,1]: {self.regime_confidence}")
        if self.time_horizon_minutes < 0:
            raise ValueError("time_horizon_minutes must be >= 0")


@dataclass(frozen=True)
class Recommendation:
    """The actual trade. Use NO_TRADE strategy when no edge."""
    instrument: Instrument
    strategy: StrategyFamily
    legs: tuple[TradeLeg, ...]
    entry_zone_low: float            # spot/futures price low end of entry zone
    entry_zone_high: float           # high end
    suggested_size_lots: int         # total lots across all legs (informational)
    entry_window_start: datetime     # earliest valid entry
    entry_window_end: datetime       # latest valid entry (e.g. cutoff time)
    stop_loss_inr: float             # negative number, e.g. -3000
    target_inr: float                # positive number, e.g. +6000
    time_stop: datetime              # force-exit time (e.g. 15:25 IST)
    reason_codes: tuple[str, ...]    # short codes explaining the recommendation

    def __post_init__(self) -> None:
        if self.strategy == StrategyFamily.NO_TRADE:
            return  # NO_TRADE bypasses the rest
        if self.entry_zone_high < self.entry_zone_low:
            raise ValueError("entry_zone_high < entry_zone_low")
        if self.entry_window_end < self.entry_window_start:
            raise ValueError("entry_window_end < entry_window_start")
        if self.stop_loss_inr >= 0:
            raise ValueError("stop_loss_inr must be negative")
        if self.target_inr <= 0:
            raise ValueError("target_inr must be positive")
        if self.suggested_size_lots < 1:
            raise ValueError("suggested_size_lots must be >= 1")
        if not self.legs:
            raise ValueError("non-NO_TRADE recommendation must have legs")


@dataclass(frozen=True)
class Risk:
    expected_edge_inr: float          # signed; negative is allowed but should rarely ship
    expected_edge_confidence: float   # [0, 1]
    tail_risk_warning: Optional[str]  # human-readable, or None
    do_not_trade: bool
    do_not_trade_reason: Optional[str]  # required if do_not_trade=True

    def __post_init__(self) -> None:
        if not (0.0 <= self.expected_edge_confidence <= 1.0):
            raise ValueError("expected_edge_confidence out of [0,1]")
        if self.do_not_trade and not self.do_not_trade_reason:
            raise ValueError("do_not_trade=True requires a reason")


@dataclass(frozen=True)
class TradeCard:
    """The canonical router output. Always emitted, even on NO_TRADE."""
    timestamp: datetime               # decision time
    symbol: str                       # NIFTY / BANKNIFTY (the underlying)
    market_state: MarketState
    recommendation: Recommendation
    risk: Risk
    feature_snapshot: dict[str, Any]  # raw features used (for replay)
    model_version: str                # e.g. "router-v0.1"

    # ----- consistency invariants -----------------------------------------
    def __post_init__(self) -> None:
        # If do_not_trade is True, the recommendation MUST be NO_TRADE
        if self.risk.do_not_trade and self.recommendation.strategy != StrategyFamily.NO_TRADE:
            raise ValueError(
                f"Risk says do_not_trade=True but recommendation is "
                f"{self.recommendation.strategy.value}; must be NO_TRADE."
            )
        # If recommendation is NO_TRADE, risk should reflect that
        # (we don't strictly require do_not_trade=True, since NO_TRADE may also
        # be issued for "weak edge" rather than "explicit halt").

    # ----- serialization --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert enums and datetimes to JSON-friendly forms
        return _normalize_for_json(d)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeCard":
        return _denormalize_from_dict(d)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _normalize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(x) for x in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _denormalize_from_dict(d: dict[str, Any]) -> TradeCard:
    ms = d["market_state"]
    market_state = MarketState(
        regime=Regime(ms["regime"]),
        regime_confidence=float(ms["regime_confidence"]),
        vol_condition=VolCondition(ms["vol_condition"]),
        time_horizon_minutes=int(ms["time_horizon_minutes"]),
        notes=tuple(ms.get("notes", [])),
    )
    rec = d["recommendation"]
    legs = tuple(
        TradeLeg(
            side=LegSide(L["side"]),
            instrument_kind=L["instrument_kind"],
            expiry=_parse_date(L["expiry"]),
            strike=L.get("strike"),
            opt_type=OptionType(L["opt_type"]) if L.get("opt_type") else None,
            lots=int(L["lots"]),
        )
        for L in rec["legs"]
    )
    recommendation = Recommendation(
        instrument=Instrument(rec["instrument"]),
        strategy=StrategyFamily(rec["strategy"]),
        legs=legs,
        entry_zone_low=float(rec["entry_zone_low"]),
        entry_zone_high=float(rec["entry_zone_high"]),
        suggested_size_lots=int(rec["suggested_size_lots"]),
        entry_window_start=_parse_dt(rec["entry_window_start"]),
        entry_window_end=_parse_dt(rec["entry_window_end"]),
        stop_loss_inr=float(rec["stop_loss_inr"]),
        target_inr=float(rec["target_inr"]),
        time_stop=_parse_dt(rec["time_stop"]),
        reason_codes=tuple(rec["reason_codes"]),
    )
    rk = d["risk"]
    risk = Risk(
        expected_edge_inr=float(rk["expected_edge_inr"]),
        expected_edge_confidence=float(rk["expected_edge_confidence"]),
        tail_risk_warning=rk.get("tail_risk_warning"),
        do_not_trade=bool(rk["do_not_trade"]),
        do_not_trade_reason=rk.get("do_not_trade_reason"),
    )
    return TradeCard(
        timestamp=_parse_dt(d["timestamp"]),
        symbol=d["symbol"],
        market_state=market_state,
        recommendation=recommendation,
        risk=risk,
        feature_snapshot=dict(d.get("feature_snapshot", {})),
        model_version=d["model_version"],
    )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_no_trade_card(
    *, timestamp: datetime, symbol: str, market_state: MarketState,
    reason: str, model_version: str,
    feature_snapshot: Optional[dict[str, Any]] = None,
) -> TradeCard:
    """Construct a NO_TRADE card. Used when the router refuses to trade."""
    # Synthesize a placeholder Recommendation that satisfies the schema
    placeholder_window_start = timestamp
    placeholder_window_end = timestamp
    rec = Recommendation(
        instrument=Instrument.NIFTY_FUT,  # any value; ignored for NO_TRADE
        strategy=StrategyFamily.NO_TRADE,
        legs=tuple(),
        entry_zone_low=0.0,
        entry_zone_high=0.0,
        suggested_size_lots=1,             # placeholder; ignored for NO_TRADE
        entry_window_start=placeholder_window_start,
        entry_window_end=placeholder_window_end,
        stop_loss_inr=-1.0,                # placeholder; ignored for NO_TRADE
        target_inr=1.0,                    # placeholder; ignored
        time_stop=timestamp,
        reason_codes=(reason,),
    )
    risk = Risk(
        expected_edge_inr=0.0,
        expected_edge_confidence=0.0,
        tail_risk_warning=None,
        do_not_trade=True,
        do_not_trade_reason=reason,
    )
    return TradeCard(
        timestamp=timestamp,
        symbol=symbol,
        market_state=market_state,
        recommendation=rec,
        risk=risk,
        feature_snapshot=feature_snapshot or {},
        model_version=model_version,
    )
