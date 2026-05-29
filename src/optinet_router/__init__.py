"""OptiNet Router — market-state-to-action engine.

Architecture (frozen 2026-05-27):

    features → regime detector → family engines → router → TradeCard

Each family engine (futures / long_vol / debit_spread) implements the same
FamilyEngine interface. The router calls each engine with the current market
state and chooses among their proposals (or returns NO_TRADE if all decline).

Output is always a TradeCard — including for NO_TRADE — for full audit
trail and downstream replay.

Status:
- schema.py     — frozen Trade-Card schema (ready)
- families/     — three stubs, no implementations yet
- regime.py     — placeholder for regime detector
- router.py     — placeholder for the router itself

See docs/optinet_router_plan.md for sprint order and effort estimates.
"""
from optinet_router.schema import (
    Instrument, LegSide, MarketState, OptionType, Recommendation,
    Regime, Risk, StrategyFamily, TradeCard, TradeLeg, VolCondition,
    make_no_trade_card,
)

__all__ = [
    "Instrument", "LegSide", "MarketState", "OptionType", "Recommendation",
    "Regime", "Risk", "StrategyFamily", "TradeCard", "TradeLeg", "VolCondition",
    "make_no_trade_card",
]
