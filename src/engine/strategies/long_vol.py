"""LongVolEngine — long-volatility option strategies on NIFTY.

Sprint #2 in the OptiNet Router roadmap. NOT YET IMPLEMENTED.

This is the "mirror" sprint suggested by the V5 postmortem. The hypothesis:
the V5 gate's apparent edge under BS-reprice came partly from systematically
underestimating long-premium wins. Test the same gate concept (or a fresh
gate) for entering LONG_STRADDLE / LONG_STRANGLE structures.

Cheapest first sprint because:
- the v5_simulator label dataset already contains long-vol PnL per minute
- the realistic market-price simulator already exists
- can reuse archived V5 gate models read-only as a quick directional test

Target predictions:
- realized vol expansion within next 30-60 minutes
- post-compression breakout
- conditions where long premium is cheap (low IV-RV spread)

Output: LONG_STRADDLE or LONG_STRANGLE recommendation.

Effort estimate: 1-2 days for the directional answer; +2-3 days if the
directional answer is positive and we want to retrain a long-vol-specific gate.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from optinet_router.families.base import FamilyEngine
from optinet_router.schema import MarketState, Recommendation


class LongVolEngine(FamilyEngine):
    name = "long_vol_v0_stub"

    def __init__(self, *, symbol: str = "NIFTY"):
        self.symbol = symbol

    def score_minute(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        market_state: MarketState,
        features: dict[str, Any],
    ) -> Optional[Recommendation]:
        raise NotImplementedError(
            "LongVolEngine not implemented. Sprint #2 in the router plan."
        )

    def expected_edge_inr(self, recommendation: Recommendation) -> float:
        raise NotImplementedError("LongVolEngine.expected_edge_inr not implemented")

    def health(self) -> bool:
        return False
