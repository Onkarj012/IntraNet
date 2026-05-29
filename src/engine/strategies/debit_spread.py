"""DebitSpreadEngine — defined-risk directional option spreads on NIFTY.

Sprint #3 in the OptiNet Router roadmap. NOT YET IMPLEMENTED.

Target structures:
- CALL_DEBIT_SPREAD (bullish, defined risk)
- PUT_DEBIT_SPREAD (bearish, defined risk)
- IRON_CONDOR (range-bound, defined risk) — possibly later

Why defined-risk:
- limited downside (no naked premium tail)
- easier to risk-manage than naked options
- may survive realistic execution better than the V5 short-vol path
- useful for trend-with-reduced-tail-risk regimes

Effort estimate: 3-4 days. Requires generating defined-risk labels, training
a directional + spread-specific gate, and validating under the realistic
market-price simulator.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from optinet_router.families.base import FamilyEngine
from optinet_router.schema import MarketState, Recommendation


class DebitSpreadEngine(FamilyEngine):
    name = "debit_spread_v0_stub"

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
            "DebitSpreadEngine not implemented. Sprint #3 in the router plan."
        )

    def expected_edge_inr(self, recommendation: Recommendation) -> float:
        raise NotImplementedError("DebitSpreadEngine.expected_edge_inr not implemented")

    def health(self) -> bool:
        return False
