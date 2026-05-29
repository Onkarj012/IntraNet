"""Abstract base for strategy-family engines.

Each family engine (futures / long-vol / debit-spread / ...) must implement
the same `score_minute` interface so the router can call them uniformly.

The engine returns either:
- a Recommendation (if the family wants to trade this minute), or
- None (if the family declines).

The router then picks among engine outputs (or NO_TRADE if all decline).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from optinet_router.schema import MarketState, Recommendation


class FamilyEngine(ABC):
    """Common interface for all strategy-family engines."""

    name: str  # set by subclass — e.g. "futures_v1", "long_vol_v1"

    @abstractmethod
    def score_minute(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        market_state: MarketState,
        features: dict[str, Any],
    ) -> Optional[Recommendation]:
        """Return a Recommendation if this family wants to trade, else None.

        Implementations MUST:
        - return None if any required feature is missing
        - return None if confidence is below internal threshold
        - return a fully-populated Recommendation otherwise (no NO_TRADE here;
          NO_TRADE is the router's job when all engines decline)
        """

    @abstractmethod
    def expected_edge_inr(self, recommendation: Recommendation) -> float:
        """Return the engine's expected edge for the given recommendation."""

    def health(self) -> bool:
        """Return True if the engine is loaded and ready to score."""
        return True
