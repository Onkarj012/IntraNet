"""
Portfolio Construction for V8.

Implements diversified stock selection with:
1. Sector penalty — penalize picks in the same sector
2. Correlation penalty — penalize picks correlated to existing selections
3. Direction balance — enforce LONG/SHORT mix per regime
4. Equal risk position sizing

The core algorithm is a greedy sequential selector:
  - Start with highest-score stock
  - For next pick, apply sector and correlation penalties
  - Continue until we have N picks or no stocks pass thresholds

This replaces the simple "top-K by score" approach in V7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import PortfolioConfig


# ---------------------------------------------------------------------------
# Candidate data
# ---------------------------------------------------------------------------

@dataclass
class PortfolioCandidate:
    """A single stock candidate for portfolio inclusion."""
    symbol: str
    score: float
    probability: float
    expected_edge: float
    side: str  # LONG or SHORT
    industry: str
    tier: str
    regime_id: int
    correlation_score: float = 0.0
    selected: bool = False
    penalized_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 6),
            "probability": round(self.probability, 4),
            "expected_edge": round(self.expected_edge, 6),
            "side": self.side,
            "industry": self.industry,
            "tier": self.tier,
            "regime_id": self.regime_id,
            "selected": self.selected,
            "penalized_score": round(self.penalized_score, 6),
        }


@dataclass
class Portfolio:
    """A constructed portfolio of picks."""
    candidates: list[PortfolioCandidate]
    selected: list[PortfolioCandidate]
    date: pd.Timestamp
    regime_id: int
    regime_label: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def n_long(self) -> int:
        return sum(1 for c in self.selected if c.side == "LONG")

    @property
    def n_short(self) -> int:
        return sum(1 for c in self.selected if c.side == "SHORT")

    @property
    def avg_score(self) -> float:
        if not self.selected:
            return 0.0
        return float(np.mean([c.score for c in self.selected]))

    def summary(self) -> str:
        if not self.selected:
            return f"[{self.date.date()}] No picks (regime: {self.regime_label})"

        lines = [f"[{self.date.date()}] Portfolio ({self.regime_label}):"]
        for i, c in enumerate(self.selected):
            lines.append(
                f"  {i + 1}. {c.symbol:12s} {c.side:5s} "
                f"prob={c.probability:.3f} edge={c.expected_edge:.4f} "
                f"industry={c.industry}"
            )
        lines.append(f"  LONG: {self.n_long} | SHORT: {self.n_short} | Avg score: {self.avg_score:.4f}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.selected:
            return pd.DataFrame()
        return pd.DataFrame([c.to_dict() for c in self.selected])


# ---------------------------------------------------------------------------
# Portfolio Constructor
# ---------------------------------------------------------------------------

class PortfolioConstructor:
    """
    Greedy diversified portfolio construction.

    Algorithm:
    1. Score all stocks: score = probability × expected_edge
    2. Filter by minimum thresholds
    3. Select best stock
    4. For next pick:
       a. Compute sector penalty for stocks in same industry
       b. Compute correlation penalty for stocks correlated to selections
       c. Apply penalties: penalized_score = score × (1 - sector_penalty) × (1 - corr_penalty)
       d. Pick highest penalized_score
    5. Enforce direction balance
    6. Size positions with equal risk allocation
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def build(
        self,
        candidates_df: pd.DataFrame,
        date: pd.Timestamp,
        regime_id: int,
        regime_label: str,
        *,
        returns_matrix: pd.DataFrame | None = None,
        max_long: int = 4,
        max_short: int = 2,
    ) -> Portfolio:
        """
        Build a portfolio from candidate scores.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Must have columns: symbol, score, probability, expected_edge,
            side, industry, tier.
        date : pd.Timestamp
            Portfolio date.
        regime_id : int
            Current market regime.
        regime_label : str
            Regime name.
        returns_matrix : pd.DataFrame, optional
            Historical returns for correlation computation.
            Rows = dates, columns = symbols.
        max_long : int
            Maximum LONG picks.
        max_short : int
            Maximum SHORT picks.

        Returns
        -------
        Portfolio
            Constructed portfolio with selected picks.
        """
        candidates = self._prepare_candidates(candidates_df, regime_id)
        selected, all_candidates = self._greedy_select(
            candidates, returns_matrix, max_long, max_short,
        )

        return Portfolio(
            candidates=all_candidates,
            selected=selected,
            date=date,
            regime_id=regime_id,
            regime_label=regime_label,
            metrics=self._compute_portfolio_metrics(selected),
        )

    def _prepare_candidates(
        self,
        df: pd.DataFrame,
        regime_id: int,
    ) -> list[PortfolioCandidate]:
        """Convert DataFrame to candidate list with filtering."""
        cfg = self.config

        candidates = []
        for _, row in df.iterrows():
            score = float(row.get("score", 0))
            probability = float(row.get("probability", 0))
            expected_edge = float(row.get("expected_edge", 0))

            # Apply tier-specific confidence adjustment
            tier = str(row.get("tier", "tier_1"))
            tier_mult = 1.0
            if tier == "tier_2":
                tier_mult = 1.1
            elif tier == "tier_3":
                tier_mult = 1.25

            adjusted_prob = probability

            if adjusted_prob < cfg.min_confidence * tier_mult:
                continue
            if expected_edge < cfg.min_expected_value:
                continue

            candidates.append(PortfolioCandidate(
                symbol=str(row.get("symbol", "")),
                score=score,
                probability=probability,
                expected_edge=expected_edge,
                side=str(row.get("side", "LONG")),
                industry=str(row.get("industry", "")),
                tier=tier,
                regime_id=regime_id,
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _greedy_select(
        self,
        candidates: list[PortfolioCandidate],
        returns_matrix: pd.DataFrame | None,
        max_long: int,
        max_short: int,
    ) -> tuple[list[PortfolioCandidate], list[PortfolioCandidate]]:
        """
        Greedy selection with sector and correlation penalties.

        Returns (selected, all_candidates_with_penalized_scores).
        """
        selected: list[PortfolioCandidate] = []
        selected_symbols: set[str] = set()
        selected_industries: dict[str, int] = {}
        n_long = 0
        n_short = 0

        for candidate in candidates:
            # Direction balance check
            if candidate.side == "LONG" and n_long >= max_long:
                continue
            if candidate.side == "SHORT" and n_short >= max_short:
                continue

            # Compute penalized score
            penalized = self._apply_penalties(
                candidate, selected_symbols, selected_industries,
                returns_matrix, selected,
            )
            candidate.penalized_score = penalized

        # Re-sort by penalized score and select
        candidates.sort(key=lambda c: c.penalized_score, reverse=True)

        for candidate in candidates:
            if len(selected) >= self.config.max_picks:
                break

            if candidate.side == "LONG" and n_long >= max_long:
                continue
            if candidate.side == "SHORT" and n_short >= max_short:
                continue

            candidate.selected = True
            selected.append(candidate)
            selected_symbols.add(candidate.symbol)

            if candidate.industry:
                selected_industries[candidate.industry] = \
                    selected_industries.get(candidate.industry, 0) + 1

            if candidate.side == "LONG":
                n_long += 1
            else:
                n_short += 1

        return selected, candidates

    def _apply_penalties(
        self,
        candidate: PortfolioCandidate,
        selected_symbols: set[str],
        selected_industries: dict[str, int],
        returns_matrix: pd.DataFrame | None,
        selected: list[PortfolioCandidate],
    ) -> float:
        """Apply sector and correlation penalties to raw score."""
        score = candidate.score
        cfg = self.config

        # Sector penalty
        if candidate.industry and candidate.industry in selected_industries:
            sector_exposure = selected_industries[candidate.industry]
            max_sector_picks = max(1, int(cfg.max_picks * cfg.max_sector_exposure))
            if sector_exposure >= max_sector_picks:
                score *= (1.0 - cfg.sector_penalty * 2)  # double penalty if sector is full

        # Correlation penalty
        if returns_matrix is not None and selected:
            corr_penalty = self._compute_correlation_penalty(
                candidate.symbol,
                [s.symbol for s in selected],
                returns_matrix,
            )
            score *= (1.0 - corr_penalty * cfg.correlation_penalty)

        return score

    def _compute_correlation_penalty(
        self,
        candidate_symbol: str,
        selected_symbols: list[str],
        returns_matrix: pd.DataFrame,
    ) -> float:
        """
        Compute penalty based on correlation with already-selected stocks.
        Higher average correlation → higher penalty.
        """
        if candidate_symbol not in returns_matrix.columns:
            return 0.0

        valid_selected = [s for s in selected_symbols if s in returns_matrix.columns]
        if not valid_selected:
            return 0.0

        lookback = self.config.correlation_lookback_days
        recent_returns = returns_matrix.tail(lookback)

        try:
            candidate_rets = recent_returns[candidate_symbol].dropna()
            if len(candidate_rets) < 5:
                return 0.0

            correlations = []
            for sel_symbol in valid_selected:
                sel_rets = recent_returns[sel_symbol].dropna()
                common_idx = candidate_rets.index.intersection(sel_rets.index)
                if len(common_idx) < 5:
                    continue
                corr = candidate_rets.loc[common_idx].corr(sel_rets.loc[common_idx])
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            if not correlations:
                return 0.0

            return float(np.mean(correlations))
        except Exception:
            return 0.0

    def _compute_portfolio_metrics(
        self,
        selected: list[PortfolioCandidate],
    ) -> dict[str, float]:
        """Compute aggregated portfolio metrics."""
        if not selected:
            return {}

        scores = [c.score for c in selected]
        probs = [c.probability for c in selected]
        edges = [c.expected_edge for c in selected]

        return {
            "n_picks": len(selected),
            "n_long": sum(1 for c in selected if c.side == "LONG"),
            "n_short": sum(1 for c in selected if c.side == "SHORT"),
            "avg_score": float(np.mean(scores)),
            "avg_probability": float(np.mean(probs)),
            "avg_edge": float(np.mean(edges)),
            "industry_count": len(set(c.industry for c in selected if c.industry)),
            "max_same_industry": max(
                sum(1 for c in selected if c.industry == ind)
                for ind in set(c.industry for c in selected if c.industry)
            ) if selected else 0,
        }


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

def compute_position_sizes(
    portfolio: Portfolio,
    capital: float = 1000000.0,
    risk_per_trade_pct: float = 0.02,
    stop_loss_pct: float = 0.01,
    max_position_pct: float = 0.20,
) -> dict[str, float]:
    """
    Compute position sizes using equal risk allocation.

    Each pick gets: risk_budget / stop_loss_pct of capital,
    capped at max_position_pct of total capital.

    Returns dict mapping symbol → position value.
    """
    if not portfolio.selected:
        return {}

    risk_per_trade = capital * risk_per_trade_pct
    max_position = capital * max_position_pct
    positions = {}

    for candidate in portfolio.selected:
        position = risk_per_trade / stop_loss_pct
        position = min(position, max_position)
        position = max(position, capital * 0.01)  # minimum 1%
        positions[candidate.symbol] = round(position, 2)

    # Scale down if total exceeds capital
    total_position = sum(positions.values())
    if total_position > capital:
        scale = capital / total_position
        positions = {k: round(v * scale, 2) for k, v in positions.items()}

    return positions
