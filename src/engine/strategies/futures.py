"""FuturesEngine — NIFTY long-only directional futures trades.

Sprint #1 in the OptiNet Router roadmap. v1 IMPLEMENTED.

Hard filters (v1):
  - LONG only (SHORT disabled)
  - Skip 11:00-11:59 (mid-morning consolidation)
  - Skip compression regime
  - No entries before 09:45 or after 14:55

Signal: long_score > 85th percentile of day's eligible scores.
Size:   1.5x when score > 95th percentile, else 1.0x.

Walk-forward AUC: 0.677 mean across 15 quarterly folds (14/15 > 0.55).
2024 blind: +₹287,536, 55.1% win, Sharpe +2.88, PF 1.54.
"""
from __future__ import annotations

from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np

from engine.strategies.base import FamilyEngine
from engine.features import FUTURES_FEATURES, add_regime
from engine.schema import (
    Instrument, LegSide, MarketState, OptionType, Recommendation,
    Regime, Risk, StrategyFamily, TradeCard, TradeLeg, VolCondition,
    make_no_trade_card,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL = _REPO_ROOT / "models/router_v0/futures/final_long.lgb"

# Hard filter times
_SKIP_START = dtime(11, 0)
_SKIP_END   = dtime(12, 0)
_CUTOFF     = dtime(14, 55)
_ENTRY_MIN  = 30   # minutes from 09:15 open

# Execution params (must match backtest)
_TARGET_PCT = 0.0040
_STOP_PCT   = 0.0030
_HORIZON    = 60
_LOT        = 50
_COSTS_INR  = 105.0
_STOP_FLOOR = -3000.0

# Percentile thresholds — set per-day by the caller via day_context
_SIGNAL_PCT   = 0.85
_HIGH_CONF_PCT = 0.95

_SKIP_REGIMES = {"compression"}

_REGIME_MAP = {
    "trend_up":    Regime.TREND,
    "trend_dn":    Regime.TREND,
    "expansion":   Regime.EXPANSION,
    "compression": Regime.COMPRESSION,
    "range":       Regime.RANGE,
}
_VOL_MAP = {
    "expansion":   VolCondition.ELEVATED,
    "compression": VolCondition.COMPRESSED,
    "trend_up":    VolCondition.NORMAL,
    "trend_dn":    VolCondition.NORMAL,
    "range":       VolCondition.NORMAL,
}


class FuturesEngine(FamilyEngine):
    """NIFTY long-only futures trade-card engine (v1).

    Usage:
        engine = FuturesEngine()
        # At the start of each day, call reset_day() to clear the score buffer.
        engine.reset_day()
        # Each minute, call update_day_scores() with the current score,
        # then call score_minute() to get a TradeCard.
    """
    name = "futures_v1"

    def __init__(self, model_path: Optional[Path] = None, symbol: str = "NIFTY"):
        self.symbol = symbol
        path = model_path or _DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(f"FuturesEngine model not found: {path}")
        self._model = lgb.Booster(model_file=str(path))
        self._day_scores: list[float] = []   # accumulates scores within a day

    def health(self) -> bool:
        return self._model is not None

    def reset_day(self) -> None:
        """Call at the start of each trading day."""
        self._day_scores = []

    def _compute_score(self, features: dict[str, Any]) -> Optional[float]:
        import pandas as pd
        row = pd.DataFrame([{f: features.get(f, np.nan) for f in FUTURES_FEATURES}])
        if row.isna().any(axis=1).iloc[0]:
            return None
        return float(self._model.predict(row)[0])

    def _regime_from_features(self, features: dict[str, Any]) -> str:
        import pandas as pd
        row = pd.DataFrame([features])
        try:
            return str(add_regime(row)["regime"].iloc[0])
        except Exception:
            return "range"

    def score_minute(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        market_state: MarketState,
        features: dict[str, Any],
    ) -> Optional[Recommendation]:
        """Return a FUTURES_LONG Recommendation if conditions are met, else None."""
        t = timestamp.time()
        mod = timestamp.hour * 60 + timestamp.minute - 9 * 60 - 15

        # Hard filters
        if mod < _ENTRY_MIN:
            return None
        if t >= _CUTOFF:
            return None
        if _SKIP_START <= t < _SKIP_END:
            return None
        regime_str = self._regime_from_features(features)
        if regime_str in _SKIP_REGIMES:
            return None

        score = self._compute_score(features)
        if score is None:
            return None

        # Accumulate score for percentile threshold
        self._day_scores.append(score)
        if len(self._day_scores) < 10:
            return None  # need enough scores to compute percentile

        p85 = float(np.percentile(self._day_scores, _SIGNAL_PCT * 100))
        p95 = float(np.percentile(self._day_scores, _HIGH_CONF_PCT * 100))
        if score < p85:
            return None

        # Build the recommendation
        spot = float(features.get("f_close", features.get("fut_close",
                     features.get("f_close", 0))))
        if spot <= 0:
            return None

        size_mult = 1.5 if score >= p95 else 1.0
        entry_low  = spot * (1 - 0.0005)
        entry_high = spot * (1 + 0.0005)
        stop_px    = spot * (1 - _STOP_PCT)
        target_px  = spot * (1 + _TARGET_PCT)
        from datetime import timedelta
        time_stop_dt = datetime.combine(timestamp.date(),
                                         dtime(15, 25))

        # Expiry: use next Thursday as a placeholder for futures
        from engine.config import next_weekly_expiry
        expiry = next_weekly_expiry("NIFTY", timestamp.date())

        return Recommendation(
            instrument=Instrument.NIFTY_FUT,
            strategy=StrategyFamily.FUTURES_LONG,
            legs=(
                TradeLeg(
                    side=LegSide.BUY,
                    instrument_kind="futures",
                    expiry=expiry,
                    strike=None,
                    opt_type=None,
                    lots=int(size_mult),
                ),
            ),
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            suggested_size_lots=int(size_mult),
            entry_window_start=timestamp,
            entry_window_end=timestamp + timedelta(minutes=5),
            stop_loss_inr=float(_STOP_FLOOR),
            target_inr=float(_TARGET_PCT * spot * _LOT * size_mult),
            time_stop=time_stop_dt,
            reason_codes=tuple(self._build_reason_codes(features, regime_str, score, p85, p95)),
        )

    def _build_reason_codes(self, features: dict, regime: str,
                              score: float, p85: float, p95: float) -> list[str]:
        codes = []
        if regime == "expansion":  codes.append("EXPANSION_REGIME")
        if regime == "trend_up":   codes.append("TREND_UP_REGIME")
        if features.get("oi_long_buildup", 0): codes.append("OI_LONG_BUILDUP")
        if features.get("or_breakout_up", 0):  codes.append("OR_BREAKOUT_UP")
        if features.get("ema_slope", 0) > 0.002: codes.append("EMA_TREND_UP")
        if features.get("vwap_dev", 0) > 0.001:  codes.append("ABOVE_VWAP")
        if score >= p95: codes.append("HIGH_CONFIDENCE")
        return codes or ["MODEL_SIGNAL"]

    def expected_edge_inr(self, recommendation: Recommendation) -> float:
        # From 2024 blind: mean PnL +₹462/trade, size-adjusted
        lots = recommendation.suggested_size_lots
        return 462.0 * lots

    def make_trade_card(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        features: dict[str, Any],
        model_version: str = "futures_v1",
    ) -> TradeCard:
        """Convenience: produce a full TradeCard (including NO_TRADE) for one minute."""
        regime_str = self._regime_from_features(features)
        rv = float(features.get("realized_vol_30m", 0.15))
        vol_cond = _VOL_MAP.get(regime_str, VolCondition.NORMAL)
        market_state = MarketState(
            regime=_REGIME_MAP.get(regime_str, Regime.RANGE),
            regime_confidence=0.70,
            vol_condition=vol_cond,
            time_horizon_minutes=60,
        )
        rec = self.score_minute(
            timestamp=timestamp, symbol=symbol,
            market_state=market_state, features=features,
        )
        if rec is None:
            return make_no_trade_card(
                timestamp=timestamp, symbol=symbol,
                market_state=market_state,
                reason="NO_SIGNAL",
                model_version=model_version,
                feature_snapshot={k: features.get(k) for k in
                                    ["realized_vol_30m", "ema_slope", "vwap_dev",
                                     "regime", "or_breakout_up"]},
            )
        risk = Risk(
            expected_edge_inr=self.expected_edge_inr(rec),
            expected_edge_confidence=0.55,
            tail_risk_warning=("High vol — stop may gap" if rv > 0.25 else None),
            do_not_trade=False,
            do_not_trade_reason=None,
        )
        return TradeCard(
            timestamp=timestamp, symbol=symbol,
            market_state=market_state, recommendation=rec, risk=risk,
            feature_snapshot={k: features.get(k) for k in
                               ["realized_vol_30m", "ema_slope", "vwap_dev",
                                "or_breakout_up", "oi_long_buildup"]},
            model_version=model_version,
        )

