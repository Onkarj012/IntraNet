"""
Market Regime Detection for IntradayNet LightGBM V2.

Detects 5 market regimes and provides parameter adjustments:
- CALM_BULL: Low VIX, uptrend → relaxed thresholds
- CALM_BEAR: Low VIX, downtrend → prefer SHORT
- VOLATILE_BULL: High VIX, uptrend → stricter thresholds
- VOLATILE_BEAR: High VIX, downtrend → very strict, fewer positions
- EXTREME: VIX spike → no trading

Usage:
    regime, should_trade, reason = detect_regime(vix, vix_change, ...)
    adjustments = get_regime_adjustments(regime)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, Optional


class MarketRegime(Enum):
    CALM_BULL = "calm_bull"
    CALM_BEAR = "calm_bear"
    VOLATILE_BULL = "volatile_bull"
    VOLATILE_BEAR = "volatile_bear"
    EXTREME = "extreme"


@dataclass
class RegimeConfig:
    vix_low: float = 15.0
    vix_high: float = 22.0
    vix_extreme: float = 28.0
    trend_window: int = 10
    max_gap_pct: float = 1.5
    skip_expiry: bool = False


def detect_regime(
    vix: float,
    vix_change: float,
    nifty_returns_10d: Optional[np.ndarray] = None,
    gap_pct: float = 0.0,
    is_expiry: bool = False,
    config: RegimeConfig = RegimeConfig(),
) -> Tuple[MarketRegime, bool, str]:
    if vix > config.vix_extreme:
        return (
            MarketRegime.EXTREME,
            False,
            f"VIX={vix:.1f} > {config.vix_extreme}",
        )

    if vix_change > 0.20:
        return (
            MarketRegime.EXTREME,
            False,
            f"VIX spiked {vix_change*100:.1f}%",
        )

    if abs(gap_pct) > config.max_gap_pct:
        return (
            MarketRegime.EXTREME,
            False,
            f"Gap={gap_pct:.2f}%",
        )

    if is_expiry and config.skip_expiry:
        return (
            MarketRegime.EXTREME,
            False,
            "Expiry day",
        )

    if nifty_returns_10d is not None and len(nifty_returns_10d) > 0:
        trend = np.mean(nifty_returns_10d) > 0
    else:
        trend = True

    is_volatile = vix > config.vix_high

    if is_volatile:
        regime = MarketRegime.VOLATILE_BULL if trend else MarketRegime.VOLATILE_BEAR
    else:
        regime = MarketRegime.CALM_BULL if trend else MarketRegime.CALM_BEAR

    return regime, True, f"Regime={regime.value}, VIX={vix:.1f}"


def get_regime_adjustments(regime: MarketRegime) -> Dict:
    adjustments = {
        MarketRegime.CALM_BULL: {
            "dir_threshold": 0.58,
            "min_confidence": 0.55,
            "stop_loss_pct": 0.008,
            "max_positions": 5,
            "prefer_direction": "LONG",
        },
        MarketRegime.CALM_BEAR: {
            "dir_threshold": 0.60,
            "min_confidence": 0.58,
            "stop_loss_pct": 0.010,
            "max_positions": 4,
            "prefer_direction": "SHORT",
        },
        MarketRegime.VOLATILE_BULL: {
            "dir_threshold": 0.63,
            "min_confidence": 0.60,
            "stop_loss_pct": 0.012,
            "max_positions": 3,
            "prefer_direction": "LONG",
        },
        MarketRegime.VOLATILE_BEAR: {
            "dir_threshold": 0.65,
            "min_confidence": 0.62,
            "stop_loss_pct": 0.015,
            "max_positions": 2,
            "prefer_direction": "SHORT",
        },
        MarketRegime.EXTREME: {
            "dir_threshold": 1.0,
            "min_confidence": 1.0,
            "stop_loss_pct": 0.02,
            "max_positions": 0,
            "prefer_direction": None,
        },
    }
    return adjustments.get(regime, adjustments[MarketRegime.CALM_BULL])


def get_regime_from_market_data(
    nifty50_path: str = "market_data_cache/nifty50.csv",
    india_vix_path: str = "market_data_cache/india_vix.csv",
    date: str = None,
) -> Tuple[MarketRegime, bool, str]:
    try:
        nifty = pd.read_csv(nifty50_path, parse_dates=["Date"], index_col="Date")
        vix_df = pd.read_csv(india_vix_path, parse_dates=["Date"], index_col="Date")
    except Exception:
        return MarketRegime.CALM_BULL, True, "Data unavailable, using default"

    if date is not None:
        ref_date = pd.Timestamp(date)
    else:
        ref_date = pd.Timestamp.now()

    prev_date = ref_date - pd.Timedelta(days=30)
    nifty_window = nifty.loc[prev_date:ref_date, "close"] if "close" in nifty.columns else None
    vix_window = vix_df.loc[ref_date:ref_date, "close"] if "close" in vix_df.columns else None

    if nifty_window is None or len(nifty_window) < 5:
        return MarketRegime.CALM_BULL, True, "Insufficient data"

    nifty_ret = nifty_window.pct_change().dropna().values
    vix_level = vix_window.iloc[0] if vix_window is not None and len(vix_window) > 0 else 15.0

    if len(nifty_ret) > 1:
        vix_change = (vix_level - vix_window.iloc[-2]) / vix_window.iloc[-2] if vix_window is not None else 0
    else:
        vix_change = 0

    return detect_regime(
        vix=vix_level,
        vix_change=vix_change,
        nifty_returns_10d=nifty_ret[-10:] if len(nifty_ret) >= 10 else nifty_ret,
        gap_pct=0.0,
        is_expiry=False,
    )
