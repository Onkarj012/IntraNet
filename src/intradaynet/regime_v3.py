"""
4-State Market Regime Classifier for IntradayNet v3.0

Classifies market into 4 regimes + extreme:
- TRENDING_CALM: VIX < 15, ADX > 25 - Best regime, momentum works
- TRENDING_VOLATILE: VIX 15-22, ADX > 25 - Momentum works but wider stops needed
- CHOPPY_CALM: VIX < 15, ADX < 20 - Mean reversion works, momentum fails
- CHOPPY_VOLATILE: VIX > 22, ADX < 20 - Don't trade, or 50% size
- EXTREME: VIX spike or other extreme conditions - No trading

Usage:
    from intradaynet.regime_v3 import RegimeClassifierV3, MarketRegime
    
    classifier = RegimeClassifierV3()
    regime = classifier.classify(vix, nifty_data, date)
    
    if regime == MarketRegime.CHOPPY_VOLATILE:
        skip_trading = True
    elif regime == MarketRegime.TRENDING_CALM:
        target_multiplier = 1.5  # Let winners run
        stop_multiplier = 0.8
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path


class MarketRegime(Enum):
    """4 primary market regimes + extreme state."""
    TRENDING_CALM = "trending_calm"
    TRENDING_VOLATILE = "trending_volatile"
    CHOPPY_CALM = "choppy_calm"
    CHOPPY_VOLATILE = "choppy_volatile"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


@dataclass
class RegimeThresholds:
    """Thresholds for regime classification."""
    vix_low: float = 15.0
    vix_high: float = 22.0
    vix_extreme: float = 28.0
    adx_trend_threshold: float = 25.0
    adx_chop_threshold: float = 20.0
    max_gap_pct: float = 1.5
    max_vix_change_pct: float = 20.0
    trend_window: int = 20


@dataclass
class RegimeAdjustments:
    """Trading adjustments for each regime."""
    # Entry/exit thresholds
    direction_threshold: float = 0.58
    min_confidence: float = 0.55
    
    # Position sizing
    max_positions: int = 5
    size_multiplier: float = 1.0  # 0.5 for choppy_volatile
    
    # Stop/target multipliers (relative to ATR)
    target_atr_multiplier: float = 2.0
    stop_atr_multiplier: float = 1.0
    
    # Trailing stop settings
    use_trailing_stop: bool = True
    trailing_activation_pct: float = 0.5  # Activate at +0.5%
    trailing_distance_pct: float = 0.3    # Trail at +0.3%
    
    # Time-based exit
    max_holding_bars: int = 60  # ~1 hour
    force_exit_eod: bool = True
    
    # Should we trade at all?
    allow_trading: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'direction_threshold': self.direction_threshold,
            'min_confidence': self.min_confidence,
            'max_positions': self.max_positions,
            'size_multiplier': self.size_multiplier,
            'target_atr_multiplier': self.target_atr_multiplier,
            'stop_atr_multiplier': self.stop_atr_multiplier,
            'use_trailing_stop': self.use_trailing_stop,
            'trailing_activation_pct': self.trailing_activation_pct,
            'trailing_distance_pct': self.trailing_distance_pct,
            'max_holding_bars': self.max_holding_bars,
            'force_exit_eod': self.force_exit_eod,
            'allow_trading': self.allow_trading,
        }


class RegimeClassifierV3:
    """
    4-state regime classifier with regime-conditional parameters.
    
    Key improvement over v2: Uses ADX (Average Directional Index) to measure
    trend strength, creating 4 distinct regimes rather than just 2.
    """
    
    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.thresholds = thresholds or RegimeThresholds()
        
    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Compute Average Directional Index (ADX).
        
        ADX measures trend strength (not direction):
        - ADX > 25: Strong trend
        - ADX < 20: Weak trend / choppy
        """
        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smooth TR and DM
        atr = tr.ewm(span=period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr.replace(0, np.nan)
        
        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, min_periods=period).mean()
        
        return adx.fillna(20)  # Neutral default
    
    def _compute_nifty_trend(self, nifty_df: pd.DataFrame) -> float:
        """Compute Nifty 20-day trend strength and direction."""
        if len(nifty_df) < 20:
            return 0.0
        
        # Use closing prices
        returns = nifty_df['close'].pct_change().tail(20)
        return returns.mean()  # Positive = uptrend, negative = downtrend
    
    def classify(
        self,
        vix_level: float,
        vix_change_pct: float,
        nifty_df: Optional[pd.DataFrame] = None,
        gap_pct: float = 0.0,
        date: Optional[str] = None,
    ) -> Tuple[MarketRegime, str, RegimeAdjustments]:
        """
        Classify market regime and return trading adjustments.
        
        Args:
            vix_level: Current VIX value
            vix_change_pct: VIX change from previous day (in %)
            nifty_df: DataFrame with Nifty OHLCV data
            gap_pct: Overnight gap percentage
            date: Date string for logging
            
        Returns:
            (regime, reason, adjustments)
        """
        t = self.thresholds
        
        # Check extreme conditions first
        if vix_level > t.vix_extreme:
            return (
                MarketRegime.EXTREME,
                f"VIX spike: {vix_level:.1f} > {t.vix_extreme}",
                self._get_extreme_adjustments()
            )
        
        if abs(vix_change_pct) > t.max_vix_change_pct:
            return (
                MarketRegime.EXTREME,
                f"VIX jump: {vix_change_pct:.1f}% > {t.max_vix_change_pct}%",
                self._get_extreme_adjustments()
            )
        
        if abs(gap_pct) > t.max_gap_pct:
            return (
                MarketRegime.EXTREME,
                f"Large gap: {gap_pct:.2f}% > {t.max_gap_pct}%",
                self._get_extreme_adjustments()
            )
        
        # Compute ADX and trend
        if nifty_df is not None and len(nifty_df) >= 20:
            adx = self._compute_adx(
                nifty_df['high'], 
                nifty_df['low'], 
                nifty_df['close']
            ).iloc[-1] if 'high' in nifty_df.columns else 20
            trend = self._compute_nifty_trend(nifty_df)
        else:
            adx = 20  # Neutral
            trend = 0.0
        
        # Classify based on VIX and ADX
        is_volatile = vix_level > t.vix_high
        is_calm = vix_level < t.vix_low
        is_trending = adx > t.adx_trend_threshold
        is_choppy = adx < t.adx_chop_threshold
        
        # Determine regime
        if is_trending:
            if is_volatile or vix_level > t.vix_high:
                regime = MarketRegime.TRENDING_VOLATILE
                reason = f"Trending (ADX={adx:.1f}) + Volatile (VIX={vix_level:.1f})"
            else:
                regime = MarketRegime.TRENDING_CALM
                reason = f"Trending (ADX={adx:.1f}) + Calm (VIX={vix_level:.1f})"
        elif is_choppy or adx < 25:
            if is_volatile or vix_level > t.vix_high:
                regime = MarketRegime.CHOPPY_VOLATILE
                reason = f"Choppy (ADX={adx:.1f}) + Volatile (VIX={vix_level:.1f})"
            else:
                regime = MarketRegime.CHOPPY_CALM
                reason = f"Choppy (ADX={adx:.1f}) + Calm (VIX={vix_level:.1f})"
        else:
            # Middle zone - default based on trend
            if trend > 0:
                regime = MarketRegime.TRENDING_CALM
                reason = f"Slight trend (ADX={adx:.1f}, trend={trend:.3f})"
            else:
                regime = MarketRegime.CHOPPY_CALM
                reason = f"Slight chop (ADX={adx:.1f}, trend={trend:.3f})"
        
        adjustments = self._get_adjustments_for_regime(regime, trend)
        
        return regime, reason, adjustments
    
    def _get_adjustments_for_regime(
        self, 
        regime: MarketRegime,
        trend: float = 0
    ) -> RegimeAdjustments:
        """Get trading adjustments for each regime."""
        
        if regime == MarketRegime.TRENDING_CALM:
            # Best regime - wider targets, let winners run
            return RegimeAdjustments(
                direction_threshold=0.56,
                min_confidence=0.54,
                max_positions=5,
                size_multiplier=1.0,
                target_atr_multiplier=2.5,  # Wide target
                stop_atr_multiplier=1.0,    # Normal stop
                use_trailing_stop=True,
                trailing_activation_pct=0.5,
                trailing_distance_pct=0.3,
                max_holding_bars=80,  # Hold longer in trends
                allow_trading=True,
            )
        
        elif regime == MarketRegime.TRENDING_VOLATILE:
            # Trending but volatile - wider stops, fewer positions
            return RegimeAdjustments(
                direction_threshold=0.60,
                min_confidence=0.58,
                max_positions=4,
                size_multiplier=0.8,
                target_atr_multiplier=2.0,
                stop_atr_multiplier=1.3,  # Wider stop for volatility
                use_trailing_stop=True,
                trailing_activation_pct=0.8,  # Need more profit before trailing
                trailing_distance_pct=0.5,     # Wider trail
                max_holding_bars=60,
                allow_trading=True,
            )
        
        elif regime == MarketRegime.CHOPPY_CALM:
            # Choppy but calm - tight targets, quick profits
            return RegimeAdjustments(
                direction_threshold=0.62,
                min_confidence=0.60,
                max_positions=3,
                size_multiplier=0.8,
                target_atr_multiplier=1.2,  # Tight target
                stop_atr_multiplier=0.9,    # Tight stop
                use_trailing_stop=True,
                trailing_activation_pct=0.3,
                trailing_distance_pct=0.2,
                max_holding_bars=40,  # Quick exit
                allow_trading=True,
            )
        
        elif regime == MarketRegime.CHOPPY_VOLATILE:
            # Worst regime - minimal trading or skip
            return RegimeAdjustments(
                direction_threshold=0.65,
                min_confidence=0.62,
                max_positions=2,
                size_multiplier=0.5,  # Half size
                target_atr_multiplier=1.0,
                stop_atr_multiplier=0.8,
                use_trailing_stop=False,
                trailing_activation_pct=0.5,
                trailing_distance_pct=0.3,
                max_holding_bars=30,
                allow_trading=False,  # Skip by default
            )
        
        else:  # EXTREME
            return self._get_extreme_adjustments()
    
    def _get_extreme_adjustments(self) -> RegimeAdjustments:
        """No trading in extreme conditions."""
        return RegimeAdjustments(
            direction_threshold=1.0,  # Impossible to reach
            min_confidence=1.0,
            max_positions=0,
            size_multiplier=0.0,
            target_atr_multiplier=1.0,
            stop_atr_multiplier=1.0,
            use_trailing_stop=False,
            allow_trading=False,
        )
    
    def get_regime_from_market_data(
        self,
        nifty50_path: str = "market_data_cache/nifty50.csv",
        india_vix_path: str = "market_data_cache/india_vix.csv",
        date: Optional[str] = None,
    ) -> Tuple[MarketRegime, str, RegimeAdjustments]:
        """
        Load market data and classify regime for a given date.
        
        Should be called at 8:45 AM IST before market open.
        """
        try:
            # Load data
            nifty = pd.read_csv(nifty50_path, parse_dates=["Date"], index_col="Date")
            vix_df = pd.read_csv(india_vix_path, parse_dates=["Date"], index_col="Date")
        except Exception as e:
            # Default to calm if data unavailable
            return (
                MarketRegime.UNKNOWN,
                f"Data unavailable: {e}",
                self._get_adjustments_for_regime(MarketRegime.TRENDING_CALM)
            )
        
        if date is not None:
            ref_date = pd.Timestamp(date)
        else:
            ref_date = pd.Timestamp.now()
        
        # Get previous day's data
        prev_date = ref_date - pd.Timedelta(days=1)
        
        # Get VIX data
        vix_window = vix_df.loc[:ref_date]
        if len(vix_window) < 1:
            return (
                MarketRegime.UNKNOWN,
                "No VIX data available",
                self._get_adjustments_for_regime(MarketRegime.TRENDING_CALM)
            )
        
        vix_level = vix_window['close'].iloc[-1]
        
        # Compute VIX change
        if len(vix_window) >= 2:
            vix_change = (vix_level - vix_window['close'].iloc[-2]) / vix_window['close'].iloc[-2] * 100
        else:
            vix_change = 0.0
        
        # Get Nifty data for ADX calculation (need 30 days)
        nifty_window = nifty.loc[prev_date - pd.Timedelta(days=35):prev_date]
        
        # Compute gap (if any)
        gap_pct = 0.0  # Would come from pre-market data
        
        return self.classify(
            vix_level=vix_level,
            vix_change_pct=vix_change,
            nifty_df=nifty_window if len(nifty_window) >= 20 else None,
            gap_pct=gap_pct,
            date=date,
        )


# Convenience function for backward compatibility
def detect_regime_v3(
    vix: float,
    vix_change: float,
    nifty_returns_10d: Optional[np.ndarray] = None,
    gap_pct: float = 0.0,
    is_expiry: bool = False,
) -> Tuple[MarketRegime, str, RegimeAdjustments]:
    """
    Simple wrapper for regime detection.
    
    Compatible with old regime.py interface but returns new 4-state regime.
    """
    classifier = RegimeClassifierV3()
    
    # Create dummy nifty_df if returns provided
    nifty_df = None
    if nifty_returns_10d is not None and len(nifty_returns_10d) >= 10:
        # Approximate from returns
        closes = 100 * (1 + np.cumsum(nifty_returns_10d))
        nifty_df = pd.DataFrame({
            'close': closes,
            'high': closes * 1.01,
            'low': closes * 0.99,
        })
    
    return classifier.classify(
        vix_level=vix,
        vix_change_pct=vix_change * 100,  # Convert to pct
        nifty_df=nifty_df,
        gap_pct=gap_pct,
    )


if __name__ == "__main__":
    # Test the classifier
    classifier = RegimeClassifierV3()
    
    test_cases = [
        (12, 5, "Low VIX, trending"),
        (18, 5, "Medium VIX, trending"),
        (12, -5, "Low VIX, choppy"),
        (25, -5, "High VIX, choppy"),
        (30, 10, "VIX spike"),
    ]
    
    print("4-State Regime Classifier Test:")
    print("-" * 60)
    
    for vix, trend, desc in test_cases:
        # Create mock data
        dates = pd.date_range(end='2025-01-15', periods=30, freq='D')
        closes = 20000 * (1 + np.cumsum(np.random.randn(30) * 0.001 + trend * 0.0001))
        nifty_df = pd.DataFrame({
            'close': closes,
            'high': closes * (1 + abs(np.random.randn(30) * 0.005)),
            'low': closes * (1 - abs(np.random.randn(30) * 0.005)),
        }, index=dates)
        
        regime, reason, adj = classifier.classify(
            vix_level=vix,
            vix_change_pct=0,
            nifty_df=nifty_df,
            gap_pct=0,
        )
        
        print(f"\n{desc}:")
        print(f"  VIX={vix}, Trend={'Up' if trend > 0 else 'Down'}")
        print(f"  Regime: {regime.value}")
        print(f"  Reason: {reason}")
        print(f"  Trading: {'YES' if adj.allow_trading else 'NO'}")
        print(f"  Max Positions: {adj.max_positions}")
        print(f"  Target ATR Multiplier: {adj.target_atr_multiplier:.1f}x")
        print(f"  Stop ATR Multiplier: {adj.stop_atr_multiplier:.1f}x")
