"""
ATR-Based Dynamic Targets for IntradayNet v3.0

Replaces fixed percentage targets with ATR-based dynamic levels
that adapt to market regime and stock volatility.

Key improvements:
1. Target = Entry ± (ATR_14 × target_multiplier)
2. Stop = Entry ∓ (ATR_14 × stop_multiplier)
3. Multipliers are functions of regime and model confidence
   - High confidence + Trending: target_mult = 2.5, stop_mult = 1.0
   - Low confidence + Choppy: target_mult = 1.0, stop_mult = 0.8

This widens the win/loss ratio and protects the thin edge.

Usage:
    from intradaynet.dynamic_targets import DynamicTargetManager
    from intradaynet.regime_v3 import RegimeClassifierV3
    
    classifier = RegimeClassifierV3()
    target_mgr = DynamicTargetManager(classifier)
    
    # Before each trade
    regime = classifier.classify(vix, nifty_data, date)
    entry_price = 1000
    atr = 15  # ATR14
    confidence = 0.65
    
    target, stop = target_mgr.compute_levels(
        entry_price=entry_price,
        atr=atr,
        side="LONG",
        regime=regime,
        confidence=confidence,
    )
    
    # Returns: target=1037.5 (2.5×ATR), stop=985 (1.0×ATR)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from intradaynet.regime_v3 import MarketRegime, RegimeClassifierV3, RegimeAdjustments


@dataclass
class DynamicTargetConfig:
    """Configuration for dynamic target calculation."""
    
    # Base multipliers (will be scaled by regime and confidence)
    base_target_atr_mult: float = 2.0
    base_stop_atr_mult: float = 1.0
    
    # Confidence scaling
    high_confidence_threshold: float = 0.65
    low_confidence_threshold: float = 0.55
    
    high_confidence_target_boost: float = 0.5  # Add 0.5 to multiplier
    low_confidence_target_reduction: float = 0.5  # Subtract 0.5
    
    # Minimum/maximum bounds (as percentages)
    min_target_pct: float = 0.005  # 0.5%
    max_target_pct: float = 0.03   # 3%
    min_stop_pct: float = 0.003  # 0.3%
    max_stop_pct: float = 0.02   # 2%
    
    # Trailing stop settings
    trailing_activation_atr_mult: float = 1.0  # Activate at +1×ATR profit
    trailing_distance_atr_mult: float = 0.5    # Trail at 0.5×ATR below peak


class DynamicTargetManager:
    """
    Manages ATR-based dynamic targets with regime and confidence adjustments.
    
    Key principle: In trending markets, let winners run (wider targets).
    In choppy markets, take quick profits (tighter targets).
    """
    
    def __init__(self, config: Optional[DynamicTargetConfig] = None):
        self.config = config or DynamicTargetConfig()
        self.regime_classifier = RegimeClassifierV3()
    
    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float:
        """
        Compute Average True Range for a price series.
        
        Returns the most recent ATR value.
        """
        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR using Wilder's smoothing
        atr = tr.ewm(span=period, min_periods=period).mean()
        
        return float(atr.iloc[-1])
    
    def compute_atr_from_ohlcv(self, ohlcv: np.ndarray, period: int = 14) -> float:
        """
        Compute ATR from OHLCV numpy array.
        
        Args:
            ohlcv: (n_bars, 5) array [open, high, low, close, volume]
        """
        h = ohlcv[:, 1]
        l = ohlcv[:, 2]
        c = ohlcv[:, 3]
        
        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        
        tr1 = h - l
        tr2 = np.abs(h - prev_c)
        tr3 = np.abs(l - prev_c)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        alpha = 2.0 / (period + 1)
        for i in range(1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        return float(atr[-1])
    
    def get_multipliers(
        self,
        regime: MarketRegime,
        confidence: float,
    ) -> Tuple[float, float]:
        """
        Get target and stop multipliers based on regime and confidence.
        
        Returns (target_mult, stop_mult)
        """
        config = self.config
        
        # Get regime base multipliers from regime adjustments
        _, _, adj = self.regime_classifier.classify(
            vix_level=15,  # Dummy - would use actual
            vix_change_pct=0,
        )
        
        # Start with regime-specific base
        if regime == MarketRegime.TRENDING_CALM:
            target_mult = 2.5
            stop_mult = 1.0
        elif regime == MarketRegime.TRENDING_VOLATILE:
            target_mult = 2.0
            stop_mult = 1.3  # Wider stop for volatility
        elif regime == MarketRegime.CHOPPY_CALM:
            target_mult = 1.2
            stop_mult = 0.9  # Tight stop
        elif regime == MarketRegime.CHOPPY_VOLATILE:
            target_mult = 1.0
            stop_mult = 0.8  # Very tight
        elif regime == MarketRegime.EXTREME:
            return (0.0, 0.0)  # No trading
        else:
            target_mult = 2.0
            stop_mult = 1.0
        
        # Adjust by confidence
        if confidence >= config.high_confidence_threshold:
            # High confidence: widen target, tighten stop
            target_mult += config.high_confidence_target_boost
            stop_mult = max(0.8, stop_mult - 0.1)
        elif confidence <= config.low_confidence_threshold:
            # Low confidence: tighten everything
            target_mult = max(1.0, target_mult - config.low_confidence_target_reduction)
            stop_mult = min(1.5, stop_mult + 0.2)
        
        return target_mult, stop_mult
    
    def compute_levels(
        self,
        entry_price: float,
        atr: float,
        side: str,
        regime: MarketRegime,
        confidence: float,
    ) -> Tuple[float, float, Dict]:
        """
        Compute target and stop-loss levels for a trade.
        
        Args:
            entry_price: Trade entry price
            atr: 14-period ATR value
            side: "LONG" or "SHORT"
            regime: Current market regime
            confidence: Model confidence (0-1)
            
        Returns:
            (target_price, stop_price, metadata_dict)
        """
        if entry_price <= 0 or atr <= 0:
            raise ValueError(f"Invalid inputs: entry={entry_price}, atr={atr}")
        
        # Get multipliers
        target_mult, stop_mult = self.get_multipliers(regime, confidence)
        
        if target_mult == 0 or stop_mult == 0:
            # No trading in extreme regime
            return (0.0, 0.0, {'skip_trade': True, 'reason': 'extreme_regime'})
        
        # Compute price distances
        target_distance = atr * target_mult
        stop_distance = atr * stop_mult
        
        # Apply min/max bounds as percentages
        min_target_dist = entry_price * self.config.min_target_pct
        max_target_dist = entry_price * self.config.max_target_pct
        min_stop_dist = entry_price * self.config.min_stop_pct
        max_stop_dist = entry_price * self.config.max_stop_pct
        
        target_distance = np.clip(target_distance, min_target_dist, max_target_dist)
        stop_distance = np.clip(stop_distance, min_stop_dist, max_stop_dist)
        
        # Compute levels based on side
        side = side.upper()
        if side == "LONG":
            target_price = entry_price + target_distance
            stop_price = entry_price - stop_distance
        else:  # SHORT
            target_price = entry_price - target_distance
            stop_price = entry_price + stop_distance
        
        # Compute risk/reward ratio
        risk = stop_distance
        reward = target_distance
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Compile metadata
        metadata = {
            'entry_price': entry_price,
            'atr': atr,
            'target_mult': target_mult,
            'stop_mult': stop_mult,
            'target_distance': target_distance,
            'stop_distance': stop_distance,
            'target_distance_pct': target_distance / entry_price * 100,
            'stop_distance_pct': stop_distance / entry_price * 100,
            'risk_reward_ratio': risk_reward_ratio,
            'regime': regime.value,
            'confidence': confidence,
            'side': side,
        }
        
        return target_price, stop_price, metadata
    
    def compute_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float,  # Highest (LONG) or lowest (SHORT) price since entry
        atr: float,
        side: str,
        regime: MarketRegime,
    ) -> Tuple[Optional[float], Dict]:
        """
        Compute trailing stop level if activated.
        
        Returns (trailing_stop_price, metadata) or (None, {}) if not activated.
        """
        config = self.config
        side = side.upper()
        
        # Compute profit in ATR terms
        if side == "LONG":
            profit_atr = (current_price - entry_price) / atr
            activation_threshold = config.trailing_activation_atr_mult
            
            # Check if trailing should activate
            if profit_atr < activation_threshold:
                return (None, {'activated': False, 'reason': 'insufficient_profit'})
            
            # Compute trailing stop
            trail_distance = atr * config.trailing_distance_atr_mult
            trailing_stop = peak_price - trail_distance
            
        else:  # SHORT
            profit_atr = (entry_price - current_price) / atr
            activation_threshold = config.trailing_activation_atr_mult
            
            if profit_atr < activation_threshold:
                return (None, {'activated': False, 'reason': 'insufficient_profit'})
            
            trail_distance = atr * config.trailing_distance_atr_mult
            trailing_stop = peak_price + trail_distance
        
        metadata = {
            'activated': True,
            'peak_price': peak_price,
            'profit_atr': profit_atr,
            'trail_distance': trail_distance,
            'trailing_stop': trailing_stop,
        }
        
        return trailing_stop, metadata
    
    def get_regime_adjustments(self, regime: MarketRegime) -> Dict:
        """
        Get full trading adjustments for a regime.
        
        Returns dict with all parameters needed for position sizing,
        entry/exit logic, etc.
        """
        _, _, adj = self.regime_classifier.classify(
            vix_level=15,
            vix_change_pct=0,
        )
        
        return {
            'regime': regime.value,
            'allow_trading': adj.allow_trading,
            'max_positions': adj.max_positions,
            'size_multiplier': adj.size_multiplier,
            'direction_threshold': adj.direction_threshold,
            'min_confidence': adj.min_confidence,
            'target_atr_multiplier': adj.target_atr_multiplier,
            'stop_atr_multiplier': adj.stop_atr_multiplier,
            'use_trailing_stop': adj.use_trailing_stop,
            'max_holding_bars': adj.max_holding_bars,
        }


def main():
    """Demonstrate dynamic targets across different regimes."""
    print("\n" + "="*70)
    print("ATR-BASED DYNAMIC TARGETS - REGIME EXAMPLES")
    print("="*70)
    
    manager = DynamicTargetManager()
    
    # Example scenario
    entry_price = 1000.0
    atr = 15.0  # 1.5% of price
    
    regimes = [
        (MarketRegime.TRENDING_CALM, "Trending + Calm (Best)"),
        (MarketRegime.TRENDING_VOLATILE, "Trending + Volatile"),
        (MarketRegime.CHOPPY_CALM, "Choppy + Calm"),
        (MarketRegime.CHOPPY_VOLATILE, "Choppy + Volatile (Avoid)"),
    ]
    
    confidences = [0.70, 0.60, 0.55]  # High, medium, low
    
    for regime, desc in regimes:
        print(f"\n{desc}")
        print("-" * 70)
        
        for conf in confidences:
            target, stop, meta = manager.compute_levels(
                entry_price=entry_price,
                atr=atr,
                side="LONG",
                regime=regime,
                confidence=conf,
            )
            
            if meta.get('skip_trade'):
                print(f"  Conf={conf:.0%}: SKIP TRADE")
                continue
            
            print(f"  Conf={conf:.0%}: "
                  f"Target=₹{target:.1f} (+{meta['target_distance_pct']:.2f}%), "
                  f"Stop=₹{stop:.1f} (-{meta['stop_distance_pct']:.2f}%), "
                  f"R/R={meta['risk_reward_ratio']:.2f}")
    
    # Example: Trailing stop
    print("\n" + "="*70)
    print("TRAILING STOP EXAMPLE")
    print("="*70)
    
    entry = 1000
    atr = 15
    
    for profit_pct in [0.3, 0.5, 1.0, 1.5, 2.0]:
        current = entry * (1 + profit_pct / 100)
        peak = current  # Assume at peak
        
        trail_stop, meta = manager.compute_trailing_stop(
            entry_price=entry,
            current_price=current,
            peak_price=peak,
            atr=atr,
            side="LONG",
            regime=MarketRegime.TRENDING_CALM,
        )
        
        if trail_stop:
            print(f"  Profit={profit_pct:.1f}%: Trail stop at ₹{trail_stop:.1f} "
                  f"(locks in {(trail_stop - entry) / entry * 100:.2f}%)")
        else:
            print(f"  Profit={profit_pct:.1f}%: Trail not activated "
                  f"(need {manager.config.trailing_activation_atr_mult:.1f}×ATR)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
