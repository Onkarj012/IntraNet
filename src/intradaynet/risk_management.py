"""
Risk Management for IntradayNet v3.0 - Phase 4

Complete risk management system:
4.1: Dynamic position sizing (ATR-based with regime adjustment)
4.2: Correlation-aware portfolio construction (uncorrelated selection)
4.3: Intraday exit logic (trailing stops, time exit, adverse momentum)
4.4: Daily circuit breakers (loss limits, consecutive loss limits)

Usage:
    from intradaynet.risk_management import RiskManager
    
    # Initialize
    risk_mgr = RiskManager(
        account_value=100000,
        max_account_risk_per_trade=0.005,  # 0.5%
        max_position_value=25000,  # ₹25K hard cap
    )
    
    # Compute position size for trade
    position_size = risk_mgr.compute_position_size(
        entry_price=1000,
        atr=15,
        stop_distance=12,
        regime_adjustments=regime_adj,
    )
    
    # Build correlation-aware portfolio
    portfolio = risk_mgr.build_portfolio(
        candidates=candidate_trades,
        max_positions=5,
        max_correlation=0.4,
    )
    
    # Check circuit breakers
    should_stop = risk_mgr.check_circuit_breakers(
        daily_pnl=-1500,
        consecutive_losses=3,
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
import logging

logger = logging.getLogger("intradaynet.risk")


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    account_value: float = 100000.0  # ₹1L default
    max_account_risk_per_trade: float = 0.005  # 0.5%
    max_position_value: float = 25000.0  # ₹25K hard cap
    min_position_value: float = 5000.0  # ₹5K minimum
    
    # ATR-based sizing
    use_atr_based_sizing: bool = True
    atr_multiplier_for_risk: float = 1.0  # Risk = ATR × multiplier
    
    # Position size formula:
    # size = (account_risk) / (stop_distance × stock_volatility_multiplier)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    max_positions: int = 5
    max_positions_per_sector: int = 2
    max_correlation: float = 0.4  # Max pairwise correlation
    
    # Correlation calculation
    correlation_lookback: int = 20  # Days


@dataclass
class ExitConfig:
    """Configuration for exit logic."""
    # Time-based exit
    use_time_exit: bool = True
    force_exit_time: time = time(14, 30)  # 2:30 PM
    
    # Trailing stop
    use_trailing_stop: bool = True
    trailing_activation_pct: float = 0.5  # Activate at +0.5% profit
    trailing_distance_pct: float = 0.3  # Trail at +0.3% below peak
    
    # Adverse momentum exit
    use_adverse_momentum_exit: bool = True
    vwap_exit_threshold: float = 0.5  # Exit if crosses VWAP by this amount
    
    # Gap protection
    max_gap_against_pct: float = 1.5  # Skip if gaps > 1.5% against position


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    # Account value for calculations
    account_value: float = 100000.0
    
    # Daily loss limit
    daily_loss_limit_pct: float = 0.015  # -1.5% of capital
    
    # Daily win limit (for tightening stops)
    daily_win_limit_pct: float = 0.03  # +3.0%
    tighten_stops_when_winning: bool = True
    
    # Consecutive loss limit
    max_consecutive_losses: int = 3
    pause_minutes_after_consecutive_losses: int = 30
    
    # Position halt
    halt_after_consecutive_stop_losses: bool = True


class DynamicPositionSizer:
    """
    4.1: Dynamic position sizing based on ATR and volatility.
    
    Formula: position_size = (account_risk) / (stop_distance × vol_mult)
    """
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
    
    def compute_position_size(
        self,
        entry_price: float,
        atr: float,
        stop_distance: Optional[float] = None,
        regime_adjustments: Optional[Dict] = None,
        stock_beta: float = 1.0,
    ) -> Tuple[float, Dict]:
        """
        Compute position size for a trade.
        
        Returns:
            (position_value, metadata)
        """
        if entry_price <= 0:
            return 0.0, {'error': 'invalid_price'}
        
        # Account risk per trade
        account_risk = self.config.account_value * self.config.max_account_risk_per_trade
        
        # Determine stop distance
        if stop_distance is None:
            # Use ATR-based stop
            stop_distance = atr * self.config.atr_multiplier_for_risk
        
        # Volatility multiplier (higher beta = smaller position)
        vol_multiplier = max(0.5, min(2.0, stock_beta))
        
        # Effective risk
        effective_risk = stop_distance * vol_multiplier
        
        if effective_risk <= 0:
            return 0.0, {'error': 'zero_risk'}
        
        # Base position size
        position_value = account_risk / (effective_risk / entry_price)
        
        # Apply regime adjustment
        if regime_adjustments:
            regime_mult = regime_adjustments.get('size_multiplier', 1.0)
            position_value *= regime_mult
        
        # Apply hard caps
        position_value = min(position_value, self.config.max_position_value)
        position_value = max(position_value, self.config.min_position_value)
        
        # Compute quantity (round to nearest lot if needed)
        quantity = int(position_value / entry_price)
        
        # Recalculate actual position value
        actual_position_value = quantity * entry_price
        
        metadata = {
            'account_risk': account_risk,
            'stop_distance': stop_distance,
            'stop_distance_pct': stop_distance / entry_price * 100,
            'vol_multiplier': vol_multiplier,
            'regime_multiplier': regime_adjustments.get('size_multiplier', 1.0) if regime_adjustments else 1.0,
            'raw_position_value': position_value,
            'capped_position_value': actual_position_value,
            'quantity': quantity,
            'risk_pct': (stop_distance * quantity) / self.config.account_value * 100,
        }
        
        return actual_position_value, metadata


class CorrelationAwarePortfolio:
    """
    4.2: Build portfolio with uncorrelated positions.
    
    Prevents "all 5 picks are banking stocks" scenario.
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
    
    def compute_pairwise_correlation(
        self,
        symbol1: str,
        symbol2: str,
        price_history: Dict[str, pd.DataFrame],
    ) -> float:
        """Compute correlation between two symbols."""
        if symbol1 not in price_history or symbol2 not in price_history:
            return 0.0
        
        df1 = price_history[symbol1]
        df2 = price_history[symbol2]
        
        # Get daily returns
        daily1 = df1.resample('D').last()['close'].pct_change().dropna()
        daily2 = df2.resample('D').last()['close'].pct_change().dropna()
        
        # Align
        aligned = pd.concat([daily1, daily2], axis=1).dropna()
        
        if len(aligned) < self.config.correlation_lookback:
            return 0.0
        
        # Use recent window
        recent = aligned.tail(self.config.correlation_lookback)
        
        try:
            corr = recent.iloc[:, 0].corr(recent.iloc[:, 1])
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def build_portfolio(
        self,
        candidates: List[Dict],
        price_history: Dict[str, pd.DataFrame],
        max_positions: Optional[int] = None,
    ) -> List[Dict]:
        """
        Build correlation-aware portfolio.
        
        Greedy selection:
        1. Pick highest-scoring candidate
        2. Pick next highest with correlation < max_correlation to all selected
        3. Repeat until max_positions
        
        Also enforces sector cap.
        """
        max_positions = max_positions or self.config.max_positions
        
        if not candidates:
            return []
        
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
        
        selected = []
        sector_counts = {}
        
        for candidate in sorted_candidates:
            if len(selected) >= max_positions:
                break
            
            symbol = candidate.get('symbol')
            sector = candidate.get('sector', 'UNKNOWN')
            
            # Check sector cap
            sector_counts[sector] = sector_counts.get(sector, 0)
            if sector_counts[sector] >= self.config.max_positions_per_sector:
                continue
            
            # Check correlation to all selected
            if selected:
                correlations = []
                for existing in selected:
                    corr = self.compute_pairwise_correlation(
                        symbol, existing.get('symbol'), price_history
                    )
                    correlations.append(abs(corr))
                
                max_corr = max(correlations) if correlations else 0.0
                
                if max_corr > self.config.max_correlation:
                    logger.info(f"Skipping {symbol} - correlation {max_corr:.2f} > threshold")
                    continue
            
            # Add to portfolio
            selected.append(candidate)
            sector_counts[sector] += 1
        
        logger.info(f"Portfolio built: {len(selected)} positions")
        for pos in selected:
            logger.info(f"  - {pos.get('symbol')} ({pos.get('sector')})")
        
        return selected


class IntradayExitManager:
    """
    4.3: Advanced exit logic for positions.
    """
    
    def __init__(self, config: ExitConfig):
        self.config = config
    
    def check_time_exit(
        self,
        entry_time: datetime,
        current_time: datetime,
        bars_in_position: int,
        max_bars: int = 60,
    ) -> Tuple[bool, str]:
        """
        Check if time-based exit should trigger.
        
        Returns (should_exit, reason)
        """
        if not self.config.use_time_exit:
            return False, ""
        
        # Check hard time limit (2:30 PM)
        if current_time.time() >= self.config.force_exit_time:
            return True, f"time_exit_{self.config.force_exit_time}"
        
        # Check max bars
        if bars_in_position >= max_bars:
            return True, f"max_bars_{max_bars}"
        
        return False, ""
    
    def check_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float,  # Highest (LONG) or lowest (SHORT) since entry
        side: str,
        atr: Optional[float] = None,
    ) -> Tuple[Optional[float], str]:
        """
        Check trailing stop activation.
        
        Returns (trailing_stop_price, reason) or (None, "") if not activated.
        """
        if not self.config.use_trailing_stop:
            return None, ""
        
        side = side.upper()
        
        # Calculate profit
        if side == "LONG":
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price * 100
        
        # Check activation threshold
        activation_pct = self.config.trailing_activation_pct
        if profit_pct < activation_pct:
            return None, ""  # Not activated yet
        
        # Calculate trailing stop level
        trail_distance_pct = self.config.trailing_distance_pct
        
        if atr:
            # ATR-based trailing
            trail_distance = atr / entry_price * 100
            trail_distance = max(trail_distance, trail_distance_pct)
        else:
            trail_distance = trail_distance_pct
        
        if side == "LONG":
            # Trail below peak
            trailing_stop = peak_price * (1 - trail_distance / 100)
        else:
            # Trail above trough (peak in this context)
            trailing_stop = peak_price * (1 + trail_distance / 100)
        
        return trailing_stop, f"trailing_{trail_distance:.2f}pct"
    
    def check_adverse_momentum_exit(
        self,
        position_side: str,
        entry_price: float,
        current_price: float,
        vwap: float,
        volume: float,
        avg_volume: float,
    ) -> Tuple[bool, str]:
        """
        Exit if stock reverses and crosses VWAP against position with rising volume.
        
        Don't wait for stop-loss - exit immediately on adverse momentum.
        """
        if not self.config.use_adverse_momentum_exit:
            return False, ""
        
        position_side = position_side.upper()
        
        # Check if price crossed VWAP against position
        if position_side == "LONG":
            # Long position - exit if price drops below VWAP
            if current_price < vwap:
                # Check distance
                distance_pct = (vwap - current_price) / entry_price * 100
                
                if distance_pct > self.config.vwap_exit_threshold:
                    # Check rising volume
                    if volume > avg_volume * 1.2:  # 20% above average
                        return True, "adverse_momentum_vwap_long"
        else:  # SHORT
            # Short position - exit if price rises above VWAP
            if current_price > vwap:
                distance_pct = (current_price - vwap) / entry_price * 100
                
                if distance_pct > self.config.vwap_exit_threshold:
                    if volume > avg_volume * 1.2:
                        return True, "adverse_momentum_vwap_short"
        
        return False, ""
    
    def check_gap_protection(
        self,
        entry_price: float,
        predicted_stop: float,
        opening_price: float,
        side: str,
    ) -> Tuple[bool, str]:
        """
        Skip trade if stock gaps against you by more than 1.5× stop distance.
        """
        side = side.upper()
        stop_distance = abs(entry_price - predicted_stop)
        
        # Calculate gap
        gap = abs(opening_price - entry_price)
        gap_pct = gap / entry_price * 100
        
        # Threshold is 1.5× stop distance
        threshold = stop_distance * 1.5
        
        if gap > threshold:
            return True, f"gap_protection_{gap_pct:.2f}pct"
        
        return False, ""


class CircuitBreakerSystem:
    """
    4.4: Daily circuit breakers to protect capital.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        
        # State tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_stop_losses = 0
        self.trading_halted = False
        self.halt_until = None
        self.daily_high_pnl = 0.0
    
    def reset_daily(self):
        """Reset daily tracking."""
        self.daily_pnl = 0.0
        self.daily_high_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_stop_losses = 0
        self.trading_halted = False
        self.halt_until = None
    
    def update_trade_result(
        self,
        pnl: float,
        exit_reason: str,
    ) -> Dict[str, Any]:
        """
        Update circuit breaker state with trade result.
        
        Returns dict with status and any actions to take.
        """
        result = {
            'halt_trading': False,
            'tighten_stops': False,
            'pause_minutes': 0,
            'reason': '',
        }
        
        # Update P&L
        self.daily_pnl += pnl
        
        if pnl > 0:
            # Winning trade
            self.consecutive_losses = 0
            self.consecutive_stop_losses = 0
            
            # Update daily high
            if self.daily_pnl > self.daily_high_pnl:
                self.daily_high_pnl = self.daily_pnl
        
        else:
            # Losing trade
            self.consecutive_losses += 1
            
            if 'stop_loss' in exit_reason:
                self.consecutive_stop_losses += 1
            
            # Check loss limit
            loss_limit = self.config.account_value * self.config.daily_loss_limit_pct
            if abs(self.daily_pnl) >= loss_limit and self.daily_pnl < 0:
                self.trading_halted = True
                result['halt_trading'] = True
                result['reason'] = f'daily_loss_limit_{abs(self.daily_pnl):.0f}'
            
            # Check consecutive loss limit
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                result['pause_minutes'] = self.config.pause_minutes_after_consecutive_losses
                result['reason'] = f'consecutive_losses_{self.consecutive_losses}'
            
            # Check consecutive stop losses
            if (self.consecutive_stop_losses >= 3 and 
                self.config.halt_after_consecutive_stop_losses):
                self.trading_halted = True
                result['halt_trading'] = True
                result['reason'] = f'consecutive_stops_{self.consecutive_stop_losses}'
        
        # Check win limit for tightening stops
        if self.config.tighten_stops_when_winning:
            win_limit = self.config.account_value * self.config.daily_win_limit_pct
            if self.daily_pnl >= win_limit:
                result['tighten_stops'] = True
                result['reason'] = f'win_limit_{self.daily_pnl:.0f}'
        
        return result
    
    def check_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'trading_halted': self.trading_halted,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.config.account_value * 100,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_stop_losses': self.consecutive_stop_losses,
            'halt_until': self.halt_until,
        }


class RiskManager:
    """
    Unified risk management combining all Phase 4 components.
    """
    
    def __init__(
        self,
        account_value: float = 100000.0,
        position_config: Optional[PositionSizingConfig] = None,
        portfolio_config: Optional[PortfolioConfig] = None,
        exit_config: Optional[ExitConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.account_value = account_value
        
        # Initialize subsystems
        self.position_sizer = DynamicPositionSizer(
            position_config or PositionSizingConfig(account_value=account_value)
        )
        
        self.portfolio_builder = CorrelationAwarePortfolio(
            portfolio_config or PortfolioConfig()
        )
        
        self.exit_manager = IntradayExitManager(
            exit_config or ExitConfig()
        )
        
        self.circuit_breaker = CircuitBreakerSystem(
            circuit_config or CircuitBreakerConfig(account_value=account_value)
        )
        
        # Track active positions
        self.active_positions: Dict[str, Dict] = {}
    
    def compute_position_size(
        self,
        entry_price: float,
        atr: float,
        **kwargs
    ) -> Tuple[float, Dict]:
        """Compute position size."""
        return self.position_sizer.compute_position_size(entry_price, atr, **kwargs)
    
    def build_portfolio(self, candidates: List[Dict], **kwargs) -> List[Dict]:
        """Build correlation-aware portfolio."""
        return self.portfolio_builder.build_portfolio(candidates, **kwargs)
    
    def check_exits(
        self,
        symbol: str,
        position: Dict,
        current_bar: Dict,
    ) -> Tuple[bool, Optional[float], str]:
        """
        Check all exit conditions for a position.
        
        Returns: (should_exit, exit_price, reason)
        """
        entry_time = position['entry_time']
        current_time = current_bar['time']
        bars_in_position = current_bar['bar_number'] - position['entry_bar']
        
        # Time exit
        should_exit, reason = self.exit_manager.check_time_exit(
            entry_time, current_time, bars_in_position
        )
        if should_exit:
            return True, current_bar['close'], reason
        
        # Trailing stop
        trailing_stop, reason = self.exit_manager.check_trailing_stop(
            position['entry_price'],
            current_bar['close'],
            position.get('peak_price', current_bar['close']),
            position['side'],
            position.get('atr'),
        )
        if trailing_stop and current_bar['close'] <= trailing_stop:
            return True, trailing_stop, reason
        
        # Adverse momentum
        should_exit, reason = self.exit_manager.check_adverse_momentum_exit(
            position['side'],
            position['entry_price'],
            current_bar['close'],
            current_bar.get('vwap', current_bar['close']),
            current_bar['volume'],
            current_bar.get('avg_volume', current_bar['volume']),
        )
        if should_exit:
            return True, current_bar['close'], reason
        
        return False, None, ""
    
    def update_trade(self, pnl: float, exit_reason: str) -> Dict:
        """Update circuit breaker with trade result."""
        return self.circuit_breaker.update_trade_result(pnl, exit_reason)
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        status = self.circuit_breaker.check_status()
        return not status['trading_halted']


def main():
    """Demo risk management components."""
    print("\n" + "="*70)
    print("Risk Management v3.0 Demo")
    print("="*70)
    
    # 1. Position Sizing
    print("\n1. DYNAMIC POSITION SIZING")
    print("-"*70)
    
    sizer = DynamicPositionSizer(PositionSizingConfig(account_value=100000))
    
    test_cases = [
        (1000, 15, 1.0, "Normal volatility"),
        (1000, 30, 1.5, "High volatility"),
        (1000, 8, 0.5, "Low volatility (regime: choppy)"),
    ]
    
    for price, atr, beta, desc in test_cases:
        size, meta = sizer.compute_position_size(price, atr, stock_beta=beta)
        print(f"\n{desc}:")
        print(f"  Entry: ₹{price:.0f}, ATR: {atr:.1f}, Beta: {beta:.1f}")
        print(f"  Position size: ₹{size:,.0f}")
        print(f"  Risk: {meta['risk_pct']:.2f}% of account")
    
    # 2. Circuit Breakers
    print("\n\n2. CIRCUIT BREAKERS")
    print("-"*70)
    
    cb = CircuitBreakerSystem(CircuitBreakerConfig(account_value=100000))
    
    # Simulate trading day
    trades = [
        (-500, "stop_loss"),
        (-300, "stop_loss"),
        (-400, "stop_loss"),  # 3 consecutive stops
        (200, "take_profit"),  # Won't reach here
    ]
    
    for i, (pnl, reason) in enumerate(trades, 1):
        result = cb.update_trade_result(pnl, reason)
        status = cb.check_status()
        
        print(f"\nTrade {i}: P&L={pnl:+.0f}, Reason={reason}")
        print(f"  Daily P&L: ₹{status['daily_pnl']:+.0f} ({status['daily_pnl_pct']:.2f}%)")
        print(f"  Consecutive losses: {status['consecutive_losses']}")
        print(f"  Trading halted: {status['trading_halted']}")
        
        if result['halt_trading']:
            print(f"  ⚠️ TRADING HALTED: {result['reason']}")
            break
    
    print("\n" + "="*70)
    print("Risk Management Demo Complete")
    print("="*70)


if __name__ == "__main__":
    main()
