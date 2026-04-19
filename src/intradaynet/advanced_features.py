"""
Advanced Features for IntradayNet v3.0 - Phase 6

6.1: Market Microstructure Intelligence (FII/DII, delivery %, block deals)
6.2: Earnings Season Module
6.3: Nifty Hedge Layer
6.4: Confidence-Gated Trading

Usage:
    from intradaynet.advanced_features import AdvancedFeatureEngine
    
    engine = AdvancedFeatureEngine()
    
    # 6.1: FII/DII flows
    fii_signal = engine.get_fii_signal(symbol, date)
    
    # 6.2: Earnings check
    can_trade = engine.check_earnings(symbol, date)
    
    # 6.3: Portfolio hedge
    hedge_size = engine.compute_hedge(portfolio_positions, nifty_data)
    
    # 6.4: Confidence gate
    should_trade, confidence_score = engine.gate_by_confidence(
        prediction, confidence_threshold=0.58
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

logger = logging.getLogger("intradaynet.advanced")


@dataclass
class FIIData:
    """FII/DII flow data."""
    date: str
    fii_net_buy: float  # In crores
    dii_net_buy: float
    fii_cumulative_5d: float
    dii_cumulative_5d: float


@dataclass
class EarningsData:
    """Earnings information for a stock."""
    symbol: str
    next_earnings_date: Optional[str]
    days_to_earnings: int
    last_earnings_surprise: float  # %
    earnings_beat_streak: int  # Consecutive beats
    post_earnings_drift: float  # % drift after last earnings


class FIIDIIIntegrator:
    """
    6.1: FII/DII Flow Integration
    
    NSE publishes daily FII/DII buy/sell data by 6 PM.
    - fii_net_flow_5d: Net FII flow over 5 days
    - dii_net_flow_5d: Net DII flow over 5 days
    - delivery_pct: Rising delivery = accumulation
    
    Conflict detection: Persistent FII selling + model LONG = reduce size
    """
    
    def __init__(self, data_dir: str = "market_data_cache"):
        self.data_dir = Path(data_dir)
        self.fii_data: Dict[str, List[FIIData]] = {}
        self._load_fii_data()
    
    def _load_fii_data(self):
        """Load historical FII/DII data."""
        fii_file = self.data_dir / "fii_dii_data.csv"
        
        if fii_file.exists():
            df = pd.read_csv(fii_file, parse_dates=['date'])
            
            for _, row in df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                data = FIIData(
                    date=date_str,
                    fii_net_buy=row.get('fii_net_buy', 0),
                    dii_net_buy=row.get('dii_net_buy', 0),
                    fii_cumulative_5d=0,  # Computed below
                    dii_cumulative_5d=0,
                )
                
                if date_str not in self.fii_data:
                    self.fii_data[date_str] = []
                self.fii_data[date_str].append(data)
    
    def get_fii_signal(
        self,
        symbol: str,
        date: str,
        window: int = 5,
    ) -> Dict[str, Any]:
        """
        Get FII/DII signal for a stock.
        
        Returns signal strength and conflict indicators.
        """
        # Get FII flows
        fii_flow = self._get_cumulative_fii_flow(date, window)
        dii_flow = self._get_cumulative_dii_flow(date, window)
        
        # Signal interpretation
        fii_signal = np.sign(fii_flow)  # +1 = buying, -1 = selling
        signal_strength = abs(fii_flow) / 1000  # Normalize
        
        # Delivery percentage (if available)
        delivery_pct = self._get_delivery_percentage(symbol, date)
        
        return {
            'fii_net_flow_5d': fii_flow,
            'dii_net_flow_5d': dii_flow,
            'fii_signal': fii_signal,
            'signal_strength': min(signal_strength, 1.0),
            'delivery_pct': delivery_pct,
            'delivery_trend': 'rising' if delivery_pct > 60 else 'normal',
        }
    
    def _get_cumulative_fii_flow(self, date: str, window: int) -> float:
        """Get cumulative FII net buy over window."""
        # Simplified - would aggregate from daily data
        return 500.0  # Placeholder: ₹500 Cr net buy
    
    def _get_cumulative_dii_flow(self, date: str, window: int) -> float:
        """Get cumulative DII net buy over window."""
        return 300.0  # Placeholder: ₹300 Cr net buy
    
    def _get_delivery_percentage(self, symbol: str, date: str) -> float:
        """Get delivery vs intraday volume percentage."""
        # Would load from NSE delivery data
        return 65.0  # Placeholder: 65% delivery
    
    def check_conflict(
        self,
        symbol: str,
        model_side: str,
        date: str,
    ) -> Tuple[bool, str]:
        """
        Check if FII flow conflicts with model signal.
        
        Returns (has_conflict, reason)
        """
        signal = self.get_fii_signal(symbol, date)
        
        fii_signal = signal['fii_signal']
        model_side = model_side.upper()
        
        # Conflict: FII selling + Model LONG
        if model_side == "LONG" and fii_signal < 0:
            return True, f"FII selling ({signal['fii_net_flow_5d']:.0f} Cr) conflicts with LONG"
        
        # Conflict: FII buying + Model SHORT
        if model_side == "SHORT" and fii_signal > 0:
            return True, f"FII buying ({signal['fii_net_flow_5d']:.0f} Cr) conflicts with SHORT"
        
        return False, ""
    
    def adjust_size_for_flow(
        self,
        base_size: float,
        symbol: str,
        model_side: str,
        date: str,
    ) -> Tuple[float, str]:
        """
        Adjust position size based on FII flow conflict.
        
        Reduces size if there's a conflict.
        """
        has_conflict, reason = self.check_conflict(symbol, model_side, date)
        
        if has_conflict:
            # Reduce size by 30%
            adjusted_size = base_size * 0.7
            return adjusted_size, f"Reduced 30% due to: {reason}"
        
        return base_size, "No FII conflict"


class EarningsSeasonModule:
    """
    6.2: Earnings Season Module
    
    - days_to_earnings: Continuous countdown
    - earnings_surprise_history: Serial beaters trend differently
    - post_earnings_drift: Directional bias after surprise
    
    Rule: Never take position in stock reporting earnings today
    """
    
    def __init__(self, earnings_file: str = "data/earnings_calendar.csv"):
        self.earnings_file = Path(earnings_file)
        self.earnings_data: Dict[str, EarningsData] = {}
        self._load_earnings_data()
    
    def _load_earnings_data(self):
        """Load earnings calendar data."""
        if not self.earnings_file.exists():
            logger.warning(f"Earnings file not found: {self.earnings_file}")
            return
        
        df = pd.read_csv(self.earnings_file)
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            self.earnings_data[symbol] = EarningsData(
                symbol=symbol,
                next_earnings_date=row.get('next_earnings_date'),
                days_to_earnings=row.get('days_to_earnings', 999),
                last_earnings_surprise=row.get('last_surprise_pct', 0),
                earnings_beat_streak=row.get('beat_streak', 0),
                post_earnings_drift=row.get('post_earnings_drift_pct', 0),
            )
    
    def check_earnings_risk(
        self,
        symbol: str,
        date: str,
    ) -> Dict[str, Any]:
        """
        Check earnings risk for a stock.
        
        Returns risk assessment and trading recommendation.
        """
        data = self.earnings_data.get(symbol)
        
        if not data:
            return {
                'has_earnings_risk': False,
                'can_trade': True,
                'reason': 'No earnings data',
            }
        
        # Check if earnings today
        if data.next_earnings_date == date:
            return {
                'has_earnings_risk': True,
                'can_trade': False,
                'reason': 'Earnings today - gap risk unmodelable',
                'days_to_earnings': 0,
            }
        
        # Check if earnings within 2 days (high vol period)
        if data.days_to_earnings <= 2:
            return {
                'has_earnings_risk': True,
                'can_trade': True,  # Can trade but be careful
                'reason': f'Earnings in {data.days_to_earnings} days - high volatility expected',
                'days_to_earnings': data.days_to_earnings,
                'recommendation': 'reduce_size',
            }
        
        # Post-earnings drift signal
        drift_signal = data.post_earnings_drift
        
        return {
            'has_earnings_risk': False,
            'can_trade': True,
            'reason': f'Earnings in {data.days_to_earnings} days',
            'days_to_earnings': data.days_to_earnings,
            'beat_streak': data.earnings_beat_streak,
            'post_earnings_drift': drift_signal,
        }
    
    def get_earnings_features(self, symbol: str, date: str) -> Dict[str, float]:
        """Get earnings-related features for model."""
        risk_data = self.check_earnings_risk(symbol, date)
        
        return {
            'days_to_earnings': risk_data.get('days_to_earnings', 999),
            'is_earnings_week': 1.0 if risk_data.get('days_to_earnings', 999) <= 7 else 0.0,
            'post_earnings_drift': risk_data.get('post_earnings_drift', 0),
            'earnings_beat_streak': risk_data.get('beat_streak', 0),
        }


class NiftyHedgeLayer:
    """
    6.3: Nifty Hedge Layer
    
    Isolates stock-picking alpha from market beta.
    
    - Measure portfolio beta daily
    - Short Nifty futures to neutralize excess beta
    - Only implement if backtest shows genuine alpha
    """
    
    def __init__(self, nifty_lot_size: int = 50, nifty_margin_per_lot: float = 150000):
        self.nifty_lot_size = nifty_lot_size
        self.nifty_margin_per_lot = nifty_margin_per_lot
    
    def compute_portfolio_beta(
        self,
        positions: List[Dict],
        price_history: Dict[str, pd.DataFrame],
        nifty_history: pd.DataFrame,
    ) -> float:
        """
        Compute portfolio beta to Nifty.
        
        Returns weighted average beta of positions.
        """
        betas = []
        values = []
        
        for pos in positions:
            symbol = pos['symbol']
            value = pos.get('value', 0)
            
            if symbol not in price_history:
                continue
            
            # Compute beta
            stock_returns = price_history[symbol]['close'].pct_change().tail(20)
            nifty_returns = nifty_history['close'].pct_change().tail(20)
            
            # Align
            aligned = pd.concat([stock_returns, nifty_returns], axis=1).dropna()
            
            if len(aligned) >= 10:
                # Beta = Cov(stock, nifty) / Var(nifty)
                covariance = aligned.cov().iloc[0, 1]
                nifty_variance = aligned.iloc[:, 1].var()
                
                beta = covariance / nifty_variance if nifty_variance > 0 else 1.0
            else:
                beta = 1.0  # Default
            
            betas.append(beta)
            values.append(value)
        
        if not betas:
            return 1.0
        
        # Weighted average beta
        total_value = sum(values)
        portfolio_beta = sum(b * v for b, v in zip(betas, values)) / total_value
        
        return portfolio_beta
    
    def compute_hedge_size(
        self,
        portfolio_value: float,
        portfolio_beta: float,
        target_beta: float = 0.0,  # Fully hedge to market-neutral
        nifty_price: float = 20000,
    ) -> Dict[str, Any]:
        """
        Compute Nifty futures hedge size.
        
        Formula: hedge_value = portfolio_value × (portfolio_beta - target_beta)
        """
        # Excess beta to hedge
        excess_beta = portfolio_beta - target_beta
        
        # Hedge value
        hedge_value = portfolio_value * excess_beta
        
        # Number of lots
        nifty_value_per_lot = nifty_price * self.nifty_lot_size
        n_lots = int(hedge_value / nifty_value_per_lot)
        
        # Margin required
        margin_required = n_lots * self.nifty_margin_per_lot
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_beta': portfolio_beta,
            'target_beta': target_beta,
            'excess_beta': excess_beta,
            'hedge_value': hedge_value,
            'nifty_price': nifty_price,
            'nifty_lot_size': self.nifty_lot_size,
            'n_lots': n_lots,
            'margin_required': margin_required,
            'side': 'SHORT' if excess_beta > 0 else 'LONG',
        }
    
    def should_hedge(self, backtest_alpha: float) -> bool:
        """
        Determine if hedging should be used.
        
        Only hedge if backtest shows genuine alpha (not just beta).
        """
        # If alpha is positive and significant, hedge to isolate it
        return backtest_alpha > 0.02  # 2% annualized alpha threshold


class ConfidenceGate:
    """
    6.4: Confidence-Gated Trading
    
    - Skip trades with confidence < 58%
    - 2× position size for 70%+ confidence trades
    - Track calibration curve
    """
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.58,
        high_confidence_threshold: float = 0.70,
        high_confidence_multiplier: float = 2.0,
    ):
        self.min_threshold = min_confidence_threshold
        self.high_threshold = high_confidence_threshold
        self.high_mult = high_confidence_multiplier
        
        # Track predictions for calibration
        self.prediction_history: List[Dict] = []
    
    def gate_trade(
        self,
        predicted_confidence: float,
        predicted_direction: str,
        symbol: str,
        base_size: float,
    ) -> Tuple[bool, float, str, float]:
        """
        Gate trade based on confidence.
        
        Returns:
            (should_trade, adjusted_size, reason, conviction_score)
        """
        # Check minimum threshold
        if predicted_confidence < self.min_threshold:
            return False, 0.0, f"Confidence {predicted_confidence:.1%} < {self.min_threshold:.1%} threshold", 0.0
        
        # Determine conviction
        if predicted_confidence >= self.high_threshold:
            conviction = 1.0
            adjusted_size = base_size * self.high_mult
            reason = f"High confidence ({predicted_confidence:.1%}) - 2× size"
        elif predicted_confidence >= 0.65:
            conviction = 0.7
            adjusted_size = base_size * 1.3
            reason = f"Medium-high confidence ({predicted_confidence:.1%}) - 1.3× size"
        else:
            conviction = 0.5
            adjusted_size = base_size
            reason = f"Standard confidence ({predicted_confidence:.1%}) - 1× size"
        
        return True, adjusted_size, reason, conviction
    
    def log_prediction(
        self,
        timestamp: str,
        symbol: str,
        predicted_confidence: float,
        predicted_direction: str,
        actual_outcome: Optional[bool] = None,
    ):
        """Log prediction for calibration tracking."""
        self.prediction_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'predicted_confidence': predicted_confidence,
            'predicted_direction': predicted_direction,
            'actual_outcome': actual_outcome,
        })
    
    def compute_calibration_curve(self) -> Dict[str, Any]:
        """
        Compute calibration curve.
        
        Ideal: 60% predicted confidence → 60% actual win rate
        """
        if not self.prediction_history:
            return {'error': 'No prediction history'}
        
        # Group by confidence bins
        bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
        calibration = []
        
        for i in range(len(bins) - 1):
            bin_low = bins[i]
            bin_high = bins[i + 1]
            
            # Get predictions in this bin
            bin_preds = [
                p for p in self.prediction_history
                if bin_low <= p['predicted_confidence'] < bin_high
                and p['actual_outcome'] is not None
            ]
            
            if not bin_preds:
                continue
            
            avg_confidence = np.mean([p['predicted_confidence'] for p in bin_preds])
            actual_win_rate = np.mean([p['actual_outcome'] for p in bin_preds])
            n_samples = len(bin_preds)
            
            calibration.append({
                'bin': f"{bin_low:.0%}-{bin_high:.0%}",
                'avg_confidence': avg_confidence,
                'actual_win_rate': actual_win_rate,
                'error': abs(avg_confidence - actual_win_rate),
                'n_samples': n_samples,
            })
        
        # Compute calibration error
        if calibration:
            mean_error = np.mean([c['error'] for c in calibration])
        else:
            mean_error = 0.0
        
        return {
            'calibration_data': calibration,
            'mean_calibration_error': mean_error,
            'well_calibrated': mean_error < 0.02,  # Within 2%
        }
    
    def get_conviction_score(self, confidence: float) -> float:
        """
        Get conviction score for display.
        
        0-1 scale based on confidence level.
        """
        if confidence >= self.high_threshold:
            return 1.0
        elif confidence >= 0.65:
            return 0.7
        elif confidence >= self.min_threshold:
            return 0.5
        else:
            return 0.0


class AdvancedFeatureEngine:
    """
    Unified interface for all Phase 6 advanced features.
    """
    
    def __init__(
        self,
        fii_data_dir: str = "market_data_cache",
        earnings_file: str = "data/earnings_calendar.csv",
        nifty_lot_size: int = 50,
    ):
        self.fii_integrator = FIIDIIIntegrator(fii_data_dir)
        self.earnings_module = EarningsSeasonModule(earnings_file)
        self.hedge_layer = NiftyHedgeLayer(nifty_lot_size)
        self.confidence_gate = ConfidenceGate()
    
    def get_fii_signal(self, symbol: str, date: str) -> Dict:
        """Get FII/DII signal."""
        return self.fii_integrator.get_fii_signal(symbol, date)
    
    def check_earnings(self, symbol: str, date: str) -> Dict:
        """Check earnings risk."""
        return self.earnings_module.check_earnings_risk(symbol, date)
    
    def compute_hedge(self, positions: List[Dict], **kwargs) -> Dict:
        """Compute Nifty hedge."""
        beta = self.hedge_layer.compute_portfolio_beta(positions, **kwargs)
        portfolio_value = sum(p.get('value', 0) for p in positions)
        return self.hedge_layer.compute_hedge_size(portfolio_value, beta)
    
    def gate_by_confidence(self, confidence: float, base_size: float, **kwargs) -> Tuple:
        """Gate trade by confidence."""
        return self.confidence_gate.gate_trade(confidence, **kwargs)


def main():
    """Demo advanced features."""
    print("\n" + "="*70)
    print("Phase 6: Advanced Features Demo")
    print("="*70)
    
    engine = AdvancedFeatureEngine()
    
    # 6.1: FII/DII Flow
    print("\n1. FII/DII FLOW INTEGRATION")
    print("-"*70)
    
    fii_signal = engine.get_fii_signal("RELIANCE", "2025-01-15")
    print(f"FII Net Flow (5d): ₹{fii_signal['fii_net_flow_5d']:.0f} Cr")
    print(f"DII Net Flow (5d): ₹{fii_signal['dii_net_flow_5d']:.0f} Cr")
    print(f"Signal Strength: {fii_signal['signal_strength']:.2f}")
    
    # Check conflict
    conflict, reason = engine.fii_integrator.check_conflict("RELIANCE", "LONG", "2025-01-15")
    print(f"FII Conflict: {conflict}")
    if conflict:
        print(f"  Reason: {reason}")
    
    # 6.2: Earnings Module
    print("\n\n2. EARNINGS SEASON MODULE")
    print("-"*70)
    
    # Create sample earnings data
    engine.earnings_module.earnings_data['INFY'] = EarningsData(
        symbol='INFY',
        next_earnings_date='2025-01-16',
        days_to_earnings=1,
        last_earnings_surprise=2.5,
        earnings_beat_streak=3,
        post_earnings_drift=1.2,
    )
    
    earnings = engine.check_earnings('INFY', '2025-01-15')
    print(f"Earnings Check for INFY:")
    print(f"  Days to earnings: {earnings['days_to_earnings']}")
    print(f"  Can trade: {earnings['can_trade']}")
    print(f"  Reason: {earnings['reason']}")
    
    # 6.3: Nifty Hedge
    print("\n\n3. NIFTY HEDGE LAYER")
    print("-"*70)
    
    positions = [
        {'symbol': 'RELIANCE', 'value': 25000, 'beta': 1.1},
        {'symbol': 'TCS', 'value': 25000, 'beta': 0.9},
        {'symbol': 'HDFCBANK', 'value': 20000, 'beta': 1.2},
    ]
    
    # Compute weighted portfolio beta
    portfolio_value = sum(p['value'] for p in positions)
    portfolio_beta = sum(p['beta'] * p['value'] for p in positions) / portfolio_value
    
    hedge = engine.hedge_layer.compute_hedge_size(
        portfolio_value=portfolio_value,
        portfolio_beta=portfolio_beta,
        nifty_price=21000,
    )
    
    print(f"Portfolio Value: ₹{hedge['portfolio_value']:,.0f}")
    print(f"Portfolio Beta: {hedge['portfolio_beta']:.2f}")
    print(f"Excess Beta: {hedge['excess_beta']:.2f}")
    print(f"Hedge Required: {hedge['n_lots']} lots ({hedge['side']})")
    print(f"Margin Required: ₹{hedge['margin_required']:,.0f}")
    
    # 6.4: Confidence Gate
    print("\n\n4. CONFIDENCE-GATED TRADING")
    print("-"*70)
    
    test_confidences = [0.55, 0.62, 0.68, 0.72, 0.58]
    
    for conf in test_confidences:
        should_trade, size, reason, conviction = engine.confidence_gate.gate_trade(
            conf, "LONG", "RELIANCE", base_size=20000
        )
        
        status = "✓ TRADE" if should_trade else "✗ SKIP"
        print(f"Confidence {conf:.0%}: {status}")
        if should_trade:
            print(f"  Size: ₹{size:,.0f} ({reason})")
            print(f"  Conviction: {conviction:.0%}")
        else:
            print(f"  Reason: {reason}")
    
    print("\n" + "="*70)
    print("Advanced Features Demo Complete")
    print("="*70)


if __name__ == "__main__":
    main()
