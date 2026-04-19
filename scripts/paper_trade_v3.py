"""
Paper Trading Simulation for IntradayNet v3.0

Simulates 20 trading days to validate model performance before going live.
Tracks execution quality, fill rates, and compares to backtest expectations.

Usage:
    python scripts/paper_trade_v3.py --days 20 --start 2025-01-01
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("paper_trading")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.models.specialized import SpecializedModelSuite
from intradaynet.regime_v3 import RegimeClassifierV3, MarketRegime
from intradaynet.dynamic_targets import DynamicTargetManager
from intradaynet.features.v3_features import EnhancedFeatureEngineer
from intradaynet.risk_management import (
    RiskManager, PositionSizingConfig, PortfolioConfig,
    ExitConfig, CircuitBreakerConfig
)
from intradaynet.execution import PaperTradeLogger


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    date: str
    symbol: str
    side: str
    predicted_entry: float
    predicted_target: float
    predicted_stop: float
    confidence: float
    regime: str
    
    # Actual execution
    actual_open: Optional[float] = None
    actual_high: Optional[float] = None
    actual_low: Optional[float] = None
    actual_close: Optional[float] = None
    
    # Execution quality
    entry_slippage_pct: Optional[float] = None
    target_hit: bool = False
    stop_hit: bool = False
    
    # Outcome
    gross_return_pct: Optional[float] = None
    net_return_pct: Optional[float] = None
    outcome: Optional[str] = None


class PaperTradingSimulator:
    """
    20-day paper trading simulation.
    
    Gates live trading based on:
    - Win rate within 5% of backtest
    - Fill rate >= 80%
    - 20+ trading days
    """
    
    def __init__(
        self,
        model_dir: str = "models/v3_production",
        data_dir: str = "nifty500",
        output_dir: str = "paper_trading_results",
        account_value: float = 100000,
        min_confidence: float = 0.55,  # Slightly lower for more trades
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.account_value = account_value
        self.min_confidence = min_confidence
        
        # Load models
        self.suite = SpecializedModelSuite()
        self.suite.load(model_dir)
        
        # Initialize components
        self.regime_classifier = RegimeClassifierV3()
        self.target_manager = DynamicTargetManager()
        self.feature_engineer = EnhancedFeatureEngineer()
        self.risk_manager = RiskManager(
            account_value=account_value,
            position_config=PositionSizingConfig(account_value=account_value),
        )
        
        # Track trades
        self.trades: List[PaperTrade] = []
        self.daily_pnl: List[float] = []
        
        logger.info("Paper Trading Simulator initialized")
        logger.info(f"  Account: ₹{account_value:,.0f}")
        logger.info(f"  Min confidence: {min_confidence:.0%}")
    
    def simulate_trading_day(
        self,
        date: str,
        symbols: List[str],
        simulated_vix: float = 15.0,
    ) -> Dict:
        """
        Simulate one trading day.
        
        Returns daily statistics.
        """
        logger.info(f"\nSimulating {date}...")
        
        # Detect regime
        regime, _, adj = self.regime_classifier.classify(
            vix_level=simulated_vix,
            vix_change_pct=0,
        )
        
        if not adj.allow_trading:
            logger.info(f"  Skipping - {regime.value}")
            return {'trades': 0, 'pnl': 0}
        
        day_trades = 0
        day_pnl = 0.0
        
        # Process each symbol
        for symbol in symbols[:10]:  # Max 10 per day
            try:
                # Load data
                csv_path = self.data_dir / f"{symbol}_minute.csv"
                if not csv_path.exists():
                    continue
                
                df = pd.read_csv(csv_path, parse_dates=['date'])
                df = df.set_index('date')
                df = df[df.index.date == pd.Timestamp(date).date()]
                
                if len(df) < 200:
                    continue
                
                # Compute features at 11:00 AM (midday)
                midday_time = pd.Timestamp(f"{date} 11:00:00")
                midday_mask = df.index >= midday_time
                if not midday_mask.any():
                    continue
                
                midday_idx = midday_mask.idxmax()
                feat_window = self.feature_engineer.compute_all_features(
                    df.loc[:midday_idx], symbol
                )
                
                if len(feat_window) < 120:
                    continue
                
                # Feature vector
                feat_vector = feat_window.mean().values.reshape(1, -1)
                
                # Predict
                preds = self.suite.predict(feat_vector)
                
                direction_prob = preds['direction_prob'][0]
                magnitude = preds['magnitude_estimate'][0]
                confidence = preds['confidence_score'][0]
                
                # Trading decision (relaxed threshold for paper trading)
                if confidence < self.min_confidence:
                    continue
                
                side = "LONG" if direction_prob > 0.5 else "SHORT"
                
                # Entry price (use actual open or close at midday)
                entry_price = df.loc[midday_idx, 'close']
                
                # Compute dynamic targets
                atr = (df['high'] - df['low']).tail(20).mean()
                
                target, stop, target_meta = self.target_manager.compute_levels(
                    entry_price=entry_price,
                    atr=atr,
                    side=side,
                    regime=regime,
                    confidence=confidence,
                )
                
                if target_meta.get('skip_trade'):
                    continue
                
                # Get afternoon data (after midday)
                afternoon_df = df.loc[midday_idx:]
                if len(afternoon_df) < 30:
                    continue
                
                # Simulate execution
                # Predicted entry vs actual first afternoon price (simulating slippage)
                actual_entry = afternoon_df['open'].iloc[0] if len(afternoon_df) > 0 else entry_price
                slippage = (actual_entry - entry_price) / entry_price * 100
                
                # Track outcome
                if side == "LONG":
                    target_hit = afternoon_df['high'].max() >= target
                    stop_hit = afternoon_df['low'].min() <= stop
                    
                    if target_hit:
                        exit_price = target
                        outcome = "target_hit"
                        gross_return = (target - actual_entry) / actual_entry
                    elif stop_hit:
                        exit_price = stop
                        outcome = "stop_hit"
                        gross_return = (stop - actual_entry) / actual_entry
                    else:
                        exit_price = afternoon_df['close'].iloc[-1]
                        outcome = "time_exit"
                        gross_return = (exit_price - actual_entry) / actual_entry
                else:  # SHORT
                    target_hit = afternoon_df['low'].min() <= target
                    stop_hit = afternoon_df['high'].max() >= stop
                    
                    if target_hit:
                        exit_price = target
                        outcome = "target_hit"
                        gross_return = (actual_entry - target) / actual_entry
                    elif stop_hit:
                        exit_price = stop
                        outcome = "stop_hit"
                        gross_return = (actual_entry - stop) / actual_entry
                    else:
                        exit_price = afternoon_df['close'].iloc[-1]
                        outcome = "time_exit"
                        gross_return = (actual_entry - exit_price) / actual_entry
                
                # Net return (with costs)
                costs = 0.001  # 0.1% transaction costs
                net_return = gross_return - costs
                
                # Record trade
                trade = PaperTrade(
                    date=date,
                    symbol=symbol,
                    side=side,
                    predicted_entry=entry_price,
                    predicted_target=target,
                    predicted_stop=stop,
                    confidence=confidence,
                    regime=regime.value,
                    actual_open=afternoon_df['open'].iloc[0] if len(afternoon_df) > 0 else None,
                    actual_high=afternoon_df['high'].max(),
                    actual_low=afternoon_df['low'].min(),
                    actual_close=afternoon_df['close'].iloc[-1],
                    entry_slippage_pct=slippage,
                    target_hit=target_hit,
                    stop_hit=stop_hit,
                    gross_return_pct=gross_return * 100,
                    net_return_pct=net_return * 100,
                    outcome=outcome,
                )
                
                self.trades.append(trade)
                day_pnl += net_return * 100  # Store as percentage
                day_trades += 1
                
                logger.info(f"  {symbol}: {side} {outcome} | Return: {net_return*100:.2f}%")
                
            except Exception as e:
                logger.debug(f"Error with {symbol}: {e}")
                continue
        
        self.daily_pnl.append(day_pnl)
        
        return {
            'trades': day_trades,
            'pnl': day_pnl,
            'regime': regime.value,
        }
    
    def run_simulation(
        self,
        start_date: str = "2025-01-01",
        n_days: int = 20,
        symbols: List[str] = None,
    ):
        """Run full paper trading simulation."""
        logger.info("="*70)
        logger.info(f"PAPER TRADING SIMULATION: {n_days} Days")
        logger.info("="*70)
        
        if symbols is None:
            # Get first 30 stocks from data directory
            symbols = sorted([
                p.stem.replace("_minute", "")
                for p in self.data_dir.glob("*_minute.csv")
            ])[:30]
        
        logger.info(f"Using {len(symbols)} symbols")
        
        # Generate trading days
        current_date = pd.Timestamp(start_date)
        trading_days = 0
        
        # Vary VIX to simulate different regimes
        vix_values = [12, 14, 16, 18, 20, 22, 24, 15, 17, 19] * 2  # 20 values
        
        while trading_days < n_days:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += pd.Timedelta(days=1)
                continue
            
            date_str = current_date.strftime('%Y-%m-%d')
            vix = vix_values[trading_days % len(vix_values)]
            
            # Simulate trading day
            daily_stats = self.simulate_trading_day(date_str, symbols, vix)
            
            trading_days += 1
            current_date += pd.Timedelta(days=1)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate paper trading validation report."""
        logger.info("\n" + "="*70)
        logger.info("PAPER TRADING RESULTS")
        logger.info("="*70)
        
        if not self.trades:
            logger.warning("No trades generated!")
            return
        
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        
        # Calculate metrics
        n_trades = len(trades_df)
        wins = len(trades_df[trades_df['net_return_pct'] > 0])
        losses = n_trades - wins
        win_rate = wins / n_trades if n_trades > 0 else 0
        
        avg_win = trades_df[trades_df['net_return_pct'] > 0]['net_return_pct'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['net_return_pct'] < 0]['net_return_pct'].mean() if losses > 0 else 0
        
        total_return = trades_df['net_return_pct'].sum()
        avg_trade = trades_df['net_return_pct'].mean()
        
        # Fill rate (% of predicted entries within 0.1% of actual)
        fill_rate = (trades_df['entry_slippage_pct'].abs() <= 0.1).mean() * 100
        avg_slippage = trades_df['entry_slippage_pct'].mean()
        
        # Sharpe (simplified)
        returns = trades_df['net_return_pct'].values
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
        
        logger.info(f"\nTrading Statistics:")
        logger.info(f"  Trading Days: {len(self.daily_pnl)}")
        logger.info(f"  Total Trades: {n_trades}")
        logger.info(f"  Win Rate: {win_rate:.1%} ({wins}/{n_trades})")
        logger.info(f"  Avg Win: {avg_win:.2f}%")
        logger.info(f"  Avg Loss: {avg_loss:.2f}%")
        logger.info(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Avg per Trade: {avg_trade:.3f}%")
        logger.info(f"  Sharpe (annualized): {sharpe:.2f}")
        
        logger.info(f"\nExecution Quality:")
        logger.info(f"  Fill Rate: {fill_rate:.1f}%")
        logger.info(f"  Avg Slippage: {avg_slippage:.3f}%")
        
        # By regime
        logger.info(f"\nPerformance by Regime:")
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            regime_wr = (regime_trades['net_return_pct'] > 0).mean()
            regime_pnl = regime_trades['net_return_pct'].sum()
            logger.info(f"  {regime}: {regime_wr:.1%} WR, {len(regime_trades)} trades, {regime_pnl:.2f}%")
        
        # Validation gate
        logger.info(f"\n" + "="*70)
        logger.info("VALIDATION GATE CHECK")
        logger.info("="*70)
        
        backtest_win_rate = 0.60  # Expected from backtest
        win_rate_diff = abs(win_rate - backtest_win_rate) * 100
        within_tolerance = win_rate_diff <= 5
        fill_rate_ok = fill_rate >= 80
        min_days_met = len(self.daily_pnl) >= 20
        
        logger.info(f"Backtest Win Rate: {backtest_win_rate:.1%}")
        logger.info(f"Paper Trade Win Rate: {win_rate:.1%}")
        logger.info(f"Difference: {win_rate_diff:.1f}%")
        logger.info(f"Within 5% Tolerance: {within_tolerance}")
        logger.info(f"Fill Rate >= 80%: {fill_rate_ok} ({fill_rate:.1f}%)")
        logger.info(f"20+ Trading Days: {min_days_met} ({len(self.daily_pnl)} days)")
        
        can_go_live = within_tolerance and fill_rate_ok and min_days_met
        
        logger.info(f"\n{'='*70}")
        if can_go_live:
            logger.info("✅ VALIDATION PASSED - READY FOR LIVE TRADING")
        else:
            logger.info("❌ VALIDATION FAILED - CONTINUE PAPER TRADING")
            if not within_tolerance:
                logger.info(f"   Reason: Win rate deviation {win_rate_diff:.1f}% > 5%")
            if not fill_rate_ok:
                logger.info(f"   Reason: Fill rate {fill_rate:.1f}% < 80%")
        logger.info(f"{'='*70}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_trading_days': len(self.daily_pnl),
            'n_trades': n_trades,
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'total_return_pct': float(total_return),
            'avg_trade_pct': float(avg_trade),
            'sharpe_ratio': float(sharpe),
            'fill_rate': float(fill_rate),
            'avg_slippage': float(avg_slippage),
            'validation': {
                'within_tolerance': within_tolerance,
                'fill_rate_ok': fill_rate_ok,
                'min_days_met': min_days_met,
                'can_go_live': can_go_live,
            },
            'trades': [t.__dict__ for t in self.trades[:50]],  # Save first 50
        }
        
        output_file = self.output_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trading Simulation")
    parser.add_argument("--start", default="2025-01-01", help="Start date")
    parser.add_argument("--days", type=int, default=20, help="Number of trading days")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Min confidence threshold")
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--model-dir", default="models/v3_production")
    
    args = parser.parse_args()
    
    simulator = PaperTradingSimulator(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        min_confidence=args.min_confidence,
    )
    
    simulator.run_simulation(
        start_date=args.start,
        n_days=args.days,
    )


if __name__ == "__main__":
    main()
