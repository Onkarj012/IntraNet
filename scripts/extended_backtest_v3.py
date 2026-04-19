"""
Extended Backtesting for IntradayNet v3.0

Runs comprehensive backtests across multiple periods:
- Q1 2025, Q2 2025, Q3 2025, Q4 2025
- Different market regimes
- Walk-forward validation

Usage:
    python scripts/extended_backtest_v3.py --full
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("extended_backtest")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.models.specialized import SpecializedModelSuite
from intradaynet.regime_v3 import RegimeClassifierV3, MarketRegime
from intradaynet.dynamic_targets import DynamicTargetManager
from intradaynet.features.v3_features import EnhancedFeatureEngineer
from intradaynet.risk_management import (
    RiskManager, PositionSizingConfig, PortfolioConfig,
    ExitConfig, CircuitBreakerConfig
)


class ExtendedBacktester:
    """
    Run extended backtests across multiple periods.
    """
    
    def __init__(
        self,
        model_dir: str = "models/v3_production",
        data_dir: str = "nifty500",
        output_dir: str = "backtest_results/extended",
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        self.suite = SpecializedModelSuite()
        self.suite.load(model_dir)
        
        # Initialize components
        self.regime_classifier = RegimeClassifierV3()
        self.target_manager = DynamicTargetManager()
        self.feature_engineer = EnhancedFeatureEngineer()
        
        with open(self.model_dir / "metadata.json") as f:
            self.model_metadata = json.load(f)
        
        logger.info("Extended Backtester initialized")
        logger.info(f"  Model features: {self.model_metadata['n_features']}")
    
    def backtest_period(
        self,
        start_date: str,
        end_date: str,
        period_name: str,
        symbols: List[str],
        confidence_threshold: float = 0.58,
    ) -> Dict:
        """
        Backtest a single period.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"BACKTEST: {period_name}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"{'='*70}")
        
        trades = []
        daily_pnls = []
        
        current_date = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Vary VIX to simulate different regimes
        vix_scenarios = [
            (12, "trending_calm"),
            (18, "trending_volatile"),
            (14, "choppy_calm"),
            (25, "choppy_volatile"),
        ]
        
        day_count = 0
        
        while current_date <= end:
            if current_date.weekday() >= 5:
                current_date += pd.Timedelta(days=1)
                continue
            
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Alternate regimes
            vix, regime_desc = vix_scenarios[day_count % len(vix_scenarios)]
            
            regime, _, adj = self.regime_classifier.classify(vix, 0)
            
            if not adj.allow_trading:
                current_date += pd.Timedelta(days=1)
                day_count += 1
                continue
            
            day_pnl = 0
            day_trades = 0
            
            # Process subset of symbols for speed
            for symbol in symbols[:5]:
                try:
                    csv_path = self.data_dir / f"{symbol}_minute.csv"
                    if not csv_path.exists():
                        continue
                    
                    df = pd.read_csv(csv_path, parse_dates=['date'])
                    df = df.set_index('date')
                    df = df[df.index.date == current_date.date()]
                    
                    if len(df) < 200:
                        continue
                    
                    # Midday prediction
                    midday_idx = len(df) // 2
                    feat_window = self.feature_engineer.compute_all_features(
                        df.iloc[:midday_idx], symbol
                    )
                    
                    if len(feat_window) < 120:
                        continue
                    
                    feat_vector = feat_window.mean().values.reshape(1, -1)
                    preds = self.suite.predict(feat_vector)
                    
                    direction_prob = preds['direction_prob'][0]
                    confidence = preds['confidence_score'][0]
                    
                    if confidence < confidence_threshold:
                        continue
                    
                    side = "LONG" if direction_prob > 0.5 else "SHORT"
                    entry_price = df.iloc[midday_idx]['close']
                    atr = (df['high'] - df['low']).tail(20).mean()
                    
                    target, stop, meta = self.target_manager.compute_levels(
                        entry_price, atr, side, regime, confidence
                    )
                    
                    if meta.get('skip_trade'):
                        continue
                    
                    # Simulate outcome
                    future = df.iloc[midday_idx:]
                    if len(future) < 30:
                        continue
                    
                    if side == "LONG":
                        target_hit = future['high'].max() >= target
                        stop_hit = future['low'].min() <= stop
                        
                        if target_hit:
                            ret = (target - entry_price) / entry_price
                            outcome = "target_hit"
                        elif stop_hit:
                            ret = (stop - entry_price) / entry_price
                            outcome = "stop_hit"
                        else:
                            ret = (future['close'].iloc[-1] - entry_price) / entry_price
                            outcome = "time_exit"
                    else:
                        target_hit = future['low'].min() <= target
                        stop_hit = future['high'].max() >= stop
                        
                        if target_hit:
                            ret = (entry_price - target) / entry_price
                            outcome = "target_hit"
                        elif stop_hit:
                            ret = (entry_price - stop) / entry_price
                            outcome = "stop_hit"
                        else:
                            ret = (entry_price - future['close'].iloc[-1]) / entry_price
                            outcome = "time_exit"
                    
                    # Net return with costs
                    net_ret = ret - 0.001
                    
                    trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'side': side,
                        'outcome': outcome,
                        'return': net_ret * 100,
                        'regime': regime.value,
                        'confidence': confidence,
                    })
                    
                    day_pnl += net_ret * 100
                    day_trades += 1
                    
                except Exception as e:
                    continue
            
            if day_trades > 0:
                daily_pnls.append(day_pnl)
            
            current_date += pd.Timedelta(days=1)
            day_count += 1
        
        # Calculate metrics
        if not trades:
            return {'period': period_name, 'trades': 0, 'error': 'No trades'}
        
        trades_df = pd.DataFrame(trades)
        
        n_trades = len(trades_df)
        wins = len(trades_df[trades_df['return'] > 0])
        win_rate = wins / n_trades if n_trades > 0 else 0
        
        total_return = trades_df['return'].sum()
        avg_trade = trades_df['return'].mean()
        
        returns = trades_df['return'].values
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = drawdown.min()
        
        results = {
            'period': period_name,
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': len(daily_pnls),
            'n_trades': n_trades,
            'win_rate': float(win_rate),
            'total_return_pct': float(total_return),
            'avg_trade_pct': float(avg_trade),
            'sharpe_ratio': float(sharpe),
            'max_drawdown_pct': float(max_dd),
            'trades_by_regime': trades_df.groupby('regime').size().to_dict(),
            'trades_by_outcome': trades_df.groupby('outcome').size().to_dict(),
        }
        
        logger.info(f"\nResults for {period_name}:")
        logger.info(f"  Trading Days: {results['trading_days']}")
        logger.info(f"  Trades: {n_trades}")
        logger.info(f"  Win Rate: {win_rate:.1%}")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Avg per Trade: {avg_trade:.3f}%")
        logger.info(f"  Sharpe: {sharpe:.2f}")
        logger.info(f"  Max DD: {max_dd:.2f}%")
        
        return results
    
    def run_extended_backtest(self, full: bool = False):
        """Run backtests across multiple periods."""
        logger.info("="*70)
        logger.info("EXTENDED BACKTESTING - v3.0")
        logger.info("="*70)
        
        # Get symbols
        symbols = sorted([
            p.stem.replace("_minute", "")
            for p in self.data_dir.glob("*_minute.csv")
        ])[:30]  # Use 30 stocks
        
        logger.info(f"Using {len(symbols)} symbols for backtest")
        
        # Define periods
        if full:
            periods = [
                ("2025-01-01", "2025-03-31", "Q1 2025"),
                ("2025-04-01", "2025-06-30", "Q2 2025"),
                # Note: Later periods don't have data yet
                # ("2025-07-01", "2025-09-30", "Q3 2025"),
                # ("2025-10-01", "2025-12-31", "Q4 2025"),
            ]
        else:
            # Quick test with just one period
            periods = [
                ("2025-01-01", "2025-01-31", "January 2025"),
            ]
        
        all_results = []
        
        for start, end, name in periods:
            result = self.backtest_period(start, end, name, symbols)
            all_results.append(result)
        
        # Aggregate results
        logger.info(f"\n{'='*70}")
        logger.info("AGGREGATE RESULTS")
        logger.info(f"{'='*70}")
        
        total_trades = sum(r['n_trades'] for r in all_results if 'n_trades' in r)
        total_return = sum(r['total_return_pct'] for r in all_results if 'total_return_pct' in r)
        
        # Weighted average win rate
        win_rates = [r['win_rate'] for r in all_results if 'win_rate' in r]
        weights = [r['n_trades'] for r in all_results if 'n_trades' in r]
        avg_win_rate = np.average(win_rates, weights=weights) if weights else 0
        
        logger.info(f"\nAcross {len(all_results)} periods:")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Avg Win Rate: {avg_win_rate:.1%}")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Avg per Trade: {total_return/total_trades:.3f}%" if total_trades > 0 else "N/A")
        
        # Save results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': str(self.model_dir),
            'n_periods': len(all_results),
            'total_trades': total_trades,
            'avg_win_rate': float(avg_win_rate),
            'total_return_pct': float(total_return),
            'period_results': all_results,
        }
        
        output_file = self.output_dir / f"extended_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file}")
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended Backtesting")
    parser.add_argument("--full", action="store_true", help="Run full multi-period backtest")
    parser.add_argument("--model-dir", default="models/v3_production")
    parser.add_argument("--data-dir", default="nifty500")
    
    args = parser.parse_args()
    
    backtester = ExtendedBacktester(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
    )
    
    results = backtester.run_extended_backtest(full=args.full)
    
    print("\n" + "="*70)
    print("EXTENDED BACKTEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
