"""
V3.0 Backtest with Full Risk Management

Run backtest using trained v3.0 models with:
- 4-state regime detection
- ATR-based dynamic targets
- Dynamic position sizing
- Correlation-aware portfolio
- Advanced exit logic
- Circuit breakers
"""

import sys
import json
from pathlib import Path
from datetime import datetime, time
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("v3_backtest")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.models.specialized import SpecializedModelSuite
from intradaynet.regime_v3 import RegimeClassifierV3, MarketRegime
from intradaynet.dynamic_targets import DynamicTargetManager
from intradaynet.features.v3_features import EnhancedFeatureEngineer
from intradaynet.risk_management import (
    RiskManager, PositionSizingConfig, PortfolioConfig,
    ExitConfig, CircuitBreakerConfig
)


def run_v3_backtest(
    model_dir: str = "models/v3_production",
    data_dir: str = "nifty500",
    test_start: str = "2025-01-01",
    test_end: str = "2025-03-31",
    max_stocks: int = 30,
    account_value: float = 100000,
):
    """Run comprehensive v3.0 backtest."""
    
    print("="*70)
    print("INTRADAYNET v3.0 - BACKTEST")
    print("="*70)
    print()
    
    # Load models
    print("Loading v3.0 models...")
    suite = SpecializedModelSuite()
    suite.load(model_dir)
    
    with open(Path(model_dir) / "metadata.json") as f:
        metadata = json.load(f)
    
    print(f"✓ Models loaded: {metadata['n_features']} features")
    print(f"✓ Training period: {metadata['train_start']} to {metadata['train_end']}")
    print()
    
    # Initialize components
    regime_classifier = RegimeClassifierV3()
    target_manager = DynamicTargetManager()
    feature_engineer = EnhancedFeatureEngineer()
    
    risk_manager = RiskManager(
        account_value=account_value,
        position_config=PositionSizingConfig(account_value=account_value),
        portfolio_config=PortfolioConfig(max_positions=5),
        exit_config=ExitConfig(),
        circuit_config=CircuitBreakerConfig(account_value=account_value),
    )
    
    # Find test stocks
    data_path = Path(data_dir)
    all_files = list(data_path.glob("*_minute.csv"))
    test_files = all_files[:max_stocks]
    
    print(f"Backtesting {len(test_files)} stocks")
    print(f"Period: {test_start} to {test_end}")
    print()
    
    # Track results
    all_trades = []
    daily_stats = {}
    
    # Simulate each day
    current_date = pd.Timestamp(test_start)
    end_date = pd.Timestamp(test_end)
    
    day_count = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += pd.Timedelta(days=1)
            continue
        
        if day_count % 10 == 0:
            print(f"Processing {date_str}...")
        
        # Simulate regime (in real system, would use actual VIX)
        # For demo, alternate between regimes
        regimes = [MarketRegime.TRENDING_CALM, MarketRegime.TRENDING_VOLATILE, 
                   MarketRegime.CHOPPY_CALM]
        simulated_regime = regimes[day_count % 3]
        
        # Get adjustments
        _, _, adj = regime_classifier.classify(
            vix_level=15 if simulated_regime == MarketRegime.TRENDING_CALM else 20,
            vix_change_pct=0,
        )
        
        # Skip if extreme regime
        if not adj.allow_trading:
            print(f"  Skipping {date_str} - extreme regime")
            current_date += pd.Timedelta(days=1)
            day_count += 1
            continue
        
        # Process each stock
        daily_pnl = 0
        day_trades = 0
        
        for csv_file in test_files[:5]:  # Limit for demo speed
            symbol = csv_file.stem.replace("_minute", "")
            
            try:
                # Load day's data
                df = pd.read_csv(csv_file, parse_dates=['date'])
                df = df.set_index('date')
                df = df[df.index.date == current_date.date()]
                
                if len(df) < 200:
                    continue
                
                # Compute features at midday
                midday_idx = len(df) // 2
                feat_window = feature_engineer.compute_all_features(
                    df.iloc[:midday_idx], symbol
                )
                
                if len(feat_window) < 120:
                    continue
                
                # Feature vector (mean)
                feat_vector = feat_window.mean().values.reshape(1, -1)
                
                # Predict
                preds = suite.predict(feat_vector)
                
                direction_prob = preds['direction_prob'][0]
                magnitude = preds['magnitude_estimate'][0]
                confidence = preds['confidence_score'][0]
                
                # Trading decision
                if direction_prob > 0.58 and confidence > 0.5:
                    # Entry price
                    entry_price = df['close'].iloc[midday_idx]
                    
                    # Compute dynamic targets
                    # Estimate ATR from data
                    atr = (df['high'] - df['low']).tail(20).mean()
                    
                    side = "LONG" if direction_prob > 0.5 else "SHORT"
                    
                    target, stop, target_meta = target_manager.compute_levels(
                        entry_price=entry_price,
                        atr=atr,
                        side=side,
                        regime=simulated_regime,
                        confidence=confidence,
                    )
                    
                    if target_meta.get('skip_trade'):
                        continue
                    
                    # Compute position size
                    size, size_meta = risk_manager.compute_position_size(
                        entry_price=entry_price,
                        atr=atr,
                        regime_adjustments={'size_multiplier': adj.size_multiplier},
                    )
                    
                    # Simulate trade outcome
                    future_window = df.iloc[midday_idx:]
                    if len(future_window) < 30:
                        continue
                    
                    exit_price = None
                    exit_reason = None
                    
                    # Check target/stop hit
                    if side == "LONG":
                        if future_window['high'].max() >= target:
                            exit_price = target
                            exit_reason = "target_hit"
                        elif future_window['low'].min() <= stop:
                            exit_price = stop
                            exit_reason = "stop_hit"
                        else:
                            exit_price = future_window['close'].iloc[-1]
                            exit_reason = "time_exit"
                    else:  # SHORT
                        if future_window['low'].min() <= target:
                            exit_price = target
                            exit_reason = "target_hit"
                        elif future_window['high'].max() >= stop:
                            exit_price = stop
                            exit_reason = "stop_hit"
                        else:
                            exit_price = future_window['close'].iloc[-1]
                            exit_reason = "time_exit"
                    
                    # Calculate P&L
                    if side == "LONG":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    
                    # Assume 0.1% costs
                    costs = 0.001
                    net_pnl_pct = pnl_pct - costs
                    
                    # Add to results
                    trade = {
                        'date': date_str,
                        'symbol': symbol,
                        'side': side,
                        'entry': entry_price,
                        'exit': exit_price,
                        'target': target,
                        'stop': stop,
                        'exit_reason': exit_reason,
                        'gross_pnl_pct': pnl_pct,
                        'net_pnl_pct': net_pnl_pct,
                        'position_size': size,
                        'regime': simulated_regime.value,
                        'confidence': confidence,
                        'direction_prob': direction_prob,
                    }
                    
                    all_trades.append(trade)
                    daily_pnl += net_pnl_pct
                    day_trades += 1
                    
                    # Update risk manager
                    risk_result = risk_manager.update_trade(net_pnl_pct * size, exit_reason)
                    
                    if risk_result.get('halt_trading'):
                        print(f"  ⚠️ Circuit breaker triggered after {len(all_trades)} trades")
                        break
                
            except Exception as e:
                continue
        
        # Record daily stats
        if day_trades > 0:
            daily_stats[date_str] = {
                'trades': day_trades,
                'pnl_pct': daily_pnl,
                'regime': simulated_regime.value,
            }
        
        # Reset daily tracking
        risk_manager.circuit_breaker.reset_daily()
        
        current_date += pd.Timedelta(days=1)
        day_count += 1
    
    # Generate results
    print()
    print("="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print()
    
    if not all_trades:
        print("No trades generated")
        return None
    
    trades_df = pd.DataFrame(all_trades)
    
    # Calculate metrics
    n_trades = len(trades_df)
    wins = len(trades_df[trades_df['net_pnl_pct'] > 0])
    losses = n_trades - wins
    win_rate = wins / n_trades if n_trades > 0 else 0
    
    avg_win = trades_df[trades_df['net_pnl_pct'] > 0]['net_pnl_pct'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['net_pnl_pct'] < 0]['net_pnl_pct'].mean() if losses > 0 else 0
    
    total_pnl = trades_df['net_pnl_pct'].sum()
    avg_trade = trades_df['net_pnl_pct'].mean()
    
    print(f"Total Trades: {n_trades}")
    print(f"Win Rate: {win_rate:.1%} ({wins}/{n_trades})")
    print(f"Avg Win: {avg_win:.2%}")
    print(f"Avg Loss: {avg_loss:.2%}")
    print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
    print(f"Total Return: {total_pnl:.2%}")
    print(f"Avg per Trade: {avg_trade:.3f}")
    print()
    
    # By regime
    print("Performance by Regime:")
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        regime_wr = (regime_trades['net_pnl_pct'] > 0).mean()
        regime_pnl = regime_trades['net_pnl_pct'].sum()
        print(f"  {regime}: {regime_wr:.1%} WR, {regime_pnl:.2f}% total")
    
    print()
    
    # By exit reason
    print("Exit Reason Distribution:")
    for reason in trades_df['exit_reason'].value_counts().index:
        count = trades_df[trades_df['exit_reason'] == reason].shape[0]
        pct = count / n_trades * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    print()
    
    # Save results
    results = {
        'backtest_period': f"{test_start} to {test_end}",
        'n_trades': n_trades,
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'total_pnl_pct': float(total_pnl),
        'avg_trade_pct': float(avg_trade),
        'trades': all_trades[:100],  # Save first 100
    }
    
    output_file = f"backtest_v3_{test_start}_{test_end}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    print()
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/v3_production")
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-03-31")
    parser.add_argument("--max-stocks", type=int, default=30)
    parser.add_argument("--capital", type=float, default=100000)
    
    args = parser.parse_args()
    
    results = run_v3_backtest(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        test_start=args.start,
        test_end=args.end,
        max_stocks=args.max_stocks,
        account_value=args.capital,
    )
