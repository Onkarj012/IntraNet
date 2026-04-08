#!/usr/bin/env python3
"""
Backtester - Risk-Based Trading on Saved Predictions.

Reads predictions from CSV and applies risk management strategies.

Usage:
    python scripts/backtest_predictions.py --predictions predictions_2025.csv --risk balanced
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class RiskConfig:
    name: str
    stop_loss_pct: float
    target_pct: float
    trailing_start: float
    trailing_stop_pct: float
    max_trades_per_day: int
    position_size: float
    min_confidence: float


RISK_PROFILES = {
    "conservative": RiskConfig(
        name="Conservative",
        stop_loss_pct=0.005,
        target_pct=0.010,
        trailing_start=0.005,
        trailing_stop_pct=0.003,
        max_trades_per_day=3,
        position_size=50000,
        min_confidence=0.008,
    ),
    "balanced": RiskConfig(
        name="Balanced",
        stop_loss_pct=0.010,
        target_pct=0.015,
        trailing_start=0.008,
        trailing_stop_pct=0.005,
        max_trades_per_day=5,
        position_size=100000,
        min_confidence=0.012,
    ),
    "aggressive": RiskConfig(
        name="Aggressive",
        stop_loss_pct=0.020,
        target_pct=0.025,
        trailing_start=0.015,
        trailing_stop_pct=0.010,
        max_trades_per_day=8,
        position_size=200000,
        min_confidence=0.015,
    ),
}


COST_PER_TRADE = 182


def load_price_data(symbol: str, date: str, data_dir: Path) -> pd.DataFrame:
    """Load minute data for a symbol on a specific date."""
    csv_path = data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = df.columns.str.lower()
        
        # Filter to specific date
        day_data = df[df.index.date == pd.to_datetime(date).date()]
        return day_data if len(day_data) > 0 else None
    except:
        return None


def simulate_trade(day_data: pd.DataFrame, direction: str, entry_price: float,
                   config: RiskConfig) -> Dict:
    """Simulate trade with multiple exit strategies."""
    if direction == "LONG":
        target = entry_price * (1 + config.target_pct)
        stop = entry_price * (1 - config.stop_loss_pct)
    else:
        target = entry_price * (1 - config.target_pct)
        stop = entry_price * (1 + config.stop_loss_pct)
    
    highs = day_data["high"].values
    lows = day_data["low"].values
    closes = day_data["close"].values
    
    # Exit at 3 PM (approx 330th minute)
    exit_3pm_idx = min(330, len(day_data) - 1)
    
    max_favorable = 0.0
    max_adverse = 0.0
    trailing_stop = None
    exit_price = closes[-1]
    exit_time = len(day_data) - 1
    exit_reason = "EOD"
    hit_target_level = False
    
    for i in range(len(day_data)):
        high = highs[i]
        low = lows[i]
        
        # Track P&L
        if direction == "LONG":
            current_pnl = (high - entry_price) / entry_price
            current_dd = (low - entry_price) / entry_price
            
            # Check if target level was reached (for hit rate)
            if high >= target:
                hit_target_level = True
        else:
            current_pnl = (entry_price - low) / entry_price
            current_dd = (entry_price - high) / entry_price
            
            if low <= target:
                hit_target_level = True
        
        max_favorable = max(max_favorable, current_pnl)
        max_adverse = min(max_adverse, current_dd)
        
        # Trailing stop
        if trailing_stop is None and abs(current_pnl) >= config.trailing_start:
            if direction == "LONG":
                trailing_stop = high * (1 - config.trailing_stop_pct)
            else:
                trailing_stop = low * (1 + config.trailing_stop_pct)
        
        # Check exits
        if direction == "LONG":
            if high >= target:
                exit_price = target
                exit_time = i
                exit_reason = "TARGET"
                break
            elif low <= stop:
                exit_price = stop
                exit_time = i
                exit_reason = "STOP"
                break
            elif trailing_stop is not None and low <= trailing_stop:
                exit_price = trailing_stop
                exit_time = i
                exit_reason = "TRAILING"
                break
        else:
            if low <= target:
                exit_price = target
                exit_time = i
                exit_reason = "TARGET"
                break
            elif high >= stop:
                exit_price = stop
                exit_time = i
                exit_reason = "STOP"
                break
            elif trailing_stop is not None and high >= trailing_stop:
                exit_price = trailing_stop
                exit_time = i
                exit_reason = "TRAILING"
                break
        
        # 3 PM exit
        if i >= exit_3pm_idx:
            exit_price = closes[i]
            exit_time = i
            exit_reason = "3PM"
            break
    
    # Calculate P&L
    if direction == "LONG":
        gross_pct = (exit_price - entry_price) / entry_price
    else:
        gross_pct = (entry_price - exit_price) / entry_price
    
    quantity = int(config.position_size / entry_price)
    gross_pnl = gross_pct * config.position_size
    costs = COST_PER_TRADE * (config.position_size / 100000)  # Scale costs
    net_pnl = gross_pnl - costs
    
    return {
        "hit_target": hit_target_level,
        "exit_reason": exit_reason,
        "gross_pct": gross_pct,
        "net_pnl": net_pnl,
        "max_favorable_pct": max_favorable,
        "max_adverse_pct": max_adverse,
        "quantity": quantity,
        "exit_price": exit_price,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--risk", type=str, default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--output", type=str, default="backtest_results.json")
    
    args = parser.parse_args()
    
    config = RISK_PROFILES[args.risk]
    
    print("=" * 80)
    print(f"BACKTEST: {config.name.upper()} RISK PROFILE")
    print("=" * 80)
    print(f"Predictions file: {args.predictions}")
    print(f"Stop: {config.stop_loss_pct*100:.1f}% | Target: {config.target_pct*100:.1f}%")
    print(f"Position: ₹{config.position_size:,.0f} | Min confidence: {config.min_confidence*100:.2f}%")
    print("=" * 80)
    
    # Load predictions
    print("\nLoading predictions...")
    pred_df = pd.read_csv(args.predictions)
    print(f"✓ Loaded {len(pred_df)} predictions")
    print(f"  Dates: {pred_df['date'].min()} to {pred_df['date'].max()}")
    print(f"  Symbols: {pred_df['symbol'].nunique()}")
    
    # Filter by confidence
    pred_df = pred_df[pred_df["confidence"] >= config.min_confidence]
    print(f"  After confidence filter: {len(pred_df)} predictions")
    
    # Backtest
    data_dir = Path("nifty500")
    trades = []
    
    print(f"\nSimulating trades...")
    print("-" * 80)
    
    for idx, row in pred_df.iterrows():
        date = row["date"]
        symbol = row["symbol"]
        predicted_return = row["predicted_return"]
        direction = row["direction"]
        
        # Load price data
        day_data = load_price_data(symbol, date, data_dir)
        if day_data is None or len(day_data) < 30:
            continue
        
        entry_price = day_data["open"].iloc[0]
        if entry_price <= 0:
            continue
        
        # Simulate trade
        result = simulate_trade(day_data, direction, entry_price, config)
        
        trades.append({
            "date": date,
            "symbol": symbol,
            "direction": direction,
            "predicted_return": predicted_return,
            "entry_price": entry_price,
            **result,
        })
        
        if len(trades) % 100 == 0:
            print(f"  Processed {len(trades)} trades...")
    
    if not trades:
        print("\n❌ No trades simulated!")
        return
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pct"] * 100 for t in trades]
    
    n_trades = len(trades)
    n_wins = sum(1 for p in net_pnls if p > 0)
    n_hit_target = sum(1 for t in trades if t["hit_target"])
    
    win_rate = n_wins / n_trades
    hit_rate = n_hit_target / n_trades
    
    total_net = sum(net_pnls)
    total_gross = sum(gross_pnls)
    
    avg_win = np.mean([p for p in net_pnls if p > 0]) if n_wins > 0 else 0
    avg_loss = np.mean([p for p in net_pnls if p <= 0]) if n_trades - n_wins > 0 else 0
    
    # Risk metrics
    daily_pnls = pd.DataFrame(trades).groupby("date")["net_pnl"].sum()
    daily_returns = daily_pnls / config.position_size
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    cumulative = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.min(cumulative - running_max)
    
    print(f"\n📊 Trading Statistics:")
    print(f"  Total Trades:       {n_trades}")
    print(f"  Win Rate:           {win_rate:.1%} ({n_wins}/{n_trades})")
    print(f"  Hit Rate:           {hit_rate:.1%} ({n_hit_target}/{n_trades})")
    
    print(f"\n💰 P&L Summary:")
    print(f"  Gross P&L:          ₹{total_gross:>12,.0f}")
    print(f"  Total Costs:        ₹{len(trades) * COST_PER_TRADE * (config.position_size/100000):>12,.0f}")
    print(f"  Net P&L:            ₹{total_net:>12,.0f}")
    
    print(f"\n📈 Risk Metrics:")
    print(f"  Sharpe Ratio:       {sharpe:.3f}")
    print(f"  Max Drawdown:       ₹{max_dd:>12,.0f}")
    
    print(f"\n📉 Per Trade:")
    print(f"  Avg Win:            ₹{avg_win:>12,.0f}")
    print(f"  Avg Loss:           ₹{avg_loss:>12,.0f}")
    
    # By direction
    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]
    
    print(f"\n🎯 Direction Breakdown:")
    print(f"  Long:  {len(long_trades)} trades, Win: {sum(1 for t in long_trades if t['net_pnl'] > 0)/len(long_trades):.1%}" if long_trades else "  Long:  0 trades")
    print(f"  Short: {len(short_trades)} trades, Win: {sum(1 for t in short_trades if t['net_pnl'] > 0)/len(short_trades):.1%}" if short_trades else "  Short: 0 trades")
    
    print(f"\n{'='*80}")
    
    # Save results
    output = {
        "risk_profile": config.name,
        "summary": {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "hit_rate": hit_rate,
            "total_net": total_net,
            "sharpe": sharpe,
            "max_drawdown": float(max_dd),
        },
        "trades": trades,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"✓ Results saved: {args.output}")


if __name__ == "__main__":
    main()
