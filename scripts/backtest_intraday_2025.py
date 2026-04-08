#!/usr/bin/env python3
"""
Risk-Based Backtesting for Intraday Movement Model.

Tests 3 risk profiles on 2025 blind test:
- Conservative: 0.5% stop, trailing SL, fixed target
- Balanced: 1.0% stop, trailing SL, fixed target  
- Aggressive: 2.0% stop, trailing SL, fixed target

Exit strategies:
1. Stop Loss: Exit at fixed loss
2. Trailing Stop: After X% profit, trail at Y% from high
3. Target: Exit at fixed profit
4. Time: Exit at 3 PM if none hit

Tracks hit rate vs win rate.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder


@dataclass
class RiskConfig:
    name: str
    stop_loss_pct: float      # Fixed stop loss
    target_pct: float         # Fixed profit target
    trailing_start: float     # When to start trailing (e.g., 0.5% profit)
    trailing_stop_pct: float  # Trail distance
    max_trades_per_day: int
    position_size: float
    min_confidence: float     # Minimum prediction confidence


RISK_PROFILES = {
    "conservative": RiskConfig(
        name="Conservative",
        stop_loss_pct=0.005,      # 0.5%
        target_pct=0.010,         # 1.0% target
        trailing_start=0.005,     # Start trail at 0.5% profit
        trailing_stop_pct=0.003,  # 0.3% trail
        max_trades_per_day=3,
        position_size=50000,
        min_confidence=0.65,
    ),
    "balanced": RiskConfig(
        name="Balanced",
        stop_loss_pct=0.010,      # 1.0%
        target_pct=0.015,         # 1.5% target
        trailing_start=0.008,     # Start trail at 0.8% profit
        trailing_stop_pct=0.005,  # 0.5% trail
        max_trades_per_day=5,
        position_size=100000,
        min_confidence=0.60,
    ),
    "aggressive": RiskConfig(
        name="Aggressive",
        stop_loss_pct=0.020,      # 2.0%
        target_pct=0.025,         # 2.5% target
        trailing_start=0.015,     # Start trail at 1.5% profit
        trailing_stop_pct=0.010,  # 1.0% trail
        max_trades_per_day=8,
        position_size=200000,
        min_confidence=0.55,
    ),
}


def simulate_trade(day_data: pd.DataFrame, direction: str, entry_price: float,
                   config: RiskConfig) -> Dict:
    """
    Simulate trade with multiple exit strategies.
    
    Returns:
    - hit_target: Did price reach target?
    - hit_stop: Did price hit stop?
    - hit_trailing: Did trailing stop trigger?
    - exit_price: Final exit price
    - exit_time: When exit occurred (minute index)
    - max_profit: Best unrealized profit
    - max_drawdown: Worst unrealized loss
    - gross_pnl: Final P&L
    """
    if direction == "LONG":
        target = entry_price * (1 + config.target_pct)
        stop = entry_price * (1 - config.stop_loss_pct)
        trail_trigger = entry_price * (1 + config.trailing_start)
    else:  # SHORT
        target = entry_price * (1 - config.target_pct)
        stop = entry_price * (1 + config.stop_loss_pct)
        trail_trigger = entry_price * (1 - config.trailing_start)
    
    highs = day_data["high"].values
    lows = day_data["low"].values
    closes = day_data["close"].values
    times = day_data.index
    
    # Exit at 3 PM (approximately 330th minute of 375)
    exit_3pm_idx = min(330, len(day_data) - 1)
    
    max_profit = 0.0
    max_drawdown = 0.0
    trailing_stop = None
    exit_price = closes[-1]  # Default: EOD
    exit_time = len(day_data) - 1
    exit_reason = "EOD"
    
    for i in range(len(day_data)):
        high = highs[i]
        low = lows[i]
        
        # Calculate current P&L
        if direction == "LONG":
            current_pnl = (high - entry_price) / entry_price
            current_dd = (low - entry_price) / entry_price
        else:
            current_pnl = (entry_price - low) / entry_price
            current_dd = (entry_price - high) / entry_price
        
        max_profit = max(max_profit, current_pnl)
        max_drawdown = min(max_drawdown, current_dd)
        
        # Update trailing stop
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
        else:  # SHORT
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
    
    # Calculate gross P&L
    if direction == "LONG":
        gross = (exit_price - entry_price) / entry_price
    else:
        gross = (entry_price - exit_price) / entry_price
    
    return {
        "hit_target": exit_reason == "TARGET",
        "hit_stop": exit_reason == "STOP",
        "hit_trailing": exit_reason == "TRAILING",
        "exit_price": exit_price,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "max_profit_pct": max_profit,
        "max_drawdown_pct": max_drawdown,
        "gross_pct": gross,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/intraday_model.pkl")
    parser.add_argument("--risk", type=str, default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    
    args = parser.parse_args()
    
    config = RISK_PROFILES[args.risk]
    
    print("=" * 80)
    print(f"2025 BLIND TEST - {config.name.upper()}")
    print("=" * 80)
    print(f"Stop Loss: {config.stop_loss_pct*100:.1f}%")
    print(f"Target: {config.target_pct*100:.1f}%")
    print(f"Trailing: Start at {config.trailing_start*100:.1f}%, Trail {config.trailing_stop_pct*100:.1f}%")
    print(f"Position Size: ₹{config.position_size:,.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
