#!/usr/bin/env python3
"""
Realistic Execution Engine with Slippage and Market Impact.

Models real-world execution:
1. Entry: Market order at open (with slippage)
2. Exit: Stop loss, target, trailing stop, or time
3. Slippage: Random 0.03-0.08% per side
4. Market impact: Larger positions = worse fills
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    # Costs
    brokerage_per_order: float = 20.0
    stt_pct: float = 0.00025
    exchange_txn_pct: float = 0.0000345
    gst_brokerage_pct: float = 0.18
    stamp_duty_pct: float = 0.00003
    
    # Slippage (per side)
    slippage_min_pct: float = 0.0003  # 0.03%
    slippage_max_pct: float = 0.0008  # 0.08%
    
    # Market impact
    impact_per_lakh: float = 0.0001  # 0.01% per ₹1L
    
    def calculate_costs(self, position_value: float) -> float:
        """Calculate total round-trip costs including slippage."""
        # Brokerage
        brokerage = self.brokerage_per_order * 2
        
        # STT (sell side only for intraday)
        stt = position_value * self.stt_pct
        
        # Exchange charges
        exchange = position_value * self.exchange_txn_pct * 2
        
        # GST
        gst = brokerage * self.gst_brokerage_pct
        
        # Stamp duty
        stamp = position_value * self.stamp_duty_pct
        
        # Slippage (both sides)
        avg_slippage = (self.slippage_min_pct + self.slippage_max_pct) / 2
        slippage = position_value * avg_slippage * 2
        
        return brokerage + stt + exchange + gst + stamp + slippage
    
    def calculate_entry_price(self, theoretical_open: float, direction: str, 
                             position_value: float) -> float:
        """Calculate realistic entry price with slippage and impact."""
        # Random slippage
        slippage = np.random.uniform(self.slippage_min_pct, self.slippage_max_pct)
        
        # Market impact (larger positions = worse fills)
        impact = (position_value / 100000) * self.impact_per_lakh
        
        total_impact = slippage + impact
        
        if direction == "LONG":
            # Buy at worse price
            return theoretical_open * (1 + total_impact)
        else:
            # Short at worse price
            return theoretical_open * (1 - total_impact)
    
    def calculate_exit_price(self, theoretical_price: float, direction: str,
                            position_value: float) -> float:
        """Calculate realistic exit price with slippage."""
        slippage = np.random.uniform(self.slippage_min_pct, self.slippage_max_pct)
        impact = (position_value / 100000) * self.impact_per_lakh
        total_impact = slippage + impact
        
        if direction == "LONG":
            # Sell at worse price
            return theoretical_price * (1 - total_impact)
        else:
            # Cover at worse price
            return theoretical_price * (1 + total_impact)


class RealisticExecution:
    """Simulate realistic trade execution."""
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
    
    def simulate_trade(self, minute_data: pd.DataFrame, direction: str,
                       entry_price: float, target_price: float, stop_price: float,
                       position_value: float, use_trailing: bool = False,
                       trailing_start: float = 0.0, trailing_stop_pct: float = 0.0) -> Dict:
        """
        Simulate trade with realistic execution.
        
        Args:
            minute_data: Minute-by-minute data for the day
            direction: "LONG" or "SHORT"
            entry_price: Theoretical entry (open)
            target_price: Target exit price
            stop_price: Stop loss price
            position_value: Position size in rupees
            use_trailing: Whether to use trailing stop
            trailing_start: Profit % to activate trailing
            trailing_stop_pct: Trailing distance
        
        Returns:
            Dict with execution details
        """
        # Realistic entry
        actual_entry = self.config.calculate_entry_price(entry_price, direction, position_value)
        
        highs = minute_data["high"].values
        lows = minute_data["low"].values
        closes = minute_data["close"].values
        
        # Exit at 3 PM
        exit_3pm_idx = min(330, len(minute_data) - 1)
        
        exit_price = closes[-1]
        exit_reason = "EOD"
        exit_time = len(minute_data) - 1
        
        max_favorable = 0.0
        max_adverse = 0.0
        trailing_stop = None
        hit_target_level = False
        
        for i in range(len(minute_data)):
            high = highs[i]
            low = lows[i]
            
            # Track P&L
            if direction == "LONG":
                current_pnl = (high - actual_entry) / actual_entry
                current_dd = (low - actual_entry) / actual_entry
                if high >= target_price:
                    hit_target_level = True
            else:
                current_pnl = (actual_entry - low) / actual_entry
                current_dd = (actual_entry - high) / actual_entry
                if low <= target_price:
                    hit_target_level = True
            
            max_favorable = max(max_favorable, current_pnl)
            max_adverse = min(max_adverse, current_dd)
            
            # Trailing stop logic
            if use_trailing and trailing_stop is None and abs(current_pnl) >= trailing_start:
                if direction == "LONG":
                    trailing_stop = high * (1 - trailing_stop_pct)
                else:
                    trailing_stop = low * (1 + trailing_stop_pct)
            
            # Update trailing stop
            if use_trailing and trailing_stop is not None:
                if direction == "LONG":
                    new_stop = high * (1 - trailing_stop_pct)
                    trailing_stop = max(trailing_stop, new_stop)
                else:
                    new_stop = low * (1 + trailing_stop_pct)
                    trailing_stop = min(trailing_stop, new_stop)
            
            # Check exits
            if direction == "LONG":
                if high >= target_price:
                    exit_price = self.config.calculate_exit_price(target_price, direction, position_value)
                    exit_time = i
                    exit_reason = "TARGET"
                    break
                elif low <= stop_price:
                    exit_price = self.config.calculate_exit_price(stop_price, direction, position_value)
                    exit_time = i
                    exit_reason = "STOP"
                    break
                elif trailing_stop is not None and low <= trailing_stop:
                    exit_price = self.config.calculate_exit_price(trailing_stop, direction, position_value)
                    exit_time = i
                    exit_reason = "TRAILING"
                    break
            else:  # SHORT
                if low <= target_price:
                    exit_price = self.config.calculate_exit_price(target_price, direction, position_value)
                    exit_time = i
                    exit_reason = "TARGET"
                    break
                elif high >= stop_price:
                    exit_price = self.config.calculate_exit_price(stop_price, direction, position_value)
                    exit_time = i
                    exit_reason = "STOP"
                    break
                elif trailing_stop is not None and high >= trailing_stop:
                    exit_price = self.config.calculate_exit_price(trailing_stop, direction, position_value)
                    exit_time = i
                    exit_reason = "TRAILING"
                    break
            
            # 3 PM exit
            if i >= exit_3pm_idx:
                exit_price = self.config.calculate_exit_price(closes[i], direction, position_value)
                exit_time = i
                exit_reason = "3PM"
                break
        
        # Calculate P&L
        if direction == "LONG":
            gross_pct = (exit_price - actual_entry) / actual_entry
        else:
            gross_pct = (actual_entry - exit_price) / actual_entry
        
        gross_pnl = gross_pct * position_value
        costs = self.config.calculate_costs(position_value)
        net_pnl = gross_pnl - costs
        
        return {
            "entry_price": actual_entry,
            "exit_price": exit_price,
            "gross_pct": gross_pct,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "net_pnl": net_pnl,
            "exit_reason": exit_reason,
            "exit_time": exit_time,
            "hit_target": hit_target_level,
            "max_favorable_pct": max_favorable,
            "max_adverse_pct": max_adverse,
        }
