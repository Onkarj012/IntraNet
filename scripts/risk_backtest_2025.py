#!/usr/bin/env python3
"""
Risk-Based Backtesting for 2025 Blind Test.

Tests 3 risk profiles on full year 2025 with hit rate tracking.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder


class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskConfig:
    name: str
    confidence_long: float
    confidence_short: float
    stop_loss_pct: float
    target_pct: float
    max_trades_per_day: int
    position_size: float


RISK_PROFILES = {
    RiskProfile.CONSERVATIVE: RiskConfig(
        name="Conservative",
        confidence_long=0.70,
        confidence_short=0.30,
        stop_loss_pct=0.005,
        target_pct=0.005,
        max_trades_per_day=2,
        position_size=50000,
    ),
    RiskProfile.BALANCED: RiskConfig(
        name="Balanced",
        confidence_long=0.60,
        confidence_short=0.40,
        stop_loss_pct=0.010,
        target_pct=0.010,
        max_trades_per_day=4,
        position_size=100000,
    ),
    RiskProfile.AGGRESSIVE: RiskConfig(
        name="Aggressive",
        confidence_long=0.55,
        confidence_short=0.45,
        stop_loss_pct=0.020,
        target_pct=0.020,
        max_trades_per_day=8,
        position_size=200000,
    ),
}


def simulate_with_hit_rate(day_data: pd.DataFrame, direction: str, 
                           entry_price: float, target_pct: float, 
                           stop_pct: float) -> Dict:
    """Simulate trade tracking hit rate and intraday metrics."""
    if direction == "LONG":
        target = entry_price * (1 + target_pct)
        stop = entry_price * (1 - stop_pct)
    else:
        target = entry_price * (1 - target_pct)
        stop = entry_price * (1 + stop_pct)
    
    hit = False
    hit_target = False
    hit_stop = False
    exit_price = day_data["close"].iloc[-1]
    exit_type = "EOD"
    
    max_fav = 0.0  # Best profit % during day
    max_adv = 0.0  # Worst drawdown % during day
    
    highs = day_data["high"].values
    lows = day_data["low"].values
    closes = day_data["close"].values
    
    for i in range(len(day_data)):
        high = highs[i]
        low = lows[i]
        
        # Track max favorable
        if direction == "LONG":
            pnl = (high - entry_price) / entry_price
            dd = (low - entry_price) / entry_price
        else:
            pnl = (entry_price - low) / entry_price
            dd = (entry_price - high) / entry_price
        
        max_fav = max(max_fav, pnl)
        max_adv = min(max_adv, dd)
        
        # Check exits
        if direction == "LONG":
            if high >= target:
                hit_target = True
                hit = True
                exit_price = target
                exit_type = "TARGET"
                break
            elif low <= stop:
                hit_stop = True
                exit_price = stop
                exit_type = "STOP"
                break
        else:
            if low <= target:
                hit_target = True
                hit = True
                exit_price = target
                exit_type = "TARGET"
                break
            elif high >= stop:
                hit_stop = True
                exit_price = stop
                exit_type = "STOP"
                break
    
    # Calculate final P&L
    if direction == "LONG":
        gross = (exit_price - entry_price) / entry_price
    else:
        gross = (entry_price - exit_price) / entry_price
    
    return {
        "hit": hit or hit_target,
        "hit_target": hit_target,
        "hit_stop": hit_stop,
        "win": gross > 0,
        "gross_pct": gross,
        "max_favorable": max_fav,
        "max_adverse": max_adv,
        "exit_price": exit_price,
        "exit_type": exit_type,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/nifty100_model_2021_2024.pkl")
    parser.add_argument("--risk", type=str, default="balanced", choices=["conservative", "balanced", "aggressive"])
    
    args = parser.parse_args()
    
    risk_profile = RiskProfile(args.risk.upper())
    config = RISK_PROFILES[risk_profile]
    
    print("=" * 80)
    print(f"2025 BLIND TEST - {config.name.upper()} RISK PROFILE")
    print("=" * 80)
    print(f"Confidence: Long >{config.confidence_long}, Short <{config.confidence_short}")
    print(f"Stop/Target: {config.stop_loss_pct*100:.1f}%/{config.target_pct*100:.1f}%")
    print(f"Position Size: ₹{config.position_size:,.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
