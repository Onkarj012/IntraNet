"""
Gap prediction targets for IntradayNet.

Targets designed for morning picks (pre-market prediction):
1. Gap Direction: Will the stock gap up or down at open?
2. Gap Magnitude: How large will the gap be?
3. Gap Fill: Will the gap fill within the first hour?
4. Gap Reversal: Will price reverse after the gap?

All targets use data available before market open (9:15 AM IST).
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class GapTargetConfig:
    """Configuration for gap target computation."""
    direction_threshold: float = 0.002  # 0.2% minimum gap to be significant
    fill_time_horizon: int = 60  # Check for fill within first 60 minutes
    magnitude_clip: float = 0.05  # Clip gaps at ±5%


@dataclass 
class GapTargets:
    """Container for all gap-related targets."""
    # Direction: -1 (down), 0 (no gap), 1 (up)
    direction: int
    
    # Magnitude: signed gap size (clipped)
    magnitude: float
    
    # Binary classification targets
    gaps_up: bool          # Gap up > threshold
    gaps_down: bool        # Gap down > threshold
    any_gap: bool          # Gap in either direction > threshold
    
    # Gap fill targets
    fills_gap: bool        # Gap fills within horizon
    gap_fill_time: int     # Minutes to fill (or -1 if doesn't fill)
    
    # Trading targets
    profitable_long: bool  # Can buy at open and profit
    profitable_short: bool # Can short at open and profit
    
    # Cost-adjusted
    net_gap_long: float    # Gap - costs
    net_gap_short: float   # Gap - costs (absolute)


def compute_gap_targets(
    daily_df: pd.DataFrame,
    next_day_minute: Optional[pd.DataFrame] = None,
    config: GapTargetConfig = GapTargetConfig(),
    costs: float = 0.002,  # 0.2% round-trip cost
) -> Optional[GapTargets]:
    """
    Compute gap targets from daily data.
    
    Args:
        daily_df: DataFrame with daily OHLCV (need at least 2 days)
        next_day_minute: Optional minute data for gap fill analysis
        config: Target configuration
        costs: Transaction costs as fraction
        
    Returns:
        GapTargets object or None if insufficient data
    """
    if len(daily_df) < 2:
        return None
    
    # Get yesterday's close and today's open
    prev_close = daily_df["close"].iloc[-2]
    today_open = daily_df["open"].iloc[-1]
    today_high = daily_df["high"].iloc[-1]
    today_low = daily_df["low"].iloc[-1]
    today_close = daily_df["close"].iloc[-1]
    
    if prev_close <= 0:
        return None
    
    # Calculate gap
    gap = (today_open / prev_close) - 1
    gap_clipped = np.clip(gap, -config.magnitude_clip, config.magnitude_clip)
    
    # Direction target
    if gap > config.direction_threshold:
        direction = 1
    elif gap < -config.direction_threshold:
        direction = -1
    else:
        direction = 0
    
    # Binary targets
    gaps_up = gap > config.direction_threshold
    gaps_down = gap < -config.direction_threshold
    any_gap = gaps_up or gaps_down
    
    # Gap fill analysis (if minute data available)
    fills_gap = False
    gap_fill_time = -1
    
    if next_day_minute is not None and len(next_day_minute) > 0:
        if gaps_up:
            # Gap fills if price drops below previous close
            for i, (idx, row) in enumerate(next_day_minute.iterrows()):
                if i >= config.fill_time_horizon:
                    break
                if row["low"] <= prev_close:
                    fills_gap = True
                    gap_fill_time = i
                    break
        elif gaps_down:
            # Gap fills if price rises above previous close
            for i, (idx, row) in enumerate(next_day_minute.iterrows()):
                if i >= config.fill_time_horizon:
                    break
                if row["high"] >= prev_close:
                    fills_gap = True
                    gap_fill_time = i
                    break
    else:
        # Approximate fill using daily data
        if gaps_up and today_low <= prev_close:
            fills_gap = True
            gap_fill_time = 30  # Estimate
        elif gaps_down and today_high >= prev_close:
            fills_gap = True
            gap_fill_time = 30  # Estimate
    
    # Profitable targets (can we trade the gap?)
    # Buy at open, sell at best price during day
    best_long_exit = max(today_high, today_open) if gaps_up else today_close
    long_profit = (best_long_exit / today_open - 1) - costs if today_open > 0 else -costs
    profitable_long = long_profit > 0
    
    # Short at open, cover at best price during day  
    best_short_exit = min(today_low, today_open) if gaps_down else today_close
    short_profit = (today_open / best_short_exit - 1) - costs if best_short_exit > 0 else -costs
    profitable_short = short_profit > 0
    
    # Cost-adjusted gaps
    net_gap_long = gap - costs
    net_gap_short = abs(gap) - costs
    
    return GapTargets(
        direction=direction,
        magnitude=gap_clipped,
        gaps_up=gaps_up,
        gaps_down=gaps_down,
        any_gap=any_gap,
        fills_gap=fills_gap,
        gap_fill_time=gap_fill_time,
        profitable_long=profitable_long,
        profitable_short=profitable_short,
        net_gap_long=net_gap_long,
        net_gap_short=net_gap_short,
    )


def compute_gap_target_series(
    daily_df: pd.DataFrame,
    config: GapTargetConfig = GapTargetConfig(),
    costs: float = 0.002,
) -> pd.DataFrame:
    """
    Compute gap targets for all days in a dataframe.
    
    Returns DataFrame with columns for each target type.
    """
    n = len(daily_df)
    if n < 2:
        return pd.DataFrame()
    
    targets = {
        "gap_direction": np.zeros(n),
        "gap_magnitude": np.zeros(n),
        "gaps_up": np.zeros(n, dtype=bool),
        "gaps_down": np.zeros(n, dtype=bool),
        "any_gap": np.zeros(n, dtype=bool),
        "gap_fills": np.zeros(n, dtype=bool),
        "profitable_long": np.zeros(n, dtype=bool),
        "profitable_short": np.zeros(n, dtype=bool),
        "net_gap_long": np.zeros(n),
        "net_gap_short": np.zeros(n),
    }
    
    # Compute for each day (starting from day 1)
    for i in range(1, n):
        prev_close = daily_df["close"].iloc[i-1]
        today_open = daily_df["open"].iloc[i]
        today_high = daily_df["high"].iloc[i]
        today_low = daily_df["low"].iloc[i]
        today_close = daily_df["close"].iloc[i]
        
        if prev_close <= 0:
            continue
        
        gap = (today_open / prev_close) - 1
        
        # Direction
        if gap > config.direction_threshold:
            targets["gap_direction"][i] = 1
            targets["gaps_up"][i] = True
            targets["any_gap"][i] = True
        elif gap < -config.direction_threshold:
            targets["gap_direction"][i] = -1
            targets["gaps_down"][i] = True
            targets["any_gap"][i] = True
        
        targets["gap_magnitude"][i] = np.clip(gap, -config.magnitude_clip, config.magnitude_clip)
        
        # Fill check (using daily approximation)
        if targets["gaps_up"][i] and today_low <= prev_close:
            targets["gap_fills"][i] = True
        elif targets["gaps_down"][i] and today_high >= prev_close:
            targets["gap_fills"][i] = True
        
        # Profitability
        if targets["gaps_up"][i]:
            long_profit = (today_high / today_open - 1) - costs if today_open > 0 else -costs
            targets["profitable_long"][i] = long_profit > 0
        
        if targets["gaps_down"][i]:
            short_profit = (today_open / today_low - 1) - costs if today_low > 0 else -costs
            targets["profitable_short"][i] = short_profit > 0
        
        targets["net_gap_long"][i] = gap - costs
        targets["net_gap_short"][i] = abs(gap) - costs
    
    return pd.DataFrame(targets, index=daily_df.index)


def get_target_statistics(targets_df: pd.DataFrame) -> Dict:
    """Compute statistics about gap targets."""
    stats = {}
    
    n = len(targets_df)
    if n == 0:
        return stats
    
    stats["total_days"] = n
    stats["gap_up_pct"] = targets_df["gaps_up"].mean() * 100
    stats["gap_down_pct"] = targets_df["gaps_down"].mean() * 100
    stats["any_gap_pct"] = targets_df["any_gap"].mean() * 100
    stats["mean_gap_mag"] = targets_df["gap_magnitude"].abs().mean() * 100
    
    gap_up_days = targets_df[targets_df["gaps_up"]]
    gap_down_days = targets_df[targets_df["gaps_down"]]
    
    stats["gap_up_fill_pct"] = gap_up_days["gap_fills"].mean() * 100 if len(gap_up_days) > 0 else 0
    stats["gap_down_fill_pct"] = gap_down_days["gap_fills"].mean() * 100 if len(gap_down_days) > 0 else 0
    
    stats["profitable_long_pct"] = gap_up_days["profitable_long"].mean() * 100 if len(gap_up_days) > 0 else 0
    stats["profitable_short_pct"] = gap_down_days["profitable_short"].mean() * 100 if len(gap_down_days) > 0 else 0
    
    return stats


# Target names for model training
GAP_TARGET_NAMES = {
    "direction": "gap_direction",      # -1, 0, 1 classification
    "up": "gaps_up",                   # Binary: will it gap up?
    "down": "gaps_down",               # Binary: will it gap down?
    "magnitude": "gap_magnitude",      # Regression: signed gap size
    "fills": "gap_fills",              # Binary: will gap fill?
    "profitable_long": "profitable_long",    # Binary: profitable to buy gap?
    "profitable_short": "profitable_short",  # Binary: profitable to short gap?
}
