#!/usr/bin/env python3
"""
Fast Backtester - Optimized for large prediction files.

Processes predictions in batches for speed.
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import multiprocessing as mp

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


def load_and_cache_data(symbols: List[str], dates: List[str], data_dir: Path) -> Dict:
    """Pre-load all required price data into memory."""
    cached_data = {}
    
    print(f"Pre-loading price data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            df.columns = df.columns.str.lower()
            
            # Filter to relevant dates only
            date_filter = df.index.normalize().isin(pd.to_datetime(dates))
            df = df[date_filter]
            
            if len(df) > 0:
                cached_data[symbol] = df
        except:
            continue
    
    print(f"✓ Loaded data for {len(cached_data)} symbols")
    return cached_data


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
        
        if direction == "LONG":
            current_pnl = (high - entry_price) / entry_price
            current_dd = (low - entry_price) / entry_price
            if high >= target:
                hit_target_level = True
        else:
            current_pnl = (entry_price - low) / entry_price
            current_dd = (entry_price - high) / entry_price
            if low <= target:
                hit_target_level = True
        
        max_favorable = max(max_favorable, current_pnl)
        max_adverse = min(max_adverse, current_dd)
        
        if trailing_stop is None and abs(current_pnl) >= config.trailing_start:
            if direction == "LONG":
                trailing_stop = high * (1 - config.trailing_stop_pct)
            else:
                trailing_stop = low * (1 + config.trailing_stop_pct)
        
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
        
        if i >= exit_3pm_idx:
            exit_price = closes[i]
            exit_time = i
            exit_reason = "3PM"
            break
    
    if direction == "LONG":
        gross_pct = (exit_price - entry_price) / entry_price
    else:
        gross_pct = (entry_price - exit_price) / entry_price
    
    gross_pnl = gross_pct * config.position_size
    costs = COST_PER_TRADE * (config.position_size / 100000)
    net_pnl = gross_pnl - costs
    
    return {
        "hit_target": hit_target_level,
        "exit_reason": exit_reason,
        "gross_pct": gross_pct,
        "net_pnl": net_pnl,
        "max_favorable_pct": max_favorable,
        "max_adverse_pct": max_adverse,
        "exit_price": exit_price,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--risk", type=str, default="balanced", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--output", type=str, default="backtest_results.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of predictions (0 = all)")
    
    args = parser.parse_args()
    
    config = RISK_PROFILES[args.risk]
    
    print("=" * 80)
    print(f"FAST BACKTEST: {config.name.upper()}")
    print("=" * 80)
    
    # Load predictions
    print("\nLoading predictions...")
    pred_df = pd.read_csv(args.predictions)
    
    if args.limit > 0:
        pred_df = pred_df.head(args.limit)
    
    print(f"✓ Loaded {len(pred_df)} predictions")
    
    # Filter by confidence
    pred_df = pred_df[pred_df["confidence"] >= config.min_confidence]
    print(f"  After confidence filter: {len(pred_df)}")
    
    # Pre-load price data
    symbols = pred_df["symbol"].unique().tolist()
    dates = pred_df["date"].unique().tolist()
    data_dir = Path("nifty500")
    
    cached_data = load_and_cache_data(symbols, dates, data_dir)
    
    # Backtest
    print(f"\nSimulating {len(pred_df)} trades...")
    trades = []
    
    for idx, row in pred_df.iterrows():
        date = row["date"]
        symbol = row["symbol"]
        predicted_return = row["predicted_return"]
        direction = row["direction"]
        
        if symbol not in cached_data:
            continue
        
        day_data = cached_data[symbol][cached_data[symbol].index.date == pd.to_datetime(date).date()]
        
        if len(day_data) < 30:
            continue
        
        entry_price = day_data["open"].iloc[0]
        if entry_price <= 0:
            continue
        
        result = simulate_trade(day_data, direction, entry_price, config)
        
        trades.append({
            "date": date,
            "symbol": symbol,
            "direction": direction,
            "predicted_return": float(predicted_return),
            "entry_price": float(entry_price),
            **result,
        })
        
        if len(trades) % 500 == 0:
            print(f"  Processed {len(trades)} trades...")
    
    if not trades:
        print("\n❌ No trades!")
        return
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    net_pnls = [t["net_pnl"] for t in trades]
    n_trades = len(trades)
    n_wins = sum(1 for p in net_pnls if p > 0)
    n_hit_target = sum(1 for t in trades if t["hit_target"])
    
    win_rate = n_wins / n_trades
    hit_rate = n_hit_target / n_trades
    total_net = sum(net_pnls)
    
    avg_win = np.mean([p for p in net_pnls if p > 0]) if n_wins > 0 else 0
    avg_loss = np.mean([p for p in net_pnls if p <= 0]) if n_trades - n_wins > 0 else 0
    
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
    print(f"  Total Net P&L:      ₹{total_net:>12,.0f}")
    print(f"  Avg per Trade:      ₹{total_net/n_trades:>12,.0f}")
    
    print(f"\n📈 Risk Metrics:")
    print(f"  Sharpe Ratio:       {sharpe:.3f}")
    print(f"  Max Drawdown:       ₹{max_dd:>12,.0f}")
    
    print(f"\n📉 Per Trade:")
    print(f"  Avg Win:            ₹{avg_win:>12,.0f}")
    print(f"  Avg Loss:           ₹{avg_loss:>12,.0f}")
    
    # By exit reason
    reasons = pd.DataFrame(trades)["exit_reason"].value_counts()
    print(f"\n🎯 Exit Reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count} ({count/len(trades):.1%})")
    
    print(f"\n{'='*80}")
    
    # Save
    output = {
        "risk_profile": config.name,
        "summary": {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "hit_rate": hit_rate,
            "total_net": float(total_net),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        },
        "trades": trades,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"✓ Results saved: {args.output}")


if __name__ == "__main__":
    main()
