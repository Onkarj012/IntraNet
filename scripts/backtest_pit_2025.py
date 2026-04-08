#!/usr/bin/env python3
"""
Realistic Backtesting - 2025 Blind Test with Proper Execution.

Uses:
- Point-in-time predictions (no lookahead)
- Realistic execution with slippage
- Multiple risk profiles
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.realistic_execution import RealisticExecution, ExecutionConfig


def load_price_data(symbol: str, date: str, data_dir: Path) -> pd.DataFrame:
    """Load minute data for a symbol on a specific date."""
    csv_path = data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = df.columns.str.lower()
        return df[df.index.date == pd.to_datetime(date).date()]
    except:
        return None


def backtest_risk_profile(predictions_df: pd.DataFrame, risk_config: Dict, 
                          output_file: str):
    """Backtest a single risk profile."""
    
    print("=" * 80)
    print(f"BACKTEST: {risk_config['name'].upper()}")
    print("=" * 80)
    print(f"Target: {risk_config['target_pct']*100:.1f}%")
    print(f"Stop: {risk_config['stop_loss_pct']*100:.1f}%")
    print(f"Position: ₹{risk_config['position_size']:,.0f}")
    print(f"Min confidence: {risk_config['min_confidence']*100:.2f}%")
    print("=" * 80)
    
    # Filter predictions by confidence
    pred_filtered = predictions_df[
        predictions_df["confidence"] >= risk_config["min_confidence"]
    ].copy()
    
    print(f"\nPredictions: {len(predictions_df)} → {len(pred_filtered)} after filter")
    
    # Initialize execution engine
    exec_config = ExecutionConfig()
    execution = RealisticExecution(exec_config)
    
    data_dir = Path("nifty500")
    trades = []
    
    print(f"\nSimulating {len(pred_filtered)} trades...")
    print("-" * 80)
    
    for idx, row in pred_filtered.iterrows():
        date = row["date"]
        symbol = row["symbol"]
        predicted_up = row["predicted_max_up"]
        
        # Determine direction
        if predicted_up > risk_config["min_confidence"]:
            direction = "LONG"
        elif predicted_up < -risk_config["min_confidence"]:
            direction = "SHORT"
        else:
            continue
        
        # Load price data
        day_data = load_price_data(symbol, date, data_dir)
        if day_data is None or len(day_data) < 30:
            continue
        
        entry_price = day_data["open"].iloc[0]
        if entry_price <= 0:
            continue
        
        # Calculate target/stop
        if direction == "LONG":
            target = entry_price * (1 + risk_config["target_pct"])
            stop = entry_price * (1 - risk_config["stop_loss_pct"])
        else:
            target = entry_price * (1 - risk_config["target_pct"])
            stop = entry_price * (1 + risk_config["stop_loss_pct"])
        
        # Execute trade
        result = execution.simulate_trade(
            day_data, direction, entry_price, target, stop,
            risk_config["position_size"],
            use_trailing=risk_config["use_trailing"],
            trailing_start=risk_config["trailing_start"],
            trailing_stop_pct=risk_config["trailing_stop_pct"]
        )
        
        trades.append({
            "date": date,
            "symbol": symbol,
            "direction": direction,
            "predicted_up": float(predicted_up),
            **result,
        })
        
        if len(trades) % 500 == 0:
            print(f"  Processed {len(trades)} trades...")
    
    if not trades:
        print("\n❌ No trades executed!")
        return
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    net_pnls = [t["net_pnl"] for t in trades]
    n_trades = len(trades)
    n_wins = sum(1 for p in net_pnls if p > 0)
    n_hits = sum(1 for t in trades if t["hit_target"])
    
    win_rate = n_wins / n_trades
    hit_rate = n_hits / n_trades
    total_net = sum(net_pnls)
    total_costs = sum(t["costs"] for t in trades)
    
    avg_win = np.mean([p for p in net_pnls if p > 0]) if n_wins > 0 else 0
    avg_loss = np.mean([p for p in net_pnls if p <= 0]) if n_trades - n_wins > 0 else 0
    
    # Daily metrics
    daily_df = pd.DataFrame(trades)
    daily_pnl = daily_df.groupby("date")["net_pnl"].sum()
    daily_returns = daily_pnl / risk_config["position_size"]
    
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    cumulative = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.min(cumulative - running_max)
    
    print(f"\n📊 Trading Statistics:")
    print(f"  Total Trades:       {n_trades}")
    print(f"  Win Rate:           {win_rate:.1%} ({n_wins}/{n_trades})")
    print(f"  Hit Rate:           {hit_rate:.1%} ({n_hits}/{n_trades})")
    
    print(f"\n💰 P&L Summary:")
    print(f"  Gross P&L:          ₹{sum(t['gross_pnl'] for t in trades):>12,.0f}")
    print(f"  Total Costs:        ₹{total_costs:>12,.0f}")
    print(f"  Net P&L:            ₹{total_net:>12,.0f}")
    print(f"  Avg per Trade:      ₹{total_net/n_trades:>12,.0f}")
    
    print(f"\n📈 Risk Metrics:")
    print(f"  Sharpe Ratio:       {sharpe:.3f}")
    print(f"  Max Drawdown:       ₹{max_dd:>12,.0f}")
    
    print(f"\n📉 Per Trade:")
    print(f"  Avg Win:            ₹{avg_win:>12,.0f}")
    print(f"  Avg Loss:           ₹{avg_loss:>12,.0f}")
    
    # Exit reasons
    reasons = daily_df["exit_reason"].value_counts()
    print(f"\n🎯 Exit Reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count} ({count/n_trades:.1%})")
    
    print(f"\n{'='*80}")
    
    # Save
    output = {
        "risk_profile": risk_config["name"],
        "summary": {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "hit_rate": hit_rate,
            "total_net": float(total_net),
            "total_costs": float(total_costs),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        },
        "trades": trades,
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"✓ Results saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default="predictions_2025_pit.csv")
    parser.add_argument("--risk", type=str, default="all", choices=["conservative", "balanced", "aggressive", "all"])
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions)
    print(f"✓ Loaded {len(pred_df)} predictions")
    
    # Risk profiles
    risk_profiles = {
        "conservative": {
            "name": "Conservative",
            "target_pct": 0.008,
            "stop_loss_pct": 0.005,
            "trailing_start": 0.005,
            "trailing_stop_pct": 0.003,
            "use_trailing": True,
            "position_size": 50000,
            "min_confidence": 0.008,
        },
        "balanced": {
            "name": "Balanced",
            "target_pct": 0.012,
            "stop_loss_pct": 0.008,
            "trailing_start": 0.008,
            "trailing_stop_pct": 0.005,
            "use_trailing": True,
            "position_size": 100000,
            "min_confidence": 0.010,
        },
        "aggressive": {
            "name": "Aggressive",
            "target_pct": 0.018,
            "stop_loss_pct": 0.012,
            "trailing_start": 0.012,
            "trailing_stop_pct": 0.008,
            "use_trailing": True,
            "position_size": 150000,
            "min_confidence": 0.012,
        },
    }
    
    # Run backtests
    if args.risk == "all":
        for risk_name, risk_config in risk_profiles.items():
            backtest_risk_profile(pred_df, risk_config, f"results_pit_{risk_name}.json")
            print("\n")
    else:
        backtest_risk_profile(pred_df, risk_profiles[args.risk], f"results_pit_{args.risk}.json")


if __name__ == "__main__":
    main()
