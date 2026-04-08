#!/usr/bin/env python3
"""
Paper Trading System with P&L Tracking.

Simulates 40+ trading sessions with full P&L tracking including:
- Entry/exit prices
- Transaction costs (NSE realistic)
- Daily and cumulative P&L
- Sharpe ratio, max drawdown, win rate
- Comparison to buy-and-hold

Usage:
    python scripts/paper_trade.py --model models/test_model.pkl --n-sessions 40
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder


# NSE realistic costs
COSTS = {
    "brokerage_per_order": 20,  # ₹20 per order (Zerodha style)
    "stt_pct": 0.00025,         # 0.025% on sell side (intraday)
    "exchange_txn_pct": 0.0000345,  # Exchange transaction fee
    "gst_brokerage_pct": 0.18,  # 18% GST on brokerage
    "stamp_duty_pct": 0.00003,  # Stamp duty
    "slippage_pct": 0.0005,     # 0.05% slippage per side
}


def compute_total_costs(position_value: float, is_intraday: bool = True) -> float:
    """Compute total round-trip costs in rupees."""
    # Brokerage (entry + exit)
    brokerage = COSTS["brokerage_per_order"] * 2
    
    # STT (only on sell side for intraday)
    stt = position_value * COSTS["stt_pct"]
    
    # Exchange transaction charges (both sides)
    exchange = position_value * COSTS["exchange_txn_pct"] * 2
    
    # GST on brokerage
    gst = brokerage * COSTS["gst_brokerage_pct"]
    
    # Stamp duty (buy side)
    stamp = position_value * COSTS["stamp_duty_pct"]
    
    # Slippage (both sides)
    slippage = position_value * COSTS["slippage_pct"] * 2
    
    return brokerage + stt + exchange + gst + stamp + slippage


@dataclass
class Trade:
    """Single trade record."""
    date: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    costs: float
    net_pnl: float
    exit_reason: str  # "target", "stop_loss", "eod"
    

@dataclass
class DailyResult:
    """Daily P&L summary."""
    date: str
    n_trades: int
    n_wins: int
    n_losses: int
    gross_pnl: float
    costs: float
    net_pnl: float
    cumulative_pnl: float
    trades: List[Trade]


class PaperTrader:
    """Paper trading system with realistic cost modeling."""
    
    def __init__(self, model_path: str, position_size: float = 100000):
        """Initialize with trained model."""
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.features = model_data["features"]
        self.position_size = position_size
        
        self.daily_results: List[DailyResult] = []
        self.cumulative_pnl = 0.0
        
    def compute_features(self, minute_df: pd.DataFrame, market_builder: MarketFeatureBuilder) -> pd.DataFrame:
        """Compute features for prediction."""
        daily = minute_df.resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        if len(daily) < 30:
            return None
        
        features = pd.DataFrame(index=daily.index)
        
        # Core features
        features["overnight_gap"] = daily["open"] / daily["close"].shift(1) - 1
        features["prev_day_volatility"] = daily["close"].pct_change().rolling(21).std()
        features["prev_gap_size"] = features["overnight_gap"].shift(1).abs()
        
        # Volume
        minute_df["date_only"] = minute_df.index.date
        vol_by_date = minute_df.groupby("date_only")["volume"].sum()
        vol_by_date.index = pd.to_datetime(vol_by_date.index)
        features["volume"] = vol_by_date.reindex(features.index)
        features["vol_momentum"] = features["volume"] / features["volume"].rolling(20).mean() - 1
        
        # VWAP
        minute_df["tp"] = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3
        minute_df["tpv"] = minute_df["tp"] * minute_df["volume"]
        vwap_daily = minute_df.groupby("date_only").apply(
            lambda x: x["tpv"].sum() / x["volume"].sum() if x["volume"].sum() > 0 else x["close"].iloc[-1],
            include_groups=False
        )
        vwap_daily.index = pd.to_datetime(vwap_daily.index)
        features["vwap"] = vwap_daily.reindex(features.index)
        features["price_vs_vwap"] = daily["close"] / features["vwap"] - 1
        
        # Close vs day high
        high_daily = minute_df.groupby("date_only")["high"].max()
        high_daily.index = pd.to_datetime(high_daily.index)
        features["day_high"] = high_daily.reindex(features.index)
        features["close_vs_day_high"] = daily["close"] / features["day_high"] - 1
        
        # Volume pace
        last_30_vol = minute_df.groupby("date_only").apply(
            lambda x: x["volume"].tail(30).sum(),
            include_groups=False
        )
        last_30_vol.index = pd.to_datetime(last_30_vol.index)
        features["last_30_vol"] = last_30_vol.reindex(features.index)
        features["volume_pace"] = features["last_30_vol"] / (features["volume"] / 6) - 1
        
        # Add market data
        market_features = market_builder.get_features(features.index)
        india_features = market_builder.get_india_market_features(features.index)
        
        for col in market_features.columns:
            features[col] = market_features[col]
        for key, series in india_features.items():
            features[key] = series
        
        # Select only model features
        available = [f for f in self.features if f in features.columns]
        return features[available]
    
    def simulate_day(self, symbol: str, date: str, minute_df: pd.DataFrame, 
                    prediction: float) -> List[Trade]:
        """Simulate trading for one day."""
        trades = []
        
        # Filter to trading day
        day_data = minute_df[minute_df.index.date == pd.to_datetime(date).date()]
        if len(day_data) < 30:
            return trades
        
        # Entry at open
        entry_price = day_data["open"].iloc[0]
        if entry_price <= 0:
            return trades
        
        # Calculate quantity
        quantity = int(self.position_size / entry_price)
        if quantity == 0:
            return trades
        
        # Direction based on prediction (> 0.5 = LONG)
        direction = "LONG" if prediction > 0.5 else "SHORT"
        
        # Simulate exit
        day_high = day_data["high"].max()
        day_low = day_data["low"].min()
        day_close = day_data["close"].iloc[-1]
        
        # Simple exit: hold until end of day
        exit_price = day_close
        exit_reason = "eod"
        
        # Calculate P&L
        if direction == "LONG":
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity
        
        # Costs
        position_value = entry_price * quantity
        costs = compute_total_costs(position_value)
        net_pnl = gross_pnl - costs
        
        trade = Trade(
            date=date,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=gross_pnl,
            costs=costs,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
        )
        trades.append(trade)
        
        return trades
    
    def run(self, universe: str = "nifty50", n_sessions: int = 40, 
            start_date: str = "2024-01-01"):
        """Run paper trading simulation."""
        print("=" * 80)
        print("PAPER TRADING SIMULATION")
        print("=" * 80)
        print(f"Universe: {universe}")
        print(f"Sessions: {n_sessions}")
        print(f"Position size: ₹{self.position_size:,.0f}")
        print("=" * 80)
        
        # Load market data
        print("\nLoading market data...")
        market_builder = MarketFeatureBuilder()
        market_builder.download(start=start_date)
        print("✓ Market data ready")
        
        symbols = get_universe(universe)
        data_dir = Path("nifty500")
        
        # Collect predictions for each day
        all_predictions = []
        
        print(f"\nSimulating {n_sessions} trading sessions...")
        
        for session_num in range(n_sessions):
            session_date = (pd.to_datetime(start_date) + timedelta(days=session_num)).strftime("%Y-%m-%d")
            
            daily_trades = []
            daily_gross = 0.0
            daily_costs = 0.0
            daily_net = 0.0
            n_wins = 0
            n_losses = 0
            
            for symbol in symbols[:20]:  # Top 20 for speed
                csv_path = data_dir / f"{symbol}_minute.csv"
                if not csv_path.exists():
                    continue
                
                try:
                    minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
                    minute_df.columns = minute_df.columns.str.lower()
                    
                    # Check if we have data for this date
                    day_data = minute_df[minute_df.index.date == pd.to_datetime(session_date).date()]
                    if len(day_data) < 30:
                        continue
                    
                    # Compute features
                    features = self.compute_features(minute_df, market_builder)
                    if features is None:
                        continue
                    
                    # Get feature row for prediction (yesterday's data)
                    pred_date = pd.to_datetime(session_date) - timedelta(days=1)
                    if pred_date not in features.index:
                        continue
                    
                    feature_row = features.loc[[pred_date]]
                    
                    # Predict
                    pred_proba = self.model.predict_proba(feature_row)[0, 1]
                    
                    # Only trade if confident (> 0.55 or < 0.45)
                    if 0.45 <= pred_proba <= 0.55:
                        continue
                    
                    # Simulate trade
                    trades = self.simulate_day(symbol, session_date, minute_df, pred_proba)
                    
                    for trade in trades:
                        daily_trades.append(trade)
                        daily_gross += trade.gross_pnl
                        daily_costs += trade.costs
                        daily_net += trade.net_pnl
                        
                        if trade.net_pnl > 0:
                            n_wins += 1
                        else:
                            n_losses += 1
                    
                except Exception as e:
                    continue
            
            # Update cumulative
            self.cumulative_pnl += daily_net
            
            # Record daily result
            daily_result = DailyResult(
                date=session_date,
                n_trades=len(daily_trades),
                n_wins=n_wins,
                n_losses=n_losses,
                gross_pnl=daily_gross,
                costs=daily_costs,
                net_pnl=daily_net,
                cumulative_pnl=self.cumulative_pnl,
                trades=daily_trades,
            )
            self.daily_results.append(daily_result)
            
            print(f"  {session_date}: {len(daily_trades):2d} trades, "
                  f"P&L: ₹{daily_net:>8,.0f}, Cum: ₹{self.cumulative_pnl:>10,.0f}")
        
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
    
    def get_stats(self) -> Dict:
        """Compute trading statistics."""
        if not self.daily_results:
            return {}
        
        all_trades = []
        for day in self.daily_results:
            all_trades.extend(day.trades)
        
        if not all_trades:
            return {}
        
        net_pnls = [t.net_pnl for t in all_trades]
        daily_pnls = [day.net_pnl for day in self.daily_results]
        
        n_trades = len(all_trades)
        n_wins = sum(1 for p in net_pnls if p > 0)
        n_losses = n_trades - n_wins
        
        win_rate = n_wins / n_trades if n_trades > 0 else 0
        
        avg_win = np.mean([p for p in net_pnls if p > 0]) if n_wins > 0 else 0
        avg_loss = np.mean([p for p in net_pnls if p < 0]) if n_losses > 0 else 0
        
        # Sharpe (assuming 252 trading days)
        daily_returns = np.array(daily_pnls) / self.position_size
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Max drawdown
        cumulative = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown)
        
        return {
            "n_sessions": len(self.daily_results),
            "n_trades": n_trades,
            "n_wins": n_wins,
            "n_losses": n_losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_gross": sum(t.gross_pnl for t in all_trades),
            "total_costs": sum(t.costs for t in all_trades),
            "total_net": sum(net_pnls),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
    
    def print_report(self):
        """Print detailed trading report."""
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("PAPER TRADING PERFORMANCE REPORT")
        print("=" * 80)
        print(f"\nTrading Statistics:")
        print(f"  Sessions:           {stats['n_sessions']}")
        print(f"  Total Trades:       {stats['n_trades']}")
        print(f"  Win Rate:           {stats['win_rate']:.1%} ({stats['n_wins']}/{stats['n_trades']})")
        print(f"\nP&L Summary:")
        print(f"  Gross P&L:          ₹{stats['total_gross']:>12,.0f}")
        print(f"  Total Costs:        ₹{stats['total_costs']:>12,.0f}")
        print(f"  Net P&L:            ₹{stats['total_net']:>12,.0f}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:       {stats['sharpe']:.3f}")
        print(f"  Max Drawdown:       ₹{stats['max_drawdown']:>12,.0f}")
        print(f"\nPer Trade:")
        print(f"  Avg Win:            ₹{stats['avg_win']:>12,.0f}")
        print(f"  Avg Loss:           ₹{stats['avg_loss']:>12,.0f}")
        print(f"  Profit Factor:      {abs(stats['avg_win'] * stats['n_wins'] / (stats['avg_loss'] * stats['n_losses'])):.2f}" 
              if stats['avg_loss'] != 0 else "  N/A")
        print("=" * 80)
        
        # Save to file
        output = {
            "stats": stats,
            "daily_results": [
                {
                    "date": day.date,
                    "n_trades": day.n_trades,
                    "net_pnl": day.net_pnl,
                    "cumulative_pnl": day.cumulative_pnl,
                }
                for day in self.daily_results
            ],
            "trades": [
                asdict(trade)
                for day in self.daily_results
                for trade in day.trades
            ],
        }
        
        with open("paper_trade_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print("\nDetailed results saved to: paper_trade_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/test_model.pkl")
    parser.add_argument("--universe", type=str, default="nifty50")
    parser.add_argument("--n-sessions", type=int, default=40)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--position-size", type=float, default=100000)
    
    args = parser.parse_args()
    
    trader = PaperTrader(args.model, args.position_size)
    trader.run(args.universe, args.n_sessions, args.start_date)
    trader.print_report()


if __name__ == "__main__":
    main()
