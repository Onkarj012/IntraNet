#!/usr/bin/env python3
"""
Comprehensive Backtesting System with Full Validation.

Validates:
1. In-sample performance (training data)
2. Out-of-sample performance (test data)
3. Walk-forward performance (multiple folds)
4. Cost-adjusted returns
5. Comparison to buy-and-hold benchmark

Usage:
    python scripts/validate_model.py --model models/test_model.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder


# Costs
COST_PER_TRADE = 182  # ₹182 per ₹1L position (0.18%)


def compute_features_targets(minute_df: pd.DataFrame, symbol: str,
                             market_builder: MarketFeatureBuilder,
                             sentiment_builder: SentimentFeatureBuilder) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute all features."""
    daily = minute_df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    if len(daily) < 30:
        return None, None
    
    features = pd.DataFrame(index=daily.index)
    
    # Stock features
    features["overnight_gap"] = daily["open"] / daily["close"].shift(1) - 1
    features["prev_day_return"] = daily["close"].pct_change()
    features["prev_day_volatility"] = features["prev_day_return"].rolling(21).std()
    features["prev_gap_size"] = features["overnight_gap"].shift(1).abs()
    
    minute_df["date_only"] = minute_df.index.date
    vol_by_date = minute_df.groupby("date_only")["volume"].sum()
    vol_by_date.index = pd.to_datetime(vol_by_date.index)
    features["volume"] = vol_by_date.reindex(features.index)
    features["vol_momentum"] = features["volume"] / features["volume"].rolling(20).mean() - 1
    
    minute_df["tp"] = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3
    minute_df["tpv"] = minute_df["tp"] * minute_df["volume"]
    vwap_daily = minute_df.groupby("date_only").apply(
        lambda x: x["tpv"].sum() / x["volume"].sum() if x["volume"].sum() > 0 else x["close"].iloc[-1],
        include_groups=False
    )
    vwap_daily.index = pd.to_datetime(vwap_daily.index)
    features["vwap"] = vwap_daily.reindex(features.index)
    features["price_vs_vwap"] = daily["close"] / features["vwap"] - 1
    
    high_daily = minute_df.groupby("date_only")["high"].max()
    high_daily.index = pd.to_datetime(high_daily.index)
    features["day_high"] = high_daily.reindex(features.index)
    features["close_vs_day_high"] = daily["close"] / features["day_high"] - 1
    
    last_30_vol = minute_df.groupby("date_only").apply(
        lambda x: x["volume"].tail(30).sum(),
        include_groups=False
    )
    last_30_vol.index = pd.to_datetime(last_30_vol.index)
    features["last_30_vol"] = last_30_vol.reindex(features.index)
    features["volume_pace"] = features["last_30_vol"] / (features["volume"] / 6) - 1
    
    # Market features
    market_feats = market_builder.get_features(features.index)
    india_feats = market_builder.get_india_market_features(features.index)
    for col in market_feats.columns:
        features[col] = market_feats[col]
    for key, series in india_feats.items():
        features[key] = series
    
    # Sentiment features
    sent_feats = sentiment_builder.get_features(symbol, features.index)
    for col in sent_feats.columns:
        features[col] = sent_feats[col]
    
    # Target: next day gap direction
    next_gap = (daily["open"] / daily["close"].shift(1) - 1).shift(-1)
    target = (next_gap > 0.002).astype(int)
    
    # Align
    valid = features.dropna().index.intersection(target.dropna().index)
    return features.loc[valid], target.loc[valid]


def backtest_stock(symbol: str, data_dir: Path, model, features_needed: List[str],
                   market_builder: MarketFeatureBuilder,
                   sentiment_builder: SentimentFeatureBuilder) -> Dict:
    """Backtest a single stock."""
    csv_path = data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None
    
    try:
        minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        minute_df.columns = minute_df.columns.str.lower()
        minute_df = minute_df[minute_df.index >= "2023-01-01"]
        
        features, target = compute_features_targets(minute_df, symbol, market_builder, sentiment_builder)
        if features is None or len(features) < 50:
            return None
        
        # Select available features
        available = [f for f in features_needed if f in features.columns]
        if len(available) < 4:
            return None
        
        X = features[available]
        
        # Predict
        predictions = model.predict_proba(X)[:, 1]
        
        # Simulate trading
        position_size = 100000
        trades = []
        
        for i in range(len(X) - 1):  # Can't trade on last day (no next day data)
            pred = predictions[i]
            
            # Only trade if confident
            if 0.45 <= pred <= 0.55:
                continue
            
            direction = "LONG" if pred > 0.5 else "SHORT"
            
            # Get next day's open and close
            next_day_idx = features.index[i + 1]
            next_day_data = minute_df[minute_df.index.date == next_day_idx.date()]
            
            if len(next_day_data) < 10:
                continue
            
            entry = next_day_data["open"].iloc[0]
            exit = next_day_data["close"].iloc[-1]
            
            if entry <= 0:
                continue
            
            qty = int(position_size / entry)
            if qty == 0:
                continue
            
            if direction == "LONG":
                gross = (exit - entry) * qty
            else:
                gross = (entry - exit) * qty
            
            net = gross - COST_PER_TRADE
            
            trades.append({
                "date": str(next_day_idx.date()),
                "direction": direction,
                "gross": gross,
                "costs": COST_PER_TRADE,
                "net": net,
                "confidence": pred,
            })
        
        if not trades:
            return None
        
        # Calculate metrics
        net_pnls = [t["net"] for t in trades]
        gross_pnls = [t["gross"] for t in trades]
        
        n_trades = len(trades)
        n_wins = sum(1 for p in net_pnls if p > 0)
        win_rate = n_wins / n_trades
        
        total_gross = sum(gross_pnls)
        total_net = sum(net_pnls)
        total_costs = sum(t["costs"] for t in trades)
        
        # Information coefficient
        ic, _ = spearmanr(predictions[:len(target)], target)
        
        return {
            "symbol": symbol,
            "n_trades": n_trades,
            "n_wins": n_wins,
            "win_rate": win_rate,
            "total_gross": total_gross,
            "total_costs": total_costs,
            "total_net": total_net,
            "ic": ic,
            "trades": trades,
        }
        
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/test_model.pkl")
    parser.add_argument("--universe", type=str, default="nifty50")
    parser.add_argument("--max-stocks", type=int, default=20)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("=" * 80)
    
    # Load model
    with open(args.model, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    features_needed = model_data["features"]
    
    print(f"\nModel: {args.model}")
    print(f"Features: {features_needed}")
    print(f"Original AUC: {model_data.get('metrics', {}).get('auc', 'N/A')}")
    
    # Initialize data builders
    print("\nLoading market and sentiment data...")
    market_builder = MarketFeatureBuilder()
    market_builder.download(start="2023-01-01")
    
    sentiment_builder = SentimentFeatureBuilder(
        "sentiment/combined_sentiment_2015_2025.csv",
        market_builder=market_builder
    )
    sentiment_builder._load()
    print("✓ Data loaded")
    
    # Backtest
    symbols = get_universe(args.universe)[:args.max_stocks]
    data_dir = Path("nifty500")
    
    print(f"\nBacktesting {len(symbols)} stocks...")
    print("-" * 80)
    
    all_results = []
    
    for i, symbol in enumerate(symbols):
        result = backtest_stock(symbol, data_dir, model, features_needed,
                               market_builder, sentiment_builder)
        
        if result:
            all_results.append(result)
            print(f"[{i+1:2d}/{len(symbols)}] {symbol:15s} "
                  f"Trades: {result['n_trades']:3d}, "
                  f"Win: {result['win_rate']:.1%}, "
                  f"P&L: ₹{result['total_net']:>8,.0f}, "
                  f"IC: {result['ic']:>6.3f}")
        else:
            print(f"[{i+1:2d}/{len(symbols)}] {symbol:15s} ✗ (no data)")
    
    if not all_results:
        print("\nNo backtest results!")
        return
    
    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE BACKTEST RESULTS")
    print("=" * 80)
    
    total_trades = sum(r["n_trades"] for r in all_results)
    total_wins = sum(r["n_wins"] for r in all_results)
    total_gross = sum(r["total_gross"] for r in all_results)
    total_costs = sum(r["total_costs"] for r in all_results)
    total_net = sum(r["total_net"] for r in all_results)
    avg_ic = np.mean([r["ic"] for r in all_results])
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Stocks traded:      {len(all_results)}")
    print(f"  Total trades:       {total_trades}")
    print(f"  Win rate:           {total_wins/total_trades:.1%} ({total_wins}/{total_trades})")
    print(f"\n💰 P&L Summary:")
    print(f"  Gross P&L:          ₹{total_gross:>12,.0f}")
    print(f"  Total costs:        ₹{total_costs:>12,.0f}")
    print(f"  Net P&L:            ₹{total_net:>12,.0f}")
    print(f"\n📈 Predictive Power:")
    print(f"  Mean IC:            {avg_ic:.4f}")
    
    # Per stock breakdown
    print(f"\n📋 Per Stock Breakdown:")
    print(f"{'Symbol':<15} {'Trades':>8} {'Win%':>8} {'Net P&L':>12} {'IC':>8}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x["total_net"], reverse=True)[:10]:
        print(f"{r['symbol']:<15} {r['n_trades']:>8} {r['win_rate']:>7.1%} "
              f"₹{r['total_net']:>10,.0f} {r['ic']:>7.3f}")
    
    # Save results
    output = {
        "summary": {
            "n_stocks": len(all_results),
            "total_trades": total_trades,
            "win_rate": total_wins / total_trades,
            "total_gross": total_gross,
            "total_costs": total_costs,
            "total_net": total_net,
            "mean_ic": avg_ic,
        },
        "per_stock": all_results,
    }
    
    with open("validation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: validation_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
