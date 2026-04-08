#!/usr/bin/env python3
"""
Quick Signal Test - Fast verification of a few key features.

Tests only the most theoretically sound features against gap targets
using a simplified approach for quick feedback.

Usage:
    python scripts/quick_signal_test.py --stock RELIANCE
    python scripts/quick_signal_test.py --universe nifty50 --max-stocks 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe


def compute_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute only 5 simple features for quick testing.
    These are the most likely to have predictive power.
    """
    features = pd.DataFrame(index=df.index)
    
    c = df["close"]
    v = df["volume"]
    
    # Feature 1: Previous day return
    features["prev_day_return"] = c.pct_change()
    
    # Feature 2: 5-day momentum
    features["momentum_5d"] = c.pct_change(5)
    
    # Feature 3: Volume trend
    features["volume_trend"] = v.rolling(5).mean() / v.rolling(20).mean() - 1
    
    # Feature 4: Volatility regime
    features["volatility"] = c.pct_change().rolling(20).std()
    
    # Feature 5: RSI proxy (simple)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_proxy"] = 100 - (100 / (1 + rs))
    
    return features


def compute_gap(daily_df: pd.DataFrame) -> pd.Series:
    """Compute next-day gap from daily data."""
    gaps = (daily_df["open"] / daily_df["close"].shift(1) - 1).shift(-1)
    return gaps


def test_stock(symbol: str, data_dir: Path):
    """Test features for a single stock."""
    csv_path = data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = df.columns.str.lower()
    except:
        return None
    
    # Resample to daily
    daily = df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    if len(daily) < 50:
        return None
    
    # Compute features and targets
    features = compute_simple_features(daily)
    features["next_day_gap"] = compute_gap(daily)
    
    # Drop NaN
    features = features.dropna()
    
    if len(features) < 30:
        return None
    
    # Compute correlations
    results = {}
    for col in ["prev_day_return", "momentum_5d", "volume_trend", "volatility", "rsi_proxy"]:
        corr, pvalue = spearmanr(features[col], features["next_day_gap"])
        results[col] = {"correlation": corr, "pvalue": pvalue}
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, help="Test single stock")
    parser.add_argument("--universe", type=str, default="nifty50")
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="nifty500")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.stock:
        symbols = [args.stock.upper()]
    else:
        symbols = get_universe(args.universe)[:args.max_stocks]
    
    print(f"Quick Signal Test - {len(symbols)} stocks")
    print("=" * 60)
    
    all_results = []
    for symbol in symbols:
        print(f"Testing {symbol}...", end=" ")
        result = test_stock(symbol, data_dir)
        if result:
            all_results.append((symbol, result))
            print("✓")
        else:
            print("✗")
    
    if not all_results:
        print("\nNo results. Check data files.")
        return
    
    # Aggregate
    print("\n" + "=" * 60)
    print("AVERAGE CORRELATIONS WITH NEXT-DAY GAP")
    print("=" * 60)
    
    feature_names = ["prev_day_return", "momentum_5d", "volume_trend", "volatility", "rsi_proxy"]
    
    for feature in feature_names:
        corrs = [r[1][feature]["correlation"] for r in all_results if feature in r[1]]
        if corrs:
            avg_corr = np.mean(corrs)
            print(f"{feature:<20}: {avg_corr:>8.4f} (n={len(corrs)})")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("- Correlation > 0.05: Weak but real signal")
    print("- Correlation > 0.10: Moderate signal")
    print("- Correlation > 0.15: Strong signal")
    print("=" * 60)


if __name__ == "__main__":
    main()
