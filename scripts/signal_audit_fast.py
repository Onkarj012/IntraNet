#!/usr/bin/env python3
"""
Optimized Signal Audit for IntradayNet - Fast vectorized version.

Computes ICIR (Information Coefficient Information Ratio) for all features
against gap targets using efficient vectorized operations.

Usage:
    python scripts/signal_audit_fast.py --universe nifty100 --start-date 2021-01-01
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe


# Define the 36 lean features we want to test
LEAN_FEATURES = [
    # Price Action
    "overnight_gap", "prev_day_return", "prev_day_volatility", "price_vs_vwap",
    "volume_pace", "rsi_14", "bb_position", "day_trend_strength",
    # Market Context (will use mock data for now)
    "vix_level", "vix_change", "nifty_prev_return", "nifty_vs_sector",
    "market_breadth", "crude_change", "usdinr_change", "dxy_change",
    "us_10y_yield", "asia_overnight",
    # Sentiment (mock)
    "sentiment_5d_avg", "sentiment_spike", "sentiment_momentum",
    "premarket_sentiment", "news_volume", "sentiment_price_div",
    # Gap-specific
    "prev_gap_size", "prev_gap_filled", "earnings_proximity", "expiry_flag",
    # Microstructure
    "last_hour_trend", "close_vs_day_high", "volume_concentration",
    "spread_trend", "obv_slope", "vol_momentum", "momentum_5_20", "support_distance",
]


def compute_features_vectorized(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features vectorized for all days at once.
    Returns DataFrame with one row per day and all features.
    """
    # Resample to daily first
    daily = minute_df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    if len(daily) < 50:
        return pd.DataFrame()
    
    features = pd.DataFrame(index=daily.index)
    
    # === Price Action from Daily ===
    features["overnight_gap"] = daily["open"] / daily["close"].shift(1) - 1
    features["prev_day_return"] = daily["close"].pct_change()
    features["prev_day_volatility"] = features["prev_day_return"].rolling(21).std()
    
    # === Compute intraday features from minute data ===
    # Group minute data by date
    minute_df["date_only"] = minute_df.index.date
    
    # Pre-compute daily aggregates from minute data
    daily_intraday = minute_df.groupby("date_only").agg({
        "close": ["first", "last", "min", "max"],
        "volume": ["sum", lambda x: x.tail(30).sum() if len(x) >= 30 else x.sum()],  # Last 30 min volume
        "high": "max",
        "low": "min",
    })
    daily_intraday.columns = ["open_intraday", "close_intraday", "low_intraday", "high_intraday", 
                               "total_volume", "last_30min_volume", "high", "low"]
    daily_intraday.index = pd.to_datetime(daily_intraday.index)
    
    # Align with features index
    features = features.join(daily_intraday, how="left")
    
    # Price vs VWAP (simplified: close vs day's VWAP)
    # VWAP from minute data
    minute_df["tp"] = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3
    minute_df["tpv"] = minute_df["tp"] * minute_df["volume"]
    
    vwap_daily = minute_df.groupby("date_only").apply(
        lambda x: x["tpv"].sum() / x["volume"].sum() if x["volume"].sum() > 0 else x["close"].iloc[-1]
    )
    vwap_daily.index = pd.to_datetime(vwap_daily.index)
    features["vwap"] = vwap_daily
    features["price_vs_vwap"] = features["close_intraday"] / features["vwap"] - 1
    
    # Volume pace (last 30 min vs day average)
    features["volume_pace"] = features["last_30min_volume"] / (features["total_volume"] / 6) - 1  # Simplified
    
    # RSI (14-day)
    delta = daily["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Bollinger Band position
    sma20 = daily["close"].rolling(20).mean()
    std20 = daily["close"].rolling(20).std()
    features["bb_position"] = (daily["close"] - sma20) / (2 * std20.replace(0, np.nan))
    
    # Day trend strength (using daily close change as proxy)
    features["day_trend_strength"] = features["prev_day_return"]
    features["last_hour_trend"] = features["prev_day_return"]  # Simplified
    
    # Close vs day high
    features["close_vs_day_high"] = features["close_intraday"] / features["high_intraday"] - 1
    
    # Volume concentration (simplified)
    features["volume_concentration"] = 0.25  # Placeholder
    
    # OBV slope (simplified using daily)
    obv = (np.sign(daily["close"].diff()) * daily["volume"]).cumsum()
    features["obv_slope"] = obv.diff(5) / 5  # 5-day slope
    
    # Volume momentum
    features["vol_momentum"] = daily["volume"] / daily["volume"].rolling(20).mean() - 1
    
    # Momentum 5 vs 20
    mom5 = daily["close"].pct_change(5)
    mom20 = daily["close"].pct_change(20)
    features["momentum_5_20"] = mom5 - mom20
    
    # Support distance (simplified: distance from 20-day low)
    low_20 = daily["low"].rolling(20).min()
    features["support_distance"] = daily["close"] / low_20 - 1
    
    # === Gap-specific ===
    features["prev_gap_size"] = features["overnight_gap"].shift(1).abs()
    # Gap fill: did yesterday's gap fill?
    prev_gap = features["overnight_gap"].shift(1)
    prev_low = daily["low"].shift(1)
    prev_high = daily["high"].shift(1)
    prev_close = daily["close"].shift(2)
    
    gap_filled = pd.Series(0.5, index=features.index)
    up_gap = prev_gap > 0.002
    down_gap = prev_gap < -0.002
    gap_filled[up_gap] = (prev_low[up_gap] <= prev_close[up_gap]).astype(float)
    gap_filled[down_gap] = (prev_high[down_gap] >= prev_close[down_gap]).astype(float)
    features["prev_gap_filled"] = gap_filled
    
    features["earnings_proximity"] = 0  # Would need earnings calendar
    features["expiry_flag"] = 0  # Would need expiry calendar
    
    # === Mock Market Context (would need external data) ===
    for col in ["vix_level", "vix_change", "nifty_prev_return", "nifty_vs_sector",
                "market_breadth", "crude_change", "usdinr_change", "dxy_change",
                "us_10y_yield", "asia_overnight"]:
        features[col] = 0
    
    # === Mock Sentiment ===
    for col in ["sentiment_5d_avg", "sentiment_spike", "sentiment_momentum",
                "premarket_sentiment", "news_volume", "sentiment_price_div"]:
        features[col] = 0
    
    features["spread_trend"] = 0
    
    # Select only the lean features
    available_cols = [c for c in LEAN_FEATURES if c in features.columns]
    return features[available_cols]


def compute_gap_targets(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap targets from daily data."""
    targets = pd.DataFrame(index=daily_df.index)
    
    # Next day gap (shift -1 to predict future)
    targets["next_gap"] = (daily_df["open"] / daily_df["close"].shift(1) - 1).shift(-1)
    targets["gap_direction"] = np.sign(targets["next_gap"])
    targets["gap_up"] = (targets["next_gap"] > 0.002).astype(int)
    targets["gap_down"] = (targets["next_gap"] < -0.002).astype(int)
    targets["abs_gap"] = targets["next_gap"].abs()
    
    return targets


def compute_walk_forward_icir(features: pd.DataFrame, targets: pd.DataFrame, 
                               n_folds: int = 8) -> pd.DataFrame:
    """Compute ICIR for all feature-target pairs."""
    results = []
    
    # Combine and drop NaN
    combined = pd.concat([features, targets], axis=1).dropna()
    
    if len(combined) < n_folds * 20:
        return pd.DataFrame()
    
    feature_cols = [c for c in features.columns if c in combined.columns]
    target_cols = [c for c in targets.columns if c in combined.columns]
    
    fold_size = len(combined) // n_folds
    
    for target in target_cols:
        for feature in feature_cols:
            ics = []
            
            for i in range(n_folds):
                start = i * fold_size
                end = start + fold_size if i < n_folds - 1 else len(combined)
                
                fold = combined.iloc[start:end]
                if len(fold) < 10:
                    continue
                
                try:
                    ic, _ = spearmanr(fold[feature], fold[target])
                    if not np.isnan(ic):
                        ics.append(ic)
                except:
                    continue
            
            if len(ics) >= 3:
                mean_ic = np.mean(ics)
                std_ic = np.std(ics) if np.std(ics) > 0 else 1e-8
                icir = mean_ic / std_ic
                pct_pos = np.mean([ic > 0 for ic in ics])
                
                results.append({
                    "feature": feature,
                    "target": target,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "icir": icir,
                    "pct_positive": pct_pos,
                    "n_folds": len(ics),
                })
    
    return pd.DataFrame(results)


def audit_stock(symbol: str, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit a single stock - returns (features, results)."""
    csv_path = data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None, None
    
    try:
        minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        minute_df.columns = minute_df.columns.str.lower()
    except:
        return None, None
    
    # Compute features
    features = compute_features_vectorized(minute_df)
    if len(features) < 50:
        return None, None
    
    # Compute targets from daily
    daily = minute_df.resample("D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    targets = compute_gap_targets(daily)
    
    # Align features and targets
    common_index = features.index.intersection(targets.index)
    features = features.loc[common_index]
    targets = targets.loc[common_index]
    
    # Compute ICIR
    results = compute_walk_forward_icir(features, targets)
    
    if len(results) == 0:
        return None, None
    
    results["symbol"] = symbol
    return features, results


def main():
    parser = argparse.ArgumentParser(description="Fast Signal Audit")
    parser.add_argument("--universe", type=str, default="nifty100", 
                       choices=["nifty50", "nifty100", "nifty200"])
    parser.add_argument("--start-date", type=str, default="2021-01-01")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--min-icir", type=float, default=0.2)
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--output", type=str, default="signal_audit_results.json")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SIGNAL AUDIT - FAST VECTORIZED VERSION")
    print("=" * 80)
    print(f"Universe: {args.universe}")
    print(f"Start date: {args.start_date}")
    print(f"Folds: {args.n_folds}")
    print("=" * 80)
    
    symbols = get_universe(args.universe)
    if args.max_stocks > 0:
        symbols = symbols[:args.max_stocks]
    
    print(f"\nAnalyzing {len(symbols)} stocks...\n")
    
    data_dir = Path(args.data_dir)
    all_results = []
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        t0 = time.time()
        features, results = audit_stock(symbol, data_dir)
        t1 = time.time()
        
        if results is not None:
            all_results.append(results)
            print(f"[{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({len(features):4d} days, {t1-t0:.2f}s)")
        else:
            print(f"[{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ (no data)")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(symbols):.1f}s per stock)")
    
    if not all_results:
        print("\nNo results to analyze.")
        return
    
    # Aggregate
    all_df = pd.concat(all_results, ignore_index=True)
    
    # Group by feature-target
    aggregated = all_df.groupby(["feature", "target"]).agg({
        "mean_ic": "mean",
        "std_ic": "mean", 
        "icir": "mean",
        "pct_positive": "mean",
        "symbol": "count",
    }).reset_index().rename(columns={"symbol": "n_stocks"})
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS BY TARGET")
    print("=" * 80)
    
    for target in aggregated["target"].unique():
        target_df = aggregated[aggregated["target"] == target].sort_values("icir", ascending=False)
        
        print(f"\n--- {target} ---")
        print(f"{'Feature':<25} {'Mean IC':>10} {'ICIR':>10} {'% Pos':>8} {'Stocks':>8}")
        print("-" * 70)
        
        for _, row in target_df.head(10).iterrows():
            print(f"{row['feature']:<25} {row['mean_ic']:>10.4f} {row['icir']:>10.4f} "
                  f"{row['pct_positive']:>7.1%} {int(row['n_stocks']):>8}")
        
        strong = target_df[target_df["icir"] >= args.min_icir]
        print(f"\nStrong signals (ICIR >= {args.min_icir}): {len(strong)}/{len(target_df)}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    strong_signals = aggregated[aggregated["icir"] >= args.min_icir]
    print(f"Total feature-target pairs: {len(aggregated)}")
    print(f"Strong signals: {len(strong_signals)}")
    print(f"Success rate: {len(strong_signals)/len(aggregated)*100:.1f}%")
    
    if len(strong_signals) > 0:
        print("\nTop 10 strongest signals:")
        top = strong_signals.sort_values("icir", ascending=False).head(10)
        for _, row in top.iterrows():
            print(f"  {row['feature']} -> {row['target']}: ICIR={row['icir']:.3f}")
    
    # Save
    aggregated.to_json(args.output, orient="records", indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
