#!/usr/bin/env python3
"""
Enhanced Signal Audit with Real Market Data.

Integrates VIX, crude, USD/INR, global markets into feature computation.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder


def compute_features_with_market_data(minute_df: pd.DataFrame, market_builder: MarketFeatureBuilder) -> pd.DataFrame:
    """Compute features with real market data."""
    # Resample to daily
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
    
    # Price features
    features["overnight_gap"] = daily["open"] / daily["close"].shift(1) - 1
    features["prev_day_return"] = daily["close"].pct_change()
    features["prev_day_volatility"] = features["prev_day_return"].rolling(21).std()
    features["prev_gap_size"] = features["overnight_gap"].shift(1).abs()
    
    # Volume features
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
    
    # === REAL MARKET DATA ===
    market_features = market_builder.get_features(features.index)
    india_features = market_builder.get_india_market_features(features.index)
    
    # Add market features
    for col in market_features.columns:
        features[col] = market_features[col]
    
    for key, series in india_features.items():
        features[key] = series
    
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="nifty50")
    parser.add_argument("--max-stocks", type=int, default=10)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SIGNAL AUDIT WITH REAL MARKET DATA")
    print("=" * 80)
    
    # Initialize market data builder
    print("\nDownloading market data...")
    market_builder = MarketFeatureBuilder(cache_dir="market_data_cache")
    market_builder.download(start="2022-01-01")
    print("✓ Market data loaded")
    
    symbols = get_universe(args.universe)[:args.max_stocks]
    data_dir = Path("nifty500")
    
    print(f"\nAnalyzing {len(symbols)} stocks with market data...\n")
    
    all_ic_results = []
    
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        
        try:
            minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            minute_df.columns = minute_df.columns.str.lower()
            minute_df = minute_df[minute_df.index >= "2022-01-01"]
            
            features = compute_features_with_market_data(minute_df, market_builder)
            if features is None or len(features) < 50:
                continue
            
            # Target: next day gap
            daily = minute_df.resample("D").agg({"open": "first", "close": "last"}).dropna()
            next_gap = (daily["open"] / daily["close"].shift(1) - 1).shift(-1)
            
            # Align
            valid_idx = features.dropna().index.intersection(next_gap.dropna().index)
            features = features.loc[valid_idx]
            target = next_gap.loc[valid_idx]
            
            # Compute IC for each feature
            for col in features.columns:
                if col in ["volume", "vwap", "day_high", "last_30_vol"]:
                    continue
                
                ic, _ = spearmanr(features[col], target)
                if not np.isnan(ic):
                    all_ic_results.append({
                        "symbol": symbol,
                        "feature": col,
                        "ic": ic,
                    })
            
            print(f"  {symbol}: ✓ ({len(features)} days)")
            
        except Exception as e:
            print(f"  {symbol}: ✗ ({str(e)[:30]})")
    
    # Aggregate results
    if all_ic_results:
        results_df = pd.DataFrame(all_ic_results)
        summary = results_df.groupby("feature")["ic"].agg(["mean", "std", "count"]).reset_index()
        summary["icir"] = summary["mean"] / summary["std"].replace(0, 1e-8)
        summary = summary.sort_values("icir", ascending=False)
        
        print("\n" + "=" * 80)
        print("TOP FEATURES WITH MARKET DATA (by ICIR)")
        print("=" * 80)
        print(f"{'Feature':<30} {'Mean IC':>10} {'ICIR':>10} {'Stocks':>8}")
        print("-" * 80)
        
        for _, row in summary.head(15).iterrows():
            print(f"{row['feature']:<30} {row['mean']:>10.4f} {row['icir']:>10.4f} {int(row['count']):>8}")
        
        # Highlight market data features
        market_feats = ["vix_level", "vix_change", "crude_oil_return", "usdinr_change", 
                       "dxy_change", "asia_sentiment", "global_volatility_regime"]
        
        print("\n" + "=" * 80)
        print("MARKET DATA FEATURES PERFORMANCE")
        print("=" * 80)
        
        market_results = summary[summary["feature"].isin(market_feats)]
        for _, row in market_results.iterrows():
            print(f"{row['feature']:<30} {row['mean']:>10.4f} {row['icir']:>10.4f}")


if __name__ == "__main__":
    main()
