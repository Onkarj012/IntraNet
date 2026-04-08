#!/usr/bin/env python3
"""
Signal Audit for IntradayNet - Phase 1 of Rebuild.

Tests every feature's predictive power using walk-forward ICIR analysis.
Computes Information Coefficient (Spearman correlation) for each feature
against multiple target types.

Usage:
    python scripts/signal_audit.py --universe nifty100 --start-date 2021-01-01
    python scripts/signal_audit.py --universe nifty50 --n-folds 10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.lean_features import LEAN_FEATURE_NAMES, compute_lean_features
from intradaynet.gap_targets import compute_gap_target_series, GAP_TARGET_NAMES


@dataclass
class ICIRResult:
    """Result of ICIR analysis for a single feature-target pair."""
    feature: str
    target: str
    mean_ic: float
    std_ic: float
    icir: float
    pct_positive: float
    n_folds: int


def compute_walk_forward_icir(
    feature_series: pd.Series,
    target_series: pd.Series,
    n_folds: int = 8,
) -> Tuple[float, float, float, float]:
    """
    Compute walk-forward Information Coefficient Information Ratio (ICIR).
    
    ICIR = mean(IC) / std(IC) across walk-forward folds
    
    Args:
        feature_series: Feature values (should be shift(1) to be predictive)
        target_series: Target values to predict
        n_folds: Number of walk-forward folds
        
    Returns:
        (mean_ic, std_ic, icir, pct_positive)
    """
    # Ensure aligned and no NaN
    df = pd.DataFrame({"feature": feature_series, "target": target_series}).dropna()
    
    if len(df) < n_folds * 10:  # Need enough data
        return 0, 1, 0, 0.5
    
    ics = []
    fold_size = len(df) // n_folds
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(df)
        
        fold = df.iloc[start_idx:end_idx]
        if len(fold) < 10:
            continue
        
        # Spearman rank correlation (Information Coefficient)
        try:
            ic, _ = spearmanr(fold["feature"], fold["target"])
            if not np.isnan(ic):
                ics.append(ic)
        except:
            continue
    
    if len(ics) < 3:
        return 0, 1, 0, 0.5
    
    mean_ic = np.mean(ics)
    std_ic = np.std(ics) if np.std(ics) > 0 else 1e-8
    icir = mean_ic / std_ic
    pct_positive = np.mean([ic > 0 for ic in ics])
    
    return mean_ic, std_ic, icir, pct_positive


def audit_single_stock(
    symbol: str,
    data_dir: Path,
    n_folds: int = 8,
    min_samples: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Audit all features for a single stock.
    
    Returns DataFrame with ICIR results for each feature-target pair.
    """
    # Load minute data
    minute_csv = data_dir / f"{symbol}_minute.csv"
    if not minute_csv.exists():
        return None
    
    try:
        minute_df = pd.read_csv(minute_csv, parse_dates=["date"], index_col="date")
        minute_df.columns = minute_df.columns.str.lower()
    except Exception as e:
        print(f"  Error loading {symbol}: {e}")
        return None
    
    if len(minute_df) < min_samples:
        return None
    
    # Build daily data from minute data
    minute_df["date_only"] = minute_df.index.date
    daily = minute_df.groupby("date_only").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    
    if len(daily) < min_samples // 10:  # Rough estimate
        return None
    
    # Compute gap targets
    from intradaynet.gap_targets import GapTargetConfig
    targets_df = compute_gap_target_series(daily, config=GapTargetConfig())
    
    if len(targets_df) < 50:
        return None
    
    # For each day, compute features (using previous day's data)
    features_list = []
    targets_list = []
    
    for i in range(1, len(daily)):
        prev_date = daily.index[i-1]
        curr_date = daily.index[i]
        
        # Get yesterday's minute data (last 120 bars)
        prev_minute = minute_df[minute_df.index.date == prev_date.date()]
        if len(prev_minute) == 0:
            continue
        
        # Get yesterday's daily context
        prev_daily = daily.iloc[:i]
        
        # Mock market and sentiment data (will be replaced with real data)
        market_data = {
            "vix_level": 18,
            "vix_change": 0,
            "nifty_prev_return": 0,
            "nifty_vs_sector": 0,
            "market_breadth": 0.5,
            "crude_change": 0,
            "usdinr_change": 0,
            "dxy_change": 0,
            "us_10y_yield": 0,
            "asia_overnight": 0,
            "is_expiry_day": 0,
        }
        sentiment_data = {
            "sentiment_5d_avg": 0,
            "sentiment_spike": 0,
            "sentiment_momentum": 0,
            "premarket_sentiment": 0,
            "news_volume": 0,
            "sentiment_price_div": 0,
        }
        
        try:
            features = compute_lean_features(
                prev_minute.tail(120),
                prev_daily,
                market_data,
                sentiment_data,
            )
            
            if features is not None and len(features) > 0:
                features_list.append(features.iloc[0].values)
                targets_list.append({
                    "gap_direction": targets_df["gap_direction"].iloc[i],
                    "gap_magnitude": targets_df["gap_magnitude"].iloc[i],
                    "gaps_up": float(targets_df["gaps_up"].iloc[i]),
                    "gaps_down": float(targets_df["gaps_down"].iloc[i]),
                    "gap_fills": float(targets_df["gap_fills"].iloc[i]),
                    "profitable_long": float(targets_df["profitable_long"].iloc[i]),
                    "profitable_short": float(targets_df["profitable_short"].iloc[i]),
                })
        except Exception as e:
            continue
    
    if len(features_list) < 50:
        return None
    
    # Create feature and target DataFrames
    features_df = pd.DataFrame(features_list, columns=LEAN_FEATURE_NAMES)
    targets_df = pd.DataFrame(targets_list)
    
    # Compute ICIR for each feature-target pair
    results = []
    target_cols = ["gap_direction", "gap_magnitude", "gaps_up", "gaps_down", 
                   "gap_fills", "profitable_long", "profitable_short"]
    
    for feature in LEAN_FEATURE_NAMES:
        for target in target_cols:
            mean_ic, std_ic, icir, pct_pos = compute_walk_forward_icir(
                features_df[feature],
                targets_df[target],
                n_folds=n_folds,
            )
            results.append({
                "symbol": symbol,
                "feature": feature,
                "target": target,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": icir,
                "pct_positive": pct_pos,
                "n_samples": len(features_df),
            })
    
    return pd.DataFrame(results)


def aggregate_across_stocks(results_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate ICIR results across multiple stocks."""
    if not results_list:
        return pd.DataFrame()
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # Group by feature and target, compute mean ICIR
    aggregated = all_results.groupby(["feature", "target"]).agg({
        "mean_ic": "mean",
        "std_ic": "mean",
        "icir": "mean",
        "pct_positive": "mean",
        "n_samples": "sum",
    }).reset_index()
    
    # Compute robust ICIR: require consistent direction across stocks
    # Count how many stocks had positive IC for this feature-target
    positive_counts = all_results.groupby(["feature", "target"]).apply(
        lambda x: (x["mean_ic"] > 0).sum() / len(x)
    ).reset_index(name="cross_stock_consistency")
    
    aggregated = aggregated.merge(positive_counts, on=["feature", "target"])
    
    return aggregated


def print_results(results_df: pd.DataFrame, min_icir: float = 0.2):
    """Print formatted ICIR results."""
    print("\n" + "=" * 80)
    print("SIGNAL AUDIT RESULTS")
    print("=" * 80)
    
    if len(results_df) == 0:
        print("No results to display.")
        return
    
    # Top features by target
    targets = results_df["target"].unique()
    
    for target in targets:
        target_results = results_df[results_df["target"] == target].sort_values("icir", ascending=False)
        
        print(f"\n--- Target: {target} ---")
        print(f"{'Feature':<30} {'Mean IC':>10} {'ICIR':>10} {'Consistency':>12} {'Samples':>10}")
        print("-" * 80)
        
        for _, row in target_results.head(10).iterrows():
            print(f"{row['feature']:<30} {row['mean_ic']:>10.4f} {row['icir']:>10.4f} "
                  f"{row['cross_stock_consistency']:>11.2%} {int(row['n_samples']):>10,}")
        
        # Count features above threshold
        above_threshold = target_results[target_results["icir"] >= min_icir]
        print(f"\nFeatures with ICIR >= {min_icir}: {len(above_threshold)} / {len(target_results)}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    strong_signals = results_df[results_df["icir"] >= min_icir]
    print(f"Total feature-target pairs tested: {len(results_df)}")
    print(f"Strong signals (ICIR >= {min_icir}): {len(strong_signals)}")
    print(f"Percentage with signal: {len(strong_signals) / len(results_df) * 100:.1f}%")
    
    if len(strong_signals) > 0:
        print("\nStrongest signals:")
        top = strong_signals.sort_values("icir", ascending=False).head(10)
        for _, row in top.iterrows():
            print(f"  {row['feature']} -> {row['target']}: ICIR={row['icir']:.3f}")
    else:
        print("\n⚠️  WARNING: No features passed the ICIR threshold!")
        print("The prediction target may need to be reformulated.")


def main():
    parser = argparse.ArgumentParser(description="Signal Audit - Test feature predictive power")
    parser.add_argument("--universe", type=str, default="nifty100",
                       choices=["nifty50", "nifty100", "nifty200"],
                       help="Stock universe to test")
    parser.add_argument("--start-date", type=str, default="2021-01-01",
                       help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="",
                       help="End date for analysis (YYYY-MM-DD)")
    parser.add_argument("--n-folds", type=int, default=8,
                       help="Number of walk-forward folds")
    parser.add_argument("--min-icir", type=float, default=0.2,
                       help="Minimum ICIR threshold for 'useful' feature")
    parser.add_argument("--max-stocks", type=int, default=0,
                       help="Limit number of stocks (0 = all)")
    parser.add_argument("--output", type=str, default="signal_audit_results.json",
                       help="Output file for detailed results")
    parser.add_argument("--data-dir", type=str, default="nifty500",
                       help="Directory containing minute CSV files")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INTRADAYNET SIGNAL AUDIT")
    print("=" * 80)
    print(f"Universe: {args.universe}")
    print(f"Date range: {args.start_date} to {args.end_date or 'end of data'}")
    print(f"Walk-forward folds: {args.n_folds}")
    print(f"Min ICIR threshold: {args.min_icir}")
    print("=" * 80)
    
    # Get universe
    symbols = get_universe(args.universe)
    if args.max_stocks > 0:
        symbols = symbols[:args.max_stocks]
    print(f"\nAnalyzing {len(symbols)} stocks...\n")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Process each stock
    all_results = []
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] Auditing {symbol}...", end=" ")
        
        result = audit_single_stock(
            symbol=symbol,
            data_dir=data_dir,
            n_folds=args.n_folds,
        )
        
        if result is not None:
            all_results.append(result)
            print(f"✓ ({len(result)} feature-target pairs)")
        else:
            print("✗ (insufficient data)")
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(all_results)} stocks in {elapsed:.1f}s")
    
    if not all_results:
        print("\nError: No stocks could be processed. Check data directory.")
        return
    
    # Aggregate results
    aggregated = aggregate_across_stocks(all_results)
    
    # Print results
    print_results(aggregated, min_icir=args.min_icir)
    
    # Save detailed results
    output_path = Path(args.output)
    aggregated.to_json(output_path, orient="records", indent=2)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Save summary for next steps
    strong_signals = aggregated[aggregated["icir"] >= args.min_icir]
    if len(strong_signals) > 0:
        summary = {
            "n_stocks": len(all_results),
            "universe": args.universe,
            "date_range": f"{args.start_date} to {args.end_date}",
            "strong_signals": strong_signals[["feature", "target", "icir"]].to_dict("records"),
            "features_above_threshold": strong_signals["feature"].unique().tolist(),
        }
        summary_path = output_path.parent / "signal_audit_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
