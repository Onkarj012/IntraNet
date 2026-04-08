#!/usr/bin/env python3
"""
Walk-Forward Validation for IntradayNet.

Strict temporal validation to prevent look-ahead bias:
1. Train on historical data
2. Validate on subsequent period
3. Test on period after validation
4. Step forward and repeat

Usage:
    python scripts/walkforward_train.py --universe nifty100 --start-date 2021-01-01
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

# LightGBM
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.data_loader import DataPipeline


# Top features from signal audit (ICIR >= 0.5)
SELECTED_FEATURES = [
    "vol_momentum",
    "prev_day_volatility", 
    "prev_gap_size",
    "overnight_gap",
    "price_vs_vwap",
    "close_vs_day_high",
    "volume_pace",
]


def compute_features_targets(minute_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute selected features and gap target."""
    # Resample to daily
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
    
    # Core features
    features["overnight_gap"] = daily["open"] / daily["close"].shift(1) - 1
    features["prev_day_return"] = daily["close"].pct_change()
    features["prev_day_volatility"] = features["prev_day_return"].rolling(21).std()
    
    # Gap features
    features["prev_gap_size"] = features["overnight_gap"].shift(1).abs()
    
    # Volume features from minute data
    minute_df["date_only"] = minute_df.index.date
    vol_by_date = minute_df.groupby("date_only")["volume"].sum()
    vol_by_date.index = pd.to_datetime(vol_by_date.index)
    features["volume"] = vol_by_date.reindex(features.index)
    features["vol_momentum"] = features["volume"] / features["volume"].rolling(20).mean() - 1
    
    # VWAP position
    minute_df["tp"] = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3
    minute_df["tpv"] = minute_df["tp"] * minute_df["volume"]
    vwap_daily = minute_df.groupby("date_only").apply(
        lambda x: x["tpv"].sum() / x["volume"].sum() if x["volume"].sum() > 0 else x["close"].iloc[-1],
        include_groups=False
    )
    vwap_daily.index = pd.to_datetime(vwap_daily.index)
    features["vwap"] = vwap_daily.reindex(features.index)
    features["price_vs_vwap"] = features.index.map(
        lambda d: daily.loc[d, "close"] / features.loc[d, "vwap"] - 1 if d in features.index and d in daily.index else 0
    )
    
    # Close vs day high
    high_daily = minute_df.groupby("date_only")["high"].max()
    high_daily.index = pd.to_datetime(high_daily.index)
    features["day_high"] = high_daily.reindex(features.index)
    features["close_vs_day_high"] = daily["close"] / features["day_high"] - 1
    
    # Volume pace (last 30 min)
    last_30_vol = minute_df.groupby("date_only").apply(
        lambda x: x["volume"].tail(30).sum(),
        include_groups=False
    )
    last_30_vol.index = pd.to_datetime(last_30_vol.index)
    features["last_30_vol"] = last_30_vol.reindex(features.index)
    features["volume_pace"] = features["last_30_vol"] / (features["volume"] / 6) - 1
    
    # Target: Gap direction (binary classification)
    # 1 = gap up > 0.2%, 0 = gap down or small
    next_gap = (daily["open"] / daily["close"].shift(1) - 1).shift(-1)
    target = (next_gap > 0.002).astype(int)
    
    # Select only features we want
    available_features = [f for f in SELECTED_FEATURES if f in features.columns]
    features = features[available_features]
    
    # Align and drop NaN
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]
    
    return features, target


def create_folds(dates: pd.DatetimeIndex, 
                 train_months: int = 12,
                 val_months: int = 3,
                 test_months: int = 3) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Create walk-forward fold boundaries.
    
    Returns list of (train_start, train_end, val_end, test_end)
    """
    folds = []
    
    # Start with enough data for training
    start = dates[0] + pd.DateOffset(months=train_months)
    
    while True:
        train_start = dates[0]
        train_end = start
        val_end = start + pd.DateOffset(months=val_months)
        test_end = val_end + pd.DateOffset(months=test_months)
        
        if test_end > dates[-1]:
            break
        
        folds.append((train_start, train_end, val_end, test_end))
        
        # Step forward
        start = start + pd.DateOffset(months=test_months)
    
    return folds


def train_fold(train_X: pd.DataFrame, train_y: pd.Series,
               val_X: pd.DataFrame, val_y: pd.Series) -> lgb.Booster:
    """Train LightGBM on one fold."""
    
    # Minimal model - aggressively regularized
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 100,  # Small ensemble
        "max_depth": 3,       # Shallow trees
        "num_leaves": 7,      # Highly constrained
        "learning_rate": 0.05,
        "min_child_samples": 100,  # Require many samples per leaf
        "subsample": 0.7,
        "colsample_bytree": 0.7,  # Feature subsampling
        "reg_alpha": 0.5,     # L1 regularization
        "reg_lambda": 2.0,    # L2 regularization
    }
    
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    
    return model


def evaluate_fold(model: lgb.Booster, test_X: pd.DataFrame, test_y: pd.Series) -> Dict:
    """Evaluate model on test set."""
    pred_proba = model.predict_proba(test_X)[:, 1]
    pred = (pred_proba > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(test_y, pred)
    auc = roc_auc_score(test_y, pred_proba) if len(np.unique(test_y)) > 1 else 0.5
    
    # Information coefficient
    ic, _ = spearmanr(pred_proba, test_y)
    
    return {
        "accuracy": acc,
        "auc": auc,
        "ic": ic,
        "n_test": len(test_y),
        "n_positive": test_y.sum(),
    }


def walk_forward_validation(features: pd.DataFrame, target: pd.Series,
                           train_months: int = 12,
                           val_months: int = 3,
                           test_months: int = 3) -> List[Dict]:
    """Run walk-forward validation."""
    
    # Create folds
    folds = create_folds(features.index, train_months, val_months, test_months)
    
    results = []
    
    for i, (train_start, train_end, val_end, test_end) in enumerate(folds):
        print(f"\nFold {i+1}/{len(folds)}:")
        print(f"  Train: {train_start.date()} to {train_end.date()}")
        print(f"  Val:   {train_end.date()} to {val_end.date()}")
        print(f"  Test:  {val_end.date()} to {test_end.date()}")
        
        # Split data
        train_mask = (features.index >= train_start) & (features.index < train_end)
        val_mask = (features.index >= train_end) & (features.index < val_end)
        test_mask = (features.index >= val_end) & (features.index < test_end)
        
        train_X = features[train_mask]
        train_y = target[train_mask]
        val_X = features[val_mask]
        val_y = target[val_mask]
        test_X = features[test_mask]
        test_y = target[test_mask]
        
        print(f"  Samples: train={len(train_X)}, val={len(val_X)}, test={len(test_X)}")
        
        if len(train_X) < 100 or len(test_X) < 20:
            print(f"  Skipping - insufficient data")
            continue
        
        # Train
        t0 = time.time()
        model = train_fold(train_X, train_y, val_X, val_y)
        train_time = time.time() - t0
        
        # Evaluate
        metrics = evaluate_fold(model, test_X, test_y)
        metrics["fold"] = i + 1
        metrics["train_time"] = train_time
        metrics["n_trees"] = model.booster_.num_trees()
        
        print(f"  AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}, IC={metrics['ic']:.4f}")
        
        results.append(metrics)
    
    return results


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across folds."""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    return {
        "mean_auc": df["auc"].mean(),
        "std_auc": df["auc"].std(),
        "mean_accuracy": df["accuracy"].mean(),
        "std_accuracy": df["accuracy"].std(),
        "mean_ic": df["ic"].mean(),
        "std_ic": df["ic"].std(),
        "n_folds": len(results),
        "total_test_samples": df["n_test"].sum(),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Training")
    parser.add_argument("--universe", type=str, default="nifty100")
    parser.add_argument("--start-date", type=str, default="2021-01-01")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--output", type=str, default="walkforward_results.json")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Universe: {args.universe}")
    print(f"Features: {SELECTED_FEATURES}")
    print(f"Train/Val/Test: {args.train_months}/{args.val_months}/{args.test_months} months")
    print("=" * 80)
    
    # Load data
    symbols = get_universe(args.universe)
    if args.max_stocks > 0:
        symbols = symbols[:args.max_stocks]
    
    data_dir = Path(args.data_dir)
    
    # Collect features and targets for all stocks
    all_features = []
    all_targets = []
    
    print(f"\nLoading {len(symbols)} stocks...")
    
    for i, symbol in enumerate(symbols):
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        
        try:
            minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            minute_df.columns = minute_df.columns.str.lower()
            
            # Filter by date
            if args.start_date:
                minute_df = minute_df[minute_df.index >= args.start_date]
            if args.end_date:
                minute_df = minute_df[minute_df.index <= args.end_date]
            
            features, target = compute_features_targets(minute_df)
            
            if features is not None and len(features) > 50:
                features["symbol"] = symbol
                all_features.append(features)
                all_targets.append(target)
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({len(features):4d} days)")
            else:
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ (insufficient data)")
                
        except Exception as e:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ ({e})")
    
    if not all_features:
        print("\nNo data loaded!")
        return
    
    # Combine all stocks
    combined_features = pd.concat(all_features, ignore_index=False)
    combined_target = pd.concat(all_targets, ignore_index=False)
    
    print(f"\nTotal samples: {len(combined_features)}")
    print(f"Symbols: {combined_features['symbol'].nunique()}")
    print(f"Date range: {combined_features.index.min().date()} to {combined_features.index.max().date()}")
    
    # Remove symbol column for training
    feature_cols = [c for c in combined_features.columns if c != "symbol"]
    X = combined_features[feature_cols]
    y = combined_target
    
    # Run walk-forward validation
    print("\n" + "=" * 80)
    print("RUNNING WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    results = walk_forward_validation(
        X, y,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
    )
    
    # Aggregate
    summary = aggregate_results(results)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Folds completed: {summary['n_folds']}")
    print(f"Total test samples: {summary['total_test_samples']}")
    print(f"Mean AUC: {summary['mean_auc']:.4f} (±{summary['std_auc']:.4f})")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} (±{summary['std_accuracy']:.4f})")
    print(f"Mean IC: {summary['mean_ic']:.4f} (±{summary['std_ic']:.4f})")
    
    # Save
    output = {
        "config": vars(args),
        "features": SELECTED_FEATURES,
        "folds": results,
        "summary": summary,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
