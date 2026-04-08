#!/usr/bin/env python3
"""
Train Minimal LightGBM Model for Gap Prediction.

Trains an aggressively regularized model with top features from signal audit.
This is the production model for morning picks.

Usage:
    python scripts/train_minimal_model.py --universe nifty100 --start-date 2021-01-01
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe


# Top 7 features from signal audit (ICIR >= 0.5)
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
    """Compute features and target from minute data."""
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
    
    # Target: Gap up > 0.2%
    next_gap = (daily["open"] / daily["close"].shift(1) - 1).shift(-1)
    target = (next_gap > 0.002).astype(int)
    
    # Select features
    available = [f for f in SELECTED_FEATURES if f in features.columns]
    features = features[available]
    
    # Align and drop NaN
    valid = features.dropna().index.intersection(target.dropna().index)
    return features.loc[valid], target.loc[valid]


def train_model(X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    """Train minimal LightGBM model."""
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "max_depth": 3,
        "num_leaves": 7,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }
    
    # Time-based split: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    train_X, test_X = X.iloc[:split_idx], X.iloc[split_idx:]
    train_y, test_y = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining on {len(train_X)} samples, testing on {len(test_X)}")
    print(f"Train positive rate: {train_y.mean():.3f}")
    print(f"Test positive rate: {test_y.mean():.3f}")
    
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        train_X, train_y,
        eval_set=[(test_X, test_y)],
        callbacks=[lgb.early_stopping(20, verbose=True)],
    )
    
    # Evaluate
    pred_proba = model.predict_proba(test_X)[:, 1]
    pred = (pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(test_y, pred)
    auc = roc_auc_score(test_y, pred_proba)
    
    print(f"\nTest Performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']}")
    
    return model, {"accuracy": acc, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Train Minimal Model")
    parser.add_argument("--universe", type=str, default="nifty100")
    parser.add_argument("--start-date", type=str, default="2021-01-01")
    parser.add_argument("--output", type=str, default="models/minimal_gap_model.pkl")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAIN MINIMAL LIGHTGBM MODEL")
    print("=" * 80)
    print(f"Universe: {args.universe}")
    print(f"Features: {SELECTED_FEATURES}")
    print("=" * 80)
    
    symbols = get_universe(args.universe)
    data_dir = Path(args.data_dir)
    
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
            
            if args.start_date:
                minute_df = minute_df[minute_df.index >= args.start_date]
            
            features, target = compute_features_targets(minute_df)
            
            if features is not None and len(features) > 50:
                features["symbol"] = symbol
                all_features.append(features)
                all_targets.append(target)
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({len(features):4d} days)")
            else:
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗")
                
        except Exception as e:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ ({str(e)[:30]})")
    
    if not all_features:
        print("\nNo data loaded!")
        return
    
    # Combine
    combined_features = pd.concat(all_features)
    combined_target = pd.concat(all_targets)
    
    print(f"\n{'='*80}")
    print(f"Total samples: {len(combined_features)}")
    print(f"Symbols: {combined_features['symbol'].nunique()}")
    print(f"Date range: {combined_features.index.min().date()} to {combined_features.index.max().date()}")
    print(f"{'='*80}")
    
    # Train
    feature_cols = [c for c in combined_features.columns if c != "symbol"]
    X = combined_features[feature_cols]
    y = combined_target
    
    model, metrics = train_model(X, y)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": SELECTED_FEATURES,
            "metrics": metrics,
            "n_samples": len(X),
            "n_symbols": combined_features["symbol"].nunique(),
        }, f)
    
    print(f"\nModel saved to: {output_path}")
    
    # Save summary
    summary = {
        "model_path": str(output_path),
        "features": SELECTED_FEATURES,
        "metrics": metrics,
        "n_samples": len(X),
        "n_symbols": combined_features["symbol"].nunique(),
        "date_range": {
            "start": str(combined_features.index.min().date()),
            "end": str(combined_features.index.max().date()),
        }
    }
    
    summary_path = output_path.parent / "model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
