#!/usr/bin/env python3
"""
Train NIFTY100 Model (2021-2024) for 2025 Blind Test.

Trains on NIFTY100 stocks from 2021-2024 with:
- Stock features (price, volume, gaps)
- Market features (VIX, crude, USD/INR, global markets)
- Sentiment features (news sentiment)

Saves model for 2025 blind testing.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder


def compute_all_features(minute_df: pd.DataFrame, symbol: str,
                         market_builder: MarketFeatureBuilder,
                         sentiment_builder: SentimentFeatureBuilder) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute stock + market + sentiment features."""
    # Daily aggregation
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
    
    # === STOCK FEATURES ===
    features["overnight_gap"] = daily["open"] / daily["close"].shift(1) - 1
    features["prev_day_return"] = daily["close"].pct_change()
    features["prev_day_volatility"] = features["prev_day_return"].rolling(21).std()
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
    
    # === MARKET FEATURES ===
    market_feats = market_builder.get_features(features.index)
    india_feats = market_builder.get_india_market_features(features.index)
    for col in market_feats.columns:
        features[col] = market_feats[col]
    for key, series in india_feats.items():
        features[key] = series
    
    # === SENTIMENT FEATURES ===
    sent_feats = sentiment_builder.get_features(symbol, features.index)
    for col in sent_feats.columns:
        features[col] = sent_feats[col]
    
    # Target: next day gap up (> 0.2%)
    next_gap = (daily["open"] / daily["close"].shift(1) - 1).shift(-1)
    target = (next_gap > 0.002).astype(int)
    
    # Align
    valid = features.dropna().index.intersection(target.dropna().index)
    return features.loc[valid], target.loc[valid]


def main():
    print("=" * 80)
    print("TRAIN NIFTY100 MODEL (2021-2024)")
    print("=" * 80)
    
    # Initialize builders
    print("\nLoading market and sentiment data...")
    market_builder = MarketFeatureBuilder()
    market_builder.download(start="2021-01-01", end="2024-12-31")
    
    sentiment_builder = SentimentFeatureBuilder(
        "sentiment/combined_sentiment_2015_2025.csv",
        market_builder=market_builder
    )
    sentiment_builder._load()
    print("✓ Data loaded")
    
    # Load NIFTY100
    symbols = get_universe("nifty100")
    data_dir = Path("nifty500")
    
    print(f"\nProcessing {len(symbols)} NIFTY100 stocks...")
    print(f"Period: 2021-01-01 to 2024-12-31")
    print("-" * 80)
    
    all_features = []
    all_targets = []
    
    for i, symbol in enumerate(symbols):
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        
        try:
            minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            minute_df.columns = minute_df.columns.str.lower()
            
            # Filter to training period
            minute_df = minute_df[(minute_df.index >= "2021-01-01") & (minute_df.index <= "2024-12-31")]
            
            if len(minute_df) < 1000:
                continue
            
            features, target = compute_all_features(minute_df, symbol, market_builder, sentiment_builder)
            
            if features is not None and len(features) > 100:
                features["symbol"] = symbol
                all_features.append(features)
                all_targets.append(target)
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({len(features):4d} days)")
            else:
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ (insufficient data)")
                
        except Exception as e:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ ({str(e)[:40]})")
    
    if not all_features:
        print("\n❌ No data loaded!")
        return
    
    # Combine all stocks
    combined_features = pd.concat(all_features)
    combined_target = pd.concat(all_targets)
    
    print(f"\n{'='*80}")
    print(f"Total samples: {len(combined_features)}")
    print(f"Symbols: {combined_features['symbol'].nunique()}")
    print(f"Date range: {combined_features.index.min().date()} to {combined_features.index.max().date()}")
    print(f"{'='*80}")
    
    # Select feature columns (exclude symbol)
    feature_cols = [c for c in combined_features.columns if c != "symbol"]
    X = combined_features[feature_cols]
    y = combined_target
    
    # Time-based split: 80% train, 20% validation
    split_idx = int(len(X) * 0.8)
    train_X, val_X = X.iloc[:split_idx], X.iloc[split_idx:]
    train_y, val_y = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining on {len(train_X)} samples, validating on {len(val_X)}")
    print(f"Train positive rate: {train_y.mean():.3f}")
    print(f"Val positive rate: {val_y.mean():.3f}")
    
    # Train model
    print("\nTraining LightGBM model...")
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 200,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        callbacks=[lgb.early_stopping(30, verbose=True)],
    )
    
    # Evaluate
    val_pred_proba = model.predict_proba(val_X)[:, 1]
    val_pred = (val_pred_proba > 0.5).astype(int)
    
    val_acc = accuracy_score(val_y, val_pred)
    val_auc = roc_auc_score(val_y, val_pred_proba)
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"AUC: {val_auc:.4f}")
    print(f"Best iteration: {model.booster_.num_trees()}")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 15 Features:")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.0f}")
    
    # Save model
    output_path = Path("models/nifty100_model_2021_2024.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_cols,
            "metrics": {
                "val_accuracy": val_acc,
                "val_auc": val_auc,
            },
            "n_train_samples": len(train_X),
            "n_val_samples": len(val_X),
            "n_symbols": combined_features["symbol"].nunique(),
            "train_period": "2021-01-01 to 2024-12-31",
        }, f)
    
    print(f"\n{'='*80}")
    print(f"✓ Model saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
