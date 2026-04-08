#!/usr/bin/env python3
"""
Price Movement Prediction Model - 2021-2024 Training.

Trains on NIFTY100 (2021-2024) and saves predictions for 2025 backtesting.

Output format (predictions_2025.csv):
  date, symbol, predicted_return, confidence, direction
  2025-01-01, RELIANCE, 0.018, 0.72, LONG
  2025-01-01, TCS, -0.012, 0.65, SHORT

Usage:
    python scripts/train_price_model.py --train-start 2021-01-01 --train-end 2024-12-31 --predict-start 2025-01-01 --predict-end 2025-12-31
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder


def compute_features_targets(minute_df: pd.DataFrame, symbol: str,
                            market_builder: MarketFeatureBuilder,
                            sentiment_builder: SentimentFeatureBuilder) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute features and target for price movement prediction.
    
    Target: (close - open) / open = intraday return from open to close
    """
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
    
    # Target: Return from open to close (what we want to predict)
    target = (daily["close"] - daily["open"]) / daily["open"]
    
    # Features (all computed from previous day or pre-market)
    features = pd.DataFrame(index=daily.index)
    
    # Previous day data
    features["prev_close"] = daily["close"].shift(1)
    features["prev_high"] = daily["high"].shift(1)
    features["prev_low"] = daily["low"].shift(1)
    features["prev_volume"] = daily["volume"].shift(1)
    
    # Returns
    features["prev_return"] = daily["close"].pct_change()
    features["prev_return_5d"] = daily["close"].pct_change(5)
    features["prev_return_10d"] = daily["close"].pct_change(10)
    
    # Volatility
    features["prev_volatility"] = features["prev_return"].rolling(21).std()
    
    # Range
    features["prev_range"] = (daily["high"].shift(1) - daily["low"].shift(1)) / daily["open"].shift(1)
    features["prev_range_avg"] = features["prev_range"].rolling(14).mean()
    
    # Gap (open vs prev close)
    features["overnight_gap"] = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)
    features["prev_gap"] = features["overnight_gap"].shift(1)
    
    # Volume
    minute_df["date_only"] = minute_df.index.date
    vol_by_date = minute_df.groupby("date_only")["volume"].sum()
    vol_by_date.index = pd.to_datetime(vol_by_date.index)
    features["volume"] = vol_by_date.reindex(features.index)
    features["volume_avg"] = features["volume"].rolling(20).mean()
    features["volume_ratio"] = features["volume"] / features["volume_avg"]
    
    # VWAP at close
    minute_df["tp"] = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3
    minute_df["tpv"] = minute_df["tp"] * minute_df["volume"]
    vwap_daily = minute_df.groupby("date_only").apply(
        lambda x: x["tpv"].sum() / x["volume"].sum() if x["volume"].sum() > 0 else x["close"].iloc[-1],
        include_groups=False
    )
    vwap_daily.index = pd.to_datetime(vwap_daily.index)
    features["vwap"] = vwap_daily.reindex(features.index)
    features["close_vs_vwap"] = daily["close"] / features["vwap"] - 1
    
    # High/LW proximity
    features["prev_close_vs_high"] = daily["close"].shift(1) / daily["high"].shift(1) - 1
    features["prev_close_vs_low"] = daily["close"].shift(1) / daily["low"].shift(1) - 1
    
    # Market features (pre-market available)
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
    
    # Align
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    return features.loc[valid_idx], target.loc[valid_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", type=str, default="2021-01-01")
    parser.add_argument("--train-end", type=str, default="2024-12-31")
    parser.add_argument("--predict-start", type=str, default="2025-01-01")
    parser.add_argument("--predict-end", type=str, default="2025-12-31")
    parser.add_argument("--universe", type=str, default="nifty100")
    parser.add_argument("--output-model", type=str, default="models/price_model.pkl")
    parser.add_argument("--output-predictions", type=str, default="predictions_2025.csv")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PRICE MOVEMENT PREDICTION MODEL")
    print("=" * 80)
    print(f"Training: {args.train_start} to {args.train_end}")
    print(f"Predictions: {args.predict_start} to {args.predict_end}")
    print("=" * 80)
    
    # Load data builders
    print("\nLoading market and sentiment data...")
    market_builder = MarketFeatureBuilder()
    market_builder.download(start=args.train_start, end=args.predict_end)
    
    sentiment_builder = SentimentFeatureBuilder(
        "sentiment/combined_sentiment_2015_2025.csv",
        market_builder=market_builder
    )
    sentiment_builder._load()
    print("✓ Data loaded")
    
    symbols = get_universe(args.universe)
    data_dir = Path("nifty500")
    
    # ========== TRAINING PHASE ==========
    print(f"\n{'='*80}")
    print("TRAINING PHASE")
    print(f"{'='*80}")
    print(f"Processing {len(symbols)} stocks...")
    
    all_features = []
    all_targets = []
    
    for i, symbol in enumerate(symbols):
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        
        try:
            minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            minute_df.columns = minute_df.columns.str.lower()
            
            # Filter training period
            train_df = minute_df[(minute_df.index >= args.train_start) & (minute_df.index <= args.train_end)]
            
            if len(train_df) < 1000:
                continue
            
            features, target = compute_features_targets(train_df, symbol, market_builder, sentiment_builder)
            
            if features is not None and len(features) > 100:
                features["symbol"] = symbol
                all_features.append(features)
                all_targets.append(target)
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({len(features):4d} days)")
            else:
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗")
                
        except Exception as e:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ ({str(e)[:40]})")
    
    if not all_features:
        print("\n❌ No training data!")
        return
    
    # Combine training data
    X_train = pd.concat(all_features)
    y_train = pd.concat(all_targets)
    
    feature_cols = [c for c in X_train.columns if c != "symbol"]
    
    print(f"\n{'='*80}")
    print(f"Training samples: {len(X_train)}")
    print(f"Symbols: {X_train['symbol'].nunique()}")
    print(f"Mean daily return: {y_train.mean():.4f}")
    print(f"Return std: {y_train.std():.4f}")
    print(f"{'='*80}")
    
    # Train model
    print("\nTraining LightGBM regressor...")
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    # Time-based split for validation
    split_idx = int(len(X_train) * 0.8)
    model.fit(
        X_train[feature_cols].iloc[:split_idx],
        y_train.iloc[:split_idx],
        eval_set=[(X_train[feature_cols].iloc[split_idx:], y_train.iloc[split_idx:])],
        callbacks=[lgb.early_stopping(30, verbose=True)],
    )
    
    # Validate
    val_pred = model.predict(X_train[feature_cols].iloc[split_idx:])
    val_mae = mean_absolute_error(y_train.iloc[split_idx:], val_pred)
    val_r2 = r2_score(y_train.iloc[split_idx:], val_pred)
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"MAE: {val_mae:.4f}")
    print(f"R²: {val_r2:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.0f}")
    
    # Save model
    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_cols,
            "mae": val_mae,
            "r2": val_r2,
        }, f)
    
    print(f"\n✓ Model saved: {model_path}")
    
    # ========== PREDICTION PHASE ==========
    print(f"\n{'='*80}")
    print("PREDICTION PHASE (2025)")
    print(f"{'='*80}")
    print(f"Generating predictions for {args.predict_start} to {args.predict_end}...")
    
    predictions = []
    
    for i, symbol in enumerate(symbols):
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        
        try:
            minute_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            minute_df.columns = minute_df.columns.str.lower()
            
            # Filter prediction period
            predict_df = minute_df[(minute_df.index >= args.predict_start) & (minute_df.index <= args.predict_end)]
            
            if len(predict_df) < 100:
                continue
            
            # Compute features
            features, _ = compute_features_targets(predict_df, symbol, market_builder, sentiment_builder)
            
            if features is None or len(features) == 0:
                continue
            
            # Predict
            X_pred = features[feature_cols]
            pred_returns = model.predict(X_pred)
            
            # Create predictions dataframe
            for date, pred in zip(features.index, pred_returns):
                predictions.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "predicted_return": float(pred),
                    "direction": "LONG" if pred > 0 else "SHORT",
                    "confidence": abs(pred),
                })
            
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({len(features):3d} predictions)")
            
        except Exception as e:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ ({str(e)[:40]})")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(args.output_predictions, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Predictions saved: {args.output_predictions}")
    print(f"  Total predictions: {len(pred_df)}")
    print(f"  Dates: {pred_df['date'].min()} to {pred_df['date'].max()}")
    print(f"  Long predictions: {len(pred_df[pred_df['direction'] == 'LONG'])}")
    print(f"  Short predictions: {len(pred_df[pred_df['direction'] == 'SHORT'])}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
