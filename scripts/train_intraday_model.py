#!/usr/bin/env python3
"""
Complete Intraday Movement Prediction System.

Predicts intraday price movement from open:
- Will stock move +X% above open during the day? (LONG signal)
- Will stock move -X% below open during the day? (SHORT signal)

Uses gap as a feature, not the primary target.

Targets:
1. long_viable: Can we make +target% profit from open? (binary)
2. short_viable: Can we make -target% profit from open? (binary)
3. max_up_move: How much does price go up from open? (regression)
4. max_down_move: How much does price go down from open? (regression)

Usage:
    python scripts/train_intraday_model.py --target-pct 0.01
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.universe import get_universe
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder


def compute_intraday_targets(daily_df: pd.DataFrame, target_pct: float = 0.01) -> pd.DataFrame:
    """
    Compute intraday movement targets.
    
    For each day, calculate:
    - max_up: (high - open) / open
    - max_down: (open - low) / open
    - long_viable: max_up > target_pct
    - short_viable: max_down > target_pct
    """
    targets = pd.DataFrame(index=daily_df.index)
    
    # Intraday movement from open
    targets["max_up"] = (daily_df["high"] - daily_df["open"]) / daily_df["open"]
    targets["max_down"] = (daily_df["open"] - daily_df["low"]) / daily_df["open"]
    
    # Binary: Can we achieve target profit?
    targets["long_viable"] = (targets["max_up"] > target_pct).astype(int)
    targets["short_viable"] = (targets["max_down"] > target_pct).astype(int)
    
    # Gap as feature (not target)
    targets["gap"] = (daily_df["open"] - daily_df["close"].shift(1)) / daily_df["close"].shift(1)
    
    # Close vs open (for reference)
    targets["close_return"] = (daily_df["close"] - daily_df["open"]) / daily_df["open"]
    
    return targets


def compute_all_features(minute_df: pd.DataFrame, symbol: str,
                         market_builder: MarketFeatureBuilder,
                         sentiment_builder: SentimentFeatureBuilder,
                         target_pct: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute features and targets for intraday movement prediction."""
    
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
    
    # Compute targets
    targets = compute_intraday_targets(daily, target_pct)
    
    # Features
    features = pd.DataFrame(index=daily.index)
    
    # === PREVIOUS DAY FEATURES (predictors) ===
    features["prev_day_return"] = daily["close"].pct_change()
    features["prev_day_volatility"] = features["prev_day_return"].rolling(21).std()
    
    # Previous day's intraday range
    features["prev_day_range"] = (daily["high"].shift(1) - daily["low"].shift(1)) / daily["open"].shift(1)
    features["prev_day_atr"] = features["prev_day_range"].rolling(14).mean()
    
    # Gap (as feature, not target!)
    features["overnight_gap"] = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)
    features["prev_gap_size"] = features["overnight_gap"].shift(1).abs()
    
    # Volume features
    minute_df["date_only"] = minute_df.index.date
    vol_by_date = minute_df.groupby("date_only")["volume"].sum()
    vol_by_date.index = pd.to_datetime(vol_by_date.index)
    features["volume"] = vol_by_date.reindex(features.index)
    features["vol_momentum"] = features["volume"] / features["volume"].rolling(20).mean() - 1
    
    # VWAP distance at close
    minute_df["tp"] = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3
    minute_df["tpv"] = minute_df["tp"] * minute_df["volume"]
    vwap_daily = minute_df.groupby("date_only").apply(
        lambda x: x["tpv"].sum() / x["volume"].sum() if x["volume"].sum() > 0 else x["close"].iloc[-1],
        include_groups=False
    )
    vwap_daily.index = pd.to_datetime(vwap_daily.index)
    features["vwap"] = vwap_daily.reindex(features.index)
    features["close_vs_vwap"] = daily["close"] / features["vwap"] - 1
    
    # Price momentum
    features["price_momentum_5d"] = daily["close"].pct_change(5)
    features["price_momentum_10d"] = daily["close"].pct_change(10)
    
    # High/low proximity
    features["close_vs_day_high"] = daily["close"] / daily["high"] - 1
    features["close_vs_day_low"] = daily["close"] / daily["low"] - 1
    
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
    
    # Align features and targets
    valid_idx = features.dropna().index.intersection(targets.dropna().index)
    return features.loc[valid_idx], targets.loc[valid_idx]


def train_models(X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series, 
                 y_up_mag: pd.Series, y_down_mag: pd.Series) -> Dict:
    """Train 4 models: long/short binary + magnitude regression."""
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    train_X, val_X = X.iloc[:split_idx], X.iloc[split_idx:]
    
    models = {}
    metrics = {}
    
    # Model 1: LONG viable
    print("\n[1/4] Training LONG viability model...")
    m1 = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=200,
        max_depth=5, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8
    )
    m1.fit(train_X, y_long.iloc[:split_idx])
    pred1 = m1.predict_proba(val_X)[:, 1]
    models["long"] = m1
    metrics["long_auc"] = roc_auc_score(y_long.iloc[split_idx:], pred1)
    print(f"  Validation AUC: {metrics['long_auc']:.4f}")
    
    # Model 2: SHORT viable
    print("\n[2/4] Training SHORT viability model...")
    m2 = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=200,
        max_depth=5, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8
    )
    m2.fit(train_X, y_short.iloc[:split_idx])
    pred2 = m2.predict_proba(val_X)[:, 1]
    models["short"] = m2
    metrics["short_auc"] = roc_auc_score(y_short.iloc[split_idx:], pred2)
    print(f"  Validation AUC: {metrics['short_auc']:.4f}")
    
    # Model 3: Magnitude UP
    print("\n[3/4] Training UP magnitude model...")
    m3 = lgb.LGBMRegressor(
        objective="regression", metric="mae", n_estimators=200,
        max_depth=5, num_leaves=31, learning_rate=0.05
    )
    m3.fit(train_X, y_up_mag.iloc[:split_idx])
    pred3 = m3.predict(val_X)
    models["up_mag"] = m3
    metrics["up_mag_mae"] = mean_absolute_error(y_up_mag.iloc[split_idx:], pred3)
    print(f"  Validation MAE: {metrics['up_mag_mae']:.4f}")
    
    # Model 4: Magnitude DOWN
    print("\n[4/4] Training DOWN magnitude model...")
    m4 = lgb.LGBMRegressor(
        objective="regression", metric="mae", n_estimators=200,
        max_depth=5, num_leaves=31, learning_rate=0.05
    )
    m4.fit(train_X, y_down_mag.iloc[:split_idx])
    pred4 = m4.predict(val_X)
    models["down_mag"] = m4
    metrics["down_mag_mae"] = mean_absolute_error(y_down_mag.iloc[split_idx:], pred4)
    print(f"  Validation MAE: {metrics['down_mag_mae']:.4f}")
    
    return models, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pct", type=float, default=0.01, help="Target movement % (e.g., 0.01 = 1%)")
    parser.add_argument("--universe", type=str, default="nifty100")
    parser.add_argument("--output", type=str, default="models/intraday_model.pkl")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INTRADAY MOVEMENT PREDICTION MODEL")
    print("=" * 80)
    print(f"Target: +/-{args.target_pct*100:.1f}% movement from open")
    print(f"Universe: {args.universe}")
    print("=" * 80)
    
    # Load data builders
    print("\nLoading market and sentiment data...")
    market_builder = MarketFeatureBuilder()
    market_builder.download(start="2021-01-01", end="2024-12-31")
    
    sentiment_builder = SentimentFeatureBuilder(
        "sentiment/combined_sentiment_2015_2025.csv",
        market_builder=market_builder
    )
    sentiment_builder._load()
    print("✓ Data loaded")
    
    # Load stocks
    symbols = get_universe(args.universe)
    data_dir = Path("nifty500")
    
    print(f"\nProcessing {len(symbols)} stocks (2021-2024)...")
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
            minute_df = minute_df[(minute_df.index >= "2021-01-01") & (minute_df.index <= "2024-12-31")]
            
            if len(minute_df) < 1000:
                continue
            
            features, targets = compute_all_features(
                minute_df, symbol, market_builder, sentiment_builder, args.target_pct
            )
            
            if features is not None and len(features) > 100:
                features["symbol"] = symbol
                all_features.append(features)
                all_targets.append(targets)
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({len(features):4d} days)")
            else:
                print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗")
                
        except Exception as e:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✗ ({str(e)[:40]})")
    
    if not all_features:
        print("\n❌ No data loaded!")
        return
    
    # Combine
    X = pd.concat(all_features)
    y = pd.concat(all_targets)
    
    print(f"\n{'='*80}")
    print(f"Total samples: {len(X)}")
    print(f"Symbols: {X['symbol'].nunique()}")
    print(f"Long viable rate: {y['long_viable'].mean():.1%}")
    print(f"Short viable rate: {y['short_viable'].mean():.1%}")
    print(f"{'='*80}")
    
    # Train models
    feature_cols = [c for c in X.columns if c != "symbol"]
    models, metrics = train_models(
        X[feature_cols],
        y["long_viable"],
        y["short_viable"],
        y["max_up"],
        y["max_down"]
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "models": models,
            "features": feature_cols,
            "metrics": metrics,
            "target_pct": args.target_pct,
            "n_samples": len(X),
            "n_symbols": X["symbol"].nunique(),
        }, f)
    
    print(f"\n{'='*80}")
    print(f"✓ Model saved: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
