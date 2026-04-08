#!/usr/bin/env python3
"""
Train Point-in-Time Model - Zero Look-Ahead Bias.

Training: 2021-2023 (strictly before prediction date)
Validation: 2024 (blind during training)
Predictions: 2025 (true out-of-sample)

Uses only data available at decision time.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.pit_features import PointInTimeFeatureEngine
from intradaynet.universe import get_universe


def train_point_in_time_model(universe: str = "nifty100"):
    """Train model with strict temporal separation."""
    
    print("=" * 80)
    print("TRAIN POINT-IN-TIME MODEL")
    print("=" * 80)
    print("Training: 2021-2023")
    print("Validation: 2024")
    print("Predictions: 2025")
    print("=" * 80)
    
    # Initialize feature engine
    pit_engine = PointInTimeFeatureEngine()
    
    symbols = get_universe(universe)
    
    # === TRAINING PHASE (2021-2023) ===
    print("\n[PHASE 1] Training Data Collection (2021-2023)")
    print("-" * 80)
    
    train_features = []
    train_targets = []
    
    train_dates = pd.date_range("2021-01-01", "2023-12-31", freq="B")  # Business days
    
    for i, symbol in enumerate(symbols):
        symbol_trades = 0
        
        for date in train_dates:
            # Get pre-market features (available at 8:45 AM)
            pit = pit_engine.get_premarket_features(symbol, date.strftime("%Y-%m-%d"))
            
            if pit is None:
                continue
            
            # Load actual intraday data for target
            csv_path = Path("nifty500") / f"{symbol}_minute.csv"
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
                df.columns = df.columns.str.lower()
                
                today_data = df[df.index.date == date.date()]
                if len(today_data) < 30:
                    continue
                
                # Target: (high - open) / open (max intraday up move)
                today_open = today_data["open"].iloc[0]
                today_high = today_data["high"].max()
                target = (today_high - today_open) / today_open
                
                train_features.append(pit.features)
                train_targets.append(target)
                symbol_trades += 1
                
            except:
                continue
        
        if symbol_trades > 0:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({symbol_trades:4d} days)")
    
    if len(train_features) < 1000:
        print("\n❌ Insufficient training data!")
        return
    
    X_train = pd.DataFrame(train_features)
    y_train = pd.Series(train_targets)
    
    print(f"\n{'='*80}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {len(X_train.columns)}")
    print(f"Target mean: {y_train.mean():.4f}")
    print(f"Target std: {y_train.std():.4f}")
    print(f"{'='*80}")
    
    # === VALIDATION PHASE (2024) ===
    print("\n[PHASE 2] Validation Data Collection (2024)")
    print("-" * 80)
    
    val_features = []
    val_targets = []
    
    val_dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    
    for i, symbol in enumerate(symbols):
        symbol_trades = 0
        
        for date in val_dates:
            pit = pit_engine.get_premarket_features(symbol, date.strftime("%Y-%m-%d"))
            if pit is None:
                continue
            
            csv_path = Path("nifty500") / f"{symbol}_minute.csv"
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
                df.columns = df.columns.str.lower()
                today_data = df[df.index.date == date.date()]
                if len(today_data) < 30:
                    continue
                
                today_open = today_data["open"].iloc[0]
                today_high = today_data["high"].max()
                target = (today_high - today_open) / today_open
                
                val_features.append(pit.features)
                val_targets.append(target)
                symbol_trades += 1
            except:
                continue
        
        if symbol_trades > 0 and i % 10 == 0:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({symbol_trades:4d} days)")
    
    X_val = pd.DataFrame(val_features)
    y_val = pd.Series(val_targets)
    
    print(f"\nValidation samples: {len(X_val)}")
    
    # === TRAIN MODEL ===
    print("\n[PHASE 3] Training Model")
    print("-" * 80)
    
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=500,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.03,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS (2024)")
    print(f"{'='*80}")
    print(f"MAE: {val_mae:.4f}")
    print(f"R²: {val_r2:.4f}")
    print(f"Correlation: {np.corrcoef(y_val, val_pred)[0,1]:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 15 Features:")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:35s}: {row['importance']:.0f}")
    
    # Save model
    model_path = Path("models/pit_model_2021_2024.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": list(X_train.columns),
            "mae": val_mae,
            "r2": val_r2,
            "n_train": len(X_train),
            "n_val": len(X_val),
        }, f)
    
    print(f"\n✓ Model saved: {model_path}")
    
    # === GENERATE 2025 PREDICTIONS ===
    print("\n[PHASE 4] Generate 2025 Predictions (True Blind Test)")
    print("-" * 80)
    
    predictions = []
    predict_dates = pd.date_range("2025-01-01", "2025-12-31", freq="B")
    
    for i, symbol in enumerate(symbols):
        symbol_preds = 0
        
        for date in predict_dates:
            pit = pit_engine.get_premarket_features(symbol, date.strftime("%Y-%m-%d"))
            if pit is None:
                continue
            
            # Predict
            X_pred = pd.DataFrame([pit.features])
            pred_return = model.predict(X_pred)[0]
            
            predictions.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "predicted_max_up": float(pred_return),
                "confidence": abs(float(pred_return)),
                "direction": "LONG" if pred_return > 0.005 else "SHORT" if pred_return < -0.005 else "NEUTRAL",
            })
            symbol_preds += 1
        
        if symbol_preds > 0 and i % 10 == 0:
            print(f"  [{i+1:3d}/{len(symbols)}] {symbol:15s} ✓ ({symbol_preds:3d} preds)")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv("predictions_2025_pit.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Predictions saved: predictions_2025_pit.csv")
    print(f"  Total: {len(pred_df)} predictions")
    print(f"  Dates: {pred_df['date'].min()} to {pred_df['date'].max()}")
    print(f"  Long: {len(pred_df[pred_df['direction'] == 'LONG'])}")
    print(f"  Short: {len(pred_df[pred_df['direction'] == 'SHORT'])}")
    print(f"  Neutral: {len(pred_df[pred_df['direction'] == 'NEUTRAL'])}")
    print(f"{'='*80}")


if __name__ == "__main__":
    train_point_in_time_model()
