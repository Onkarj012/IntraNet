#!/usr/bin/env python3
"""
QUICK TRAINING - NIFTY50, 2023 Only (30-60 min runtime).

For fast testing before full training.

Training: 2023 (1 year)
Validation: 2024 Q1 (3 months)
Predictions: 2025 (full year)
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


def train_quick():
    """Quick training on 2023 only."""
    
    print("=" * 80)
    print("QUICK TRAINING - NIFTY50, 2023 ONLY")
    print("=" * 80)
    print("Training: 2023 (1 year)")
    print("Validation: 2024 Q1 (Jan-Mar)")
    print("Predictions: 2025")
    print("=" * 80)
    
    # Initialize feature engine
    pit_engine = PointInTimeFeatureEngine()
    
    symbols = get_universe("nifty50")
    
    # === TRAINING PHASE (2023 only) ===
    print("\n[PHASE 1] Training Data Collection (2023)")
    print("-" * 80)
    
    train_features = []
    train_targets = []
    
    train_dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    
    for i, symbol in enumerate(symbols):
        symbol_trades = 0
        
        for date in train_dates:
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
                
                # Target: (high - open) / open
                today_open = today_data["open"].iloc[0]
                today_high = today_data["high"].max()
                target = (today_high - today_open) / today_open
                
                train_features.append(pit.features)
                train_targets.append(target)
                symbol_trades += 1
                
            except:
                continue
        
        if symbol_trades > 0:
            print(f"  [{i+1:2d}/{len(symbols)}] {symbol:15s} ✓ ({symbol_trades:3d} days)")
    
    if len(train_features) < 500:
        print("\n❌ Insufficient training data!")
        return
    
    X_train = pd.DataFrame(train_features)
    y_train = pd.Series(train_targets)
    
    print(f"\n{'='*80}")
    print(f"Training samples: {len(X_train)}")
    print(f"Target mean: {y_train.mean():.4f}")
    print(f"Target std: {y_train.std():.4f}")
    print(f"{'='*80}")
    
    # === VALIDATION PHASE (2024 Q1 only) ===
    print("\n[PHASE 2] Validation Data Collection (2024 Q1)")
    print("-" * 80)
    
    val_features = []
    val_targets = []
    
    val_dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")  # Q1 only
    
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
        
        if symbol_trades > 0 and i % 5 == 0:
            print(f"  [{i+1:2d}/{len(symbols)}] {symbol:15s} ✓ ({symbol_trades:3d} days)")
    
    X_val = pd.DataFrame(val_features)
    y_val = pd.Series(val_targets)
    
    print(f"\nValidation samples: {len(X_val)}")
    
    # === TRAIN MODEL ===
    print("\n[PHASE 3] Training Model")
    print("-" * 80)
    
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=200,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=True)],
    )
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS (2024 Q1)")
    print(f"{'='*80}")
    print(f"MAE: {val_mae:.4f}")
    print(f"R²: {val_r2:.4f}")
    print(f"Correlation: {np.corrcoef(y_val, val_pred)[0,1]:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:35s}: {row['importance']:.0f}")
    
    # Save model
    model_path = Path("models/pit_model_quick.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": list(X_train.columns),
            "mae": val_mae,
            "r2": val_r2,
            "train_period": "2023",
            "val_period": "2024 Q1",
        }, f)
    
    print(f"\n✓ Model saved: {model_path}")
    
    # === GENERATE 2025 PREDICTIONS ===
    print("\n[PHASE 4] Generate 2025 Predictions")
    print("-" * 80)
    
    predictions = []
    predict_dates = pd.date_range("2025-01-01", "2025-12-31", freq="B")
    
    for i, symbol in enumerate(symbols):
        symbol_preds = 0
        
        for date in predict_dates:
            pit = pit_engine.get_premarket_features(symbol, date.strftime("%Y-%m-%d"))
            if pit is None:
                continue
            
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
        
        if symbol_preds > 0 and i % 5 == 0:
            print(f"  [{i+1:2d}/{len(symbols)}] {symbol:15s} ✓ ({symbol_preds:3d} preds)")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv("predictions_2025_quick.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Predictions saved: predictions_2025_quick.csv")
    print(f"  Total: {len(pred_df)} predictions")
    print(f"  Long: {len(pred_df[pred_df['direction'] == 'LONG'])}")
    print(f"  Short: {len(pred_df[pred_df['direction'] == 'SHORT'])}")
    print(f"{'='*80}")


if __name__ == "__main__":
    train_quick()
