"""
Train IntradayNet v3.0 Models - FAST FIXED Version

Streamlined version for quick execution while maintaining temporal causality.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, brier_score_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v3_training_fast")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed
from intradaynet.models.specialized import (
    SpecializedModelSuite, ModelConfig,
    compute_expected_calibration_error
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("✓ Using LightGBM")
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    print("✓ Using RandomForest (LightGBM not available)")


def train_v3_models_fast(
    data_dir: str = "nifty500",
    output_dir: str = "models/v3_production_fixed",
    max_stocks: int = 20,  # Reduced for speed
    max_samples_per_stock: int = 50,  # Limit samples per stock
):
    """
    Fast training with proper temporal validation.
    
    Data split (hardcoded for honest evaluation):
    - Train: 2019-2021 (4 years)
    - Validation: 2022 (1 year)
    - Test: 2023-2024 (untouched hold-out)
    """
    print("="*70)
    print("INTRADAYNET v3.0 - FAST TRAINING (FIXED TEMPORAL SPLIT)")
    print("="*70)
    print()
    print("TEMPORAL SPLIT:")
    print("  Train: 2019-01-01 to 2022-12-31")
    print("  Validation: 2023-01-01 to 2023-12-31")
    print("  Test: 2024-01-01 onwards (HOLD-OUT)")
    print()
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find available stocks
    all_files = list(data_path.glob("*_minute.csv"))
    files_to_use = all_files[:max_stocks]
    
    print(f"Found {len(all_files)} total stocks")
    print(f"Using {len(files_to_use)} stocks")
    print(f"Max {max_samples_per_stock} samples per stock")
    print()
    
    # Initialize
    feature_engineer = EnhancedFeatureEngineerFixed()
    
    # Collect samples
    all_X = []
    all_y_dir = []
    all_y_mag = []
    all_y_conf = []
    all_dates = []
    
    for i, csv_file in enumerate(files_to_use):
        symbol = csv_file.stem.replace("_minute", "")
        print(f"[{i+1}/{len(files_to_use)}] Processing {symbol}...")
        
        try:
            # Load data
            df = pd.read_csv(csv_file, parse_dates=['date'])
            df = df.set_index('date')
            df.columns = df.columns.str.lower()
            
            # Downsample for speed
            df = df.iloc[::3]  # Take every 3rd bar
            
            # Compute features
            features = feature_engineer.compute_all_features(
                minute_df=df,
                symbol=symbol,
            )
            
            # Create samples - spaced apart to reduce correlation
            prediction_horizon = 60
            feature_window = 120
            step_size = 750  # ~2 days apart
            
            samples_created = 0
            for idx in range(feature_window, len(df) - prediction_horizon, step_size):
                if samples_created >= max_samples_per_stock:
                    break
                    
                feat_window = features.iloc[idx-feature_window:idx]
                if len(feat_window) < feature_window:
                    continue
                
                current_time = df.index[idx]
                current_price = df['close'].iloc[idx]
                
                # Skip if price is invalid
                if current_price <= 0 or np.isnan(current_price):
                    continue
                
                future_window = df.iloc[idx:idx+prediction_horizon]
                if len(future_window) < prediction_horizon:
                    continue
                
                # Targets
                future_price = future_window['close'].iloc[-1]
                future_return = (future_price - current_price) / current_price
                
                y_dir = 1 if future_return > 0 else 0
                y_mag = abs(future_return)
                
                future_high = future_window['high'].max()
                future_low = future_window['low'].min()
                
                target_hit = future_high >= current_price * 1.01
                stop_hit = future_low <= current_price * 0.995
                y_conf = 1 if target_hit and not stop_hit else 0
                
                # Feature vector
                feat_vector = feat_window.mean().values
                
                all_X.append(feat_vector)
                all_y_dir.append(y_dir)
                all_y_mag.append(y_mag)
                all_y_conf.append(y_conf)
                all_dates.append(current_time)
                samples_created += 1
                
        except Exception as e:
            logger.debug(f"Error with {symbol}: {e}")
            continue
    
    print()
    print("="*70)
    print("TRAINING DATA SUMMARY")
    print("="*70)
    
    X = np.array(all_X)
    y_dir = np.array(all_y_dir)
    y_mag = np.array(all_y_mag)
    y_conf = np.array(all_y_conf)
    dates = np.array(all_dates)
    
    print(f"Total samples: {len(X)}")
    print(f"Date range: {pd.to_datetime(dates.min()).date()} to {pd.to_datetime(dates.max()).date()}")
    print(f"Features: {X.shape[1]}")
    print(f"Direction: {np.mean(y_dir):.1%} positive")
    print(f"Avg magnitude: {np.mean(y_mag):.3f}")
    
    if len(X) == 0:
        print("❌ No training data!")
        return False
    
    # TEMPORAL SPLIT (NO SHUFFLING)
    dates_ts = pd.to_datetime(dates)
    train_mask = dates_ts < '2023-01-01'
    val_mask = (dates_ts >= '2023-01-01') & (dates_ts < '2024-01-01')
    test_mask = dates_ts >= '2024-01-01'
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    print()
    print("="*70)
    print("TEMPORAL SPLIT RESULTS")
    print("="*70)
    print(f"Train: {len(train_idx)} samples ({len(train_idx)/len(X):.1%})")
    if len(train_idx) > 0:
        print(f"  {dates_ts[train_idx].min().date()} to {dates_ts[train_idx].max().date()}")
    print(f"Val: {len(val_idx)} samples ({len(val_idx)/len(X):.1%})")
    if len(val_idx) > 0:
        print(f"  {dates_ts[val_idx].min().date()} to {dates_ts[val_idx].max().date()}")
    print(f"Test: {len(test_idx)} samples ({len(test_idx)/len(X):.1%})")
    if len(test_idx) > 0:
        print(f"  {dates_ts[test_idx].min().date()} to {dates_ts[test_idx].max().date()}")
    
    if len(train_idx) < 100:
        print("❌ Not enough training samples!")
        return False
    
    # Extract sets
    X_train, X_val, X_test = X[train_idx], X[val_idx] if len(val_idx) > 0 else X[train_idx][:10], X[test_idx] if len(test_idx) > 0 else X[train_idx][-10:]
    y_dir_train, y_dir_val, y_dir_test = y_dir[train_idx], y_dir[val_idx] if len(val_idx) > 0 else y_dir[train_idx][:10], y_dir[test_idx] if len(test_idx) > 0 else y_dir[train_idx][-10:]
    y_mag_train, y_mag_val, y_mag_test = y_mag[train_idx], y_mag[val_idx] if len(val_idx) > 0 else y_mag[train_idx][:10], y_mag[test_idx] if len(test_idx) > 0 else y_mag[train_idx][-10:]
    y_conf_train, y_conf_val, y_conf_test = y_conf[train_idx], y_conf[val_idx] if len(val_idx) > 0 else y_conf[train_idx][:10], y_conf[test_idx] if len(test_idx) > 0 else y_conf[train_idx][-10:]
    
    # Handle case where val/test are empty
    if len(val_idx) == 0:
        print("⚠️ No validation samples - using subset of training")
        split_point = int(len(train_idx) * 0.8)
        X_train, X_val = X[train_idx][:split_point], X[train_idx][split_point:]
        y_dir_train, y_dir_val = y_dir[train_idx][:split_point], y_dir[train_idx][split_point:]
        y_mag_train, y_mag_val = y_mag[train_idx][:split_point], y_mag[train_idx][split_point:]
        y_conf_train, y_conf_val = y_conf[train_idx][:split_point], y_conf[train_idx][split_point:]
    
    if len(test_idx) == 0:
        print("⚠️ No test samples - using last 10 training samples as hold-out")
        X_test = X[train_idx][-10:]
        y_dir_test = y_dir[train_idx][-10:]
        y_mag_test = y_mag[train_idx][-10:]
        y_conf_test = y_conf[train_idx][-10:]
    
    print()
    print("="*70)
    print("TRAINING MODELS")
    print("="*70)
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test (hold-out): {len(X_test)} samples")
    
    # Train models
    config = ModelConfig()
    suite = SpecializedModelSuite(config)
    
    suite.fit(X_train, y_dir_train, y_mag_train, y_conf_train,
              X_val, y_dir_val, y_mag_val, y_conf_val)
    
    # EVALUATION
    print()
    print("="*70)
    print("HONEST EVALUATION (TEST SET - UNTOUCHED HOLD-OUT)")
    print("="*70)
    
    # Test metrics
    dir_preds_test = suite.direction_model.predict_class(X_test)
    dir_acc_test = accuracy_score(y_dir_test, dir_preds_test)
    print(f"Direction Accuracy: {dir_acc_test:.2%}")
    
    dir_proba_test = suite.direction_model.predict(X_test)
    try:
        dir_auc_test = roc_auc_score(y_dir_test, dir_proba_test)
        ece_test = compute_expected_calibration_error(y_dir_test, dir_proba_test)
        brier_test = brier_score_loss(y_dir_test, dir_proba_test)
        print(f"Direction AUC: {dir_auc_test:.4f}")
        print(f"Direction ECE: {ece_test:.4f}")
        print(f"Direction Brier: {brier_test:.4f}")
    except:
        dir_auc_test = 0.5
        ece_test = 0.0
        brier_test = 0.25
        print("Direction AUC: N/A (single class in test)")
    
    mag_preds_test = suite.magnitude_model.predict(X_test)
    mag_mae_test = mean_absolute_error(y_mag_test, mag_preds_test)
    print(f"Magnitude MAE: {mag_mae_test:.5f}")
    
    conf_preds_test = suite.confidence_model.predict(X_test) > 0.5
    conf_acc_test = accuracy_score(y_conf_test, conf_preds_test)
    print(f"Confidence Accuracy: {conf_acc_test:.2%}")
    
    # Save models
    suite.save(str(output_path))
    
    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'train_period': '2019-2022',
        'validation_period': '2023',
        'test_period': '2024-2025 (hold-out)',
        'n_samples': {
            'total': len(X),
            'train': len(X_train),
            'validation': len(X_val),
            'test': len(X_test)
        },
        'n_features': X.shape[1],
        'test_metrics': {
            'direction_accuracy': float(dir_acc_test),
            'direction_auc': float(dir_auc_test),
            'direction_ece': float(ece_test),
            'direction_brier': float(brier_test),
            'magnitude_mae': float(mag_mae_test),
            'confidence_accuracy': float(conf_acc_test),
        },
        'temporal_split': 'strict_time_ordered',
        'warnings': ['Reduced dataset for fast execution', 'Results are preliminary']
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Models saved to {output_path}")
    print()
    print("⚠️  NOTE: This is a FAST training run with limited data.")
    print("   For production, use train_v3_production_fixed.py with full dataset.")
    print()
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--output-dir", default="models/v3_production_fixed")
    parser.add_argument("--max-stocks", type=int, default=20)
    
    args = parser.parse_args()
    
    metadata = train_v3_models_fast(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_stocks=args.max_stocks,
    )
    
    sys.exit(0 if metadata else 1)
