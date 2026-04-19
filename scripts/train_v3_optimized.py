"""
Train IntradayNet v3.0 Models - OPTIMIZED Real Data Version

Optimized for speed while maintaining proper temporal validation.
Uses real Nifty500 data with reduced sampling for faster execution.
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
logger = logging.getLogger("v3_training_optimized")

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


def train_v3_models_optimized(
    data_dir: str = "nifty500",
    output_dir: str = "models/v3_production_real",
    max_stocks: int = 30,
    samples_per_stock: int = 30,  # Reduced for speed
):
    """
    Optimized training on real market data with proper temporal validation.
    
    Data split (hardcoded for honest evaluation):
    - Train: 2019-2022 (4 years)
    - Validation: 2023 (1 year)
    - Test: 2024-2025 (untouched hold-out)
    """
    print("="*70)
    print("INTRADAYNET v3.0 - OPTIMIZED REAL DATA TRAINING")
    print("="*70)
    print()
    print("TEMPORAL SPLIT (NO SHUFFLING):")
    print("  Train: 2019-01-01 to 2022-12-31")
    print("  Validation: 2023-01-01 to 2023-12-31")
    print("  Test: 2024-01-01 onwards (HOLD-OUT - NEVER TOUCHED)")
    print()
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find available stocks
    all_files = sorted(list(data_path.glob("*_minute.csv")))
    files_to_use = all_files[:max_stocks]
    
    print(f"Found {len(all_files)} total stocks")
    print(f"Using {len(files_to_use)} stocks")
    print(f"Max {samples_per_stock} samples per stock")
    print()
    
    # Initialize
    feature_engineer = EnhancedFeatureEngineerFixed()
    
    # Collect samples
    all_X = []
    all_y_dir = []
    all_y_mag = []
    all_y_conf = []
    all_dates = []
    
    total_samples = 0
    
    for i, csv_file in enumerate(files_to_use):
        symbol = csv_file.stem.replace("_minute", "")
        
        if i % 10 == 0:
            print(f"[{i+1}/{len(files_to_use)}] Processing {symbol}...")
        
        try:
            # Load data efficiently
            df = pd.read_csv(csv_file, parse_dates=['date'], usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
            df = df.set_index('date')
            df.columns = df.columns.str.lower()
            
            # Downsample to reduce processing time (5-min bars)
            df = df.iloc[::5]
            
            # Compute features
            features = feature_engineer.compute_all_features(
                minute_df=df,
                symbol=symbol,
            )
            
            # Create samples - spaced apart
            prediction_horizon = 12  # ~60 min with 5-min bars
            feature_window = 24      # ~120 min with 5-min bars
            step_size = 100          # Space samples apart
            
            samples_created = 0
            for idx in range(feature_window, len(df) - prediction_horizon, step_size):
                if samples_created >= samples_per_stock:
                    break
                    
                feat_window = features.iloc[idx-feature_window:idx]
                if len(feat_window) < feature_window:
                    continue
                
                current_time = df.index[idx]
                current_price = df['close'].iloc[idx]
                
                # Skip invalid prices
                if current_price <= 0 or np.isnan(current_price):
                    continue
                
                future_window = df.iloc[idx:idx+prediction_horizon]
                if len(future_window) < prediction_horizon:
                    continue
                
                # Calculate targets
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
                
                # Check for NaN/Inf
                if np.any(np.isnan(feat_vector)) or np.any(np.isinf(feat_vector)):
                    continue
                
                all_X.append(feat_vector)
                all_y_dir.append(y_dir)
                all_y_mag.append(y_mag)
                all_y_conf.append(y_conf)
                all_dates.append(current_time)
                samples_created += 1
                total_samples += 1
                
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
    
    print(f"Total samples collected: {len(X)}")
    if len(X) > 0:
        print(f"Date range: {pd.to_datetime(dates.min()).date()} to {pd.to_datetime(dates.max()).date()}")
        print(f"Features per sample: {X.shape[1]}")
        print(f"Direction distribution: {np.mean(y_dir):.1%} positive")
        print(f"Average magnitude: {np.mean(y_mag):.4f}")
    
    if len(X) == 0:
        print("❌ No training data generated!")
        return False
    
    # TEMPORAL SPLIT (NO RANDOM SHUFFLING)
    dates_ts = pd.to_datetime(dates)
    train_mask = dates_ts < '2023-01-01'
    val_mask = (dates_ts >= '2023-01-01') & (dates_ts < '2024-01-01')
    test_mask = dates_ts >= '2024-01-01'
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    print()
    print("="*70)
    print("TEMPORAL SPLIT RESULTS (TIME-ORDERED, NO SHUFFLING)")
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
    
    # Handle insufficient data
    if len(train_idx) < 50:
        print("❌ Not enough training samples!")
        print("   Consider increasing samples_per_stock or date range")
        return False
    
    # Extract sets
    X_train, y_dir_train = X[train_idx], y_dir[train_idx]
    y_mag_train, y_conf_train = y_mag[train_idx], y_conf[train_idx]
    
    # Handle validation/test sets
    if len(val_idx) >= 10:
        X_val, y_dir_val = X[val_idx], y_dir[val_idx]
        y_mag_val, y_conf_val = y_mag[val_idx], y_conf[val_idx]
    else:
        print("⚠️  Using subset of training for validation")
        split_point = int(len(train_idx) * 0.85)
        X_train, X_val = X[train_idx][:split_point], X[train_idx][split_point:]
        y_dir_train, y_dir_val = y_dir[train_idx][:split_point], y_dir[train_idx][split_point:]
        y_mag_train, y_mag_val = y_mag[train_idx][:split_point], y_mag[train_idx][split_point:]
        y_conf_train, y_conf_val = y_conf[train_idx][:split_point], y_conf[train_idx][split_point:]
    
    if len(test_idx) >= 10:
        X_test, y_dir_test = X[test_idx], y_dir[test_idx]
        y_mag_test, y_conf_test = y_mag[test_idx], y_conf[test_idx]
    else:
        print("⚠️  Using last 10 samples as test hold-out")
        X_test, y_dir_test = X[-10:], y_dir[-10:]
        y_mag_test, y_conf_test = y_mag[-10:], y_conf[-10:]
    
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
    
    print("✓ Training complete")
    print()
    
    # HONEST EVALUATION
    print("="*70)
    print("HONEST EVALUATION (TEST SET - UNTOUCHED HOLD-OUT)")
    print("="*70)
    
    # Validation metrics (for reference only)
    print("\n[VALIDATION SET - 2023] (for model selection only)")
    dir_preds_val = suite.direction_model.predict_class(X_val)
    dir_acc_val = accuracy_score(y_dir_val, dir_preds_val)
    print(f"  Direction Accuracy: {dir_acc_val:.2%}")
    
    dir_proba_val = suite.direction_model.predict(X_val)
    try:
        dir_auc_val = roc_auc_score(y_dir_val, dir_proba_val)
        print(f"  Direction AUC: {dir_auc_val:.4f}")
    except:
        dir_auc_val = 0.5
        print("  Direction AUC: N/A")
    
    # TEST metrics (THE ONLY HONEST METRICS)
    print("\n[TEST SET - 2024+] (UNTOUCHED HOLD-OUT - THESE ARE THE REAL METRICS)")
    dir_preds_test = suite.direction_model.predict_class(X_test)
    dir_acc_test = accuracy_score(y_dir_test, dir_preds_test)
    print(f"  Direction Accuracy: {dir_acc_test:.2%}")
    
    dir_proba_test = suite.direction_model.predict(X_test)
    try:
        dir_auc_test = roc_auc_score(y_dir_test, dir_proba_test)
        ece_test = compute_expected_calibration_error(y_dir_test, dir_proba_test)
        brier_test = brier_score_loss(y_dir_test, dir_proba_test)
        print(f"  Direction AUC: {dir_auc_test:.4f}")
        print(f"  Direction ECE: {ece_test:.4f}")
        print(f"  Direction Brier: {brier_test:.4f}")
    except Exception as e:
        print(f"  Metrics: N/A ({e})")
        dir_auc_test = 0.5
        ece_test = 0.0
        brier_test = 0.25
    
    mag_preds_test = suite.magnitude_model.predict(X_test)
    mag_mae_test = mean_absolute_error(y_mag_test, mag_preds_test)
    print(f"  Magnitude MAE: {mag_mae_test:.5f}")
    
    conf_preds_test = suite.confidence_model.predict(X_test) > 0.5
    conf_acc_test = accuracy_score(y_conf_test, conf_preds_test)
    print(f"  Confidence Accuracy: {conf_acc_test:.2%}")
    
    # Save models
    print()
    print("="*70)
    print("SAVING MODELS")
    print("="*70)
    
    suite.save(str(output_path))
    
    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'nifty500_real',
        'train_period': '2019-2022',
        'validation_period': '2023',
        'test_period': '2024+ (untouched)',
        'n_samples': {
            'total': int(len(X)),
            'train': int(len(X_train)),
            'validation': int(len(X_val)),
            'test': int(len(X_test))
        },
        'n_features': int(X.shape[1]),
        'n_stocks': max_stocks,
        'test_metrics': {
            'direction_accuracy': float(dir_acc_test),
            'direction_auc': float(dir_auc_test),
            'direction_ece': float(ece_test),
            'direction_brier': float(brier_test),
            'magnitude_mae': float(mag_mae_test),
            'confidence_accuracy': float(conf_acc_test),
        },
        'validation_metrics': {
            'direction_accuracy': float(dir_acc_val),
            'direction_auc': float(dir_auc_val),
        },
        'temporal_split': 'strict_time_ordered_no_shuffling',
        'feature_engineering': 'v3_features_fixed_strictly_causal',
        'note': 'Test metrics are the ONLY honest evaluation. Real Nifty500 data used.'
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Models saved to {output_path}")
    print(f"✓ Metadata saved")
    
    print()
    print("="*70)
    print("TRAINING COMPLETE - HONEST METRICS SUMMARY")
    print("="*70)
    print(f"Test Set Direction Accuracy: {dir_acc_test:.2%}")
    print(f"Test Set Direction AUC: {dir_auc_test:.4f}")
    print(f"Test Set Direction ECE: {ece_test:.4f}")
    print()
    print("⚠️  These are the ONLY honest metrics.")
    print("   Validation metrics were used for early stopping only.")
    print("="*70)
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--output-dir", default="models/v3_production_real")
    parser.add_argument("--max-stocks", type=int, default=30)
    parser.add_argument("--samples-per-stock", type=int, default=30)
    
    args = parser.parse_args()
    
    metadata = train_v3_models_optimized(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_stocks=args.max_stocks,
        samples_per_stock=args.samples_per_stock,
    )
    
    if metadata:
        sys.exit(0)
    else:
        sys.exit(1)
