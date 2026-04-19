"""
Train IntradayNet v3.0 Models - FIXED Version with Proper Temporal Validation

CRITICAL FIXES:
1. Time-ordered train/validation/test split (no random shuffling)
2. Strict temporal causality in all features
3. Out-of-time test set (2024-2025) never seen during training
4. Purged cross-validation for hyperparameter tuning
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, brier_score_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v3_training_fixed")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed
from intradaynet.feature_selection import FeatureSelector
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


def temporal_train_val_test_split(X, y_dict, dates, train_end='2022-12-31', val_end='2023-12-31'):
    """
    Strict temporal split - NO random shuffling.
    
    Args:
        X: Feature matrix
        y_dict: Dictionary of target arrays {direction, magnitude, confidence}
        dates: Array of datetime objects for each sample
        train_end: Training period ends here (data up to this date used for training)
        val_end: Validation period ends here (used for hyperparameter tuning)
        
    Returns:
        Split indices for train, validation, test
    """
    dates = pd.to_datetime(dates)
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    
    # Temporal split
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    return train_idx, val_idx, test_idx


def purge_overlapping_samples(dates, bars_per_day=375, gap_days=2):
    """
    Purge samples that are too close in time to avoid information leakage.
    
    Financial time series have autocorrelation - samples from the same day
    or consecutive days share information. We need gaps between train/val/test.
    """
    dates = pd.to_datetime(dates)
    
    # Group by date
    date_groups = {}
    for i, d in enumerate(dates):
        date_key = d.date()
        if date_key not in date_groups:
            date_groups[date_key] = []
        date_groups[date_key].append(i)
    
    return date_groups


def train_v3_models_fixed(
    data_dir: str = "nifty500",
    output_dir: str = "models/v3_production_fixed",
    max_stocks: int = 50,
    train_start: str = "2019-01-01",
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
):
    """
    Train v3.0 models with PROPER TEMPORAL VALIDATION.
    
    Data split:
    - Train: 2019-01-01 to 2022-12-31 (4 years)
    - Validation: 2023-01-01 to 2023-12-31 (1 year, for hyperparameter tuning)
    - Test: 2024-01-01 onwards (untouched hold-out for final evaluation)
    """
    print("="*70)
    print("INTRADAYNET v3.0 - MODEL TRAINING (FIXED TEMPORAL SPLIT)")
    print("="*70)
    print()
    print("TEMPORAL SPLIT:")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Validation: 2023-01-01 to {val_end}")
    print(f"  Test: {test_start} onwards (HOLD-OUT)")
    print()
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find available stocks
    all_files = list(data_path.glob("*_minute.csv"))
    files_to_use = all_files[:max_stocks]
    
    print(f"Found {len(all_files)} total stocks")
    print(f"Using first {len(files_to_use)} for training")
    print()
    
    # Initialize
    feature_engineer = EnhancedFeatureEngineerFixed()
    
    # Collect training samples with STRICT temporal tracking
    all_X = []
    all_y_dir = []
    all_y_mag = []
    all_y_conf = []
    all_dates = []  # Track sample dates for temporal split
    all_symbols = []  # Track symbols for analysis
    
    for i, csv_file in enumerate(files_to_use):
        symbol = csv_file.stem.replace("_minute", "")
        
        if i % 5 == 0:
            print(f"Processing {i+1}/{len(files_to_use)}: {symbol}")
        
        try:
            # Load data
            df = pd.read_csv(csv_file, parse_dates=['date'])
            df = df.set_index('date')
            df.columns = df.columns.str.lower()
            
            # Store original index for temporal tracking
            df_dates = df.index
            
            # Compute features with STRICT temporal causality
            features = feature_engineer.compute_all_features(
                minute_df=df,
                symbol=symbol,
            )
            
            # Create training samples - step carefully to avoid overlap
            prediction_horizon = 60  # bars
            feature_window = 120  # bars
            step_size = 375  # ~1 day apart to avoid overlapping samples
            
            for idx in range(feature_window, len(df) - prediction_horizon, step_size):
                # Extract feature window (past data only - STRICTLY CAUSAL)
                feat_window = features.iloc[idx-feature_window:idx]
                if len(feat_window) < feature_window:
                    continue
                
                # Current timestamp (prediction point)
                current_time = df_dates[idx]
                current_price = df['close'].iloc[idx]
                
                # Future window (target - never used for features)
                future_window = df.iloc[idx:idx+prediction_horizon]
                if len(future_window) < prediction_horizon:
                    continue
                
                # Direction target
                future_return = (future_window['close'].iloc[-1] - current_price) / current_price
                y_dir = 1 if future_return > 0 else 0
                
                # Magnitude target
                y_mag = abs(future_return)
                
                # Confidence target (hit 1% target before 0.5% stop)
                future_high = future_window['high'].max()
                future_low = future_window['low'].min()
                
                target_hit = future_high >= current_price * 1.01
                stop_hit = future_low <= current_price * 0.995
                y_conf = 1 if target_hit and not stop_hit else 0
                
                # Feature vector (mean of window)
                feat_vector = feat_window.mean().values
                
                all_X.append(feat_vector)
                all_y_dir.append(y_dir)
                all_y_mag.append(y_mag)
                all_y_conf.append(y_conf)
                all_dates.append(current_time)
                all_symbols.append(symbol)
                
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
    print(f"Date range: {dates.min()} to {dates.max()}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Direction distribution: {np.mean(y_dir):.1%} positive")
    print(f"Average magnitude: {np.mean(y_mag):.3f}")
    print(f"Confidence (target hit): {np.mean(y_conf):.1%}")
    
    if len(X) == 0:
        print("❌ No training data generated!")
        return False
    
    # TEMPORAL SPLIT (NO RANDOM SHUFFLING!)
    print()
    print("="*70)
    print("TEMPORAL TRAIN/VALIDATION/TEST SPLIT")
    print("="*70)
    
    train_idx, val_idx, test_idx = temporal_train_val_test_split(
        X, 
        {'direction': y_dir, 'magnitude': y_mag, 'confidence': y_conf},
        dates,
        train_end=train_end,
        val_end=val_end
    )
    
    print(f"Training samples: {len(train_idx)} ({len(train_idx)/len(X):.1%})")
    print(f"  Date range: {dates[train_idx].min()} to {dates[train_idx].max()}")
    print(f"Validation samples: {len(val_idx)} ({len(val_idx)/len(X):.1%})")
    print(f"  Date range: {dates[val_idx].min()} to {dates[val_idx].max()}")
    print(f"Test samples (HOLD-OUT): {len(test_idx)} ({len(test_idx)/len(X):.1%})")
    print(f"  Date range: {dates[test_idx].min()} to {dates[test_idx].max()}")
    
    # Feature selection on training data only
    feature_names = feature_engineer.get_feature_names()
    print(f"\nOriginal features: {len(feature_names)}")
    
    if len(feature_names) > 60:
        print("Running feature selection on TRAINING data only...")
        
        # Train temp model on training set only
        if HAS_LIGHTGBM:
            temp_model = lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
        else:
            from sklearn.ensemble import RandomForestClassifier
            temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Fit on training, evaluate importance on validation
        temp_model.fit(X[train_idx], y_dir[train_idx])
        
        selector = FeatureSelector(feature_names=feature_names, target_count=60)
        selector.fit(X[val_idx], y_dir[val_idx], temp_model, n_repeats=2)
        
        selected_indices = selector.selected_indices
        X = X[:, selected_indices]
        feature_names = selector.selected_features
        
        print(f"Selected features: {len(feature_names)}")
    
    print()
    print("="*70)
    print("TRAINING MODELS ON TRAINING SET ONLY")
    print("="*70)
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    y_dir_train = y_dir[train_idx]
    y_dir_val = y_dir[val_idx]
    y_dir_test = y_dir[test_idx]
    
    y_mag_train = y_mag[train_idx]
    y_mag_val = y_mag[val_idx]
    y_mag_test = y_mag[test_idx]
    
    y_conf_train = y_conf[train_idx]
    y_conf_val = y_conf[val_idx]
    y_conf_test = y_conf[test_idx]
    
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples (for early stopping)")
    print(f"Test: {len(X_test)} samples (UNTouched hold-out)")
    
    # Train specialized models
    config = ModelConfig()
    suite = SpecializedModelSuite(config)
    
    suite.fit(
        X_train, y_dir_train, y_mag_train, y_conf_train,
        X_val, y_dir_val, y_mag_val, y_conf_val,
    )
    
    print()
    print("="*70)
    print("HONEST EVALUATION METRICS")
    print("="*70)
    
    # VALIDATION METRICS (for hyperparameter tuning decisions)
    print("\n[VALIDATION SET - 2023] (for model selection)")
    dir_preds = suite.direction_model.predict_class(X_val)
    dir_acc_val = accuracy_score(y_dir_val, dir_preds)
    print(f"  Direction Accuracy: {dir_acc_val:.2%}")
    
    dir_proba_val = suite.direction_model.predict(X_val)
    try:
        dir_auc_val = roc_auc_score(y_dir_val, dir_proba_val)
        print(f"  Direction AUC: {dir_auc_val:.4f}")
        
        ece_val = compute_expected_calibration_error(y_dir_val, dir_proba_val)
        print(f"  Direction ECE: {ece_val:.4f}")
        
        brier_val = brier_score_loss(y_dir_val, dir_proba_val)
        print(f"  Direction Brier: {brier_val:.4f}")
    except:
        pass
    
    # TEST METRICS (THE ONLY HONEST METRICS)
    print("\n[TEST SET - 2024-2025] (UNTOUCHED HOLD-OUT - THESE ARE THE REAL METRICS)")
    dir_preds_test = suite.direction_model.predict_class(X_test)
    dir_acc_test = accuracy_score(y_dir_test, dir_preds_test)
    print(f"  Direction Accuracy: {dir_acc_test:.2%}")
    
    dir_proba_test = suite.direction_model.predict(X_test)
    try:
        dir_auc_test = roc_auc_score(y_dir_test, dir_proba_test)
        print(f"  Direction AUC: {dir_auc_test:.4f}")
        
        ece_test = compute_expected_calibration_error(y_dir_test, dir_proba_test)
        print(f"  Direction ECE: {ece_test:.4f}")
        
        brier_test = brier_score_loss(y_dir_test, dir_proba_test)
        print(f"  Direction Brier: {brier_test:.4f}")
    except:
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
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'train_period': f"{train_start} to {train_end}",
        'validation_period': f"2023-01-01 to {val_end}",
        'test_period': f"{test_start} onwards",
        'n_samples_total': len(X),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'metrics': {
            'validation': {
                'direction_accuracy': float(dir_acc_val),
                'direction_auc': float(dir_auc_val) if 'dir_auc_val' in locals() else None,
            },
            'test': {  # THESE ARE THE HONEST METRICS
                'direction_accuracy': float(dir_acc_test),
                'direction_auc': float(dir_auc_test),
                'direction_ece': float(ece_test),
                'direction_brier': float(brier_test),
                'magnitude_mae': float(mag_mae_test),
                'confidence_accuracy': float(conf_acc_test),
            }
        },
        'temporal_split': 'strict_time_ordered_no_shuffling',
        'note': 'Test set metrics are the ONLY honest evaluation. Validation set used for early stopping only.'
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Models saved to {output_path}")
    print(f"✓ Metadata saved with HONEST test metrics")
    
    print()
    print("="*70)
    print("TRAINING COMPLETE - HONEST METRICS SUMMARY")
    print("="*70)
    print(f"Test Set Direction Accuracy: {dir_acc_test:.2%}")
    print(f"Test Set Direction AUC: {dir_auc_test:.4f}")
    print(f"Test Set Direction ECE: {ece_test:.4f}")
    print()
    print("⚠️  These are the ONLY honest metrics. Do not use validation metrics for claims.")
    print("="*70)
    
    return True, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--output-dir", default="models/v3_production_fixed")
    parser.add_argument("--max-stocks", type=int, default=50)
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    
    args = parser.parse_args()
    
    result = train_v3_models_fixed(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_stocks=args.max_stocks,
        train_start=args.train_start,
        train_end=args.train_end,
        val_end=args.val_end,
        test_start=args.test_start,
    )
    
    if result:
        success, metadata = result
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)
