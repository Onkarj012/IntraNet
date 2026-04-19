"""
Train IntradayNet v3.0 Models - Production Version

Streamlined training on actual market data.
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v3_training")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.features.v3_features import EnhancedFeatureEngineer
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


def train_v3_models(
    data_dir: str = "nifty500",
    output_dir: str = "models/v3_production",
    max_stocks: int = 50,
    train_start: str = "2023-01-01",
    train_end: str = "2024-12-31",
):
    """
    Train v3.0 models on real market data.
    """
    print("="*70)
    print("INTRADAYNET v3.0 - MODEL TRAINING")
    print("="*70)
    print()
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find available stocks
    all_files = list(data_path.glob("*_minute.csv"))
    # Use first max_stocks
    files_to_use = all_files[:max_stocks]
    
    print(f"Found {len(all_files)} total stocks")
    print(f"Using first {len(files_to_use)} for training")
    print()
    
    # Initialize
    feature_engineer = EnhancedFeatureEngineer()
    
    # Collect training samples
    all_X = []
    all_y_dir = []
    all_y_mag = []
    all_y_conf = []
    
    for i, csv_file in enumerate(files_to_use):
        symbol = csv_file.stem.replace("_minute", "")
        
        if i % 5 == 0:
            print(f"Processing {i+1}/{len(files_to_use)}: {symbol}")
        
        try:
            # Load data
            df = pd.read_csv(csv_file, parse_dates=['date'])
            df = df.set_index('date')
            df.columns = df.columns.str.lower()
            
            # Filter to training period
            df = df[(df.index >= train_start) & (df.index <= train_end)]
            
            if len(df) < 500:
                continue
            
            # Sample every Nth bar to reduce data (for demo)
            df = df.iloc[::5]  # Take every 5th bar
            
            # Compute features
            features = feature_engineer.compute_all_features(
                minute_df=df,
                symbol=symbol,
            )
            
            # Create training samples
            for idx in range(120, len(df) - 60, 20):  # Step by 20
                feat_window = features.iloc[idx-120:idx]
                if len(feat_window) < 120:
                    continue
                
                current_price = df['close'].iloc[idx]
                
                # Future 60 bars
                future_window = df.iloc[idx:idx+60]
                if len(future_window) < 60:
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
    
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Direction distribution: {np.mean(y_dir):.1%} positive")
    print(f"Average magnitude: {np.mean(y_mag):.3f}")
    print(f"Confidence (target hit): {np.mean(y_conf):.1%}")
    
    if len(X) == 0:
        print("❌ No training data generated!")
        return False
    
    # Feature selection
    feature_names = feature_engineer.get_feature_names()
    print(f"\nOriginal features: {len(feature_names)}")
    
    if len(feature_names) > 60:
        print("Running feature selection...")
        
        # Train temp model for importance
        if HAS_LIGHTGBM:
            temp_model = lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier
            temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        X_tr, X_val, y_tr, y_val = train_test_split(X, y_dir, test_size=0.2, random_state=42)
        temp_model.fit(X_tr, y_tr)
        
        selector = FeatureSelector(feature_names=feature_names, target_count=60)
        selector.fit(X_val, y_val, temp_model, n_repeats=2)
        
        selected_indices = selector.selected_indices
        X = X[:, selected_indices]
        feature_names = selector.selected_features
        
        print(f"Selected features: {len(feature_names)}")
    
    print()
    print("="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # Split data
    X_train, X_val, y_dir_train, y_dir_val = train_test_split(
        X, y_dir, test_size=0.2, random_state=42
    )
    _, _, y_mag_train, y_mag_val = train_test_split(
        X, y_mag, test_size=0.2, random_state=42
    )
    _, _, y_conf_train, y_conf_val = train_test_split(
        X, y_conf, test_size=0.2, random_state=42
    )
    
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    
    # Train specialized models
    config = ModelConfig()
    suite = SpecializedModelSuite(config)
    
    suite.fit(
        X_train, y_dir_train, y_mag_train, y_conf_train,
        X_val, y_dir_val, y_mag_val, y_conf_val,
    )
    
    print()
    print("="*70)
    print("VALIDATION METRICS")
    print("="*70)
    
    # Evaluate
    dir_preds = suite.direction_model.predict_class(X_val)
    dir_acc = accuracy_score(y_dir_val, dir_preds)
    print(f"Direction Accuracy: {dir_acc:.2%}")
    
    dir_proba = suite.direction_model.predict(X_val)
    try:
        dir_auc = roc_auc_score(y_dir_val, dir_proba)
        print(f"Direction AUC: {dir_auc:.4f}")
        
        ece = compute_expected_calibration_error(y_dir_val, dir_proba)
        print(f"Direction ECE: {ece:.4f}")
        
        brier = brier_score_loss(y_dir_val, dir_proba)
        print(f"Direction Brier: {brier:.4f}")
    except:
        pass
    
    mag_preds = suite.magnitude_model.predict(X_val)
    mag_mae = mean_absolute_error(y_mag_val, mag_preds)
    print(f"Magnitude MAE: {mag_mae:.5f}")
    
    conf_preds = suite.confidence_model.predict(X_val) > 0.5
    conf_acc = accuracy_score(y_conf_val, conf_preds)
    print(f"Confidence Accuracy: {conf_acc:.2%}")
    
    # Save models
    print()
    print("="*70)
    print("SAVING MODELS")
    print("="*70)
    
    suite.save(str(output_path))
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'train_start': train_start,
        'train_end': train_end,
        'n_samples': len(X),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'metrics': {
            'direction_accuracy': float(dir_acc),
            'magnitude_mae': float(mag_mae),
            'confidence_accuracy': float(conf_acc),
        }
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Models saved to {output_path}")
    print(f"✓ Metadata saved")
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="nifty500")
    parser.add_argument("--output-dir", default="models/v3_production")
    parser.add_argument("--max-stocks", type=int, default=50)
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    
    args = parser.parse_args()
    
    success = train_v3_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_stocks=args.max_stocks,
        train_start=args.train_start,
        train_end=args.train_end,
    )
    
    sys.exit(0 if success else 1)
