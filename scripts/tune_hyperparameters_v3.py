"""
Hyperparameter Tuning for IntradayNet v3.0

Uses Optuna to optimize LightGBM hyperparameters for:
1. Direction model (maximize AUC)
2. Magnitude model (minimize MAE)
3. Confidence model (maximize accuracy)

Usage:
    python scripts/tune_hyperparameters_v3.py --n-trials 50
"""

import sys
import json
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
logger = logging.getLogger("hyperparameter_tuning")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    print("✓ Using Optuna for hyperparameter optimization")
except ImportError:
    HAS_OPTUNA = False
    print("⚠ Optuna not available, using grid search fallback")

from intradaynet.models.specialized import SpecializedModelSuite, ModelConfig

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class HyperparameterTuner:
    """Tune hyperparameters for v3.0 models."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_dir_train: np.ndarray,
        y_mag_train: np.ndarray,
        y_conf_train: np.ndarray,
        X_val: np.ndarray,
        y_dir_val: np.ndarray,
        y_mag_val: np.ndarray,
        y_conf_val: np.ndarray,
        output_dir: str = "models/v3_tuned",
    ):
        self.X_train = X_train
        self.y_dir_train = y_dir_train
        self.y_mag_train = y_mag_train
        self.y_conf_train = y_conf_train
        
        self.X_val = X_val
        self.y_dir_val = y_dir_val
        self.y_mag_val = y_mag_val
        self.y_conf_val = y_conf_val
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_params = {}
        
        logger.info(f"Hyperparameter Tuner initialized")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Features: {X_train.shape[1]}")
    
    def objective_direction(self, trial):
        """Optuna objective for direction model."""
        if not HAS_LIGHTGBM:
            # RandomForest fallback
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        else:
            # LightGBM
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 15, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 500),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1,
            }
            model = lgb.LGBMClassifier(**params)
        
        # Train
        model.fit(self.X_train, self.y_dir_train)
        
        # Evaluate
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(self.X_val)[:, 1]
            try:
                auc = roc_auc_score(self.y_dir_val, y_proba)
            except:
                auc = 0.5
        else:
            y_pred = model.predict(self.X_val)
            auc = accuracy_score(self.y_dir_val, y_pred)
        
        return auc
    
    def objective_magnitude(self, trial):
        """Optuna objective for magnitude model."""
        if not HAS_LIGHTGBM:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 15, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 500),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1,
            }
            model = lgb.LGBMRegressor(**params)
        
        model.fit(self.X_train, self.y_mag_train)
        y_pred = model.predict(self.X_val)
        mae = mean_absolute_error(self.y_mag_val, y_pred)
        
        return -mae  # Minimize MAE = maximize negative MAE
    
    def objective_confidence(self, trial):
        """Optuna objective for confidence model."""
        if not HAS_LIGHTGBM:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 64),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 300),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1,
            }
            model = lgb.LGBMClassifier(**params)
        
        model.fit(self.X_train, self.y_conf_train)
        y_pred = model.predict(self.X_val)
        acc = accuracy_score(self.y_conf_val, y_pred)
        
        return acc
    
    def tune_with_optuna(self, n_trials: int = 50):
        """Run Optuna hyperparameter optimization."""
        logger.info("="*70)
        logger.info("HYPERPARAMETER TUNING WITH OPTUNA")
        logger.info("="*70)
        
        results = {}
        
        # 1. Tune Direction Model
        logger.info("\nTuning Direction Model...")
        study_dir = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name='direction_model'
        )
        study_dir.optimize(self.objective_direction, n_trials=n_trials, show_progress_bar=True)
        
        results['direction'] = {
            'best_auc': study_dir.best_value,
            'best_params': study_dir.best_params,
        }
        
        logger.info(f"  Best AUC: {study_dir.best_value:.4f}")
        logger.info(f"  Best params: {study_dir.best_params}")
        
        # 2. Tune Magnitude Model
        logger.info("\nTuning Magnitude Model...")
        study_mag = optuna.create_study(
            direction='maximize',  # We return -MAE
            sampler=TPESampler(seed=42),
            study_name='magnitude_model'
        )
        study_mag.optimize(self.objective_magnitude, n_trials=n_trials, show_progress_bar=True)
        
        results['magnitude'] = {
            'best_mae': -study_mag.best_value,
            'best_params': study_mag.best_params,
        }
        
        logger.info(f"  Best MAE: {-study_mag.best_value:.5f}")
        logger.info(f"  Best params: {study_mag.best_params}")
        
        # 3. Tune Confidence Model
        logger.info("\nTuning Confidence Model...")
        study_conf = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name='confidence_model'
        )
        study_conf.optimize(self.objective_confidence, n_trials=n_trials, show_progress_bar=True)
        
        results['confidence'] = {
            'best_accuracy': study_conf.best_value,
            'best_params': study_conf.best_params,
        }
        
        logger.info(f"  Best Accuracy: {study_conf.best_value:.2%}")
        logger.info(f"  Best params: {study_conf.best_params}")
        
        # Save results
        self.best_params = results
        
        output_file = self.output_dir / "best_hyperparameters.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Best hyperparameters saved to {output_file}")
        
        return results
    
    def tune_with_grid_search(self):
        """Fallback grid search if Optuna not available."""
        logger.info("="*70)
        logger.info("HYPERPARAMETER TUNING (GRID SEARCH)")
        logger.info("="*70)
        
        # Simple grid search for key parameters
        best_dir_auc = 0
        best_dir_params = {}
        
        if not HAS_LIGHTGBM:
            # RandomForest grid
            param_grid = [
                {'n_estimators': 50, 'max_depth': 5},
                {'n_estimators': 100, 'max_depth': 7},
                {'n_estimators': 150, 'max_depth': 10},
            ]
        else:
            # LightGBM grid
            param_grid = [
                {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},
                {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03},
                {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1},
            ]
        
        logger.info("\nTesting Direction Model...")
        for i, params in enumerate(param_grid):
            if not HAS_LIGHTGBM:
                model = RandomForestClassifier(**params, random_state=42)
            else:
                params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
                model = lgb.LGBMClassifier(**params)
            
            model.fit(self.X_train, self.y_dir_train)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_val)[:, 1]
                try:
                    auc = roc_auc_score(self.y_dir_val, y_proba)
                except:
                    auc = 0.5
            else:
                y_pred = model.predict(self.X_val)
                auc = accuracy_score(self.y_dir_val, y_pred)
            
            logger.info(f"  Config {i+1}: AUC = {auc:.4f}")
            
            if auc > best_dir_auc:
                best_dir_auc = auc
                best_dir_params = params
        
        results = {
            'direction': {
                'best_auc': best_dir_auc,
                'best_params': best_dir_params,
            }
        }
        
        logger.info(f"\n✓ Best Direction AUC: {best_dir_auc:.4f}")
        logger.info(f"  Params: {best_dir_params}")
        
        # Save
        self.best_params = results
        output_file = self.output_dir / "best_hyperparameters.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_tuning(self, n_trials: int = 50):
        """Run full hyperparameter tuning."""
        logger.info("\n" + "="*70)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*70)
        
        if HAS_OPTUNA:
            return self.tune_with_optuna(n_trials)
        else:
            return self.tune_with_grid_search()


def load_training_data(model_dir: str = "models/v3_production"):
    """Load pre-computed training data or regenerate."""
    logger.info("Loading training data...")
    
    # For this demo, generate synthetic data
    # In production, load from saved training set
    np.random.seed(42)
    
    n_samples = 10000
    n_features = 18
    
    X = np.random.randn(n_samples, n_features)
    
    # Direction: based on first 5 features
    direction_signal = X[:, :5].sum(axis=1)
    y_dir = (direction_signal > 0).astype(int)
    
    # Magnitude
    y_mag = np.abs(direction_signal) * 0.01 + np.random.randn(n_samples) * 0.001
    
    # Confidence
    prob_success = 1 / (1 + np.exp(-np.abs(direction_signal)))
    y_conf = (np.random.random(n_samples) < prob_success).astype(int)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_dir_train, y_dir_val = y_dir[:split], y_dir[split:]
    y_mag_train, y_mag_val = y_mag[:split], y_mag[split:]
    y_conf_train, y_conf_val = y_conf[:split], y_conf[split:]
    
    logger.info(f"✓ Training data: {len(X_train)} samples")
    logger.info(f"✓ Validation data: {len(X_val)} samples")
    
    return (X_train, y_dir_train, y_mag_train, y_conf_train,
            X_val, y_dir_val, y_mag_val, y_conf_val)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune v3.0 hyperparameters")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--output-dir", default="models/v3_tuned", help="Output directory")
    
    args = parser.parse_args()
    
    # Load data
    (X_train, y_dir_train, y_mag_train, y_conf_train,
     X_val, y_dir_val, y_mag_val, y_conf_val) = load_training_data()
    
    # Run tuning
    tuner = HyperparameterTuner(
        X_train, y_dir_train, y_mag_train, y_conf_train,
        X_val, y_dir_val, y_mag_val, y_conf_val,
        output_dir=args.output_dir,
    )
    
    results = tuner.run_tuning(n_trials=args.n_trials)
    
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    
    # Print summary
    print("\nBest Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            if metric_name != 'best_params':
                print(f"  {metric_name}: {value}")


if __name__ == "__main__":
    main()
