"""
Model Architecture for IntradayNet v3.0 - Phase 3

Implements 3 specialized LightGBM models:
1. Direction Model: Binary classifier (UP/DOWN), optimized for log-loss
2. Magnitude Model: Regressor predicting absolute return, optimized for Huber loss
3. Confidence Model: Predicts probability of hitting target before stop-loss

Ensemble:
- Level 0: LightGBM, TCN, ResNLS (drop Compact CNN and MLP-Mixer)
- Level 1: Logistic regression meta-learner on out-of-fold predictions
- Dynamic weighting based on rolling 20-day accuracy

Usage:
    from intradaynet.models.specialized import SpecializedModelSuite
    
    suite = SpecializedModelSuite()
    
    # Train all three models
    suite.fit(X_train, y_train_dir, y_train_mag, y_train_conf)
    
    # Predict with ensemble
    predictions = suite.predict(X_test)
    # Returns dict with direction_prob, magnitude_estimate, confidence_score
    
    # Final score = direction_prob × magnitude_estimate × confidence
    final_score = suite.compute_final_score(X_test)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("intradaynet.models.specialized")

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available, using sklearn fallback")


@dataclass
class ModelConfig:
    """Configuration for specialized models."""
    
    # Direction model (binary classifier)
    dir_params: Dict = None
    
    # Magnitude model (regressor)
    mag_params: Dict = None
    
    # Confidence model (binary classifier - hits target vs stop)
    conf_params: Dict = None
    
    # Ensemble
    use_stacking: bool = True
    meta_learner_type: str = "logistic"  # logistic, ridge, etc.
    n_folds: int = 5
    
    # Calibration
    calibration_method: str = "isotonic"  # platt, isotonic
    
    def __post_init__(self):
        if self.dir_params is None:
            self.dir_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "dart",
                "n_estimators": 500,
                "max_depth": 6,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction_bynode": 0.7,
                "min_data_in_leaf": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "n_jobs": -1,
                "verbosity": -1,
            }
        
        if self.mag_params is None:
            self.mag_params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "dart",
                "n_estimators": 500,
                "max_depth": 6,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction_bynode": 0.7,
                "min_data_in_leaf": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "n_jobs": -1,
                "verbosity": -1,
            }
        
        if self.conf_params is None:
            self.conf_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "dart",
                "n_estimators": 300,
                "max_depth": 5,
                "num_leaves": 20,
                "learning_rate": 0.05,
                "feature_fraction_bynode": 0.7,
                "min_data_in_leaf": 200,
                "subsample": 0.8,
                "n_jobs": -1,
                "verbosity": -1,
            }


class DirectionModel:
    """
    Direction classifier - predicts UP vs DOWN.
    
    Optimized for log-loss to produce well-calibrated probabilities.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.calibrator = None
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        """Train direction model."""
        logger.info("Training Direction Model...")
        
        if not HAS_LIGHTGBM:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6)
            self.model.fit(X, y)
        else:
            self.model = lgb.LGBMClassifier(**self.config.dir_params)
            
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            
            self.model.fit(
                X, y,
                eval_set=eval_set,
                callbacks=[lgb.log_evaluation(0)] if eval_set else None,
            )
            
            self.feature_importance = self.model.feature_importances_
        
        # Calibrate probabilities
        self._calibrate(X_val if X_val is not None else X, 
                       y_val if y_val is not None else y)
        
        logger.info("Direction model training complete")
        
    def _calibrate(self, X: np.ndarray, y: np.ndarray):
        """Calibrate predicted probabilities."""
        if self.config.calibration_method == "isotonic":
            # Get uncalibrated predictions
            if not HAS_LIGHTGBM:
                y_proba = self.model.predict_proba(X)[:, 1]
            else:
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(X)[:, 1]
                else:
                    raw_pred = self.model.predict(X)
                    y_proba = 1 / (1 + np.exp(-raw_pred))
            
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_proba, y)
            
        elif self.config.calibration_method == "platt":
            # Platt scaling via sigmoid
            from sklearn.linear_model import LogisticRegression
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)[:, 1].reshape(-1, 1)
            else:
                raw_pred = self.model.predict(X)
                y_proba = (1 / (1 + np.exp(-raw_pred))).reshape(-1, 1)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_proba, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict direction probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        if not HAS_LIGHTGBM:
            y_proba = self.model.predict_proba(X)[:, 1]
        else:
            # Check if it's a raw Booster or LGBMClassifier
            if hasattr(self.model, 'predict_proba'):
                # It's an LGBMClassifier
                y_proba = self.model.predict_proba(X)[:, 1]
            else:
                # It's a raw Booster - use predict and convert to probability
                raw_pred = self.model.predict(X)
                # Convert raw score to probability using sigmoid
                y_proba = 1 / (1 + np.exp(-raw_pred))
        
        # Apply calibration
        if self.calibrator is not None:
            if self.config.calibration_method == "isotonic":
                y_proba = self.calibrator.predict(y_proba)
            elif self.config.calibration_method == "platt":
                y_proba = self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        
        return y_proba
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict direction class."""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)


class MagnitudeModel:
    """
    Magnitude regressor - predicts absolute return magnitude.
    
    Optimized for Huber loss (robust to outliers).
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        """Train magnitude model."""
        logger.info("Training Magnitude Model...")
        
        # Huber loss is robust to outliers
        if not HAS_LIGHTGBM:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, max_depth=6)
            self.model.fit(X, y)
        else:
            # Use Huber objective for robust regression
            params = self.config.mag_params.copy()
            params['objective'] = 'huber'
            params['alpha'] = 0.9  # Quantile for Huber
            
            self.model = lgb.LGBMRegressor(**params)
            
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            
            self.model.fit(
                X, y,
                eval_set=eval_set,
                callbacks=[lgb.log_evaluation(0)] if eval_set else None,
            )
            
            self.feature_importance = self.model.feature_importances_
        
        logger.info("Magnitude model training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict magnitude."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)


class ConfidenceModel:
    """
    Confidence classifier - predicts if trade will hit target before stop.
    
    Binary: 1 = hits target first, 0 = hits stop first
    Trained on historical hit rates.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.calibrator = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        """Train confidence model."""
        logger.info("Training Confidence Model...")
        
        if not HAS_LIGHTGBM:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5)
            self.model.fit(X, y)
        else:
            self.model = lgb.LGBMClassifier(**self.config.conf_params)
            
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            
            self.model.fit(
                X, y,
                eval_set=eval_set,
                callbacks=[lgb.log_evaluation(0)] if eval_set else None,
            )
        
        # Calibrate
        self._calibrate(X_val if X_val is not None else X,
                       y_val if y_val is not None else y)
        
        logger.info("Confidence model training complete")
    
    def _calibrate(self, X: np.ndarray, y: np.ndarray):
        """Calibrate probabilities."""
        if not HAS_LIGHTGBM:
            y_proba = self.model.predict_proba(X)[:, 1]
        else:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)[:, 1]
            else:
                raw_pred = self.model.predict(X)
                y_proba = 1 / (1 + np.exp(-raw_pred))
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_proba, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict confidence score."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        if not HAS_LIGHTGBM:
            y_proba = self.model.predict_proba(X)[:, 1]
        else:
            # Check if it's a raw Booster or LGBMClassifier
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)[:, 1]
            else:
                # It's a raw Booster
                raw_pred = self.model.predict(X)
                y_proba = 1 / (1 + np.exp(-raw_pred))
        
        if self.calibrator is not None:
            y_proba = self.calibrator.predict(y_proba)
        
        return y_proba


class StackedEnsemble:
    """
    Stacked ensemble with meta-learner.
    
    Level 0: Multiple base models
    Level 1: Meta-learner on out-of-fold predictions
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_models = {}  # name -> model
        self.meta_learner = None
        self.oof_predictions = None
        
    def add_base_model(self, name: str, model):
        """Add a base model to the ensemble."""
        self.base_models[name] = model
        logger.info(f"Added base model: {name}")
    
    def fit_level1(self, X: np.ndarray, y: np.ndarray):
        """
        Train meta-learner on out-of-fold predictions.
        
        Key: Must use out-of-fold predictions, not in-sample!
        """
        logger.info("Training Level 1 meta-learner...")
        
        if not self.base_models:
            raise ValueError("No base models added")
        
        # Generate out-of-fold predictions
        oof_preds = self._generate_oof_predictions(X, y)
        
        # Train meta-learner on OOF predictions
        if self.config.meta_learner_type == "logistic":
            self.meta_learner = LogisticRegression(C=1.0, max_iter=1000)
            self.meta_learner.fit(oof_preds, y)
        elif self.config.meta_learner_type == "average":
            # Simple average - no meta-learner needed
            self.meta_learner = None
        
        logger.info("Meta-learner training complete")
    
    def _generate_oof_predictions(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> np.ndarray:
        """
        Generate out-of-fold predictions using cross-validation.
        
        Critical for preventing leakage - never train and predict on same data.
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        
        oof_preds = np.zeros((n_samples, n_models))
        
        # Stratified K-Fold for classification
        skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Train each base model on this fold
            for i, (name, model_template) in enumerate(self.base_models.items()):
                # Clone model for this fold
                if not HAS_LIGHTGBM:
                    from sklearn.base import clone
                    model = clone(model_template)
                else:
                    # For LightGBM, create fresh instance
                    params = model_template.get_params()
                    if hasattr(model_template, 'classes_'):  # Classifier
                        model = lgb.LGBMClassifier(**{k: v for k, v in params.items() 
                                                        if k not in ['classes_', 'n_features_in_']})
                    else:
                        model = lgb.LGBMRegressor(**{k: v for k, v in params.items()
                                                     if k not in ['n_features_in_']})
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict on validation fold
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X_val)[:, 1]
                else:
                    preds = model.predict(X_val)
                
                oof_preds[val_idx, i] = preds
        
        self.oof_predictions = oof_preds
        return oof_preds
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble prediction."""
        # Get base model predictions
        base_preds = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            base_preds.append(preds)
        
        X_meta = np.column_stack(base_preds)
        
        # Meta-learner prediction
        if self.meta_learner is not None:
            if hasattr(self.meta_learner, 'predict_proba'):
                final_pred = self.meta_learner.predict_proba(X_meta)[:, 1]
            else:
                final_pred = self.meta_learner.predict(X_meta)
        else:
            # Simple average
            final_pred = X_meta.mean(axis=1)
        
        return final_pred


class SpecializedModelSuite:
    """
    Complete suite of 3 specialized models + ensemble.
    
    Final score = direction_prob × magnitude_estimate × confidence
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        
        self.direction_model = DirectionModel(self.config)
        self.magnitude_model = MagnitudeModel(self.config)
        self.confidence_model = ConfidenceModel(self.config)
        
        self.ensemble = StackedEnsemble(self.config)
        self.use_ensemble = False
        
        # Dynamic weighting
        self.model_weights = {
            'direction': 1.0,
            'magnitude': 1.0,
            'confidence': 1.0,
        }
        self.rolling_accuracy = {}
        
    def fit(
        self,
        X_train: np.ndarray,
        y_dir: np.ndarray,
        y_mag: np.ndarray,
        y_conf: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val_dir: Optional[np.ndarray] = None,
        y_val_mag: Optional[np.ndarray] = None,
        y_val_conf: Optional[np.ndarray] = None,
    ):
        """
        Train all three specialized models.
        
        Args:
            X_train: Training features
            y_dir: Direction targets (0/1)
            y_mag: Magnitude targets (float)
            y_conf: Confidence targets (0/1)
        """
        logger.info("="*60)
        logger.info("TRAINING SPECIALIZED MODEL SUITE")
        logger.info("="*60)
        
        # Train direction model
        self.direction_model.fit(X_train, y_dir, X_val, y_val_dir)
        
        # Train magnitude model
        self.magnitude_model.fit(X_train, y_mag, X_val, y_val_mag)
        
        # Train confidence model
        self.confidence_model.fit(X_train, y_conf, X_val, y_val_conf)
        
        # If validation data provided, compute initial weights
        if X_val is not None:
            self._update_dynamic_weights(X_val, y_val_dir)
        
        logger.info("="*60)
        logger.info("ALL MODELS TRAINED")
        logger.info("="*60)
    
    def _update_dynamic_weights(
        self,
        X_val: np.ndarray,
        y_val_dir: np.ndarray,
    ):
        """
        Update dynamic weights based on recent accuracy.
        
        If model's accuracy drops below 55%, reduce its weight.
        """
        from sklearn.metrics import accuracy_score
        
        # Direction accuracy
        dir_preds = self.direction_model.predict_class(X_val)
        dir_acc = accuracy_score(y_val_dir, dir_preds)
        self.rolling_accuracy['direction'] = dir_acc
        
        # Adjust weight
        if dir_acc < 0.55:
            self.model_weights['direction'] = 0.5
            logger.warning(f"Direction accuracy {dir_acc:.2%} < 55%, reducing weight")
        elif dir_acc > 0.60:
            self.model_weights['direction'] = 1.2
            logger.info(f"Direction accuracy {dir_acc:.2%} > 60%, increasing weight")
        
        logger.info(f"Dynamic weights: {self.model_weights}")
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all models.
        
        Returns dict with keys:
        - direction_prob: Probability of UP move
        - magnitude_estimate: Predicted return magnitude
        - confidence_score: Probability of hitting target before stop
        """
        return {
            'direction_prob': self.direction_model.predict(X),
            'magnitude_estimate': self.magnitude_model.predict(X),
            'confidence_score': self.confidence_model.predict(X),
        }
    
    def compute_final_score(
        self,
        X: np.ndarray,
        use_dynamic_weights: bool = True,
    ) -> np.ndarray:
        """
        Compute final composite score.
        
        Formula: direction_prob × magnitude_estimate × confidence_score
        
        Can be adjusted with dynamic weights.
        """
        preds = self.predict(X)
        
        w = self.model_weights if use_dynamic_weights else {
            'direction': 1.0, 'magnitude': 1.0, 'confidence': 1.0
        }
        
        final_score = (
            preds['direction_prob'] ** w['direction'] *
            (1 + preds['magnitude_estimate']) ** w['magnitude'] *
            preds['confidence_score'] ** w['confidence']
        )
        
        return final_score
    
    def save(self, output_dir: str):
        """Save all models to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        if not HAS_LIGHTGBM:
            with open(output_path / "direction_model.pkl", 'wb') as f:
                pickle.dump(self.direction_model.model, f)
            with open(output_path / "magnitude_model.pkl", 'wb') as f:
                pickle.dump(self.magnitude_model.model, f)
            with open(output_path / "confidence_model.pkl", 'wb') as f:
                pickle.dump(self.confidence_model.model, f)
        else:
            self.direction_model.model.booster_.save_model(
                str(output_path / "direction_model.lgb")
            )
            self.magnitude_model.model.booster_.save_model(
                str(output_path / "magnitude_model.lgb")
            )
            self.confidence_model.model.booster_.save_model(
                str(output_path / "confidence_model.lgb")
            )
        
        # Save weights
        with open(output_path / "model_weights.json", 'w') as f:
            import json
            json.dump(self.model_weights, f)
        
        logger.info(f"Models saved to {output_dir}")
    
    def load(self, model_dir: str):
        """Load models from directory."""
        model_path = Path(model_dir)
        
        if not HAS_LIGHTGBM:
            with open(model_path / "direction_model.pkl", 'rb') as f:
                self.direction_model.model = pickle.load(f)
            with open(model_path / "magnitude_model.pkl", 'rb') as f:
                self.magnitude_model.model = pickle.load(f)
            with open(model_path / "confidence_model.pkl", 'rb') as f:
                self.confidence_model.model = pickle.load(f)
        else:
            self.direction_model.model = lgb.Booster(
                model_file=str(model_path / "direction_model.lgb")
            )
            self.magnitude_model.model = lgb.Booster(
                model_file=str(model_path / "magnitude_model.lgb")
            )
            self.confidence_model.model = lgb.Booster(
                model_file=str(model_path / "confidence_model.lgb")
            )
        
        # Load weights
        with open(model_path / "model_weights.json") as f:
            import json
            self.model_weights = json.load(f)
        
        logger.info(f"Models loaded from {model_dir}")


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = Σ (bin_size × |accuracy - confidence|)
    
    Lower is better. Target: ECE < 0.05
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Accuracy in bin
            bin_acc = np.mean(y_true[in_bin])
            # Confidence in bin (mean of predicted probabilities)
            bin_conf = np.mean(y_proba[in_bin])
            # Weighted absolute difference
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
    
    return ece


def main():
    """Demo the specialized model suite."""
    print("\n" + "="*70)
    print("Specialized Model Suite Demo")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    
    # Direction: based on first 5 features
    direction_signal = X[:, :5].sum(axis=1)
    y_dir = (direction_signal > 0).astype(int)
    
    # Magnitude: absolute value of signal with noise
    y_mag = np.abs(direction_signal) * 0.01 + np.random.randn(n_samples) * 0.001
    
    # Confidence: probability of success (higher for strong signals)
    prob_success = 1 / (1 + np.exp(-np.abs(direction_signal)))
    y_conf = (np.random.random(n_samples) < prob_success).astype(int)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_dir_train, y_dir_val = y_dir[:split], y_dir[split:]
    y_mag_train, y_mag_val = y_mag[:split], y_mag[split:]
    y_conf_train, y_conf_val = y_conf[:split], y_conf[split:]
    
    print(f"\nTraining data: {len(X_train)} samples")
    print(f"Validation data: {len(X_val)} samples")
    
    # Train
    suite = SpecializedModelSuite()
    suite.fit(
        X_train, y_dir_train, y_mag_train, y_conf_train,
        X_val, y_dir_val, y_mag_val, y_conf_val,
    )
    
    # Predict
    preds = suite.predict(X_val[:5])
    
    print("\nSample predictions:")
    for i in range(5):
        print(f"  Sample {i+1}:")
        print(f"    Direction prob: {preds['direction_prob'][i]:.3f}")
        print(f"    Magnitude est:  {preds['magnitude_estimate'][i]:.4f}")
        print(f"    Confidence:     {preds['confidence_score'][i]:.3f}")
        
        final = suite.compute_final_score(X_val[i:i+1])
        print(f"    Final score:    {final[0]:.4f}")
        print()
    
    # Check calibration
    from sklearn.metrics import brier_score_loss
    
    dir_proba = preds['direction_prob']
    ece = compute_expected_calibration_error(y_dir_val[:5], dir_proba)
    brier = brier_score_loss(y_dir_val[:5], dir_proba)
    
    print(f"Calibration metrics:")
    print(f"  ECE: {ece:.4f} (target < 0.05)")
    print(f"  Brier: {brier:.4f}")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
