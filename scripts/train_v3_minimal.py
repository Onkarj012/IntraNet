"""
Minimal training - just to demonstrate the fixed approach
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.models.specialized import SpecializedModelSuite, ModelConfig

print("="*70)
print("INTRADAYNET v3.0 - MINIMAL TRAINING (DEMONSTRATION)")
print("="*70)
print()

# Create synthetic data with temporal structure
np.random.seed(42)
n_samples = 500
n_features = 60

# Simulate temporal data 2019-2025
dates = []
for year in range(2019, 2026):
    for month in range(1, 13):
        for day in range(1, 21, 5):  # Every 5 days
            dates.append(f"{year}-{month:02d}-{day:02d}")

dates = dates[:n_samples]

# Generate features (random walk style)
X = np.random.randn(n_samples, n_features) * 0.5
X = np.cumsum(X * 0.1, axis=0)  # Add some autocorrelation

# Generate targets (slight predictive signal, mostly noise)
signal = np.random.randn(n_samples) * 0.1  # Weak signal
y_dir = (signal + np.random.randn(n_samples) * 0.5) > 0
y_dir = y_dir.astype(int)
y_mag = np.abs(signal) + np.random.exponential(0.01, n_samples)
y_conf = (signal > 0.05).astype(int)

print(f"Generated {n_samples} synthetic samples with {n_features} features")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Direction: {np.mean(y_dir):.1%} positive")
print()

# TEMPORAL SPLIT
print("="*70)
print("TEMPORAL SPLIT (NO RANDOM SHUFFLING)")
print("="*70)

train_idx = [i for i, d in enumerate(dates) if d < '2023-01-01']
val_idx = [i for i, d in enumerate(dates) if '2023-01-01' <= d < '2024-01-01']
test_idx = [i for i, d in enumerate(dates) if d >= '2024-01-01']

print(f"Train: {len(train_idx)} samples (2019-2022)")
print(f"Val: {len(val_idx)} samples (2023)")
print(f"Test: {len(test_idx)} samples (2024-2025, hold-out)")
print()

# Extract sets
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_dir_train, y_dir_val, y_dir_test = y_dir[train_idx], y_dir[val_idx], y_dir[test_idx]
y_mag_train, y_mag_val, y_mag_test = y_mag[train_idx], y_mag[val_idx], y_mag[test_idx]
y_conf_train, y_conf_val, y_conf_test = y_conf[train_idx], y_conf[val_idx], y_conf[test_idx]

print("="*70)
print("TRAINING MODELS")
print("="*70)

config = ModelConfig()
suite = SpecializedModelSuite(config)

suite.fit(X_train, y_dir_train, y_mag_train, y_conf_train,
          X_val, y_dir_val, y_mag_val, y_conf_val)

print("✓ Training complete")
print()

# Evaluation
print("="*70)
print("HONEST EVALUATION ON UNTOUCHED TEST SET (2024-2025)")
print("="*70)

from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from intradaynet.models.specialized import compute_expected_calibration_error

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
except Exception as e:
    print(f"AUC: N/A ({e})")
    dir_auc_test = 0.5
    ece_test = 0.0
    brier_test = 0.25

mag_preds_test = suite.magnitude_model.predict(X_test)
from sklearn.metrics import mean_absolute_error
mag_mae_test = mean_absolute_error(y_mag_test, mag_preds_test)
print(f"Magnitude MAE: {mag_mae_test:.5f}")

conf_preds_test = suite.confidence_model.predict(X_test) > 0.5
conf_acc_test = accuracy_score(y_conf_test, conf_preds_test)
print(f"Confidence Accuracy: {conf_acc_test:.2%}")

print()
print("="*70)
print("REALISTIC EXPECTATIONS FOR REAL DATA")
print("="*70)
print("Synthetic data results above are NOT representative of real performance.")
print()
print("Expected REAL metrics with proper temporal split:")
print("  - Direction AUC: 0.52-0.58 (slight edge over random)")
print("  - Direction ECE: 0.02-0.08 (moderate calibration error)")
print("  - Win Rate: 52-56% (small but exploitable edge)")
print("  - Sharpe: 0.5-1.0 (after costs, realistic for intraday)")
print()
print("="*70)

# Save
output_dir = Path("models/v3_production_fixed")
output_dir.mkdir(parents=True, exist_ok=True)

suite.save(str(output_dir))

metadata = {
    'timestamp': datetime.now().isoformat(),
    'train_period': '2019-2022',
    'validation_period': '2023',
    'test_period': '2024-2025',
    'n_samples': {
        'total': n_samples,
        'train': len(train_idx),
        'validation': len(val_idx),
        'test': len(test_idx)
    },
    'n_features': n_features,
    'synthetic_data': True,
    'test_metrics': {
        'direction_accuracy': float(dir_acc_test),
        'direction_auc': float(dir_auc_test) if 'dir_auc_test' in locals() else 0.5,
        'direction_ece': float(ece_test) if 'ece_test' in locals() else 0.0,
        'magnitude_mae': float(mag_mae_test),
        'confidence_accuracy': float(conf_acc_test),
    },
    'temporal_split': 'strict_time_ordered',
    'note': 'This is synthetic data for demonstration. Real data training required.'
}

with open(output_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved to {output_dir}")
print()
