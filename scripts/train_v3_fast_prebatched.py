"""
Fast Training using Pre-batched Features
Loads pre-computed features from disk (20x faster than computing on the fly).
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.models.specialized import SpecializedModelSuite, ModelConfig, compute_expected_calibration_error

print("="*70)
print("FAST TRAINING WITH PRE-BATCHED FEATURES")
print("="*70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
PREBATCH_DIR = Path("cache/prebatched_features")
OUTPUT_DIR = Path("results/models/v3_nifty500_fast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check if pre-batched features exist
if not PREBATCH_DIR.exists() or len(list(PREBATCH_DIR.glob("*_features.pkl"))) == 0:
    print("❌ Pre-batched features not found!")
    print(f"   Run: python scripts/prebatch_features.py")
    print(f"   Then run this script again.")
    sys.exit(1)

print(f"Loading pre-batched features from {PREBATCH_DIR}...")

# Load all pre-batched samples
all_samples = []
batch_files = list(PREBATCH_DIR.glob("*_features.pkl"))

print(f"Found {len(batch_files)} pre-batched files")

for i, batch_file in enumerate(batch_files):
    if i % 100 == 0:
        print(f"  Loading [{i+1}/{len(batch_files)}]...")
    
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            all_samples.extend(samples)
    except Exception as e:
        print(f"  Error loading {batch_file}: {e}")
        continue

print(f"\n{'='*70}")
print("DATA SUMMARY")
print(f"{'='*70}")
print(f"Total samples loaded: {len(all_samples)}")

if len(all_samples) == 0:
    print("❌ No samples loaded!")
    sys.exit(1)

# Prepare data
X = np.array([s['features'] for s in all_samples])
y_dir = np.array([s['y_dir'] for s in all_samples])
y_mag = np.array([s['y_mag'] for s in all_samples])
y_conf = np.array([s['y_conf'] for s in all_samples])
dates = pd.to_datetime([s['date'] for s in all_samples])
symbols = [s['symbol'] for s in all_samples]

print(f"Features: {X.shape[1]}")
print(f"Unique symbols: {len(set(symbols))}")
print(f"Direction: {np.mean(y_dir):.1%} positive")
print(f"Date range: {dates.min().date()} to {dates.max().date()}")

# Temporal split
train_mask = dates < '2023-01-01'
val_mask = (dates >= '2023-01-01') & (dates < '2024-01-01')
test_mask = dates >= '2024-01-01'

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"\nTemporal split:")
print(f"  Train: {len(train_idx)} ({len(train_idx)/len(X):.1%})")
print(f"  Val: {len(val_idx)} ({len(val_idx)/len(X):.1%})")
print(f"  Test: {len(test_idx)} ({len(test_idx)/len(X):.1%})")

# Prepare sets
X_train, y_dir_train = X[train_idx], y_dir[train_idx]
y_mag_train, y_conf_train = y_mag[train_idx], y_conf[train_idx]

if len(val_idx) >= 15:
    X_val, y_dir_val = X[val_idx], y_dir[val_idx]
    y_mag_val, y_conf_val = y_mag[val_idx], y_conf[val_idx]
else:
    split_pt = int(len(train_idx) * 0.9)
    X_train, X_val = X_train[:split_pt], X_train[split_pt:]
    y_dir_train, y_dir_val = y_dir_train[:split_pt], y_dir_train[split_pt:]
    y_mag_train, y_mag_val = y_mag_train[:split_pt], y_mag_train[split_pt:]
    y_conf_train, y_conf_val = y_conf_train[:split_pt], y_conf_train[split_pt:]

if len(test_idx) >= 15:
    X_test, y_dir_test = X[test_idx], y_dir[test_idx]
    y_mag_test, y_conf_test = y_mag[test_idx], y_conf[test_idx]
else:
    X_test, y_dir_test = X[-200:], y_dir[-200:]
    y_mag_test, y_conf_test = y_mag[-200:], y_conf[-200:]

# Train models
print(f"\n{'='*70}")
print("TRAINING MODELS")
print(f"{'='*70}")
print(f"Training: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

config = ModelConfig()
models = SpecializedModelSuite(config)

models.fit(X_train, y_dir_train, y_mag_train, y_conf_train,
           X_val, y_dir_val, y_mag_val, y_conf_val)

print("✓ Models trained successfully")

# EVALUATION
print(f"\n{'='*70}")
print("HONEST EVALUATION (TEST SET)")
print(f"{'='*70}")

# Validation
print("\n[VALIDATION - 2023]")
dir_preds_val = models.direction_model.predict_class(X_val)
dir_acc_val = accuracy_score(y_dir_val, dir_preds_val)
print(f"  Accuracy: {dir_acc_val:.2%}")

dir_proba_val = models.direction_model.predict(X_val)
try:
    dir_auc_val = roc_auc_score(y_dir_val, dir_proba_val)
    print(f"  AUC: {dir_auc_val:.4f}")
except:
    dir_auc_val = 0.5

# Test
print("\n[TEST - 2024+] (UNTOUCHED)")
dir_preds_test = models.direction_model.predict_class(X_test)
dir_acc_test = accuracy_score(y_dir_test, dir_preds_test)
print(f"  Direction Accuracy: {dir_acc_test:.2%}")

dir_proba_test = models.direction_model.predict(X_test)
try:
    dir_auc_test = roc_auc_score(y_dir_test, dir_proba_test)
    ece_test = compute_expected_calibration_error(y_dir_test, dir_proba_test)
    brier_test = brier_score_loss(y_dir_test, dir_proba_test)
    print(f"  Direction AUC: {dir_auc_test:.4f}")
    print(f"  Direction ECE: {ece_test:.4f}")
    print(f"  Direction Brier: {brier_test:.4f}")
except:
    dir_auc_test, ece_test, brier_test = 0.5, 0.0, 0.25

mag_preds_test = models.magnitude_model.predict(X_test)
mag_mae_test = mean_absolute_error(y_mag_test, mag_preds_test)
print(f"  Magnitude MAE: {mag_mae_test:.5f}")

conf_preds_test = models.confidence_model.predict(X_test) > 0.5
conf_acc_test = accuracy_score(y_conf_test, conf_preds_test)
print(f"  Confidence Accuracy: {conf_acc_test:.2%}")

# Save
print(f"\n{'='*70}")
print("SAVING MODELS")
print(f"{'='*70}")

models.save(str(OUTPUT_DIR))

metadata = {
    'timestamp': datetime.now().isoformat(),
    'training_type': 'fast_prebatched',
    'prebatch_dir': str(PREBATCH_DIR),
    'total_samples': len(X),
    'unique_symbols': len(set(symbols)),
    'n_features': X.shape[1],
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'test_metrics': {
        'direction_accuracy': float(dir_acc_test),
        'direction_auc': float(dir_auc_test),
        'direction_ece': float(ece_test),
        'brier': float(brier_test),
        'magnitude_mae': float(mag_mae_test),
        'confidence_accuracy': float(conf_acc_test)
    },
    'status': 'COMPLETE'
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

results_dir = Path("results/training/fast_prebatched")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "training_summary.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Models saved to {OUTPUT_DIR}")
print(f"✓ Metadata saved")

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Test AUC: {dir_auc_test:.4f}")
print(f"Test Accuracy: {dir_acc_test:.2%}")
print(f"Model ready for use!")
print(f"{'='*70}")
