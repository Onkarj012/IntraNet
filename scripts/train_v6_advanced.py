"""
Train V6 Model with 30 Advanced Features (25 base + 5 advanced)
Experiment C: Add advanced features to improve AUC above 0.54
"""
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, brier_score_loss

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.models.specialized import compute_expected_calibration_error

print("=" * 70)
print("🚀 V6 TRAINING - 30 Advanced Features (Experiment C)")
print("=" * 70)

PREBATCH_DIR = Path("cache/prebatched_features_v6")

if not PREBATCH_DIR.exists():
    print(f"❌ V6 prebatch directory not found: {PREBATCH_DIR}")
    print("   Run: python scripts/create_v6_advanced_features.py")
    sys.exit(1)

print(f"\n📊 Loading V6 enhanced data (30 features)...")
all_samples = []
batch_files = list(PREBATCH_DIR.glob("*_features_v6.pkl"))

for i, batch_file in enumerate(batch_files):
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            # 50% sampling for speed
            all_samples.extend(samples[::2])
    except Exception as e:
        pass
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(batch_files)} stocks loaded...")

if len(all_samples) == 0:
    print("❌ No samples loaded!")
    sys.exit(1)

print(f"\n✅ Loaded {len(all_samples):,} samples with 30 features")

# Prepare data
X = np.array([s['features'] for s in all_samples])
y_dir = np.array([s['y_dir'] for s in all_samples])
y_mag = np.array([s['y_mag'] for s in all_samples])
y_conf = np.array([s['y_conf'] for s in all_samples])
dates = pd.to_datetime([s['date'] for s in all_samples])

print(f"Data shape: {X.shape}")

# Temporal split (strict - no data leakage)
train_mask = dates < '2023-01-01'
val_mask = (dates >= '2023-01-01') & (dates < '2024-01-01')
test_mask = dates >= '2024-01-01'

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"\n📅 Temporal Split:")
print(f"  Train: {len(train_idx):,} samples ({dates[train_idx].min()} to {dates[train_idx].max()})")
print(f"  Val:   {len(val_idx):,} samples ({dates[val_idx].min() if len(val_idx) > 0 else 'N/A'} to {dates[val_idx].max() if len(val_idx) > 0 else 'N/A'})")
print(f"  Test:  {len(test_idx):,} samples ({dates[test_idx].min() if len(test_idx) > 0 else 'N/A'} to {dates[test_idx].max() if len(test_idx) > 0 else 'N/A'})")

X_train, y_dir_train = X[train_idx], y_dir[train_idx]
y_mag_train, y_conf_train = y_mag[train_idx], y_conf[train_idx]

# Handle validation split
if len(val_idx) >= 500:
    X_val, y_dir_val = X[val_idx], y_dir[val_idx]
    y_mag_val, y_conf_val = y_mag[val_idx], y_conf[val_idx]
else:
    # Use last 10% of training
    split_pt = int(len(train_idx) * 0.9)
    X_train, X_val = X_train[:split_pt], X_train[split_pt:]
    y_dir_train, y_dir_val = y_dir_train[:split_pt], y_dir_train[split_pt:]
    y_mag_train, y_mag_val = y_mag_train[:split_pt], y_mag_train[split_pt:]
    y_conf_train, y_conf_val = y_conf_train[:split_pt], y_conf_train[split_pt:]
    print(f"  Using last {len(X_val):,} training samples for validation")

# Test set
if len(test_idx) >= 100:
    X_test, y_dir_test = X[test_idx], y_dir[test_idx]
    y_mag_test, y_conf_test = y_mag[test_idx], y_conf[test_idx]
else:
    X_test = X[-10000:]
    y_dir_test = y_dir[-10000:]
    y_mag_test = y_mag[-10000:]
    y_conf_test = y_conf[-10000:]
    print(f"  Using last {len(X_test):,} samples for test")

# Train models
print(f"\n🧠 Training V6 models with {len(X_train):,} samples...")

# Direction model
print("  Training direction model...")
train_data = lgb.Dataset(X_train, label=y_dir_train)
val_data = lgb.Dataset(X_val, label=y_dir_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'max_depth': 8,
}

direction_model = lgb.train(params, train_data, num_boost_round=150, valid_sets=[val_data])

# Magnitude model
print("  Training magnitude model...")
mag_params = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1,
}
magnitude_model = lgb.train(mag_params, lgb.Dataset(X_train, label=y_mag_train), num_boost_round=100)

# Confidence model
print("  Training confidence model...")
conf_model = lgb.train(params, lgb.Dataset(X_train, label=y_conf_train), num_boost_round=100)

# Evaluate
print(f"\n📊 Evaluating on test set ({len(X_test):,} samples)...")

dir_proba = direction_model.predict(X_test)
dir_preds = (dir_proba > 0.5).astype(int)
dir_acc = accuracy_score(y_dir_test, dir_preds)
dir_auc = roc_auc_score(y_dir_test, dir_proba)
ece = compute_expected_calibration_error(y_dir_test, dir_proba)
brier = brier_score_loss(y_dir_test, dir_proba)

mag_preds = magnitude_model.predict(X_test)
mag_mae = mean_absolute_error(y_mag_test, mag_preds)

conf_preds = conf_model.predict(X_test) > 0.5
conf_acc = accuracy_score(y_conf_test, conf_preds)

# Results
print(f"\n" + "=" * 70)
print("🎯 V6 RESULTS (30 Features - 25 base + 5 advanced)")
print("=" * 70)
print(f"  Direction AUC:         {dir_auc:.4f}")
print(f"  Direction Accuracy:    {dir_acc:.2%}")
print(f"  Confidence Accuracy:   {conf_acc:.2%}")
print(f"  Magnitude MAE:         {mag_mae:.5f}")
print(f"  Calibration (ECE):     {ece:.4f}")
print(f"  Brier Score:           {brier:.4f}")
print("=" * 70)

# Comparison
v3_auc = 0.5141
v5_auc = 0.5266

print(f"\n📊 COMPARISON:")
print(f"  V3 (18 features):      AUC = {v3_auc:.4f}")
print(f"  V5 (25 features):      AUC = {v5_auc:.4f}")
print(f"  V6 (30 features):      AUC = {dir_auc:.4f}")

improvement_v5 = dir_auc - v5_auc
improvement_v3 = dir_auc - v3_auc

print(f"\n  vs V5: {'+' if improvement_v5 > 0 else ''}{improvement_v5:.4f}")
print(f"  vs V3: {'+' if improvement_v3 > 0 else ''}{improvement_v3:.4f}")

if dir_auc > 0.54:
    print(f"\n🎉🎉🎉 EXCELLENT! AUC > 0.54 threshold achieved! 🎉🎉🎉")
    print("   This model is ready for live paper trading!")
elif dir_auc > v5_auc:
    print(f"\n✅ Improvement over V5, but still below 0.54 threshold")
    print("   Consider adding real sentiment/options data")
else:
    print(f"\n⚠️ No improvement with synthetic advanced features")
    print("   Need real external data (sentiment, options, order book)")

# Save models
OUTPUT_DIR = Path("results/models/v6_advanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

direction_model.save_model(str(OUTPUT_DIR / "direction_model.lgb"))
magnitude_model.save_model(str(OUTPUT_DIR / "magnitude_model.lgb"))
conf_model.save_model(str(OUTPUT_DIR / "confidence_model.lgb"))

metadata = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v6_advanced',
    'n_features': 30,
    'feature_breakdown': {'base': 25, 'advanced': 5},
    'total_samples': len(X),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'test_metrics': {
        'direction_auc': float(dir_auc),
        'direction_accuracy': float(dir_acc),
        'direction_ece': float(ece),
        'brier': float(brier),
        'magnitude_mae': float(mag_mae),
        'confidence_accuracy': float(conf_acc)
    },
    'comparison': {
        'v3_baseline': v3_auc,
        'v5_selected': v5_auc,
        'v6_advanced': float(dir_auc),
        'improvement_v5': float(improvement_v5),
        'improvement_v3': float(improvement_v3)
    }
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ V6 model saved to: {OUTPUT_DIR}/")
print(f"   - direction_model.lgb")
print(f"   - magnitude_model.lgb")
print(f"   - confidence_model.lgb")
print(f"   - metadata.json")
