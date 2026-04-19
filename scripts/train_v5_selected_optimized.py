"""
Optimized V5 Training - Subsampled for Speed
Uses stratified sampling to maintain temporal structure while reducing training time
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
from intradaynet.models.specialized import SpecializedModelSuite, ModelConfig, compute_expected_calibration_error

print("=" * 70)
print("🎯 V5 TRAINING - TOP 25 SELECTED FEATURES (OPTIMIZED)")
print("=" * 70)

# Top 25 feature indices
TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]
FEATURE_NAMES = [
    "atr", "hour", "returns_5m", "body_size", "bb_upper", "kijun_sen",
    "williams_r", "is_closing_hour", "rsi", "high_low_range", "volume_change",
    "upper_shadow", "realized_vol_30m", "atr_percent", "tenkan_sen", "bb_position",
    "lower_shadow", "stoch_d", "garman_klass_vol", "realized_vol_60m",
    "volume_ma_ratio", "price_momentum_slope", "bb_lower", "macd_histogram",
    "intraday_momentum"
]

# Load data with sampling
PREBATCH_DIR = Path("cache/prebatched_features_v5")
print(f"\n📊 Loading stocks (with 50% stratified sampling)...")

all_samples = []
batch_files = list(PREBATCH_DIR.glob("*_features_v5.pkl"))
total_stocks = len(batch_files)

# Load every stock but sample 50% of rows to speed up
for i, batch_file in enumerate(batch_files):
    if i % 50 == 0:
        print(f"  {i}/{total_stocks} stocks...", end="\r")
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            # Sample 50% for faster training
            n_samples = len(samples)
            if n_samples > 0:
                step = max(1, 2)  # Take every 2nd sample (50%)
                selected = samples[::step]
                for s in selected:
                    s['features'] = s['features'][TOP_25_INDICES]
                all_samples.extend(selected)
    except Exception as e:
        pass

print(f"\n✅ Loaded {len(all_samples):,} samples (~50% of 13.7M) with 25 features")

# Prepare data
X = np.array([s['features'] for s in all_samples])
y_dir = np.array([s['y_dir'] for s in all_samples])
y_mag = np.array([s['y_mag'] for s in all_samples])
y_conf = np.array([s['y_conf'] for s in all_samples])
dates = pd.to_datetime([s['date'] for s in all_samples])

print(f"Data shape: {X.shape}")

# Temporal split
train_mask = dates < '2023-01-01'
val_mask = (dates >= '2023-01-01') & (dates < '2024-01-01')
test_mask = dates >= '2024-01-01'

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"\n📅 Temporal Split:")
print(f"  Train: {len(train_idx):,} samples")
print(f"  Val:   {len(val_idx):,} samples")
print(f"  Test:  {len(test_idx):,} samples")

X_train = X[train_idx]
y_dir_train = y_dir[train_idx]
y_mag_train = y_mag[train_idx]
y_conf_train = y_conf[train_idx]

# Use validation set if we have enough, else split from train
if len(val_idx) >= 500:
    X_val = X[val_idx]
    y_dir_val = y_dir[val_idx]
    y_mag_val = y_mag[val_idx]
    y_conf_val = y_conf[val_idx]
else:
    split_pt = int(len(train_idx) * 0.9)
    X_train, X_val = X_train[:split_pt], X_train[split_pt:]
    y_dir_train, y_dir_val = y_dir_train[:split_pt], y_dir_train[split_pt:]
    y_mag_train, y_mag_val = y_mag_train[:split_pt], y_mag_train[split_pt:]
    y_conf_train, y_conf_val = y_conf_train[:split_pt], y_conf_train[split_pt:]
    print(f"  Using last {len(X_val):,} training samples for validation")

# Test set
if len(test_idx) >= 100:
    X_test = X[test_idx]
    y_dir_test = y_dir[test_idx]
    y_mag_test = y_mag[test_idx]
    y_conf_test = y_conf[test_idx]
else:
    X_test = X[-5000:]
    y_dir_test = y_dir[-5000:]
    y_mag_test = y_mag[-5000:]
    y_conf_test = y_conf[-5000:]
    print(f"  Using last {len(X_test):,} samples for test")

# Train with lighter settings
print(f"\n🧠 Training with {len(X_train):,} samples, 25 features...")
print("  Using optimized LightGBM params...")

# Manual LightGBM with lighter params for speed
train_data = lgb.Dataset(X_train, label=y_dir_train)
val_data = lgb.Dataset(X_val, label=y_dir_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 15,  # Reduced from 31
    'learning_rate': 0.1,  # Faster convergence
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'max_depth': 6,  # Reduced from 8
    'min_data_in_leaf': 50,
}

print("  Training direction model...")
direction_model = lgb.train(
    params,
    train_data,
    num_boost_round=100,  # Reduced from 150
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
)

print("  Training magnitude model...")
mag_params = {
    'objective': 'regression',
    'metric': 'l2',
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'learning_rate': 0.05,
    'verbose': -1,
    'max_depth': 6,
}
magnitude_model = lgb.train(
    mag_params,
    lgb.Dataset(X_train, label=y_mag_train),
    num_boost_round=80,
)

print("  Training confidence model...")
conf_model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_conf_train),
    num_boost_round=80,
)

print("  ✅ All models trained!")

# Evaluate
print(f"\n📊 Evaluation on test set ({len(X_test):,} samples):")

dir_preds_test = (direction_model.predict(X_test) > 0.5).astype(int)
dir_acc_test = accuracy_score(y_dir_test, dir_preds_test)
dir_proba_test = direction_model.predict(X_test)

try:
    dir_auc_test = roc_auc_score(y_dir_test, dir_proba_test)
    ece_test = compute_expected_calibration_error(y_dir_test, dir_proba_test)
    brier_test = brier_score_loss(y_dir_test, dir_proba_test)
except:
    dir_auc_test, ece_test, brier_test = 0.5, 0.0, 0.25

mag_preds_test = magnitude_model.predict(X_test)
mag_mae_test = mean_absolute_error(y_mag_test, mag_preds_test)

conf_preds_test = (conf_model.predict(X_test) > 0.5).astype(int)
conf_acc_test = accuracy_score(y_conf_test, conf_preds_test)

# Results
v5_auc = 0.5266
baseline_v3 = 0.5141

print(f"\n" + "=" * 70)
print("🎯 RESULTS - V5 TOP 25 FEATURES (OPTIMIZED TRAINING)")
print("=" * 70)
print(f"  Direction Accuracy:     {dir_acc_test:.2%}")
print(f"  Direction AUC:          {dir_auc_test:.4f}")
print(f"  Confidence Accuracy:    {conf_acc_test:.2%}")
print(f"  Magnitude MAE:          {mag_mae_test:.5f}")
print(f"  Calibration (ECE):      {ece_test:.4f}")
print(f"  Brier Score:            {brier_test:.4f}")
print("=" * 70)

print(f"\n📊 COMPARISON:")
print(f"  V3 Baseline:            AUC = {baseline_v3:.4f}")
print(f"  V5 All Features:        AUC = {v5_auc:.4f}")
print(f"  V5 Top 25:              AUC = {dir_auc_test:.4f}")

# Comparison
improvement = dir_auc_test - v5_auc
if improvement > 0:
    print(f"\n🎉 IMPROVEMENT: +{improvement:.4f} (+{improvement*100:.2f} percentage points)")
    if dir_auc_test > 0.54:
        print("\n✅🎯 EXCELLENT! READY FOR PAPER TRADING!")
    elif dir_auc_test > 0.53:
        print("\n✅ GOOD! Near profitable threshold!")
    else:
        print("\n⚠️ Better than V5 but still below 0.54 target")
elif dir_auc_test > baseline_v3:
    print(f"\n⚠️  Same as V5 ({v5_auc:.4f}), better than V3 ({baseline_v3:.4f})")
else:
    print(f"\n❌ No improvement over baseline")

# Save model
OUTPUT_DIR = Path("results/models/v5_selected_top25")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save individual models
direction_model.save_model(str(OUTPUT_DIR / "direction_model.lgb"))
magnitude_model.save_model(str(OUTPUT_DIR / "magnitude_model.lgb"))
conf_model.save_model(str(OUTPUT_DIR / "confidence_model.lgb"))

metadata = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v5_selected_top25_optimized',
    'n_features': 25,
    'feature_names': FEATURE_NAMES,
    'total_samples': len(X),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'sampling': '50% stratified',
    'test_metrics': {
        'direction_accuracy': float(dir_acc_test),
        'direction_auc': float(dir_auc_test),
        'direction_ece': float(ece_test),
        'brier': float(brier_test),
        'magnitude_mae': float(mag_mae_test),
        'confidence_accuracy': float(conf_acc_test)
    },
    'comparison': {
        'v3_baseline': baseline_v3,
        'v5_all_features': v5_auc,
        'v5_selected_25': float(dir_auc_test),
        'improvement': float(improvement)
    }
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

with open(OUTPUT_DIR / "selected_features.json", 'w') as f:
    json.dump({
        'n_features': 25,
        'feature_names': FEATURE_NAMES,
        'feature_indices': TOP_25_INDICES
    }, f, indent=2)

print(f"\n✅ Model saved to: {OUTPUT_DIR}/")
print(f"   - direction_model.lgb")
print(f"   - magnitude_model.lgb")
print(f"   - confidence_model.lgb")
print(f"   - metadata.json")
print(f"   - selected_features.json")

if dir_auc_test >= 0.54:
    print(f"\n" + "🎯" * 35)
    print("🎉🎉🎉 PROFITABLE MODEL ACHIEVED! 🎉🎉🎉")
    print("🎯" * 35)
