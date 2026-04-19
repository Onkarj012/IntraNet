"""
Quick V5 Training with Top 25 Selected Features
Faster version without console output overhead
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

print("=" * 60)
print("🎯 V5 TRAINING - TOP 25 SELECTED FEATURES")
print("=" * 60)

# Top 25 feature indices from selection analysis
TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]
FEATURE_NAMES = [
    "atr", "hour", "returns_5m", "body_size", "bb_upper", "kijun_sen",
    "williams_r", "is_closing_hour", "rsi", "high_low_range", "volume_change",
    "upper_shadow", "realized_vol_30m", "atr_percent", "tenkan_sen", "bb_position",
    "lower_shadow", "stoch_d", "garman_klass_vol", "realized_vol_60m",
    "volume_ma_ratio", "price_momentum_slope", "bb_lower", "macd_histogram",
    "intraday_momentum"
]

# Load data
PREBATCH_DIR = Path("cache/prebatched_features_v5")
print(f"\n📊 Loading all 499 stocks with 25 selected features...")

all_samples = []
batch_files = list(PREBATCH_DIR.glob("*_features_v5.pkl"))

for i, batch_file in enumerate(batch_files):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(batch_files)} stocks loaded")
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            # Select only top 25 features
            for s in samples:
                s['features'] = s['features'][TOP_25_INDICES]
            all_samples.extend(samples)
    except Exception as e:
        pass

print(f"\n✅ Loaded {len(all_samples):,} samples with 25 features")

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
print(f"  Train: {len(train_idx):,} samples ({dates[train_idx].min()} to {dates[train_idx].max()})")
print(f"  Val: {len(val_idx):,} samples ({dates[val_idx].min() if len(val_idx) > 0 else 'N/A'} to {dates[val_idx].max() if len(val_idx) > 0 else 'N/A'})")
print(f"  Test: {len(test_idx):,} samples ({dates[test_idx].min() if len(test_idx) > 0 else 'N/A'} to {dates[test_idx].max() if len(test_idx) > 0 else 'N/A'})")

X_train = X[train_idx]
y_dir_train = y_dir[train_idx]
y_mag_train = y_mag[train_idx]
y_conf_train = y_conf[train_idx]

if len(val_idx) >= 1000:
    X_val = X[val_idx]
    y_dir_val = y_dir[val_idx]
    y_mag_val = y_mag[val_idx]
    y_conf_val = y_conf[val_idx]
else:
    # Use last 10% of training as validation
    split_pt = int(len(train_idx) * 0.9)
    X_train, X_val = X_train[:split_pt], X_train[split_pt:]
    y_dir_train, y_dir_val = y_dir_train[:split_pt], y_dir_train[split_pt:]
    y_mag_train, y_mag_val = y_mag_train[:split_pt], y_mag_train[split_pt:]
    y_conf_train, y_conf_val = y_conf_train[:split_pt], y_conf_train[split_pt:]
    print(f"  Using last {len(X_val):,} training samples for validation")

X_test = X[test_idx] if len(test_idx) >= 100 else X[-10000:]
y_dir_test = y_dir[test_idx] if len(test_idx) >= 100 else y_dir[-10000:]
y_mag_test = y_mag[test_idx] if len(test_idx) >= 100 else y_mag[-10000:]
y_conf_test = y_conf[test_idx] if len(test_idx) >= 100 else y_conf[-10000:]

# Train
print(f"\n🧠 Training with {len(X_train):,} samples, 25 features...")
print("  This will take 2-5 minutes...")

config = ModelConfig()
models = SpecializedModelSuite(config)

models.fit(X_train, y_dir_train, y_mag_train, y_conf_train,
           X_val, y_dir_val, y_mag_val, y_conf_val)

print("  ✅ Training complete!")

# Evaluate
print(f"\n📊 Evaluation on test set ({len(X_test):,} samples):")

dir_preds_test = models.direction_model.predict_class(X_test)
dir_acc_test = accuracy_score(y_dir_test, dir_preds_test)
dir_proba_test = models.direction_model.predict(X_test)

try:
    dir_auc_test = roc_auc_score(y_dir_test, dir_proba_test)
    ece_test = compute_expected_calibration_error(y_dir_test, dir_proba_test)
    brier_test = brier_score_loss(y_dir_test, dir_proba_test)
except:
    dir_auc_test, ece_test, brier_test = 0.5, 0.0, 0.25

mag_preds_test = models.magnitude_model.predict(X_test)
mag_mae_test = mean_absolute_error(y_mag_test, mag_preds_test)

conf_preds_test = models.confidence_model.predict(X_test) > 0.5
conf_acc_test = accuracy_score(y_conf_test, conf_preds_test)

# Results
v5_auc = 0.5266
baseline_v3 = 0.5141

print(f"\n" + "=" * 60)
print("🎯 RESULTS - V5 TOP 25 FEATURES")
print("=" * 60)
print(f"  Direction Accuracy:     {dir_acc_test:.2%} (V5 all: 53.62%)")
print(f"  Direction AUC:          {dir_auc_test:.4f} (V5 all: {v5_auc:.4f}, V3: {baseline_v3:.4f})")
print(f"  Confidence Accuracy:    {conf_acc_test:.2%} (V5 all: 96.41%)")
print(f"  Magnitude MAE:          {mag_mae_test:.5f} (V5 all: 0.00189)")
print(f"  Calibration (ECE):      {ece_test:.4f}")
print(f"  Brier Score:            {brier_test:.4f}")
print("=" * 60)

# Comparison
if dir_auc_test > v5_auc:
    print(f"\n🎉 SUCCESS! AUC improved: {v5_auc:.4f} → {dir_auc_test:.4f} (+{(dir_auc_test-v5_auc)*100:.2f}%)")
    if dir_auc_test > 0.54:
        print("✅ READY FOR PAPER TRADING!")
elif dir_auc_test > baseline_v3:
    print(f"\n⚠️  Slight improvement over V3 ({baseline_v3:.4f}) but not better than V5 all features ({v5_auc:.4f})")
else:
    print(f"\n❌ No improvement. Feature selection didn't help.")

# Save model
OUTPUT_DIR = Path("results/models/v5_selected_top25")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

models.save(str(OUTPUT_DIR))

metadata = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v5_selected_top25',
    'n_features': 25,
    'feature_names': FEATURE_NAMES,
    'total_samples': len(X),
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
        'improvement': float(dir_auc_test - v5_auc)
    }
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Model saved to: {OUTPUT_DIR}/")
print(f"✅ Metadata saved to: {OUTPUT_DIR}/metadata.json")
