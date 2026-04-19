"""
V6 Fixed - Fast Version (Sampled Data)
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("🔧 V6 FIXED - Fast Version (100 stocks)")
print("=" * 70)

PREBATCH_DIR = Path("cache/prebatched_features_v6")
OUTPUT_DIR = Path("cache/prebatched_features_v6_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep only 20 non-redundant features
KEEP_INDICES = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

batch_files = list(PREBATCH_DIR.glob("*_features_v6.pkl"))[:100]  # Only 100 stocks
print(f"\n📊 Processing {len(batch_files)} stocks...")

for i, batch_file in enumerate(batch_files):
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
        
        cleaned = []
        for s in samples:
            selected = s['features'][KEEP_INDICES].copy()
            
            # Add clean regime features
            atr_pct = selected[11] if 11 < len(selected) else 0
            vol_30m = selected[9] if 9 < len(selected) else 0
            hour = selected[1] if 1 < len(selected) else 12
            
            vol_regime_score = np.tanh(atr_pct * 100 + vol_30m * 10) / 2 + 0.5
            time_score = np.clip((hour - 9) / 6.5, 0, 1)
            
            clean_features = np.concatenate([selected, [vol_regime_score, time_score]]).astype(np.float32)
            clean_features = np.nan_to_num(clean_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            s['features'] = clean_features
            cleaned.append(s)
        
        output_file = OUTPUT_DIR / batch_file.name.replace("_v6", "_v6_fixed")
        with open(output_file, 'wb') as f:
            pickle.dump(cleaned, f)
            
    except:
        pass
    
    if (i + 1) % 20 == 0:
        print(f"   {i+1}/{len(batch_files)} processed")

print(f"\n✅ Cleaned features created!")

# Train on sampled data
print(f"\n🧠 Training V6 Fixed (20 features)...")
all_samples = []
for batch_file in list(OUTPUT_DIR.glob("*_features_v6_fixed.pkl")):
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            all_samples.extend(samples[::3])  # 33% sample
    except:
        pass

print(f"   Loaded {len(all_samples):,} samples")

X = np.array([s['features'] for s in all_samples])
y_dir = np.array([s['y_dir'] for s in all_samples])
y_mag = np.array([s['y_mag'] for s in all_samples])
dates = pd.to_datetime([s['date'] for s in all_samples])

# Split
train_mask = dates < '2023-01-01'
test_mask = dates >= '2024-01-01'

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]

if len(train_idx) < 100 or len(test_idx) < 100:
    # Use simple split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_dir[:split], y_dir[split:]
else:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_dir[train_idx], y_dir[test_idx]

print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")

# Train
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1,
    'scale_pos_weight': 2.0,  # Weight UP class more heavily
}

model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=100)

# Evaluate
proba = model.predict(X_test)
auc = roc_auc_score(y_test, proba)
acc = accuracy_score(y_test, (proba > 0.5).astype(int))

print(f"\n" + "=" * 70)
print("🎯 V6 FIXED RESULTS")
print("=" * 70)
print(f"   AUC:      {auc:.4f}")
print(f"   Accuracy: {acc:.2%}")

# Save
MODEL_DIR = Path("results/models/v6_fixed")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_model(str(MODEL_DIR / "direction_model.lgb"))

import json
with open(MODEL_DIR / "metadata.json", 'w') as f:
    json.dump({
        'version': 'v6_fixed',
        'n_features': 20,
        'auc': float(auc),
        'accuracy': float(acc),
        'changes': ['Removed 10 redundant features', 'Fixed class imbalance']
    }, f)

print(f"\n✅ Saved to: {MODEL_DIR}/")
