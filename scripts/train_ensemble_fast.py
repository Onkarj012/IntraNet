"""
Ensemble Fast - V5 + Regime (50 stocks only)
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("🎯 ENSEMBLE FAST - V5 + Regime (50 stocks)")
print("=" * 70)

V5_MODEL_DIR = Path("results/models/v5_selected_top25")
PREBATCH_DIR = Path("cache/prebatched_features_v5")
OUTPUT_DIR = Path("results/v6_deep_analysis")

TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]

print("\n📦 Loading V5 Model...")
direction_model = lgb.Booster(model_file=str(V5_MODEL_DIR / "direction_model.lgb"))
confidence_model = lgb.Booster(model_file=str(V5_MODEL_DIR / "confidence_model.lgb"))

print("📊 Loading 50 stocks...")
all_samples = []
for batch_file in list(PREBATCH_DIR.glob("*_features_v5.pkl"))[:50]:
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            for s in samples:
                if pd.to_datetime(s['date']).year == 2024:
                    s['features'] = s['features'][TOP_25_INDICES]
                    all_samples.append(s)
    except:
        pass

print(f"✅ Loaded {len(all_samples):,} samples")

X = np.array([s['features'] for s in all_samples])
y_dir = np.array([s['y_dir'] for s in all_samples])
dates = pd.to_datetime([s['date'] for s in all_samples])

# Find regime features
atr_pct_idx = None
hour_idx = None
for i, idx in enumerate(TOP_25_INDICES):
    if idx == 13: atr_pct_idx = i
    if idx == 26: hour_idx = i

print(f"   atr_pct_idx={atr_pct_idx}, hour_idx={hour_idx}")

# Get predictions
base_proba = direction_model.predict(X)
conf_proba = confidence_model.predict(X)

base_auc = roc_auc_score(y_dir, base_proba)
print(f"\n🎯 V5 Base: AUC={base_auc:.4f}")

# Ensemble tests
print("\n" + "=" * 70)
print("🔍 ENSEMBLE TESTS")
print("=" * 70)

# Strategy 1: High confidence only
high_conf = conf_proba > 0.6
if high_conf.sum() > 10:
    auc_high = roc_auc_score(y_dir[high_conf], base_proba[high_conf])
    print(f"High Conf (>0.6):  AUC={auc_high:.4f} on {high_conf.sum()} samples")

# Strategy 2: Closing hour only (if hour feature exists)
if hour_idx is not None:
    hours = X[:, hour_idx]
    closing = hours >= 14
    if closing.sum() > 10:
        auc_close = roc_auc_score(y_dir[closing], base_proba[closing])
        print(f"Closing Hour:      AUC={auc_close:.4f} on {closing.sum()} samples")
    
    # Combined: Closing + High Conf
    combined = closing & (conf_proba > 0.55)
    if combined.sum() > 5:
        auc_combo = roc_auc_score(y_dir[combined], base_proba[combined])
        acc_combo = accuracy_score(y_dir[combined], (base_proba[combined] > 0.55).astype(int))
        print(f"Closing + Conf:    AUC={auc_combo:.4f}, Acc={acc_combo:.2%} on {combined.sum()} samples")

# Strategy 3: Low volatility (if atr feature exists)
if atr_pct_idx is not None:
    atr = X[:, atr_pct_idx]
    low_vol = atr < 0.015
    if low_vol.sum() > 10:
        auc_lowvol = roc_auc_score(y_dir[low_vol], base_proba[low_vol])
        print(f"Low Vol (<1.5%):   AUC={auc_lowvol:.4f} on {low_vol.sum()} samples")

# Stacked (simplified)
print("\n🧠 Training meta-classifier...")
# Use first 80% for train, last 20% for test
split = int(len(X) * 0.8)
X_meta = np.column_stack([base_proba, conf_proba])
X_train, X_test = X_meta[:split], X_meta[split:]
y_train, y_test = y_dir[:split], y_dir[split:]

meta = LGBMClassifier(n_estimators=50, learning_rate=0.05, verbose=-1)
meta.fit(X_train, y_train)
meta_proba = meta.predict_proba(X_test)[:, 1]
meta_auc = roc_auc_score(y_test, meta_proba)

print(f"\n🎯 Stacked Meta:    AUC={meta_auc:.4f}")

print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"V5 Base:           {base_auc:.4f}")
if 'auc_combo' in locals():
    print(f"Best Ensemble:     {auc_combo:.4f} (Closing + Conf)")
    print(f"vs Base:           {auc_combo - base_auc:+.4f}")
print(f"Stacked:           {meta_auc:.4f}")
