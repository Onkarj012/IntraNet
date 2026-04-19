"""
V6 Prediction Error Analysis
Analyze when and why the model fails
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("❌ V6 PREDICTION ERROR ANALYSIS")
print("=" * 80)

# Load data
PREBATCH_DIR = Path("cache/prebatched_features_v6")
MODEL_DIR = Path("results/models/v6_advanced")
OUTPUT_DIR = Path("results/v6_deep_analysis")

import lightgbm as lgb
direction_model = lgb.Booster(model_file=str(MODEL_DIR / "direction_model.lgb"))

print("\n📊 Loading 2024 data...")
all_data = []
for batch_file in list(PREBATCH_DIR.glob("*_features_v6.pkl"))[:75]:
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            for s in samples:
                if pd.to_datetime(s['date']).year == 2024:
                    all_data.append(s)
    except:
        pass

X = np.array([s['features'] for s in all_data])
y_dir = np.array([s['y_dir'] for s in all_data])
y_mag = np.array([s['y_mag'] for s in all_data])
dates = pd.to_datetime([s['date'] for s in all_data])

proba = direction_model.predict(X)
preds = (proba > 0.5).astype(int)

# Identify errors
errors = preds != y_dir
error_indices = np.where(errors)[0]

print(f"\n📊 Error Statistics:")
print(f"   Total samples:     {len(y_dir):,}")
print(f"   Total errors:        {len(error_indices):,}")
print(f"   Error rate:          {len(error_indices)/len(y_dir):.2%}")

# Analyze errors
error_analysis = {
    'false_positives': [],  # Predicted UP, actual DOWN
    'false_negatives': []   # Predicted DOWN, actual UP
}

for idx in error_indices:
    error_type = 'false_positives' if preds[idx] == 1 else 'false_negatives'
    error_analysis[error_type].append({
        'index': idx,
        'predicted_proba': proba[idx],
        'actual': y_dir[idx],
        'magnitude': y_mag[idx],
        'date': dates[idx],
        'features': X[idx]
    })

fp_count = len(error_analysis['false_positives'])
fn_count = len(error_analysis['false_negatives'])

print(f"\n📊 Error Breakdown:")
print(f"   False Positives:   {fp_count:,} ({fp_count/len(error_indices):.1%})")
print(f"   False Negatives:   {fn_count:,} ({fn_count/len(error_indices):.1%})")

# Analyze confidence of errors
fp_probas = [e['predicted_proba'] for e in error_analysis['false_positives']] if fp_count > 0 else []
fn_probas = [e['predicted_proba'] for e in error_analysis['false_negatives']] if fn_count > 0 else []

if fp_probas:
    print(f"\n📊 False Positive Confidence:")
    print(f"   Mean:    {np.mean(fp_probas):.4f}")
    print(f"   Median:  {np.median(fp_probas):.4f}")
    print(f"   Std:     {np.std(fp_probas):.4f}")
    print(f"   High conf (>0.7): {sum(1 for p in fp_probas if p > 0.7)}/{len(fp_probas)}")

if fn_probas:
    print(f"\n📊 False Negative Confidence:")
    print(f"   Mean:    {np.mean(fn_probas):.4f}")
    print(f"   Median:  {np.median(fn_probas):.4f}")
    print(f"   Std:     {np.std(fn_probas):.4f}")

# Magnitude of errors
fp_mags = [e['magnitude'] for e in error_analysis['false_positives']] if fp_count > 0 else []
fn_mags = [e['magnitude'] for e in error_analysis['false_negatives']] if fn_count > 0 else []

if fp_mags:
    print(f"\n📊 Magnitude of Errors:")
    print(f"   False Positives avg mag: {np.mean(fp_mags):.4f}")
    print(f"   False Negatives avg mag: {np.mean(fn_mags):.4f}")

# Time-based error analysis
fp_hours = [e['date'].hour for e in error_analysis['false_positives']] if fp_count > 0 else []
fn_hours = [e['date'].hour for e in error_analysis['false_negatives']] if fn_count > 0 else []

if fp_hours:
    print(f"\n📊 Errors by Hour of Day:")
    for hour in range(9, 16):
        fp_at_hour = sum(1 for h in fp_hours if h == hour)
        fn_at_hour = sum(1 for h in fn_hours if h == hour)
        total_at_hour = fp_at_hour + fn_at_hour
        if total_at_hour > 0:
            print(f"   {hour:2d}:00  FP={fp_at_hour:3d}  FN={fn_at_hour:3d}  Total={total_at_hour:3d}")

# Feature patterns in errors
if error_indices.size > 0:
    error_features = X[error_indices]
    correct_features = X[~errors]
    
    FEATURE_NAMES = [
        "atr", "hour", "returns_5m", "body_size", "bb_upper", "kijun_sen",
        "williams_r", "is_closing_hour", "rsi", "high_low_range", "volume_change",
        "upper_shadow", "realized_vol_30m", "atr_percent", "tenkan_sen", "bb_position",
        "lower_shadow", "stoch_d", "garman_klass_vol", "realized_vol_60m",
        "volume_ma_ratio", "price_momentum_slope", "bb_lower", "macd_histogram",
        "intraday_momentum", "sentiment", "vol_regime", "trend_strength", 
        "volume_anomaly", "momentum_divergence"
    ]
    
    print(f"\n📊 Feature Patterns in Errors:")
    print(f"{'Feature':<25} {'Error Mean':<12} {'Correct Mean':<14} {'Diff':<10}")
    print("-" * 70)
    
    for i, feat_name in enumerate(FEATURE_NAMES[:15]):  # Top 15 features
        error_mean = error_features[:, i].mean()
        correct_mean = correct_features[:, i].mean()
        diff = error_mean - correct_mean
        print(f"{feat_name:<25} {error_mean:<12.4f} {correct_mean:<14.4f} {diff:<+10.4f}")

# High-confidence errors (model was sure but wrong)
high_conf_threshold = 0.7
high_conf_errors = [i for i in error_indices if proba[i] > high_conf_threshold or proba[i] < (1-high_conf_threshold)]

print(f"\n🎯 High Confidence Errors (proba > {high_conf_threshold} or < {1-high_conf_threshold}):")
print(f"   Count: {len(high_conf_errors)}")
print(f"   These are the 'surprising' errors where model was very confident but wrong")

if len(high_conf_errors) > 0:
    print(f"\n   Sample high-confidence errors:")
    for i in high_conf_errors[:5]:
        print(f"      Date: {dates[i]} | Pred: {proba[i]:.4f} | Actual: {y_dir[i]} | Mag: {y_mag[i]:.4f}")

# Save error analysis
import json
error_report = {
    'total_errors': int(len(error_indices)),
    'error_rate': float(len(error_indices)/len(y_dir)),
    'false_positives': fp_count,
    'false_negatives': fn_count,
    'fp_confidence_mean': float(np.mean(fp_probas)) if fp_probas else 0,
    'fn_confidence_mean': float(np.mean(fn_probas)) if fn_probas else 0,
    'high_confidence_errors': len(high_conf_errors)
}

with open(OUTPUT_DIR / "error_analysis.json", 'w') as f:
    json.dump(error_report, f, indent=2)

print(f"\n✅ Error analysis saved to: {OUTPUT_DIR}/error_analysis.json")

print("\n💡 Recommendations to Reduce Errors:")
print("   1. Add regime-specific models (train separate models for high/low vol)")
print("   2. Increase training weight on high-magnitude moves")
print("   3. Add ensemble methods to reduce high-confidence errors")
print("   4. Filter trades during error-prone hours")
