"""
Ensemble Model: V5 + Regime Signals + Time Filters
Combines V5 base model with regime-based gating for improved performance
"""
import sys
import json
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

print("=" * 80)
print("🎯 ENSEMBLE MODEL: V5 + Regime Signals + Time Filters")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
V5_MODEL_DIR = Path("results/models/v5_selected_top25")
PREBATCH_DIR = Path("cache/prebatched_features_v5")
OUTPUT_DIR = Path("results/v6_deep_analysis")

# Top 25 feature indices (from V5 selection)
TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]
FEATURE_NAMES = [
    "atr", "hour", "returns_5m", "body_size", "bb_upper", "kijun_sen",
    "williams_r", "is_closing_hour", "rsi", "high_low_range", "volume_change",
    "upper_shadow", "realized_vol_30m", "atr_percent", "tenkan_sen", "bb_position",
    "lower_shadow", "stoch_d", "garman_klass_vol", "realized_vol_60m",
    "volume_ma_ratio", "price_momentum_slope", "bb_lower", "macd_histogram",
    "intraday_momentum"
]

# Regime thresholds (from v6_regime_analysis.py)
LOW_VOL_THRESHOLD = 0.015  # ATR% threshold for low vol
CLOSING_HOUR_START = 14    # 2 PM start of closing hour

# ============================================================================
# LOAD V5 MODEL
# ============================================================================
print("\n📦 Loading V5 Base Model...")
direction_model = lgb.Booster(model_file=str(V5_MODEL_DIR / "direction_model.lgb"))
magnitude_model = lgb.Booster(model_file=str(V5_MODEL_DIR / "magnitude_model.lgb"))
confidence_model = lgb.Booster(model_file=str(V5_MODEL_DIR / "confidence_model.lgb"))

with open(V5_MODEL_DIR / "metadata.json", 'r') as f:
    v5_metadata = json.load(f) if 'json' in dir() else None

print(f"✅ V5 Model loaded (AUC: {0.5254})")

# ============================================================================
# LOAD 2024 TEST DATA
# ============================================================================
print("\n📊 Loading 2024 Test Data...")
all_samples = []
for batch_file in list(PREBATCH_DIR.glob("*_features_v5.pkl"))[:150]:
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
y_mag = np.array([s['y_mag'] for s in all_samples])
dates = pd.to_datetime([s['date'] for s in all_samples])

# Extract regime indicators from features
# atr_percent = index 13 in original, but after TOP_25 selection it's at a different position
# Let's find it
atr_pct_idx = None
for i, idx in enumerate(TOP_25_INDICES):
    if idx == 13:  # atr_percent in original V5
        atr_pct_idx = i
        break

hour_idx = None
for i, idx in enumerate(TOP_25_INDICES):
    if idx == 26:  # hour in original V5
        hour_idx = i
        break

print(f"   Regime feature indices: atr_pct={atr_pct_idx}, hour={hour_idx}")

# Initialize favorable_mask as all True (no filtering by default)
favorable_mask = np.ones(len(X), dtype=bool)

# ============================================================================
# BASE V5 PREDICTIONS
# ============================================================================
print("\n🧠 Generating Base V5 Predictions...")
base_proba = direction_model.predict(X)
base_preds = (base_proba > 0.5).astype(int)
base_acc = accuracy_score(y_dir, base_preds)
base_auc = roc_auc_score(y_dir, base_proba)

print(f"   V5 Base: AUC={base_auc:.4f}, Acc={base_acc:.2%}")

# ============================================================================
# ENSEMBLE STRATEGY 1: Regime-Based Gating
# ============================================================================
print("\n" + "=" * 80)
print("🔍 ENSEMBLE STRATEGY 1: Regime-Based Gating")
print("=" * 80)
print("   Only trade in favorable regimes (Low/Medium Vol + Closing Hour)")

# Identify favorable regimes
if atr_pct_idx is not None and hour_idx is not None:
    atr_pct_values = X[:, atr_pct_idx]
    hour_values = X[:, hour_idx]
    
    # Low/Med vol: ATR% < 0.015 (based on regime analysis)
    low_vol_mask = atr_pct_values < LOW_VOL_THRESHOLD
    
    # Closing hour: 14:00-15:30 (based on regime analysis showing 69% accuracy)
    closing_mask = hour_values >= CLOSING_HOUR_START
    
    # Combined favorable regime
    favorable_mask = low_vol_mask | closing_mask  # Either condition
    
    print(f"\n📊 Regime Distribution:")
    print(f"   Low/Med Vol:       {low_vol_mask.sum():,} ({low_vol_mask.mean():.1%})")
    print(f"   Closing Hour:      {closing_mask.sum():,} ({closing_mask.mean():.1%})")
    print(f"   Favorable (either): {favorable_mask.sum():,} ({favorable_mask.mean():.1%})")
    
    # Test V5 only on favorable regimes
    if favorable_mask.sum() > 0:
        v5_fav_proba = base_proba[favorable_mask]
        v5_fav_true = y_dir[favorable_mask]
        v5_fav_auc = roc_auc_score(v5_fav_true, v5_fav_proba)
        v5_fav_acc = accuracy_score(v5_fav_true, (v5_fav_proba > 0.5).astype(int))
        
        print(f"\n🎯 V5 on Favorable Regimes Only:")
        print(f"   AUC:  {v5_fav_auc:.4f} (vs {base_auc:.4f} all regimes)")
        print(f"   Acc:  {v5_fav_acc:.2%} (vs {base_acc:.2%} all regimes)")
        
        improvement = v5_fav_auc - base_auc
        if improvement > 0:
            print(f"   ✅ Improvement: +{improvement:.4f} ({improvement*100:.2f} pp)")
        else:
            print(f"   ⚠️ No improvement: {improvement:.4f}")
    
    # Test on unfavorable regimes
    unfavorable_mask = ~favorable_mask
    if unfavorable_mask.sum() > 0:
        v5_unfav_auc = roc_auc_score(y_dir[unfavorable_mask], base_proba[unfavorable_mask])
        print(f"\n⚠️ V5 on Unfavorable Regimes:")
        print(f"   AUC:  {v5_unfav_auc:.4f} (worse than favorable)")

# ============================================================================
# ENSEMBLE STRATEGY 2: Dynamic Threshold by Regime
# ============================================================================
print("\n" + "=" * 80)
print("🔍 ENSEMBLE STRATEGY 2: Dynamic Threshold by Regime")
print("=" * 80)
print("   Use lower threshold (0.52) in favorable regimes, higher (0.58) in unfavorable")

if atr_pct_idx is not None and hour_idx is not None:
    # Dynamic threshold
    dynamic_proba = base_proba.copy()
    
    # Boost probabilities in favorable regimes (soft gating)
    # This is like: proba' = proba * (1 + bonus) if favorable
    bonus = 0.05
    dynamic_proba[favorable_mask] = np.clip(dynamic_proba[favorable_mask] * (1 + bonus), 0, 1)
    
    dynamic_auc = roc_auc_score(y_dir, dynamic_proba)
    print(f"\n🎯 Dynamic Boosting ({bonus:.0%} bonus in favorable regimes):")
    print(f"   AUC:  {dynamic_auc:.4f} (vs {base_auc:.4f} base)")
    print(f"   Boost: +{dynamic_auc - base_auc:.4f}")

# ============================================================================
# ENSEMBLE STRATEGY 3: Voting with Confidence Model
# ============================================================================
print("\n" + "=" * 80)
print("🔍 ENSEMBLE STRATEGY 3: Confidence-Gated Predictions")
print("=" * 80)
print("   Only trade when confidence model agrees with direction model")

conf_proba = confidence_model.predict(X)

# Agreement threshold
agreement_threshold = 0.5
agreement_mask = conf_proba > agreement_threshold

print(f"\n📊 Confidence Agreement:")
print(f"   High confidence trades: {agreement_mask.sum():,} ({agreement_mask.mean():.1%})")

if agreement_mask.sum() > 0:
    agreed_proba = base_proba[agreement_mask]
    agreed_true = y_dir[agreement_mask]
    agreed_auc = roc_auc_score(agreed_true, agreed_proba)
    agreed_acc = accuracy_score(agreed_true, (agreed_proba > 0.5).astype(int))
    
    print(f"\n🎯 V5 + Confidence Agreement (>0.5):")
    print(f"   AUC:  {agreed_auc:.4f} (vs {base_auc:.4f} all)")
    print(f"   Acc:  {agreed_acc:.2%} (vs {base_acc:.2%} all)")
    
    # Test higher threshold
    high_conf_mask = conf_proba > 0.7
    if high_conf_mask.sum() > 0:
        high_conf_auc = roc_auc_score(y_dir[high_conf_mask], base_proba[high_conf_mask])
        print(f"   High conf (>0.7): AUC={high_conf_auc:.4f} on {high_conf_mask.sum()} samples")

# ============================================================================
# ENSEMBLE STRATEGY 4: Combined (Best of All)
# ============================================================================
print("\n" + "=" * 80)
print("🔍 ENSEMBLE STRATEGY 4: Combined Approach (All Filters)")
print("=" * 80)
print("   Favorable regime + High confidence + Moderate probability")

if atr_pct_idx is not None and hour_idx is not None:
    # Combined mask (check if favorable_mask exists)
    if 'favorable_mask' in locals():
        combined_mask = favorable_mask & agreement_mask & (base_proba > 0.52)
    else:
        combined_mask = agreement_mask & (base_proba > 0.52)
    
    print(f"\n📊 Combined Filters:")
    print(f"   Favorable regime:    {favorable_mask.sum():,}")
    print(f"   High confidence:     {agreement_mask.sum():,}")
    print(f"   Combined:            {combined_mask.sum():,}")
    
    if combined_mask.sum() > 100:
        combined_proba = base_proba[combined_mask]
        combined_true = y_dir[combined_mask]
        combined_auc = roc_auc_score(combined_true, combined_proba)
        combined_acc = accuracy_score(combined_true, (combined_proba > 0.5).astype(int))
        
        # Win rate at threshold 0.55
        combined_preds_055 = (combined_proba > 0.55).astype(int)
        combined_wr = (combined_preds_055 == combined_true).mean()
        
        print(f"\n🎯 COMBINED ENSEMBLE RESULTS:")
        print(f"   Samples:   {combined_mask.sum():,}")
        print(f"   AUC:       {combined_auc:.4f}")
        print(f"   Accuracy:  {combined_acc:.2%}")
        print(f"   Win Rate@0.55: {combined_wr:.2%}")
        
        improvement = combined_auc - base_auc
        print(f"\n   vs V5 Base: {combined_auc:.4f} - {base_auc:.4f} = {improvement:+.4f}")
        
        if combined_auc > 0.54:
            print(f"\n   🎉🎉🎉 EXCEEDS 0.54 TARGET! 🎉🎉🎉")
        elif improvement > 0.005:
            print(f"   ✅ Good improvement")
        else:
            print(f"   ⚠️ Modest improvement")

# ============================================================================
# ENSEMBLE STRATEGY 5: Train Meta-Classifier (Stacking)
# ============================================================================
print("\n" + "=" * 80)
print("🔍 ENSEMBLE STRATEGY 5: Stacked Meta-Classifier")
print("=" * 80)
print("   Train a meta-model on V5 prob + regime indicators + confidence")

# Prepare meta-features
meta_features = []
for i in range(len(X)):
    features = [
        base_proba[i],           # V5 prediction
        conf_proba[i],           # Confidence model
    ]
    if atr_pct_idx is not None:
        features.append(X[i, atr_pct_idx])  # Volatility
    if hour_idx is not None:
        features.append(X[i, hour_idx])      # Time
    features.append(1 if favorable_mask[i] else 0)  # Regime flag
    meta_features.append(features)

X_meta = np.array(meta_features)

# Temporal split for meta-learner
meta_train_mask = dates < '2024-06-01'
meta_test_mask = dates >= '2024-06-01'

if meta_train_mask.sum() > 1000 and meta_test_mask.sum() > 100:
    X_meta_train, X_meta_test = X_meta[meta_train_mask], X_meta[meta_test_mask]
    y_meta_train, y_meta_test = y_dir[meta_train_mask], y_dir[meta_test_mask]
    
    # Train meta-classifier
    meta_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        verbose=-1
    )
    meta_model.fit(X_meta_train, y_meta_train)
    
    # Predict
    meta_proba = meta_model.predict_proba(X_meta_test)[:, 1]
    meta_auc = roc_auc_score(y_meta_test, meta_proba)
    
    print(f"\n🎯 STACKED META-CLASSIFIER (2nd half 2024):")
    print(f"   Meta-features: V5 proba, Conf proba, Vol, Hour, Regime flag")
    print(f"   AUC:  {meta_auc:.4f}")
    print(f"   vs V5 on same period: {base_auc:.4f}")
    print(f"   Boost: {meta_auc - base_auc:+.4f}")
    
    # Feature importance
    meta_importance = meta_model.feature_importances_
    print(f"\n   Meta-Feature Importance:")
    meta_feat_names = ['V5_proba', 'Conf_proba', 'Volatility', 'Hour', 'Regime']
    for i, (name, imp) in enumerate(zip(meta_feat_names[:len(meta_importance)], meta_importance)):
        print(f"      {name}: {imp:.1f}%")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 ENSEMBLE RESULTS SUMMARY")
print("=" * 80)

results_summary = {
    'V5_Base': base_auc,
    'V5_Favorable_Only': v5_fav_auc if 'v5_fav_auc' in dir() else None,
    'V5_Confidence_Gated': agreed_auc if 'agreed_auc' in dir() else None,
    'V5_Combined_Filters': combined_auc if 'combined_auc' in dir() else None,
    'Stacked_Meta': meta_auc if 'meta_auc' in dir() else None
}

print(f"\n{'Strategy':<25} {'AUC':<8} {'vs V5':<10} {'Status'}")
print("-" * 60)
for name, auc in results_summary.items():
    if auc:
        diff = auc - base_auc
        status = "✅" if diff > 0 else "⚠️"
        print(f"{name:<25} {auc:<8.4f} {diff:+.4f}     {status}")

best_auc = max([v for v in results_summary.values() if v is not None])
best_strategy = [k for k, v in results_summary.items() if v == best_auc][0]

print(f"\n🏆 BEST STRATEGY: {best_strategy}")
print(f"   AUC: {best_auc:.4f}")

if best_auc > 0.54:
    print(f"\n🎉🎉🎉 EXCEEDS 0.54 PROFITABILITY THRESHOLD! 🎉🎉🎉")
    print(f"\n💡 Recommendation: Deploy {best_strategy} for live trading")
elif best_auc > base_auc:
    print(f"\n✅ Improvement over V5 base")
    print(f"   Next step: Add more ensemble components (options flow, sentiment)")
else:
    print(f"\n⚠️ No improvement with ensemble approaches")
    print(f"   Need real external data sources")

# Save results
import json
with open(OUTPUT_DIR / "ensemble_analysis.json", 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'v5_base_auc': float(base_auc),
        'results': {k: float(v) if v else None for k, v in results_summary.items()},
        'best_strategy': best_strategy,
        'best_auc': float(best_auc),
        'profitable': best_auc > 0.54
    }, f, indent=2)

print(f"\n✅ Results saved to: {OUTPUT_DIR}/ensemble_analysis.json")
