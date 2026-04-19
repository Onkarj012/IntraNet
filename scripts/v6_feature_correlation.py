"""
V6 Feature Correlation & Interaction Analysis
Deep dive into feature relationships and predictive power
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("🔗 V6 FEATURE CORRELATION & INTERACTION ANALYSIS")
print("=" * 80)

# Load data
PREBATCH_DIR = Path("cache/prebatched_features_v6")
OUTPUT_DIR = Path("results/v6_deep_analysis")

print("\n📊 Loading sample data for correlation analysis...")
all_data = []
for batch_file in list(PREBATCH_DIR.glob("*_features_v6.pkl"))[:50]:  # Sample for speed
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

FEATURE_NAMES = [
    "atr", "hour", "returns_5m", "body_size", "bb_upper", "kijun_sen",
    "williams_r", "is_closing_hour", "rsi", "high_low_range", "volume_change",
    "upper_shadow", "realized_vol_30m", "atr_percent", "tenkan_sen", "bb_position",
    "lower_shadow", "stoch_d", "garman_klass_vol", "realized_vol_60m",
    "volume_ma_ratio", "price_momentum_slope", "bb_lower", "macd_histogram",
    "intraday_momentum", "sentiment", "vol_regime", "trend_strength", 
    "volume_anomaly", "momentum_divergence"
]

# Create DataFrame
df_features = pd.DataFrame(X, columns=FEATURE_NAMES)
df_features['target'] = y_dir

print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features\n")

# ============================================================================
# CORRELATION WITH TARGET
# ============================================================================
print("📊 Feature-Target Correlations:")
print(f"{'Feature':<25} {'Pearson':<10} {'Spearman':<10} {'Mutual Info':<12}")
print("-" * 65)

correlations = []
for i, feat_name in enumerate(FEATURE_NAMES):
    feat_values = X[:, i]
    
    # Pearson correlation
    pearson_r, _ = pearsonr(feat_values, y_dir)
    
    # Spearman correlation
    spearman_r, _ = spearmanr(feat_values, y_dir)
    
    correlations.append({
        'feature': feat_name,
        'pearson': pearson_r,
        'spearman': spearman_r,
        'abs_pearson': abs(pearson_r),
        'abs_spearman': abs(spearman_r)
    })
    
    print(f"{feat_name:<25} {pearson_r:<10.4f} {spearman_r:<10.4f}")

# Sort by absolute correlation
corr_df = pd.DataFrame(correlations).sort_values('abs_pearson', ascending=False)

print(f"\n🏆 Top 10 Features by Pearson Correlation:")
for _, row in corr_df.head(10).iterrows():
    print(f"   {row['feature']:<25} r={row['pearson']:+.4f}")

print(f"\n📉 Bottom 5 Features by Correlation:")
for _, row in corr_df.tail(5).iterrows():
    print(f"   {row['feature']:<25} r={row['pearson']:+.4f}")

# ============================================================================
# FEATURE INTERACTIONS
# ============================================================================
print("\n" + "=" * 80)
print("🔗 Feature Interactions (Top Correlated Pairs):")
print("=" * 80)

# Compute correlation matrix
corr_matrix = df_features[FEATURE_NAMES].corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(FEATURE_NAMES)):
    for j in range(i+1, len(FEATURE_NAMES)):
        if corr_matrix.iloc[i, j] > 0.8:  # High correlation threshold
            high_corr_pairs.append({
                'feat1': FEATURE_NAMES[i],
                'feat2': FEATURE_NAMES[j],
                'correlation': corr_matrix.iloc[i, j]
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)

if len(high_corr_df) > 0:
    print(f"\n⚠️ Highly Correlated Feature Pairs (>0.8):")
    print(f"{'Feature 1':<20} {'Feature 2':<20} {'Corr':<8}")
    print("-" * 55)
    for _, row in high_corr_df.head(10).iterrows():
        print(f"{row['feat1']:<20} {row['feat2']:<20} {row['correlation']:<8.4f}")
else:
    print("\n✅ No highly correlated feature pairs found (>0.8)")

# ============================================================================
# V6 ADVANCED FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🚀 V6 Advanced Features Deep Dive:")
print("=" * 80)

adv_features = ['sentiment', 'vol_regime', 'trend_strength', 'volume_anomaly', 'momentum_divergence']

print(f"\n📊 Advanced Features vs Base Features Correlation:")
for adv_feat in adv_features:
    if adv_feat in FEATURE_NAMES:
        idx = FEATURE_NAMES.index(adv_feat)
        print(f"\n   {adv_feat}:")
        
        # Find most correlated base features
        correlations = []
        for i, base_feat in enumerate(FEATURE_NAMES[:25]):  # Base features only
            corr = np.corrcoef(X[:, idx], X[:, i])[0, 1]
            correlations.append((base_feat, abs(corr), corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        for base_feat, abs_corr, corr in correlations[:3]:
            print(f"      ↔ {base_feat:<20} r={corr:+.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
corr_df.to_csv(OUTPUT_DIR / "feature_correlations.csv", index=False)
if len(high_corr_df) > 0:
    high_corr_df.to_csv(OUTPUT_DIR / "high_correlation_pairs.csv", index=False)

print(f"\n✅ Results saved:")
print(f"   - {OUTPUT_DIR}/feature_correlations.csv")
print(f"   - {OUTPUT_DIR}/high_correlation_pairs.csv")
