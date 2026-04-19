"""
V6 Market Regime Analysis
Analyze model performance across different market conditions
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("🌊 V6 MARKET REGIME ANALYSIS")
print("=" * 80)

# Load data
PREBATCH_DIR = Path("cache/prebatched_features_v6")
MODEL_DIR = Path("results/models/v6_advanced")
OUTPUT_DIR = Path("results/v6_deep_analysis")

# Load model
import lightgbm as lgb
direction_model = lgb.Booster(model_file=str(MODEL_DIR / "direction_model.lgb"))

# Load 2024 data
print("\n📊 Loading data...")
all_data = []
for batch_file in list(PREBATCH_DIR.glob("*_features_v6.pkl"))[:100]:
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            for s in samples:
                if pd.to_datetime(s['date']).year == 2024:
                    all_data.append(s)
    except:
        pass

if len(all_data) == 0:
    print("❌ No data loaded!")
    sys.exit(1)

X = np.array([s['features'] for s in all_data])
y_dir = np.array([s['y_dir'] for s in all_data])
y_mag = np.array([s['y_mag'] for s in all_data])
dates = pd.to_datetime([s['date'] for s in all_data])

# Get predictions
proba = direction_model.predict(X)
preds = (proba > 0.5).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'y_true': y_dir,
    'y_pred': preds,
    'y_proba': proba,
    'magnitude': y_mag
})

# Define regimes based on features (using feature indices)
# atr_percent = 13, realized_vol_30m = 12
atr_values = X[:, 13]  # atr_percent
vol_values = X[:, 12]  # realized_vol_30m
returns_5m = X[:, 2]

# Regime classification
df['volatility_regime'] = pd.cut(atr_values, 
                                  bins=[0, 0.005, 0.015, 0.03, 1.0],
                                  labels=['Very Low', 'Low', 'Medium', 'High'])

df['trend_regime'] = pd.cut(returns_5m,
                             bins=[-1, -0.002, 0.002, 1],
                             labels=['Downtrend', 'Sideways', 'Uptrend'])

# Time-based regimes
df['hour'] = df['date'].dt.hour
df['time_regime'] = pd.cut(df['hour'],
                            bins=[0, 11, 13, 15, 24],
                            labels=['Opening', 'Midday', 'Afternoon', 'Close'])

print("\n📊 Performance by Volatility Regime:")
vol_perf = df.groupby('volatility_regime').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df.loc[x.index, 'y_true']).mean(),
    'y_proba': 'mean'
}).rename(columns={'y_true': 'count', 'y_pred': 'accuracy', 'y_proba': 'avg_pred'})
print(vol_perf)

print("\n📊 Performance by Trend Regime:")
trend_perf = df.groupby('trend_regime').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df.loc[x.index, 'y_true']).mean(),
    'y_proba': 'mean'
}).rename(columns={'y_true': 'count', 'y_pred': 'accuracy', 'y_proba': 'avg_pred'})
print(trend_perf)

print("\n📊 Performance by Time of Day:")
time_perf = df.groupby('time_regime').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df.loc[x.index, 'y_true']).mean(),
    'y_proba': 'mean'
}).rename(columns={'y_true': 'count', 'y_pred': 'accuracy', 'y_proba': 'avg_pred'})
print(time_perf)

# Combined regimes
print("\n📊 Performance by Combined Regimes (Volatility + Trend):")
df['combined_regime'] = df['volatility_regime'].astype(str) + ' + ' + df['trend_regime'].astype(str)
combined_perf = df.groupby('combined_regime').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df.loc[x.index, 'y_true']).mean()
}).rename(columns={'y_true': 'count', 'y_pred': 'accuracy'})
combined_perf = combined_perf[combined_perf['count'] >= 100]  # Filter for significance
combined_perf = combined_perf.sort_values('accuracy', ascending=False)
print(combined_perf.head(10))

# Save results
regime_results = {
    'volatility': vol_perf.to_dict(),
    'trend': trend_perf.to_dict(),
    'time': time_perf.to_dict(),
    'combined': combined_perf.head(10).to_dict()
}

import json
with open(OUTPUT_DIR / "regime_analysis.json", 'w') as f:
    json.dump(regime_results, f, indent=2, default=str)

print(f"\n✅ Regime analysis saved to: {OUTPUT_DIR}/regime_analysis.json")
print("\n💡 Key Insights:")
print(f"   • Best regime: {combined_perf.index[0]} (acc: {combined_perf.iloc[0]['accuracy']:.2%})")
print(f"   • Worst regime: {combined_perf.index[-1]} (acc: {combined_perf.iloc[-1]['accuracy']:.2%})")
print(f"   • Volatility sweet spot: {vol_perf['accuracy'].idxmax()}")
print(f"   • Best time to trade: {time_perf['accuracy'].idxmax()}")
