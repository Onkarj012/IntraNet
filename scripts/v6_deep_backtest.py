"""
V6 Deep Backtesting & Model Analysis
Comprehensive in-depth analysis of V6 model performance
"""
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                            precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report,
                            brier_score_loss)
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.models.specialized import compute_expected_calibration_error

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("🔬 V6 DEEP BACKTESTING & MODEL ANALYSIS")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_DIR = Path("results/models/v6_advanced")
PREBATCH_DIR = Path("cache/prebatched_features_v6")
OUTPUT_DIR = Path("results/v6_deep_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Top 30 feature names for V6
FEATURE_NAMES = [
    # V5 Base 25
    "atr", "hour", "returns_5m", "body_size", "bb_upper", "kijun_sen",
    "williams_r", "is_closing_hour", "rsi", "high_low_range", "volume_change",
    "upper_shadow", "realized_vol_30m", "atr_percent", "tenkan_sen", "bb_position",
    "lower_shadow", "stoch_d", "garman_klass_vol", "realized_vol_60m",
    "volume_ma_ratio", "price_momentum_slope", "bb_lower", "macd_histogram",
    "intraday_momentum",
    # V6 Advanced 5
    "sentiment", "vol_regime", "trend_strength", "volume_anomaly", "momentum_divergence"
]

# Trading parameters
INITIAL_CAPITAL = 25000
RISK_PER_TRADE = 0.05  # 5% per trade
THRESHOLDS = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62]

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n📦 Loading V6 Model...")
try:
    direction_model = lgb.Booster(model_file=str(MODEL_DIR / "direction_model.lgb"))
    magnitude_model = lgb.Booster(model_file=str(MODEL_DIR / "magnitude_model.lgb"))
    confidence_model = lgb.Booster(model_file=str(MODEL_DIR / "confidence_model.lgb"))
    
    with open(MODEL_DIR / "metadata.json", 'r') as f:
        model_metadata = json.load(f)
    
    print(f"✅ Model loaded (AUC: {model_metadata['test_metrics']['direction_auc']:.4f})")
    print(f"   Features: {model_metadata['n_features']}")
    print(f"   Training samples: {model_metadata.get('training_samples', 'N/A'):,}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading 2024 Test Data (all available stocks)...")
all_samples = []
stocks_processed = 0

batch_files = list(PREBATCH_DIR.glob("*_features_v6.pkl"))
print(f"   Found {len(batch_files)} prebatch files")

for i, batch_file in enumerate(batch_files):
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            for s in samples:
                date = pd.to_datetime(s['date'])
                if date.year == 2024:
                    all_samples.append(s)
        stocks_processed += 1
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(batch_files)} stocks...")
    except Exception as e:
        pass

print(f"\n✅ Loaded {len(all_samples):,} samples from {stocks_processed} stocks")

if len(all_samples) == 0:
    print("❌ No data loaded!")
    sys.exit(1)

# Prepare data
X = np.array([s['features'] for s in all_samples])
y_dir = np.array([s['y_dir'] for s in all_samples])
y_mag = np.array([s['y_mag'] for s in all_samples])
y_conf = np.array([s['y_conf'] for s in all_samples])
dates = pd.to_datetime([s['date'] for s in all_samples])
stocks = [s.get('stock', 'UNKNOWN') for s in all_samples]

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================
print("\n🧠 Generating Predictions...")
dir_proba = direction_model.predict(X)
dir_preds = (dir_proba > 0.5).astype(int)
mag_preds = magnitude_model.predict(X)
conf_preds = confidence_model.predict(X)

# ============================================================================
# SECTION 1: MODEL PERFORMANCE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("📊 SECTION 1: MODEL PERFORMANCE METRICS")
print("=" * 80)

metrics = {}

# Basic metrics
metrics['accuracy'] = accuracy_score(y_dir, dir_preds)
metrics['auc'] = roc_auc_score(y_dir, dir_proba)
metrics['precision'] = precision_score(y_dir, dir_preds, zero_division=0)
metrics['recall'] = recall_score(y_dir, dir_preds, zero_division=0)
metrics['f1'] = f1_score(y_dir, dir_preds, zero_division=0)
metrics['brier'] = brier_score_loss(y_dir, dir_proba)
metrics['ece'] = compute_expected_calibration_error(y_dir, dir_proba)

# Confusion matrix
cm = confusion_matrix(y_dir, dir_preds)
tn, fp, fn, tp = cm.ravel()
metrics['true_positives'] = int(tp)
metrics['false_positives'] = int(fp)
metrics['true_negatives'] = int(tn)
metrics['false_negatives'] = int(fn)

print(f"\n🎯 Classification Metrics:")
print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
print(f"   AUC:         {metrics['auc']:.4f}")
print(f"   Precision:   {metrics['precision']:.4f}")
print(f"   Recall:      {metrics['recall']:.4f}")
print(f"   F1-Score:    {metrics['f1']:.4f}")
print(f"   Brier Score: {metrics['brier']:.4f}")
print(f"   ECE:         {metrics['ece']:.4f}")

print(f"\n📊 Confusion Matrix:")
print(f"                Predicted")
print(f"   Actual    DOWN    UP")
print(f"   DOWN      {tn:5d}   {fp:5d}  (Specificity: {tn/(tn+fp):.2%})")
print(f"   UP        {fn:5d}   {tp:5d}  (Sensitivity: {tp/(tp+fn):.2%})")

# Classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_dir, dir_preds, target_names=['DOWN', 'UP']))

# ============================================================================
# SECTION 2: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🔍 SECTION 2: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importance
importance_gain = direction_model.feature_importance(importance_type='gain')
importance_split = direction_model.feature_importance(importance_type='split')

# Normalize
importance_gain = importance_gain / (importance_gain.sum() + 1e-10)
importance_split = importance_split / (importance_split.sum() + 1e-10)

# Sort by gain
sorted_idx = np.argsort(importance_gain)[::-1]

print(f"\n🏆 Top 15 Most Important Features (by Gain):")
print(f"{'Rank':<6} {'Feature':<25} {'Gain %':<10} {'Split %':<10}")
print("-" * 60)
for rank, idx in enumerate(sorted_idx[:15], 1):
    feat_name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
    print(f"{rank:<6} {feat_name:<25} {importance_gain[idx]*100:<10.2f} {importance_split[idx]*100:<10.2f}")

print(f"\n📉 Bottom 10 Features:")
for rank, idx in enumerate(sorted_idx[-10:], len(sorted_idx)-9):
    feat_name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
    print(f"{rank:<6} {feat_name:<25} {importance_gain[idx]*100:<10.2f} {importance_split[idx]*100:<10.2f}")

# V6 advanced features importance
print(f"\n🚀 V6 Advanced Features Performance:")
adv_indices = [25, 26, 27, 28, 29]  # Last 5 features
for i, idx in enumerate(adv_indices):
    if idx < len(FEATURE_NAMES):
        feat_name = FEATURE_NAMES[idx]
        print(f"   {feat_name:<20} Gain: {importance_gain[idx]*100:.2f}%  (Rank: {list(sorted_idx).index(idx)+1})")

# Save feature importance
feature_imp_df = pd.DataFrame({
    'feature': [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}" for i in range(len(importance_gain))],
    'importance_gain': importance_gain,
    'importance_split': importance_split,
    'rank': [list(sorted_idx).index(i)+1 for i in range(len(importance_gain))]
}).sort_values('importance_gain', ascending=False)

feature_imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
print(f"\n✅ Feature importance saved to: {OUTPUT_DIR}/feature_importance.csv")

# ============================================================================
# SECTION 3: PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📈 SECTION 3: PREDICTION DISTRIBUTION ANALYSIS")
print("=" * 80)

# Probability distribution
proba_up = dir_proba[y_dir == 1]
proba_down = dir_proba[y_dir == 0]

print(f"\n📊 Probability Distribution:")
print(f"   UP actual:    mean={proba_up.mean():.4f}, std={proba_up.std():.4f}")
print(f"   DOWN actual:  mean={proba_down.mean():.4f}, std={proba_down.std():.4f}")
print(f"   Separation:   {proba_up.mean() - proba_down.mean():.4f}")

# Confidence distribution
conf_up = conf_preds[y_dir == 1]
conf_down = conf_preds[y_dir == 0]

print(f"\n📊 Confidence Distribution:")
print(f"   UP actual:    mean={conf_up.mean():.4f}, std={conf_up.std():.4f}")
print(f"   DOWN actual:  mean={conf_down.mean():.4f}, std={conf_down.std():.4f}")

# Prediction bins
bins = [0, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 1.0]
bin_labels = ['<0.45', '0.45-0.48', '0.48-0.50', '0.50-0.52', '0.52-0.55', '0.55-0.60', '>0.60']

print(f"\n📊 Predictions by Probability Bin:")
for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
    mask = (dir_proba >= low) & (dir_proba < high)
    count = mask.sum()
    if count > 0:
        accuracy = (dir_preds[mask] == y_dir[mask]).mean()
        actual_up_rate = y_dir[mask].mean()
        print(f"   {bin_labels[i]:<10} n={count:6d}  acc={accuracy:.2%}  actual_up={actual_up_rate:.2%}")

# ============================================================================
# SECTION 4: TRADING SIMULATION
# ============================================================================
print("\n" + "=" * 80)
print("💰 SECTION 4: TRADING SIMULATION & BACKTESTING")
print("=" * 80)

trading_results = {}

for threshold in THRESHOLDS:
    trades = []
    
    for i in range(len(X)):
        if dir_proba[i] > threshold and conf_preds[i] > 0.5:
            # Enter LONG position
            actual_return = y_mag[i] if y_dir[i] == 1 else -y_mag[i]
            predicted_dir = 1 if dir_proba[i] > 0.5 else 0
            won = (predicted_dir == y_dir[i])
            pnl = actual_return if won else -actual_return
            
            trades.append({
                'date': dates[i],
                'stock': stocks[i],
                'predicted_proba': dir_proba[i],
                'predicted_dir': predicted_dir,
                'actual_dir': y_dir[i],
                'actual_return': actual_return,
                'magnitude_pred': mag_preds[i],
                'confidence': conf_preds[i],
                'won': won,
                'pnl': pnl,
                'features': X[i]
            })
    
    if len(trades) < 5:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = trades_df['won'].sum()
    win_rate = winning_trades / total_trades
    avg_pnl = trades_df['pnl'].mean()
    total_pnl = trades_df['pnl'].sum()
    
    # Risk metrics
    position_size = INITIAL_CAPITAL * RISK_PER_TRADE
    profits = trades_df['pnl'] * position_size
    total_profit = profits.sum()
    
    # Sharpe (simplified)
    returns = trades_df['pnl'].values
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 1 else 0
    
    # Max drawdown calculation
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    # Daily aggregation
    trades_df['date_only'] = trades_df['date'].dt.date
    daily_stats = trades_df.groupby('date_only').agg({
        'pnl': ['count', 'sum', 'mean'],
        'won': 'sum'
    }).reset_index()
    daily_stats.columns = ['date', 'trades', 'daily_pnl', 'avg_pnl', 'wins']
    daily_stats['win_rate'] = daily_stats['wins'] / daily_stats['trades']
    
    profitable_days = (daily_stats['daily_pnl'] > 0).sum()
    total_days = len(daily_stats)
    
    trading_results[threshold] = {
        'threshold': threshold,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'total_profit_inr': total_profit,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'profitable_days': profitable_days,
        'total_days': total_days,
        'day_win_rate': profitable_days / total_days if total_days > 0 else 0,
        'trades_df': trades_df
    }
    
    print(f"\n📈 Threshold {threshold:.2f}:")
    print(f"   Trades:      {total_trades:4d} | Win Rate: {win_rate:.1%}")
    print(f"   Avg P&L:     {avg_pnl:+.4f}% | Total: {total_pnl:+.2f}%")
    print(f"   Profit:      ₹{total_profit:,.2f}")
    print(f"   Sharpe:      {sharpe:.2f}")
    print(f"   Max DD:      {max_dd:.2f}%")
    print(f"   Days:        {profitable_days}/{total_days} ({profitable_days/total_days:.1%})")

# Find best threshold
if trading_results:
    best_threshold = max(trading_results.items(), 
                          key=lambda x: x[1]['win_rate'] if x[1]['win_rate'] >= 0.54 else -1)[0]
    best = trading_results[best_threshold]
    
    print(f"\n🏆 BEST THRESHOLD: {best_threshold:.2f}")
    print(f"   Win Rate:    {best['win_rate']:.1%}")
    print(f"   Profit:      ₹{best['total_profit_inr']:,.2f}")
    print(f"   Sharpe:      {best['sharpe']:.2f}")
    
    # Save trades
    best['trades_df'].to_csv(OUTPUT_DIR / f"trades_threshold_{best_threshold:.2f}.csv", index=False)
    print(f"\n✅ Trades saved to: {OUTPUT_DIR}/trades_threshold_{best_threshold:.2f}.csv")

# ============================================================================
# SECTION 5: TEMPORAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📅 SECTION 5: TEMPORAL ANALYSIS")
print("=" * 80)

# Performance by month
df_analysis = pd.DataFrame({
    'date': dates,
    'y_true': y_dir,
    'y_pred': dir_preds,
    'y_proba': dir_proba,
    'confidence': conf_preds,
    'magnitude': y_mag
})
df_analysis['month'] = df_analysis['date'].dt.to_period('M')
df_analysis['hour'] = df_analysis['date'].dt.hour
df_analysis['day_of_week'] = df_analysis['date'].dt.day_name()

monthly_perf = df_analysis.groupby('month').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df_analysis.loc[x.index, 'y_true']).mean()
}).rename(columns={'y_true': 'samples', 'y_pred': 'accuracy'})

print(f"\n📊 Monthly Performance (2024):")
print(monthly_perf.to_string())

# Performance by hour
hourly_perf = df_analysis.groupby('hour').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df_analysis.loc[x.index, 'y_true']).mean()
}).rename(columns={'y_true': 'samples', 'y_pred': 'accuracy'})

print(f"\n⏰ Performance by Hour of Day:")
for hour, row in hourly_perf.iterrows():
    print(f"   {hour:2d}:00  n={row['samples']:6.0f}  acc={row['accuracy']:.2%}")

# Performance by day of week
dow_perf = df_analysis.groupby('day_of_week').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df_analysis.loc[x.index, 'y_true']).mean()
}).rename(columns={'y_true': 'samples', 'y_pred': 'accuracy'})

print(f"\n📅 Performance by Day of Week:")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
for day in day_order:
    if day in dow_perf.index:
        row = dow_perf.loc[day]
        print(f"   {day:<10} n={row['samples']:6.0f}  acc={row['accuracy']:.2%}")

# ============================================================================
# SECTION 6: PER-STOCK ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🏢 SECTION 6: PER-STOCK ANALYSIS")
print("=" * 80)

df_analysis['stock'] = stocks
stock_perf = df_analysis.groupby('stock').agg({
    'y_true': 'count',
    'y_pred': lambda x: (x == df_analysis.loc[x.index, 'y_true']).mean(),
    'y_proba': 'mean',
    'magnitude': 'mean'
}).rename(columns={'y_true': 'samples', 'y_pred': 'accuracy', 
                   'y_proba': 'avg_prediction', 'magnitude': 'avg_magnitude'})

# Filter stocks with at least 50 samples
stock_perf_filtered = stock_perf[stock_perf['samples'] >= 50].sort_values('accuracy', ascending=False)

print(f"\n🏆 Top 10 Best Performing Stocks (min 50 samples):")
print(f"{'Stock':<15} {'Samples':<8} {'Accuracy':<10} {'Avg Pred':<10}")
print("-" * 50)
for stock, row in stock_perf_filtered.head(10).iterrows():
    print(f"{stock:<15} {row['samples']:<8.0f} {row['accuracy']:<10.2%} {row['avg_prediction']:<10.4f}")

print(f"\n📉 Bottom 5 Worst Performing Stocks:")
for stock, row in stock_perf_filtered.tail(5).iterrows():
    print(f"{stock:<15} {row['samples']:<8.0f} {row['accuracy']:<10.2%} {row['avg_prediction']:<10.4f}")

# ============================================================================
# SECTION 7: CONFIDENCE VS ACCURACY
# ============================================================================
print("\n" + "=" * 80)
print("🎯 SECTION 7: CONFIDENCE CALIBRATION")
print("=" * 80)

conf_bins = np.arange(0.5, 1.0, 0.05)
print(f"\n📊 Accuracy by Confidence Bin:")
print(f"{'Conf Range':<12} {'Count':<8} {'Accuracy':<10} {'Expected':<10}")
print("-" * 50)

for i in range(len(conf_bins)-1):
    low, high = conf_bins[i], conf_bins[i+1]
    mask = (conf_preds >= low) & (conf_preds < high)
    count = mask.sum()
    if count > 0:
        acc = (dir_preds[mask] == y_dir[mask]).mean()
        expected = (low + high) / 2
        print(f"{low:.2f}-{high:.2f}     {count:<8} {acc:<10.2%} {expected:<10.2f}")

# High confidence trades
high_conf_mask = conf_preds > 0.7
if high_conf_mask.sum() > 0:
    high_conf_acc = (dir_preds[high_conf_mask] == y_dir[high_conf_mask]).mean()
    print(f"\n✨ High Confidence (>0.7) Trades:")
    print(f"   Count: {high_conf_mask.sum()}")
    print(f"   Accuracy: {high_conf_acc:.2%}")

# ============================================================================
# SECTION 8: RISK ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("⚠️  SECTION 8: RISK ANALYSIS")
print("=" * 80)

# Using best threshold results
if trading_results:
    best_trades = trading_results[best_threshold]['trades_df']
    returns = best_trades['pnl'].values
    
    # Risk metrics
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    cvar_95 = returns[returns <= var_95].mean() if any(returns <= var_95) else 0
    
    # Win/Loss ratio
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_loss_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else float('inf')
    
    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Consecutive wins/losses
    best_trades['win_int'] = best_trades['won'].astype(int)
    consecutive = []
    current = 1
    for i in range(1, len(best_trades)):
        if best_trades['win_int'].iloc[i] == best_trades['win_int'].iloc[i-1]:
            current += 1
        else:
            consecutive.append(current)
            current = 1
    consecutive.append(current)
    max_consecutive_wins = max([c for c, w in zip(consecutive, best_trades['win_int']) if w == 1], default=0)
    max_consecutive_losses = max([c for c, w in zip(consecutive, best_trades['win_int']) if w == 0], default=0)
    
    print(f"\n📊 Risk Metrics (Threshold {best_threshold:.2f}):")
    print(f"   VaR (95%):          {var_95:.4f}%")
    print(f"   VaR (99%):          {var_99:.4f}%")
    print(f"   CVaR (95%):         {cvar_95:.4f}%")
    print(f"   Win/Loss Ratio:     {win_loss_ratio:.2f}")
    print(f"   Profit Factor:      {profit_factor:.2f}")
    print(f"   Max Consec. Wins:   {max_consecutive_wins}")
    print(f"   Max Consec. Losses: {max_consecutive_losses}")

# ============================================================================
# SAVE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("💾 SAVING COMPREHENSIVE REPORT")
print("=" * 80)

report = {
    'timestamp': datetime.now().isoformat(),
    'model': 'v6_advanced',
    'n_features': 30,
    'test_samples': len(X),
    'metrics': metrics,
    'feature_importance': feature_imp_df.head(20).to_dict(orient='records'),
    'trading_results': {k: {key: v[key] for key in v if key != 'trades_df'} 
                       for k, v in trading_results.items()},
    'best_threshold': best_threshold if trading_results else None,
    'temporal_analysis': {
        'monthly': monthly_perf.to_dict(),
        'hourly': hourly_perf.to_dict(),
        'daily': dow_perf.to_dict()
    }
}

with open(OUTPUT_DIR / "v6_deep_analysis_report.json", 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"✅ Report saved: {OUTPUT_DIR}/v6_deep_analysis_report.json")

# Summary
print("\n" + "=" * 80)
print("🎉 V6 DEEP ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n📁 Output Directory: {OUTPUT_DIR}/")
print(f"   - feature_importance.csv")
print(f"   - trades_threshold_*.csv")
print(f"   - v6_deep_analysis_report.json")
print(f"\n✨ Key Takeaways:")
print(f"   • Model AUC: {metrics['auc']:.4f}")
print(f"   • Best Threshold: {best_threshold:.2f if trading_results else 'N/A'}")
print(f"   • Best Win Rate: {trading_results[best_threshold]['win_rate']:.1% if trading_results else 'N/A'}")
print(f"   • Top Feature: {FEATURE_NAMES[sorted_idx[0]]} ({importance_gain[sorted_idx[0]]*100:.1f}%)")
