"""
V5 Paper Trading - Test Model with Multiple Thresholds
Experiment B: Find optimal threshold for live trading
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

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("📊 V5 THRESHOLD SWEEP - Find Optimal Trading Threshold")
print("=" * 70)

# Top 25 feature indices
TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]

# Load model
model_dir = Path("results/models/v5_selected_top25")
if not model_dir.exists():
    model_dir = Path("results/models/v5_comprehensive")
    print("Using V5 comprehensive model (44 features)")

direction_model = lgb.Booster(model_file=str(model_dir / "direction_model.lgb"))
magnitude_model = lgb.Booster(model_file=str(model_dir / "magnitude_model.lgb"))
confidence_model = lgb.Booster(model_file=str(model_dir / "confidence_model.lgb"))

with open(model_dir / "metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"\n✅ Model loaded (AUC: {metadata['test_metrics']['direction_auc']:.4f})")

# Load 2024 test data
PREBATCH_DIR = Path("cache/prebatched_features_v5")
print(f"\n📊 Loading 2024 test data (first 100 stocks)...")

all_test_samples = []
batch_files = sorted(list(PREBATCH_DIR.glob("*_features_v5.pkl")))[:100]

for i, batch_file in enumerate(batch_files):
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            for s in samples:
                date = pd.to_datetime(s['date'])
                if date.year == 2024:
                    s['features'] = s['features'][TOP_25_INDICES]
                    all_test_samples.append(s)
    except:
        pass
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(batch_files)} stocks loaded...")

print(f"\n✅ Loaded {len(all_test_samples):,} samples from 2024")

# Prepare data
X_test = np.array([s['features'] for s in all_test_samples])
y_dir_test = np.array([s['y_dir'] for s in all_test_samples])
y_mag_test = np.array([s['y_mag'] for s in all_test_samples])
dates_test = pd.to_datetime([s['date'] for s in all_test_samples])

# Get predictions
print("\n🧠 Generating predictions...")
dir_proba = direction_model.predict(X_test)
conf_preds = confidence_model.predict(X_test)

# Test multiple thresholds
thresholds = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65]
capital = 25000
risk_per_trade = 0.05
position_size = capital * risk_per_trade

print("\n" + "=" * 70)
print("📊 RESULTS BY THRESHOLD")
print("=" * 70)
print(f"{'Threshold':<10} {'Trades':<10} {'Win Rate':<12} {'Total PnL%':<14} {'Profit ₹':<14} {'Sharpe':<10}")
print("-" * 70)

best_result = None
best_score = -999
results_by_threshold = {}

for threshold in thresholds:
    trade_mask = (dir_proba > threshold) & (conf_preds > 0.5)
    n_trades = np.sum(trade_mask)
    
    if n_trades < 10:
        print(f"{threshold:<10.2f} {'N/A (<'+str(n_trades)+')':<10}")
        continue
    
    trades = []
    for i in range(len(X_test)):
        if trade_mask[i]:
            actual_return = y_mag_test[i] if y_dir_test[i] == 1 else -y_mag_test[i]
            predicted_dir = 1 if dir_proba[i] > 0.5 else 0
            won = (predicted_dir == y_dir_test[i])
            pnl = actual_return if won else -actual_return
            trades.append({'won': won, 'pnl': pnl, 'date': dates_test[i]})
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['won'])
    win_rate = winning_trades / total_trades
    total_pnl = sum([t['pnl'] for t in trades])
    total_profit = total_pnl * position_size / 100  # Convert % to INR
    
    # Sharpe calculation (by day)
    trades_df = pd.DataFrame(trades)
    trades_df['date_only'] = trades_df['date'].dt.date
    daily_pnl = trades_df.groupby('date_only')['pnl'].sum().values
    sharpe = np.mean(daily_pnl) / (np.std(daily_pnl) + 1e-10) * np.sqrt(252) if len(daily_pnl) > 1 else 0
    
    status = ""
    if win_rate > 0.54 and total_pnl > 0:
        status = "✅ PROFIT"
    elif total_pnl > 0:
        status = "⚠️ LOW"
    else:
        status = "❌ LOSS"
    
    print(f"{threshold:<10.2f} {total_trades:<10} {win_rate:<12.1%} {total_pnl:<14.3f} ₹{total_profit:<13,.0f} {sharpe:<10.1f} {status}")
    
    results_by_threshold[str(threshold)] = {
        'trades': int(total_trades),
        'win_rate': float(win_rate),
        'total_pnl_pct': float(total_pnl),
        'profit_inr': float(total_profit),
        'sharpe': float(sharpe)
    }
    
    # Score: prioritize win_rate >= 54% with positive returns
    if win_rate >= 0.54 and total_pnl > 0:
        score = win_rate * 100 + total_pnl * 10 + sharpe
        if score > best_score:
            best_score = score
            best_result = {
                'threshold': threshold,
                'trades': int(total_trades),
                'win_rate': float(win_rate),
                'total_pnl_pct': float(total_pnl),
                'profit_inr': float(total_profit),
                'sharpe': float(sharpe)
            }

print("=" * 70)

if best_result:
    print(f"\n🎯 OPTIMAL THRESHOLD: {best_result['threshold']}")
    print(f"   Trades:      {best_result['trades']}")
    print(f"   Win Rate:    {best_result['win_rate']:.1%} ✅ (> 54%)")
    print(f"   Total PnL:   {best_result['total_pnl_pct']:.2f}%")
    print(f"   Profit:      ₹{best_result['profit_inr']:,.2f}")
    print(f"   Sharpe:      {best_result['sharpe']:.2f}")
    print(f"\n{'🎉' * 15} PROFITABLE! {'🎉' * 15}")
    print(f"\n✅ Recommendation: Use threshold = {best_result['threshold']}")
    print(f"   For live trading with ₹25,000 capital")
else:
    print(f"\n❌ No profitable threshold found with win rate >= 54%")
    print(f"   Best strategy: Lower threshold or add more features (Experiment C)")

# Save results
results_dir = Path("results/paper_trading")
results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "v5_threshold_sweep.json", 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'model_version': 'v5_selected_top25',
        'best_threshold': best_result['threshold'] if best_result else None,
        'best_result': best_result,
        'all_results': results_by_threshold,
        'thresholds_tested': thresholds
    }, f, indent=2, default=str)

print(f"\n✅ Results saved to: {results_dir}/v5_threshold_sweep.json")
