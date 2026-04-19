"""
Fast V5 Backtest - Quick simulation using sampled 2024 data
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

# Top 25 feature indices
TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]

print("=" * 70)
print("📊 V5 PAPER TRADING SIMULATION (Fast Version)")
print("=" * 70)

# Load model
model_dir = Path("results/models/v5_selected_top25")
if not model_dir.exists():
    model_dir = Path("results/models/v5_comprehensive")

direction_model = lgb.Booster(model_file=str(model_dir / "direction_model.lgb"))
magnitude_model = lgb.Booster(model_file=str(model_dir / "magnitude_model.lgb"))
confidence_model = lgb.Booster(model_file=str(model_dir / "confidence_model.lgb"))

with open(model_dir / "metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"\n✅ Model loaded (AUC: {metadata['test_metrics']['direction_auc']:.4f})")

# Sample 50 stocks for speed
PREBATCH_DIR = Path("cache/prebatched_features_v5")
print(f"\n📊 Loading 50 sample stocks from 2024...")

all_test_samples = []
batch_files = sorted(list(PREBATCH_DIR.glob("*_features_v5.pkl")))[:50]  # First 50 only

for i, batch_file in enumerate(batch_files):
    try:
        with open(batch_file, 'rb') as f:
            samples = pickle.load(f)
            for s in samples:
                date = pd.to_datetime(s['date'])
                if date.year == 2024:
                    s['features'] = s['features'][TOP_25_INDICES]
                    all_test_samples.append(s)
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i+1}/50 stocks...")
    except:
        pass

if len(all_test_samples) == 0:
    print("❌ No 2024 test data found!")
    sys.exit(1)

print(f"\n✅ Loaded {len(all_test_samples):,} samples from 2024\n")

# Prepare data
X_test = np.array([s['features'] for s in all_test_samples])
y_dir_test = np.array([s['y_dir'] for s in all_test_samples])
y_mag_test = np.array([s['y_mag'] for s in all_test_samples])
dates_test = pd.to_datetime([s['date'] for s in all_test_samples])

# Get predictions
print("🧠 Generating predictions...")
dir_proba = direction_model.predict(X_test)
conf_preds = confidence_model.predict(X_test)

# Trading simulation - test multiple thresholds
thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
results_by_threshold = {}

trades = []
for i in range(len(X_test)):
    if trade_mask[i]:
        actual_return = y_mag_test[i] if y_dir_test[i] == 1 else -y_mag_test[i]
        predicted_dir = 1 if dir_proba[i] > 0.5 else 0
        won = (predicted_dir == y_dir_test[i])
        pnl = actual_return if won else -actual_return
        
        trades.append({
            'date': dates_test[i],
            'won': won,
            'pnl': pnl,
            'confidence': conf_preds[i]
        })

if len(trades) == 0:
    print("⚠️ No trades met the threshold criteria")
    sys.exit(1)

print(f"\n📈 EXECUTED {len(trades)} TRADES")
print("=" * 70)

# Statistics
total_trades = len(trades)
winning_trades = sum(1 for t in trades if t['won'])
win_rate = winning_trades / total_trades
avg_pnl = np.mean([t['pnl'] for t in trades])
total_pnl = sum([t['pnl'] for t in trades])

# Capital simulation
capital = 25000
risk_per_trade = 0.05
position_size = capital * risk_per_trade
profits = [t['pnl'] * position_size for t in trades]
total_profit = sum(profits)

# Daily aggregation
trades_df = pd.DataFrame(trades)
trades_df['date_only'] = trades_df['date'].dt.date
daily_stats = trades_df.groupby('date_only').agg({
    'pnl': ['count', 'sum', 'mean'],
    'won': 'sum'
}).reset_index()
daily_stats.columns = ['date', 'trades', 'daily_pnl', 'avg_pnl', 'wins']
daily_stats['win_rate'] = daily_stats['wins'] / daily_stats['trades']

profitable_days = sum(1 for p in daily_stats['daily_pnl'] if p > 0)
total_days = len(daily_stats)
daily_returns = daily_stats['daily_pnl'].values
sharpe_like = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)

print(f"  Total Trades:        {total_trades}")
print(f"  Trading Days:        {total_days}")
print(f"  Win Rate:            {win_rate:.1%} {'✅ > 54%' if win_rate > 0.54 else '❌ < 54%'}")
print(f"  Profitable Days:     {profitable_days}/{total_days} ({profitable_days/total_days:.1%})")
print(f"  Avg Return/Trade:    {avg_pnl:.4f}%")
print(f"  Total Return:        {total_pnl:.2f}%")
print(f"  Sharpe-like Ratio:   {sharpe_like:.2f}")
print(f"  Simulated Profit:    ₹{total_profit:,.2f}")
print("=" * 70)

# Save results
results_dir = Path("results/paper_trading")
results_dir.mkdir(parents=True, exist_ok=True)

results_data = {
    'timestamp': datetime.now().isoformat(),
    'model_version': 'v5_selected_top25',
    'simulation': '2024_test_data_fast',
    'stocks_tested': 50,
    'capital': 25000,
    'threshold': threshold,
    'metrics': {
        'total_trades': total_trades,
        'win_rate': float(win_rate),
        'profitable_days': int(profitable_days),
        'total_days': int(total_days),
        'day_profitability_rate': float(profitable_days/total_days),
        'avg_pnl': float(avg_pnl),
        'total_pnl': float(total_pnl),
        'sharpe_like': float(sharpe_like),
        'simulated_profit_inr': float(total_profit)
    }
}

with open(results_dir / "v5_backtest_fast_results.json", 'w') as f:
    json.dump(results_data, f, indent=2, default=str)

print(f"\n✅ Results saved to: {results_dir}/v5_backtest_fast_results.json")

# Verdict
if win_rate > 0.54 and total_profit > 0 and sharpe_like > 0.8:
    print(f"\n🎉🎉🎉 PROFITABLE! Ready for live trading! 🎉🎉🎉")
elif win_rate > 0.50 and total_profit > 0:
    print(f"\n⚠️ Slightly profitable. Need improvement to reach 54% win rate.")
else:
    print(f"\n❌ Not profitable. Need Experiment C (advanced features).")
