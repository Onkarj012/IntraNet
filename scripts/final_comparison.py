"""
Final Comparison of All 3 Experiments
Generate comprehensive comparison and recommendation
"""
import json
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("📊 FINAL COMPARISON: ALL 3 EXPERIMENTS COMPLETE")
print("=" * 70)

results = {}

# === Experiment A: Feature Selection ===
print("\n🔍 Experiment A: Feature Selection (25 from 44)")
print("-" * 70)
v5_selected = Path("results/models/v5_selected_top25/metadata.json")
if v5_selected.exists():
    with open(v5_selected) as f:
        data = json.load(f)
    results['A_FeatureSelection'] = {
        'auc': data['test_metrics']['direction_auc'],
        'accuracy': data['test_metrics']['direction_accuracy'],
        'features': data.get('n_features', 25),
        'training_samples': data.get('training_samples', 'N/A'),
        'note': 'Kept top 25 features from 44 total'
    }
    print(f"  AUC:           {data['test_metrics']['direction_auc']:.4f}")
    print(f"  Accuracy:      {data['test_metrics']['direction_accuracy']:.2%}")
    print(f"  Features:      25 (selected from 44)")
    print(f"  Status:        {'✅ Same performance, more efficient' if data['test_metrics']['direction_auc'] > 0.52 else '⚠️ Below target'}")
else:
    print("  ❌ Results not found")

# === Experiment B: Paper Trading ===
print("\n📈 Experiment B: Paper Trading (Threshold Sweep)")
print("-" * 70)
paper = Path("results/paper_trading/v5_threshold_sweep.json")
if paper.exists():
    with open(paper) as f:
        data = json.load(f)
    best = data.get('best_result', {})
    results['B_PaperTrading'] = {
        'best_threshold': best.get('threshold'),
        'trades': best.get('trades'),
        'win_rate': best.get('win_rate'),
        'profit_inr': best.get('profit_inr'),
        'sharpe': best.get('sharpe')
    }
    if best:
        print(f"  Best Threshold:  {best.get('threshold')}")
        print(f"  Trades:          {best.get('trades')}")
        print(f"  Win Rate:        {best.get('win_rate'):.1%}")
        print(f"  Simulated Profit: ₹{best.get('profit_inr'):,.2f}")
        print(f"  Sharpe Ratio:    {best.get('sharpe'):.2f}")
        if best.get('win_rate', 0) > 0.54:
            print(f"  Status:          ✅ PROFITABLE!")
        else:
            print(f"  Status:          ⚠️ Below 54% win rate target")
    else:
        print("  ❌ No profitable threshold found")
else:
    print("  ❌ Results not found")

# === Experiment C: Advanced Features ===
print("\n🚀 Experiment C: Advanced Features (V6 - 30 features)")
print("-" * 70)
v6 = Path("results/models/v6_advanced/metadata.json")
if v6.exists():
    with open(v6) as f:
        data = json.load(f)
    results['C_AdvancedFeatures'] = {
        'auc': data['test_metrics']['direction_auc'],
        'accuracy': data['test_metrics']['direction_accuracy'],
        'features': data.get('n_features', 30),
        'improvement_v5': data.get('comparison', {}).get('improvement_v5', 0),
        'note': 'Added 5 synthetic advanced features'
    }
    print(f"  AUC:           {data['test_metrics']['direction_auc']:.4f}")
    print(f"  Accuracy:      {data['test_metrics']['direction_accuracy']:.2%}")
    print(f"  Features:      30 (25 base + 5 advanced)")
    imp = data.get('comparison', {}).get('improvement_v5', 0)
    if imp > 0:
        print(f"  vs V5:         +{imp:.4f} ✅ Improved")
    else:
        print(f"  vs V5:         {imp:.4f} ⚠️ No improvement")
    if data['test_metrics']['direction_auc'] > 0.54:
        print(f"  Status:        ✅🎯 EXCEEDS 0.54 THRESHOLD!")
    else:
        print(f"  Status:        ⚠️ Still below 0.54 target")
else:
    print("  ❌ Results not found")

# === Comparison Summary ===
print("\n" + "=" * 70)
print("📊 HEAD-TO-HEAD COMPARISON")
print("=" * 70)

# Collect all AUCs
aucs = {}
if 'A_FeatureSelection' in results:
    aucs['V5_25feat'] = results['A_FeatureSelection']['auc']
if 'C_AdvancedFeatures' in results:
    aucs['V6_30feat'] = results['C_AdvancedFeatures']['auc']

# Baselines
print(f"\n  V3 Baseline (18 features):   AUC = 0.5141")
print(f"  V5 All (44 features):        AUC = 0.5266")

for name, auc in aucs.items():
    status = ""
    if auc > 0.54:
        status = "✅🎯 PROFITABLE!"
    elif auc > 0.5266:
        status = "✅ Better than V5"
    elif auc > 0.5141:
        status = "⚠️ Better than V3"
    else:
        status = "❌ Below baseline"
    print(f"  {name:<27}  AUC = {auc:.4f}  {status}")

# === Recommendation ===
print("\n" + "=" * 70)
print("🎯 FINAL RECOMMENDATION")
print("=" * 70)

if aucs:
    best_model = max(aucs.items(), key=lambda x: x[1])
    best_auc = best_model[1]
    
    if best_auc > 0.54:
        print(f"\n✅🚀 DEPLOY: {best_model[0]}")
        print(f"   AUC = {best_auc:.4f} exceeds 0.54 threshold")
        print(f"   This model is ready for live paper trading!")
        
        if 'B_PaperTrading' in results and results['B_PaperTrading'].get('win_rate', 0) > 0.54:
            print(f"\n📋 Trading Configuration:")
            print(f"   Threshold:    {results['B_PaperTrading'].get('best_threshold')}")
            print(f"   Capital:      ₹25,000")
            print(f"   Risk/Trade:   5%")
            print(f"   Expected Win:  {results['B_PaperTrading'].get('win_rate'):.1%}")
            
    elif best_auc > 0.53:
        print(f"\n⚠️ {best_model[0]} shows promise (AUC = {best_auc:.4f})")
        print(f"   But below 0.54 profitability threshold.")
        print(f"\n📋 Next Steps:")
        print(f"   1. Try paper trading with threshold 0.52-0.55")
        print(f"   2. Add REAL sentiment data (not synthetic)")
        print(f"   3. Add options flow data")
        print(f"   4. Consider ensemble of multiple models")
        
    else:
        print(f"\n❌ Best model ({best_model[0]}) AUC = {best_auc:.4f}")
        print(f"   Below 0.54 profitability threshold.")
        print(f"\n📋 Required Improvements:")
        print(f"   1. Get REAL external data sources:")
        print(f"      - Social media sentiment APIs")
        print(f"      - Options flow data (buy/sell pressure)")
        print(f"      - Institutional order book data")
        print(f"   2. Increase training data size")
        print(f"   3. Try alternative architectures (XGBoost, ensemble)")
        print(f"   4. Add sector rotation features")
        
else:
    print("\n⚠️ No experiment results found.")
    print("   Run experiments first:")
    print("   1. python scripts/train_v5_selected.py")
    print("   2. python scripts/paper_trade_v5_threshold_sweep.py")
    print("   3. python scripts/create_v6_advanced_features.py")
    print("   4. python scripts/train_v6_advanced.py")

# Save final report
with open("results/FINAL_COMPARISON.json", 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'best_model': best_model[0] if aucs else None,
        'best_auc': best_auc if aucs else None,
        'profitable': best_auc > 0.54 if aucs else False,
        'recommendation': 'DEPLOY' if (aucs and best_auc > 0.54) else 'NEED_IMPROVEMENT'
    }, f, indent=2)

print(f"\n✅ Final report saved to: results/FINAL_COMPARISON.json")
print("\n" + "=" * 70)
