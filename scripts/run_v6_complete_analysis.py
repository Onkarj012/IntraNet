"""
V6 Complete Analysis Suite - Run All Deep Analysis Scripts
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

SCRIPTS = [
    "scripts/v6_deep_backtest.py",
    "scripts/v6_regime_analysis.py", 
    "scripts/v6_feature_correlation.py",
    "scripts/v6_error_analysis.py"
]

print("=" * 80)
print("🔬 V6 COMPLETE ANALYSIS SUITE")
print("=" * 80)
print(f"\n🚀 Running {len(SCRIPTS)} comprehensive analysis modules...")
print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

results = {}

for i, script in enumerate(SCRIPTS, 1):
    script_name = Path(script).name
    print(f"\n[{i}/{len(SCRIPTS)}] Running {script_name}...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per script
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and "Error" in result.stderr:
            print(f"⚠️  Warnings/Errors:\n{result.stderr}")
        
        results[script_name] = {
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ {script_name} timed out after 10 minutes")
        results[script_name] = {'status': 'TIMEOUT'}
    except Exception as e:
        print(f"❌ {script_name} failed: {e}")
        results[script_name] = {'status': 'ERROR', 'error': str(e)}

# Summary
print("\n" + "=" * 80)
print("📊 ANALYSIS SUITE SUMMARY")
print("=" * 80)

success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
print(f"\n✅ Completed: {success_count}/{len(SCRIPTS)} analyses")

print(f"\n📁 Results saved to: results/v6_deep_analysis/")
print("   - v6_deep_analysis_report.json (comprehensive report)")
print("   - feature_importance.csv")
print("   - trades_threshold_*.csv")
print("   - regime_analysis.json")
print("   - feature_correlations.csv")
print("   - error_analysis.json")

print("\n💡 Next Steps:")
print("   1. Review results/v6_deep_analysis/ for insights")
print("   2. Identify best performing regimes")
print("   3. Consider feature removal for correlated pairs")
print("   4. Address high-confidence error patterns")
print(f"\n⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
