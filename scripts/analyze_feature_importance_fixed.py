"""
Analyze Feature Importance for Trained Model - FIXED VERSION
Uses LightGBM's built-in feature importance instead of sklearn's permutation_importance.
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
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed

console = Console()


def analyze_feature_importance():
    """Analyze which features are actually helping - using LightGBM's native importance."""
    console.print(Panel.fit(
        "[bold blue]FEATURE IMPORTANCE ANALYSIS[/bold blue]\n"
        f"[cyan]Analyzing 18 v3.0 features using LightGBM native importance...[/cyan]",
        title="🔍 Analysis",
        border_style="blue"
    ))
    
    # Load model
    model_dir = Path("results/models/v3_nifty500_fast")
    if not model_dir.exists():
        console.print("[red]❌ Model not found![/red]")
        return
    
    console.print(f"[green]✓ Loading model from {model_dir}...[/green]")
    
    # Load LightGBM models directly
    try:
        direction_model = lgb.Booster(model_file=str(model_dir / "direction_model.lgb"))
        console.print("[green]✓ Direction model loaded[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error loading model: {e}[/red]")
        return
    
    # Load pre-batched features
    PREBATCH_DIR = Path("cache/prebatched_features")
    if not PREBATCH_DIR.exists():
        console.print("[red]❌ Pre-batched features not found![/red]")
        return
    
    # Load subset of samples
    batch_files = list(PREBATCH_DIR.glob("*_features.pkl"))[:30]
    
    console.print(f"[yellow]Loading {len(batch_files)} stock samples...[/yellow]")
    
    all_samples = []
    for batch_file in batch_files[:15]:
        try:
            with open(batch_file, 'rb') as f:
                samples = pickle.load(f)
                all_samples.extend(samples)
        except:
            pass
    
    if len(all_samples) == 0:
        console.print("[red]❌ No samples loaded![/red]")
        return
    
    # Prepare data
    X = np.array([s['features'] for s in all_samples])
    y_dir = np.array([s['y_dir'] for s in all_samples])
    dates = pd.to_datetime([s['date'] for s in all_samples])
    
    # Use test set only
    test_mask = dates >= '2024-01-01'
    test_idx = np.where(test_mask)[0]
    
    if len(test_idx) < 100:
        test_idx = np.arange(len(X))[-300:]
    
    X_test = X[test_idx]
    y_test = y_dir[test_idx]
    
    console.print(f"[green]✓ Using {len(X_test)} test samples[/green]\n")
    
    # Feature names
    feature_names = [
        'relative_volume_15m', 'price_acceleration', 'tick_imbalance',
        'bar_entropy', 'volume_price_correlation', 'consecutive_direction',
        'sector_momentum_rank', 'sector_flow_score', 'relative_strength_vs_nifty',
        'correlation_to_nifty_20d', 'vix_percentile_60d', 'realized_vs_implied_vol',
        'overnight_gap_zscore', 'intraday_range_percentile', 'pcr_change',
        'max_pain_distance', 'iv_skew', 'oi_buildup_signal'
    ]
    
    # Get LightGBM feature importance
    console.print("[bold yellow]Computing LightGBM feature importance...[/bold yellow]")
    
    importance_gain = direction_model.feature_importance(importance_type='gain')
    importance_split = direction_model.feature_importance(importance_type='split')
    
    # Normalize
    importance_gain = importance_gain / (importance_gain.sum() + 1e-10)
    importance_split = importance_split / (importance_split.sum() + 1e-10)
    
    # Sort by gain importance
    sorted_idx = np.argsort(importance_gain)[::-1]
    
    # Create results table
    table = Table(
        title="🎯 Feature Importance Analysis (by Gain)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Feature", style="cyan")
    table.add_column("Gain", style="green")
    table.add_column("Split", style="yellow")
    table.add_column("Status", style="bold")
    
    # Add rows
    for rank, idx in enumerate(sorted_idx, 1):
        gain_score = importance_gain[idx]
        split_score = importance_split[idx]
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        
        # Status based on gain importance
        if gain_score > 0.08:
            status = "✅ Strong"
            status_color = "green"
        elif gain_score > 0.04:
            status = "🟡 Moderate"
            status_color = "yellow"
        else:
            status = "❌ Weak"
            status_color = "red"
        
        table.add_row(
            str(rank),
            name,
            f"{gain_score:.3f}",
            f"{split_score:.3f}",
            f"[{status_color}]{status}[/{status_color}]"
        )
    
    console.print(table)
    console.print()
    
    # Summary
    strong_features = sum(1 for score in importance_gain if score > 0.08)
    moderate_features = sum(1 for score in importance_gain if 0.04 < score <= 0.08)
    weak_features = sum(1 for score in importance_gain if score <= 0.04)
    
    summary_table = Table(title="📊 Summary", box=box.ROUNDED)
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Count", style="green")
    summary_table.add_column("Action", style="yellow")
    
    summary_table.add_row("Strong (Gain > 8%)", str(strong_features), "Keep & enhance")
    summary_table.add_row("Moderate (4-8%)", str(moderate_features), "Refine")
    summary_table.add_row("Weak (< 4%)", str(weak_features), "Replace with v4")
    
    console.print(summary_table)
    console.print()
    
    # Top 5 and bottom 5
    console.print("[bold cyan]Top 5 Most Important:[/bold cyan]")
    for i in range(5):
        idx = sorted_idx[i]
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        console.print(f"  {i+1}. {name}: {importance_gain[idx]:.3f}")
    
    console.print("\n[bold red]Bottom 5 (Consider Replacing):[/bold red]")
    for i in range(5):
        idx = sorted_idx[-(i+1)]
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        console.print(f"  {i+1}. {name}: {importance_gain[idx]:.3f}")
    
    console.print()
    
    # Save analysis
    results_dir = Path("results/training/feature_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'model': str(model_dir),
        'samples_analyzed': len(X_test),
        'method': 'lightgbm_native_gain',
        'feature_importance': [
            {
                'rank': rank,
                'feature': feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                'gain': float(importance_gain[idx]),
                'split': float(importance_split[idx]),
                'status': 'strong' if importance_gain[idx] > 0.08 else ('moderate' if importance_gain[idx] > 0.04 else 'weak')
            }
            for rank, idx in enumerate(sorted_idx, 1)
        ],
        'summary': {
            'strong': strong_features,
            'moderate': moderate_features,
            'weak': weak_features
        },
        'recommendations': {
            'keep_and_enhance': [feature_names[idx] for idx in sorted_idx[:strong_features]],
            'replace': [feature_names[idx] for idx in sorted_idx[-weak_features:]]
        }
    }
    
    with open(results_dir / "feature_importance_analysis.json", 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    console.print(f"[green]✅ Analysis saved to {results_dir}/feature_importance_analysis.json[/green]\n")
    
    # Recommendations
    console.print(Panel.fit(
        "[bold cyan]🎯 RECOMMENDATIONS[/bold cyan]\n\n"
        "[white]1. [green]Keep strong features[/green] - enhance with longer windows[/white]\n"
        "[white]2. [yellow]Replace weak features[/yellow] with v4 microstructure:[/white]\n"
        "   • volume_delta (buy/sell pressure)\n"
        "   • vwap_deviation (institutional level)\n"
        "   • tick_imbalance_v2 (order flow)\n"
        "   • price_momentum_slope (trend strength)\n"
        "[white]3. [green]Add v4 features[/green] to boost AUC from 0.51 → 0.54+[/white]",
        title="💡 Next Steps",
        border_style="green"
    ))


if __name__ == "__main__":
    try:
        analyze_feature_importance()
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Interrupted[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
