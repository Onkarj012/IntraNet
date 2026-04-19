"""
Analyze Feature Importance for Trained Model
Shows which of the 18 features actually help/hurt performance.
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.models.specialized import SpecializedModelSuite
from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed

console = Console()

def analyze_feature_importance():
    """Analyze which features are actually helping."""
    console.print(Panel.fit(
        "[bold blue]FEATURE IMPORTANCE ANALYSIS[/bold blue]\n"
        f"[cyan]Analyzing 18 v3.0 features...[/cyan]",
        title="🔍 Analysis",
        border_style="blue"
    ))
    
    # Load model
    model_dir = Path("results/models/v3_nifty500_fast")
    if not model_dir.exists():
        console.print("[red]❌ Model not found![/red]")
        return
    
    console.print(f"[green]✓ Loading model from {model_dir}...[/green]")
    
    config = type('Config', (), {})()
    models = SpecializedModelSuite(config)
    models.load(str(model_dir))
    
    # Load pre-batched features
    PREBATCH_DIR = Path("cache/prebatched_features")
    if not PREBATCH_DIR.exists():
        console.print("[red]❌ Pre-batched features not found![/red]")
        return
    
    # Load subset of samples for analysis
    batch_files = list(PREBATCH_DIR.glob("*_features.pkl"))[:50]  # Sample 50 stocks
    
    console.print(f"[yellow]Loading {len(batch_files)} stock samples for analysis...[/yellow]")
    
    all_samples = []
    for batch_file in batch_files[:20]:  # Use 20 stocks for speed
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
        test_idx = np.arange(len(X))[-500:]  # Use last 500 samples
    
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
    
    # Compute permutation importance
    console.print("[bold yellow]Computing feature importance (this may take 1-2 minutes)...[/bold yellow]")
    
    with console.status("[bold green]Analyzing feature contributions..."):
        perm_importance = permutation_importance(
            models.direction_model.model,
            X_test,
            y_test,
            n_repeats=3,
            random_state=42,
            n_jobs=-1
        )
    
    # Sort by importance
    importance_scores = perm_importance.importances_mean
    importance_std = perm_importance.importances_std
    
    sorted_idx = np.argsort(importance_scores)[::-1]
    
    # Create results table
    table = Table(
        title="🎯 Feature Importance Analysis (Top 18)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", style="green")
    table.add_column("Std Dev", style="yellow")
    table.add_column("Status", style="bold")
    
    # Add rows
    for rank, idx in enumerate(sorted_idx[:18], 1):
        score = importance_scores[idx]
        std = importance_std[idx]
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        
        # Status
        if score > 0.005:
            status = "✅ Strong"
            status_color = "green"
        elif score > 0.001:
            status = "🟡 Weak"
            status_color = "yellow"
        else:
            status = "❌ Noise"
            status_color = "red"
        
        table.add_row(
            str(rank),
            name,
            f"{score:.5f}",
            f"±{std:.5f}",
            f"[{status_color}]{status}[/{status_color}]"
        )
    
    console.print(table)
    console.print()
    
    # Summary
    strong_features = sum(1 for score in importance_scores if score > 0.005)
    weak_features = sum(1 for score in importance_scores if 0.001 < score <= 0.005)
    noise_features = sum(1 for score in importance_scores if score <= 0.001)
    
    summary_table = Table(title="📊 Summary", box=box.ROUNDED)
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Count", style="green")
    summary_table.add_column("Action", style="yellow")
    
    summary_table.add_row("Strong Features", str(strong_features), "Keep & enhance")
    summary_table.add_row("Weak Features", str(weak_features), "Improve or replace")
    summary_table.add_row("Noise Features", str(noise_features), "Remove")
    
    console.print(summary_table)
    console.print()
    
    # Save analysis
    results_dir = Path("results/training/feature_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'model': str(model_dir),
        'samples_analyzed': len(X_test),
        'feature_importance': [
            {
                'rank': rank,
                'feature': feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                'importance': float(importance_scores[idx]),
                'std': float(importance_std[idx]),
                'status': 'strong' if importance_scores[idx] > 0.005 else ('weak' if importance_scores[idx] > 0.001 else 'noise')
            }
            for rank, idx in enumerate(sorted_idx, 1)
        ],
        'summary': {
            'strong': strong_features,
            'weak': weak_features,
            'noise': noise_features
        }
    }
    
    with open(results_dir / "feature_importance_analysis.json", 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    console.print(f"[green]✅ Analysis saved to {results_dir}/feature_importance_analysis.json[/green]\n")
    
    # Recommendations
    console.print(Panel.fit(
        "[bold cyan]🎯 RECOMMENDATIONS[/bold cyan]\n\n"
        "[white]1. [green]Keep strong features[/green] and add more like them[/white]\n"
        "[white]2. [yellow]Improve weak features[/yellow] with better calculations[/white]\n"
        "[white]3. [red]Remove noise features[/red] - they hurt performance[/white]\n"
        "[white]4. Add [bold]microstructure features[/bold] (order flow, tick data)[/white]",
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
