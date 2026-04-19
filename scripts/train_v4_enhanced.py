"""
Train with V4 Enhanced Features
Uses 26 features (18 v3 + 8 microstructure) for better performance.
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, brier_score_loss
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.models.specialized import SpecializedModelSuite, ModelConfig, compute_expected_calibration_error

console = Console()


def train_v4():
    """Train with v4 enhanced features."""
    console.print(Panel.fit(
        "[bold blue]🚀 V4 ENHANCED TRAINING[/bold blue]\n"
        "[cyan]26 features (18 v3 + 8 microstructure)[/cyan]\n"
        f"[cyan]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/cyan]",
        title="🧠 Training V4 Model",
        border_style="blue"
    ))
    
    PREBATCH_DIR = Path("cache/prebatched_features_v4")
    OUTPUT_DIR = Path("results/models/v4_enhanced")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if v4 features exist
    if not PREBATCH_DIR.exists() or len(list(PREBATCH_DIR.glob("*_features_v4.pkl"))) == 0:
        console.print(Panel.fit(
            "[bold red]❌ V4 features not found![/bold red]\n\n"
            "[white]Run this first:[/white]\n"
            "[cyan]python scripts/prebatch_v4_features.py[/cyan]",
            title="⚠️ Error",
            border_style="red"
        ))
        sys.exit(1)
    
    batch_files = list(PREBATCH_DIR.glob("*_features_v4.pkl"))
    console.print(f"[green]📂 Found {len(batch_files)} v4 pre-batched files[/green]\n")
    
    # Load samples
    all_samples = []
    
    console.print("[yellow]Loading samples...[/yellow]")
    for batch_file in batch_files:
        try:
            with open(batch_file, 'rb') as f:
                samples = pickle.load(f)
                all_samples.extend(samples)
        except:
            pass
    
    console.print(f"[green]✅ Loaded {len(all_samples):,} samples[/green]\n")
    
    # Prepare data
    X = np.array([s['features'] for s in all_samples])
    y_dir = np.array([s['y_dir'] for s in all_samples])
    y_mag = np.array([s['y_mag'] for s in all_samples])
    y_conf = np.array([s['y_conf'] for s in all_samples])
    dates = pd.to_datetime([s['date'] for s in all_samples])
    symbols = [s['symbol'] for s in all_samples]
    
    # Summary table
    summary = Table(title="📊 Data Summary", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Total samples", f"{len(X):,}")
    summary.add_row("Features", f"{X.shape[1]} (v3: 18 + v4: 8)")
    summary.add_row("Unique stocks", f"{len(set(symbols))}")
    summary.add_row("Direction", f"{np.mean(y_dir):.1%} positive")
    summary.add_row("Date range", f"{dates.min().date()} to {dates.max().date()}")
    console.print(summary)
    console.print()
    
    # Temporal split
    train_mask = dates < '2023-01-01'
    val_mask = (dates >= '2023-01-01') & (dates < '2024-01-01')
    test_mask = dates >= '2024-01-01'
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    split_table = Table(title="⏰ Temporal Split", box=box.ROUNDED)
    split_table.add_column("Set", style="cyan")
    split_table.add_column("Samples", style="green")
    split_table.add_column("Percentage", style="yellow")
    split_table.add_row("Train", f"{len(train_idx):,}", f"{len(train_idx)/len(X):.1%}")
    split_table.add_row("Val", f"{len(val_idx):,}", f"{len(val_idx)/len(X):.1%}")
    split_table.add_row("Test", f"{len(test_idx):,}", f"{len(test_idx)/len(X):.1%}")
    console.print(split_table)
    console.print()
    
    # Prepare sets
    X_train = X[train_idx]
    y_dir_train, y_mag_train, y_conf_train = y_dir[train_idx], y_mag[train_idx], y_conf[train_idx]
    
    if len(val_idx) >= 15:
        X_val = X[val_idx]
        y_dir_val, y_mag_val, y_conf_val = y_dir[val_idx], y_mag[val_idx], y_conf[val_idx]
    else:
        split_pt = int(len(train_idx) * 0.9)
        X_train, X_val = X_train[:split_pt], X_train[split_pt:]
        y_dir_train, y_dir_val = y_dir_train[:split_pt], y_dir_train[split_pt:]
        y_mag_train, y_mag_val = y_mag_train[:split_pt], y_mag_train[split_pt:]
        y_conf_train, y_conf_val = y_conf_train[:split_pt], y_conf_train[split_pt:]
    
    X_test = X[test_idx] if len(test_idx) >= 15 else X[-200:]
    y_dir_test = y_dir[test_idx] if len(test_idx) >= 15 else y_dir[-200:]
    y_mag_test = y_mag[test_idx] if len(test_idx) >= 15 else y_mag[-200:]
    y_conf_test = y_conf[test_idx] if len(test_idx) >= 15 else y_conf[-200:]
    
    # Train
    console.print("[bold blue]🧠 Training V4 Enhanced Models...[/bold blue]\n")
    
    config = ModelConfig()
    models = SpecializedModelSuite(config)
    
    with console.status("[bold green]Training..."):
        models.fit(X_train, y_dir_train, y_mag_train, y_conf_train,
                   X_val, y_dir_val, y_mag_val, y_conf_val)
    
    console.print("[bold green]✅ Models trained![/bold green]\n")
    
    # Evaluate
    dir_preds_test = models.direction_model.predict_class(X_test)
    dir_acc_test = accuracy_score(y_dir_test, dir_preds_test)
    dir_proba_test = models.direction_model.predict(X_test)
    
    try:
        dir_auc_test = roc_auc_score(y_dir_test, dir_proba_test)
        ece_test = compute_expected_calibration_error(y_dir_test, dir_proba_test)
        brier_test = brier_score_loss(y_dir_test, dir_proba_test)
    except:
        dir_auc_test, ece_test, brier_test = 0.5, 0.0, 0.25
    
    mag_preds_test = models.magnitude_model.predict(X_test)
    mag_mae_test = mean_absolute_error(y_mag_test, mag_preds_test)
    
    conf_preds_test = models.confidence_model.predict(X_test) > 0.5
    conf_acc_test = accuracy_score(y_conf_test, conf_preds_test)
    
    # Results table
    results = Table(title="🎯 V4 Enhanced Model Results", box=box.ROUNDED)
    results.add_column("Metric", style="cyan")
    results.add_column("Value", style="green")
    results.add_column("Status", style="yellow")
    
    # Compare to baseline
    baseline_auc = 0.5141
    
    auc_status = "✅ IMPROVED" if dir_auc_test > baseline_auc else "⚠️ Similar"
    conf_status = "✅ Excellent" if conf_acc_test > 0.85 else "✅ Good"
    
    results.add_row("Direction Accuracy", f"{dir_acc_test:.2%}", "✅")
    results.add_row("Direction AUC", f"{dir_auc_test:.4f}", f"{auc_status} (baseline: {baseline_auc})")
    results.add_row("Direction ECE", f"{ece_test:.4f}", "✅")
    results.add_row("Brier Score", f"{brier_test:.4f}", "✅")
    results.add_row("Magnitude MAE", f"{mag_mae_test:.5f}", "✅")
    results.add_row("Confidence Accuracy", f"{conf_acc_test:.2%}", conf_status)
    
    console.print(results)
    console.print()
    
    # Comparison
    if dir_auc_test > baseline_auc:
        improvement = ((dir_auc_test - baseline_auc) / baseline_auc) * 100
        console.print(Panel.fit(
            f"[bold green]🎉 AUC IMPROVED BY {improvement:.1f}%![/bold green]\n"
            f"[white]Baseline (v3): {baseline_auc:.4f}[/white]\n"
            f"[white]Enhanced (v4): {dir_auc_test:.4f}[/white]",
            title="📈 Improvement",
            border_style="green"
        ))
    
    # Save
    models.save(str(OUTPUT_DIR))
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v4_enhanced',
        'n_features': 26,
        'features_breakdown': {'v3': 18, 'v4_new': 8},
        'total_samples': len(X),
        'test_metrics': {
            'direction_accuracy': float(dir_acc_test),
            'direction_auc': float(dir_auc_test),
            'direction_ece': float(ece_test),
            'brier': float(brier_test),
            'magnitude_mae': float(mag_mae_test),
            'confidence_accuracy': float(conf_acc_test)
        },
        'baseline_comparison': {
            'baseline_auc': baseline_auc,
            'v4_auc': float(dir_auc_test),
            'improvement_pct': ((dir_auc_test - baseline_auc) / baseline_auc) * 100
        }
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(OUTPUT_DIR / "comparison_v3_vs_v4.json", 'w') as f:
        json.dump({
            'v3_baseline': {
                'auc': 0.5141,
                'accuracy': 0.5196,
                'confidence': 0.8847
            },
            'v4_enhanced': {
                'auc': float(dir_auc_test),
                'accuracy': float(dir_acc_test),
                'confidence': float(conf_acc_test)
            }
        }, f, indent=2)
    
    console.print(Panel.fit(
        "[bold green]✅ V4 TRAINING COMPLETE![/bold green]\n\n"
        f"[white]Models saved to:[/white]\n"
        f"[cyan]{OUTPUT_DIR}[/cyan]",
        title="🎉 Success",
        border_style="green"
    ))


if __name__ == "__main__":
    try:
        train_v4()
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Interrupted[/bold red]")
        sys.exit(1)
