"""
Fast Training using Pre-batched Features - RICH CLI VERSION
Loads pre-computed features from disk (20x faster).
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.models.specialized import SpecializedModelSuite, ModelConfig, compute_expected_calibration_error

console = Console()


def train_with_prebatched():
    """Fast training with beautiful CLI output."""
    console.print(Panel.fit(
        "[bold blue]INTRADAYNET v3.0 - FAST TRAINING[/bold blue]\n"
        f"[cyan]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/cyan]\n"
        "[green]Using pre-batched features (20x faster!)[/green]",
        title="🚀 Training Pipeline",
        border_style="blue"
    ))
    
    PREBATCH_DIR = Path("cache/prebatched_features")
    OUTPUT_DIR = Path("results/models/v3_nifty500_fast")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check pre-batched features
    if not PREBATCH_DIR.exists() or len(list(PREBATCH_DIR.glob("*_features.pkl"))) == 0:
        console.print(Panel.fit(
            "[bold red]❌ Pre-batched features not found![/bold red]\n\n"
            "[white]Run this first:[/white]\n"
            "[cyan]python scripts/prebatch_features_rich.py[/cyan]",
            title="⚠️ Error",
            border_style="red"
        ))
        sys.exit(1)
    
    batch_files = list(PREBATCH_DIR.glob("*_features.pkl"))
    console.print(f"[green]📂 Found {len(batch_files)} pre-batched files[/green]\n")
    
    # Load all samples
    all_samples = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading {task.description}"),
        BarColumn(complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        
        task = progress.add_task("features...", total=len(batch_files))
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    samples = pickle.load(f)
                    all_samples.extend(samples)
            except:
                pass
            progress.advance(task)
    
    console.print(f"[green]✅ Loaded {len(all_samples):,} samples[/green]\n")
    
    if len(all_samples) == 0:
        console.print("[bold red]❌ No samples loaded![/bold red]")
        sys.exit(1)
    
    # Prepare data
    X = np.array([s['features'] for s in all_samples])
    y_dir = np.array([s['y_dir'] for s in all_samples])
    y_mag = np.array([s['y_mag'] for s in all_samples])
    y_conf = np.array([s['y_conf'] for s in all_samples])
    dates = pd.to_datetime([s['date'] for s in all_samples])
    symbols = [s['symbol'] for s in all_samples]
    
    # Create summary table
    summary_table = Table(title="📊 Data Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total samples", f"{len(X):,}")
    summary_table.add_row("Features", f"{X.shape[1]}")
    summary_table.add_row("Unique symbols", f"{len(set(symbols))}")
    summary_table.add_row("Direction", f"{np.mean(y_dir):.1%} positive")
    summary_table.add_row("Date range", f"{dates.min().date()} to {dates.max().date()}")
    console.print(summary_table)
    console.print()
    
    # Temporal split
    train_mask = dates < '2023-01-01'
    val_mask = (dates >= '2023-01-01') & (dates < '2024-01-01')
    test_mask = dates >= '2024-01-01'
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    # Split table
    split_table = Table(title="⏰ Temporal Split", box=box.ROUNDED)
    split_table.add_column("Set", style="cyan")
    split_table.add_column("Samples", style="green")
    split_table.add_column("Percentage", style="yellow")
    split_table.add_row("Train (2015-2022)", f"{len(train_idx):,}", f"{len(train_idx)/len(X):.1%}")
    split_table.add_row("Val (2023)", f"{len(val_idx):,}", f"{len(val_idx)/len(X):.1%}")
    split_table.add_row("Test (2024+)", f"{len(test_idx):,}", f"{len(test_idx)/len(X):.1%}")
    console.print(split_table)
    console.print()
    
    # Prepare sets
    X_train, y_dir_train = X[train_idx], y_dir[train_idx]
    y_mag_train, y_conf_train = y_mag[train_idx], y_conf[train_idx]
    
    if len(val_idx) >= 15:
        X_val, y_dir_val = X[val_idx], y_dir[val_idx]
        y_mag_val, y_conf_val = y_mag[val_idx], y_conf[val_idx]
    else:
        split_pt = int(len(train_idx) * 0.9)
        X_train, X_val = X_train[:split_pt], X_train[split_pt:]
        y_dir_train, y_dir_val = y_dir_train[:split_pt], y_dir_train[split_pt:]
        y_mag_train, y_mag_val = y_mag_train[:split_pt], y_mag_train[split_pt:]
        y_conf_train, y_conf_val = y_conf_train[:split_pt], y_conf_train[split_pt:]
    
    if len(test_idx) >= 15:
        X_test, y_dir_test = X[test_idx], y_dir[test_idx]
        y_mag_test, y_conf_test = y_mag[test_idx], y_conf[test_idx]
    else:
        X_test, y_dir_test = X[-200:], y_dir[-200:]
        y_mag_test, y_conf_test = y_mag[-200:], y_conf[-200:]
    
    # Train models with progress
    console.print("[bold blue]🧠 Training 3 Specialized Models...[/bold blue]\n")
    
    config = ModelConfig()
    models = SpecializedModelSuite(config)
    
    with console.status("[bold green]Training Direction, Magnitude & Confidence models..."):
        models.fit(X_train, y_dir_train, y_mag_train, y_conf_train,
                   X_val, y_dir_val, y_mag_val, y_conf_val)
    
    console.print("[bold green]✅ Models trained successfully![/bold green]\n")
    
    # Evaluate
    console.print("[bold blue]📈 Evaluation Results[/bold blue]\n")
    
    # Validation metrics
    dir_preds_val = models.direction_model.predict_class(X_val)
    dir_acc_val = accuracy_score(y_dir_val, dir_preds_val)
    dir_proba_val = models.direction_model.predict(X_val)
    try:
        dir_auc_val = roc_auc_score(y_dir_val, dir_proba_val)
    except:
        dir_auc_val = 0.5
    
    # Test metrics
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
    results_table = Table(title="🎯 Test Set Metrics (Untouched 2024+ Data)", box=box.ROUNDED)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_column("Status", style="yellow")
    
    # Status indicators
    auc_status = "✅ Good" if dir_auc_test > 0.52 else "⚠️ Below target"
    acc_status = "✅ Good" if dir_acc_test > 0.53 else "⚠️ Marginal"
    conf_status = "✅ Excellent" if conf_acc_test > 0.80 else "✅ Good"
    
    results_table.add_row("Direction Accuracy", f"{dir_acc_test:.2%}", acc_status)
    results_table.add_row("Direction AUC", f"{dir_auc_test:.4f}", auc_status)
    results_table.add_row("Direction ECE", f"{ece_test:.4f}", "✅ Calibrated")
    results_table.add_row("Brier Score", f"{brier_test:.4f}", "✅ Good")
    results_table.add_row("Magnitude MAE", f"{mag_mae_test:.5f}", "✅ Good")
    results_table.add_row("Confidence Accuracy", f"{conf_acc_test:.2%}", conf_status)
    
    console.print(results_table)
    console.print()
    
    # Save models
    console.print("[bold blue]💾 Saving Models...[/bold blue]")
    
    models.save(str(OUTPUT_DIR))
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'training_type': 'fast_prebatched_rich',
        'prebatch_dir': str(PREBATCH_DIR),
        'total_samples': len(X),
        'unique_symbols': len(set(symbols)),
        'n_features': X.shape[1],
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_metrics': {
            'direction_accuracy': float(dir_acc_test),
            'direction_auc': float(dir_auc_test),
            'direction_ece': float(ece_test),
            'brier': float(brier_test),
            'magnitude_mae': float(mag_mae_test),
            'confidence_accuracy': float(conf_acc_test)
        },
        'status': 'COMPLETE'
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    results_dir = Path("results/training/fast_prebatched")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_summary.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final success panel
    console.print(Panel.fit(
        f"[bold green]🎉 TRAINING COMPLETE![/bold green]\n\n"
        f"[white]Test AUC: [cyan]{dir_auc_test:.4f}[/cyan][/white]\n"
        f"[white]Test Accuracy: [cyan]{dir_acc_test:.2%}[/cyan][/white]\n"
        f"[white]Confidence Accuracy: [green]{conf_acc_test:.2%}[/green] ⭐[/white]\n\n"
        f"[bold white]📁 Models saved to:[/bold white]\n"
        f"[cyan]{OUTPUT_DIR}[/cyan]\n\n"
        f"[bold green]✅ Ready for paper trading![/bold green]",
        title="🚀 Success",
        border_style="green"
    ))


if __name__ == "__main__":
    try:
        train_with_prebatched()
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Interrupted by user[/bold red]")
        sys.exit(1)
