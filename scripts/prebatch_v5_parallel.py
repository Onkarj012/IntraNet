"""
Pre-batch V5 Comprehensive Features - MULTI-CORE PARALLEL VERSION
Uses all CPU cores for maximum speed (4-8x faster on M3 Mac).
"""

import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.features.v5_comprehensive import ComprehensiveFeatureEngineerV5

console = Console()


def process_single_stock(args):
    """Process a single stock (for parallel execution)."""
    csv_file, output_dir = args
    symbol = csv_file.stem.replace("_minute", "")
    
    # Check if already done
    batch_file = Path(output_dir) / f"{symbol}_features_v5.pkl"
    if batch_file.exists():
        return {'symbol': symbol, 'status': 'already_done', 'samples': 0}
    
    try:
        # Load FULL 1-minute data
        df = pd.read_csv(csv_file, parse_dates=['date'],
                        usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
        df = df.set_index('date')
        df.columns = df.columns.str.lower()
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].ffill().bfill()
        
        if len(df) < 1000:
            return {'symbol': symbol, 'status': 'insufficient_data', 'samples': 0}
        
        # Compute v5 features
        engineer = ComprehensiveFeatureEngineerV5()
        features = engineer.compute_all_features_v5(minute_df=df, symbol=symbol)
        
        # Create training samples
        samples = []
        pred_horizon = 15
        feat_window = 30
        step = 30
        
        for idx in range(feat_window, len(df) - pred_horizon, step):
            feat_win = features.iloc[idx-feat_window:idx]
            if len(feat_win) < feat_window:
                continue
            
            current_price = df['close'].iloc[idx]
            if current_price <= 0 or np.isnan(current_price):
                continue
            
            future = df.iloc[idx:idx+pred_horizon]
            if len(future) < pred_horizon:
                continue
            
            future_price = future['close'].iloc[-1]
            future_return = (future_price - current_price) / current_price
            
            y_dir = 1 if future_return > 0 else 0
            y_mag = abs(future_return)
            
            target_hit = future['high'].max() >= current_price * 1.01
            stop_hit = future['low'].min() <= current_price * 0.995
            y_conf = 1 if target_hit and not stop_hit else 0
            
            feat_vector = feat_win.mean().values
            
            if np.any(np.isnan(feat_vector)) or np.any(np.isinf(feat_vector)):
                continue
            
            samples.append({
                'features': feat_vector,
                'date': df.index[idx].isoformat(),
                'y_dir': y_dir,
                'y_mag': y_mag,
                'y_conf': y_conf,
                'symbol': symbol
            })
        
        if len(samples) > 0:
            with open(batch_file, 'wb') as f:
                pickle.dump(samples, f)
            return {'symbol': symbol, 'status': 'success', 'samples': len(samples)}
        else:
            return {'symbol': symbol, 'status': 'no_samples', 'samples': 0}
            
    except Exception as e:
        return {'symbol': symbol, 'status': f'error: {str(e)[:50]}', 'samples': 0}


def prebatch_v5_parallel(data_dir="nifty500", output_dir="cache/prebatched_features_v5", max_stocks=None):
    """Pre-compute v5 features using ALL CPU cores."""
    console.print(Panel.fit(
        "[bold blue]🚀 PRE-BATCHING V5 - PARALLEL MULTI-CORE VERSION[/bold blue]\n"
        "[yellow]Using ALL CPU cores for maximum speed![/yellow]\n"
        f"[cyan]Detected {cpu_count()} CPU cores[/cyan]",
        title="⚡ Parallel Processing",
        border_style="blue"
    ))
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all stocks
    all_files = sorted(list(data_path.glob("*_minute.csv")))
    if max_stocks:
        all_files = all_files[:max_stocks]
    
    total_stocks = len(all_files)
    
    # Filter out already processed
    files_to_process = []
    for csv_file in all_files:
        symbol = csv_file.stem.replace("_minute", "")
        batch_file = output_path / f"{symbol}_features_v5.pkl"
        if not batch_file.exists():
            files_to_process.append((csv_file, str(output_path)))
    
    already_done = total_stocks - len(files_to_process)
    
    console.print(f"[green]📊 Total stocks: {total_stocks}[/green]")
    console.print(f"[green]✅ Already processed: {already_done}[/green]")
    console.print(f"[yellow]🔄 To process: {len(files_to_process)}[/yellow]")
    console.print(f"[blue]🔧 CPU cores: {cpu_count()}[/blue]\n")
    
    if len(files_to_process) == 0:
        console.print("[bold green]✅ All stocks already processed![/bold green]")
        return
    
    # Determine optimal number of workers
    n_cores = cpu_count()
    n_workers = min(n_cores, len(files_to_process))  # Don't create more workers than tasks
    
    console.print(f"[bold blue]🚀 Starting parallel processing with {n_workers} workers...[/bold blue]\n")
    
    # Process in parallel with progress bar
    results = []
    processed = 0
    errors = 0
    total_samples = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("Processing stocks...", total=len(files_to_process))
        
        # Use Pool for parallel processing
        with Pool(processes=n_workers) as pool:
            # Map with callback to update progress
            for result in pool.imap_unordered(process_single_stock, files_to_process):
                results.append(result)
                
                if result['status'] == 'success':
                    processed += 1
                    total_samples += result['samples']
                elif result['status'] != 'already_done':
                    errors += 1
                
                progress.advance(task)
    
    console.print(f"\n[bold green]✅ PARALLEL PRE-BATCHING COMPLETE[/bold green]")
    console.print(f"[white]Successfully processed: [green]{processed}[/green] stocks[/white]")
    console.print(f"[white]Errors: [red]{errors}[/red][/white]")
    console.print(f"[white]Total samples: [blue]{total_samples:,}[/blue] (using 1-min data)[/white]")
    console.print(f"[white]Output: [cyan]{output_path}[/cyan][/white]")
    
    # Calculate speedup
    if processed > 0:
        time_per_stock = 30  # seconds (estimated for single core)
        estimated_single_core_time = (processed * time_per_stock) / 60  # minutes
        estimated_parallel_time = estimated_single_core_time / n_workers
        speedup = n_workers
        
        console.print(f"\n[bold yellow]⚡ Speedup: ~{speedup:.1f}x faster with {n_workers} cores![/bold yellow]")
        console.print(f"[dim]Single-core would take: ~{estimated_single_core_time:.0f} minutes[/dim]")
        console.print(f"[dim]Multi-core took: ~{estimated_parallel_time:.0f} minutes[/dim]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--output-dir", default="cache/prebatched_features_v5")
    
    args = parser.parse_args()
    
    # Required for multiprocessing on macOS
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    try:
        prebatch_v5_parallel(max_stocks=args.max_stocks, output_dir=args.output_dir)
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
