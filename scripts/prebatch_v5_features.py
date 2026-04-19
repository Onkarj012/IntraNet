"""
Pre-batch V5 Comprehensive Features - USES FULL 1-MINUTE DATA

This creates 50+ features per sample using raw 1-minute bars.
Much more signal than downsampled 5-minute data.
"""

import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.features.v5_comprehensive import ComprehensiveFeatureEngineerV5

console = Console()


def prebatch_v5_features(data_dir="nifty500", output_dir="cache/prebatched_features_v5", max_stocks=None):
    """Pre-compute v5 comprehensive features using FULL 1-minute data."""
    console.print("[bold blue]🚀 PRE-BATCHING V5 COMPREHENSIVE FEATURES[/bold blue]")
    console.print("[yellow]Using FULL 1-minute data (not downsampled)[/yellow]")
    console.print("[yellow]Computing 50+ features per sample...[/yellow]\n")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all stocks
    all_files = sorted(list(data_path.glob("*_minute.csv")))
    if max_stocks:
        all_files = all_files[:max_stocks]
    
    total_stocks = len(all_files)
    console.print(f"[green]📊 Found {total_stocks} stocks to process[/green]")
    
    feature_engineer = ComprehensiveFeatureEngineerV5()
    
    processed = 0
    errors = 0
    total_samples = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("Processing stocks...", total=total_stocks)
        
        for i, csv_file in enumerate(all_files):
            symbol = csv_file.stem.replace("_minute", "")
            
            # Check if already done
            batch_file = output_path / f"{symbol}_features_v5.pkl"
            if batch_file.exists():
                progress.advance(task)
                processed += 1
                continue
            
            try:
                # Load FULL 1-minute data (no downsampling!)
                df = pd.read_csv(csv_file, parse_dates=['date'],
                                usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
                df = df.set_index('date')
                df.columns = df.columns.str.lower()
                
                # Clean data
                df = df.replace([np.inf, -np.inf], np.nan)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].ffill().bfill()
                
                if len(df) < 1000:  # Need substantial data
                    progress.advance(task)
                    continue
                
                # Compute v5 features (50+ features!)
                features = feature_engineer.compute_all_features_v5(minute_df=df, symbol=symbol)
                
                # Create training samples (use 1-min resolution)
                samples = []
                pred_horizon = 15  # 15 minutes ahead
                feat_window = 30   # 30 minutes of history
                step = 30          # Sample every 30 minutes
                
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
                    
                    # Calculate targets
                    future_price = future['close'].iloc[-1]
                    future_return = (future_price - current_price) / current_price
                    
                    y_dir = 1 if future_return > 0 else 0
                    y_mag = abs(future_return)
                    
                    target_hit = future['high'].max() >= current_price * 1.01
                    stop_hit = future['low'].min() <= current_price * 0.995
                    y_conf = 1 if target_hit and not stop_hit else 0
                    
                    # Mean of feature window
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
                    processed += 1
                    total_samples += len(samples)
                    
            except Exception as e:
                errors += 1
                pass
            
            progress.advance(task)
    
    console.print(f"\n[bold green]✅ V5 PRE-BATCHING COMPLETE[/bold green]")
    console.print(f"[white]Processed: [green]{processed}[/green] stocks[/white]")
    console.print(f"[white]Total samples: [blue]{total_samples:,}[/blue] (using 1-min data)[/white]")
    console.print(f"[white]Output: [cyan]{output_path}[/cyan][/white]")
    console.print(f"\n[bold yellow]⚡ Ready for training with 50+ features per sample![/bold yellow]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--output-dir", default="cache/prebatched_features_v5")
    
    args = parser.parse_args()
    
    prebatch_v5_features(max_stocks=args.max_stocks, output_dir=args.output_dir)
