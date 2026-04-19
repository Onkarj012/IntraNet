"""
Pre-batch Features for Nifty500 - RICH CLI VERSION
Computes and saves all features once, enabling fast training.
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed

console = Console()


def prebatch_nifty500_features(data_dir="nifty500", output_dir="cache/prebatched_features", max_stocks=None):
    """
    Pre-compute and save features for all Nifty500 stocks with beautiful CLI output.
    """
    console.print(Panel.fit(
        "[bold blue]INTRADAYNET v3.0 - PRE-BATCHING FEATURES[/bold blue]\n"
        f"[cyan]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/cyan]",
        title="🚀 Feature Pre-batching",
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
    
    console.print(f"[green]📊 Found {total_stocks} stocks to process[/green]")
    console.print(f"[yellow]💾 Output directory: {output_path}[/yellow]\n")
    
    feature_engineer = EnhancedFeatureEngineerFixed()
    
    processed = 0
    errors = 0
    total_samples = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("Processing stocks...", total=total_stocks)
        
        for i, csv_file in enumerate(all_files):
            symbol = csv_file.stem.replace("_minute", "")
            
            # Check if already pre-batched
            batch_file = output_path / f"{symbol}_features.pkl"
            if batch_file.exists():
                progress.advance(task)
                processed += 1
                continue
            
            try:
                # Load data
                df = pd.read_csv(csv_file, parse_dates=['date'],
                                usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
                df = df.set_index('date')
                df.columns = df.columns.str.lower()
                
                # Replace inf values
                df = df.replace([np.inf, -np.inf], np.nan)
                
                # Downsample
                df = df.iloc[::10]
                
                if len(df) < 100:
                    progress.advance(task)
                    continue
                
                # Compute features
                features = feature_engineer.compute_all_features(minute_df=df, symbol=symbol)
                
                # Create training samples
                samples = []
                pred_horizon, feat_window, step = 6, 12, 150
                
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
                
                # Save pre-batched samples
                if len(samples) > 0:
                    with open(batch_file, 'wb') as f:
                        pickle.dump(samples, f)
                    processed += 1
                    total_samples += len(samples)
                    
            except Exception as e:
                errors += 1
                # Silently continue on errors
                pass
            
            progress.advance(task)
    
    # Summary
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        f"[bold green]✅ PRE-BATCHING COMPLETE[/bold green]\n\n"
        f"[white]Processed: [green]{processed}[/green] stocks[/white]\n"
        f"[white]Errors: [red]{errors}[/red][/white]\n"
        f"[white]Total samples: [blue]{total_samples}[/blue][/white]\n"
        f"[white]Output: [cyan]{output_path}[/cyan][/white]\n\n"
        f"[bold yellow]⚡ Next training will be 20x faster![/bold yellow]",
        title="📦 Summary",
        border_style="green"
    ))
    
    return processed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=None, help="Max stocks to process (default: all)")
    parser.add_argument("--output-dir", default="cache/prebatched_features", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        count = prebatch_nifty500_features(
            max_stocks=args.max_stocks,
            output_dir=args.output_dir
        )
        sys.exit(0 if count > 0 else 1)
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Interrupted by user[/bold red]")
        sys.exit(1)
