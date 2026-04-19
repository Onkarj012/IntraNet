"""
Pre-batch V4 Enhanced Features
Computes all 26 features (18 v3 + 8 new microstructure).
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
from intradaynet.features.v4_enhanced import EnhancedFeatureEngineerV4

console = Console()


def prebatch_v4_features(data_dir="nifty500", output_dir="cache/prebatched_features_v4", max_stocks=None):
    """Pre-compute v4 enhanced features."""
    console.print("[bold blue]🚀 PRE-BATCHING V4 ENHANCED FEATURES[/bold blue]\n")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_files = sorted(list(data_path.glob("*_minute.csv")))
    if max_stocks:
        all_files = all_files[:max_stocks]
    
    total_stocks = len(all_files)
    console.print(f"[green]📊 Found {total_stocks} stocks to process[/green]")
    console.print(f"[yellow]Computing 26 features per stock (18 v3 + 8 new)[/yellow]\n")
    
    feature_engineer = EnhancedFeatureEngineerV4()
    
    processed = 0
    errors = 0
    total_samples = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("Processing stocks...", total=total_stocks)
        
        for i, csv_file in enumerate(all_files):
            symbol = csv_file.stem.replace("_minute", "")
            
            # Check if already done
            batch_file = output_path / f"{symbol}_features_v4.pkl"
            if batch_file.exists():
                progress.advance(task)
                processed += 1
                continue
            
            try:
                df = pd.read_csv(csv_file, parse_dates=['date'],
                                usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
                df = df.set_index('date')
                df.columns = df.columns.str.lower()
                df = df.replace([np.inf, -np.inf], np.nan)
                
                # Downsample
                df = df.iloc[::10]
                
                if len(df) < 100:
                    progress.advance(task)
                    continue
                
                # Compute V4 features (26 total)
                features = feature_engineer.compute_all_features_v4(minute_df=df, symbol=symbol)
                
                # Create samples
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
                    
                    # Targets
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
                    processed += 1
                    total_samples += len(samples)
                    
            except Exception as e:
                errors += 1
                pass
            
            progress.advance(task)
    
    console.print(f"\n[bold green]✅ COMPLETE[/bold green]")
    console.print(f"Processed: {processed} stocks")
    console.print(f"Total samples: {total_samples:,}")
    console.print(f"Output: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--output-dir", default="cache/prebatched_features_v4")
    
    args = parser.parse_args()
    
    prebatch_v4_features(max_stocks=args.max_stocks, output_dir=args.output_dir)
