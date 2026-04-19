"""
Pre-batch Features for Nifty500
Computes and saves all features once, enabling fast training.
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prebatch_features")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed


def prebatch_nifty500_features(data_dir="nifty500", output_dir="cache/prebatched_features", max_stocks=None):
    """
    Pre-compute and save features for all Nifty500 stocks.
    This is done ONCE, then training is 20x faster.
    """
    print("="*70)
    print("PRE-BATCHING NIFTY500 FEATURES")
    print("="*70)
    print(f"\nThis will compute features for all stocks and save them.")
    print(f"Subsequent trainings will load from disk (20x faster).")
    print()
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all stocks
    all_files = sorted(list(data_path.glob("*_minute.csv")))
    if max_stocks:
        all_files = all_files[:max_stocks]
    
    total_stocks = len(all_files)
    print(f"Found {total_stocks} stocks to process")
    print()
    
    feature_engineer = EnhancedFeatureEngineerFixed()
    
    processed = 0
    errors = 0
    
    for i, csv_file in enumerate(all_files):
        symbol = csv_file.stem.replace("_minute", "")
        
        if i % 50 == 0:
            print(f"[{i+1}/{total_stocks}] Processing {symbol}...")
        
        # Check if already pre-batched
        batch_file = output_path / f"{symbol}_features.pkl"
        if batch_file.exists():
            processed += 1
            continue
        
        try:
            # Load data
            df = pd.read_csv(csv_file, parse_dates=['date'],
                            usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
            df = df.set_index('date')
            df.columns = df.columns.str.lower()
            
            # Downsample
            df = df.iloc[::10]
            
            if len(df) < 100:
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
                
        except Exception as e:
            errors += 1
            logger.debug(f"Error with {symbol}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("PRE-BATCHING COMPLETE")
    print(f"{'='*70}")
    print(f"Processed: {processed} stocks")
    print(f"Errors: {errors}")
    print(f"Output directory: {output_path}")
    print()
    print(f"Next training will load from disk (20x faster!)")
    print(f"{'='*70}")
    
    return processed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--output-dir", default="cache/prebatched_features")
    
    args = parser.parse_args()
    
    count = prebatch_nifty500_features(
        max_stocks=args.max_stocks,
        output_dir=args.output_dir
    )
    
    sys.exit(0 if count > 0 else 1)
