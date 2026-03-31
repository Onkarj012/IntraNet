#!/usr/bin/env python3
"""
Pre-batch training data into contiguous numpy arrays for fast loading.

Reads pre-computed .npz feature files and builds flat arrays ready for
direct DataLoader consumption. Eliminates per-sample I/O during training.

Usage:
    python scripts/prebatch_training_data.py --config configs/intraday_config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from intradaynet.config import load_config
from intradaynet.features.per_bar_features import PER_BAR_FEATURE_NAMES
from intradaynet.features.session_features import SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-batch training data")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--features-cache", type=str, default="features_cache")
    parser.add_argument("--output-dir", type=str, default="prebatched")
    parser.add_argument("--subset-stocks", type=str, default="")
    parser.add_argument("--max-stocks", type=int, default=0)
    return parser.parse_args()


def build_split(
    symbols, features_cache, date_start, date_end, seq_length, horizons,
    sample_interval, split_name, output_dir, progress, task_id,
):
    """Build pre-batched arrays for one split (train/val/test)."""

    non_eod = [h for h in horizons if h < 375]
    max_non_eod = max(non_eod) if non_eod else 60
    effective_fwd = max(max_non_eod, 60)

    date_start_str = str(date_start.date()) if hasattr(date_start, 'date') else date_start
    date_end_str = str(date_end.date()) if hasattr(date_end, 'date') else date_end

    # First pass: count total samples
    total_samples = 0
    stock_info = []

    for symbol in symbols:
        npz_path = features_cache / f"{symbol}.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path, allow_pickle=True)
        dates = data["per_bar_dates"]
        mask = (dates >= date_start_str) & (dates <= date_end_str)

        if mask.sum() == 0:
            continue

        filtered_dates = dates[mask]
        unique_dates, counts = np.unique(filtered_dates, return_counts=True)

        n_samples = 0
        for ud, n_bars in zip(unique_dates, counts):
            if n_bars < seq_length + effective_fwd:
                continue
            start = seq_length
            end = n_bars - effective_fwd
            if start >= end:
                continue
            n_samples += len(range(start, end, sample_interval))

        if n_samples > 0:
            stock_info.append((symbol, n_samples))
            total_samples += n_samples

    if total_samples == 0:
        console.print(f"  [yellow]⚠ No samples for {split_name}[/yellow]")
        return 0

    # Allocate arrays
    X_per_bar = np.zeros((total_samples, seq_length, len(PER_BAR_FEATURE_NAMES)), dtype=np.float32)
    X_context = np.zeros((total_samples, len(SESSION_FEATURE_NAMES)), dtype=np.float32)
    X_sentiment = np.zeros((total_samples, len(SENTIMENT_FEATURE_NAMES)), dtype=np.float32)
    Y_direction = np.zeros((total_samples, len(horizons)), dtype=np.float32)
    Y_magnitude = np.zeros((total_samples, len(horizons)), dtype=np.float32)
    T_time_norm = np.zeros(total_samples, dtype=np.float32)

    # Second pass: fill arrays
    idx = 0
    for si, (symbol, _) in enumerate(stock_info):
        progress.update(task_id, completed=si)

        npz_path = features_cache / f"{symbol}.npz"
        data = np.load(npz_path, allow_pickle=True)

        per_bar_feats = data["per_bar_features"]
        close = data["close"]
        dates = data["per_bar_dates"]
        sess_feats = data["session_features"]
        sess_dates = data["session_dates"]
        sent_feats = data["sentiment_features"]

        # Filter by date range
        mask = (dates >= date_start_str) & (dates <= date_end_str)
        per_bar_feats = per_bar_feats[mask]
        close = close[mask]
        dates = dates[mask]

        if len(dates) == 0:
            continue

        # Build session lookups
        sess_feat_map = {d: sess_feats[i] for i, d in enumerate(sess_dates) if d >= date_start_str and d <= date_end_str}
        sent_feat_map = {d: sent_feats[i] for i, d in enumerate(sess_dates) if d >= date_start_str and d <= date_end_str}

        # Session slices
        unique_dates = np.unique(dates)
        for ud in unique_dates:
            session_mask = dates == ud
            session_indices = np.where(session_mask)[0]
            n_bars = len(session_indices)
            sess_start = session_indices[0]

            if n_bars < seq_length + effective_fwd:
                continue

            start = seq_length
            end = n_bars - effective_fwd

            if start >= end:
                continue

            for bar_offset in range(start, end, sample_interval):
                # Per-bar features window
                global_start = sess_start + bar_offset - seq_length
                global_end = sess_start + bar_offset
                X_per_bar[idx] = per_bar_feats[global_start:global_end]

                # Context & sentiment
                if ud in sess_feat_map:
                    X_context[idx] = sess_feat_map[ud]
                if ud in sent_feat_map:
                    X_sentiment[idx] = sent_feat_map[ud]

                # Targets
                anchor = close[sess_start + bar_offset]
                for hi, h in enumerate(horizons):
                    tgt_idx = sess_start + bar_offset + h
                    if h >= 375 or tgt_idx >= sess_start + n_bars:
                        tgt_close = close[sess_start + n_bars - 1]
                    else:
                        tgt_close = close[tgt_idx]

                    ret = float((tgt_close - anchor) / max(anchor, 1e-10))
                    Y_direction[idx, hi] = 1.0 if ret > 0 else 0.0
                    Y_magnitude[idx, hi] = ret

                T_time_norm[idx] = bar_offset / max(n_bars, 1)
                idx += 1

    progress.update(task_id, completed=len(stock_info))

    # Trim if needed (edge cases may produce fewer samples)
    if idx < total_samples:
        X_per_bar = X_per_bar[:idx]
        X_context = X_context[:idx]
        X_sentiment = X_sentiment[:idx]
        Y_direction = Y_direction[:idx]
        Y_magnitude = Y_magnitude[:idx]
        T_time_norm = T_time_norm[:idx]

    # Save
    out_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        X_per_bar=X_per_bar, X_context=X_context, X_sentiment=X_sentiment,
        Y_direction=Y_direction, Y_magnitude=Y_magnitude, T_time_norm=T_time_norm,
    )

    return idx


def main():
    args = parse_args()
    cfg = load_config(args.config)
    features_cache = Path(args.features_cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    # Symbol selection
    if args.subset_stocks:
        symbols = [s.strip() for s in args.subset_stocks.split(",")]
    elif args.max_stocks > 0:
        all_npz = sorted(features_cache.glob("*.npz"))
        symbols = [f.stem for f in all_npz[:args.max_stocks]]
    else:
        symbols = sorted([f.stem for f in features_cache.glob("*.npz")])

    console.print(f"\n[bold cyan]╔══ IntradayNet Pre-Batcher ══╗[/bold cyan]")
    console.print(f"  Stocks: [green]{len(symbols)}[/green]")
    console.print(f"  Features cache: [dim]{features_cache}[/dim]")
    console.print(f"  Output: [dim]{output_dir}[/dim]")
    console.print()

    splits = [
        ("train", cfg.splits.train_start, cfg.splits.train_end),
        ("val", cfg.splits.val_start, cfg.splits.val_end),
        ("test", cfg.splits.test_start, cfg.splits.test_end),
    ]

    t0 = time.time()
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} stocks"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for split_name, start, end in splits:
            task_id = progress.add_task(
                f"Building [cyan]{split_name}[/cyan]...",
                total=len(symbols),
            )
            n_samples = build_split(
                symbols, features_cache,
                pd.Timestamp(start), pd.Timestamp(end),
                cfg.model.sequence_length, cfg.horizons,
                cfg.train.sample_interval, split_name,
                output_dir, progress, task_id,
            )
            results[split_name] = n_samples

    elapsed = time.time() - t0

    # Summary table
    table = Table(title="Pre-batched Data Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")
    table.add_column("File Size", style="dim", justify="right")

    for split_name, n_samples in results.items():
        fpath = output_dir / f"{split_name}.npz"
        fsize = f"{fpath.stat().st_size / 1e6:.1f} MB" if fpath.exists() else "N/A"
        table.add_row(split_name, f"{n_samples:,}", fsize)

    console.print()
    console.print(table)
    console.print(f"\n[bold green]✓ Done in {elapsed:.1f}s[/bold green]")
    console.print(f"  Saved to: [dim]{output_dir.resolve()}[/dim]\n")


if __name__ == "__main__":
    main()
