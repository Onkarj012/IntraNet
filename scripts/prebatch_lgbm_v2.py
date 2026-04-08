#!/usr/bin/env python3
"""
Prebatch pipeline for IntradayNet LightGBM V2.

Reads Parquet minute-bar data, computes features on-the-fly using
leak-free source formulas, extracts sliding windows, computes
cost-adjusted targets, and saves flat numpy arrays.

Usage:
    python scripts/prebatch_lgbm_v2.py
    python scripts/prebatch_lgbm_v2.py --source csv --data-dir nifty500
    python scripts/prebatch_lgbm_v2.py --max-stocks 50 --dry-run
"""

import argparse
import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from intradaynet.config import load_config
from intradaynet.features.per_bar_features import compute_per_bar_features, PER_BAR_FEATURE_NAMES
from intradaynet.features.session_features import compute_session_features, SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES

console = Console()

HORIZONS = {"H15": 15, "H30": 30, "H60": 60}
SEQ_LENGTH = 120
SAMPLE_INTERVAL = 15
MIN_SESSION_BARS = 150


def parse_args():
    parser = argparse.ArgumentParser(description="Prebatch LightGBM V2 training data")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--data-dir", type=str, default="nifty500_parquet",
                        help="Directory with Parquet files (or CSV if --source csv)")
    parser.add_argument("--source", type=str, default="parquet", choices=["parquet", "csv"],
                        help="Input data format")
    parser.add_argument("--sentiment-csv", type=str, default="sentiment/combined_sentiment_2015_2025.csv")
    parser.add_argument("--output-dir", type=str, default="prebatched_v2")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="Limit stocks for testing (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process 1 stock per split and exit")
    return parser.parse_args()


def load_stock_data(symbol: str, source: str, data_dir: Path) -> pd.DataFrame:
    """Load minute-bar data for one stock."""
    if source == "parquet":
        pq_path = data_dir / f"{symbol}.parquet"
        if not pq_path.exists():
            return None
        df = pd.read_parquet(pq_path)
    else:
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.set_index("datetime")

    df.columns = df.columns.str.lower()
    return df


def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-bar and session features from raw OHLCV."""
    per_bar = compute_per_bar_features(df)
    session_feats = compute_session_features(df)
    return per_bar, session_feats


def extract_windows(
    df: pd.DataFrame,
    per_bar: pd.DataFrame,
    session_feats: pd.DataFrame,
    target_df: pd.DataFrame,
    symbol: str,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
) -> dict:
    """
    Extract sliding windows from one stock's data.
    Processes one session at a time for efficient batching.
    """
    close = df["close"].values
    dates = df.index.date if hasattr(df.index, "date") else np.array([d.date() for d in df.index])
    unique_dates = np.unique(dates)

    all_windows = []
    all_targets = {h: [] for h in HORIZONS}
    all_valid = {h: [] for h in HORIZONS}

    non_eod_max = max((h for h in HORIZONS.values() if h < 375), default=60)
    per_bar_vals = per_bar.values
    split_start_date = split_start.date()
    split_end_date = split_end.date()

    for date in unique_dates:
        if date < split_start_date or date > split_end_date:
            continue

        session_mask = dates == date
        session_indices = np.where(session_mask)[0]
        n_bars = len(session_indices)

        if n_bars < MIN_SESSION_BARS:
            continue

        sess_start = session_indices[0]
        sess_end = sess_start + n_bars

        offsets = np.arange(SEQ_LENGTH, n_bars - non_eod_max, SAMPLE_INTERVAL)
        if len(offsets) == 0:
            continue

        start_indices = sess_start + offsets - SEQ_LENGTH
        end_indices = sess_start + offsets

        for si, ei in zip(start_indices, end_indices):
            all_windows.append(per_bar_vals[si:ei])

        anchor_close = close[sess_start + offsets]

        for h_name, h_bars in HORIZONS.items():
            future_idx = sess_start + offsets + h_bars
            session_end = sess_start + n_bars

            future_close = np.where(
                future_idx < session_end,
                close[future_idx],
                np.where(h_bars >= 375, close[session_end - 1], np.nan)
            )

            valid = future_idx < session_end
            raw_ret = np.where(
                valid & (anchor_close > 1e-10),
                (future_close - anchor_close) / anchor_close,
                0.0
            )
            raw_ret = np.clip(raw_ret, -0.05, 0.05)

            cost_adj = np.where(raw_ret > 0, raw_ret - 0.001, raw_ret + 0.001)
            cost_adj = np.abs(cost_adj) * np.sign(np.where(raw_ret != 0, raw_ret, 1))

            all_targets[h_name].append(raw_ret)
            dir_signal = np.where(cost_adj > 0.003, 1.0, np.where(cost_adj < -0.003, 0.0, np.nan))
            all_valid[h_name].append(dir_signal)

    if not all_windows:
        return None

    all_windows = np.stack(all_windows).astype(np.float32)
    nan_mask = ~np.isnan(all_windows).any(axis=(1, 2))
    all_windows = all_windows[nan_mask]

    result = {"windows": all_windows}
    for h_name in HORIZONS:
        t = np.concatenate(all_targets[h_name], axis=0)[nan_mask].astype(np.float32)
        v = np.concatenate(all_valid[h_name], axis=0)[nan_mask].astype(np.float32)
        result[f"target_{h_name}"] = t
        result[f"valid_{h_name}"] = v

    return result


def flatten_windows_vectorized(
    windows: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """
    Vectorized flattening: (N, 120, 25) → (N, F).

    Windows at 5 scales: 5, 15, 30, 60, 120
    Per window: mean, std, min, max
    Plus: last value, first value, diff (last - first)
    """
    N, L, F = windows.shape
    parts = []

    for w in [5, 15, 30, 60, 120]:
        if w > L:
            continue
        window = windows[:, -w:, :]
        parts.append(np.nanmean(window, axis=1))
        parts.append(np.nanstd(window, axis=1))
        parts.append(np.nanmin(window, axis=1))
        parts.append(np.nanmax(window, axis=1))

    parts.append(windows[:, -1, :])
    parts.append(windows[:, 0, :])
    diff = windows[:, -1, :] - windows[:, 0, :]
    parts.append(diff)

    if L >= 30:
        parts.append(np.nanmean(windows[:, -5:, :], axis=1) - np.nanmean(windows[:, -30:, :], axis=1))
    if L >= 60:
        parts.append(np.nanmean(windows[:, -15:, :], axis=1) - np.nanmean(windows[:, -60:, :], axis=1))

    flat = np.concatenate(parts, axis=1)
    return np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def process_stock(
    symbol: str,
    source: str,
    data_dir: Path,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    dry_run: bool,
) -> dict | None:
    """Process one stock and return extracted data."""
    df = load_stock_data(symbol, source, data_dir)
    if df is None or len(df) < 50:
        return None

    try:
        per_bar, session_feats = compute_features(df)
    except Exception as e:
        return None

    data = extract_windows(df, per_bar, session_feats, None, symbol, split_start, split_end)
    if data is None:
        return None

    return {
        "windows": data["windows"],
        **{k: v for k, v in data.items() if k.startswith("target_") or k.startswith("valid_")},
    }


def build_split(
    symbols: list[str],
    source: str,
    data_dir: Path,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    dry_run: bool,
    progress: Progress,
    task_id: int,
) -> dict:
    """Process all stocks for one split and return aggregated arrays."""
    stock_results = []
    n_skipped = 0

    stocks_to_process = symbols[:5] if dry_run else symbols

    for i, symbol in enumerate(stocks_to_process):
        progress.update(task_id, description=f"[cyan]{split_start.year}[/cyan] {symbol} ({i+1}/{len(stocks_to_process)})")
        result = process_stock(symbol, source, data_dir, split_start, split_end, dry_run)
        if result is not None:
            stock_results.append(result)
        else:
            n_skipped += 1

        if (i + 1) % 10 == 0:
            gc.collect()

    if not stock_results:
        return None

    n_total = sum(r["windows"].shape[0] for r in stock_results)
    flat_features = np.zeros((n_total, 0), dtype=np.float32)

    all_windows = np.concatenate([r["windows"] for r in stock_results], axis=0)
    flat_features = flatten_windows_vectorized(all_windows, PER_BAR_FEATURE_NAMES)

    output = {"X_flat": flat_features}
    for h_name in HORIZONS:
        tgt_key = f"target_{h_name}"
        val_key = f"valid_{h_name}"
        output[tgt_key] = np.concatenate([r[tgt_key] for r in stock_results], axis=0)
        output[val_key] = np.concatenate([r[val_key] for r in stock_results], axis=0)

    return output


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]╔══ IntradayNet Prebatcher V2 ══╗[/bold cyan]")
    console.print(f"  Source: [dim]{data_dir} ({args.source})[/dim]")
    console.print(f"  Output: [dim]{output_dir}[/dim]")
    console.print(f"  Dry run: [yellow]{args.dry_run}[/yellow]")

    if not data_dir.exists():
        console.print(f"\n[red]✗ Data directory not found: {data_dir}[/red]")
        if args.source == "csv":
            console.print(f"  → Falling back to nifty500/ CSV source")
            data_dir = Path("nifty500")
            args.source = "csv"
        else:
            console.print(f"  → Run: python scripts/convert_to_parquet.py")
            console.print(f"  → Or: python scripts/prebatch_lgbm_v2.py --source csv --data-dir nifty500")
            return

    ext = "parquet" if args.source == "parquet" else "csv"
    suffix = "" if args.source == "parquet" else "_minute"
    symbols = sorted([
        p.stem.replace("_minute", "") if suffix else p.stem
        for p in data_dir.glob(f"*.{ext}")
        if not p.name.startswith(".")
    ])
    if args.max_stocks > 0:
        symbols = symbols[:args.max_stocks]

    console.print(f"  Stocks: [green]{len(symbols)}[/green]")
    if args.dry_run:
        console.print(f"  [yellow]DRY RUN: processing 5 stocks only[/yellow]")
    console.print()

    if not symbols:
        console.print("[red]✗ No data files found[/red]")
        return

    cfg = load_config(args.config) if Path(args.config).exists() else None

    splits = [
        ("train", pd.Timestamp(cfg.splits.train_start if cfg else "2022-01-01"),
                   pd.Timestamp(cfg.splits.train_end if cfg else "2023-12-31")),
        ("val",   pd.Timestamp(cfg.splits.val_start if cfg else "2024-01-01"),
                   pd.Timestamp(cfg.splits.val_end if cfg else "2024-12-31")),
        ("test",  pd.Timestamp(cfg.splits.test_start if cfg else "2025-01-01"),
                   pd.Timestamp(cfg.splits.test_end if cfg else "2025-12-31")),
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
                f"Processing [cyan]{split_name}[/cyan] ({start.date()} → {end.date()})...",
                total=len(symbols),
            )

            split_data = build_split(
                symbols, args.source, data_dir,
                start, end, args.dry_run,
                progress, task_id,
            )

            if split_data is not None:
                out_path = output_dir / f"{split_name}.npz"
                np.savez_compressed(out_path, **split_data)
                n_samples = split_data["X_flat"].shape[0]
                n_feats = split_data["X_flat"].shape[1]
                results[split_name] = (n_samples, n_feats)
                progress.update(task_id, description=f"[green]✓ {split_name}: {n_samples:,} × {n_feats}[/green]")
            else:
                results[split_name] = (0, 0)
                progress.update(task_id, description=f"[yellow]⚠ {split_name}: no samples[/yellow]")

    elapsed = time.time() - t0

    table = Table(title="Prebatched Data Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")
    table.add_column("Features", style="dim", justify="right")
    table.add_column("File Size", style="dim", justify="right")

    for split_name, (n_samples, n_feats) in results.items():
        fpath = output_dir / f"{split_name}.npz"
        fsize = f"{fpath.stat().st_size / 1e6:.1f} MB" if fpath.exists() and n_samples > 0 else "N/A"
        table.add_row(split_name, f"{n_samples:,}", str(n_feats), fsize)

    console.print()
    console.print(table)
    console.print(f"\n[bold green]✓ Done in {elapsed:.1f}s[/bold green]")
    console.print(f"  Output: [dim]{output_dir.resolve()}[/dim]\n")


if __name__ == "__main__":
    main()
