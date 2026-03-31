#!/usr/bin/env python3
"""
LightGBM training for intraday prediction — optimized for large datasets.

Flattens the 120-bar window into aggregate features and trains
gradient-boosted trees. Uses chunked processing to avoid RAM issues.

Usage:
    python scripts/train_lgbm.py --config configs/intraday_config.yaml
    python scripts/train_lgbm.py --prebatched prebatched/
"""

import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from intradaynet.config import load_config

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM models")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--prebatched", type=str, default="prebatched",
                        help="Directory with pre-batched .npz files")
    parser.add_argument("--output-dir", type=str, default="runs/lgbm")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int, default=500_000,
                        help="Max training samples (0 = use all). Subsampling speeds up training hugely.")
    return parser.parse_args()


def flatten_chunk(per_bar_chunk, context_chunk, sentiment_chunk):
    """
    Flatten a chunk of data. Works on any size array.

    From (N, 120, 25) → (N, ~F) flat features.
    Simplified: use last bar, mean, std, min, max (fewer windows = faster).
    """
    N, L, F = per_bar_chunk.shape
    agg_features = []

    # Window-based aggregates (fewer windows for speed)
    for w in [5, 30, 120]:
        w = min(w, L)
        window = per_bar_chunk[:, -w:, :]

        agg_features.append(np.nanmean(window, axis=1))  # (N, F)
        agg_features.append(np.nanstd(window, axis=1))   # (N, F)

        # Slope
        if w > 1:
            slope = (window[:, -1, :] - window[:, 0, :]) / w
            agg_features.append(slope)

    # Last bar value
    agg_features.append(per_bar_chunk[:, -1, :])

    # Min/max of full window
    agg_features.append(np.nanmin(per_bar_chunk, axis=1))
    agg_features.append(np.nanmax(per_bar_chunk, axis=1))

    flat = np.concatenate(agg_features + [context_chunk, sentiment_chunk], axis=1)
    return np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def load_and_flatten(npz_path, max_samples=0):
    """
    Load prebatched .npz with memory mapping and flatten in chunks.
    Returns flat feature matrix + targets.
    """
    console.print(f"  Loading [cyan]{npz_path.name}[/cyan] (memory-mapped)...")
    data = np.load(npz_path, mmap_mode='r')

    N_total = len(data['Y_direction'])
    console.print(f"  Total samples: [green]{N_total:,}[/green]")

    # Subsample if needed
    if max_samples > 0 and N_total > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(N_total, max_samples, replace=False)
        indices.sort()
        console.print(f"  Subsampling to [yellow]{max_samples:,}[/yellow] samples...")
    else:
        indices = np.arange(N_total)
        max_samples = N_total

    # Process in chunks to avoid RAM explosion
    chunk_size = 50_000
    n_chunks = (len(indices) + chunk_size - 1) // chunk_size
    flat_chunks = []

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task(f"Flattening {npz_path.name}...", total=n_chunks)

        for i in range(n_chunks):
            chunk_idx = indices[i * chunk_size: (i + 1) * chunk_size]

            # Read only this chunk from memory-mapped arrays
            per_bar = np.array(data['X_per_bar'][chunk_idx])
            context = np.array(data['X_context'][chunk_idx])
            sentiment = np.array(data['X_sentiment'][chunk_idx])

            flat_chunks.append(flatten_chunk(per_bar, context, sentiment))
            del per_bar, context, sentiment

            progress.update(task, advance=1)

    X_flat = np.concatenate(flat_chunks, axis=0)
    del flat_chunks

    # Targets (small — just read directly)
    Y_direction = np.array(data['Y_direction'][indices])
    Y_magnitude = np.array(data['Y_magnitude'][indices])

    console.print(f"  Flat features: [green]{X_flat.shape[1]}[/green] per sample")

    return X_flat, Y_direction, Y_magnitude


def train_horizon_model(X_train, y_train, X_val, y_val, horizon_name,
                        is_classifier, params, progress, task_id):
    """Train a single LightGBM model for one horizon."""

    if is_classifier:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)],
        )
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, val_pred)
        try:
            auc = roc_auc_score(y_val, val_prob)
        except ValueError:
            auc = 0.5
        return model, {"accuracy": acc, "auc": auc}
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)],
        )
        val_pred = model.predict(X_val)
        mae = np.mean(np.abs(val_pred - y_val))
        corr = np.corrcoef(val_pred, y_val)[0, 1] if len(np.unique(y_val)) > 1 else 0
        return model, {"mae": mae, "correlation": corr}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    prebatched = Path(args.prebatched)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]IntradayNet — LightGBM Trainer (Optimized)[/bold cyan]",
        border_style="cyan",
    ))

    t0 = time.time()

    # Load and flatten with chunked processing
    X_train, Y_dir_train, Y_mag_train = load_and_flatten(
        prebatched / "train.npz", max_samples=args.max_samples
    )
    X_val, Y_dir_val, Y_mag_val = load_and_flatten(
        prebatched / "val.npz", max_samples=min(args.max_samples // 3, 200_000)
    )

    console.print(f"\n  Loaded in {time.time() - t0:.1f}s\n")

    # LightGBM parameters
    clf_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbosity": -1,
        "random_state": 42,
    }
    reg_params = {**clf_params, "objective": "regression"}

    horizon_names = [f"H{h}" for h in cfg.horizons]
    all_metrics = {}

    t_train = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Train direction classifiers
        for hi, hname in enumerate(horizon_names):
            task_id = progress.add_task(f"Direction {hname}...", total=1)

            y_tr = Y_dir_train[:, hi].astype(int)
            y_va = Y_dir_val[:, hi].astype(int)

            model, metrics = train_horizon_model(
                X_train, y_tr, X_val, y_va, hname,
                is_classifier=True, params=clf_params,
                progress=progress, task_id=task_id,
            )

            model.booster_.save_model(str(output_dir / f"dir_{hname}.lgb"))
            all_metrics[f"dir_{hname}"] = metrics
            progress.update(task_id, completed=1)

        # Train magnitude regressors
        for hi, hname in enumerate(horizon_names):
            task_id = progress.add_task(f"Magnitude {hname}...", total=1)

            y_tr = Y_mag_train[:, hi]
            y_va = Y_mag_val[:, hi]

            model, metrics = train_horizon_model(
                X_train, y_tr, X_val, y_va, hname,
                is_classifier=False, params=reg_params,
                progress=progress, task_id=task_id,
            )

            model.booster_.save_model(str(output_dir / f"mag_{hname}.lgb"))
            all_metrics[f"mag_{hname}"] = metrics
            progress.update(task_id, completed=1)

    train_time = time.time() - t_train

    # Results table
    table = Table(title=f"LightGBM Results — trained in {train_time:.1f}s")
    table.add_column("Horizon", style="cyan")
    table.add_column("Dir Accuracy", style="green", justify="right")
    table.add_column("Dir AUC", style="green", justify="right")
    table.add_column("Mag MAE", style="yellow", justify="right")
    table.add_column("Mag Corr", style="yellow", justify="right")

    for hname in horizon_names:
        dm = all_metrics[f"dir_{hname}"]
        mm = all_metrics[f"mag_{hname}"]
        table.add_row(
            hname,
            f"{dm['accuracy']:.3f}",
            f"{dm['auc']:.3f}",
            f"{mm['mae']:.5f}",
            f"{mm['correlation']:.3f}",
        )

    console.print()
    console.print(table)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    console.print(f"\n[bold green]✓ Models saved to {output_dir}[/bold green]")
    console.print(f"  Total training time: [cyan]{train_time:.1f}s[/cyan]\n")


if __name__ == "__main__":
    main()
