#!/usr/bin/env python3
"""
LightGBM V2 training for IntradayNet.

Uses prebatched_v2 data with:
- Direction classifiers + magnitude regressors per horizon (H15, H30, H60)
- Walk-forward validation (temporal, not random)
- Smart subsampling via intradaynet.sampling
- Cost-adjusted valid mask for direction targets
- Feature importance extraction

Usage:
    python scripts/train_lgbm_v2.py
    python scripts/train_lgbm_v2.py --prebatched prebatched_v2 --output runs/lgbm_v2
    python scripts/train_lgbm_v2.py --max-train 500000 --max-val 100000
"""

import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from intradaynet.sampling import smart_subsample, stratified_subsample

console = Console()

HORIZONS = ["H15", "H30", "H60"]
N_FEATURES = 625


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM V2 models")
    parser.add_argument("--prebatched", type=str, default="prebatched_v2",
                        help="Directory with prebatched_v2 .npz files")
    parser.add_argument("--output-dir", type=str, default="runs/lgbm_v2",
                        help="Output directory for models")
    parser.add_argument("--max-train", type=int, default=500_000,
                        help="Max training samples (0 = all)")
    parser.add_argument("--max-val", type=int, default=100_000,
                        help="Max validation samples (0 = all)")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.7)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-subsample", action="store_true",
                        help="Disable smart subsampling (use all data)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-model training progress")
    return parser.parse_args()


def load_split(npz_path: Path, max_samples: int = 0, seed: int = 42):
    """Load one split's arrays from .npz file."""
    data = np.load(npz_path, mmap_mode="r")
    X = data["X_flat"][:]
    targets = {h: data[f"target_{h}"][:] for h in HORIZONS}
    valids = {h: data[f"valid_{h}"][:] for h in HORIZONS}
    return X, targets, valids


def apply_subsample(X, y_dir, y_mag, valid, max_samples, use_smart, seed):
    """Apply subsampling to training data."""
    if max_samples > 0 and len(X) > max_samples:
        if use_smart and y_dir is not None:
            valid_mask = ~np.isnan(y_dir)
            return smart_subsample(X, y_dir, y_mag, valid_mask,
                                   max_samples=max_samples, seed=seed)
        else:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), max_samples, replace=False)
            idx.sort()
            return X[idx], (y_dir[idx] if y_dir is not None else None), (y_mag[idx] if y_mag is not None else None)
    return X, y_dir, y_mag


def train_direction_model(X_train, y_train, X_val, y_val, horizon,
                          params, verbose):
    """Train a direction classifier for one horizon."""
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train.astype(int),
        eval_set=[(X_val, y_val.astype(int))],
        callbacks=[lgb.log_evaluation(0)],
    )
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, val_pred)
    try:
        auc = roc_auc_score(y_val, val_prob)
    except ValueError:
        auc = 0.5
    importance = model.feature_importances_.tolist()
    return model, {"accuracy": float(acc), "auc": float(auc)}, importance


def train_magnitude_model(X_train, y_train, X_val, y_val, horizon,
                         params, verbose):
    """Train a magnitude regressor for one horizon."""
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(0)])
    val_pred = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, val_pred))
    try:
        corr = float(np.corrcoef(val_pred, y_val)[0, 1])
    except Exception:
        corr = 0.0
    importance = model.feature_importances_.tolist()
    return model, {"mae": mae, "correlation": corr}, importance


def get_top_features(importance, top_n=30):
    """Return indices of top-N most important features."""
    imp = np.array(importance)
    return np.argsort(imp)[::-1][:top_n].tolist()


def main():
    args = parse_args()
    prebatched = Path(args.prebatched)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]IntradayNet LightGBM V2 Trainer[/bold cyan]",
        border_style="cyan",
        subtitle=f"Horizons: {', '.join(HORIZONS)} | Features: {N_FEATURES}",
    ))

    train_path = prebatched / "train.npz"
    val_path = prebatched / "val.npz"

    if not train_path.exists() or not val_path.exists():
        console.print(f"[red]✗ Prebatched data not found at {prebatched}[/red]")
        console.print(f"  Run: python scripts/prebatch_lgbm_v2.py --source csv --data-dir nifty500")
        return

    t0 = time.time()

    console.print("\n[bold]Loading data...[/bold]")
    X_train_raw, targets_train, valids_train = load_split(train_path, 0, args.seed)
    X_val_raw, targets_val, valids_val = load_split(val_path, 0, args.seed)

    console.print(f"  Train: {X_train_raw.shape[0]:,} × {X_train_raw.shape[1]}")
    console.print(f"  Val:   {X_val_raw.shape[0]:,} × {X_val_raw.shape[1]}")

    lgb_params_base = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_samples": args.min_child_samples,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbosity": -1,
        "random_state": args.seed,
    }
    clf_params = {**lgb_params_base, "objective": "binary", "metric": "binary_logloss"}
    reg_params = {**lgb_params_base, "objective": "regression"}

    all_metrics = {}
    top_features = {}
    t_train = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=not args.verbose,
    ) as progress:

        for horizon in HORIZONS:
            task_dir = progress.add_task(f"[cyan]Direction {horizon}[/cyan]", total=1)
            task_mag = progress.add_task(f"[yellow]Magnitude {horizon}[/yellow]", total=1)

            y_dir_train = valids_train[horizon]
            y_mag_train = targets_train[horizon]
            y_dir_val = valids_val[horizon]
            y_mag_val = targets_val[horizon]

            valid_train = ~np.isnan(y_dir_train)
            valid_val = ~np.isnan(y_dir_val)
            n_valid_train = valid_train.sum()
            n_valid_val = valid_val.sum()

            console.print(f"\n  {horizon}: {n_valid_train:,} train valid / {n_valid_val:,} val valid")

            X_tr_dir, y_tr_dir, y_tr_mag = apply_subsample(
                X_train_raw[valid_train],
                y_dir_train[valid_train],
                y_mag_train[valid_train],
                valid_train,
                args.max_train,
                not args.no_subsample,
                args.seed,
            )
            X_va_dir, y_va_dir, y_va_mag = apply_subsample(
                X_val_raw[valid_val],
                y_dir_val[valid_val],
                y_mag_val[valid_val],
                valid_val,
                args.max_val,
                False,
                args.seed,
            )

            console.print(f"    Subsampled: {len(X_tr_dir):,} train / {len(X_va_dir):,} val")

            n_up = (y_tr_dir == 1).sum()
            n_down = (y_tr_dir == 0).sum()
            console.print(f"    Class balance: {n_up:,} up ({n_up/len(y_tr_dir)*100:.1f}%) / "
                          f"{n_down:,} down ({n_down/len(y_tr_dir)*100:.1f}%)")

            model_dir, metrics_dir, imp_dir = train_direction_model(
                X_tr_dir, y_tr_dir, X_va_dir, y_va_dir,
                horizon, clf_params, args.verbose,
            )
            model_dir.booster_.save_model(str(output_dir / f"dir_{horizon}.lgb"))
            all_metrics[f"dir_{horizon}"] = metrics_dir
            top_features[f"dir_{horizon}"] = get_top_features(imp_dir, 30)
            progress.update(task_dir, completed=1)

            X_tr_mag, _, y_tr_mag_m = apply_subsample(
                X_train_raw, y_mag_train, y_mag_train, valid_train,
                args.max_train, not args.no_subsample, args.seed,
            )
            X_va_mag, _, y_va_mag_m = apply_subsample(
                X_val_raw, y_mag_val, y_mag_val, valid_val,
                args.max_val, False, args.seed,
            )

            model_mag, metrics_mag, imp_mag = train_magnitude_model(
                X_tr_mag, y_tr_mag_m, X_va_mag, y_va_mag_m,
                horizon, reg_params, args.verbose,
            )
            model_mag.booster_.save_model(str(output_dir / f"mag_{horizon}.lgb"))
            all_metrics[f"mag_{horizon}"] = metrics_mag
            top_features[f"mag_{horizon}"] = get_top_features(imp_mag, 30)
            progress.update(task_mag, completed=1)

    train_time = time.time() - t_train

    table = Table(title=f"LightGBM V2 Results — {train_time:.1f}s")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("AUC", style="green", justify="right")
    table.add_column("MAE", style="yellow", justify="right")
    table.add_column("Corr", style="yellow", justify="right")

    for h in HORIZONS:
        dm = all_metrics.get(f"dir_{h}", {})
        mm = all_metrics.get(f"mag_{h}", {})
        acc = f"{dm.get('accuracy', 0):.3f}" if dm else "—"
        auc = f"{dm.get('auc', 0):.3f}" if dm else "—"
        mae = f"{mm.get('mae', 0):.5f}" if mm else "—"
        corr = f"{mm.get('correlation', 0):.3f}" if mm else "—"
        table.add_row(f"dir_{h}", acc, auc, "—", "—")
        table.add_row(f"mag_{h}", "—", "—", mae, corr)

    console.print()
    console.print(table)

    if top_features:
        console.print("\n[bold]Top Features (direction H60):[/bold]")
        tf = top_features.get("dir_H60", [])
        feature_labels = [f"F{i}" for i in tf[:15]]
        console.print(f"  {', '.join(feature_labels)}")

    metadata = {
        "horizons": HORIZONS,
        "n_features": N_FEATURES,
        "metrics": all_metrics,
        "top_features": top_features,
        "args": vars(args),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"\n[bold green]✓ Models saved to {output_dir}[/bold green]")
    console.print(f"  Total time: [cyan]{time.time()-t0:.1f}s[/cyan]\n")


if __name__ == "__main__":
    main()
