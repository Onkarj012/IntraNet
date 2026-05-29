#!/usr/bin/env python3
"""
Train the live LightGBM backend bundle with calibrated direction models.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.config import load_config
from intradaynet.costs import DEFAULT_COSTS
from intradaynet.feature_contract import FEATURE_NAMES
from intradaynet.model_bundle import (
    HorizonBundleMetadata,
    ModelBundleManifest,
    save_manifest,
    validate_feature_contract,
)
from intradaynet.recommendation import probability_strength
from intradaynet.run_logging import command_string, start_run_logging


HORIZONS = ("H15", "H30", "H60")
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train live backend bundle")
    parser.add_argument("--config", default="configs/intraday_config.yaml")
    parser.add_argument("--prebatched", default="prebatched_live")
    parser.add_argument("--output-dir", default="runs/live_backend")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", default="H15,H30,H60",
                        help="Comma-separated horizons to train")
    parser.add_argument("--resume", action="store_true",
                        help="Skip horizons whose models and calibrator already exist")
    return parser.parse_args()


def load_split(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    feature_names = data["feature_names"].tolist()
    validate_feature_contract(feature_names)
    return {key: data[key] for key in data.files}


def fit_calibrator(raw_probs: np.ndarray, y_true: np.ndarray):
    if len(y_true) >= 5000:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_true)
        return calibrator, "isotonic"

    model = LogisticRegression(max_iter=500)
    model.fit(raw_probs.reshape(-1, 1), y_true)
    return model, "platt"


def apply_calibrator(calibrator, raw_probs: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(raw_probs)
    return calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]


def evaluate_direction(y_true: np.ndarray, raw_probs: np.ndarray, cal_probs: np.ndarray) -> dict:
    pred = (cal_probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, cal_probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "avg_confidence": float(np.mean(np.abs(cal_probs - 0.5) * 2.0)),
    }


def evaluate_regression(y_true: np.ndarray, pred: np.ndarray) -> dict:
    corr = float(np.corrcoef(y_true, pred)[0, 1]) if len(y_true) > 1 else 0.0
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "correlation": 0.0 if np.isnan(corr) else corr,
    }


def evaluate_topk(
    edge_true: np.ndarray,
    edge_pred: np.ndarray,
    dir_prob: np.ndarray,
    k: int = 25,
) -> dict:
    scores = edge_pred + probability_strength(dir_prob) * 0.001
    order = np.argsort(scores)[::-1][: min(k, len(scores))]
    chosen = edge_true[order]
    return {
        "topk_mean_edge": float(np.mean(chosen)) if len(chosen) else 0.0,
        "topk_hit_rate": float(np.mean(chosen > 0)) if len(chosen) else 0.0,
        "coverage": int(len(order)),
    }


def main():
    args = parse_args()
    run_name = f"train_live_backend_{Path(args.output_dir).name}"
    with start_run_logging(project_root=PROJECT_ROOT, log_group="training", run_name=run_name) as run_logger:
        global console
        console = Console()

        cfg = load_config(args.config)
        prebatched = Path(args.prebatched)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        requested_horizons = tuple(h.strip() for h in args.horizons.split(",") if h.strip())

        console.print(
            Panel.fit(
                "[bold cyan]IntradayNet Live Backend Trainer[/bold cyan]\n"
                f"[dim]Prebatched: {prebatched} | Output: {output_dir}[/dim]",
                border_style="cyan",
            )
        )
        console.print(f"[dim]Command:[/dim] {command_string()}")
        console.print(f"[dim]Run log:[/dim] {run_logger.log_path}")

        train = load_split(prebatched / "train.npz")
        val = load_split(prebatched / "val.npz")

        X_train = train["X_flat"]
        X_val = val["X_flat"]

        base_params = {
            "n_estimators": max(args.n_estimators, 1000),
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "min_child_samples": 100,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "verbosity": -1,
            "random_state": args.seed,
            "force_col_wise": True,
        }

        metrics: dict[str, dict] = {}
        horizon_files: dict[str, HorizonBundleMetadata] = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            active_horizons = [h for h in HORIZONS if h in requested_horizons]
            task = progress.add_task("Training horizons", total=len(active_horizons))
            for horizon in active_horizons:
                progress.update(task, description=f"Training {horizon}")
                dir_filename = f"dir_{horizon}.lgb"
                ret_filename = f"ret_{horizon}.lgb"
                edge_filename = f"edge_{horizon}.lgb"
                calibrator_filename = f"calibrator_{horizon}.pkl"

                if args.resume and all(
                    (output_dir / name).exists()
                    for name in (dir_filename, ret_filename, edge_filename, calibrator_filename)
                ):
                    horizon_files[horizon] = HorizonBundleMetadata(
                        direction_model=dir_filename,
                        gross_return_model=ret_filename,
                        net_edge_model=edge_filename,
                        calibrator=calibrator_filename,
                    )
                    progress.advance(task)
                    continue

                y_dir_train = train[f"dir_{horizon}"]
                y_dir_val = val[f"dir_{horizon}"]
                valid_train = ~np.isnan(y_dir_train)
                valid_val = ~np.isnan(y_dir_val)
                callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

                dir_model = lgb.LGBMClassifier(objective="binary", **base_params)
                dir_model.fit(
                    X_train[valid_train],
                    y_dir_train[valid_train].astype(int),
                    eval_set=[(X_val[valid_val], y_dir_val[valid_val].astype(int))],
                    callbacks=callbacks,
                )
                raw_probs = dir_model.predict_proba(X_val[valid_val])[:, 1]
                calibrator, calibrator_method = fit_calibrator(raw_probs, y_dir_val[valid_val].astype(int))
                cal_probs = apply_calibrator(calibrator, raw_probs)
                metrics[f"dir_{horizon}"] = evaluate_direction(y_dir_val[valid_val].astype(int), raw_probs, cal_probs)
                metrics[f"dir_{horizon}"]["calibration_method"] = calibrator_method
                metrics[f"dir_{horizon}"]["best_iteration"] = int(dir_model.best_iteration_ or dir_model.n_estimators)

                ret_model = lgb.LGBMRegressor(objective="regression", **base_params)
                ret_model.fit(
                    X_train,
                    train[f"gross_{horizon}"],
                    eval_set=[(X_val, val[f"gross_{horizon}"])],
                    callbacks=callbacks,
                )
                ret_pred = ret_model.predict(X_val)
                metrics[f"gross_{horizon}"] = evaluate_regression(val[f"gross_{horizon}"], ret_pred)
                metrics[f"gross_{horizon}"]["best_iteration"] = int(ret_model.best_iteration_ or ret_model.n_estimators)

                edge_model = lgb.LGBMRegressor(objective="regression", **base_params)
                edge_model.fit(
                    X_train,
                    train[f"edge_{horizon}"],
                    eval_set=[(X_val, val[f"edge_{horizon}"])],
                    callbacks=callbacks,
                )
                edge_pred = edge_model.predict(X_val)
                metrics[f"edge_{horizon}"] = evaluate_regression(val[f"edge_{horizon}"], edge_pred)
                metrics[f"edge_{horizon}"]["best_iteration"] = int(edge_model.best_iteration_ or edge_model.n_estimators)
                metrics[f"ranking_{horizon}"] = evaluate_topk(val[f"edge_{horizon}"], edge_pred, cal_probs, k=5)

                dir_model.booster_.save_model(str(output_dir / dir_filename))
                ret_model.booster_.save_model(str(output_dir / ret_filename))
                edge_model.booster_.save_model(str(output_dir / edge_filename))
                with open(output_dir / calibrator_filename, "wb") as f:
                    pickle.dump(calibrator, f)

                horizon_files[horizon] = HorizonBundleMetadata(
                    direction_model=dir_filename,
                    gross_return_model=ret_filename,
                    net_edge_model=edge_filename,
                    calibrator=calibrator_filename,
                )
                progress.advance(task)

        metrics["run_log"] = {"path": str(run_logger.log_path)}
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        manifest = ModelBundleManifest(
            bundle_name="live_lightgbm_backend",
            horizons=list(active_horizons),
            training_windows={
                "train_start": cfg.splits.train_start,
                "train_end": cfg.splits.train_end,
                "val_start": cfg.splits.val_start,
                "val_end": cfg.splits.val_end,
            },
            cost_summary={
                "position_value": 100_000.0,
                "round_trip_cost_example": DEFAULT_COSTS.estimate_for_position(100_000.0),
            },
            metrics=metrics,
            horizon_files=horizon_files,
            feature_names=list(FEATURE_NAMES),
            feature_count=len(FEATURE_NAMES),
        )
        save_manifest(output_dir, manifest)

        table = Table(title="Training Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Metric A", justify="right", style="green")
        table.add_column("Metric B", justify="right", style="yellow")

        for horizon in active_horizons:
            dm = metrics.get(f"dir_{horizon}")
            rm = metrics.get(f"gross_{horizon}")
            em = metrics.get(f"edge_{horizon}")
            if dm is not None:
                table.add_row(
                    f"dir_{horizon}",
                    f"acc={dm['accuracy']:.3f}",
                    f"auc={dm['auc']:.3f} @ {dm['best_iteration']}",
                )
            else:
                table.add_row(f"dir_{horizon}", "skipped", "resume")
            if rm is not None:
                table.add_row(
                    f"ret_{horizon}",
                    f"mae={rm['mae']:.4f}",
                    f"corr={rm['correlation']:.3f} @ {rm['best_iteration']}",
                )
            else:
                table.add_row(f"ret_{horizon}", "skipped", "resume")
            if em is not None:
                table.add_row(
                    f"edge_{horizon}",
                    f"mae={em['mae']:.4f}",
                    f"corr={em['correlation']:.3f} @ {em['best_iteration']}",
                )
            else:
                table.add_row(f"edge_{horizon}", "skipped", "resume")

        console.print()
        console.print(table)
        console.print(
            Panel.fit(
                f"[bold green]Saved live backend bundle[/bold green]\n"
                f"[dim]{output_dir}[/dim]\n"
                f"[bold]Run log:[/bold] [dim]{run_logger.log_path}[/dim]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
