#!/usr/bin/env python3
"""
Probability Calibration — calibrate LightGBM probabilities using isotonic regression.

After calibration:
- When model says P=0.70, stocks actually go up ~70% of the time
- Critical for setting confidence thresholds

Usage:
    python src/intradaynet/calibration.py --model runs/lgbm_v2/ --data prebatched_v2/ --horizon H60
    python src/intradaynet/calibration.py --model runs/lgbm_v2/ --data prebatched_v2/ --horizon H60 --method isotonic --output runs/lgbm_v2/calibrator_H60.pkl
"""

import argparse
import pickle
import json
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from intradaynet.feature_names import FEATURE_NAMES

console = Console()


def calibrate(
    booster: lgb.Booster,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> IsotonicRegression:
    """Calibrate using isotonic or sigmoid regression directly on probabilities."""
    raw_probs = booster.predict(X_cal)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_probs, y_cal)
    return ir


def evaluate_calibration(
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Check calibration quality by binning."""
    bins = np.linspace(0, 1, 11)
    rows = []
    overall_cal_error = 0.0
    total_weight = 0.0

    for i in range(len(bins) - 1):
        mask = (cal_probs >= bins[i]) & (cal_probs < bins[i + 1])
        if mask.sum() > 0:
            actual = float(y_true[mask].mean())
            predicted = float(cal_probs[mask].mean())
            n = int(mask.sum())
            gap = abs(actual - predicted)
            weight = n
            overall_cal_error += gap * weight
            total_weight += weight
            status = "OK" if gap < 0.05 else "WARN" if gap < 0.10 else "BAD"
            rows.append({
                "bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                "n": n,
                "predicted": predicted,
                "actual": actual,
                "gap": gap,
                "status": status,
            })

    ece = overall_cal_error / max(total_weight, 1)

    table = Table(title="Calibration Quality by Bin")
    table.add_column("Bin", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Predicted", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Status", width=8)

    for r in rows:
        color = "[green]" if r["status"] == "OK" else "[yellow]" if r["status"] == "WARN" else "[red]"
        table.add_row(
            r["bin"], str(r["n"]),
            f"{r['predicted']:.3f}", f"{r['actual']:.3f}",
            f"{r['gap']:.3f}",
            f"{color}{r['status']}[/{color}]",
        )

    console.print(table)
    console.print(f"Expected Calibration Error (ECE): {ece:.4f}")

    return {"ece": ece, "bins": rows}


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate LightGBM probabilities")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--data", required=True, help="Path to lgbm_dataset.npz")
    parser.add_argument("--horizon", default="H60", choices=["H15", "H30", "H60"])
    parser.add_argument("--method", default="isotonic", choices=["isotonic"])
    parser.add_argument("--cal-size", type=int, default=100_000,
                        help="Max calibration set size")
    parser.add_argument("--output", default="",
                        help="Save calibrator to pickle file")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model)
    data_path = Path(args.data)

    target_key = f"target_{args.horizon}"
    valid_key = f"valid_{args.horizon}"

    if data_path.is_dir():
        train_path = data_path / "val.npz"
    else:
        train_path = data_path

    if not train_path.exists():
        console.print(f"[red]ERROR: Data not found at {train_path}[/red]")
        return
    if not (model_dir / f"dir_{args.horizon}.lgb").exists():
        console.print(f"[red]ERROR: Model not found at {model_dir / f'dir_{args.horizon}.lgb'}[/red]")
        return

    console.print(f"Loading data from {train_path}...")
    data = np.load(train_path, allow_pickle=True)
    X = data["X_flat"]
    y_class = data[valid_key]
    valid_mask = ~np.isnan(y_class)

    X_valid = X[valid_mask]
    y_valid = y_class[valid_mask].astype(int)

    console.print(f"Valid samples: {len(X_valid):,}")

    if len(X_valid) > args.cal_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_valid), size=args.cal_size, replace=False)
        X_cal = X_valid[idx]
        y_cal = y_valid[idx]
    else:
        X_cal, y_cal = X_valid, y_valid

    console.print(f"Calibration set: {len(X_cal):,} samples")
    console.print(f"Class balance: {y_cal.mean()*100:.1f}% up / {(1-y_cal.mean())*100:.1f}% down")

    dir_model = lgb.Booster(model_file=str(model_dir / f"dir_{args.horizon}.lgb"))

    console.print(f"\nCalibrating direction model ({args.method})...")
    calibrator = calibrate(dir_model, X_cal, y_cal)

    raw_probs = dir_model.predict(X_cal)
    cal_probs = calibrator.predict(raw_probs)

    console.print(f"\n[bold]Direction Model Calibration — {args.horizon}[/bold]")
    eval_result = evaluate_calibration(raw_probs, cal_probs, y_cal)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "wb") as f:
            pickle.dump(calibrator, f)
        console.print(f"\n[green]Calibrator saved to {out_path}[/green]")

    cal_metrics = {
        "horizon": args.horizon,
        "method": args.method,
        "n_calibration_samples": len(X_cal),
        "ece": eval_result["ece"],
    }
    metrics_path = model_dir / f"calibration_{args.horizon}.json"
    with open(metrics_path, "w") as f:
        json.dump(cal_metrics, f, indent=2)
    console.print(f"Calibration metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
