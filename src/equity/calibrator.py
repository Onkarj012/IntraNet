"""
Probability calibration for IntradayNet model outputs.

LightGBM raw probabilities are notoriously uncalibrated — they cluster near
0 and 1. This module provides Platt scaling and isotonic regression via
sklearn's CalibratedClassifierCV, integrated into the model bundle workflow.

Usage:
    # Training
    calibrator = train_calibrator(
        model=raw_lgbm_model,
        X_val=val_features,
        y_val=val_labels,
        method="isotonic",
    )
    save_calibrator(calibrator, path)

    # Inference
    cal = load_calibrator(path)
    calibrated_probs = cal.predict_proba(features)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalibrationMethod = Literal["sigmoid", "isotonic"]


def train_calibrator(
    model,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    *,
    method: CalibrationMethod = "isotonic",
    cv: int | str = "prefit",
) -> CalibratedClassifierCV:
    """
    Train a probability calibrator on validation data.

    Args:
        model: Trained LightGBM Booster or sklearn classifier (must have predict_proba).
        X_val: Validation features, shape (n_samples, n_features).
        y_val: Validation labels. For direction models: ints in {0, 1, 2}
               (NO_TRADE=0, LONG=1, SHORT=2) or {0, 1} for binary.
        method: 'sigmoid' for Platt scaling (faster, fewer samples needed)
                or 'isotonic' for isotonic regression (more flexible, more data needed).
        cv: Number of CV folds for calibration, or 'prefit' if model is already fitted.

    Returns:
        Fitted CalibratedClassifierCV.
    """
    cal = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
        n_jobs=-1,
    )
    cal.fit(X_val, np.asarray(y_val).ravel())
    return cal


def train_platt_scaler(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:
    """
    Platt scaling (sigmoid calibration) from raw probabilities.

    Unlike CalibratedClassifierCV, this works directly on probability outputs,
    allowing calibration of any model without modifying its internals.

    Args:
        raw_probs: Shape (n_samples, n_classes). Raw model probabilities.
        y_true: Shape (n_samples,). True class labels (0-indexed ints).

    Returns:
        Fitted LogisticRegression calibration model. Apply with:
            calibrated = platt.predict_proba(raw_probs)
    """
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    platt.fit(raw_probs, y_true.ravel())
    return platt


def train_isotonic_regressor(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> IsotonicRegression:
    """
    Isotonic regression calibration for a single probability column.

    Use this for binary classification or one class at a time.

    Args:
        raw_probs: Shape (n_samples,). Raw probabilities for a single class.
        y_true: Shape (n_samples,). Binary labels (0 or 1).
    """
    iso = IsotonicRegression(
        y_min=y_min,
        y_max=y_max,
        out_of_bounds="clip",
    )
    iso.fit(raw_probs, y_true)
    return iso


def save_calibrator(calibrator, path: str | Path) -> None:
    """Save calibrator to disk via pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path: str | Path):
    """
    Load calibrator from disk.

    Returns None if file doesn't exist or can't be loaded.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def apply_calibration(
    calibrator,
    raw_probs: np.ndarray,
) -> np.ndarray:
    """
    Apply a fitted calibrator to raw model probabilities.

    Supports CalibratedClassifierCV, LogisticRegression (Platt wrapper),
    and IsotonicRegression objects.
    """
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(raw_probs)
    if isinstance(calibrator, IsotonicRegression):
        probs = raw_probs.ravel()
        calibrated = calibrator.transform(probs)
        return calibrated.reshape(-1, 1)
    raise TypeError(f"Unsupported calibrator type: {type(calibrator)}")


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Compute reliability curve for calibration diagnostics.

    Returns dict with:
        'prob_true': True fraction of positives in each bin
        'prob_pred': Mean predicted probability in each bin
        'counts': Number of samples in each bin
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    bins = np.digitize(y_prob, np.linspace(0, 1, n_bins + 1))
    counts = np.bincount(bins, minlength=n_bins + 2)[1:-1]
    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "counts": counts,
    }


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> str:
    """
    Generate a human-readable calibration report.

    Returns a formatted string showing bin-by-bin calibration quality.
    """
    curve = compute_calibration_curve(y_true, y_prob, n_bins)
    lines = [
        "Calibration Report",
        "=" * 50,
        f"{'Bin':>6s}  {'Pred':>7s}  {'Actual':>7s}  {'Count':>6s}  {'Diff':>7s}",
        "-" * 50,
    ]
    for i in range(len(curve["prob_pred"])):
        diff = curve["prob_pred"][i] - curve["prob_true"][i]
        lines.append(
            f"{i+1:>6d}  {curve['prob_pred'][i]:>7.3f}  {curve['prob_true'][i]:>7.3f}  "
            f"{curve['counts'][i]:>6.0f}  {diff:>+7.3f}"
        )
    prob_pred_arr = np.asarray(curve["prob_pred"])
    prob_true_arr = np.asarray(curve["prob_true"])
    if len(prob_pred_arr) > 0 and len(prob_true_arr) > 0:
        ece = float(np.mean(np.abs(prob_pred_arr - prob_true_arr)))
    else:
        ece = 0.0
    lines.extend([
        "-" * 50,
        f"ECE (Expected Calibration Error): {ece:.4f}",
    ])
    return "\n".join(lines)


def calibrate_direction_probs(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    method: CalibrationMethod = "isotonic",
) -> dict[str, object]:
    """
    Calibrate multiclass direction probabilities (LONG/SHORT/NO_TRADE → 3 classes).

    Fits a CalibratedClassifierCV on the raw LightGBM probabilities against
    true labels. Returns the fitted calibrator and calibration report.

    Args:
        raw_probs: Shape (n_samples, n_classes). Raw model probabilities.
        y_true: Shape (n_samples,). True class labels as 0-indexed ints.
        method: 'sigmoid' or 'isotonic'.

    Returns:
        dict with 'calibrator', 'report', and 'ece'.
    """
    from sklearn.base import clone

    wrapper = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    cal = CalibratedClassifierCV(
        estimator=wrapper,
        method=method,
        cv=3,
        n_jobs=-1,
    )
    cal.fit(raw_probs, y_true.ravel())
    calibrated_probs = cal.predict_proba(raw_probs)

    reports = {}
    n_classes = raw_probs.shape[1]
    class_names = {0: "NO_TRADE", 1: "LONG", 2: "SHORT"}
    ece_values = []

    for c in range(n_classes):
        name = class_names.get(c, f"class_{c}")
        curve = compute_calibration_curve(
            (y_true == c).astype(int), raw_probs[:, c]
        )
        prob_pred = np.asarray(curve["prob_pred"])
        prob_true = np.asarray(curve["prob_true"])
        if len(prob_pred) > 0 and len(prob_true) > 0:
            ece = float(np.mean(np.abs(prob_pred - prob_true)))
        else:
            ece = 0.0
        ece_values.append(ece)
        reports[name] = {
            "ece": ece,
            "prob_pred": prob_pred.tolist(),
            "prob_true": prob_true.tolist(),
        }

    return {
        "calibrator": cal,
        "report": calibration_report((y_true == 1).astype(int), raw_probs[:, 1]),
        "ece_per_class": dict(zip(class_names.values(), ece_values)),
        "mean_ece": float(np.mean(ece_values)),
    }
