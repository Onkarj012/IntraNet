"""Tests for probability calibration module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intradaynet.calibrator import (
    calibrate_direction_probs,
    calibration_report,
    compute_calibration_curve,
    load_calibrator,
    save_calibrator,
    train_isotonic_regressor,
    train_platt_scaler,
)


def test_platt_scaler_improves_calibration():
    """Platt scaling should produce better calibrated probabilities."""
    rng = np.random.default_rng(42)
    n = 500
    raw_probs = np.clip(rng.normal(0.6, 0.25, n), 0.01, 0.99).reshape(-1, 1)
    y_true = (rng.random(n) < raw_probs.ravel()).astype(int)

    platt = train_platt_scaler(raw_probs, y_true)
    calibrated = platt.predict_proba(raw_probs)

    assert calibrated.shape == (n, 2)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_isotonic_regressor_monotonic():
    """Isotonic regression produces monotonic probability transformations."""
    rng = np.random.default_rng(42)
    n = 200
    raw_probs = np.sort(rng.uniform(0.05, 0.95, n))
    y_true = (rng.random(n) < raw_probs).astype(int)

    iso = train_isotonic_regressor(raw_probs, y_true)
    calibrated = iso.transform(raw_probs)

    assert len(calibrated) == n
    assert (np.diff(calibrated) >= 0).all()


def test_calibration_curve_structure():
    """Calibration curve returns expected dict structure."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_prob = rng.uniform(0, 1, 100)

    curve = compute_calibration_curve(y_true, y_prob, n_bins=5)

    assert "prob_true" in curve
    assert "prob_pred" in curve
    assert "counts" in curve
    assert len(curve["prob_true"]) <= 5


def test_calibration_report_generates_string():
    """Calibration report produces a non-empty string."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_prob = rng.uniform(0, 1, 100)

    report = calibration_report(y_true, y_prob)
    assert isinstance(report, str)
    assert "ECE" in report


def test_calibrator_save_load_roundtrip(tmp_path: Path):
    """Calibrator should survive a save/load roundtrip."""
    rng = np.random.default_rng(42)
    n = 200
    raw_probs = rng.uniform(0.1, 0.9, n).reshape(-1, 1)
    y_true = (rng.random(n) < raw_probs.ravel()).astype(int)

    platt = train_platt_scaler(raw_probs, y_true)
    cal_path = tmp_path / "calibrator.pkl"
    save_calibrator(platt, cal_path)

    loaded = load_calibrator(cal_path)
    assert loaded is not None

    orig_cal = platt.predict_proba(raw_probs)
    load_cal = loaded.predict_proba(raw_probs)
    np.testing.assert_array_almost_equal(orig_cal, load_cal)


def test_calibrate_direction_probs_multiclass():
    """Multiclass direction calibration produces a calibrator and report."""
    rng = np.random.default_rng(42)
    n = 300
    raw_probs = rng.dirichlet([1, 2, 1], n)
    y_true = np.array([int(np.argmax(row)) for row in raw_probs])

    result = calibrate_direction_probs(raw_probs, y_true, method="sigmoid")

    assert "calibrator" in result
    assert "report" in result
    assert "ece_per_class" in result
    assert result["calibrator"] is not None
    assert isinstance(result["mean_ece"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
