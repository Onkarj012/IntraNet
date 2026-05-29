"""
Model bundle utilities for the live LightGBM backend.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb

from intradaynet.calibrator import load_calibrator as _load_calibrator_file
from intradaynet.feature_contract import FEATURE_SCHEMA


@dataclass
class HorizonBundleMetadata:
    direction_model: str
    gross_return_model: str
    net_edge_model: str
    calibrator: str | None = None


@dataclass
class ModelBundleManifest:
    bundle_name: str
    bundle_version: str = "live_v2"
    schema_version: str = FEATURE_SCHEMA.version
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_SCHEMA.feature_names))
    feature_count: int = FEATURE_SCHEMA.feature_count
    horizons: list[str] = field(default_factory=list)
    label_version: str = "labels_v2"
    preprocessing_version: str = "preprocessing_v1"
    training_windows: dict[str, str] = field(default_factory=dict)
    cost_summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    horizon_files: dict[str, HorizonBundleMetadata] = field(default_factory=dict)
    calibration_config: dict[str, Any] = field(default_factory=dict)


def validate_feature_contract(feature_names: list[str]) -> None:
    if feature_names != list(FEATURE_SCHEMA.feature_names):
        raise ValueError(
            "Feature contract mismatch between runtime features and bundle schema"
        )


def save_manifest(output_dir: Path, manifest: ModelBundleManifest) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    serializable = asdict(manifest)
    serializable["horizon_files"] = {
        key: asdict(value) for key, value in manifest.horizon_files.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    return path


def load_manifest(bundle_dir: Path) -> ModelBundleManifest:
    path = bundle_dir / "manifest.json"
    with open(path, "r") as f:
        data = json.load(f)
    horizon_files = {
        key: HorizonBundleMetadata(**value)
        for key, value in data.get("horizon_files", {}).items()
    }
    data["horizon_files"] = horizon_files
    if "bundle_version" not in data:
        data["bundle_version"] = "live_v1"
    if "label_version" not in data:
        data["label_version"] = "labels_v1"
    if "calibration_config" not in data:
        data["calibration_config"] = {}
    manifest = ModelBundleManifest(**data)
    if manifest.feature_count != len(manifest.feature_names):
        raise ValueError("Manifest feature_count does not match feature_names length")
    return manifest


def load_calibrator(calibrator_path: Path):
    """
    Load a calibrator from disk.

    Uses the canonical calibrator loader from calibrator.py.
    Falls back to direct pickle load for backward compatibility.
    """
    cal = _load_calibrator_file(calibrator_path)
    if cal is not None:
        return cal
    if not calibrator_path.exists():
        return None
    try:
        with open(calibrator_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def load_bundle(bundle_dir: str | Path) -> tuple[ModelBundleManifest, dict[str, dict[str, Any]]]:
    bundle_path = Path(bundle_dir)
    manifest = load_manifest(bundle_path)
    validate_feature_contract(manifest.feature_names)

    models: dict[str, dict[str, Any]] = {}
    for horizon, files in manifest.horizon_files.items():
        models[horizon] = {
            "dir": lgb.Booster(model_file=str(bundle_path / files.direction_model)),
            "ret": lgb.Booster(model_file=str(bundle_path / files.gross_return_model)),
            "edge": lgb.Booster(model_file=str(bundle_path / files.net_edge_model)),
            "calibrator": load_calibrator(bundle_path / files.calibrator)
            if files.calibrator
            else None,
        }
    return manifest, models


def calibrate_probabilities(
    models: dict[str, dict[str, Any]],
    horizon: str,
    raw_probs: Any,
) -> Any:
    """
    Apply calibrator to raw model probabilities for a given horizon.

    If no calibrator is available, returns raw_probs unchanged.
    """
    horizon_models = models.get(horizon, {})
    calibrator = horizon_models.get("calibrator")
    if calibrator is None:
        return raw_probs

    import numpy as np
    from intradaynet.calibrator import apply_calibration

    return apply_calibration(calibrator, np.asarray(raw_probs))


def bundle_has_calibration(manifest: ModelBundleManifest) -> bool:
    """Check if the bundle contains calibration data."""
    if manifest.calibration_config:
        return True
    for files in manifest.horizon_files.values():
        if files.calibrator:
            return True
    return False
