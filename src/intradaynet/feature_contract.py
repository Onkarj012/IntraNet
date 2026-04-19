"""
Shared feature contract for the fast LightGBM trading backend.

This module is the single source of truth for:
- flattened feature names
- flattened feature order
- training/inference feature generation

All training, validation, inference, and backtesting code should go through
this contract so we never silently drift into a schema mismatch again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from intradaynet.features.per_bar_features import PER_BAR_FEATURE_NAMES
from intradaynet.features.session_features import SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES


FEATURE_SCHEMA_VERSION = "live_v2"
FLAT_WINDOWS = (5, 15, 30, 60, 120)
FLAT_STATS = ("mean", "std", "min", "max")


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    feature_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


def build_feature_names() -> list[str]:
    names: list[str] = []

    for feature_name in PER_BAR_FEATURE_NAMES:
        for window in FLAT_WINDOWS:
            for stat in FLAT_STATS:
                names.append(f"{feature_name}_{stat}_{window}")
        names.append(f"{feature_name}_last")
        names.append(f"{feature_name}_first")
        names.append(f"{feature_name}_diff_last_first")
        names.append(f"{feature_name}_diff_5v30")
        names.append(f"{feature_name}_diff_15v60")

    names.extend(f"session::{name}" for name in SESSION_FEATURE_NAMES)
    names.extend(f"sentiment::{name}" for name in SENTIMENT_FEATURE_NAMES)
    return names


FEATURE_NAMES = build_feature_names()
FEATURE_SCHEMA = FeatureSchema(
    version=FEATURE_SCHEMA_VERSION,
    feature_names=tuple(FEATURE_NAMES),
)


def _safe_stat(segment: np.ndarray, stat: str) -> float:
    if stat == "mean":
        return float(np.nanmean(segment))
    if stat == "std":
        return float(np.nanstd(segment))
    if stat == "min":
        return float(np.nanmin(segment))
    if stat == "max":
        return float(np.nanmax(segment))
    raise ValueError(f"Unsupported stat: {stat}")


def flatten_intraday_window(
    window: np.ndarray,
    session: Sequence[float],
    sentiment: Sequence[float],
) -> np.ndarray:
    """
    Flatten a single (L, F) window into the shared backend feature vector.
    """
    if window.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {window.shape}")
    if window.shape[1] != len(PER_BAR_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(PER_BAR_FEATURE_NAMES)} per-bar features, got {window.shape[1]}"
        )

    session_arr = np.asarray(session, dtype=np.float32)
    sentiment_arr = np.asarray(sentiment, dtype=np.float32)

    if session_arr.shape[0] != len(SESSION_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(SESSION_FEATURE_NAMES)} session features, got {session_arr.shape[0]}"
        )
    if sentiment_arr.shape[0] != len(SENTIMENT_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(SENTIMENT_FEATURE_NAMES)} sentiment features, got {sentiment_arr.shape[0]}"
        )

    length = window.shape[0]
    values: list[float] = []

    for col_idx in range(window.shape[1]):
        column = window[:, col_idx]

        for flat_window in FLAT_WINDOWS:
            segment = column[-flat_window:] if flat_window <= length else column
            for stat in FLAT_STATS:
                values.append(_safe_stat(segment, stat))

        values.append(float(column[-1]))
        values.append(float(column[0]))
        values.append(float(column[-1] - column[0]))

        last_5 = column[-5:] if length >= 5 else column
        last_30 = column[-30:] if length >= 30 else column
        last_15 = column[-15:] if length >= 15 else column
        last_60 = column[-60:] if length >= 60 else column

        values.append(float(np.nanmean(last_5) - np.nanmean(last_30)))
        values.append(float(np.nanmean(last_15) - np.nanmean(last_60)))

    values.extend(float(v) for v in session_arr)
    values.extend(float(v) for v in sentiment_arr)

    flat = np.asarray(values, dtype=np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0)

    if flat.shape[0] != FEATURE_SCHEMA.feature_count:
        raise RuntimeError(
            f"Feature contract mismatch: built {flat.shape[0]} features, "
            f"expected {FEATURE_SCHEMA.feature_count}"
        )
    return flat


def flatten_intraday_batch(
    windows: np.ndarray,
    session_batch: np.ndarray,
    sentiment_batch: np.ndarray,
) -> np.ndarray:
    """
    Flatten a batch of windows using the exact same feature order as inference.
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D windows array, got {windows.shape}")
    if windows.shape[0] != len(session_batch) or windows.shape[0] != len(sentiment_batch):
        raise ValueError("Batch size mismatch between windows, session, and sentiment")

    batch = [
        flatten_intraday_window(windows[i], session_batch[i], sentiment_batch[i])
        for i in range(windows.shape[0])
    ]
    return np.stack(batch, axis=0).astype(np.float32)
