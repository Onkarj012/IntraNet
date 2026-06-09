"""
Backwards-compatible wrapper around the shared feature contract.
"""

from typing import List, Tuple

import numpy as np

from equity.feature_contract import (
    FEATURE_NAMES,
    flatten_intraday_batch,
    flatten_intraday_window,
)


def flatten_window_for_lgbm(
    window: np.ndarray,
    session: np.ndarray,
    sentiment: np.ndarray,
    feature_names: List[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    flat = flatten_intraday_window(window, session, sentiment)
    return flat, list(FEATURE_NAMES)


def flatten_chunk_fast(
    windows: np.ndarray,
    session_batch: np.ndarray,
    sentiment_batch: np.ndarray,
    feature_names: List[str] | None = None,
) -> np.ndarray:
    return flatten_intraday_batch(windows, session_batch, sentiment_batch)
