"""
Smart subsampling for IntradayNet LightGBM V2.

Keeps ALL extreme-move samples (the signal) while subsampling
the middle 60% (the noise). Balances up/down classes.

Strategy:
1. Keep ALL samples in top/bottom 20% by magnitude
2. Randomly sample from the middle 60%
3. Balance up/down classes

Usage:
    X_sub, y_dir_sub, y_mag_sub = smart_subsample(
        X, y_dir, y_mag, valid_mask,
        max_samples=2_000_000,
        extreme_percentile=80
    )
"""

import numpy as np
from typing import Tuple


def smart_subsample(
    X: np.ndarray,
    y_dir: np.ndarray,
    y_mag: np.ndarray,
    valid_mask: np.ndarray,
    max_samples: int = 2_000_000,
    extreme_percentile: float = 80,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smart subsample preserving extreme moves.

    Args:
        X: Feature matrix (N, F)
        y_dir: Direction labels (N,) — 0/1 or NaN
        y_mag: Magnitude values (N,) — float
        valid_mask: Boolean mask of valid samples (N,)
        max_samples: Maximum number of samples to return
        extreme_percentile: Keep samples above this percentile of |magnitude|
        seed: Random seed

    Returns:
        X_sub, y_dir_sub, y_mag_sub — subsampled arrays
    """
    rng = np.random.RandomState(seed)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return X, y_dir, y_mag

    X_valid = X[valid_idx]
    y_dir_valid = y_dir[valid_idx]
    y_mag_valid = y_mag[valid_idx]

    abs_mag = np.abs(y_mag_valid)
    threshold = np.percentile(abs_mag[~np.isnan(abs_mag)], extreme_percentile)

    extreme_mask = abs_mag >= threshold
    extreme_idx = valid_idx[np.where(extreme_mask)[0]]

    normal_mask = ~extreme_mask
    normal_idx = valid_idx[np.where(normal_mask)[0]]

    n_extreme = len(extreme_idx)
    n_normal_budget = max_samples - n_extreme

    if n_normal_budget <= 0:
        final_idx = extreme_idx
    else:
        up_normal = normal_idx[y_dir_valid[np.isin(valid_idx, normal_idx)] == 1]
        down_normal = normal_idx[y_dir_valid[np.isin(valid_idx, normal_idx)] == 0]

        n_up = min(len(up_normal), n_normal_budget // 2)
        n_down = min(len(down_normal), n_normal_budget // 2)

        sampled_up = rng.choice(up_normal, size=n_up, replace=False) if len(up_normal) > 0 else np.array([], dtype=int)
        sampled_down = rng.choice(down_normal, size=n_down, replace=False) if len(down_normal) > 0 else np.array([], dtype=int)
        sampled_normal = np.concatenate([sampled_up, sampled_down])

        final_idx = np.concatenate([extreme_idx, sampled_normal])

    rng.shuffle(final_idx)

    valid_mask_final = np.isin(valid_idx, final_idx)
    X_sub = X_valid[valid_mask_final]
    y_dir_sub = y_dir_valid[valid_mask_final]
    y_mag_sub = y_mag_valid[valid_mask_final]

    up_count = np.sum(y_dir_sub == 1)
    down_count = np.sum(y_dir_sub == 0)
    n_total = len(y_dir_sub)

    print(f"Subsampling: {len(valid_idx):,} → {n_total:,} samples")
    print(f"  Extreme (top {100 - extreme_percentile:.0f}%): {n_extreme:,}")
    print(f"  Normal (sampled): {n_total - n_extreme:,}")
    print(f"  Class balance: {up_count/n_total*100:.1f}% up / {down_count/n_total*100:.1f}% down")

    return X_sub, y_dir_sub, y_mag_sub


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 500_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple stratified subsample balancing classes.

    Args:
        X: Feature matrix
        y: Labels (0/1)
        max_samples: Maximum samples
        seed: Random seed

    Returns:
        X_sub, y_sub
    """
    rng = np.random.RandomState(seed)

    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    n_0 = min(len(idx_0), max_samples // 2)
    n_1 = min(len(idx_1), max_samples // 2)

    sampled_0 = rng.choice(idx_0, size=n_0, replace=False)
    sampled_1 = rng.choice(idx_1, size=n_1, replace=False)

    final_idx = np.concatenate([sampled_0, sampled_1])
    rng.shuffle(final_idx)

    return X[final_idx], y[final_idx]
