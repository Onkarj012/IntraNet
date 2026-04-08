"""
Feature name utilities for IntradayNet LightGBM V2.

Reconstructs the 625 feature names used in prebatching.
Naming scheme from flatten_windows_vectorized in prebatch_lgbm_v2.py:

  - 5 windows × 4 stats × 25 per-bar = 500 features
  - last bar: 25 features
  - first bar: 25 features
  - diff (last - first): 25 features
  - diff_5v30: 25 features
  - diff_15v60: 25 features
  Total: 625 features
"""

from intradaynet.features.per_bar_features import PER_BAR_FEATURE_NAMES

PER_BAR_NAMES = PER_BAR_FEATURE_NAMES

WINDOWS = [5, 15, 30, 60, 120]
STATS = ["mean", "std", "min", "max"]


def get_flat_feature_names() -> list[str]:
    """Generate the 625 feature names used in prebatching."""
    names = []

    for w in WINDOWS:
        for stat in STATS:
            for pb in PER_BAR_NAMES:
                names.append(f"{pb}_w{w}_{stat}")

    for pb in PER_BAR_NAMES:
        names.append(f"{pb}_last")

    for pb in PER_BAR_NAMES:
        names.append(f"{pb}_first")

    for pb in PER_BAR_NAMES:
        names.append(f"{pb}_diff")

    for pb in PER_BAR_NAMES:
        names.append(f"{pb}_diff_5v30")

    for pb in PER_BAR_NAMES:
        names.append(f"{pb}_diff_15v60")

    assert len(names) == 625, f"Expected 625 features, got {len(names)}"
    return names


FEATURE_NAMES = get_flat_feature_names()
