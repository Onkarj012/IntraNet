"""
Thresholded target construction for IntradayNet LightGBM V2.

Computes direction and magnitude targets with:
- Move threshold: filters out noise (< 0.3% moves are not tradeable)
- Cost adjustment: subtracts round-trip transaction costs
- Valid mask: excludes bars without enough future data
- Clipping: bounds extreme moves at ±5%

Usage:
    targets = compute_targets(df, config=TargetConfig())
    stats = get_target_stats(targets)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

from intradaynet.costs import DEFAULT_COSTS, estimate_liquidity_penalty


HORIZONS = {
    "H15": 15,
    "H30": 30,
    "H60": 60,
    "H375": 375,
}


@dataclass
class TargetConfig:
    horizons: dict = field(default_factory=lambda: HORIZONS)
    move_threshold: float = 0.003
    magnitude_clip: float = 0.05
    cost_adjustment: float = 0.001
    sample_interval: int = 15
    min_session_bars: int = 120
    position_value: float = 100_000.0
    liquidity_penalty_floor: float = 0.0001


def compute_targets(
    df: pd.DataFrame,
    config: TargetConfig = TargetConfig(),
) -> pd.DataFrame:
    """
    Compute direction and magnitude targets for each horizon.

    Args:
        df: DataFrame with Close, DatetimeIndex, date column
        config: Target configuration

    Returns:
        DataFrame with columns:
            dir_H*    : direction labels (0/1 or NaN)
            gross_H*  : clipped signed forward return
            edge_H*   : signed forward return after cost/liquidity penalties
            mag_H*    : backwards-compatible alias of gross_H*
            valid_H*  : valid direction labels
    """
    targets = pd.DataFrame(index=df.index)
    close_col = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
    volume_col = "Volume" if "Volume" in df.columns else "volume" if "volume" in df.columns else None

    if close_col is None:
        raise KeyError("compute_targets expects a 'close' or 'Close' column")

    close = df[close_col].values

    if "date" in df.columns:
        dates = df["date"].values
    else:
        dates = np.array([d.date() for d in df.index])

    unique_dates = np.unique(dates)

    for name, bars in config.horizons.items():
        direction = np.full(len(df), np.nan, dtype=np.float32)
        gross_return = np.zeros(len(df), dtype=np.float32)
        net_edge = np.zeros(len(df), dtype=np.float32)
        valid = np.zeros(len(df), dtype=bool)

        for date in unique_dates:
            session_mask = dates == date
            session_indices = np.where(session_mask)[0]
            n_bars = len(session_indices)

            if n_bars < config.min_session_bars:
                continue

            session_close = close[session_indices]

            for i, idx in enumerate(session_indices):
                bar_pos = i
                future_pos = bar_pos + bars

                if future_pos >= n_bars:
                    continue

                anchor = session_close[bar_pos]
                if anchor <= 0:
                    continue

                future_close = session_close[future_pos]

                raw_return = (future_close - anchor) / anchor

                round_trip_cost = DEFAULT_COSTS.estimate_round_trip_fraction(
                    entry_price=anchor,
                    position_value=config.position_value,
                )

                if volume_col is not None:
                    session_volume = df[volume_col].values[session_indices]
                    avg_daily_traded_value = float(np.mean(session_close * session_volume))
                    median_minute_turnover = float(np.median(session_close * session_volume))
                else:
                    avg_daily_traded_value = 0.0
                    median_minute_turnover = 0.0

                liquidity_penalty = estimate_liquidity_penalty(
                    avg_daily_traded_value=avg_daily_traded_value,
                    median_minute_turnover=median_minute_turnover,
                )
                total_penalty = max(
                    round_trip_cost + liquidity_penalty,
                    config.cost_adjustment + config.liquidity_penalty_floor,
                )

                net_return = raw_return - total_penalty if raw_return > 0 else raw_return + total_penalty
                net_return = abs(net_return) * np.sign(raw_return) if raw_return != 0 else 0.0

                gross_return[idx] = np.clip(
                    raw_return,
                    -config.magnitude_clip,
                    config.magnitude_clip,
                )
                net_edge[idx] = np.clip(
                    net_return,
                    -config.magnitude_clip,
                    config.magnitude_clip,
                )

                if net_return > config.move_threshold:
                    direction[idx] = 1.0
                    valid[idx] = True
                elif net_return < -config.move_threshold:
                    direction[idx] = 0.0
                    valid[idx] = True

        targets[f"dir_{name}"] = direction
        targets[f"gross_{name}"] = gross_return
        targets[f"edge_{name}"] = net_edge
        targets[f"mag_{name}"] = gross_return
        targets[f"valid_{name}"] = valid

    return targets


def get_target_stats(targets: pd.DataFrame) -> Dict:
    """Print and return target distribution statistics."""
    stats = {}
    for h in HORIZONS:
        valid = targets[f"valid_{h}"]
        dirs = targets.loc[valid, f"dir_{h}"]
        mags = targets.loc[valid, f"mag_{h}"]

        n_up = (dirs == 1.0).sum() if len(dirs) > 0 else 0
        n_down = (dirs == 0.0).sum() if len(dirs) > 0 else 0
        n_total = valid.sum()

        stats[h] = {
            "total_samples": int(n_total),
            "pct_up": float(n_up / max(n_total, 1) * 100),
            "pct_down": float(n_down / max(n_total, 1) * 100),
            "mean_magnitude": float(mags.abs().mean() * 100) if len(mags) > 0 else 0.0,
            "median_magnitude": float(mags.abs().median() * 100) if len(mags) > 0 else 0.0,
            "n_up": int(n_up),
            "n_down": int(n_down),
        }

    return stats


def print_target_stats(stats: Dict):
    """Print formatted target statistics."""
    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION STATISTICS")
    print("=" * 60)
    for h, s in stats.items():
        print(f"\n{h}:")
        print(f"  Samples:      {s['total_samples']:,}")
        print(f"  Up/Down:      {s['pct_up']:.1f}% / {s['pct_down']:.1f}%")
        print(f"  Mean |move|: {s['mean_magnitude']:.2f}%")
        print(f"  Median |move|: {s['median_magnitude']:.2f}%")
    print("=" * 60)
