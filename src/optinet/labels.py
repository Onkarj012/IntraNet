from __future__ import annotations

import numpy as np
import pandas as pd

from optinet.config import PROFILE_SPECS, RiskProfileSpec


def _profile_targets(index_bars: pd.DataFrame, profile: RiskProfileSpec) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index_name, daily in index_bars.sort_values(["index", "date"]).groupby("index"):
        daily = daily.set_index("date").sort_index()
        next_open = daily["open"].shift(-1)
        next_high = daily["high"].shift(-1)
        next_low = daily["low"].shift(-1)
        next_close = daily["close"].shift(-1)
        up_move = (next_high - next_open) / next_open.replace(0, np.nan)
        down_move = (next_open - next_low) / next_open.replace(0, np.nan)
        close_ret = (next_close - next_open) / next_open.replace(0, np.nan)
        long_ok = (up_move >= profile.target_pct + profile.cost_buffer_pct) & (
            up_move - down_move >= profile.cost_buffer_pct
        )
        short_ok = (down_move >= profile.target_pct + profile.cost_buffer_pct) & (
            down_move - up_move >= profile.cost_buffer_pct
        )
        trade_label = pd.Series("NO_TRADE", index=daily.index, dtype="object")
        trade_label.loc[long_ok & ~short_ok] = "LONG"
        trade_label.loc[short_ok & ~long_ok] = "SHORT"
        target = pd.DataFrame(
            {
                "index": index_name,
                "date": daily.index,
                f"{profile.name}_long_label": (trade_label == "LONG").astype(int),
                f"{profile.name}_short_label": (trade_label == "SHORT").astype(int),
                f"{profile.name}_trade_label": trade_label,
                "up_magnitude": up_move.clip(lower=0),
                "down_magnitude": down_move.clip(lower=0),
                "next_open": next_open,
                "next_close": next_close,
                "next_close_return": close_ret,
            }
        )
        rows.append(target.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_labels(index_bars: pd.DataFrame) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for profile in PROFILE_SPECS.values():
        labels = _profile_targets(index_bars, profile)
        if merged is None:
            merged = labels
        else:
            cols = ["index", "date", f"{profile.name}_long_label", f"{profile.name}_short_label", f"{profile.name}_trade_label"]
            merged = merged.merge(labels[cols], on=["index", "date"], how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["next_open"])


def merge_features_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    frame = features.merge(labels, on=["index", "date"], how="inner")
    return frame.sort_values(["index", "date"]).reset_index(drop=True)
