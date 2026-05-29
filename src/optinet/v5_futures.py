"""V5 futures feature loader.

Computes 8 minute-level features from intraday futures CSVs at
data/option_data/{nifty,banknifty}_data/{nifty,banknifty}_fut/*.csv.

Schema of input: date,time,symbol,open,high,low,close,oi,volume

Features per minute (after merging with spot):
  fut_basis                    = (fut_close - spot) / spot
  fut_basis_change_30m         = fut_basis(t) - fut_basis(t-30)
  fut_oi_change_1m             = (oi(t) - oi(t-1)) / max(oi(t-1), 1)
  fut_oi_change_5m             = (oi(t) - oi(t-5)) / max(oi(t-5), 1)
  fut_oi_change_30m            = (oi(t) - oi(t-30)) / max(oi(t-30), 1)
  fut_volume_oi_ratio          = vol(t) / max(oi(t), 1)
  fut_session_position         = (close - low_so_far) / (high_so_far - low_so_far + eps)
  fut_oi_x_price_long_buildup  = 1 if (price up AND oi up) else 0  (over 5-min window)
  fut_oi_x_price_short_buildup = 1 if (price down AND oi up) else 0
  fut_oi_x_price_short_cover   = 1 if (price up AND oi down) else 0
  fut_oi_x_price_long_unwind   = 1 if (price down AND oi down) else 0
"""
from __future__ import annotations

from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd


def discover_fut_days(root: Path, index_name: str) -> list[tuple[date_cls, Path]]:
    """Find all (date, fut_csv_path) for one index across years."""
    if index_name == "NIFTY":
        fut_root = root / "nifty_data/nifty_fut"
        prefix = "nifty_fut_"
    elif index_name == "BANKNIFTY":
        fut_root = root / "banknifty_data/banknifty_fut"
        prefix = "banknifty_fut_"
    else:
        raise ValueError(f"Unknown index: {index_name}")

    out: list[tuple[date_cls, Path]] = []
    for f in sorted(fut_root.rglob(f"{prefix}*.csv")):
        stem = f.stem.replace(prefix, "")
        try:
            dd, mm, yyyy = stem.split("_")
            d = date_cls(int(yyyy), int(mm), int(dd))
        except (ValueError, AttributeError):
            continue
        out.append((d, f))
    return out


def load_fut_day(path: Path, index_name: str) -> pd.DataFrame:
    """Load one day's intraday futures with the standard schema."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.rename(columns={"close": "fut_close", "oi": "fut_oi", "volume": "fut_vol"})
    df["index"] = index_name
    return df[["index", "datetime", "fut_close", "fut_oi", "fut_vol"]] \
        .sort_values("datetime").reset_index(drop=True)


def compute_fut_features(fut_day: pd.DataFrame, spot_day: pd.DataFrame) -> pd.DataFrame:
    """Merge fut + spot, compute 11 minute-level futures features."""
    if fut_day.empty or spot_day.empty:
        return pd.DataFrame()

    spot = spot_day[["datetime", "spot"]].copy()
    df = fut_day.merge(spot, on="datetime", how="inner")
    if df.empty:
        return df

    df = df.sort_values("datetime").reset_index(drop=True)

    # Basis
    df["fut_basis"] = (df["fut_close"] - df["spot"]) / df["spot"]
    df["fut_basis_change_30m"] = df["fut_basis"] - df["fut_basis"].shift(30)

    # OI deltas
    df["fut_oi_change_1m"] = df["fut_oi"].diff(1) / df["fut_oi"].shift(1).replace(0, np.nan)
    df["fut_oi_change_5m"] = df["fut_oi"].diff(5) / df["fut_oi"].shift(5).replace(0, np.nan)
    df["fut_oi_change_30m"] = df["fut_oi"].diff(30) / df["fut_oi"].shift(30).replace(0, np.nan)

    # Volume / OI
    df["fut_volume_oi_ratio"] = df["fut_vol"] / df["fut_oi"].replace(0, np.nan)

    # Session position (rolling high/low so-far)
    df["fut_session_high"] = df["fut_close"].cummax()
    df["fut_session_low"] = df["fut_close"].cummin()
    rng = df["fut_session_high"] - df["fut_session_low"]
    df["fut_session_position"] = (df["fut_close"] - df["fut_session_low"]) / rng.replace(0, np.nan)

    # OI×price interaction over 5m window
    price_up = (df["fut_close"].diff(5) > 0)
    oi_up = (df["fut_oi"].diff(5) > 0)
    df["fut_oi_x_long_buildup"]  = (price_up & oi_up).astype(np.int8)
    df["fut_oi_x_short_buildup"] = (~price_up & oi_up).astype(np.int8)
    df["fut_oi_x_short_cover"]   = (price_up & ~oi_up).astype(np.int8)
    df["fut_oi_x_long_unwind"]   = (~price_up & ~oi_up).astype(np.int8)

    out_cols = ["index", "datetime",
                "fut_basis", "fut_basis_change_30m",
                "fut_oi_change_1m", "fut_oi_change_5m", "fut_oi_change_30m",
                "fut_volume_oi_ratio", "fut_session_position",
                "fut_oi_x_long_buildup", "fut_oi_x_short_buildup",
                "fut_oi_x_short_cover", "fut_oi_x_long_unwind"]
    return df[out_cols].copy()


def build_fut_day(fut_path: Path, spot_path: Path, index_name: str) -> pd.DataFrame:
    """End-to-end: load fut + spot, compute features, return 11-column frame."""
    from optinet.v4_chain import load_spot

    fut_day = load_fut_day(fut_path, index_name)
    spot_day = load_spot(spot_path, index_name)
    return compute_fut_features(fut_day, spot_day)
