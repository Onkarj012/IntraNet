"""Regime classifier for OptiNet.

Computes per (index, date) a feature set and a hard-filter flag used to skip
trading on volatile or extreme days. Uses:
  - india_vix from market_data_cache/india_vix.csv (cached daily series)
  - index OHLC for gap, ATR, trend
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Default thresholds chosen to match the 2026 March crash analysis: skipping
# `india_vix > 20` AND large gaps would have removed ~80% of stop-outs.
DEFAULT_VIX_THRESHOLD = 20.0
DEFAULT_GAP_THRESHOLD = 0.008
DEFAULT_ATR_RATIO_THRESHOLD = 1.5

DEFAULT_VIX_PATH = Path("market_data_cache/india_vix.csv")


def _load_vix(vix_path: str | Path = DEFAULT_VIX_PATH) -> pd.Series:
    path = Path(vix_path)
    if not path.exists():
        return pd.Series(dtype=float, name="india_vix")
    df = pd.read_csv(path)
    if df.empty:
        return pd.Series(dtype=float, name="india_vix")

    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime"}), df.columns[0])
    close_col = next((c for c in df.columns if c.lower() in {"close", "adj_close", "close_price", "value"}),
                      df.columns[-1])
    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    daily = df.groupby("date")["close"].last().rename("india_vix")
    return daily


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window // 2).mean()
    return atr / close.replace(0, np.nan)


def _classify_regime(row: pd.Series, vix_threshold: float, gap_threshold: float,
                     atr_ratio_threshold: float) -> str:
    vix = row.get("india_vix", np.nan)
    gap = abs(row.get("gap_pct", 0.0))
    atr_ratio = row.get("atr_ratio_60d", 1.0)
    trend = row.get("trend_regime", 0.0)

    if (pd.notna(vix) and vix > vix_threshold * 1.4) or atr_ratio > 2.0:
        return "crash"
    if (pd.notna(vix) and vix > vix_threshold) or gap > gap_threshold or atr_ratio > atr_ratio_threshold:
        return "volatile"
    if abs(trend) >= 0.5:
        return "trend"
    return "calm"


def compute_regime(
    index_bars: pd.DataFrame,
    *,
    vix_path: str | Path = DEFAULT_VIX_PATH,
    vix_threshold: float = DEFAULT_VIX_THRESHOLD,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    atr_ratio_threshold: float = DEFAULT_ATR_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """Per (index, date) regime features + hard-filter flag.

    Output columns:
        index, date,
        india_vix, vix_change_5d, vix_percentile_60d,
        gap_pct, abs_gap_pct,
        atr_14_pct, atr_ratio_60d,
        trend_regime, regime_label, regime_block
    """
    if index_bars.empty:
        return pd.DataFrame()

    vix = _load_vix(vix_path)

    frames: list[pd.DataFrame] = []
    for idx_name, df in index_bars.sort_values(["index", "date"]).groupby("index"):
        df = df.set_index("date").sort_index()
        out = pd.DataFrame(index=df.index)
        out["index"] = idx_name
        out["date"] = df.index
        out["gap_pct"] = df["open"] / df["close"].shift(1) - 1.0
        out["abs_gap_pct"] = out["gap_pct"].abs()
        out["atr_14_pct"] = _atr_pct(df["high"], df["low"], df["close"], 14)
        out["atr_ratio_60d"] = (out["atr_14_pct"] /
                                out["atr_14_pct"].rolling(60, min_periods=20).median())
        ema20 = df["close"].ewm(span=20, adjust=False, min_periods=10).mean()
        ema50 = df["close"].ewm(span=50, adjust=False, min_periods=20).mean()
        out["trend_regime"] = np.sign(ema20 - ema50)
        # Join VIX (one series shared across indices)
        out["india_vix"] = out.index.map(vix.to_dict())
        out["vix_change_5d"] = out["india_vix"].pct_change(5)
        out["vix_percentile_60d"] = out["india_vix"].rolling(60, min_periods=20) \
                                     .rank(pct=True)
        frames.append(out.reset_index(drop=True))

    regime = pd.concat(frames, ignore_index=True)

    # Classification + hard filter
    regime["regime_label"] = regime.apply(
        lambda r: _classify_regime(r, vix_threshold, gap_threshold, atr_ratio_threshold),
        axis=1,
    )
    regime["regime_block"] = regime["regime_label"].isin(["crash", "volatile"])

    cols = ["index", "date", "india_vix", "vix_change_5d", "vix_percentile_60d",
            "gap_pct", "abs_gap_pct", "atr_14_pct", "atr_ratio_60d",
            "trend_regime", "regime_label", "regime_block"]
    return regime[cols].replace([np.inf, -np.inf], np.nan)
