"""
Per-bar feature engineering for intraday minute data.

Computes 25 features per minute bar from raw OHLCV data.
All computations are vectorized using pandas/numpy.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("intradaynet.features.per_bar")

# Ordered list of all per-bar feature names (25 total)
PER_BAR_FEATURE_NAMES = [
    "log_return",
    "volume_ratio",
    "vwap_distance",
    "ema_9_distance",
    "ema_20_distance",
    "rsi_14",
    "bb_zscore",
    "bb_width",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "spread_pct",
    "volume_pace",
    "time_normalized",
    "orb_high_dist",
    "orb_low_dist",
    "day_return",
    "momentum_5",
    "momentum_20",
    "vol_momentum",
    "atr_14",
    "close_vs_running_range",
    "session_volatility",
    "obv_slope",
    "trade_intensity",
]


def compute_per_bar_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 25 per-bar features from raw minute OHLCV data.

    Args:
        minute_df: DataFrame with columns [open, high, low, close, volume]
                   and DatetimeIndex. Should be filtered to market hours.

    Returns:
        DataFrame with 25 feature columns, same index as input.
    """
    df = minute_df.copy()

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # Session boundaries (group by date)
    df["_date"] = df.index.date

    features = pd.DataFrame(index=df.index)

    # ── 1. log_return ──
    price_ratio = (c / c.shift(1).replace(0, np.nan)).fillna(1.0).clip(lower=1e-10)
    features["log_return"] = np.log(price_ratio).fillna(0.0)

    # ── 2. volume_ratio: Volume_t / EMA_20(Volume) ──
    vol_ema20 = v.ewm(span=20, min_periods=1).mean().replace(0, 1)
    features["volume_ratio"] = (v / vol_ema20).clip(0, 50)

    # ── 3. vwap_distance: (Close - cumulative_VWAP) / VWAP ──
    tp = (h + l + c) / 3.0
    tpv = tp * v
    cum_tpv = tpv.groupby(df["_date"]).cumsum()
    cum_vol = v.groupby(df["_date"]).cumsum().replace(0, 1)
    vwap = cum_tpv / cum_vol
    features["vwap_distance"] = ((c - vwap) / vwap.replace(0, 1)).clip(-5, 5)

    # ── 4-5. EMA distances ──
    ema_9 = c.ewm(span=9, min_periods=1).mean()
    ema_20 = c.ewm(span=20, min_periods=1).mean()
    features["ema_9_distance"] = ((c - ema_9) / ema_9.replace(0, 1)).clip(-5, 5)
    features["ema_20_distance"] = ((c - ema_20) / ema_20.replace(0, 1)).clip(-5, 5)

    # ── 6. RSI-14 ──
    features["rsi_14"] = _compute_rsi(c, period=14) / 100.0  # normalize to [0, 1]

    # ── 7-8. Bollinger Band features ──
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std().fillna(1e-10)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_half_width = (bb_upper - bb_mid).replace(0, 1e-10)
    features["bb_zscore"] = ((c - bb_mid) / bb_half_width).clip(-5, 5)
    features["bb_width"] = ((bb_upper - bb_lower) / bb_mid.replace(0, 1)).clip(0, 5)

    # ── 9-11. Candle features ──
    hl_range = (h - l).replace(0, 1e-10)
    features["body_ratio"] = (abs(c - o) / hl_range).clip(0, 1)
    features["upper_shadow_ratio"] = ((h - np.maximum(o, c)) / hl_range).clip(0, 1)
    features["lower_shadow_ratio"] = ((np.minimum(o, c) - l) / hl_range).clip(0, 1)

    # ── 12. spread_pct: (H-L)/C — intrabar volatility ──
    features["spread_pct"] = ((h - l) / c.replace(0, 1)).clip(0, 0.5)

    # ── 13. volume_pace: current volume / rolling 30-bar avg volume ──
    # FIXED: Was cum_volume_pct which used full-session volume (LEAK).
    # Now uses rolling volume pace — purely causal, no future data.
    vol_roll_avg = v.rolling(30, min_periods=5).mean().replace(0, 1)
    features["volume_pace"] = (v / vol_roll_avg).clip(0, 50)

    # ── 14. time_normalized: position in session [0, 1] ──
    bar_in_session = v.groupby(df["_date"]).cumcount()
    session_len = v.groupby(df["_date"]).transform("count").replace(0, 1)
    features["time_normalized"] = (bar_in_session / session_len).clip(0, 1)

    # ── 15-16. ORB (Opening Range Breakout) features ──
    # ORB = first 15 minutes high/low
    orb_high, orb_low = _compute_orb(df, n_bars=15)
    orb_range = (orb_high - orb_low).replace(0, 1e-10)
    features["orb_high_dist"] = ((c - orb_high) / orb_range).clip(-10, 10)
    features["orb_low_dist"] = ((c - orb_low) / orb_range).clip(-10, 10)

    # ── 17. day_return ──
    day_open = o.groupby(df["_date"]).transform("first").replace(0, 1)
    features["day_return"] = ((c - day_open) / day_open).clip(-0.2, 0.2)

    # ── 18-19. Momentum ──
    features["momentum_5"] = (c / c.shift(5).replace(0, 1) - 1).clip(-0.1, 0.1).fillna(0)
    features["momentum_20"] = (c / c.shift(20).replace(0, 1) - 1).clip(-0.2, 0.2).fillna(0)

    # ── 20. vol_momentum: (up_volume - down_volume) / total_volume over 20 bars ──
    close_diff = c.diff()
    up_vol = v.where(close_diff > 0, 0)
    dn_vol = v.where(close_diff < 0, 0)
    up_vol_20 = up_vol.rolling(20, min_periods=1).sum()
    dn_vol_20 = dn_vol.rolling(20, min_periods=1).sum()
    total_vol_20 = v.rolling(20, min_periods=1).sum().replace(0, 1)
    features["vol_momentum"] = ((up_vol_20 - dn_vol_20) / total_vol_20).clip(-1, 1)

    # ── 21. atr_14: normalized ATR ──
    features["atr_14"] = _compute_atr(h, l, c, period=14)

    # ── 22. close_vs_running_range ──
    # FIXED: Was close_vs_day_range which used cummax/cummin (LEAK).
    # Now uses expanding min/max — only past+current bars within session.
    day_high = h.groupby(df["_date"]).transform(lambda x: x.expanding().max())
    day_low = l.groupby(df["_date"]).transform(lambda x: x.expanding().min())
    day_range = (day_high - day_low).replace(0, 1e-10)
    features["close_vs_running_range"] = ((c - day_low) / day_range).clip(0, 1)

    # ── 23. session_volatility: rolling std of log returns ──
    features["session_volatility"] = (
        features["log_return"].rolling(20, min_periods=1).std().fillna(0).clip(0, 0.1)
    )

    # ── 24. obv_slope: linear regression slope of OBV over 20 bars ──
    features["obv_slope"] = _compute_obv_slope(c, v, window=20)

    # ── 25. trade_intensity: Volume × spread ──
    trade_int = v * (h - l)
    trade_int_mean = trade_int.rolling(20, min_periods=1).mean().replace(0, 1)
    features["trade_intensity"] = (trade_int / trade_int_mean).clip(0, 50)

    # Fill remaining NaNs
    features = features.fillna(0.0)

    # Final clip for safety
    features = features.clip(-50, 50)

    logger.debug(f"Computed {len(features.columns)} per-bar features for {len(features)} bars")
    return features[PER_BAR_FEATURE_NAMES]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
    """Compute normalized ATR (ATR / Close)."""
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=1).mean()
    return (atr / close.replace(0, 1)).clip(0, 0.5).fillna(0)


def _compute_orb(df: pd.DataFrame, n_bars: int = 15):
    """
    Compute Opening Range Breakout high/low (first N bars of each session).
    Returns two Series aligned to df index.
    """
    grouped = df.groupby("_date")

    def _orb_high(g):
        orb_h = g["high"].iloc[:n_bars].max()
        return pd.Series(orb_h, index=g.index)

    def _orb_low(g):
        orb_l = g["low"].iloc[:n_bars].min()
        return pd.Series(orb_l, index=g.index)

    orb_high = grouped.apply(_orb_high, include_groups=False).droplevel(0)
    orb_low = grouped.apply(_orb_low, include_groups=False).droplevel(0)
    return orb_high, orb_low


def _compute_obv_slope(close: pd.Series, volume: pd.Series,
                       window: int = 20) -> pd.Series:
    """Compute linear regression slope of OBV over a rolling window.
    Uses vectorized rolling covariance for O(n) performance."""
    # OBV
    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * volume).cumsum()

    # Rolling linear regression slope via vectorized formula:
    # slope = Cov(x, y) / Var(x) where x = [0..w-1]
    # For fixed x window, Var(x) is constant = w*(w-1)/12 * (w+1)... simplified:
    # Use rolling correlation between index and OBV
    x = pd.Series(np.arange(len(obv), dtype=np.float64), index=obv.index)

    # Rolling means
    obv_mean = obv.rolling(window, min_periods=window).mean()
    x_mean = x.rolling(window, min_periods=window).mean()

    # Rolling covariance and variance
    xy = (x * obv).rolling(window, min_periods=window).mean() - x_mean * obv_mean
    xx = (x * x).rolling(window, min_periods=window).mean() - x_mean * x_mean

    slope = xy / xx.replace(0, 1e-10)

    # Normalize
    slope_std = slope.rolling(100, min_periods=1).std().replace(0, 1)
    return (slope / slope_std).clip(-5, 5).fillna(0)
