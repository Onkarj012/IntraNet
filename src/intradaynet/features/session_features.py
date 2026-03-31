"""
Session-level context features for intraday predictions.

Computes 20 features per trading session from daily data and
the first bars of the current session. These are static within
a session (same value for every minute bar in a given day).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("intradaynet.features.session_features")

SESSION_FEATURE_NAMES = [
    "prev_day_rsi",
    "prev_day_macd",
    "prev_day_macd_hist",
    "prev_day_bb_zscore",
    "prev_day_trend_strength",
    "prev_day_regime",
    "prev_day_volatility_21",
    "prev_day_adx",
    "overnight_return",
    "gap_size",
    "gap_direction",
    "prev_day_close_location",
    "prev_day_volume_zscore",
    "day_of_week",
    "is_expiry_week",
    "is_monthly_expiry",
    "is_result_season",
    "days_since_52w_high",
    "days_since_52w_low",
    "avg_intraday_range",
]


def compute_session_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session-level context features from minute data.

    For each trading day, computes 20 context features from previous-day
    aggregates and current-day session start information.

    Args:
        minute_df: DataFrame with columns [open, high, low, close, volume]
                   and DatetimeIndex, filtered to market hours.

    Returns:
        DataFrame indexed by date with 20 feature columns.
        One row per trading session.
    """
    df = minute_df.copy()
    df.columns = df.columns.str.lower()
    df["_date"] = df.index.date

    # Build daily OHLCV from minute data
    grouped = df.groupby("_date")
    daily = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "volume": grouped["volume"].sum().astype(float),
    })
    daily.index = pd.to_datetime(daily.index)

    features = pd.DataFrame(index=daily.index)

    c = daily["close"]
    h = daily["high"]
    l = daily["low"]
    o = daily["open"]
    v = daily["volume"]

    # ── 1. prev_day_rsi (14-day RSI) ──
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features["prev_day_rsi"] = (rsi.shift(1).fillna(50) / 100.0)  # normalize to [0,1]

    # ── 2-3. prev_day_macd, prev_day_macd_hist ──
    ema12 = c.ewm(span=12, min_periods=1).mean()
    ema26 = c.ewm(span=26, min_periods=1).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, min_periods=1).mean()
    macd_hist = macd_line - signal_line
    # Normalize by price
    features["prev_day_macd"] = (macd_line.shift(1) / c.shift(1).replace(0, 1)).fillna(0).clip(-0.1, 0.1)
    features["prev_day_macd_hist"] = (macd_hist.shift(1) / c.shift(1).replace(0, 1)).fillna(0).clip(-0.1, 0.1)

    # ── 4. prev_day_bb_zscore ──
    bb_mid = c.rolling(20, min_periods=1).mean()
    bb_std = c.rolling(20, min_periods=1).std().fillna(1e-10)
    bb_half = (2 * bb_std).replace(0, 1e-10)
    features["prev_day_bb_zscore"] = ((c.shift(1) - bb_mid.shift(1)) / bb_half.shift(1)).fillna(0).clip(-3, 3)

    # ── 5. prev_day_trend_strength: (close - SMA20) / SMA20 ──
    sma20 = c.rolling(20, min_periods=1).mean()
    features["prev_day_trend_strength"] = ((c.shift(1) - sma20.shift(1)) / sma20.shift(1).replace(0, 1)).fillna(0).clip(-0.5, 0.5)

    # ── 6. prev_day_regime: -1 (bear), 0 (sideways), 1 (bull) ──
    sma50 = c.rolling(50, min_periods=1).mean()
    sma200 = c.rolling(200, min_periods=1).mean()
    regime = np.where(
        c.shift(1) > sma50.shift(1), 1,
        np.where(c.shift(1) < sma200.shift(1), -1, 0)
    )
    features["prev_day_regime"] = pd.Series(regime, index=daily.index).astype(float)

    # ── 7. prev_day_volatility_21: 21-day rolling std of returns ──
    daily_ret = c.pct_change()
    features["prev_day_volatility_21"] = daily_ret.rolling(21, min_periods=1).std().shift(1).fillna(0).clip(0, 0.2)

    # ── 8. prev_day_adx (simplified — use DI spread as proxy) ──
    plus_dm = (h.diff()).where(h.diff() > l.diff().abs(), 0).clip(lower=0)
    minus_dm = (l.diff().abs()).where(l.diff().abs() > h.diff(), 0).clip(lower=0)
    atr = _true_range(h, l, c).rolling(14, min_periods=1).mean()
    plus_di = (plus_dm.rolling(14, min_periods=1).mean() / atr.replace(0, 1)) * 100
    minus_di = (minus_dm.rolling(14, min_periods=1).mean() / atr.replace(0, 1)) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)) * 100
    adx = dx.rolling(14, min_periods=1).mean()
    features["prev_day_adx"] = (adx.shift(1).fillna(25) / 100.0).clip(0, 1)

    # ── 9-11. Overnight / Gap features ──
    prev_close = c.shift(1)
    features["overnight_return"] = ((o - prev_close) / prev_close.replace(0, 1)).fillna(0).clip(-0.1, 0.1)
    features["gap_size"] = features["overnight_return"].abs()
    features["gap_direction"] = np.sign(features["overnight_return"])

    # ── 12. prev_day_close_location ──
    prev_range = (h.shift(1) - l.shift(1)).replace(0, 1e-10)
    features["prev_day_close_location"] = ((c.shift(1) - l.shift(1)) / prev_range).fillna(0.5).clip(0, 1)

    # ── 13. prev_day_volume_zscore ──
    vol_mean = v.rolling(20, min_periods=1).mean()
    vol_std = v.rolling(20, min_periods=1).std().replace(0, 1)
    features["prev_day_volume_zscore"] = ((v.shift(1) - vol_mean.shift(1)) / vol_std.shift(1)).fillna(0).clip(-3, 3)

    # ── 14. day_of_week (0=Mon, 4=Fri) ──
    features["day_of_week"] = daily.index.dayofweek.astype(float) / 4.0  # normalize to [0, 1]

    # ── 15-16. Expiry flags ──
    features["is_expiry_week"] = _is_expiry_week(daily.index).astype(float)
    features["is_monthly_expiry"] = _is_monthly_expiry(daily.index).astype(float)

    # ── 17. is_result_season (Jan-Feb, Jul-Aug) ──
    month = daily.index.month
    features["is_result_season"] = ((month.isin([1, 2, 7, 8]))).astype(float)

    # ── 18-19. Days since 52-week high/low ──
    rolling_high = c.rolling(252, min_periods=1).max()
    rolling_low = c.rolling(252, min_periods=1).min()
    # Approximate: count days since price was at rolling max/min
    is_high = (c >= rolling_high * 0.999)
    is_low = (c <= rolling_low * 1.001)
    features["days_since_52w_high"] = _days_since_flag(is_high).clip(0, 252) / 252.0
    features["days_since_52w_low"] = _days_since_flag(is_low).clip(0, 252) / 252.0

    # ── 20. avg_intraday_range ──
    daily_range = (h - l) / c.replace(0, 1)
    features["avg_intraday_range"] = daily_range.rolling(20, min_periods=1).mean().fillna(0).clip(0, 0.2)

    features = features.fillna(0.0)
    logger.debug(f"Computed {len(features.columns)} session features for {len(features)} days")
    return features[SESSION_FEATURE_NAMES]


def _true_range(high, low, close):
    """Compute True Range."""
    prev_close = close.shift(1).fillna(close)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)


def _is_expiry_week(dates: pd.DatetimeIndex) -> pd.Series:
    """Approximate F&O expiry week (last Thursday of month)."""
    last_thu = dates.to_series().groupby([dates.year, dates.month]).transform(
        lambda g: g[g.dt.dayofweek == 3].max() if (g.dt.dayofweek == 3).any() else g.max()
    )
    # Expiry week = within 5 calendar days of last Thursday
    return ((last_thu - dates.to_series()).dt.days.abs() <= 4).astype(float)


def _is_monthly_expiry(dates: pd.DatetimeIndex) -> pd.Series:
    """Flag the last Thursday of each month (monthly F&O expiry)."""
    s = dates.to_series()
    last_thu = s.groupby([dates.year, dates.month]).transform(
        lambda g: g[g.dt.dayofweek == 3].max() if (g.dt.dayofweek == 3).any() else pd.NaT
    )
    return (s.dt.date == last_thu.dt.date).astype(float)


def _days_since_flag(flag_series: pd.Series) -> pd.Series:
    """Count days since flag was last True."""
    groups = (~flag_series).cumsum()
    return flag_series.groupby(groups).cumcount()
