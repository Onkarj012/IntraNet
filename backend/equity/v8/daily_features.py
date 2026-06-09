"""
Daily Feature Builder for V8 Pipeline.

Computes ~60 engineered daily features from minute session data, market data,
and sentiment data. All features align with the 5 specialist models
(momentum, reversal, breakout, sentiment, macro) defined in signal_models.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .per_stock_sentiment import compute_per_stock_sentiment_features

# ---------------------------------------------------------------------------
# Industry -> Sector mapping (extends V7's PRIMARY_INDEX_BY_INDUSTRY)
# ---------------------------------------------------------------------------

INDUSTRY_TO_SECTOR = {
    "Automobile and Auto Components": "auto",
    "Automobiles": "auto",
    "Banks": "bank",
    "Banking": "bank",
    "Financial Services": "financial_services",
    "Finance": "financial_services",
    "NBFC": "financial_services",
    "Insurance": "financial_services",
    "Fast Moving Consumer Goods": "fmcg",
    "FMCG": "fmcg",
    "Consumer Staples": "fmcg",
    "Healthcare": "healthcare",
    "Pharmaceuticals & Biotechnology": "pharma",
    "Pharma": "pharma",
    "Information Technology": "it",
    "IT": "it",
    "IT Services": "it",
    "Software": "it",
    "Media Entertainment & Publication": "media",
    "Media": "media",
    "Entertainment": "media",
    "Metals & Mining": "metal",
    "Metal": "metal",
    "Steel": "metal",
    "Oil Gas & Consumable Fuels": "oil_gas",
    "Oil & Gas": "oil_gas",
    "Petroleum": "oil_gas",
    "Energy": "oil_gas",
    "Realty": "realty",
    "Real Estate": "realty",
    "Construction": "realty",
    "Construction Materials": "realty",
    "Cement & Cement Products": "realty",
    "Power": None,
    "Telecommunication": None,
    "Chemicals": None,
    "Consumer Durables": None,
    "Capital Goods": None,
    "Textiles": None,
    "Fertilisers & Pesticides": None,
    "Industrial Manufacturing": None,
    "Food Products": "fmcg",
    "Services": None,
    "Diversified": None,
}

KNOWN_SECTORS = [
    "auto", "bank", "financial_services", "fmcg", "healthcare",
    "it", "media", "metal", "oil_gas", "pharma", "psu_bank", "realty",
]


# ---------------------------------------------------------------------------
# Daily Feature Builder class
# ---------------------------------------------------------------------------

class DailyFeatureBuilder:
    """Compute all daily features for V8 from minute sessions + external data."""

    def __init__(
        self,
        market_data_dir: str | Path = "market_data_cache",
        sentiment_df: pd.DataFrame | None = None,
    ):
        self.market_df = self._load_market_data(market_data_dir)
        self.sector_dfs = self._load_sector_data(market_data_dir)
        self.sentiment_df = sentiment_df

    def build_for_stock(
        self,
        sessions: dict,
        symbol: str,
        industry: str = "",
    ) -> pd.DataFrame:
        """Compute all daily features for a single stock.
        
        Parameters
        ----------
        sessions : dict
            Date -> minute DataFrame mapping (from extract_sessions).
        symbol : str
            Stock symbol (e.g. "RELIANCE").
        industry : str
            Industry name for sector features.
            
        Returns
        -------
        pd.DataFrame
            Index: date (pd.Timestamp), columns: feature names.
        """
        if not sessions:
            return pd.DataFrame()

        daily = _build_eod_from_sessions(sessions)
        if daily.empty:
            return pd.DataFrame()

        features = pd.DataFrame(index=daily.index)
        features.index.name = "date"

        # Price action
        _add_return_features(features, daily)
        _add_momentum_features(features, daily)
        _add_price_position_features(features, daily)
        _add_gap_features(features, daily)
        _add_volatility_features(features, daily)

        # Technical
        _add_rsi_14d(features, daily)
        _add_bollinger_position(features, daily)
        _add_atr_features(features, daily)
        _add_inside_day_count(features, daily)
        _add_narrow_range_features(features, daily)
        _add_prev_day_range_pct(features, daily)

        # Volume
        _add_volume_features(features, daily)

        # Intraday session features
        _add_intraday_features(features, sessions, daily)

        # Sector relative strength
        _add_sector_features(features, daily, industry, self.sector_dfs)

        # Market / macro
        _add_market_features(features, self.market_df, industry, self.sector_dfs)

        # Calendar
        _add_calendar_features(features)

        # Sentiment (per-stock)
        _add_sentiment_features(features, symbol, self.sentiment_df)

        # Ensure float32
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32)

        return features

    @staticmethod
    def _load_market_data(market_data_dir: str | Path) -> pd.DataFrame:
        """Load all macro market CSVs into a wide DataFrame indexed by date."""
        market_data_dir = Path(market_data_dir)
        macro_files = {
            "nifty50": "nifty50.csv",
            "india_vix": "india_vix.csv",
            "cboe_vix": "cboe_vix.csv",
            "sp500": "sp500.csv",
            "dow": "dow.csv",
            "nasdaq": "nasdaq.csv",
            "nikkei": "nikkei.csv",
            "hangseng": "hangseng.csv",
            "shanghai": "shanghai.csv",
            "crude_brent": "crude_brent.csv",
            "gold": "gold.csv",
            "usdinr": "usdinr.csv",
            "us10y": "us10y.csv",
            "dxy": "dxy.csv",
        }

        dfs = {}
        for name, filename in macro_files.items():
            fpath = market_data_dir / filename
            if not fpath.exists():
                continue
            try:
                df = pd.read_csv(fpath, parse_dates=["Date"])
                df = df.rename(columns={"close": name})
                df = df.set_index("Date")[name].sort_index()
                dfs[name] = df
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        market = pd.DataFrame(dfs)
        market.index = pd.DatetimeIndex(market.index).normalize()
        return market.sort_index()

    @staticmethod
    def _load_sector_data(market_data_dir: str | Path) -> dict:
        """Load sector index data from market_data_cache."""
        market_data_dir = Path(market_data_dir)
        sector_dfs = {}
        for sector in KNOWN_SECTORS:
            fpath = market_data_dir / f"{sector}.csv"
            if fpath.exists():
                try:
                    df = pd.read_csv(fpath, parse_dates=["Date"])
                    df = df.set_index("Date")["close"].sort_index()
                    df.index = pd.DatetimeIndex(df.index).normalize()
                    sector_dfs[sector] = df
                except Exception:
                    continue
        return sector_dfs


# ---------------------------------------------------------------------------
# EOD builder
# ---------------------------------------------------------------------------

def _build_eod_from_sessions(sessions: dict) -> pd.DataFrame:
    """Build daily OHLCV DataFrame from minute session data."""
    rows = []
    for date, df in sorted(sessions.items()):
        if len(df) == 0:
            continue
        rows.append({
            "date": pd.Timestamp(date),
            "open": float(df["open"].iloc[0]),
            "high": float(df["high"].max()),
            "low": float(df["low"].min()),
            "close": float(df["close"].iloc[-1]),
            "volume": float(df["volume"].sum()),
            "n_bars": int(len(df)),
        })
    if not rows:
        return pd.DataFrame()
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    return daily


def _safe_div(a, b):
    """Divide a by b, return 0 if b is 0 or very close to it."""
    if abs(float(b)) < 1e-12:
        return 0.0
    return float(a) / float(b)


# ---------------------------------------------------------------------------
# Price action features
# ---------------------------------------------------------------------------

def _add_return_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute return-based features."""
    close = daily["close"]
    features["return_1d"] = close.pct_change(1)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_21d"] = close.pct_change(21)
    features["return_63d"] = close.pct_change(63)


def _add_momentum_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute momentum and SMA relative features."""
    close = daily["close"]
    features["momentum_5d"] = close.pct_change(5)
    features["momentum_10d"] = close.pct_change(10)
    features["momentum_21d"] = close.pct_change(21)

    sma_21 = close.rolling(21, min_periods=5).mean()
    sma_63 = close.rolling(63, min_periods=21).mean()
    features["close_vs_sma_21d"] = (close - sma_21) / sma_21.replace(0, np.nan)
    features["close_vs_sma_63d"] = (close - sma_63) / sma_63.replace(0, np.nan)


def _add_price_position_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute price position within rolling range (0=bottom, 1=top)."""
    close = daily["close"]
    for window in (20, 63):
        roll_high = daily["high"].rolling(window, min_periods=5).max()
        roll_low = daily["low"].rolling(window, min_periods=5).min()
        roll_range = roll_high - roll_low
        features[f"price_position_{window}d"] = (
            (close - roll_low) / roll_range.replace(0, np.nan)
        )


def _add_gap_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute overnight gap features."""
    prev_close = daily["close"].shift(1)
    overnight = (daily["open"] - prev_close) / prev_close.replace(0, np.nan)
    features["overnight_return"] = overnight
    features["gap_size"] = overnight.abs()


def _add_volatility_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute Parkinson and Garman-Klass volatility."""
    h = daily["high"]
    l = daily["low"]
    c = daily["close"]
    o = daily["open"]

    # Safe log ratios (avoid log(0) and divide-by-zero)
    eps = 1e-12
    hl_ratio = np.where(l.abs() > eps, h / l, np.nan)
    co_ratio = np.where(o.abs() > eps, c / o, np.nan)
    log_hl = np.log(np.maximum(hl_ratio, eps))
    log_co = np.log(np.maximum(co_ratio, eps))
    log_hl_sq = log_hl ** 2

    # Parkinson: (1 / (4 * ln(2))) * E[(ln(H/L))^2]
    parkinson_daily = pd.Series(log_hl_sq / (4.0 * np.log(2.0)), index=daily.index)

    # Garman-Klass: more efficient estimator using OHLC
    gk_daily = pd.Series(
        0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co ** 2, index=daily.index
    )

    for window in (5, 21):
        features[f"parkinson_vol_{window}d"] = np.sqrt(
            parkinson_daily.rolling(window, min_periods=3).mean()
        )
        features[f"gk_vol_{window}d"] = np.sqrt(
            gk_daily.rolling(window, min_periods=3).mean().clip(lower=0)
        )


# ---------------------------------------------------------------------------
# Technical features
# ---------------------------------------------------------------------------

def _add_rsi_14d(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute 14-day RSI (normalized 0-1)."""
    close = daily["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    features["rsi_14d"] = rsi / 100.0


def _add_bollinger_position(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute Bollinger Band position (0=lower band, 1=upper band)."""
    close = daily["close"]
    sma = close.rolling(20, min_periods=5).mean()
    std = close.rolling(20, min_periods=5).std()
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    bb_range = upper - lower
    features["bollinger_position"] = (close - lower) / bb_range.replace(0, np.nan)


def _add_atr_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute ATR-based features."""
    h, l, c = daily["high"], daily["low"], daily["close"]
    prev_c = c.shift(1)

    tr = np.maximum(
        h - l,
        np.maximum((h - prev_c).abs(), (l - prev_c).abs()),
    )
    atr = tr.ewm(span=14, min_periods=1).mean()
    norm_atr = atr / c.replace(0, np.nan)
    features["avg_true_range_14d"] = norm_atr

    # ATR percentile over 63 days
    atr_rank = norm_atr.rolling(63, min_periods=21).rank(pct=True)
    features["atr_percentile"] = atr_rank


def _add_inside_day_count(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Count inside days in rolling 5-day window."""
    inside = (
        (daily["high"] < daily["high"].shift(1))
        & (daily["low"] > daily["low"].shift(1))
    ).astype(float)
    features["inside_day_count_5d"] = inside.rolling(5, min_periods=1).sum()


def _add_narrow_range_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Close position within 10-day high-low range."""
    close = daily["close"]
    nr_high = daily["high"].rolling(10, min_periods=5).max()
    nr_low = daily["low"].rolling(10, min_periods=5).min()
    nr_range = nr_high - nr_low
    features["close_vs_narrow_range"] = (
        (close - nr_low) / nr_range.replace(0, np.nan)
    )
    # high_low_range_pct: today's range as pct of close
    features["high_low_range_pct"] = (
        (daily["high"] - daily["low"]) / daily["close"].replace(0, np.nan)
    )


def _add_prev_day_range_pct(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Previous day's range as percentage of open."""
    prev_high = daily["high"].shift(1)
    prev_low = daily["low"].shift(1)
    prev_open = daily["open"].shift(1)
    features["prev_day_range_pct"] = (
        (prev_high - prev_low) / prev_open.replace(0, np.nan)
    )


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

def _add_volume_features(features: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Compute volume-based features."""
    vol = daily["volume"].replace(0, np.nan)

    # Relative volume vs 20-day average
    vol_mean_5 = vol.rolling(5, min_periods=3).mean()
    vol_mean_20 = vol.rolling(20, min_periods=5).mean()
    vol_mean_21 = vol.rolling(21, min_periods=5).mean()
    vol_mean_63 = vol.rolling(63, min_periods=21).mean()

    features["rel_volume_20d"] = vol / vol_mean_20.replace(0, np.nan)
    features["volume_trend_5d"] = vol_mean_5 / vol_mean_20.replace(0, np.nan) - 1.0

    # Volume contraction: (short_term_vol / long_term_vol) - 1
    features["vol_contraction_5d"] = vol_mean_5 / vol_mean_20.replace(0, np.nan) - 1.0
    features["vol_contraction_21d"] = vol_mean_21 / vol_mean_63.replace(0, np.nan) - 1.0

    # Volume dryup ratio: current volume / max volume in last 5 days
    vol_max_5 = vol.rolling(5, min_periods=3).max()
    features["volume_dryup_ratio"] = vol / vol_max_5.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Intraday session features (computed from minute data, not EOD)
# ---------------------------------------------------------------------------

def _add_intraday_features(
    features: pd.DataFrame,
    sessions: dict,
    daily: pd.DataFrame,
) -> None:
    """Compute close_vs_vwap and afternoon_vs_morning from minute data."""
    close_vs_vwap_vals = {}
    afternoon_morning_vals = {}

    for date, df in sessions.items():
        date_ts = pd.Timestamp(date)
        if len(df) < 10:
            close_vs_vwap_vals[date_ts] = np.nan
            afternoon_morning_vals[date_ts] = np.nan
            continue

        # VWAP
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        tpv = tp * df["volume"]
        vwap = tpv.sum() / df["volume"].sum() if df["volume"].sum() > 0 else df["close"].mean()
        if vwap > 0:
            close_vs_vwap_vals[date_ts] = (df["close"].iloc[-1] - vwap) / vwap
        else:
            close_vs_vwap_vals[date_ts] = 0.0

        # Morning vs Afternoon
        n = len(df)
        mid = n // 2
        morning = df.iloc[:mid]
        afternoon = df.iloc[mid:]
        mor_open = float(morning["open"].iloc[0]) if len(morning) > 0 else 0.0
        mor_close = float(morning["close"].iloc[-1]) if len(morning) > 0 else 0.0
        aft_open = float(afternoon["open"].iloc[0]) if len(afternoon) > 0 else 0.0
        aft_close = float(afternoon["close"].iloc[-1]) if len(afternoon) > 0 else 0.0
        morning_ret = _safe_div(mor_close - mor_open, mor_open)
        afternoon_ret = _safe_div(aft_close - aft_open, aft_open)
        afternoon_morning_vals[date_ts] = afternoon_ret - morning_ret

    features["close_vs_vwap"] = pd.Series(close_vs_vwap_vals, index=features.index)
    features["afternoon_vs_morning"] = pd.Series(afternoon_morning_vals, index=features.index)


# ---------------------------------------------------------------------------
# Sector features
# ---------------------------------------------------------------------------

def _add_sector_features(
    features: pd.DataFrame,
    daily: pd.DataFrame,
    industry: str,
    sector_dfs: dict,
) -> None:
    """Compute relative strength vs sector."""
    close = daily["close"]
    stock_ret_5d = close.pct_change(5)
    stock_ret_21d = close.pct_change(21)

    # Determine sector
    sector = INDUSTRY_TO_SECTOR.get(industry) if industry else None
    if sector is None:
        # Try fuzzy match
        for key, val in INDUSTRY_TO_SECTOR.items():
            if isinstance(industry, str) and val and (key.lower() in industry.lower() or industry.lower() in key.lower()):
                sector = val
                break

    if sector is None or sector not in sector_dfs:
        features["rs_vs_sector_5d"] = np.nan
        features["rs_vs_sector_21d"] = np.nan
        features["sector_return_1d"] = np.nan
        features["sector_return_5d"] = np.nan
        return

    sector_close = sector_dfs[sector]
    # Reindex to daily dates
    aligned = sector_close.reindex(daily.index, method="ffill")
    aligned = aligned.ffill()

    sector_ret_1d = aligned.pct_change(1, fill_method=None)
    sector_ret_5d = aligned.pct_change(5, fill_method=None)
    sector_ret_21d = aligned.pct_change(21, fill_method=None)

    features["rs_vs_sector_5d"] = stock_ret_5d - sector_ret_5d
    features["rs_vs_sector_21d"] = stock_ret_21d - sector_ret_21d
    features["sector_return_1d"] = sector_ret_1d
    features["sector_return_5d"] = sector_ret_5d


# ---------------------------------------------------------------------------
# Market / macro features
# ---------------------------------------------------------------------------

def _add_market_features(
    features: pd.DataFrame,
    market_df: pd.DataFrame,
    industry: str,
    sector_dfs: dict,
) -> None:
    """Add market-level features (VIX, Nifty, SP500, crude, gold, etc.)."""
    if market_df.empty:
        return

    # Align market data to feature dates
    aligned = market_df.reindex(features.index, method="ffill")
    aligned = aligned.ffill()

    # VIX features
    if "india_vix" in aligned.columns:
        features["vix_level"] = aligned["india_vix"] / 100.0
        features["vix_trend_5d"] = aligned["india_vix"].pct_change(5, fill_method=None)

    # Nifty vs moving averages
    if "nifty50" in aligned.columns:
        nifty = aligned["nifty50"]
        nifty_sma_50 = nifty.rolling(50, min_periods=20).mean()
        nifty_sma_200 = nifty.rolling(200, min_periods=100).mean()
        features["nifty_vs_50dma"] = (nifty - nifty_sma_50) / nifty_sma_50.replace(0, np.nan)
        features["nifty_vs_200dma"] = (nifty - nifty_sma_200) / nifty_sma_200.replace(0, np.nan)
        # Breadth proxy: % of sector indices above 20dma (simplified)
        breadth_vals = []
        for name, series in sector_dfs.items():
            sec_aligned = series.reindex(features.index, method="ffill").ffill()
            sec_sma_20 = sec_aligned.rolling(20, min_periods=10).mean()
            breadth_vals.append(sec_aligned > sec_sma_20)
        if breadth_vals:
            breadth_df = pd.concat(breadth_vals, axis=1)
            features["breadth_pct_above_20dma"] = breadth_df.mean(axis=1)
        else:
            # Fallback: Nifty itself vs 20dma
            nifty_sma_20 = nifty.rolling(20, min_periods=10).mean()
            features["breadth_pct_above_20dma"] = (nifty > nifty_sma_20).astype(float)

    # SP500 overnight (daily return)
    if "sp500" in aligned.columns:
        features["sp500_overnight"] = aligned["sp500"].pct_change(1, fill_method=None)

    # Commodity / FX features
    if "usdinr" in aligned.columns:
        features["usdinr_change"] = aligned["usdinr"].pct_change(1, fill_method=None)

    if "crude_brent" in aligned.columns:
        crude_ret = aligned["crude_brent"].pct_change(1, fill_method=None)
        features["crude_change"] = crude_ret
        features["crude_oil_return"] = crude_ret

    if "gold" in aligned.columns:
        features["gold_return"] = aligned["gold"].pct_change(1, fill_method=None)

    if "dxy" in aligned.columns:
        features["dxy_change"] = aligned["dxy"].pct_change(1, fill_method=None)

    # Asia sentiment: mean of nikkei, hangseng, shanghai returns
    asia_cols = ["nikkei", "hangseng", "shanghai"]
    asia_available = [c for c in asia_cols if c in aligned.columns]
    if asia_available:
        asia_rets = aligned[asia_available].pct_change(1, fill_method=None)
        features["asia_sentiment"] = asia_rets.mean(axis=1)

    # Risk-on signal: mean of asia, dow, nasdaq, sp500, -dxy, -vix/cboe
    risk_cols = []
    for c in asia_available:
        risk_cols.append(aligned[c].pct_change(1, fill_method=None))
    for c in ["sp500", "dow", "nasdaq"]:
        if c in aligned.columns:
            risk_cols.append(aligned[c].pct_change(1, fill_method=None))
    for c in ["dxy", "cboe_vix"]:
        if c in aligned.columns:
            if c == "cboe_vix":
                risk_cols.append(-aligned[c].pct_change(1, fill_method=None) / 100.0)
            else:
                risk_cols.append(-aligned[c].pct_change(1, fill_method=None))
    if risk_cols:
        features["risk_on_signal"] = pd.concat(risk_cols, axis=1).mean(axis=1)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def _add_calendar_features(features: pd.DataFrame) -> None:
    """Add calendar-based features."""
    idx = features.index
    features["day_of_week"] = idx.dayofweek / 4.0
    features["month"] = (idx.month - 1) / 11.0

    # Expiry week: last Thursday of the month
    # Check if date is within 4 days of the month's last Thursday
    last_day = idx + pd.offsets.MonthEnd(0)
    last_thu = last_day - pd.to_timedelta((last_day.dayofweek - 3) % 7, unit="D")
    features["expiry_week"] = (
        (idx >= last_thu - pd.Timedelta(days=3)) & (idx <= last_thu + pd.Timedelta(days=1))
    ).astype(float)

    # Budget day: Feb 1 (approximate, India's Union Budget usually Feb 1)
    features["budget_day"] = (
        (idx.month == 2) & (idx.day == 1)
    ).astype(float)


# ---------------------------------------------------------------------------
# Sentiment features (per-stock)
# ---------------------------------------------------------------------------

def _add_sentiment_features(
    features: pd.DataFrame,
    symbol: str,
    sentiment_df: pd.DataFrame | None,
) -> None:
    """Add per-stock sentiment features."""
    # Initialize columns
    sent_cols = [
        "sentiment_score_1d", "sentiment_score_3d",
        "sentiment_momentum_5d", "sentiment_volatility_5d",
        "article_count_1d", "article_count_3d",
        "headline_sentiment_bias", "news_to_price_ratio",
    ]
    for col in sent_cols:
        features[col] = np.nan

    if sentiment_df is None or sentiment_df.empty:
        return

    # Pre-filter sentiment data for this symbol
    sym_articles = sentiment_df[
        sentiment_df["symbol"].str.upper() == symbol.upper()
    ].copy()
    if sym_articles.empty:
        return

    if "timestamp" not in sym_articles.columns:
        return

    sym_articles["date"] = pd.to_datetime(sym_articles["timestamp"]).dt.normalize()

    # Compute daily sentiment aggregates
    daily_sent = sym_articles.groupby("date").agg(
        mean_score=("score", "mean"),
        count=("score", "count"),
    ).sort_index()

    # For each feature date, compute sentiment features
    for date_ts in features.index:
        # Lookback windows
        cutoff_1d = date_ts - pd.Timedelta(days=1)
        cutoff_3d = date_ts - pd.Timedelta(days=3)
        cutoff_5d = date_ts - pd.Timedelta(days=5)

        # 1-day
        recent_1d = daily_sent[daily_sent.index >= cutoff_1d]
        if len(recent_1d) >= 2:
            features.loc[date_ts, "sentiment_score_1d"] = float(recent_1d["mean_score"].mean())
            features.loc[date_ts, "article_count_1d"] = float(recent_1d["count"].sum())

        # 3-day
        recent_3d = daily_sent[daily_sent.index >= cutoff_3d]
        if len(recent_3d) >= 2:
            features.loc[date_ts, "sentiment_score_3d"] = float(recent_3d["mean_score"].mean())
            features.loc[date_ts, "article_count_3d"] = float(recent_3d["count"].sum())

        # 5-day momentum and volatility
        recent_5d = daily_sent[daily_sent.index >= cutoff_5d]
        if len(recent_5d) >= 3:
            features.loc[date_ts, "sentiment_momentum_5d"] = float(
                recent_5d["mean_score"].iloc[-1] - recent_5d["mean_score"].iloc[0]
            )
            features.loc[date_ts, "sentiment_volatility_5d"] = float(
                recent_5d["mean_score"].std()
            )

        # Headline sentiment bias
        recent_articles = sym_articles[sym_articles["date"] >= cutoff_3d]
        if "headline" in recent_articles.columns:
            headlines = recent_articles["headline"].dropna().unique()
            if len(headlines) > 0:
                # Simple keyword-based sentiment inference
                headline_scores = []
                for h in headlines:
                    s = _quick_headline_polarity(h)
                    headline_scores.append(s)
                features.loc[date_ts, "headline_sentiment_bias"] = float(np.mean(headline_scores))


def _quick_headline_polarity(headline: str) -> float:
    """Fast keyword-based headline sentiment (-1 to +1)."""
    if not isinstance(headline, str):
        return 0.0
    h = headline.lower()
    pos_words = ["beat", "surge", "gain", "bullish", "profit", "upgrade",
                 "rise", "growth", "strong", "positive", "outperform",
                 "jump", "rally", "boost", "record"]
    neg_words = ["miss", "drop", "fall", "weak", "downgrade", "fraud",
                 "decline", "loss", "negative", "underperform", "crash",
                 "plunge", "cut", "warn", "probe"]
    score = 0
    words = h.split()
    for w in words:
        if w in pos_words:
            score += 1
        elif w in neg_words:
            score -= 1
    return max(-1.0, min(1.0, score / 4.0))
