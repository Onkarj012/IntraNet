"""
Point-in-time pre-open intraday feature matrix builder.

Assembles all 4 feature streams for the full universe on a given morning,
using only data available before 09:15 IST (strictly T-1 daily data,
overnight macro, close→open news window).

Streams:
  A. Price-action / technical  — daily OHLCV panel (T-1 close)
  B. Macro + sector context    — MarketFeatureBuilder (yfinance cached)
  C. Sentiment (close→open)    — SentimentFeatureBuilder premarket mode
  D. Barrier target (label)    — for training only; None at inference time

Usage:
  from equity.intraday_features import IntradayFeatureAssembler
  asm = IntradayFeatureAssembler()
  asm.refresh_macro()                          # once per morning
  X = asm.build(date="2026-06-05", symbols=symbols)  # → pd.DataFrame
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("equity.intraday_features")

_ROOT = Path(__file__).resolve().parents[2]
_PANEL_PATH = _ROOT / "cache/v8/daily_panel_nifty500_adj.parquet"
_MACRO_DIR = _ROOT / "market_data_cache"
_SENTIMENT_PARQUET = _ROOT / "data/sentiment/sentiment_cache_full.parquet"
_SENTIMENT_CSV = _ROOT / "data/sentiment/combined_sentiment_latest.csv"
_UNIVERSE_META = _ROOT / "data/sentiment/ind_nifty500list.csv"


# ---------------------------------------------------------------------------
# Phase A: Opening-range features (09:15–09:20, first 5-min bar)
# ---------------------------------------------------------------------------

def fetch_opening_bar(symbol: str, date: pd.Timestamp) -> dict[str, float]:
    """
    Fetch the first 5-min bar (09:15–09:20 IST) for symbol on date via yfinance.
    Returns opening-range features: gap, first-bar direction, volume surge, ORB context.
    All features are causal: only use data from 09:15–09:20.
    """
    try:
        import yfinance as yf
        start = date.strftime("%Y-%m-%d")
        end   = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(f"{symbol}.NS", start=start, end=end, interval="5m",
                          auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        # Convert to IST
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert("Asia/Kolkata")
        # First bar: 09:15–09:20 IST
        first_bars = raw.between_time("09:15", "09:20")
        if first_bars.empty:
            return {}
        bar = first_bars.iloc[0]
        o, h, l, c, v = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["volume"])
        if o <= 0:
            return {}
        return {
            "orb_open":         o,
            "orb_first_ret":    (c - o) / o,                     # first-bar direction
            "orb_range_pct":    (h - l) / o,                     # opening range size
            "orb_close_loc":    (c - l) / max(h - l, 1e-6),      # where did it close in range
            "orb_body_ratio":   abs(c - o) / max(h - l, 1e-6),   # conviction of first bar
            "orb_volume_pct":   v,                                 # filled below if vol context
            "orb_upper_shadow": (h - max(o, c)) / max(h - l, 1e-6),
            "orb_lower_shadow": (min(o, c) - l) / max(h - l, 1e-6),
        }
    except Exception:
        return {}


def enrich_with_opening_bars(X: pd.DataFrame, symbols: list[str],
                              date: pd.Timestamp) -> pd.DataFrame:
    """
    Add opening-range features to an existing feature matrix.
    Called at 09:20 IST after the first 5-min bar closes.
    Only used in live inference / post-09:15 backtest mode.
    """
    orb_cols = ["orb_first_ret", "orb_range_pct", "orb_close_loc",
                "orb_body_ratio", "orb_upper_shadow", "orb_lower_shadow"]
    for col in orb_cols:
        if col not in X.columns:
            X[col] = 0.0

    for sym in symbols:
        if sym not in X.index:
            continue
        feats = fetch_opening_bar(sym, date)
        for col in orb_cols:
            if col in feats:
                X.at[sym, col] = feats[col]
    return X


# ---------------------------------------------------------------------------
# Stream A helpers: price-action features from the OHLCV panel
# ---------------------------------------------------------------------------

def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=14, min_periods=7).mean()


def build_price_action_features(ohlcv: dict[str, pd.DataFrame], symbol: str,
                                 date: pd.Timestamp) -> dict[str, float]:
    """
    Compute ~30 daily price-action features for one symbol up to T-1.
    All look-back windows use data strictly <= T-1.
    """
    try:
        c = ohlcv["close"][symbol].dropna()
        v = ohlcv["volume"][symbol].dropna()
        h = ohlcv.get("high", {symbol: c})[symbol].reindex(c.index)
        lo = ohlcv.get("low", {symbol: c})[symbol].reindex(c.index)
        o = ohlcv.get("open", {symbol: c})[symbol].reindex(c.index)
    except KeyError:
        return {}

    # Only data through T-1
    c = c[c.index <= date - pd.Timedelta(days=1)]
    if len(c) < 22:
        return {}
    h, lo, o, v = h.reindex(c.index), lo.reindex(c.index), o.reindex(c.index), v.reindex(c.index)

    last = c.index[-1]
    p = float(c.iloc[-1])

    feats: dict[str, float] = {}

    # Returns
    for lag, name in [(1, "return_1d"), (5, "return_5d"), (10, "return_10d"),
                      (21, "return_21d"), (63, "return_63d")]:
        feats[name] = float(c.pct_change(lag).iloc[-1]) if len(c) > lag else 0.0

    # Overnight gap (open vs prev close)
    feats["overnight_return"] = float((o.iloc[-1] / c.iloc[-2] - 1)) if len(c) >= 2 else 0.0
    feats["gap_size"] = abs(feats["overnight_return"])
    feats["gap_direction"] = float(np.sign(feats["overnight_return"]))

    # Volatility: Parkinson & Garman-Klass (5d and 21d)
    for window, sfx in [(5, "5d"), (21, "21d")]:
        if len(h) >= window:
            hl = np.log(h / lo).replace(0, np.nan).dropna()
            park = float(hl.rolling(window).std().iloc[-1]) if len(hl) >= window else 0.0
            feats[f"parkinson_vol_{sfx}"] = park
            co = np.log(c / o).fillna(0)
            oc = np.log(o / c.shift(1)).fillna(0)
            gk = float(np.sqrt((0.5 * (np.log(h / lo) ** 2) - (2 * np.log(2) - 1) * co ** 2)
                               .rolling(window).mean().iloc[-1])) if len(c) >= window else 0.0
            feats[f"gk_vol_{sfx}"] = max(gk, 0.0)

    # ATR
    atr_series = _atr14(h, lo, c)
    feats["avg_true_range_14d"] = float(atr_series.iloc[-1]) / p if p > 0 else 0.0
    feats["atr_percentile"] = float(atr_series.rank(pct=True).iloc[-1])

    # Price position (close vs rolling high-low range)
    for window, sfx in [(20, "20d"), (63, "63d")]:
        roll_h = h.rolling(window).max()
        roll_l = lo.rolling(window).min()
        rng = (roll_h - roll_l).replace(0, np.nan)
        feats[f"price_position_{sfx}"] = float(((c - roll_l) / rng).iloc[-1]) if len(c) >= window else 0.5

    # RSI-14
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    al = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = ag / al.replace(0, 1e-10)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]
    feats["rsi_14d"] = float(rsi) / 100.0

    # Bollinger position
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std().replace(0, 1)
    feats["bollinger_position"] = float(((c - bb_mid) / (2 * bb_std)).iloc[-1])

    # Relative volume
    vol_mean = v.rolling(21).mean().replace(0, 1)
    feats["rel_volume_20d"] = float((v / vol_mean).iloc[-1])
    feats["volume_trend_5d"] = float(v.pct_change(5).iloc[-1]) if len(v) > 5 else 0.0
    feats["volume_dryup_ratio"] = float((v.rolling(5).mean() / v.rolling(21).mean().replace(0, 1)).iloc[-1])

    # Vol contraction (ATR compression)
    feats["vol_contraction_5d"] = float((atr_series.rolling(5).mean() / atr_series.rolling(21).mean().replace(0, 1)).iloc[-1]) if len(atr_series) >= 21 else 1.0
    feats["vol_contraction_21d"] = float((atr_series.rolling(21).mean() / atr_series.rolling(63).mean().replace(0, 1)).iloc[-1]) if len(atr_series) >= 63 else 1.0

    # High-low range pct
    feats["high_low_range_pct"] = float(((h - lo) / c.replace(0, np.nan)).iloc[-1])
    feats["prev_day_range_pct"] = feats["high_low_range_pct"]

    # Momentum (short horizons for specialists)
    for lag, name in [(5, "momentum_5d"), (10, "momentum_10d"), (21, "momentum_21d")]:
        feats[name] = float(c.pct_change(lag).iloc[-1]) if len(c) > lag else 0.0

    # SMA distances (used by momentum specialist)
    for window, name in [(21, "close_vs_sma_21d"), (63, "close_vs_sma_63d")]:
        sma = c.rolling(window).mean()
        feats[name] = float((c / sma.replace(0, np.nan) - 1).iloc[-1]) if len(c) >= window else 0.0

    # Inside-day count (last 5 days)
    inside = ((h <= h.shift(1)) & (lo >= lo.shift(1))).rolling(5).sum()
    feats["inside_day_count_5d"] = float(inside.iloc[-1]) if not np.isnan(inside.iloc[-1]) else 0.0

    # Narrow range (current range vs 7-day range)
    feats["close_vs_narrow_range"] = float(((h - lo) / h.rolling(7).max().replace(0, 1)).iloc[-1])
    feats["close_vs_vwap"] = 0.0  # requires intraday data; leave at 0 (pre-open)
    feats["afternoon_vs_morning"] = 0.0  # intraday — not available pre-open

    return {k: float(np.nan_to_num(v, nan=0.0)) for k, v in feats.items()}


# ---------------------------------------------------------------------------
# Stream B: macro + sector (wraps existing MarketFeatureBuilder)
# ---------------------------------------------------------------------------

def build_macro_features(market_builder, date: pd.Timestamp,
                          industry: str = "") -> dict[str, float]:
    dates_idx = pd.DatetimeIndex([date])
    try:
        mf = market_builder.get_features(dates_idx)
        india_f = market_builder.get_india_market_features(dates_idx, industry=industry)
        row: dict[str, float] = {k: float(mf[k].iloc[0]) for k in mf.columns}
        for k, s in india_f.items():
            row[k] = float(s.iloc[0])
        # Extra macro signals needed by specialists
        nifty = market_builder._get_close("nifty50")
        india_vix = market_builder._get_close("india_vix")
        if not nifty.empty:
            row["nifty_vs_50dma"] = float((nifty / nifty.rolling(50).mean()).iloc[-1] - 1) if len(nifty) >= 50 else 0.0
            row["nifty_vs_200dma"] = float((nifty / nifty.rolling(200).mean()).iloc[-1] - 1) if len(nifty) >= 200 else 0.0
            row["breadth_pct_above_20dma"] = float(((nifty / nifty.rolling(20).mean()) > 1).astype(float).iloc[-1])
            row["nifty_adx"] = 0.3  # placeholder — daily Nifty ADX
        if not india_vix.empty:
            row["vix_trend_5d"] = float(india_vix.pct_change(5).iloc[-1])
        row["sp500_overnight"] = row.get("sp500_overnight_return", 0.0)
        row["crude_change"] = row.get("crude_oil_return", 0.0)
        # Sector returns under expected names
        row["sector_return_1d"] = row.get("sector_index_prev_return", 0.0)
        row["sector_return_5d"] = row.get("sector_index_5d_return", 0.0)
        # Calendar flags
        row["day_of_week"] = float(date.dayofweek) / 4.0
        row["month"] = float(date.month) / 12.0
        row["expiry_week"] = float(_is_expiry_week(date))
        row["budget_day"] = float(date.month == 2 and date.day == 1)
        row["market_cap_category"] = 0.5  # filled per-symbol later
    except Exception as e:
        logger.warning("macro features failed: %s", e)
        row = {}
    return {k: float(np.nan_to_num(v, nan=0.0)) for k, v in row.items()}


def _is_expiry_week(date: pd.Timestamp) -> bool:
    import calendar
    year, month = date.year, date.month
    days_in_month = calendar.monthrange(year, month)[1]
    thursdays = [d for d in range(1, days_in_month + 1)
                 if pd.Timestamp(year, month, d).dayofweek == 3]
    if not thursdays:
        return False
    last_thursday = pd.Timestamp(year, month, thursdays[-1])
    return abs((date - last_thursday).days) <= 4


# ---------------------------------------------------------------------------
# Stream C: sentiment (premarket, close → open window)
# ---------------------------------------------------------------------------

def build_sentiment_features(sentiment_builder, symbol: str,
                               date: pd.Timestamp) -> dict[str, float]:
    dates_idx = pd.DatetimeIndex([date])
    try:
        sf = sentiment_builder.get_features(symbol, dates_idx)
        return {k: float(np.nan_to_num(sf[k].iloc[0], nan=0.0)) for k in sf.columns}
    except Exception as e:
        logger.debug("sentiment features skipped for %s: %s", symbol, e)
        return {}


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

class IntradayFeatureAssembler:
    """
    Assembles the full point-in-time pre-open feature matrix for the universe.

    Parameters
    ----------
    panel_path : Path, optional
        OHLCV panel parquet. Defaults to the nifty500 adj panel.
    macro_dir : Path, optional
        Directory with cached macro/sector CSVs.
    sentiment_csv : Path, optional
        Path to historical sentiment CSV.
    universe_meta_csv : Path, optional
        Path to ind_nifty500list.csv for industry lookup.
    """

    def __init__(
        self,
        panel_path: Path | str | None = None,
        macro_dir: Path | str | None = None,
        sentiment_csv: Path | str | None = None,
        universe_meta_csv: Path | str | None = None,
    ):
        self.panel_path = Path(panel_path or _PANEL_PATH)
        self.macro_dir = Path(macro_dir or _MACRO_DIR)
        # Prefer the dense parquet cache if it exists; fall back to CSV
        if sentiment_csv is not None:
            self.sentiment_csv = Path(sentiment_csv)
        elif _SENTIMENT_PARQUET.exists():
            self.sentiment_csv = _SENTIMENT_PARQUET
            logger.info("Using dense sentiment cache: %s", _SENTIMENT_PARQUET)
        else:
            self.sentiment_csv = _SENTIMENT_CSV
        self.universe_meta_csv = str(universe_meta_csv or _UNIVERSE_META)

        self._ohlcv: dict[str, pd.DataFrame] | None = None
        self._market_builder = None
        self._sentiment_builder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_macro(self, start: str = "2021-01-01") -> None:
        """Download / update macro + sector cache from yfinance. Call once per morning."""
        from equity.features.market_features import MarketFeatureBuilder
        self._market_builder = MarketFeatureBuilder(str(self.macro_dir))
        self._market_builder.download(start=start)
        logger.info("Macro cache refreshed")

    def build(
        self,
        date: str | pd.Timestamp,
        symbols: Sequence[str],
        *,
        live_news: bool = False,
    ) -> pd.DataFrame:
        """
        Build the pre-open feature matrix for all symbols on ``date``.

        Parameters
        ----------
        date : str | Timestamp
            The trading date (features use data through T-1).
        symbols : list[str]
            Universe to score.
        live_news : bool
            If True, fetch overnight news via yfinance for each symbol.
            Slower but adds live sentiment.

        Returns
        -------
        pd.DataFrame
            Index = symbol, columns = all feature names.
        """
        date = pd.Timestamp(date).normalize()
        self._ensure_loaded(live_news, symbols, date)

        rows = []
        for sym in symbols:
            row = {}

            # Stream A: price-action
            pa = build_price_action_features(self._ohlcv, sym, date)
            if not pa:
                continue  # skip symbol if no price history
            row.update(pa)

            # Stream B: macro + sector
            industry = self._get_industry(sym)
            if self._market_builder is not None:
                row.update(build_macro_features(self._market_builder, date, industry))
            row["market_cap_category"] = self._market_cap_category(sym)

            # Stream C: sentiment (premarket)
            if self._sentiment_builder is not None:
                row.update(build_sentiment_features(self._sentiment_builder, sym, date))

            # Relative-strength vs sector (needs all symbols to be processed first — added below)
            row["__rs_tmp_close"] = float(self._ohlcv["close"].get(sym, pd.Series(dtype=float)).iloc[-1]) \
                if sym in self._ohlcv.get("close", {}).columns else np.nan

            row["symbol"] = sym
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        X = pd.DataFrame(rows).set_index("symbol")

        # Compute cross-sectional relative strength (needs all rows)
        self._add_relative_strength(X, date)
        X.drop(columns=["__rs_tmp_close"], errors="ignore", inplace=True)

        # Fill NaN → 0, clip extreme values
        X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        X = X.clip(-10, 10)

        logger.info("Feature matrix built: %d symbols × %d features on %s",
                    len(X), len(X.columns), date.date())
        return X

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self, live_news: bool, symbols: Sequence[str],
                       date: pd.Timestamp) -> None:
        """Lazy-load OHLCV panel, sentiment builder."""
        if self._ohlcv is None:
            if not self.panel_path.exists():
                raise FileNotFoundError(
                    f"OHLCV panel not found: {self.panel_path}\n"
                    "Run: python scripts/data/equity_eod_panel.py --update"
                )
            from equity.momentum import load_ohlcv
            self._ohlcv = load_ohlcv(self.panel_path)
            logger.info("Loaded OHLCV panel: %s → %s",
                        self._ohlcv["close"].index.min().date(),
                        self._ohlcv["close"].index.max().date())

        if self._market_builder is None:
            # Load from cache without downloading (download done explicitly via refresh_macro)
            from equity.features.market_features import MarketFeatureBuilder
            self._market_builder = MarketFeatureBuilder(str(self.macro_dir))
            self._market_builder._load_cached()

        if self._sentiment_builder is None:
            from equity.features.sentiment_features import SentimentFeatureBuilder
            articles = None
            if live_news:
                try:
                    from equity.v8.per_stock_sentiment import fetch_live_sentiment
                    articles_df, _ = fetch_live_sentiment(
                        list(symbols), date,
                        universe_metadata_csv=self.universe_meta_csv,
                    )
                except Exception as e:
                    logger.warning("Live news fetch failed, using historical only: %s", e)

            self._sentiment_builder = SentimentFeatureBuilder(
                csv_path=str(self.sentiment_csv),
                market_builder=self._market_builder,
                mode="premarket",
                market_open_time="09:15",
                universe_metadata_csv=self.universe_meta_csv,
                articles_df=articles,
            )

    def _get_industry(self, symbol: str) -> str:
        from equity.universe import get_symbol_metadata
        return get_symbol_metadata(symbol, self.universe_meta_csv).get("industry", "")

    def _market_cap_category(self, symbol: str) -> float:
        """0=small, 0.5=mid, 1=large — based on universe tier (rough proxy from ADV)."""
        from equity.universe import get_universe
        if symbol in get_universe("nifty50"):
            return 1.0
        if symbol in get_universe("nifty100"):
            return 0.75
        if symbol in get_universe("nifty200"):
            return 0.5
        return 0.25

    def _add_relative_strength(self, X: pd.DataFrame, date: pd.Timestamp) -> None:
        """Add cross-sectional RS features (needs all symbols in X simultaneously)."""
        if "__rs_tmp_close" not in X.columns:
            return
        closes = X["__rs_tmp_close"]
        # RS vs sector: group by sector, rank within group
        sector_col = X.get("sector_return_1d", pd.Series(0.0, index=X.index))
        # Cross-sectional momentum rank as RS proxy
        for lag, col in [(5, "return_5d"), (21, "return_21d")]:
            if col in X.columns:
                rank_col = f"rs_vs_sector_{lag}d"
                X[rank_col] = X[col].rank(pct=True) - 0.5  # centered [-0.5, 0.5]
