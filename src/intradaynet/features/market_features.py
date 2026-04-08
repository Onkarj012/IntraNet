"""
Global market & macro feature builder.

Downloads and caches index/commodity/currency data from yfinance,
then computes 10 global macro features per trading date:

  15. crude_oil_return        — Brent crude daily return
  16. crude_oil_5d_change     — 5-day crude oil trend
  17. gold_return             — Gold daily return
  18. usdinr_change           — USD/INR daily change
  19. us_10y_yield_change     — US 10Y bond yield daily change
  20. dxy_change              — Dollar index daily change
  21. asia_sentiment          — Avg Asian index overnight return
  22. dow_overnight_return    — Dow Jones previous close return
  23. nasdaq_overnight_return — NASDAQ previous close return
  24. global_volatility_regime— CBOE VIX level (normalized)
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("intradaynet.features.market_features")

# Tickers for macro data (all available on yfinance)
MACRO_TICKERS = {
    "india_vix":   "^INDIAVIX",
    "nifty50":     "^NSEI",
    "sp500":       "^GSPC",
    "crude_brent": "BZ=F",
    "gold":        "GC=F",
    "usdinr":      "INR=X",
    "us10y":       "^TNX",
    "dxy":         "DX-Y.NYB",
    "dow":         "^DJI",
    "nasdaq":      "^IXIC",
    "nikkei":      "^N225",
    "hangseng":    "^HSI",
    "shanghai":    "000001.SS",
    "cboe_vix":    "^VIX",
}

# Sector → yfinance sector index ticker
SECTOR_INDEX_MAP = {
    "IT":          "^CNXIT",
    "BANK":        "^NSEBANK",
    "PHARMA":      "^CNXPHARMA",
    "AUTO":        "^CNXAUTO",
    "FMCG":        "^CNXFMCG",
    "METAL":       "^CNXMETAL",
    "ENERGY":      "^CNXENERGY",
    "REALTY":      "^CNXREALTY",
    "DEFAULT":     "^NSEI",
}

# 10 new global/macro feature names
MARKET_FEATURE_NAMES = [
    "crude_oil_return",
    "crude_oil_5d_change",
    "gold_return",
    "usdinr_change",
    "us_10y_yield_change",
    "dxy_change",
    "asia_sentiment",
    "dow_overnight_return",
    "nasdaq_overnight_return",
    "global_volatility_regime",
]


class MarketFeatureBuilder:
    """
    Downloads macro/global data from yfinance, caches locally,
    and computes per-date market features.

    Usage:
        builder = MarketFeatureBuilder(cache_dir="market_data_cache")
        builder.download(start="2022-01-01", end="2026-03-11")
        features = builder.get_features(dates)  # → DataFrame (N, 10)
    """

    def __init__(self, cache_dir: str = "market_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data = {}  # ticker_name → DataFrame

    def download(self, start: str = "2021-01-01", end: str = None):
        """Download (or update) macro data from yfinance and cache as CSVs."""
        import yfinance as yf

        if end is None:
            end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        for name, ticker in MACRO_TICKERS.items():
            csv_path = self.cache_dir / f"{name}.csv"

            try:
                # Check if we have cached data and when it ends
                if csv_path.exists():
                    existing = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
                    if len(existing) > 0:
                        last_date = existing.index.max()
                        # Always download the new portion
                        dl_start = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")
                    else:
                        dl_start = start
                else:
                    dl_start = start

                df = yf.download(ticker, start=dl_start, end=end,
                                 interval="1d", progress=False, auto_adjust=True)
                if df.empty:
                    logger.warning(f"No data for {name} ({ticker})")
                    continue

                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[["Close"]].rename(columns={"Close": "close"})
                df.index.name = "Date"

                # Merge with existing if present
                if csv_path.exists():
                    existing = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
                    df = pd.concat([existing, df])
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.sort_index()

                df.to_csv(csv_path)
                self._data[name] = df
                logger.debug(f"Downloaded {name}: {len(df)} rows")

            except Exception as e:
                logger.warning(f"Failed to download {name} ({ticker}): {e}")

    def _load_cached(self):
        """Load all cached CSVs into memory."""
        for name in MACRO_TICKERS:
            if name in self._data:
                continue
            csv_path = self.cache_dir / f"{name}.csv"
            if csv_path.exists():
                self._data[name] = pd.read_csv(
                    csv_path, parse_dates=["Date"], index_col="Date"
                )

    def _get_close(self, name: str) -> pd.Series:
        """Get close price series for a macro ticker."""
        self._load_cached()
        if name in self._data and "close" in self._data[name].columns:
            return self._data[name]["close"]
        return pd.Series(dtype=float)

    def _safe_return(self, series: pd.Series) -> pd.Series:
        """Compute daily return safely."""
        return series.pct_change().fillna(0).clip(-0.2, 0.2)

    def get_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Compute 10 global/macro features for each date.

        Args:
            dates: DatetimeIndex of dates to get features for.

        Returns:
            DataFrame with 10 market feature columns, indexed by date.
        """
        self._load_cached()
        features = pd.DataFrame(0.0, index=dates, columns=MARKET_FEATURE_NAMES)

        # Helper: align a series to our dates (forward-fill for holidays)
        def aligned(series):
            if series.empty:
                return pd.Series(0.0, index=dates)
            return series.reindex(dates, method="ffill").fillna(0)

        # ── Feature 15: crude_oil_return ──
        crude = self._get_close("crude_brent")
        crude_ret = self._safe_return(crude)
        features["crude_oil_return"] = aligned(crude_ret)

        # ── Feature 16: crude_oil_5d_change ──
        crude_5d = crude.pct_change(5).fillna(0).clip(-0.5, 0.5) if len(crude) else pd.Series()
        features["crude_oil_5d_change"] = aligned(crude_5d)

        # ── Feature 17: gold_return ──
        gold = self._get_close("gold")
        features["gold_return"] = aligned(self._safe_return(gold))

        # ── Feature 18: usdinr_change ──
        usdinr = self._get_close("usdinr")
        features["usdinr_change"] = aligned(self._safe_return(usdinr))

        # ── Feature 19: us_10y_yield_change ──
        us10y = self._get_close("us10y")
        # Yield is in %, change in basis points / 100
        us10y_change = us10y.diff().fillna(0).clip(-1, 1) / 10.0
        features["us_10y_yield_change"] = aligned(us10y_change)

        # ── Feature 20: dxy_change ──
        dxy = self._get_close("dxy")
        features["dxy_change"] = aligned(self._safe_return(dxy))

        # ── Feature 21: asia_sentiment ──
        # Average of Nikkei, Hang Seng, Shanghai returns (they close before India opens)
        nikkei_ret = self._safe_return(self._get_close("nikkei"))
        hsi_ret = self._safe_return(self._get_close("hangseng"))
        shanghai_ret = self._safe_return(self._get_close("shanghai"))

        asia_avg = pd.DataFrame({
            "n": nikkei_ret, "h": hsi_ret, "s": shanghai_ret
        }).mean(axis=1).fillna(0)
        features["asia_sentiment"] = aligned(asia_avg)

        # ── Feature 22: dow_overnight_return ──
        dow = self._get_close("dow")
        features["dow_overnight_return"] = aligned(self._safe_return(dow))

        # ── Feature 23: nasdaq_overnight_return ──
        nasdaq = self._get_close("nasdaq")
        features["nasdaq_overnight_return"] = aligned(self._safe_return(nasdaq))

        # ── Feature 24: global_volatility_regime ──
        # CBOE VIX normalized: divide by 100 (typically 10-40 → 0.1-0.4)
        cboe_vix = self._get_close("cboe_vix")
        vix_norm = (cboe_vix / 100.0).clip(0, 1).fillna(0.2)
        features["global_volatility_regime"] = aligned(vix_norm)

        return features.fillna(0.0)

    def get_india_market_features(self, dates: pd.DatetimeIndex) -> dict:
        """
        Get India-specific market features for filling the 6 previously stubbed ones.

        Returns dict with keys matching SENTIMENT_FEATURE_NAMES[8:14]:
          - nifty_intraday_return (feature 9)
          - sector_intraday_return (feature 10) — default NIFTY
          - vix_level (feature 11)
          - vix_change (feature 12)
          - market_breadth (feature 13)
          - global_cue (feature 14)
        """
        self._load_cached()

        def aligned(series):
            if series.empty:
                return pd.Series(0.0, index=dates)
            return series.reindex(dates, method="ffill").fillna(0)

        result = {}

        # Feature 9: nifty_intraday_return
        # FIXED: Was using today's close-to-close return (LEAK at 9:15 AM).
        # Now uses yesterday's daily return — known at market open.
        nifty = self._get_close("nifty50")
        result["nifty_intraday_return"] = aligned(
            nifty.pct_change().shift(1)  # previous day's close-to-close return
        ).clip(-0.1, 0.1)

        # Feature 10: sector_intraday_return (default: NIFTY50)
        # FIXED: Inherited from nifty_intraday_return fix above.
        result["sector_intraday_return"] = result["nifty_intraday_return"]

        # Feature 11: vix_level (India VIX / 100)
        india_vix = self._get_close("india_vix")
        result["vix_level"] = aligned(
            (india_vix / 100.0).clip(0, 1).fillna(0.15)
        )

        # Feature 12: vix_change
        result["vix_change"] = aligned(
            self._safe_return(india_vix)
        ).clip(-1, 1)

        # Feature 13: market_breadth (approx: NIFTY 5d momentum as proxy)
        nifty_5d = nifty.pct_change(5).fillna(0).clip(-0.2, 0.2) if len(nifty) else pd.Series()
        result["market_breadth"] = aligned(nifty_5d)

        # Feature 14: global_cue — S&P 500 previous day return
        sp500 = self._get_close("sp500")
        result["global_cue"] = aligned(
            self._safe_return(sp500)
        ).clip(-0.1, 0.1)

        return result
