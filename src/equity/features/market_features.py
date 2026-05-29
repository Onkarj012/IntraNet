"""
Global market, macro, and sector/index feature builder.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("intradaynet.features.market_features")

MACRO_TICKERS = {
    "india_vix": "^INDIAVIX",
    "nifty50": "^NSEI",
    "sp500": "^GSPC",
    "crude_brent": "BZ=F",
    "gold": "GC=F",
    "usdinr": "INR=X",
    "us10y": "^TNX",
    "dxy": "DX-Y.NYB",
    "dow": "^DJI",
    "nasdaq": "^IXIC",
    "nikkei": "^N225",
    "hangseng": "^HSI",
    "shanghai": "000001.SS",
    "cboe_vix": "^VIX",
}

SECTOR_INDEX_REGISTRY = {
    "auto": {"display_name": "Nifty Auto Index", "ticker": "^CNXAUTO", "industries": ["Automobile and Auto Components"]},
    "bank": {"display_name": "Nifty Bank Index", "ticker": "^NSEBANK", "industries": []},
    "financial_services": {"display_name": "Nifty Financial Services Index", "ticker": "^CNXFIN", "industries": ["Financial Services"]},
    "fmcg": {"display_name": "Nifty FMCG Index", "ticker": "^CNXFMCG", "industries": ["Fast Moving Consumer Goods"]},
    "healthcare": {"display_name": "Nifty Healthcare Index", "ticker": "^CNXPHARMA", "industries": ["Healthcare"]},
    "it": {"display_name": "Nifty IT Index", "ticker": "^CNXIT", "industries": ["Information Technology"]},
    "media": {"display_name": "Nifty Media Index", "ticker": "^CNXMEDIA", "industries": ["Media Entertainment & Publication"]},
    "metal": {"display_name": "Nifty Metal Index", "ticker": "^CNXMETAL", "industries": ["Metals & Mining"]},
    "pharma": {"display_name": "Nifty Pharma Index", "ticker": "^CNXPHARMA", "industries": []},
    "private_bank": {"display_name": "Nifty Private Bank Index", "ticker": "^NIFTYPVTBANK", "industries": []},
    "psu_bank": {"display_name": "Nifty PSU Bank Index", "ticker": "^CNXPSUBANK", "industries": []},
    "realty": {"display_name": "Nifty Realty Index", "ticker": "^CNXREALTY", "industries": ["Realty"]},
    "oil_gas": {"display_name": "Nifty Oil and Gas Index", "ticker": "^CNXENERGY", "industries": ["Oil Gas & Consumable Fuels"]},
}

PRIMARY_INDEX_BY_INDUSTRY = {
    "Automobile and Auto Components": "auto",
    "Fast Moving Consumer Goods": "fmcg",
    "Financial Services": "financial_services",
    "Healthcare": "healthcare",
    "Information Technology": "it",
    "Media Entertainment & Publication": "media",
    "Metals & Mining": "metal",
    "Oil Gas & Consumable Fuels": "oil_gas",
    "Realty": "realty",
}

SECONDARY_INDEX_BY_INDUSTRY = {
    "Financial Services": ["private_bank", "psu_bank"],
    "Healthcare": ["pharma"],
}

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
    "india_vix_percentile",
    "nifty_5d_return",
    "sp500_overnight_return",
    "commodity_pressure",
    "dollar_yield_pressure",
    "risk_on_signal",
]


class MarketFeatureBuilder:
    def __init__(self, cache_dir: str = "market_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, pd.DataFrame] = {}

    def download(self, start: str = "2021-01-01", end: str | None = None):
        import yfinance as yf

        if end is None:
            end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_dt = pd.Timestamp(start).normalize()
        end_dt = pd.Timestamp(end).normalize()
        all_tickers = {**MACRO_TICKERS, **{key: value["ticker"] for key, value in SECTOR_INDEX_REGISTRY.items()}}

        for name, ticker in all_tickers.items():
            csv_path = self.cache_dir / f"{name}.csv"
            try:
                existing = None
                dl_start = start_dt
                if csv_path.exists():
                    existing = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
                    if len(existing) > 0:
                        last_date = existing.index.max()
                        if last_date >= end_dt:
                            self._data[name] = existing.sort_index()
                            continue
                        dl_start = max(start_dt, (last_date - timedelta(days=1)).normalize())
                if dl_start >= end_dt:
                    if existing is not None:
                        self._data[name] = existing.sort_index()
                    continue
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        df = yf.download(
                            ticker,
                            start=dl_start.strftime("%Y-%m-%d"),
                            end=end_dt.strftime("%Y-%m-%d"),
                            interval="1d",
                            progress=False,
                            auto_adjust=True,
                        )
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Close"]].rename(columns={"Close": "close"})
                df.index.name = "Date"
                if existing is not None:
                    df = pd.concat([existing, df]).sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                df.to_csv(csv_path)
                self._data[name] = df
            except Exception as exc:
                logger.warning("Failed to download %s (%s): %s", name, ticker, exc)

    def _load_cached(self):
        all_names = list(MACRO_TICKERS.keys()) + list(SECTOR_INDEX_REGISTRY.keys())
        for name in all_names:
            if name in self._data:
                continue
            csv_path = self.cache_dir / f"{name}.csv"
            if csv_path.exists():
                self._data[name] = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    def _get_close(self, name: str) -> pd.Series:
        self._load_cached()
        if name in self._data and "close" in self._data[name].columns:
            return self._data[name]["close"]
        return pd.Series(dtype=float)

    def _safe_return(self, series: pd.Series) -> pd.Series:
        return series.pct_change().fillna(0).clip(-0.2, 0.2)

    def get_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        self._load_cached()
        features = pd.DataFrame(0.0, index=dates, columns=MARKET_FEATURE_NAMES)

        def aligned(series: pd.Series) -> pd.Series:
            if series.empty:
                return pd.Series(0.0, index=dates)
            return series.reindex(dates, method="ffill").fillna(0.0)

        crude = self._get_close("crude_brent")
        features["crude_oil_return"] = aligned(self._safe_return(crude))
        features["crude_oil_5d_change"] = aligned(crude.pct_change(5).fillna(0).clip(-0.5, 0.5) if len(crude) else pd.Series(dtype=float))

        gold = self._get_close("gold")
        features["gold_return"] = aligned(self._safe_return(gold))

        usdinr = self._get_close("usdinr")
        features["usdinr_change"] = aligned(self._safe_return(usdinr))

        us10y = self._get_close("us10y")
        features["us_10y_yield_change"] = aligned(us10y.diff().fillna(0).clip(-1, 1) / 10.0)

        dxy = self._get_close("dxy")
        features["dxy_change"] = aligned(self._safe_return(dxy))

        nikkei_ret = self._safe_return(self._get_close("nikkei"))
        hsi_ret = self._safe_return(self._get_close("hangseng"))
        shanghai_ret = self._safe_return(self._get_close("shanghai"))
        features["asia_sentiment"] = aligned(pd.DataFrame({"n": nikkei_ret, "h": hsi_ret, "s": shanghai_ret}).mean(axis=1).fillna(0))

        features["dow_overnight_return"] = aligned(self._safe_return(self._get_close("dow")))
        features["nasdaq_overnight_return"] = aligned(self._safe_return(self._get_close("nasdaq")))
        cboe_vix = self._get_close("cboe_vix")
        features["global_volatility_regime"] = aligned((cboe_vix / 100.0).clip(0, 1).fillna(0.2))

        india_vix = self._get_close("india_vix")
        india_vix_ffill = india_vix.reindex(dates, method="ffill")
        features["india_vix_percentile"] = india_vix_ffill.rolling(252, min_periods=20).rank(pct=True).fillna(0.5).clip(0, 1)

        nifty50 = self._get_close("nifty50")
        features["nifty_5d_return"] = aligned(nifty50.pct_change(5).fillna(0).clip(-0.2, 0.2))
        sp500 = self._get_close("sp500")
        features["sp500_overnight_return"] = aligned(self._safe_return(sp500))
        features["commodity_pressure"] = (features["crude_oil_return"] - features["gold_return"]).clip(-0.2, 0.2)
        features["dollar_yield_pressure"] = (
            features["dxy_change"] + features["us_10y_yield_change"] + features["usdinr_change"]
        ).clip(-0.3, 0.3)
        risk_stack = pd.DataFrame(
            {
                "asia": features["asia_sentiment"],
                "dow": features["dow_overnight_return"],
                "nasdaq": features["nasdaq_overnight_return"],
                "sp500": features["sp500_overnight_return"],
                "dxy": -features["dxy_change"],
                "vix": -features["global_volatility_regime"],
            },
            index=dates,
        )
        features["risk_on_signal"] = risk_stack.mean(axis=1).clip(-1, 1)
        return features.fillna(0.0)

    def get_india_market_features(
        self,
        dates: pd.DatetimeIndex,
        *,
        symbol: str | None = None,
        industry: str | None = None,
    ) -> dict[str, pd.Series]:
        self._load_cached()

        def aligned(series: pd.Series) -> pd.Series:
            if series.empty:
                return pd.Series(0.0, index=dates)
            return series.reindex(dates, method="ffill").fillna(0.0)

        nifty = self._get_close("nifty50")
        india_vix = self._get_close("india_vix")
        sp500 = self._get_close("sp500")

        result: dict[str, pd.Series] = {}
        result["nifty_intraday_return"] = aligned(nifty.pct_change().shift(1)).clip(-0.1, 0.1)
        result["vix_level"] = aligned((india_vix / 100.0).clip(0, 1).fillna(0.15))
        result["vix_change"] = aligned(self._safe_return(india_vix)).clip(-1, 1)
        result["market_breadth"] = aligned(nifty.pct_change(5).fillna(0).clip(-0.2, 0.2))
        result["global_cue"] = aligned(self._safe_return(sp500)).clip(-0.1, 0.1)

        sector_context = self.get_sector_context(dates, industry=industry)
        result["sector_intraday_return"] = sector_context["sector_index_prev_return"]
        result["sector_index_prev_return"] = sector_context["sector_index_prev_return"]
        result["sector_index_5d_return"] = sector_context["sector_index_5d_return"]
        result["sector_index_volatility"] = sector_context["sector_index_volatility"]
        result["industry_relative_strength_rank"] = sector_context["industry_relative_strength_rank"]
        result["sector_breadth_proxy"] = sector_context["sector_breadth_proxy"]
        result["secondary_sector_confirmation"] = sector_context["secondary_sector_confirmation"]
        result["stock_vs_sector_1d"] = pd.Series(0.0, index=dates)
        result["stock_vs_sector_5d"] = pd.Series(0.0, index=dates)
        return result

    def get_sector_context(self, dates: pd.DatetimeIndex, *, industry: str | None = None) -> dict[str, pd.Series]:
        self._load_cached()

        def aligned(series: pd.Series) -> pd.Series:
            if series.empty:
                return pd.Series(0.0, index=dates)
            return series.reindex(dates, method="ffill").fillna(0.0)

        primary_key = PRIMARY_INDEX_BY_INDUSTRY.get(industry or "")
        secondary_keys = SECONDARY_INDEX_BY_INDUSTRY.get(industry or "", [])
        primary_close = self._get_close(primary_key) if primary_key else pd.Series(dtype=float)

        sector_prev_return = aligned(primary_close.pct_change().shift(1)).clip(-0.2, 0.2)
        sector_5d_return = aligned(primary_close.pct_change(5).shift(1)).clip(-0.4, 0.4)
        sector_volatility = aligned(primary_close.pct_change().rolling(20, min_periods=5).std().shift(1)).clip(0, 0.2)
        sector_breadth_proxy = sector_5d_return.copy()

        secondary_returns = []
        for key in secondary_keys:
            close = self._get_close(key)
            if close.empty:
                continue
            secondary_returns.append(aligned(close.pct_change().shift(1)))
        if secondary_returns:
            secondary_mean = pd.concat(secondary_returns, axis=1).mean(axis=1).clip(-0.2, 0.2)
            secondary_confirmation = (np.sign(sector_prev_return).replace(0, 0) * np.sign(secondary_mean).replace(0, 0)).clip(-1, 1)
        else:
            secondary_confirmation = pd.Series(0.0, index=dates)

        all_sector_5d = []
        for key in SECTOR_INDEX_REGISTRY:
            close = self._get_close(key)
            if close.empty:
                continue
            series = aligned(close.pct_change(5).shift(1)).rename(key)
            all_sector_5d.append(series)
        if all_sector_5d and primary_key:
            sector_panel = pd.concat(all_sector_5d, axis=1)
            sector_rank = sector_panel.rank(axis=1, pct=True).get(primary_key, pd.Series(0.5, index=dates)).fillna(0.5)
        else:
            sector_rank = pd.Series(0.5, index=dates)

        return {
            "sector_index_prev_return": sector_prev_return.fillna(0.0),
            "sector_index_5d_return": sector_5d_return.fillna(0.0),
            "sector_index_volatility": sector_volatility.fillna(0.0),
            "industry_relative_strength_rank": sector_rank.clip(0, 1),
            "sector_breadth_proxy": sector_breadth_proxy.fillna(0.0),
            "secondary_sector_confirmation": secondary_confirmation.fillna(0.0),
        }
