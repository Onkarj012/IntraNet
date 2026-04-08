#!/usr/bin/env python3
"""
Point-in-Time Feature Engine - Zero Look-Ahead Bias.

For any timestamp T, features use ONLY data from <= T.

Two modes:
1. Pre-market (before 9:15): Uses yesterday's close + overnight data
2. Intraday (after 9:15): Uses data up to current bar

Usage:
    from pit_features import PointInTimeFeatures
    
    # Pre-market prediction (8:30 AM)
    features = pit.get_premarket_features(symbol, date)
    
    # Intraday update (9:30 AM)
    features = pit.get_intraday_features(symbol, date, current_time="09:30")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.features.sentiment_features import SentimentFeatureBuilder


@dataclass
class PITFeatures:
    """Point-in-time feature container."""
    timestamp: pd.Timestamp
    is_premarket: bool
    features: Dict[str, float]
    

class PointInTimeFeatureEngine:
    """
    Computes features with strict causality - no future data.
    
    For live trading:
    - At 8:45 AM: Use yesterday's data + overnight markets
    - At 9:30 AM: Use first 15 mins of today's data
    - At 10:00 AM: Use first 45 mins of today's data
    - And so on...
    """
    
    def __init__(self, data_dir: str = "nifty500"):
        self.data_dir = Path(data_dir)
        self.market_builder = MarketFeatureBuilder()
        self.sentiment_builder = None  # Lazy load
        
        # Cache for loaded data
        self._price_cache = {}
        self._market_cache = {}
        
    def _load_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and cache price data for a symbol."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        
        csv_path = self.data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            return None
        
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = df.columns.str.lower()
        self._price_cache[symbol] = df
        return df
    
    def get_premarket_features(self, symbol: str, trade_date: str) -> Optional[PITFeatures]:
        """
        Get features available BEFORE market open (9:15 AM).
        
        Uses:
        - Yesterday's full day data (close, high, low, volume)
        - Yesterday's intraday features (calculated EOD)
        - Overnight global markets (US, Asia)
        - Pre-market sentiment
        """
        trade_date = pd.to_datetime(trade_date)
        yesterday = trade_date - pd.Timedelta(days=1)
        
        # Load price data
        df = self._load_price_data(symbol)
        if df is None:
            return None
        
        # Get yesterday's data (complete, available before 9:15)
        yesterday_data = df[df.index.date == yesterday.date()]
        if len(yesterday_data) < 30:
            return None
        
        # Get yesterday's daily bar (computed EOD yesterday)
        yesterday_daily = yesterday_data.resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).iloc[0]
        
        features = {}
        
        # === YESTERDAY'S CLOSE DATA (available pre-market) ===
        features["yest_close"] = yesterday_daily["close"]
        features["yest_high"] = yesterday_daily["high"]
        features["yest_low"] = yesterday_daily["low"]
        features["yest_volume"] = yesterday_daily["volume"]
        features["yest_range"] = (yesterday_daily["high"] - yesterday_daily["low"]) / yesterday_daily["open"]
        
        # Yesterday's return (close-to-close)
        yest_prev_close = self._get_previous_close(df, yesterday)
        if yest_prev_close:
            features["yest_return"] = (yesterday_daily["close"] - yest_prev_close) / yest_prev_close
        else:
            features["yest_return"] = 0
        
        # === YESTERDAY'S INTRADAY FEATURES (computed EOD) ===
        # VWAP (calculated after market close yesterday)
        yesterday_data["tp"] = (yesterday_data["high"] + yesterday_data["low"] + yesterday_data["close"]) / 3
        yesterday_data["tpv"] = yesterday_data["tp"] * yesterday_data["volume"]
        vwap_yest = yesterday_data["tpv"].sum() / yesterday_data["volume"].sum()
        features["yest_close_vs_vwap"] = (yesterday_daily["close"] - vwap_yest) / vwap_yest
        
        # Intraday momentum (yesterday)
        features["yest_first_hour"] = self._first_hour_return(yesterday_data)
        features["yest_last_hour"] = self._last_hour_return(yesterday_data)
        
        # === HISTORICAL ROLLING FEATURES (up to yesterday) ===
        # Get last 20 days of daily data for rolling calculations
        hist_data = df[df.index < trade_date].resample("D").agg({
            "close": "last",
            "high": "max",
            "low": "min",
            "volume": "sum",
        }).dropna().tail(20)
        
        if len(hist_data) >= 10:
            hist_returns = hist_data["close"].pct_change().dropna()
            features["volatility_20d"] = hist_returns.std()
            features["return_5d"] = (hist_data["close"].iloc[-1] / hist_data["close"].iloc[-5] - 1) if len(hist_data) >= 5 else 0
            features["return_10d"] = (hist_data["close"].iloc[-1] / hist_data["close"].iloc[-10] - 1) if len(hist_data) >= 10 else 0
        else:
            features["volatility_20d"] = 0.02
            features["return_5d"] = 0
            features["return_10d"] = 0
        
        # === OVERNIGHT MARKET DATA (available pre-market) ===
        market_feats = self.market_builder.get_features(pd.DatetimeIndex([trade_date]))
        for col in market_feats.columns:
            features[f"market_{col}"] = market_feats[col].iloc[0]
        
        # India market features
        india_feats = self.market_builder.get_india_market_features(pd.DatetimeIndex([trade_date]))
        for key, series in india_feats.items():
            features[f"india_{key}"] = series.iloc[0]
        
        # === SENTIMENT (pre-market available) ===
        if self.sentiment_builder is None:
            self.sentiment_builder = SentimentFeatureBuilder(
                "sentiment/combined_sentiment_2015_2025.csv",
                market_builder=self.market_builder
            )
        
        sent_feats = self.sentiment_builder.get_features(symbol, pd.DatetimeIndex([trade_date]))
        for col in sent_feats.columns:
            features[f"sentiment_{col}"] = sent_feats[col].iloc[0]
        
        return PITFeatures(
            timestamp=pd.Timestamp(trade_date.strftime("%Y-%m-%d") + " 08:45:00"),
            is_premarket=True,
            features=features
        )
    
    def get_intraday_features(self, symbol: str, trade_date: str, 
                             current_time: str) -> Optional[PITFeatures]:
        """
        Get features at specific intraday time.
        
        Uses data up to current_time only - no future data.
        """
        trade_date = pd.to_datetime(trade_date)
        current_dt = pd.to_datetime(f"{trade_date.date()} {current_time}")
        
        # Load price data
        df = self._load_price_data(symbol)
        if df is None:
            return None
        
        # Get data up to current time (strictly <=)
        intraday_data = df[df.index <= current_dt]
        today_data = intraday_data[intraday_data.index.date == trade_date.date()]
        
        if len(today_data) < 5:
            # Not enough data yet, use pre-market features
            return self.get_premarket_features(symbol, trade_date)
        
        # Get yesterday's pre-market features as base
        pit = self.get_premarket_features(symbol, trade_date)
        if pit is None:
            return None
        
        features = pit.features.copy()
        
        # === INTRADAY FEATURES (up to current time) ===
        features["bars_since_open"] = len(today_data)
        
        # Current price relative to open
        today_open = today_data["open"].iloc[0]
        current_price = today_data["close"].iloc[-1]
        features["current_vs_open"] = (current_price - today_open) / today_open
        
        # Intraday VWAP (up to current time only!)
        today_data["tp"] = (today_data["high"] + today_data["low"] + today_data["close"]) / 3
        today_data["tpv"] = today_data["tp"] * today_data["volume"]
        vwap_current = today_data["tpv"].sum() / today_data["volume"].sum()
        features["current_vs_vwap"] = (current_price - vwap_current) / vwap_current
        
        # Intraday range
        features["intraday_high"] = today_data["high"].max()
        features["intraday_low"] = today_data["low"].min()
        features["intraday_range"] = (features["intraday_high"] - features["intraday_low"]) / today_open
        
        # Volume pace
        current_volume = today_data["volume"].sum()
        avg_volume = features.get("yest_volume", current_volume)
        features["volume_pace"] = current_volume / (avg_volume * (len(today_data) / 375)) - 1
        
        # Momentum within day
        if len(today_data) >= 15:
            first_15 = today_data.head(15)["close"].iloc[-1]
            features["first_15min_return"] = (first_15 - today_open) / today_open
        else:
            features["first_15min_return"] = features["current_vs_open"]
        
        return PITFeatures(
            timestamp=current_dt,
            is_premarket=False,
            features=features
        )
    
    def _get_previous_close(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        """Get previous trading day's close."""
        prev_data = df[df.index.date < date.date()].tail(1)
        return prev_data["close"].iloc[0] if len(prev_data) > 0 else None
    
    def _first_hour_return(self, day_data: pd.DataFrame) -> float:
        """Calculate first hour return (first 60 bars)."""
        if len(day_data) < 60:
            return 0
        open_price = day_data["open"].iloc[0]
        first_hour_close = day_data["close"].iloc[59]
        return (first_hour_close - open_price) / open_price
    
    def _last_hour_return(self, day_data: pd.DataFrame) -> float:
        """Calculate last hour return (last 60 bars)."""
        if len(day_data) < 60:
            return 0
        last_hour_open = day_data["open"].iloc[-60]
        close_price = day_data["close"].iloc[-1]
        return (close_price - last_hour_open) / last_hour_open
