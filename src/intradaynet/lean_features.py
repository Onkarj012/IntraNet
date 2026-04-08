"""
Lean feature definitions for IntradayNet Signal-First Rebuild.

Reduced from 625+ features to ~36 essential features based on:
1. Financial theory (what should matter)
2. Computational efficiency (works on 16GB Mac)
3. Interpretability

Categories:
- Price Action (8): End-of-day microstructure for gap prediction
- Market Context (10): Global and local market indicators
- Sentiment (6): News and sentiment flow
- Gap-Specific (4): Historical gap patterns
- Microstructure (8): Volume, flow, and technical indicators
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

# Feature names in order - this is the contract
LEAN_FEATURE_NAMES: List[str] = [
    # === Price Action (8) ===
    "overnight_gap",              # Previous day's gap
    "prev_day_return",            # Previous day close-to-close
    "prev_day_volatility",        # 21-day realized volatility
    "price_vs_vwap",              # Last price vs VWAP
    "volume_pace",                # Last 30min volume vs average
    "rsi_14",                     # RSI at previous close
    "bb_position",                # Bollinger band position (-1 to 1)
    "day_trend_strength",         # Linear slope of last hour
    
    # === Market Context (10) ===
    "vix_level",                  # VIX level
    "vix_change",                 # VIX change from prev day
    "nifty_prev_return",          # Nifty previous day return
    "nifty_vs_sector",            # Stock vs sector relative strength
    "market_breadth",             # Advance-decline ratio
    "crude_change",               # Crude oil overnight change
    "usdinr_change",              # USD/INR change
    "dxy_change",                 # Dollar index change
    "us_10y_yield",               # US 10Y yield change
    "asia_overnight",             # Asian markets composite
    
    # === Sentiment (6) ===
    "sentiment_5d_avg",           # 5-day average sentiment
    "sentiment_spike",            # Unusual sentiment activity
    "sentiment_momentum",         # Sentiment trend
    "premarket_sentiment",        # Today's pre-market sentiment
    "news_volume",                # Number of news items
    "sentiment_price_div",        # Sentiment-price divergence
    
    # === Gap-Specific (4) ===
    "prev_gap_size",              # Yesterday's gap magnitude
    "prev_gap_filled",            # Whether yesterday's gap filled
    "earnings_proximity",         # Days to/from earnings
    "expiry_flag",                # F&O expiry day proximity
    
    # === Microstructure (8) ===
    "last_hour_trend",            # Return in last hour
    "close_vs_day_high",          # Distance from day's high
    "volume_concentration",       # Volume in last hour vs day
    "spread_trend",               # Bid-ask spread trend
    "obv_slope",                  # On-balance volume slope
    "vol_momentum",               # Volume momentum
    "momentum_5_20",              # Short vs medium momentum
    "support_distance",           # Distance to nearest support
]


def compute_lean_features(
    minute_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    market_data: dict,
    sentiment_data: dict,
) -> pd.DataFrame:
    """
    Compute all 36 lean features from available data.
    
    Args:
        minute_df: Last day's minute bars (for microstructure)
        daily_df: Historical daily bars (for trends, volatility)
        market_data: Dict with VIX, Nifty, global market data
        sentiment_data: Dict with sentiment scores
        
    Returns:
        DataFrame with single row of 36 features
    """
    features = {}
    
    # === Price Action (from daily_df) ===
    if len(daily_df) >= 2:
        prev_close = daily_df["close"].iloc[-2]
        prev_open = daily_df["open"].iloc[-1]
        last_close = daily_df["close"].iloc[-1]
        
        # Overnight gap from previous day
        features["overnight_gap"] = (prev_open / prev_close - 1) if prev_close > 0 else 0
        
        # Previous day return
        prev_prev_close = daily_df["close"].iloc[-3] if len(daily_df) >= 3 else prev_close
        features["prev_day_return"] = (prev_close / prev_prev_close - 1) if prev_prev_close > 0 else 0
        
        # 21-day volatility
        if len(daily_df) >= 21:
            returns = daily_df["close"].pct_change().dropna()
            features["prev_day_volatility"] = returns.tail(21).std()
        else:
            features["prev_day_volatility"] = 0
    else:
        features["overnight_gap"] = 0
        features["prev_day_return"] = 0
        features["prev_day_volatility"] = 0
    
    # === Price Action (from minute_df - last day only) ===
    if len(minute_df) > 0:
        c = minute_df["close"].values
        v = minute_df["volume"].values
        
        # VWAP calculation
        h = minute_df["high"].values
        l = minute_df["low"].values
        tp = (h + l + c) / 3
        vwap = np.cumsum(tp * v) / np.cumsum(v)
        
        features["price_vs_vwap"] = (c[-1] / vwap[-1] - 1) if vwap[-1] > 0 else 0
        
        # Volume pace (last 30 min vs day average)
        if len(v) >= 30:
            last_30_vol = v[-30:].mean()
            day_avg_vol = v.mean()
            features["volume_pace"] = (last_30_vol / day_avg_vol - 1) if day_avg_vol > 0 else 0
        else:
            features["volume_pace"] = 0
        
        # RSI at close (simplified)
        if len(c) >= 15:
            deltas = np.diff(c)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
            features["rsi_14"] = rsi / 100  # Normalize to 0-1
        else:
            features["rsi_14"] = 0.5
        
        # Bollinger band position
        if len(c) >= 20:
            sma20 = np.mean(c[-20:])
            std20 = np.std(c[-20:])
            if std20 > 0:
                features["bb_position"] = ((c[-1] - sma20) / (2 * std20))
            else:
                features["bb_position"] = 0
        else:
            features["bb_position"] = 0
        
        # Last hour trend
        if len(c) >= 60:
            last_hour_return = (c[-1] / c[-60] - 1)
            features["day_trend_strength"] = last_hour_return
            features["last_hour_trend"] = last_hour_return
        else:
            features["day_trend_strength"] = 0
            features["last_hour_trend"] = 0
        
        # Close vs day high
        day_high = np.max(c)
        if day_high > 0:
            features["close_vs_day_high"] = (c[-1] / day_high - 1)
        else:
            features["close_vs_day_high"] = 0
        
        # Volume concentration (last hour vs day)
        if len(v) >= 60:
            last_hour_vol = np.sum(v[-60:])
            total_vol = np.sum(v)
            features["volume_concentration"] = (last_hour_vol / total_vol) if total_vol > 0 else 0.25
        else:
            features["volume_concentration"] = 0.25
        
        # OBV slope
        obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v)
        if len(obv) >= 20:
            x = np.arange(len(obv[-20:]))
            y = obv[-20:]
            slope = np.polyfit(x, y, 1)[0]
            features["obv_slope"] = np.tanh(slope / 10000)  # Normalize
        else:
            features["obv_slope"] = 0
        
        # Volume momentum
        if len(v) >= 20:
            recent_vol = np.mean(v[-5:])
            older_vol = np.mean(v[-20:-5])
            features["vol_momentum"] = (recent_vol / older_vol - 1) if older_vol > 0 else 0
        else:
            features["vol_momentum"] = 0
        
        # Momentum 5 vs 20
        if len(c) >= 20:
            mom5 = (c[-1] / c[-5] - 1) if c[-5] > 0 else 0
            mom20 = (c[-1] / c[-20] - 1) if c[-20] > 0 else 0
            features["momentum_5_20"] = mom5 - mom20
        else:
            features["momentum_5_20"] = 0
        
        # Support distance (simplified: distance from recent low)
        if len(c) >= 20:
            recent_low = np.min(c[-20:])
            features["support_distance"] = (c[-1] / recent_low - 1) if recent_low > 0 else 0
        else:
            features["support_distance"] = 0
    else:
        # Default values if no minute data
        for col in ["price_vs_vwap", "volume_pace", "rsi_14", "bb_position",
                   "day_trend_strength", "last_hour_trend", "close_vs_day_high",
                   "volume_concentration", "obv_slope", "vol_momentum",
                   "momentum_5_20", "support_distance"]:
            features[col] = 0
    
    # === Market Context ===
    features["vix_level"] = market_data.get("vix_level", 18) / 50  # Normalize
    features["vix_change"] = market_data.get("vix_change", 0)
    features["nifty_prev_return"] = market_data.get("nifty_prev_return", 0)
    features["nifty_vs_sector"] = market_data.get("nifty_vs_sector", 0)
    features["market_breadth"] = market_data.get("market_breadth", 0.5)
    features["crude_change"] = market_data.get("crude_change", 0)
    features["usdinr_change"] = market_data.get("usdinr_change", 0)
    features["dxy_change"] = market_data.get("dxy_change", 0)
    features["us_10y_yield"] = market_data.get("us_10y_yield", 0)
    features["asia_overnight"] = market_data.get("asia_overnight", 0)
    
    # === Sentiment ===
    features["sentiment_5d_avg"] = sentiment_data.get("sentiment_5d_avg", 0)
    features["sentiment_spike"] = sentiment_data.get("sentiment_spike", 0)
    features["sentiment_momentum"] = sentiment_data.get("sentiment_momentum", 0)
    features["premarket_sentiment"] = sentiment_data.get("premarket_sentiment", 0)
    features["news_volume"] = sentiment_data.get("news_volume", 0) / 10  # Normalize
    features["sentiment_price_div"] = sentiment_data.get("sentiment_price_div", 0)
    
    # === Gap-Specific ===
    if len(daily_df) >= 2:
        # Yesterday's gap
        yest_open = daily_df["open"].iloc[-1]
        yest_prev_close = daily_df["close"].iloc[-2]
        features["prev_gap_size"] = abs(yest_open / yest_prev_close - 1) if yest_prev_close > 0 else 0
        
        # Did yesterday's gap fill?
        yest_close = daily_df["close"].iloc[-1]
        if yest_open > yest_prev_close:  # Up gap
            features["prev_gap_filled"] = 1 if yest_close <= yest_open else 0
        else:  # Down gap
            features["prev_gap_filled"] = 1 if yest_close >= yest_open else 0
    else:
        features["prev_gap_size"] = 0
        features["prev_gap_filled"] = 0.5  # Unknown
    
    # Earnings proximity (stub - would need actual earnings dates)
    features["earnings_proximity"] = 0  # 0 = not near earnings
    
    # Expiry flag (simplified)
    features["expiry_flag"] = market_data.get("is_expiry_day", 0)
    
    # Clip all features to reasonable ranges
    for key in features:
        if isinstance(features[key], (int, float)):
            features[key] = np.clip(features[key], -5, 5)
    
    return pd.DataFrame([features])[LEAN_FEATURE_NAMES]


def get_feature_count() -> int:
    """Return the number of lean features."""
    return len(LEAN_FEATURE_NAMES)
