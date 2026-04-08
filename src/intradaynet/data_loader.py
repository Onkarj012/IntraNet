"""
Data Loader for IntradayNet - Efficient loading with filtering.

Optimized for 16GB Mac:
- Streams data stock-by-stock (not all at once)
- Filters by date and universe on load
- Memory-efficient processing
"""

from pathlib import Path
from typing import Iterator, Optional, List, Dict, Tuple
import logging

import numpy as np
import pandas as pd

from intradaynet.universe import get_universe

logger = logging.getLogger(__name__)


def load_stock_data(
    symbol: str,
    data_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load minute data for a single stock with date filtering.
    
    Args:
        symbol: Stock symbol
        data_dir: Directory containing CSV files
        start_date: Filter data >= this date (YYYY-MM-DD)
        end_date: Filter data <= this date (YYYY-MM-DD)
        
    Returns:
        DataFrame with minute data or None if not found/error
    """
    csv_path = data_dir / f"{symbol}_minute.csv"
    
    if not csv_path.exists():
        return None
    
    try:
        # Use parse_dates and index_col for efficiency
        df = pd.read_csv(
            csv_path,
            parse_dates=["date"],
            index_col="date",
            usecols=["date", "open", "high", "low", "close", "volume"],
        )
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Filter by date
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if len(df) == 0:
            return None
        
        return df
        
    except Exception as e:
        logger.warning(f"Error loading {symbol}: {e}")
        return None


def resample_to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute data to daily OHLCV.
    
    Args:
        minute_df: DataFrame with minute-level data
        
    Returns:
        DataFrame with daily OHLCV
    """
    daily = minute_df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    
    # Drop days with no data
    daily = daily.dropna()
    
    return daily


def load_universe_data(
    universe: str = "nifty100",
    data_dir: Path = Path("nifty500"),
    start_date: str = "2021-01-01",
    end_date: Optional[str] = None,
    min_days: int = 100,
) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Load data for all stocks in a universe.
    
    Yields tuples of (symbol, minute_df, daily_df) for each stock
    that has sufficient data.
    
    Args:
        universe: 'nifty50', 'nifty100', or 'nifty200'
        data_dir: Directory containing CSV files
        start_date: Minimum date to include
        end_date: Maximum date to include
        min_days: Minimum number of trading days required
        
    Yields:
        (symbol, minute_df, daily_df) tuples
    """
    symbols = get_universe(universe)
    
    logger.info(f"Loading {universe} ({len(symbols)} stocks) from {start_date}...")
    
    loaded = 0
    skipped = 0
    
    for symbol in symbols:
        minute_df = load_stock_data(symbol, data_dir, start_date, end_date)
        
        if minute_df is None:
            skipped += 1
            continue
        
        # Resample to daily
        daily_df = resample_to_daily(minute_df)
        
        if len(daily_df) < min_days:
            skipped += 1
            continue
        
        loaded += 1
        yield symbol, minute_df, daily_df
    
    logger.info(f"Loaded {loaded} stocks, skipped {skipped}")


def compute_returns(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add return columns to daily dataframe."""
    df = daily_df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["overnight_gap"] = df["open"] / df["close"].shift(1) - 1
    df["intraday_return"] = df["close"] / df["open"] - 1
    return df


def get_market_regime(daily_df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """
    Compute market regime indicators from daily data.
    
    Returns dict with:
    - volatility: Annualized volatility
    - trend: Recent trend direction
    - is_high_vol: Whether in high volatility regime
    """
    if len(daily_df) < lookback:
        return {"volatility": 0.2, "trend": 0, "is_high_vol": False}
    
    returns = daily_df["close"].pct_change().dropna()
    recent_returns = returns.tail(lookback)
    
    # Annualized volatility
    vol = recent_returns.std() * np.sqrt(252)
    
    # Trend (slope of log prices)
    prices = daily_df["close"].tail(lookback)
    log_prices = np.log(prices)
    x = np.arange(len(log_prices))
    trend = np.polyfit(x, log_prices, 1)[0] * len(log_prices)
    
    return {
        "volatility": vol,
        "trend": trend,
        "is_high_vol": vol > 0.25,  # 25% annualized
    }


class DataPipeline:
    """
    Main data pipeline for loading and preprocessing data.
    
    Usage:
        pipeline = DataPipeline(universe="nifty100", start_date="2021-01-01")
        for symbol, minute_df, daily_df in pipeline.load_all():
            # Process each stock
            features = compute_features(minute_df, daily_df)
            targets = compute_targets(daily_df)
    """
    
    def __init__(
        self,
        universe: str = "nifty100",
        data_dir: str = "nifty500",
        start_date: str = "2021-01-01",
        end_date: Optional[str] = None,
        min_days: int = 100,
    ):
        self.universe = universe
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.min_days = min_days
        
    def load_all(self) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
        """Load all stocks in universe."""
        return load_universe_data(
            universe=self.universe,
            data_dir=self.data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
            min_days=self.min_days,
        )
    
    def load_single(self, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load a single stock."""
        minute_df = load_stock_data(
            symbol, self.data_dir, self.start_date, self.end_date
        )
        if minute_df is None:
            return None
        
        daily_df = resample_to_daily(minute_df)
        if len(daily_df) < self.min_days:
            return None
        
        return minute_df, daily_df
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded universe."""
        stats = {
            "universe": self.universe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "stocks": [],
            "total_days": 0,
            "avg_days": 0,
        }
        
        for symbol, minute_df, daily_df in self.load_all():
            stats["stocks"].append(symbol)
            stats["total_days"] += len(daily_df)
        
        if stats["stocks"]:
            stats["avg_days"] = stats["total_days"] // len(stats["stocks"])
        
        return stats
