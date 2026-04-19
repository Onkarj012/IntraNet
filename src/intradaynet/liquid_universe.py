"""
Liquid Universe Filter for IntradayNet v3.0

Filters Nifty 500 down to 120-150 liquid stocks based on:
1. Average daily turnover > ₹10 Cr (₹100M) over trailing 60 days
2. Average bid-ask spread < 0.15% (requires L2 data or estimated from OHLC)
3. Minimum 200 trading days of history

Filters are recomputed monthly to handle changing liquidity conditions.

Usage:
    from intradaynet.liquid_universe import LiquidUniverseFilter
    
    filter = LiquidUniverseFilter(data_dir="nifty500")
    liquid_stocks = filter.get_liquid_universe(as_of_date="2025-01-15")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a single stock."""
    symbol: str
    avg_daily_turnover: float  # in rupees
    avg_bid_ask_spread_pct: float  # estimated from high-low vs close
    trading_days_count: int
    avg_daily_volume: float
    price_volatility_20d: float
    
    @property
    def is_liquid(self) -> bool:
        """Check if stock meets all liquidity criteria."""
        return (
            self.avg_daily_turnover >= 100_000_000  # ₹10 Cr
            and self.avg_bid_ask_spread_pct < 0.15  # 0.15%
            and self.trading_days_count >= 200
        )
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "avg_daily_turnover_cr": self.avg_daily_turnover / 10_000_000,
            "avg_bid_ask_spread_pct": self.avg_bid_ask_spread_pct * 100,
            "trading_days_count": self.trading_days_count,
            "avg_daily_volume": self.avg_daily_volume,
            "price_volatility_20d": self.price_volatility_20d,
            "is_liquid": self.is_liquid,
        }


class LiquidUniverseFilter:
    """
    Filters stocks based on liquidity criteria.
    
    Recomputes monthly - a stock liquid in 2023 may not be liquid in 2025.
    """
    
    # Minimum criteria
    MIN_TURNOVER_RUPEES = 100_000_000  # ₹10 Cr
    MAX_SPREAD_PCT = 0.0015  # 0.15%
    MIN_TRADING_DAYS = 200
    
    # Lookback window for metrics
    LOOKBACK_DAYS = 60
    
    def __init__(self, data_dir: str = "nifty500", cache_dir: str = "liquid_universe_cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _estimate_spread_from_ohlc(self, df: pd.DataFrame) -> float:
        """
        Estimate bid-ask spread from OHLC data.
        
        Uses the high-low range relative to close as a proxy for spread.
        This is a conservative estimate since true spread requires L2 data.
        
        Formula: avg((high - low) / close) across all minute bars
        """
        if len(df) == 0:
            return float('inf')
        
        # Use intraday bars to estimate spread
        hl_range = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        return float(hl_range.mean())
    
    def _compute_metrics_for_stock(
        self, 
        symbol: str, 
        end_date: str,
        lookback_days: int = LOOKBACK_DAYS
    ) -> Optional[LiquidityMetrics]:
        """Compute liquidity metrics for a single stock."""
        csv_path = self.data_dir / f"{symbol}_minute.csv"
        
        if not csv_path.exists():
            return None
        
        try:
            # Load data
            df = pd.read_csv(csv_path, parse_dates=['date'])
            df.columns = df.columns.str.lower()
            df = df.set_index('date')
            
            # Filter to lookback period
            end_dt = pd.Timestamp(end_date)
            start_dt = end_dt - pd.Timedelta(days=lookback_days * 1.5)  # Extra buffer for weekends/holidays
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            if len(df) < 100:  # Need sufficient data
                return None
            
            # Resample to daily for turnover calculation
            daily = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(daily) < 30:  # Need at least a month of data
                return None
            
            # Daily turnover = volume * close price
            daily['turnover'] = daily['volume'] * daily['close']
            
            # Compute metrics
            avg_daily_turnover = daily['turnover'].mean()
            avg_daily_volume = daily['volume'].mean()
            trading_days_count = len(daily)
            
            # Estimate spread from intraday data
            avg_spread_pct = self._estimate_spread_from_ohlc(df)
            
            # 20-day volatility
            daily['return'] = daily['close'].pct_change()
            price_volatility_20d = daily['return'].tail(20).std()
            
            return LiquidityMetrics(
                symbol=symbol,
                avg_daily_turnover=avg_daily_turnover,
                avg_bid_ask_spread_pct=avg_spread_pct,
                trading_days_count=trading_days_count,
                avg_daily_volume=avg_daily_volume,
                price_volatility_20d=price_volatility_20d,
            )
            
        except Exception as e:
            return None
    
    def get_liquid_universe(
        self, 
        as_of_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        max_stocks: int = 150,
        min_stocks: int = 120,
        use_cache: bool = True
    ) -> List[str]:
        """
        Get list of liquid stocks as of a given date.
        
        Args:
            as_of_date: Date string (YYYY-MM-DD). If None, uses today.
            symbols: List of symbols to consider. If None, scans data_dir.
            max_stocks: Maximum number of stocks to return
            min_stocks: Minimum number of stocks to return (warns if below)
            use_cache: Whether to use cached results
            
        Returns:
            List of liquid stock symbols, sorted by turnover (highest first)
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache
        cache_file = self.cache_dir / f"liquid_universe_{as_of_date}.json"
        if use_cache and cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                return cached['liquid_symbols']
        
        # Get list of symbols to analyze
        if symbols is None:
            symbols = sorted([
                p.stem.replace('_minute', '')
                for p in self.data_dir.glob('*_minute.csv')
            ])
        
        # Compute metrics for all stocks
        all_metrics: List[LiquidityMetrics] = []
        
        for symbol in symbols:
            metrics = self._compute_metrics_for_stock(symbol, as_of_date)
            if metrics and metrics.is_liquid:
                all_metrics.append(metrics)
        
        # Sort by turnover (descending)
        all_metrics.sort(key=lambda x: x.avg_daily_turnover, reverse=True)
        
        # Take top N
        selected = all_metrics[:max_stocks]
        liquid_symbols = [m.symbol for m in selected]
        
        # Warn if below minimum
        if len(liquid_symbols) < min_stocks:
            print(f"WARNING: Only {len(liquid_symbols)} liquid stocks found (min: {min_stocks})")
        
        # Save cache
        result = {
            'as_of_date': as_of_date,
            'total_scanned': len(symbols),
            'liquid_count': len(liquid_symbols),
            'liquid_symbols': liquid_symbols,
            'metrics': [m.to_dict() for m in selected],
        }
        
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return liquid_symbols
    
    def get_universe_for_period(
        self,
        start_date: str,
        end_date: str,
        rebalance_freq: str = 'MS',  # Month Start
        max_stocks: int = 150,
    ) -> Dict[str, List[str]]:
        """
        Get liquid universe for each rebalancing date in a period.
        
        Returns dict mapping date -> list of liquid symbols.
        This handles survivorship bias by using as-of dates.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
        
        universe_by_date = {}
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            universe_by_date[date_str] = self.get_liquid_universe(
                as_of_date=date_str,
                max_stocks=max_stocks
            )
        
        return universe_by_date
    
    def analyze_universe_evolution(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Analyze how the liquid universe changes over time.
        
        Returns DataFrame with dates as rows and metrics as columns.
        """
        universe_by_date = self.get_universe_for_period(start_date, end_date)
        
        records = []
        all_symbols = set()
        
        for date, symbols in universe_by_date.items():
            all_symbols.update(symbols)
            records.append({
                'date': date,
                'count': len(symbols),
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Compute turnover (symbols entering/leaving)
        prev_set = set()
        entries = []
        exits = []
        
        for date, symbols in universe_by_date.items():
            curr_set = set(symbols)
            entries.append(len(curr_set - prev_set))
            exits.append(len(prev_set - curr_set))
            prev_set = curr_set
        
        df['new_entries'] = [0] + entries[1:]
        df['exits'] = [0] + exits[1:]
        
        return df


def main():
    """CLI for testing the liquid universe filter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Liquid Universe Filter")
    parser.add_argument("--as-of", type=str, help="Date to compute universe (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    parser.add_argument("--max-stocks", type=int, default=150)
    parser.add_argument("--analyze", action="store_true", help="Analyze universe evolution")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    
    args = parser.parse_args()
    
    filter = LiquidUniverseFilter(data_dir=args.data_dir)
    
    if args.analyze:
        print("Analyzing universe evolution...")
        df = filter.analyze_universe_evolution(args.start, args.end)
        print(f"\nUniverse Evolution Summary:")
        print(f"  Average stocks per month: {df['count'].mean():.1f}")
        print(f"  Min: {df['count'].min()}, Max: {df['count'].max()}")
        print(f"  Average monthly entries: {df['new_entries'].mean():.1f}")
        print(f"  Average monthly exits: {df['exits'].mean():.1f}")
        print(f"\nLast 12 months:")
        print(df.tail(12).to_string(index=False))
    else:
        as_of = args.as_of or datetime.now().strftime('%Y-%m-%d')
        print(f"Computing liquid universe as of {as_of}...")
        
        symbols = filter.get_liquid_universe(
            as_of_date=as_of,
            max_stocks=args.max_stocks
        )
        
        print(f"\nLiquid Universe ({len(symbols)} stocks):")
        for i, sym in enumerate(symbols, 1):
            print(f"  {i:3d}. {sym}")
        
        # Show cache location
        cache_file = filter.cache_dir / f"liquid_universe_{as_of}.json"
        print(f"\nCached to: {cache_file}")


if __name__ == "__main__":
    main()
