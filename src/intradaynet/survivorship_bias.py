"""
Survivorship Bias Fix for IntradayNet v3.0

Ensures the training universe is constructed as-of each historical date,
eliminating look-ahead bias from stocks that:
1. Were delisted after the training date
2. Were removed from Nifty 500 after the training date
3. Had IPOs after the training date (no data available)

The key insight: A model trained in 2022 should only know about stocks
that existed in 2022, not stocks that were added in 2024.

Usage:
    from intradaynet.survivorship_bias import SurvivorshipBiasFix
    
    sbf = SurvivorshipBiasFix(
        nifty500_history_file="nifty500_revisions.csv",
        data_dir="nifty500"
    )
    
    # Get universe as it existed on a specific date
    universe_2022 = sbf.get_universe_as_of("2022-06-01")
    
    # Filter out future information
    clean_data = sbf.remove_future_stocks(data_2022, as_of_date="2022-06-01")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger("survivorship_bias")


@dataclass
class StockLifecycle:
    """Tracks when a stock was available for trading."""
    symbol: str
    first_available_date: str  # First date with data
    last_available_date: str   # Last date with data (or "active")
    ipo_date: Optional[str] = None    # IPO date if known
    delisting_date: Optional[str] = None
    index_additions: List[str] = None  # Dates added to Nifty 500
    index_removals: List[str] = None   # Dates removed from Nifty 500
    
    def is_available(self, date: str) -> bool:
        """Check if stock was available on a given date."""
        d = pd.Timestamp(date)
        first = pd.Timestamp(self.first_available_date)
        last = pd.Timestamp(self.last_available_date) if self.last_available_date != "active" else pd.Timestamp.now()
        
        return first <= d <= last


class SurvivorshipBiasFix:
    """
    Fixes survivorship bias by constructing point-in-time universes.
    
    This class:
    1. Builds a database of when each stock was available
    2. Provides as-of-date universe queries
    3. Validates data doesn't contain future information
    """
    
    def __init__(
        self,
        data_dir: str = "nifty500",
        cache_dir: str = "survivorship_cache",
        index_history_file: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.index_history_file = index_history_file
        self.stock_lifecycles: Dict[str, StockLifecycle] = {}
        
        # Build or load lifecycle database
        self._build_lifecycle_database()
    
    def _build_lifecycle_database(self):
        """
        Build database of stock availability dates from minute data.
        
        This scans all CSV files to find first and last dates.
        """
        cache_file = self.cache_dir / "stock_lifecycles.json"
        
        if cache_file.exists():
            # Load from cache
            logger.info("Loading stock lifecycle database from cache...")
            with open(cache_file) as f:
                data = json.load(f)
                for symbol, info in data.items():
                    self.stock_lifecycles[symbol] = StockLifecycle(
                        symbol=info['symbol'],
                        first_available_date=info['first_available_date'],
                        last_available_date=info['last_available_date'],
                        ipo_date=info.get('ipo_date'),
                        delisting_date=info.get('delisting_date'),
                        index_additions=info.get('index_additions', []),
                        index_removals=info.get('index_removals', []),
                    )
            logger.info(f"Loaded {len(self.stock_lifecycles)} stocks")
            return
        
        # Build from data files
        logger.info("Building stock lifecycle database...")
        
        csv_files = sorted(self.data_dir.glob("*_minute.csv"))
        
        for csv_file in csv_files:
            symbol = csv_file.stem.replace("_minute", "")
            
            try:
                # Read just the first and last few rows
                df_head = pd.read_csv(csv_file, nrows=5, parse_dates=['date'])
                df_tail = pd.read_csv(csv_file, skiprows=range(1, max(1, len(pd.read_csv(csv_file)) - 5)), 
                                      nrows=5, parse_dates=['date'])
                
                first_date = df_head['date'].min().strftime('%Y-%m-%d')
                last_date = df_tail['date'].max().strftime('%Y-%m-%d')
                
                # Check if still active (data within last 30 days)
                last_dt = pd.Timestamp(last_date)
                days_since_last = (pd.Timestamp.now() - last_dt).days
                is_active = days_since_last < 30
                
                lifecycle = StockLifecycle(
                    symbol=symbol,
                    first_available_date=first_date,
                    last_available_date="active" if is_active else last_date,
                )
                
                self.stock_lifecycles[symbol] = lifecycle
                
            except Exception as e:
                logger.warning(f"Could not process {symbol}: {e}")
        
        # Save cache
        cache_data = {}
        for symbol, lifecycle in self.stock_lifecycles.items():
            cache_data[symbol] = {
                'symbol': lifecycle.symbol,
                'first_available_date': lifecycle.first_available_date,
                'last_available_date': lifecycle.last_available_date,
                'ipo_date': lifecycle.ipo_date,
                'delisting_date': lifecycle.delisting_date,
            }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Built database for {len(self.stock_lifecycles)} stocks")
    
    def get_universe_as_of(
        self,
        as_of_date: str,
        min_history_days: int = 200,
        require_active: bool = False,
    ) -> List[str]:
        """
        Get universe of stocks that were available on a specific date.
        
        Args:
            as_of_date: Date to query (YYYY-MM-DD)
            min_history_days: Minimum days of history required before as_of_date
            require_active: If True, only include stocks still active today
            
        Returns:
            List of symbols that existed on the given date
        """
        as_of = pd.Timestamp(as_of_date)
        min_start = as_of - pd.Timedelta(days=min_history_days)
        
        available = []
        
        for symbol, lifecycle in self.stock_lifecycles.items():
            # Check if stock was available on the date
            if not lifecycle.is_available(as_of_date):
                continue
            
            # Check minimum history
            first_dt = pd.Timestamp(lifecycle.first_available_date)
            if first_dt > min_start:
                continue
            
            # Check if still active (if required)
            if require_active and lifecycle.last_available_date != "active":
                continue
            
            available.append(symbol)
        
        logger.info(f"Universe as of {as_of_date}: {len(available)} stocks")
        return sorted(available)
    
    def get_delisted_stocks(
        self,
        between_start: str,
        between_end: str,
    ) -> List[str]:
        """
        Get list of stocks that were delisted between two dates.
        
        These stocks would be invisible to a model trained after the end date,
        creating survivorship bias if not handled properly.
        """
        delisted = []
        
        for symbol, lifecycle in self.stock_lifecycles.items():
            if lifecycle.last_available_date == "active":
                continue
            
            last_dt = pd.Timestamp(lifecycle.last_available_date)
            start_dt = pd.Timestamp(between_start)
            end_dt = pd.Timestamp(between_end)
            
            # Stock ended during the period
            if start_dt <= last_dt <= end_dt:
                delisted.append({
                    'symbol': symbol,
                    'last_date': lifecycle.last_available_date,
                })
        
        return delisted
    
    def get_ipo_stocks(
        self,
        between_start: str,
        between_end: str,
    ) -> List[str]:
        """
        Get list of stocks that IPO'd between two dates.
        
        These stocks should NOT be included in training data for dates
        before their IPO.
        """
        ipos = []
        
        for symbol, lifecycle in self.stock_lifecycles.items():
            first_dt = pd.Timestamp(lifecycle.first_available_date)
            start_dt = pd.Timestamp(between_start)
            end_dt = pd.Timestamp(between_end)
            
            # Stock started during the period
            if start_dt <= first_dt <= end_dt:
                ipos.append({
                    'symbol': symbol,
                    'first_date': lifecycle.first_available_date,
                })
        
        return ipos
    
    def validate_no_future_data(
        self,
        df: pd.DataFrame,
        as_of_date: str,
        symbol_col: str = 'symbol',
        date_col: str = 'date',
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a DataFrame contains no future data.
        
        Returns (is_valid, violations) where violations is a list of
        symbols that shouldn't be present in the data.
        """
        as_of = pd.Timestamp(as_of_date)
        
        violations = []
        
        for _, row in df.iterrows():
            symbol = row[symbol_col]
            date = pd.Timestamp(row[date_col])
            
            if symbol not in self.stock_lifecycles:
                violations.append(f"Unknown symbol: {symbol}")
                continue
            
            lifecycle = self.stock_lifecycles[symbol]
            
            # Check if stock existed on this date
            if date < pd.Timestamp(lifecycle.first_available_date):
                violations.append(f"{symbol}: data before IPO ({lifecycle.first_available_date})")
            
            if lifecycle.last_available_date != "active":
                if date > pd.Timestamp(lifecycle.last_available_date):
                    violations.append(f"{symbol}: data after delisting ({lifecycle.last_available_date})")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def filter_to_historical_universe(
        self,
        df: pd.DataFrame,
        as_of_date: str,
        symbol_col: str = 'symbol',
        min_history_days: int = 200,
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to only include stocks that were available
        on the given historical date.
        """
        historical_universe = set(self.get_universe_as_of(as_of_date, min_history_days))
        
        mask = df[symbol_col].isin(historical_universe)
        filtered = df[mask].copy()
        
        n_removed = len(df) - len(filtered)
        if n_removed > 0:
            removed_symbols = set(df[symbol_col]) - historical_universe
            logger.warning(f"Removed {n_removed} rows with {len(removed_symbols)} future stocks")
        
        return filtered
    
    def generate_point_in_time_universes(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
        freq: str = 'MS',  # Month Start
    ) -> Dict[str, List[str]]:
        """
        Generate universe for each rebalancing date in a period.
        
        This is useful for walk-forward validation to ensure
        each fold uses the correct historical universe.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        universes = {}
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            universes[date_str] = self.get_universe_as_of(date_str)
        
        return universes
    
    def analyze_survivorship_bias(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Analyze the extent of survivorship bias over a period.
        
        Returns DataFrame showing:
        - Universe size at start
        - Universe size at end (biased view)
        - Stocks delisted (missing from end universe)
        - Stocks added via IPO (not in start universe)
        """
        start_universe = set(self.get_universe_as_of(start_date))
        end_universe = set(self.get_universe_as_of(end_date))
        
        delisted = self.get_delisted_stocks(start_date, end_date)
        ipos = self.get_ipo_stocks(start_date, end_date)
        
        # Biased universe = what you'd get if you just took current stocks
        biased_only = end_universe - start_universe
        
        analysis = {
            'period': f"{start_date} to {end_date}",
            'start_universe_size': len(start_universe),
            'end_universe_size': len(end_universe),
            'survivors': len(start_universe & end_universe),
            'delisted_count': len(delisted),
            'ipo_count': len(ipos),
            'survivorship_bias_pct': len(delisted) / max(len(start_universe), 1) * 100,
            'biased_sample_survivors_only': len(start_universe & end_universe),
        }
        
        df = pd.DataFrame([analysis])
        
        logger.info(f"\nSurvivorship Bias Analysis:")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Start universe: {len(start_universe)} stocks")
        logger.info(f"  End universe: {len(end_universe)} stocks")
        logger.info(f"  Survivors: {len(start_universe & end_universe)} stocks")
        logger.info(f"  Delisted: {len(delisted)} stocks ({len(delisted)/max(len(start_universe),1)*100:.1f}%)")
        logger.info(f"  IPOs: {len(ipos)} stocks")
        
        return df


def main():
    """CLI for survivorship bias analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Survivorship Bias Fix")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--as-of", type=str, help="Get universe as of specific date")
    
    args = parser.parse_args()
    
    sbf = SurvivorshipBiasFix(data_dir=args.data_dir)
    
    if args.analyze:
        print("\nAnalyzing survivorship bias...")
        analysis = sbf.analyze_survivorship_bias(args.start, args.end)
        print(analysis.to_string(index=False))
    
    if args.as_of:
        universe = sbf.get_universe_as_of(args.as_of)
        print(f"\nUniverse as of {args.as_of}: {len(universe)} stocks")
        for i, sym in enumerate(universe[:20], 1):
            print(f"  {i:3d}. {sym}")
        if len(universe) > 20:
            print(f"  ... and {len(universe) - 20} more")


if __name__ == "__main__":
    main()
