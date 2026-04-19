"""
Enhanced Feature Engineering for IntradayNet v3.0 - Phase 2

Adds 18 new features across 4 categories:
1. Microstructure Features (6) - Order flow, volume patterns, entropy
2. Cross-Sectional Features (4) - Sector momentum, relative strength
3. Volatility Regime Features (4) - VIX percentile, gap analysis
4. Options-Derived Features (4) - PCR, IV skew, max pain

Total features: 69 (original) + 18 (new) = 87 features
Target after selection: 60-75 features

Usage:
    from intradaynet.features.v3_features import EnhancedFeatureEngineer
    
    engineer = EnhancedFeatureEngineer()
    features_df = engineer.compute_all_features(
        minute_df=minute_data,
        symbol="RELIANCE",
        sector="ENERGY",
        nifty_df=nifty_data,
        vix_data=vix_series,
        options_data=options_chain,
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from pathlib import Path
import logging

logger = logging.getLogger("intradaynet.features.v3")


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # Microstructure
    relative_volume_window: int = 20  # Days for volume comparison
    tick_imbalance_window: int = 30   # Bars for tick analysis
    entropy_window: int = 30          # Bars for entropy
    correlation_window: int = 30      # Bars for volume-price correlation
    
    # Cross-sectional
    sector_lookback: int = 5          # Days for sector momentum
    nifty_correlation_window: int = 20
    
    # Volatility
    vix_percentile_window: int = 60   # Days for VIX percentile
    gap_zscore_window: int = 60       # Days for gap analysis
    
    # Options
    options_lookback: int = 5         # Days for PCR change


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineer with 18 new v3.0 features.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    # =========================================================================
    # MICROSTRUCTURE FEATURES (6 features)
    # =========================================================================
    
    def compute_relative_volume_15m(
        self, 
        minute_df: pd.DataFrame,
        historical_volumes: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Feature 1: Volume in last 15 mins vs same 15-min window average over past 20 days.
        
        Detects institutional activity - high relative volume = informed trading.
        """
        if len(minute_df) < 15:
            return pd.Series(1.0, index=minute_df.index)
        
        # Current 15-min volume
        current_vol_15m = minute_df['volume'].tail(15).sum()
        
        if historical_volumes is not None and len(historical_volumes) >= self.config.relative_volume_window:
            # Use historical same-window volumes
            avg_hist_vol = historical_volumes.mean()
        else:
            # Use intraday pattern estimate
            avg_hist_vol = minute_df['volume'].mean() * 15
        
        relative_vol = current_vol_15m / max(avg_hist_vol, 1)
        
        return pd.Series(np.clip(relative_vol, 0.1, 50), index=minute_df.index)
    
    def compute_price_acceleration(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 2: Second derivative of price (momentum of momentum).
        
        Catches trend exhaustion before it shows in RSI.
        """
        close = minute_df['close']
        
        # First derivative (momentum)
        momentum = close.diff()
        
        # Second derivative (acceleration)
        acceleration = momentum.diff()
        
        # Normalize by price
        accel_normalized = acceleration / close.replace(0, np.nan) * 1000  # Scale up
        
        return accel_normalized.fillna(0).clip(-50, 50)
    
    def compute_tick_imbalance(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 3: Ratio of upticks to downticks in lookback window.
        
        Proxy for order flow without L2 data.
        """
        close = minute_df['close']
        
        # Count ticks
        up_ticks = (close.diff() > 0).astype(int)
        down_ticks = (close.diff() < 0).astype(int)
        
        # Rolling sum
        window = self.config.tick_imbalance_window
        up_sum = up_ticks.rolling(window, min_periods=1).sum()
        down_sum = down_ticks.rolling(window, min_periods=1).sum()
        
        # Imbalance ratio
        total_ticks = up_sum + down_sum + 1  # +1 to avoid div by zero
        imbalance = (up_sum - down_sum) / total_ticks
        
        return imbalance.clip(-1, 1)
    
    def compute_bar_entropy(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 4: Shannon entropy of return distribution over last 30 bars.
        
        High entropy = noise (random walk), low entropy = trend (directional).
        """
        returns = minute_df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return pd.Series(1.0, index=minute_df.index)
        
        # Discretize returns into bins for entropy calculation
        window = self.config.entropy_window
        
        def rolling_entropy(x):
            if len(x) < 10:
                return 1.0
            # Create histogram
            hist, _ = np.histogram(x, bins=10, range=(-0.01, 0.01))
            probs = hist / hist.sum() if hist.sum() > 0 else np.ones(10) / 10
            # Shannon entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            # Normalize by max entropy (uniform distribution)
            max_entropy = np.log2(10)
            return entropy / max_entropy if max_entropy > 0 else 1.0
        
        # Apply rolling entropy
        entropy_series = returns.rolling(window, min_periods=10).apply(
            rolling_entropy, raw=True
        )
        
        return entropy_series.reindex(minute_df.index).fillna(1.0)
    
    def compute_volume_price_correlation(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 5: Rolling correlation between volume and absolute returns.
        
        High correlation = informed trading (volume confirms moves).
        """
        close = minute_df['close']
        volume = minute_df['volume']
        
        # Absolute returns
        abs_returns = close.pct_change().abs()
        
        # Rolling correlation
        window = self.config.correlation_window
        
        def rolling_corr(x, y):
            if len(x) < 10:
                return 0.0
            try:
                corr = np.corrcoef(x, y)[0, 1]
                return corr if not np.isnan(corr) else 0.0
            except:
                return 0.0
        
        # Compute rolling correlation manually
        corr_values = []
        for i in range(len(minute_df)):
            if i < window:
                corr_values.append(0.0)
            else:
                x = abs_returns.iloc[i-window:i].values
                y = volume.iloc[i-window:i].values
                corr_values.append(rolling_corr(x, y))
        
        return pd.Series(corr_values, index=minute_df.index).clip(-1, 1)
    
    def compute_consecutive_direction_count(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 6: Number of consecutive bars closing in same direction.
        
        Mean reversion signal when extreme (e.g., 10 consecutive up bars = likely pullback).
        """
        close = minute_df['close']
        returns = close.diff()
        
        # Direction of each bar
        direction = np.sign(returns)
        
        # Count consecutive same direction
        consecutive = []
        current_streak = 0
        current_dir = 0
        
        for d in direction:
            if d == 0:
                consecutive.append(current_streak if current_streak > 0 else 0)
            elif d == current_dir:
                current_streak += 1
                consecutive.append(current_streak)
            else:
                current_streak = 1
                current_dir = d
                consecutive.append(current_streak)
        
        return pd.Series(consecutive, index=minute_df.index).clip(0, 20)
    
    # =========================================================================
    # CROSS-SECTIONAL FEATURES (4 features)
    # =========================================================================
    
    def compute_sector_momentum_rank(
        self,
        symbol: str,
        minute_df: pd.DataFrame,
        sector_returns: Dict[str, pd.Series]
    ) -> float:
        """
        Feature 7: Where does this stock's 5-day return rank within its sector?
        
        Returns 0-1 percentile (top quartile = momentum, bottom = mean reversion candidate).
        """
        if not sector_returns:
            return 0.5
        
        # Calculate 5-day return for this stock
        daily = minute_df.resample('D').last()
        if len(daily) < 6:
            return 0.5
        
        stock_return = (daily['close'].iloc[-1] / daily['close'].iloc[-6]) - 1
        
        # Get sector returns
        sector_rets = []
        for sym, returns_series in sector_returns.items():
            if len(returns_series) > 0:
                sector_rets.append(returns_series.iloc[-1])
        
        if not sector_rets:
            return 0.5
        
        # Calculate percentile
        percentile = stats.percentileofscore(sector_rets, stock_return, kind='rank') / 100
        
        return percentile
    
    def compute_sector_flow_score(
        self,
        sector_symbols: List[str],
        data_dir: Path,
        date: str
    ) -> float:
        """
        Feature 8: Average volume surge across the sector.
        
        Detects sector rotation before individual stock moves.
        """
        volume_surges = []
        
        for sym in sector_symbols[:10]:  # Limit to avoid excessive I/O
            csv_path = data_dir / f"{sym}_minute.csv"
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path, parse_dates=['date'])
                df = df[df['date'].dt.date == pd.Timestamp(date).date()]
                
                if len(df) > 30:
                    # Volume surge vs 20-day average
                    current_vol = df['volume'].mean()
                    # Would need historical data for proper calculation
                    volume_surges.append(1.0)
            except:
                continue
        
        return np.mean(volume_surges) if volume_surges else 1.0
    
    def compute_relative_strength_vs_nifty(
        self,
        minute_df: pd.DataFrame,
        nifty_df: pd.DataFrame,
        window: int = 20
    ) -> float:
        """
        Feature 9: Stock's 20-day return minus Nifty's 20-day return.
        
        Isolates stock-specific alpha from market beta.
        """
        if nifty_df is None or len(nifty_df) < window:
            return 0.0
        
        # Get daily closes
        stock_daily = minute_df.resample('D').last()
        nifty_daily = nifty_df.resample('D').last()
        
        if len(stock_daily) < window or len(nifty_daily) < window:
            return 0.0
        
        stock_return = (stock_daily['close'].iloc[-1] / stock_daily['close'].iloc[-window]) - 1
        nifty_return = (nifty_daily['close'].iloc[-1] / nifty_daily['close'].iloc[-window]) - 1
        
        alpha = stock_return - nifty_return
        
        return alpha
    
    def compute_correlation_to_nifty(
        self,
        minute_df: pd.DataFrame,
        nifty_df: pd.DataFrame,
        window: int = 20
    ) -> float:
        """
        Feature 10: Rolling beta (correlation) to Nifty.
        
        High beta stocks need different stop-loss logic than low beta.
        """
        if nifty_df is None or len(nifty_df) < window * 2:
            return 0.5
        
        # Resample to daily
        stock_daily = minute_df.resample('D').last()['close'].pct_change().dropna()
        nifty_daily = nifty_df.resample('D').last()['close'].pct_change().dropna()
        
        # Align
        aligned = pd.concat([stock_daily, nifty_daily], axis=1).dropna()
        aligned.columns = ['stock', 'nifty']
        
        if len(aligned) < window:
            return 0.5
        
        # Compute correlation
        recent = aligned.tail(window)
        if len(recent) < 10:
            return 0.5
        
        try:
            corr = recent['stock'].corr(recent['nifty'])
            return corr if not np.isnan(corr) else 0.5
        except:
            return 0.5
    
    # =========================================================================
    # VOLATILITY REGIME FEATURES (4 features)
    # =========================================================================
    
    def compute_vix_percentile_60d(
        self,
        vix_series: pd.Series,
        current_date: str
    ) -> float:
        """
        Feature 11: Where current VIX sits relative to last 60 days.
        
        0 = lowest vol in 60 days, 1 = highest vol (extreme fear).
        """
        if vix_series is None or len(vix_series) < 30:
            return 0.5
        
        current_vix = vix_series.iloc[-1]
        window = vix_series.tail(self.config.vix_percentile_window)
        
        if len(window) < 20:
            return 0.5
        
        percentile = stats.percentileofscore(window, current_vix, kind='rank') / 100
        
        return percentile
    
    def compute_realized_vs_implied_vol(
        self,
        minute_df: pd.DataFrame,
        vix_level: float
    ) -> float:
        """
        Feature 12: Realized volatility vs VIX (implied).
        
        If realized > implied, market is underpricing risk - tighten stops.
        """
        # Compute 20-day realized volatility (annualized)
        daily = minute_df.resample('D').last()
        if len(daily) < 20:
            return 1.0
        
        daily_returns = daily['close'].pct_change().dropna()
        realized_vol = daily_returns.tail(20).std() * np.sqrt(252) * 100  # Annualized, in %
        
        # Compare to VIX
        if vix_level > 0:
            ratio = realized_vol / vix_level
        else:
            ratio = 1.0
        
        return ratio
    
    def compute_overnight_gap_zscore(
        self,
        minute_df: pd.DataFrame,
        window: int = 60
    ) -> float:
        """
        Feature 13: How unusual is today's gap vs last 60 days of gaps.
        
        Large z-score = unusual event (earnings, news).
        """
        # Get daily data
        daily = minute_df.resample('D').agg({
            'open': 'first',
            'close': 'last'
        })
        
        if len(daily) < 2:
            return 0.0
        
        # Compute gaps
        daily['prev_close'] = daily['close'].shift(1)
        daily['gap'] = (daily['open'] - daily['prev_close']) / daily['prev_close']
        
        gaps = daily['gap'].dropna()
        
        if len(gaps) < 10:
            return 0.0
        
        current_gap = gaps.iloc[-1]
        
        # Z-score
        mean_gap = gaps.tail(window).mean()
        std_gap = gaps.tail(window).std()
        
        if std_gap > 0:
            zscore = (current_gap - mean_gap) / std_gap
        else:
            zscore = 0.0
        
        return np.clip(zscore, -5, 5)
    
    def compute_intraday_range_percentile(
        self,
        minute_df: pd.DataFrame,
        window: int = 20
    ) -> float:
        """
        Feature 14: Where today's developing range sits vs last 20 days.
        
        Detects range expansion/compression early.
        """
        if len(minute_df) < 30:
            return 0.5
        
        # Current day's range so far
        current_high = minute_df['high'].max()
        current_low = minute_df['low'].min()
        current_range = (current_high - current_low) / minute_df['close'].iloc[0]
        
        # Would need historical daily ranges for proper percentile
        # Simplified version:
        # Estimate from ATR
        from intradaynet.dynamic_targets import DynamicTargetManager
        manager = DynamicTargetManager()
        
        ohlcv = minute_df[['open', 'high', 'low', 'close', 'volume']].values
        atr = manager.compute_atr_from_ohlcv(ohlcv)
        
        current_price = minute_df['close'].iloc[-1]
        atr_pct = atr / current_price
        
        # Normalize to 0-1 based on typical ranges
        # 0.5% range = 0.0, 5% range = 1.0
        normalized = (atr_pct - 0.005) / (0.05 - 0.005)
        
        return np.clip(normalized, 0, 1)
    
    # =========================================================================
    # OPTIONS-DERIVED FEATURES (4 features)
    # =========================================================================
    
    def compute_pcr_change(
        self,
        pcr_series: pd.Series,
        window: int = 5
    ) -> float:
        """
        Feature 15: Put-call ratio change from previous day.
        
        Sentiment proxy - rising PCR = bearish sentiment.
        """
        if pcr_series is None or len(pcr_series) < 2:
            return 0.0
        
        current_pcr = pcr_series.iloc[-1]
        prev_pcr = pcr_series.iloc[-2]
        
        if prev_pcr > 0:
            change = (current_pcr - prev_pcr) / prev_pcr
        else:
            change = 0.0
        
        return change
    
    def compute_max_pain_distance(
        self,
        current_price: float,
        max_pain_strike: float
    ) -> float:
        """
        Feature 16: Distance of current price from options max pain.
        
        Gravitational pull effect - price tends toward max pain at expiry.
        """
        if max_pain_strike <= 0 or current_price <= 0:
            return 0.0
        
        distance_pct = (current_price - max_pain_strike) / max_pain_strike
        
        return distance_pct
    
    def compute_iv_skew(
        self,
        otm_put_iv: float,
        otm_call_iv: float
    ) -> float:
        """
        Feature 17: Difference between OTM put IV and OTM call IV.
        
        Fear gauge - high put skew = fear of downside.
        """
        if otm_call_iv <= 0:
            return 0.0
        
        skew = (otm_put_iv - otm_call_iv) / otm_call_iv
        
        return skew
    
    def compute_oi_buildup_signal(
        self,
        current_oi: float,
        prev_oi: float,
        option_type: str = 'CE'  # CE or PE
    ) -> float:
        """
        Feature 18: Net change in open interest for ATM strikes.
        
        Institutional positioning signal.
        """
        if prev_oi <= 0:
            return 0.0
        
        oi_change = (current_oi - prev_oi) / prev_oi
        
        # Signal strength
        # For calls: +OI buildup with -price = short buildup (bearish)
        # For puts: +OI buildup with +price = long buildup (bearish hedge)
        
        return oi_change
    
    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================
    
    def compute_all_features(
        self,
        minute_df: pd.DataFrame,
        symbol: str,
        sector: str = "UNKNOWN",
        nifty_df: Optional[pd.DataFrame] = None,
        vix_data: Optional[pd.Series] = None,
        options_data: Optional[Dict] = None,
        sector_data: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Compute all 18 new v3.0 features.
        
        Returns DataFrame with new features (to be merged with existing 69 features).
        """
        features = pd.DataFrame(index=minute_df.index)
        
        logger.info(f"Computing v3.0 features for {symbol}...")
        
        # Microstructure features (6)
        features['relative_volume_15m'] = self.compute_relative_volume_15m(minute_df)
        features['price_acceleration'] = self.compute_price_acceleration(minute_df)
        features['tick_imbalance'] = self.compute_tick_imbalance(minute_df)
        features['bar_entropy'] = self.compute_bar_entropy(minute_df)
        features['volume_price_correlation'] = self.compute_volume_price_correlation(minute_df)
        features['consecutive_direction'] = self.compute_consecutive_direction_count(minute_df)
        
        # Cross-sectional features (4) - computed as scalars per day
        # These would be added as constant columns for intraday data
        if sector_data:
            sector_rank = self.compute_sector_momentum_rank(symbol, minute_df, sector_data)
            features['sector_momentum_rank'] = sector_rank
        else:
            features['sector_momentum_rank'] = 0.5
        
        if nifty_df is not None:
            rs_vs_nifty = self.compute_relative_strength_vs_nifty(minute_df, nifty_df)
            corr_to_nifty = self.compute_correlation_to_nifty(minute_df, nifty_df)
            features['relative_strength_vs_nifty'] = rs_vs_nifty
            features['correlation_to_nifty_20d'] = corr_to_nifty
        else:
            features['relative_strength_vs_nifty'] = 0.0
            features['correlation_to_nifty_20d'] = 0.5
        
        # Placeholder for sector flow (would need sector data)
        features['sector_flow_score'] = 1.0
        
        # Volatility regime features (4)
        if vix_data is not None:
            vix_pctile = self.compute_vix_percentile_60d(vix_data, str(minute_df.index[-1]))
            features['vix_percentile_60d'] = vix_pctile
            
            realized_vs_implied = self.compute_realized_vs_implied_vol(
                minute_df, vix_data.iloc[-1]
            )
            features['realized_vs_implied_vol'] = realized_vs_implied
        else:
            features['vix_percentile_60d'] = 0.5
            features['realized_vs_implied_vol'] = 1.0
        
        features['overnight_gap_zscore'] = self.compute_overnight_gap_zscore(minute_df)
        features['intraday_range_percentile'] = self.compute_intraday_range_percentile(minute_df)
        
        # Options features (4) - placeholders if options_data not provided
        if options_data:
            features['pcr_change'] = self.compute_pcr_change(
                options_data.get('pcr_series')
            )
            features['max_pain_distance'] = self.compute_max_pain_distance(
                minute_df['close'].iloc[-1],
                options_data.get('max_pain_strike', 0)
            )
            features['iv_skew'] = self.compute_iv_skew(
                options_data.get('otm_put_iv', 0),
                options_data.get('otm_call_iv', 0)
            )
            features['oi_buildup_signal'] = self.compute_oi_buildup_signal(
                options_data.get('current_oi', 0),
                options_data.get('prev_oi', 1),
                options_data.get('option_type', 'CE')
            )
        else:
            features['pcr_change'] = 0.0
            features['max_pain_distance'] = 0.0
            features['iv_skew'] = 0.0
            features['oi_buildup_signal'] = 0.0
        
        logger.info(f"Computed {len(features.columns)} new features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all 18 new feature names."""
        return [
            # Microstructure (6)
            'relative_volume_15m',
            'price_acceleration',
            'tick_imbalance',
            'bar_entropy',
            'volume_price_correlation',
            'consecutive_direction',
            # Cross-sectional (4)
            'sector_momentum_rank',
            'sector_flow_score',
            'relative_strength_vs_nifty',
            'correlation_to_nifty_20d',
            # Volatility regime (4)
            'vix_percentile_60d',
            'realized_vs_implied_vol',
            'overnight_gap_zscore',
            'intraday_range_percentile',
            # Options (4)
            'pcr_change',
            'max_pain_distance',
            'iv_skew',
            'oi_buildup_signal',
        ]


def main():
    """Demonstrate feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="v3.0 Feature Engineering")
    parser.add_argument("--symbol", type=str, default="RELIANCE")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    
    args = parser.parse_args()
    
    # Load sample data
    csv_path = Path(args.data_dir) / f"{args.symbol}_minute.csv"
    
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        print("Creating synthetic demo data...")
        
        # Create synthetic data
        n_bars = 1000
        dates = pd.date_range(end='2025-01-15', periods=n_bars, freq='1min')
        
        np.random.seed(42)
        trend = np.cumsum(np.random.randn(n_bars) * 0.001)
        
        df = pd.DataFrame({
            'open': 1000 * (1 + trend) + np.random.randn(n_bars) * 2,
            'high': 1000 * (1 + trend) + np.abs(np.random.randn(n_bars)) * 3 + 2,
            'low': 1000 * (1 + trend) - np.abs(np.random.randn(n_bars)) * 3 - 2,
            'close': 1000 * (1 + trend) + np.random.randn(n_bars) * 2,
            'volume': np.random.poisson(10000, n_bars),
        }, index=dates)
        
        # Ensure high >= low
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1) + 1)
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1) - 1)
    else:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.set_index('date')
    
    # Compute features
    engineer = EnhancedFeatureEngineer()
    features = engineer.compute_all_features(
        minute_df=df,
        symbol=args.symbol,
        sector="ENERGY",
    )
    
    print("\n" + "="*70)
    print(f"v3.0 Enhanced Features for {args.symbol}")
    print("="*70)
    
    print("\nMicrostructure Features:")
    for col in ['relative_volume_15m', 'price_acceleration', 'tick_imbalance', 
                'bar_entropy', 'volume_price_correlation', 'consecutive_direction']:
        val = features[col].iloc[-1]
        print(f"  {col:30s}: {val:8.4f}")
    
    print("\nCross-Sectional Features:")
    for col in ['sector_momentum_rank', 'sector_flow_score', 
                'relative_strength_vs_nifty', 'correlation_to_nifty_20d']:
        val = features[col].iloc[-1] if hasattr(features[col], 'iloc') else features[col]
        print(f"  {col:30s}: {val:8.4f}")
    
    print("\nVolatility Regime Features:")
    for col in ['vix_percentile_60d', 'realized_vs_implied_vol', 
                'overnight_gap_zscore', 'intraday_range_percentile']:
        val = features[col].iloc[-1]
        print(f"  {col:30s}: {val:8.4f}")
    
    print("\nOptions Features:")
    for col in ['pcr_change', 'max_pain_distance', 'iv_skew', 'oi_buildup_signal']:
        val = features[col].iloc[-1]
        print(f"  {col:30s}: {val:8.4f}")
    
    print("\n" + "="*70)
    print(f"Total new features: {len(features.columns)}")
    print("="*70)


if __name__ == "__main__":
    main()
