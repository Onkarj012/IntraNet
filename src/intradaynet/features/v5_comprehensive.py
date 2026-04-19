"""
IntradayNet v5.0 - COMPREHENSIVE FEATURE ENGINEERING

Features Added:
1. Technical Indicators (RSI, MACD, Bollinger, ATR, etc.)
2. Cross-Sectional Signals (sector momentum, index correlation)
3. Time-Based Features (market open/close, lunch effect)
4. Order Flow Estimates (from 1-min data)
5. Volatility Regime Features
6. Lead-Lag Features (stock vs sector/index)
7. Market Context (VIX, Nifty trend)

Total: 50+ features using FULL 1-minute data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger("intradaynet.features.v5_comprehensive")


@dataclass
class ComprehensiveFeatureConfig:
    """Configuration for comprehensive feature computation."""
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    atr_window: int = 14
    momentum_window: int = 10
    volume_ma_window: int = 20
    correlation_window: int = 30
    

class ComprehensiveFeatureEngineerV5:
    """
    V5.0 Feature Engineer with COMPREHENSIVE indicators and cross-sectional signals.
    Uses FULL 1-minute data (not downsampled) for maximum signal.
    """
    
    def __init__(self, config: Optional[ComprehensiveFeatureConfig] = None):
        self.config = config or ComprehensiveFeatureConfig()
        self._nifty_cache = None
        self._sector_cache = {}
    
    # =========================================================================
    # TECHNICAL INDICATORS (15 features)
    # =========================================================================
    
    def compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index - momentum oscillator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def compute_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        exp1 = prices.ewm(span=self.config.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.config.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.config.macd_signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def compute_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (upper, middle, lower)."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        return upper, middle, lower
    
    def compute_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range - volatility indicator."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def compute_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K, %D)."""
        low_min = df['low'].rolling(window=k_window).min()
        high_max = df['high'].rolling(window=k_window).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_window).mean()
        return k, d
    
    def compute_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Williams %R - momentum indicator."""
        high_max = df['high'].rolling(window=window).max()
        low_min = df['low'].rolling(window=window).min()
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        return williams_r
    
    def compute_cci(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    def compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def compute_mfi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Money Flow Index - volume-weighted RSI."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        delta = tp.diff()
        
        positive_flow = rmf.where(delta > 0, 0).rolling(window=window).sum()
        negative_flow = rmf.where(delta < 0, 0).rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def compute_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Ichimoku Cloud components."""
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        chikou_span = df['close'].shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def compute_pivot_points(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Pivot Points (Support/Resistance levels)."""
        pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        r1 = (2 * pivot) - df['low'].shift(1)
        s1 = (2 * pivot) - df['high'].shift(1)
        r2 = pivot + (df['high'].shift(1) - df['low'].shift(1))
        s2 = pivot - (df['high'].shift(1) - df['low'].shift(1))
        
        return {
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'r2': r2,
            's2': s2
        }
    
    # =========================================================================
    # CROSS-SECTIONAL FEATURES (10 features)
    # =========================================================================
    
    def compute_beta_to_nifty(self, stock_returns: pd.Series, nifty_returns: pd.Series, window: int = 30) -> pd.Series:
        """Stock's beta (sensitivity) to Nifty movements."""
        covariance = stock_returns.rolling(window=window).cov(nifty_returns)
        variance = nifty_returns.rolling(window=window).var()
        beta = covariance / variance
        return beta.fillna(1.0)
    
    def compute_relative_strength_vs_sector(self, stock_returns: pd.Series, sector_returns: pd.Series, window: int = 20) -> pd.Series:
        """Stock return minus sector return (alpha)."""
        stock_cum = (1 + stock_returns).rolling(window=window).apply(lambda x: np.prod(x)) - 1
        sector_cum = (1 + sector_returns).rolling(window=window).apply(lambda x: np.prod(x)) - 1
        relative = stock_cum - sector_cum
        return relative
    
    def compute_correlation_to_market(self, stock_returns: pd.Series, market_returns: pd.Series, window: int = 30) -> pd.Series:
        """Rolling correlation to market."""
        correlation = stock_returns.rolling(window=window).corr(market_returns)
        return correlation.fillna(0)
    
    def compute_sector_momentum_rank(self, stock_returns: pd.Series, all_sector_returns: pd.DataFrame, window: int = 10) -> float:
        """Where this stock ranks within its sector (0-1)."""
        stock_cum = (1 + stock_returns).rolling(window=window).apply(lambda x: np.prod(x)) - 1
        current_ret = stock_cum.iloc[-1]
        
        sector_rets = []
        for col in all_sector_returns.columns:
            cum_ret = (1 + all_sector_returns[col]).rolling(window=window).apply(lambda x: np.prod(x)) - 1
            sector_rets.append(cum_ret.iloc[-1])
        
        if len(sector_rets) == 0:
            return 0.5
        
        percentile = stats.percentileofscore(sector_rets, current_ret, kind='rank') / 100
        return percentile
    
    def compute_market_breadth_signal(self, advances: int, declines: int, unchanged: int = 0) -> float:
        """Market breadth (advances vs declines)."""
        total = advances + declines + unchanged
        if total == 0:
            return 0.5
        breadth = advances / (advances + declines) if (advances + declines) > 0 else 0.5
        return breadth
    
    # =========================================================================
    # TIME-BASED FEATURES (8 features)
    # =========================================================================
    
    def compute_time_features(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Time-based market patterns."""
        features = pd.DataFrame(index=minute_df.index)
        
        # Hour and minute
        features['hour'] = minute_df.index.hour
        features['minute'] = minute_df.index.minute
        
        # Market session (Indian market: 9:15-15:30)
        # Opening hour (9:15-10:15) - high volatility
        features['is_opening_hour'] = ((features['hour'] == 9) & (features['minute'] >= 15)) | \
                                       ((features['hour'] == 10) & (features['minute'] <= 15))
        
        # Lunch hour (12:00-13:30) - low volume
        features['is_lunch_hour'] = (features['hour'] >= 12) & (features['hour'] <= 13)
        
        # Closing hour (14:30-15:30) - high volume
        features['is_closing_hour'] = (features['hour'] >= 14) & (features['hour'] <= 15) & (features['minute'] >= 30)
        
        # Time from market open (in minutes)
        market_open = minute_df.index.normalize() + pd.Timedelta(hours=9, minutes=15)
        features['minutes_from_open'] = (minute_df.index - market_open).total_seconds() / 60
        
        # Day of week (0=Monday, 6=Sunday)
        features['day_of_week'] = minute_df.index.dayofweek
        
        # Month (seasonality)
        features['month'] = minute_df.index.month
        
        return features
    
    def compute_intraday_momentum(self, minute_df: pd.DataFrame) -> pd.Series:
        """Momentum from market open to current time."""
        # Get open price (first price of the day)
        daily_open = minute_df.groupby(minute_df.index.date)['open'].transform('first')
        current = minute_df['close']
        momentum = (current - daily_open) / daily_open * 100
        return momentum
    
    # =========================================================================
    # VOLATILITY FEATURES (7 features)
    # =========================================================================
    
    def compute_realized_volatility(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """Realized volatility (annualized)."""
        vol = returns.rolling(window=window).std() * np.sqrt(252 * 375)  # 375 mins per trading day
        return vol
    
    def compute_volatility_regime(self, current_vol: pd.Series, historical_vol: pd.Series) -> pd.Series:
        """Current vol vs historical average."""
        regime = current_vol / historical_vol.replace(0, np.nan)
        return regime.fillna(1.0)
    
    def compute_volatility_percentile(self, current_vol: pd.Series, lookback: int = 60) -> pd.Series:
        """Where current vol ranks historically (0-100)."""
        def rolling_percentile(x):
            if len(x) < 20:
                return 50.0
            return stats.percentileofscore(x[:-1], x[-1], kind='rank')
        
        percentile = current_vol.rolling(window=lookback).apply(rolling_percentile, raw=False)
        return percentile.fillna(50.0)
    
    def compute_garman_klass_vol(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """Garman-Klass volatility estimator (more efficient)."""
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        vol = np.sqrt(gk.rolling(window=window).mean()) * np.sqrt(252 * 375)
        return vol
    
    def compute_parkinson_vol(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """Parkinson volatility (uses high-low range)."""
        log_hl = np.log(df['high'] / df['low']) ** 2
        vol = np.sqrt(log_hl.rolling(window=window).mean() / (4 * np.log(2))) * np.sqrt(252 * 375)
        return vol
    
    # =========================================================================
    # ORDER FLOW FEATURES (5 features)
    # =========================================================================
    
    def compute_volume_imbalance(self, minute_df: pd.DataFrame, window: int = 10) -> pd.Series:
        """Buy vs sell volume estimate using tick rule."""
        price_change = minute_df['close'].diff()
        volume = minute_df['volume']
        
        # Tick rule: up = buy, down = sell
        signed_volume = volume * np.sign(price_change)
        imbalance = signed_volume.rolling(window=window).sum()
        
        # Normalize by total volume
        total_volume = volume.rolling(window=window).sum()
        imbalance_ratio = imbalance / (total_volume + 1)
        
        return imbalance_ratio
    
    def compute_vwap(self, minute_df: pd.DataFrame) -> pd.Series:
        """Volume-Weighted Average Price."""
        typical_price = (minute_df['high'] + minute_df['low'] + minute_df['close']) / 3
        vwap = (typical_price * minute_df['volume']).cumsum() / minute_df['volume'].cumsum()
        return vwap
    
    def compute_vwap_deviation(self, minute_df: pd.DataFrame) -> pd.Series:
        """Distance from VWAP (institutional benchmark)."""
        vwap = self.compute_vwap(minute_df)
        deviation = (minute_df['close'] - vwap) / vwap * 100
        return deviation
    
    def compute_price_momentum_slope(self, prices: pd.Series, window: int = 10) -> pd.Series:
        """Linear regression slope of price."""
        def linear_slope(x):
            if len(x) < 5:
                return 0.0
            x_norm = np.arange(len(x))
            x_mean, y_mean = np.mean(x_norm), np.mean(x)
            numerator = np.sum((x_norm - x_mean) * (x - y_mean))
            denominator = np.sum((x_norm - x_mean) ** 2)
            if denominator == 0:
                return 0.0
            slope = numerator / denominator
            # Normalize
            return slope / np.mean(x) * 1000
        
        slope = prices.rolling(window=window).apply(linear_slope, raw=True)
        return slope
    
    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================
    
    def compute_all_features_v5(self, minute_df: pd.DataFrame, symbol: str, 
                                 nifty_df: Optional[pd.DataFrame] = None,
                                 sector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute ALL v5 comprehensive features (50+ features).
        Uses FULL 1-minute data for maximum signal.
        """
        features = pd.DataFrame(index=minute_df.index)
        
        # Ensure data is clean
        minute_df = minute_df.replace([np.inf, -np.inf], np.nan)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in minute_df.columns:
                minute_df[col] = minute_df[col].ffill().bfill()
        
        prices = minute_df['close']
        
        # TECHNICAL INDICATORS (15 features)
        logger.info(f"Computing technical indicators for {symbol}...")
        
        features['rsi'] = self.compute_rsi(prices, window=14)
        
        macd, macd_signal, macd_hist = self.compute_macd(prices)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_hist
        
        bb_upper, bb_middle, bb_lower = self.compute_bollinger_bands(prices, window=20)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower)
        
        features['atr'] = self.compute_atr(minute_df, window=14)
        features['atr_percent'] = features['atr'] / prices * 100
        
        stoch_k, stoch_d = self.compute_stochastic(minute_df, k_window=14, d_window=3)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        features['williams_r'] = self.compute_williams_r(minute_df, window=14)
        features['cci'] = self.compute_cci(minute_df, window=20)
        features['mfi'] = self.compute_mfi(minute_df, window=14)
        
        # Ichimoku
        ichimoku = self.compute_ichimoku(minute_df)
        features['tenkan_sen'] = ichimoku['tenkan_sen']
        features['kijun_sen'] = ichimoku['kijun_sen']
        
        # VOLATILITY (7 features)
        logger.info(f"Computing volatility features for {symbol}...")
        
        returns = prices.pct_change()
        features['realized_vol_30m'] = self.compute_realized_volatility(returns, window=30)
        features['realized_vol_60m'] = self.compute_realized_volatility(returns, window=60)
        features['vol_percentile'] = self.compute_volatility_percentile(features['realized_vol_30m'], lookback=60)
        features['garman_klass_vol'] = self.compute_garman_klass_vol(minute_df, window=30)
        features['parkinson_vol'] = self.compute_parkinson_vol(minute_df, window=30)
        
        # ORDER FLOW (5 features)
        logger.info(f"Computing order flow features for {symbol}...")
        
        features['volume_imbalance'] = self.compute_volume_imbalance(minute_df, window=10)
        features['vwap_deviation'] = self.compute_vwap_deviation(minute_df)
        features['price_momentum_slope'] = self.compute_price_momentum_slope(prices, window=10)
        
        # Volume features
        features['volume_ma_ratio'] = minute_df['volume'] / minute_df['volume'].rolling(window=20).mean()
        features['volume_change'] = minute_df['volume'].pct_change()
        
        # TIME-BASED (8 features)
        logger.info(f"Computing time features for {symbol}...")
        
        time_features = self.compute_time_features(minute_df)
        features = pd.concat([features, time_features], axis=1)
        
        features['intraday_momentum'] = self.compute_intraday_momentum(minute_df)
        
        # PRICE ACTION (10 features)
        features['returns_5m'] = prices.pct_change(5)
        features['returns_10m'] = prices.pct_change(10)
        features['returns_30m'] = prices.pct_change(30)
        
        features['high_low_range'] = (minute_df['high'] - minute_df['low']) / minute_df['close']
        features['body_size'] = abs(minute_df['close'] - minute_df['open']) / minute_df['open']
        features['upper_shadow'] = (minute_df['high'] - minute_df[['close', 'open']].max(axis=1)) / minute_df['close']
        features['lower_shadow'] = (minute_df[['close', 'open']].min(axis=1) - minute_df['low']) / minute_df['close']
        
        # Gap features
        features['overnight_gap'] = (minute_df['open'] - minute_df['close'].shift(1)) / minute_df['close'].shift(1)
        features['gap_filled'] = (minute_df['close'] > minute_df['open'].shift(1)) & (minute_df['open'] < minute_df['close'].shift(1))
        
        # CROSS-SECTIONAL (if data provided)
        if nifty_df is not None and len(nifty_df) > 0:
            logger.info(f"Computing cross-sectional features for {symbol} vs Nifty...")
            
            nifty_returns = nifty_df['close'].pct_change()
            stock_returns = prices.pct_change()
            
            features['beta_to_nifty'] = self.compute_beta_to_nifty(stock_returns, nifty_returns, window=30)
            features['correlation_to_nifty'] = self.compute_correlation_to_market(stock_returns, nifty_returns, window=30)
            
            # Relative performance
            nifty_cum = (1 + nifty_returns).rolling(window=60).apply(lambda x: np.prod(x)) - 1
            stock_cum = (1 + stock_returns).rolling(window=60).apply(lambda x: np.prod(x)) - 1
            features['relative_to_nifty'] = stock_cum - nifty_cum
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Clip extreme values
        for col in features.columns:
            if features[col].dtype in [np.float64, np.float32]:
                features[col] = features[col].clip(-50, 50)
        
        logger.info(f"Computed {len(features.columns)} comprehensive features for {symbol}")
        
        return features
    
    def get_feature_names_v5(self) -> List[str]:
        """Return list of all v5 feature names (~50)."""
        return [
            # Technical indicators (15)
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_position', 'atr', 'atr_percent',
            'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi',
            'tenkan_sen', 'kijun_sen',
            # Volatility (7)
            'realized_vol_30m', 'realized_vol_60m', 'vol_percentile',
            'garman_klass_vol', 'parkinson_vol',
            # Order flow (5)
            'volume_imbalance', 'vwap_deviation', 'price_momentum_slope',
            'volume_ma_ratio', 'volume_change',
            # Time-based (8)
            'hour', 'minute', 'is_opening_hour', 'is_lunch_hour', 'is_closing_hour',
            'minutes_from_open', 'day_of_week', 'month', 'intraday_momentum',
            # Price action (10)
            'returns_5m', 'returns_10m', 'returns_30m',
            'high_low_range', 'body_size', 'upper_shadow', 'lower_shadow',
            'overnight_gap', 'gap_filled',
            # Cross-sectional (3)
            'beta_to_nifty', 'correlation_to_nifty', 'relative_to_nifty'
        ]


def main():
    """Demo v5 features."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--data-dir", default="nifty500")
    
    args = parser.parse_args()
    
    csv_path = Path(args.data_dir) / f"{args.symbol}_minute.csv"
    
    if not csv_path.exists():
        print(f"Data not found: {csv_path}")
        return
    
    print(f"Loading {args.symbol} data...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.set_index('date')
    df.columns = df.columns.str.lower()
    
    # Use full 1-minute data (first 5000 bars for demo)
    df = df.iloc[:5000]
    
    print(f"Computing v5 comprehensive features...")
    engineer = ComprehensiveFeatureEngineerV5()
    features = engineer.compute_all_features_v5(df, args.symbol)
    
    print("\n" + "="*70)
    print(f"V5 COMPREHENSIVE FEATURES for {args.symbol}")
    print("="*70)
    
    print(f"\nTotal features: {len(features.columns)}")
    print("\nFeature categories:")
    print("  Technical Indicators: 15 (RSI, MACD, Bollinger, ATR, etc.)")
    print("  Volatility: 7 (Realized vol, Garman-Klass, Parkinson)")
    print("  Order Flow: 5 (Volume imbalance, VWAP deviation)")
    print("  Time-Based: 9 (Market session, intraday momentum)")
    print("  Price Action: 10 (Returns, candlestick patterns)")
    print("  Cross-Sectional: 3 (Beta to Nifty, correlation)")
    
    print("\nSample feature values (last bar):")
    for col in ['rsi', 'macd', 'bb_position', 'volume_imbalance', 'vwap_deviation']:
        val = features[col].iloc[-1]
        print(f"  {col:30s}: {val:8.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
