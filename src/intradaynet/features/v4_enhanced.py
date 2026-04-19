"""
Enhanced Feature Engineering v4.0 - MICROSTRUCTURE ENHANCED
Adds high-impact microstructure features to improve AUC.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger("intradaynet.features.v4_enhanced")


@dataclass
class EnhancedFeatureConfig:
    """Configuration for enhanced feature computation."""
    # Microstructure windows
    volume_imbalance_window: int = 20
    price_momentum_window: int = 10
    volatility_regime_window: int = 30
    order_flow_window: int = 15
    

class EnhancedFeatureEngineerV4:
    """
    V4.0 Feature Engineer with MICROSTRUCTURE enhancements.
    
    NEW FEATURES ADDED:
    1. Volume Delta (buy vs sell volume estimate)
    2. Price Momentum Slope (linear regression slope)
    3. VWAP Deviation (distance from VWAP)
    4. Tick Imbalance v2 (signed volume on upticks/downticks)
    5. Volatility Regime (realized vol vs historical)
    6. Order Flow Imbalance (bid/ask pressure)
    7. Range Expansion (current vs recent range)
    8. Gap Fill Probability (based on historical gaps)
    9. Time of Day Effect (session-based patterns)
    10. Volume Profile (at price vs average)
    
    Total: 18 (v3) + 10 (new) = 28 features
    """
    
    def __init__(self, config: Optional[EnhancedFeatureConfig] = None):
        self.config = config or EnhancedFeatureConfig()
        self.v3_engineer = None  # Will import if needed
    
    def compute_volume_delta(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 1: Volume Delta (buy vs sell volume estimate).
        
        Uses tick rule: volume classified as buy if close > close[-1], else sell.
        High positive delta = buying pressure.
        """
        volume = minute_df['volume']
        close = minute_df['close']
        
        # Tick rule classification
        price_change = close.diff()
        buy_volume = volume * (price_change > 0).astype(float)
        sell_volume = volume * (price_change < 0).astype(float)
        
        # Delta = buy - sell
        delta = buy_volume - sell_volume
        
        # Normalize by total volume
        total_volume = buy_volume + sell_volume + 1  # +1 to avoid div by zero
        delta_ratio = delta / total_volume
        
        # Rolling sum
        window = self.config.volume_imbalance_window
        delta_rolling = delta_ratio.rolling(window, min_periods=5).sum()
        
        return delta_rolling.fillna(0).clip(-10, 10)
    
    def compute_vwap_deviation(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 2: VWAP Deviation.
        
        Distance of current price from VWAP.
        Positive = above VWAP (bullish), Negative = below VWAP (bearish).
        """
        typical_price = (minute_df['high'] + minute_df['low'] + minute_df['close']) / 3
        volume = minute_df['volume']
        
        # VWAP = cumulative(TP * Volume) / cumulative(Volume)
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Deviation as percentage
        deviation = (minute_df['close'] - vwap) / vwap * 100
        
        return deviation.fillna(0).clip(-5, 5)
    
    def compute_price_momentum_slope(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 3: Price Momentum Slope.
        
        Linear regression slope over recent window.
        Positive slope = upward momentum, Negative = downward.
        """
        close = minute_df['close']
        window = self.config.price_momentum_window
        
        def linear_slope(x):
            if len(x) < 5:
                return 0.0
            x_norm = np.arange(len(x))
            # Simple linear regression
            x_mean, y_mean = x_norm.mean(), x.mean()
            numerator = ((x_norm - x_mean) * (x - y_mean)).sum()
            denominator = ((x_norm - x_mean) ** 2).sum()
            if denominator == 0:
                return 0.0
            return numerator / denominator
        
        slope = close.rolling(window, min_periods=5).apply(linear_slope, raw=True)
        
        # Normalize by price level
        slope_normalized = slope / close * 1000
        
        return slope_normalized.fillna(0).clip(-50, 50)
    
    def compute_tick_imbalance_v2(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 4: Tick Imbalance v2 (Signed Volume).
        
        Volume-weighted tick imbalance.
        Sum(volume on upticks) - Sum(volume on downticks).
        """
        close = minute_df['close']
        volume = minute_df['volume']
        
        price_change = close.diff()
        
        # Signed volume
        signed_volume = volume * np.sign(price_change)
        
        # Rolling sum
        window = self.config.order_flow_window
        imbalance = signed_volume.rolling(window, min_periods=5).sum()
        
        # Normalize by rolling volume
        total_volume = volume.rolling(window, min_periods=5).sum()
        imbalance_ratio = imbalance / (total_volume + 1)
        
        return imbalance_ratio.fillna(0).clip(-1, 1)
    
    def compute_volatility_regime(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 5: Volatility Regime.
        
        Current realized volatility vs historical average.
        > 1 = high vol regime (tighten stops), < 1 = low vol (widen stops).
        """
        returns = minute_df['close'].pct_change().abs()
        
        window = self.config.volatility_regime_window
        
        # Current vol (recent)
        current_vol = returns.rolling(window, min_periods=10).mean()
        
        # Historical vol (longer window)
        hist_vol = returns.rolling(window * 2, min_periods=20).mean()
        
        # Ratio
        vol_ratio = current_vol / (hist_vol + 1e-10)
        
        return vol_ratio.fillna(1.0).clip(0.1, 10)
    
    def compute_range_expansion(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 6: Range Expansion.
        
        Is current range larger than recent average?
        Large expansion = breakout potential, compression = consolidation.
        """
        high = minute_df['high']
        low = minute_df['low']
        close = minute_df['close']
        
        # Current range as % of price
        current_range = (high - low) / close
        
        # Historical average range
        window = 20
        avg_range = current_range.rolling(window, min_periods=5).mean()
        
        # Expansion ratio
        expansion = current_range / (avg_range + 1e-10)
        
        return expansion.fillna(1.0).clip(0.1, 5)
    
    def compute_session_intensity(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 7: Session Intensity (Time of Day Effect).
        
        Volume intensity based on time of day.
        9:15-10:00 = high intensity, 13:00-14:00 = low intensity, 14:00-15:30 = high.
        """
        # Get hour from index
        hours = minute_df.index.hour + minute_df.index.minute / 60
        
        # Define session patterns (typical Indian market)
        # Opening hour (9:15-10:15): intensity = 1.5
        # Mid-day (10:15-13:30): intensity = 0.8
        # Afternoon (13:30-14:30): intensity = 1.0
        # Closing (14:30-15:30): intensity = 1.3
        
        intensity = pd.Series(1.0, index=minute_df.index)
        
        # Opening intensity (9:15 - 10:15)
        opening_mask = (hours >= 9.25) & (hours <= 10.25)
        intensity[opening_mask] = 1.5
        
        # Low intensity mid-day (11:30 - 13:30)
        midday_mask = (hours >= 11.5) & (hours <= 13.5)
        intensity[midday_mask] = 0.7
        
        # Closing intensity (14:30 - 15:30)
        closing_mask = (hours >= 14.5) & (hours <= 15.5)
        intensity[closing_mask] = 1.3
        
        return intensity
    
    def compute_volume_profile(self, minute_df: pd.DataFrame) -> pd.Series:
        """
        Feature 8: Volume Profile at Price.
        
        Is current volume above or below average for this time of day?
        """
        volume = minute_df['volume']
        
        # Get time (hour:minute) as string for grouping
        time_str = minute_df.index.strftime('%H:%M')
        
        # Calculate expanding average for each time slot
        # Simplified: use rolling average
        window = 20
        avg_volume = volume.rolling(window, min_periods=5).mean()
        
        # Profile = current / average
        profile = volume / (avg_volume + 1)
        
        return profile.fillna(1.0).clip(0.1, 10)
    
    def compute_all_features_v4(self, minute_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute ALL features (v3 + new microstructure).
        Returns DataFrame with 28 features.
        """
        # Import v3 features
        from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed
        
        v3_engineer = EnhancedFeatureEngineerFixed()
        features_v3 = v3_engineer.compute_all_features(minute_df, symbol)
        
        # Add new v4 features
        features_v4 = pd.DataFrame(index=minute_df.index)
        
        # Microstructure features (8 new)
        features_v4['volume_delta'] = self.compute_volume_delta(minute_df)
        features_v4['vwap_deviation'] = self.compute_vwap_deviation(minute_df)
        features_v4['price_momentum_slope'] = self.compute_price_momentum_slope(minute_df)
        features_v4['tick_imbalance_v2'] = self.compute_tick_imbalance_v2(minute_df)
        features_v4['volatility_regime'] = self.compute_volatility_regime(minute_df)
        features_v4['range_expansion'] = self.compute_range_expansion(minute_df)
        features_v4['session_intensity'] = self.compute_session_intensity(minute_df)
        features_v4['volume_profile'] = self.compute_volume_profile(minute_df)
        
        # Combine v3 + v4
        combined = pd.concat([features_v3, features_v4], axis=1)
        
        logger.info(f"Computed {len(combined.columns)} features for {symbol} (v3: 18 + v4: 8)")
        
        return combined
    
    def get_feature_names_v4(self) -> List[str]:
        """Return list of all 26 feature names (18 v3 + 8 v4)."""
        from intradaynet.features.v3_features_fixed import EnhancedFeatureEngineerFixed
        
        v3_names = EnhancedFeatureEngineerFixed().get_feature_names()
        v4_names = [
            'volume_delta', 'vwap_deviation', 'price_momentum_slope',
            'tick_imbalance_v2', 'volatility_regime', 'range_expansion',
            'session_intensity', 'volume_profile'
        ]
        
        return v3_names + v4_names


def main():
    """Demo v4 features."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--data-dir", default="nifty500")
    
    args = parser.parse_args()
    
    csv_path = Path(args.data_dir) / f"{args.symbol}_minute.csv"
    
    if not csv_path.exists():
        print(f"Data not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.set_index('date')
    df.columns = df.columns.str.lower()
    
    # Sample for demo
    df = df.iloc[:1000]
    
    engineer = EnhancedFeatureEngineerV4()
    features = engineer.compute_all_features_v4(df, args.symbol)
    
    print("\n" + "="*70)
    print(f"ENHANCED V4 FEATURES for {args.symbol}")
    print("="*70)
    
    print("\nMicrostructure Features (NEW):")
    for col in ['volume_delta', 'vwap_deviation', 'price_momentum_slope', 
                'tick_imbalance_v2', 'volatility_regime', 'range_expansion',
                'session_intensity', 'volume_profile']:
        val = features[col].iloc[-1]
        print(f"  {col:30s}: {val:8.4f}")
    
    print(f"\nTotal features: {len(features.columns)}")
    print(f"  v3 features: 18")
    print(f"  v4 features: 8 (NEW)")
    print("="*70)


if __name__ == "__main__":
    main()
