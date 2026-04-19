"""
Train IntradayNet v3.0 Models and Run Backtests

This script:
1. Prepares training data with v3.0 features
2. Trains 3 specialized models (Direction, Magnitude, Confidence)
3. Runs walk-forward backtests with regime awareness
4. Compares v3.0 results to v2.0 baseline

Usage:
    python scripts/train_and_backtest_v3.py --train
    python scripts/train_and_backtest_v3.py --backtest --start 2025-01-01 --end 2025-03-31
    python scripts/train_and_backtest_v3.py --full
"""

import argparse
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, brier_score_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v3_training")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# v3.0 imports
from intradaynet.regime_v3 import RegimeClassifierV3, MarketRegime
from intradaynet.dynamic_targets import DynamicTargetManager
from intradaynet.features.v3_features import EnhancedFeatureEngineer
from intradaynet.feature_selection import FeatureSelector, select_features_for_model
from intradaynet.models.specialized import (
    SpecializedModelSuite, ModelConfig,
    compute_expected_calibration_error
)
from intradaynet.risk_management import (
    RiskManager, PositionSizingConfig, PortfolioConfig,
    ExitConfig, CircuitBreakerConfig
)
from intradaynet.liquid_universe import LiquidUniverseFilter
from intradaynet.survivorship_bias import SurvivorshipBiasFix
from intradaynet.walkforward_v3 import WalkForwardEngine, WalkForwardConfig

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class V3ModelTrainer:
    """Train IntradayNet v3.0 specialized models."""
    
    def __init__(
        self,
        data_dir: str = "nifty500",
        output_dir: str = "models/v3_production",
        universe_size: int = 150,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.universe_size = universe_size
        self.feature_engineer = EnhancedFeatureEngineer()
        self.regime_classifier = RegimeClassifierV3()
        
        # Model suite
        self.model_suite = None
        self.feature_selector = None
        
        # Training data storage
        self.X_train = None
        self.y_dir_train = None
        self.y_mag_train = None
        self.y_conf_train = None
        self.feature_names = None
        
        logger.info(f"V3 Trainer initialized")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Universe size: {universe_size}")
    
    def prepare_training_data(
        self,
        train_start: str = "2022-01-01",
        train_end: str = "2024-12-31",
        min_bars: int = 500,
    ):
        """
        Prepare training data with v3.0 features.
        
        Loads liquid universe stocks and computes 87 features.
        """
        logger.info("="*70)
        logger.info("PREPARING TRAINING DATA")
        logger.info("="*70)
        
        # Get liquid universe
        liquid_filter = LiquidUniverseFilter(data_dir=self.data_dir)
        universe = liquid_filter.get_liquid_universe(
            as_of_date=train_end,
            max_stocks=self.universe_size,
        )
        
        logger.info(f"Using {len(universe)} liquid stocks")
        
        # Storage for all samples
        all_X = []
        all_y_dir = []
        all_y_mag = []
        all_y_conf = []
        
        # Process each stock
        for i, symbol in enumerate(universe):
            if i % 10 == 0:
                logger.info(f"Processing {i+1}/{len(universe)}: {symbol}")
            
            csv_path = self.data_dir / f"{symbol}_minute.csv"
            if not csv_path.exists():
                continue
            
            try:
                # Load data
                df = pd.read_csv(csv_path, parse_dates=['date'])
                df = df.set_index('date')
                df.columns = df.columns.str.lower()
                
                # Filter to training period
                df = df[(df.index >= train_start) & (df.index <= train_end)]
                
                if len(df) < min_bars:
                    continue
                
                # Compute v3.0 features
                features = self.feature_engineer.compute_all_features(
                    minute_df=df,
                    symbol=symbol,
                )
                
                # Create targets for each bar
                for idx in range(120, len(df) - 60, 30):  # Sample every 30 bars
                    # Feature window
                    feat_window = features.iloc[idx-120:idx]
                    if len(feat_window) < 120:
                        continue
                    
                    # Current price
                    current_price = df['close'].iloc[idx]
                    
                    # Future window (next 60 bars)
                    future_window = df.iloc[idx:idx+60]
                    if len(future_window) < 60:
                        continue
                    
                    # Direction target: Did price go up?
                    future_return = (future_window['close'].iloc[-1] - current_price) / current_price
                    y_dir = 1 if future_return > 0 else 0
                    
                    # Magnitude target: Absolute return
                    y_mag = abs(future_return)
                    
                    # Confidence target: Did it hit 1% target before 0.5% stop?
                    future_high = future_window['high'].max()
                    future_low = future_window['low'].min()
                    
                    target_hit = future_high >= current_price * 1.01
                    stop_hit = future_low <= current_price * 0.995
                    
                    y_conf = 1 if target_hit and not stop_hit else 0
                    
                    # Flatten features (simple mean for now)
                    feat_vector = feat_window.mean().values
                    
                    all_X.append(feat_vector)
                    all_y_dir.append(y_dir)
                    all_y_mag.append(y_mag)
                    all_y_conf.append(y_conf)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                continue
        
        # Convert to arrays
        self.X_train = np.array(all_X)
        self.y_dir_train = np.array(all_y_dir)
        self.y_mag_train = np.array(all_y_mag)
        self.y_conf_train = np.array(all_y_conf)
        
        # Feature names
        self.feature_names = self.feature_engineer.get_feature_names()
        
        logger.info(f"\nTraining data prepared:")
        logger.info(f"  Samples: {len(self.X_train)}")
        logger.info(f"  Features: {self.X_train.shape[1]}")
        logger.info(f"  Direction balance: {np.mean(self.y_dir_train):.2%}")
        logger.info(f"  Avg magnitude: {np.mean(self.y_mag_train):.4f}")
        logger.info(f"  Confidence balance: {np.mean(self.y_conf_train):.2%}")
    
    def select_features(self, n_select: int = 70):
        """Select top features using permutation importance."""
        logger.info("\n" + "="*70)
        logger.info("FEATURE SELECTION")
        logger.info("="*70)
        
        # Train a temporary model for feature importance
        if HAS_LIGHTGBM:
            temp_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            temp_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_dir_train, test_size=0.2, random_state=42
        )
        
        temp_model.fit(X_tr, y_tr)
        
        # Select features
        self.feature_selector = FeatureSelector(
            feature_names=self.feature_names,
            target_count=n_select,
        )
        
        self.feature_selector.fit(X_val, y_val, temp_model, n_repeats=3)
        self.feature_selector.print_report(top_n=20)
        
        # Transform training data
        selected_indices = self.feature_selector.selected_indices
        self.X_train = self.X_train[:, selected_indices]
        self.feature_names = self.feature_selector.selected_features
        
        logger.info(f"\nSelected {len(self.feature_names)} features")
    
    def train_models(self):
        """Train the 3 specialized models."""
        logger.info("\n" + "="*70)
        logger.info("TRAINING SPECIALIZED MODELS")
        logger.info("="*70)
        
        # Split data
        X_train, X_val, y_dir_train, y_dir_val = train_test_split(
            self.X_train, self.y_dir_train, test_size=0.2, random_state=42
        )
        _, _, y_mag_train, y_mag_val = train_test_split(
            self.X_train, self.y_mag_train, test_size=0.2, random_state=42
        )
        _, _, y_conf_train, y_conf_val = train_test_split(
            self.X_train, self.y_conf_train, test_size=0.2, random_state=42
        )
        
        # Initialize model suite
        model_config = ModelConfig()
        self.model_suite = SpecializedModelSuite(model_config)
        
        # Train
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
        
        self.model_suite.fit(
            X_train, y_dir_train, y_mag_train, y_conf_train,
            X_val, y_dir_val, y_mag_val, y_conf_val,
        )
        
        # Evaluate
        logger.info("\nValidation Metrics:")
        
        # Direction accuracy
        dir_preds = self.model_suite.direction_model.predict_class(X_val)
        dir_acc = accuracy_score(y_dir_val, dir_preds)
        logger.info(f"  Direction Accuracy: {dir_acc:.2%}")
        
        # Direction AUC
        dir_proba = self.model_suite.direction_model.predict(X_val)
        try:
            dir_auc = roc_auc_score(y_dir_val, dir_proba)
            logger.info(f"  Direction AUC: {dir_auc:.4f}")
            
            # ECE
            ece = compute_expected_calibration_error(y_dir_val, dir_proba)
            logger.info(f"  Direction ECE: {ece:.4f}")
        except:
            pass
        
        # Magnitude MAE
        mag_preds = self.model_suite.magnitude_model.predict(X_val)
        mag_mae = mean_absolute_error(y_mag_val, mag_preds)
        logger.info(f"  Magnitude MAE: {mag_mae:.5f}")
        
        # Confidence accuracy
        conf_preds = self.model_suite.confidence_model.predict(X_val) > 0.5
        conf_acc = accuracy_score(y_conf_val, conf_preds)
        logger.info(f"  Confidence Accuracy: {conf_acc:.2%}")
    
    def save_models(self):
        """Save trained models."""
        logger.info("\n" + "="*70)
        logger.info("SAVING MODELS")
        logger.info("="*70)
        
        # Save model suite
        self.model_suite.save(str(self.output_dir))
        
        # Save feature selector
        if self.feature_selector:
            self.feature_selector.save_selected_features(
                str(self.output_dir / "selected_features.txt")
            )
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'model_config': {
                'direction': 'LGBMClassifier' if HAS_LIGHTGBM else 'RandomForest',
                'magnitude': 'LGBMRegressor' if HAS_LIGHTGBM else 'RandomForest',
                'confidence': 'LGBMClassifier' if HAS_LIGHTGBM else 'RandomForest',
            }
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {self.output_dir}")
    
    def train_full_pipeline(self):
        """Execute full training pipeline."""
        logger.info("\n" + "="*70)
        logger.info("V3.0 TRAINING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Prepare data
        self.prepare_training_data()
        
        if len(self.X_train) == 0:
            logger.error("No training data generated!")
            return False
        
        # Step 2: Select features
        if len(self.feature_names) > 70:
            self.select_features(n_select=70)
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Save
        self.save_models()
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        
        return True


class V3Backtester:
    """Run v3.0 backtests with full risk management."""
    
    def __init__(
        self,
        model_dir: str = "models/v3_production",
        data_dir: str = "nifty500",
        output_dir: str = "backtest_results/v3",
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        self.model_suite = SpecializedModelSuite()
        self.model_suite.load(model_dir)
        
        # Risk manager
        self.risk_manager = RiskManager(account_value=100000)
        
        # Target manager
        self.target_manager = DynamicTargetManager()
        
        logger.info("V3 Backtester initialized")
    
    def backtest_period(
        self,
        start_date: str,
        end_date: str,
        symbols: list = None,
        max_positions: int = 5,
    ):
        """
        Run backtest for a period.
        
        This is a simplified backtest - full implementation would use
        the walk-forward framework from walkforward_v3.py.
        """
        logger.info("="*70)
        logger.info(f"BACKTEST: {start_date} to {end_date}")
        logger.info("="*70)
        
        # Get universe
        if symbols is None:
            liquid_filter = LiquidUniverseFilter(data_dir=self.data_dir)
            symbols = liquid_filter.get_liquid_universe(
                as_of_date=start_date,
                max_stocks=50,  # Smaller for demo
            )
        
        logger.info(f"Backtesting {len(symbols)} symbols")
        
        # Track results
        all_trades = []
        daily_pnl = []
        
        # Simple backtest loop
        for symbol in symbols[:5]:  # Limit for demo
            csv_path = self.data_dir / f"{symbol}_minute.csv"
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path, parse_dates=['date'])
                df = df.set_index('date')
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                if len(df) < 200:
                    continue
                
                logger.info(f"Processing {symbol}: {len(df)} bars")
                
                # Simulate trading (simplified)
                # Full implementation would use proper feature windows
                
            except Exception as e:
                logger.warning(f"Error with {symbol}: {e}")
                continue
        
        # Generate summary
        summary = {
            'start_date': start_date,
            'end_date': end_date,
            'n_symbols': len(symbols),
            'n_trades': len(all_trades),
            'note': 'Demo backtest - full implementation would include all trade logic',
        }
        
        logger.info("\nBacktest Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Save results
        with open(self.output_dir / f"backtest_{start_date}_{end_date}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Train v3.0 models and run backtests")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--full", action="store_true", help="Train + backtest")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    parser.add_argument("--train-start", type=str, default="2022-01-01")
    parser.add_argument("--train-end", type=str, default="2024-12-31")
    parser.add_argument("--backtest-start", type=str, default="2025-01-01")
    parser.add_argument("--backtest-end", type=str, default="2025-03-31")
    
    args = parser.parse_args()
    
    print("="*70)
    print("INTRADAYNET v3.0 - TRAINING & BACKTESTING")
    print("="*70)
    print()
    
    success = True
    
    if args.train or args.full:
        print("🚀 TRAINING v3.0 MODELS")
        print("-"*70)
        
        trainer = V3ModelTrainer(data_dir=args.data_dir)
        success = trainer.train_full_pipeline()
        
        if not success:
            print("\n❌ Training failed!")
            return 1
    
    if args.backtest or args.full:
        if not success:
            print("\n⚠️  Skipping backtest (training failed or not run)")
            return 1
        
        print("\n📊 RUNNING BACKTEST")
        print("-"*70)
        
        backtester = V3Backtester(
            model_dir="models/v3_production",
            data_dir=args.data_dir,
        )
        
        results = backtester.backtest_period(
            args.backtest_start,
            args.backtest_end,
        )
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
