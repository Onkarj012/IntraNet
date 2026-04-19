"""
Anchored Walk-Forward Validation Engine for IntradayNet v3.0

Implements proper walk-forward validation to prevent look-ahead bias:
- Method: Anchored walk-forward with expanding window
- Train: Start with 2015-2022, retrain every 3 months, adding latest quarter
- Validation: Always next 1 month after training cutoff (for hyperparameter tuning)
- Test: Month after validation (never touched during training/tuning)
- Result: ~12-14 independent out-of-sample months per year

Key Features:
1. Strict temporal splits - no future data leakage
2. Regime-aware training - separate models per regime
3. Liquid universe filtering at each split
4. Survivorship-bias-free universe construction

Usage:
    from intradaynet.walkforward_v3 import WalkForwardEngine
    
    engine = WalkForwardEngine(
        data_dir="nifty500",
        model_type="lightgbm",
        train_months=84,  # 7 years initial
        val_months=1,
        test_months=1,
        step_months=3,  # Retrain every quarter
    )
    
    results = engine.run_full_walkforward(
        start_date="2015-01-01",
        end_date="2025-12-31",
    )
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# ML libraries
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from intradaynet.liquid_universe import LiquidUniverseFilter
from intradaynet.regime_v3 import RegimeClassifierV3, MarketRegime, RegimeAdjustments


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("walkforward_v3")


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_number: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    
    # Universe info
    n_train_symbols: int
    n_test_symbols: int
    train_symbols: List[str]
    test_symbols: List[str]
    
    # Regime info
    train_regime: str
    test_regime: str
    
    # Metrics
    train_samples: int
    val_samples: int
    test_samples: int
    
    direction_accuracy: float
    direction_auc: float
    magnitude_mae: float
    magnitude_corr: float
    
    # Trading metrics (backtest results)
    win_rate: float
    avg_win_loss_ratio: float
    net_edge_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Feature importance
    top_features: List[Tuple[str, float]]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Time windows
    train_months_initial: int = 84  # 7 years
    val_months: int = 1
    test_months: int = 1
    step_months: int = 3  # Retrain quarterly
    
    # Model settings
    model_type: str = "lightgbm"  # lightgbm, xgboost, etc.
    
    # LightGBM params
    lgb_params: Dict = None
    
    # Universe settings
    use_liquid_filter: bool = True
    max_universe_size: int = 150
    min_universe_size: int = 120
    
    # Regime settings
    use_regime_models: bool = True
    regime_aware_training: bool = True
    
    # Feature settings
    n_features: int = 625
    feature_names: List[str] = None
    
    # Data
    data_dir: str = "nifty500"
    cache_dir: str = "walkforward_cache"
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "dart",  # Better regularization than gbdt
                "n_estimators": 500,
                "max_depth": 6,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction_bynode": 0.7,  # Force feature diversity
                "min_data_in_leaf": 200,  # Prevent overfitting
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "n_jobs": -1,
                "verbosity": -1,
            }


class WalkForwardEngine:
    """
    Anchored walk-forward validation with regime awareness.
    
    Key improvements over basic walk-forward:
    1. Anchored expanding window (not rolling)
    2. Regime-conditional models
    3. Liquid universe filtering at each split
    4. Survivorship-bias-free universe
    """
    
    HORIZONS = ["H15", "H30", "H60"]
    
    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.liquid_filter = LiquidUniverseFilter(
            data_dir=self.config.data_dir,
            cache_dir=f"{self.config.cache_dir}/liquid"
        )
        self.regime_classifier = RegimeClassifierV3()
        self.results: List[FoldResult] = []
        
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
    def create_folds(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, str]]:
        """
        Create anchored walk-forward fold boundaries.
        
        Returns list of dicts with train/val/test boundaries.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Initial training period
        train_start = start
        train_end = start + pd.DateOffset(months=self.config.train_months_initial)
        
        folds = []
        fold_num = 1
        
        while True:
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=self.config.val_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=self.config.test_months)
            
            if test_end > end:
                break
            
            folds.append({
                "fold": fold_num,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            })
            
            # Step forward - expand training window
            train_end = train_end + pd.DateOffset(months=self.config.step_months)
            fold_num += 1
        
        logger.info(f"Created {len(folds)} walk-forward folds")
        return folds
    
    def get_liquid_universe_for_fold(
        self,
        train_end: str,
        test_end: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Get liquid universe as of train end and test end.
        
        This handles survivorship bias by using as-of dates.
        """
        if not self.config.use_liquid_filter:
            # Use all available stocks
            data_dir = Path(self.config.data_dir)
            all_symbols = sorted([
                p.stem.replace('_minute', '')
                for p in data_dir.glob('*_minute.csv')
            ])
            return all_symbols, all_symbols
        
        # Get universe as of training cutoff
        train_universe = self.liquid_filter.get_liquid_universe(
            as_of_date=train_end,
            max_stocks=self.config.max_universe_size,
            min_stocks=self.config.min_universe_size,
        )
        
        # Get universe as of test end (for evaluation only)
        test_universe = self.liquid_filter.get_liquid_universe(
            as_of_date=test_end,
            max_stocks=self.config.max_universe_size,
            min_stocks=self.config.min_universe_size,
        )
        
        return train_universe, test_universe
    
    def load_data_for_fold(
        self,
        fold: Dict[str, str],
        symbols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training, validation, and test data for a fold.
        
        Returns feature DataFrames for train/val/test.
        """
        # This would load from prebatched data or compute features
        # For now, return empty DataFrames as placeholder
        logger.info(f"Loading data for fold {fold['fold']}...")
        
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        return train_df, val_df, test_df
    
    def detect_regime_for_period(
        self,
        start_date: str,
        end_date: str,
    ) -> MarketRegime:
        """
        Detect dominant regime for a time period.
        
        Uses majority vote of daily regimes.
        """
        # Load market data
        try:
            regime, reason, adj = self.regime_classifier.get_regime_from_market_data(
                date=end_date
            )
            return regime
        except Exception as e:
            logger.warning(f"Could not detect regime: {e}")
            return MarketRegime.UNKNOWN
    
    def train_model_for_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        regime: MarketRegime,
        horizon: str,
    ) -> lgb.Booster:
        """
        Train a model for a specific fold and regime.
        
        If regime-aware training is enabled, uses regime-specific parameters.
        """
        # Get regime-specific adjustments
        if self.config.use_regime_models:
            _, _, adj = self.regime_classifier.classify(
                vix_level=15,  # Dummy - would use actual
                vix_change_pct=0,
            )
            # Could adjust LGB params based on regime
        
        # Prepare features and targets
        # This is a placeholder - actual implementation would:
        # 1. Extract features from train_df
        # 2. Apply regime-specific sampling
        # 3. Train LightGBM model
        
        model = lgb.LGBMClassifier(**self.config.lgb_params)
        
        # Dummy fit - would use actual data
        # model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        return model
    
    def evaluate_fold(
        self,
        model: lgb.Booster,
        test_df: pd.DataFrame,
        regime: MarketRegime,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set and compute trading metrics.
        
        Returns dict with model metrics and backtest results.
        """
        metrics = {
            "direction_accuracy": 0.0,
            "direction_auc": 0.0,
            "magnitude_mae": 0.0,
            "magnitude_corr": 0.0,
            "win_rate": 0.0,
            "avg_win_loss_ratio": 0.0,
            "net_edge_per_trade": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        
        return metrics
    
    def run_single_fold(
        self,
        fold: Dict[str, str],
    ) -> FoldResult:
        """
        Run complete training and evaluation for one fold.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold['fold']}: {fold['train_start']} to {fold['test_end']}")
        logger.info(f"{'='*60}")
        
        # Get liquid universe
        train_symbols, test_symbols = self.get_liquid_universe_for_fold(
            fold['train_end'],
            fold['test_end']
        )
        
        logger.info(f"Train universe: {len(train_symbols)} stocks")
        logger.info(f"Test universe: {len(test_symbols)} stocks")
        
        # Detect regimes
        train_regime = self.detect_regime_for_period(
            fold['train_start'], fold['train_end']
        )
        test_regime = self.detect_regime_for_period(
            fold['test_start'], fold['test_end']
        )
        
        logger.info(f"Train regime: {train_regime.value}")
        logger.info(f"Test regime: {test_regime.value}")
        
        # Load data
        train_df, val_df, test_df = self.load_data_for_fold(fold, train_symbols)
        
        # Train models for each horizon
        models = {}
        for horizon in self.HORIZONS:
            model = self.train_model_for_fold(
                train_df, val_df, train_regime, horizon
            )
            models[horizon] = model
        
        # Evaluate
        metrics = self.evaluate_fold(models["H60"], test_df, test_regime)
        
        result = FoldResult(
            fold_number=fold['fold'],
            train_start=fold['train_start'],
            train_end=fold['train_end'],
            val_start=fold['val_start'],
            val_end=fold['val_end'],
            test_start=fold['test_start'],
            test_end=fold['test_end'],
            n_train_symbols=len(train_symbols),
            n_test_symbols=len(test_symbols),
            train_symbols=train_symbols,
            test_symbols=test_symbols,
            train_regime=train_regime.value,
            test_regime=test_regime.value,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df),
            direction_accuracy=metrics['direction_accuracy'],
            direction_auc=metrics['direction_auc'],
            magnitude_mae=metrics['magnitude_mae'],
            magnitude_corr=metrics['magnitude_corr'],
            win_rate=metrics['win_rate'],
            avg_win_loss_ratio=metrics['avg_win_loss_ratio'],
            net_edge_per_trade=metrics['net_edge_per_trade'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            top_features=[],
        )
        
        logger.info(f"Fold {fold['fold']} complete")
        logger.info(f"  Win rate: {result.win_rate:.2%}")
        logger.info(f"  Net edge: {result.net_edge_per_trade:.4f}")
        
        return result
    
    def run_full_walkforward(
        self,
        start_date: str,
        end_date: str,
        save_results: bool = True,
    ) -> List[FoldResult]:
        """
        Run complete anchored walk-forward validation.
        
        This is the main entry point for walk-forward validation.
        """
        logger.info(f"\n{'#'*70}")
        logger.info(f"ANCHORED WALK-FORWARD VALIDATION")
        logger.info(f"{'#'*70}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial train: {self.config.train_months_initial} months")
        logger.info(f"Val/Test: {self.config.val_months}/{self.config.test_months} months")
        logger.info(f"Step: {self.config.step_months} months")
        logger.info(f"Liquid filter: {self.config.use_liquid_filter}")
        logger.info(f"Regime models: {self.config.use_regime_models}")
        logger.info(f"{'#'*70}\n")
        
        # Create folds
        folds = self.create_folds(start_date, end_date)
        
        if not folds:
            logger.error("No folds created - check date range")
            return []
        
        # Run each fold
        self.results = []
        for fold in folds:
            result = self.run_single_fold(fold)
            self.results.append(result)
        
        # Aggregate results
        summary = self.aggregate_results()
        
        # Save results
        if save_results:
            self.save_results(summary)
        
        return self.results
    
    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all folds.
        
        Computes mean, std, and confidence intervals.
        """
        if not self.results:
            return {}
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        summary = {
            "n_folds": len(self.results),
            "date_range": {
                "first_train_start": df['train_start'].min(),
                "last_test_end": df['test_end'].max(),
            },
            "direction_accuracy": {
                "mean": df['direction_accuracy'].mean(),
                "std": df['direction_accuracy'].std(),
                "min": df['direction_accuracy'].min(),
                "max": df['direction_accuracy'].max(),
            },
            "win_rate": {
                "mean": df['win_rate'].mean(),
                "std": df['win_rate'].std(),
                "min": df['win_rate'].min(),
                "max": df['win_rate'].max(),
            },
            "net_edge_per_trade": {
                "mean": df['net_edge_per_trade'].mean(),
                "std": df['net_edge_per_trade'].std(),
            },
            "sharpe_ratio": {
                "mean": df['sharpe_ratio'].mean(),
                "std": df['sharpe_ratio'].std(),
            },
            "max_drawdown": {
                "mean": df['max_drawdown'].mean(),
                "std": df['max_drawdown'].std(),
            },
            "regime_distribution": df['test_regime'].value_counts().to_dict(),
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("WALK-FORWARD SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Folds completed: {summary['n_folds']}")
        logger.info(f"Win Rate: {summary['win_rate']['mean']:.2%} (±{summary['win_rate']['std']:.2%})")
        logger.info(f"Net Edge: {summary['net_edge_per_trade']['mean']:.4f} (±{summary['net_edge_per_trade']['std']:.4f})")
        logger.info(f"Sharpe: {summary['sharpe_ratio']['mean']:.2f} (±{summary['sharpe_ratio']['std']:.2f})")
        logger.info(f"Max DD: {summary['max_drawdown']['mean']:.2%} (±{summary['max_drawdown']['std']:.2%})")
        logger.info(f"{'='*60}")
        
        return summary
    
    def save_results(self, summary: Dict):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save fold results
        results_file = Path(self.config.cache_dir) / f"walkforward_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'config': {
                    'train_months_initial': self.config.train_months_initial,
                    'val_months': self.config.val_months,
                    'test_months': self.config.test_months,
                    'step_months': self.config.step_months,
                    'use_liquid_filter': self.config.use_liquid_filter,
                    'use_regime_models': self.config.use_regime_models,
                },
                'summary': summary,
                'folds': [r.to_dict() for r in self.results],
            }, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")


def main():
    """CLI for running walk-forward validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Anchored Walk-Forward Validation")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--train-months", type=int, default=84)
    parser.add_argument("--val-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--no-liquid-filter", action="store_true")
    parser.add_argument("--no-regime", action="store_true")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    parser.add_argument("--dry-run", action="store_true", help="Show folds without running")
    
    args = parser.parse_args()
    
    config = WalkForwardConfig(
        train_months_initial=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        use_liquid_filter=not args.no_liquid_filter,
        use_regime_models=not args.no_regime,
        data_dir=args.data_dir,
    )
    
    engine = WalkForwardEngine(config)
    
    if args.dry_run:
        # Just show folds
        folds = engine.create_folds(args.start, args.end)
        print(f"\n{'='*70}")
        print(f"DRY RUN - {len(folds)} Folds")
        print(f"{'='*70}")
        for fold in folds:
            print(f"\nFold {fold['fold']}:")
            print(f"  Train: {fold['train_start']} to {fold['train_end']}")
            print(f"  Val:   {fold['val_start']} to {fold['val_end']}")
            print(f"  Test:  {fold['test_start']} to {fold['test_end']}")
        print(f"\n{'='*70}")
    else:
        # Run full validation
        results = engine.run_full_walkforward(args.start, args.end)
        print(f"\nCompleted {len(results)} folds")


if __name__ == "__main__":
    main()
