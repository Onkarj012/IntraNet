"""
Feature Selection for IntradayNet v3.0 - Phase 2.2

Implements disciplined feature selection using:
1. Permutation importance on validation sets
2. Remove features with < 0.1% importance of top feature
3. Target final count: 60-75 features (from 87 total)

Also audits low-value features like candlestick anatomy (body_ratio, 
upper_shadow_ratio, lower_shadow_ratio) which rarely survive on 1-min bars.

Usage:
    from intradaynet.feature_selection import FeatureSelector
    
    selector = FeatureSelector(
        feature_names=all_feature_names,
        target_count=70,
        min_importance_pct=0.1,
    )
    
    # Fit on validation data
    selector.fit(X_val, y_val, model)
    
    # Get selected features
    selected_features = selector.selected_features
    
    # Transform data
    X_selected = selector.transform(X)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("intradaynet.feature_selection")


@dataclass
class FeatureImportance:
    """Importance score for a single feature."""
    name: str
    importance: float
    rank: int
    survives: bool
    category: str  # 'microstructure', 'cross_sectional', 'volatility', 'options', 'original'


class FeatureSelector:
    """
    Select features using permutation importance.
    
    Process:
    1. Compute baseline score on validation set
    2. For each feature, shuffle values and compute score drop
    3. Importance = score_baseline - score_shuffled
    4. Remove features below threshold (< 0.1% of top)
    5. Target 60-75 features from 87 total
    """
    
    # Categories for the new v3.0 features
    V3_CATEGORIES = {
        # Microstructure
        'relative_volume_15m': 'microstructure',
        'price_acceleration': 'microstructure',
        'tick_imbalance': 'microstructure',
        'bar_entropy': 'microstructure',
        'volume_price_correlation': 'microstructure',
        'consecutive_direction': 'microstructure',
        # Cross-sectional
        'sector_momentum_rank': 'cross_sectional',
        'sector_flow_score': 'cross_sectional',
        'relative_strength_vs_nifty': 'cross_sectional',
        'correlation_to_nifty_20d': 'cross_sectional',
        # Volatility
        'vix_percentile_60d': 'volatility',
        'realized_vs_implied_vol': 'volatility',
        'overnight_gap_zscore': 'volatility',
        'intraday_range_percentile': 'volatility',
        # Options
        'pcr_change': 'options',
        'max_pain_distance': 'options',
        'iv_skew': 'options',
        'oi_buildup_signal': 'options',
    }
    
    # Low-value features to audit (rarely survive on 1-min bars)
    LOW_VALUE_CANDIDATES = [
        'body_ratio',
        'upper_shadow_ratio', 
        'lower_shadow_ratio',
        'time_normalized',  # Might be low value in 1-min context
    ]
    
    def __init__(
        self,
        feature_names: List[str],
        target_count: int = 70,
        min_importance_pct: float = 0.1,  # 0.1% of top feature
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.target_count = target_count
        self.min_importance_pct = min_importance_pct
        self.random_state = random_state
        
        self.importances: Dict[str, FeatureImportance] = {}
        self.selected_features: List[str] = []
        self.selected_indices: List[int] = []
        self.baseline_score: float = 0.0
        
    def _compute_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        n_repeats: int = 5,
    ) -> Dict[str, float]:
        """
        Compute permutation importance for each feature.
        
        Higher score drop = more important feature.
        """
        np.random.seed(self.random_state)
        
        # Baseline score
        baseline_score = self._score_model(model, X, y)
        self.baseline_score = baseline_score
        
        logger.info(f"Baseline score: {baseline_score:.4f}")
        
        importances = {}
        
        for i, feature_name in enumerate(self.feature_names):
            scores_shuffled = []
            
            for _ in range(n_repeats):
                # Shuffle feature values
                X_shuffled = X.copy()
                np.random.shuffle(X_shuffled[:, i])
                
                # Compute score with shuffled feature
                score_shuffled = self._score_model(model, X_shuffled, y)
                scores_shuffled.append(score_shuffled)
            
            # Importance = baseline - mean(shuffled)
            # Higher drop = more important
            importance = baseline_score - np.mean(scores_shuffled)
            importances[feature_name] = importance
            
            if i % 10 == 0:
                logger.info(f"  Feature {i+1}/{len(self.feature_names)}: {feature_name} = {importance:.6f}")
        
        return importances
    
    def _score_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute model score (AUC for classification, -MAE for regression).
        """
        from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score
        
        # Try to predict
        try:
            if hasattr(model, 'predict_proba'):
                # Classification - use AUC
                y_proba = model.predict_proba(X)[:, 1]
                score = roc_auc_score(y, y_proba)
            elif hasattr(model, 'predict'):
                # Try regression or classification
                y_pred = model.predict(X)
                
                # Check if binary classification
                if len(np.unique(y)) == 2:
                    score = accuracy_score(y, y_pred)
                else:
                    # Regression - use negative MAE (higher is better)
                    score = -mean_absolute_error(y, y_pred)
            else:
                score = 0.0
        except Exception as e:
            logger.warning(f"Could not score model: {e}")
            score = 0.0
        
        return score
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        n_repeats: int = 5,
    ) -> 'FeatureSelector':
        """
        Fit feature selector on validation data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector
            model: Trained model with predict or predict_proba
            n_repeats: Number of shuffle repeats per feature
        """
        logger.info(f"Computing permutation importance for {len(self.feature_names)} features...")
        
        # Compute importances
        raw_importances = self._compute_permutation_importance(X, y, model, n_repeats)
        
        # Sort by importance
        sorted_features = sorted(
            raw_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Determine threshold
        max_importance = sorted_features[0][1] if sorted_features else 1.0
        threshold = max_importance * (self.min_importance_pct / 100)
        
        logger.info(f"Max importance: {max_importance:.6f}")
        logger.info(f"Selection threshold: {threshold:.6f} ({self.min_importance_pct}% of max)")
        
        # Create importance records
        self.importances = {}
        for rank, (name, importance) in enumerate(sorted_features, 1):
            category = self.V3_CATEGORIES.get(name, 'original')
            survives = importance >= threshold and rank <= self.target_count
            
            self.importances[name] = FeatureImportance(
                name=name,
                importance=importance,
                rank=rank,
                survives=survives,
                category=category,
            )
        
        # Select features
        self.selected_features = [
            name for name, fi in self.importances.items()
            if fi.survives
        ]
        
        # Get indices
        self.selected_indices = [
            self.feature_names.index(name) for name in self.selected_features
        ]
        
        # Log results
        logger.info(f"\nFeature Selection Results:")
        logger.info(f"  Original features: {len(self.feature_names)}")
        logger.info(f"  Selected features: {len(self.selected_features)}")
        logger.info(f"  Removed features: {len(self.feature_names) - len(self.selected_features)}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to selected features only."""
        return X[:, self.selected_indices]
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, model)
        return self.transform(X)
    
    def get_importance_report(self) -> pd.DataFrame:
        """Get detailed importance report as DataFrame."""
        if not self.importances:
            return pd.DataFrame()
        
        records = [
            {
                'feature': fi.name,
                'importance': fi.importance,
                'rank': fi.rank,
                'survives': fi.survives,
                'category': fi.category,
            }
            for fi in self.importances.values()
        ]
        
        df = pd.DataFrame(records)
        df = df.sort_values('importance', ascending=False)
        
        return df
    
    def print_report(self, top_n: int = 30):
        """Print feature importance report."""
        df = self.get_importance_report()
        
        if df.empty:
            print("No importance data available. Call fit() first.")
            return
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE REPORT")
        print("="*80)
        
        print(f"\nBaseline score: {self.baseline_score:.4f}")
        print(f"Features selected: {len(self.selected_features)}/{len(self.feature_names)}")
        
        # Top features
        print(f"\nTop {top_n} Features:")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<30} {'Importance':>12} {'Category':<20} {'Status'}")
        print("-"*80)
        
        for i, row in df.head(top_n).iterrows():
            status = "✓ KEEP" if row['survives'] else "✗ DROP"
            print(f"{row['rank']:<6} {row['feature']:<30} {row['importance']:>12.6f} "
                  f"{row['category']:<20} {status}")
        
        # Summary by category
        print("\n\nSummary by Category:")
        print("-"*80)
        category_summary = df.groupby('category').agg({
            'feature': 'count',
            'survives': 'sum',
        }).rename(columns={'feature': 'total', 'survives': 'selected'})
        category_summary['dropped'] = category_summary['total'] - category_summary['selected']
        category_summary['survival_rate'] = (
            category_summary['selected'] / category_summary['total'] * 100
        )
        
        for cat, row in category_summary.iterrows():
            print(f"  {cat:<20}: {row['selected']:.0f}/{row['total']:.0f} "
                  f"({row['survival_rate']:.1f}% survival)")
        
        # Low-value feature audit
        print("\n\nLow-Value Feature Audit:")
        print("-"*80)
        for feature in self.LOW_VALUE_CANDIDATES:
            if feature in self.importances:
                fi = self.importances[feature]
                status = "SURVIVED" if fi.survives else "DROPPED (as expected)"
                print(f"  {feature:<30}: Rank {fi.rank:3d}, {status}")
        
        print("\n" + "="*80)
    
    def save_selected_features(self, filepath: str):
        """Save list of selected features to file."""
        with open(filepath, 'w') as f:
            for name in self.selected_features:
                f.write(f"{name}\n")
        logger.info(f"Saved {len(self.selected_features)} selected features to {filepath}")
    
    def load_selected_features(self, filepath: str):
        """Load previously selected features."""
        with open(filepath) as f:
            self.selected_features = [line.strip() for line in f if line.strip()]
        
        self.selected_indices = [
            self.feature_names.index(name) for name in self.selected_features
            if name in self.feature_names
        ]
        
        logger.info(f"Loaded {len(self.selected_features)} selected features from {filepath}")


def select_features_for_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    target_count: int = 70,
    output_file: Optional[str] = None,
) -> List[str]:
    """
    Convenience function for feature selection.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets
        feature_names: List of feature names
        target_count: Target number of features to keep
        output_file: Optional file to save selected features
        
    Returns:
        List of selected feature names
    """
    selector = FeatureSelector(
        feature_names=feature_names,
        target_count=target_count,
    )
    
    selector.fit(X_val, y_val, model)
    selector.print_report()
    
    if output_file:
        selector.save_selected_features(output_file)
    
    return selector.selected_features


def main():
    """Demo feature selection with synthetic data."""
    import argparse
    from sklearn.ensemble import RandomForestClassifier
    
    parser = argparse.ArgumentParser(description="Feature Selection Demo")
    parser.add_argument("--n-features", type=int, default=87)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--target-count", type=int, default=70)
    
    args = parser.parse_args()
    
    print("\nGenerating synthetic data...")
    
    # Create feature names
    base_features = [
        'log_return', 'volume_ratio', 'vwap_distance', 'ema_9_distance', 
        'ema_20_distance', 'rsi_14', 'bb_zscore', 'bb_width', 'body_ratio',
        'upper_shadow_ratio', 'lower_shadow_ratio', 'spread_pct', 'volume_pace',
        'time_normalized', 'orb_high_dist', 'orb_low_dist', 'day_return',
        'momentum_5', 'momentum_20', 'vol_momentum', 'atr_14',
        'close_vs_running_range', 'session_volatility', 'obv_slope', 'trade_intensity',
    ]
    
    # Add flattened window features
    windows = [5, 15, 30, 60, 120]
    stats = ['mean', 'std', 'min', 'max']
    
    feature_names = []
    for w in windows:
        for stat in stats:
            for pb in base_features:
                feature_names.append(f"{pb}_w{w}_{stat}")
    
    # Add last/first/diff features
    for pb in base_features:
        feature_names.append(f"{pb}_last")
        feature_names.append(f"{pb}_first")
        feature_names.append(f"{pb}_diff")
    
    # Add v3.0 features
    v3_features = [
        'relative_volume_15m', 'price_acceleration', 'tick_imbalance',
        'bar_entropy', 'volume_price_correlation', 'consecutive_direction',
        'sector_momentum_rank', 'sector_flow_score', 'relative_strength_vs_nifty',
        'correlation_to_nifty_20d', 'vix_percentile_60d', 'realized_vs_implied_vol',
        'overnight_gap_zscore', 'intraday_range_percentile',
        'pcr_change', 'max_pain_distance', 'iv_skew', 'oi_buildup_signal',
    ]
    feature_names.extend(v3_features)
    
    # Take first n features
    feature_names = feature_names[:args.n_features]
    
    # Generate synthetic data with some important and some noise features
    np.random.seed(42)
    n_important = 20
    n_noise = args.n_features - n_important
    
    # Important features
    X_important = np.random.randn(args.n_samples, n_important)
    
    # Noise features
    X_noise = np.random.randn(args.n_samples, n_noise) * 0.1
    
    X = np.hstack([X_important, X_noise])
    
    # Target depends only on important features
    y = (X_important.sum(axis=1) > 0).astype(int)
    
    print(f"Training model on {args.n_samples} samples...")
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Select features
    print("\nRunning feature selection...")
    selected = select_features_for_model(
        model=model,
        X_val=X,
        y_val=y,
        feature_names=feature_names,
        target_count=args.target_count,
    )
    
    print(f"\nFinal feature count: {len(selected)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
