"""
V8 IntradayNet — Complete redesign.

Design principles:
1. Learn from raw intraday curves (masked autoencoder embeddings)
2. Predict path, not point (barrier targets)
3. Ensemble of specialized models for different market regimes
4. Calibration over raw accuracy
5. Portfolio construction as a first-class problem

Architecture:
    Data Layer → Feature Store → Signal Layer (ensemble) → Recommendation Layer
"""

from .barriers import (
    BarrierTarget,
    barrier_label_distribution,
    barrier_targets_to_dataframe,
    compute_barrier_targets,
    compute_barrier_targets_batch,
    compute_multi_horizon_barriers,
)
from .config import (
    V8Config,
    TargetConfig,
    EmbeddingConfig,
    SignalModelConfig,
    MetaEnsembleConfig,
    PortfolioConfig,
    BacktestConfig,
    SentimentConfig,
    DailyFeaturesConfig,
)
from .curve_embedding import (
    CurveMaskedEncoder,
    CurveDataset,
    CurveTrainer,
    generate_embeddings,
    load_curve_encoder,
    masked_mse_loss,
    prepare_curve_data,
    save_embeddings,
)
from .data_pipeline import (
    build_training_dataset,
    downsample_curve_ohlc,
    extract_sessions,
    load_minute_data,
    load_minute_data_batch,
)
from .daily_features import (
    DailyFeatureBuilder,
)
from .per_stock_sentiment import (
    compute_per_stock_sentiment_features,
    build_historical_sentiment_cache,
    fetch_live_sentiment,
)
from .portfolio import (
    Portfolio,
    PortfolioCandidate,
    PortfolioConstructor,
    compute_position_sizes,
)
from .regime_detector import (
    RegimeDetector,
    RegimeAssignment,
    DEFAULT_REGIME_WEIGHTS,
    REGIME_CONFIDENCE_SCALE,
)
from .signal_models import (
    SignalModel,
    MetaEnsemble,
    FEATURE_GROUPS,
)
from .universe_tiers import (
    DataTier,
    TierAssignment,
    UniverseTierReport,
    classify_tiers,
)
from .walk_forward import (
    WalkForwardBacktest,
    BacktestMetrics,
    TradeRecord,
    compute_buy_hold_metrics,
    compute_random_baseline,
)

__all__ = [
    # Barriers
    "BarrierTarget",
    "barrier_label_distribution",
    "barrier_targets_to_dataframe",
    "compute_barrier_targets",
    "compute_barrier_targets_batch",
    "compute_multi_horizon_barriers",
    # Config
    "V8Config",
    "TargetConfig",
    "EmbeddingConfig",
    "SignalModelConfig",
    "MetaEnsembleConfig",
    "PortfolioConfig",
    "BacktestConfig",
    "SentimentConfig",
    "DailyFeaturesConfig",
    # Curve Embedding
    "CurveMaskedEncoder",
    "CurveDataset",
    "CurveTrainer",
    "generate_embeddings",
    "load_curve_encoder",
    "masked_mse_loss",
    "prepare_curve_data",
    "save_embeddings",
    # Data Pipeline
    "build_training_dataset",
    "downsample_curve_ohlc",
    "extract_sessions",
    "load_minute_data",
    "load_minute_data_batch",
    # Daily Features
    "DailyFeatureBuilder",
    # Sentiment
    "compute_per_stock_sentiment_features",
    "build_historical_sentiment_cache",
    "fetch_live_sentiment",
    # Portfolio
    "Portfolio",
    "PortfolioCandidate",
    "PortfolioConstructor",
    "compute_position_sizes",
    # Regime
    "RegimeDetector",
    "RegimeAssignment",
    "DEFAULT_REGIME_WEIGHTS",
    "REGIME_CONFIDENCE_SCALE",
    # Signal Models
    "SignalModel",
    "MetaEnsemble",
    "FEATURE_GROUPS",
    # Universe Tiers
    "DataTier",
    "TierAssignment",
    "UniverseTierReport",
    "classify_tiers",
    # Walk-forward
    "WalkForwardBacktest",
    "BacktestMetrics",
    "TradeRecord",
    "compute_buy_hold_metrics",
    "compute_random_baseline",
]
