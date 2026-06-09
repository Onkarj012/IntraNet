"""
V8 configuration — single source of truth for all V8 parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TargetConfig:
    """Barrier target parameters."""
    target_pct: float = 0.015          # 1.5% target
    stop_pct: float = 0.010            # 1.0% stop-loss
    cost_buffer_pct: float = 0.0018    # transaction cost estimate
    ambiguity_band_pct: float = 0.0025  # minimum edge gap to classify
    min_tradable_move_pct: float = 0.0075

    # Multi-horizon support (single-day active, others planned)
    horizons: tuple[str, ...] = ("H375",)  # future: ("H30", "H60", "H375")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Curve autoencoder configuration."""
    enabled: bool = True
    input_dim: int = 4                  # OHLC
    sequence_length: int = 75           # 5-min bars (downsampled from 1-min)
    embedding_dim: int = 128            # learned representation size
    mask_ratio: float = 0.6             # fraction of bars to mask
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 64
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_epochs: int = 50
    early_stopping_patience: int = 5

    # Training data
    min_bars_per_day: int = 200         # minimum bars for a valid session (raw 1-min)
    max_gap_pct: float = 0.15           # max allowed gap from prev close
    train_years: tuple[int, ...] = tuple(range(2015, 2024))  # 2015-2023

    # Subsampling (prevents O(n) blowup with many stocks)
    subsample_max_sessions: int = 15000  # max training sessions across all stocks
    subsample_seed: int = 42

    # Downsampling (1-min → N-min aggregation)
    downsample_minutes: int = 5         # aggregate to 5-min bars; set 1 for raw 1-min


@dataclass(frozen=True)
class SignalModelConfig:
    """Configuration for a single specialist signal model."""
    name: str
    lgb_objective: str = "binary"
    lgb_metric: str = "binary_logloss"
    lgb_num_leaves: int = 127
    lgb_max_bin: int = 255
    lgb_min_child_samples: int = 200
    lgb_subsample: float = 0.7
    lgb_colsample_bytree: float = 0.7
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 0.1
    lgb_n_estimators: int = 5000
    lgb_early_stopping_rounds: int = 100
    lgb_scale_pos_weight: Optional[float] = None  # auto-computed from class balance
    use_is_unbalance: bool = False

    # Specialist emphasis — feature group weights for this model
    feature_group_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaEnsembleConfig:
    """Meta-ensemble configuration."""
    regime_labels: tuple[str, ...] = (
        "strong_trend_up",
        "strong_trend_down",
        "choppy_reverting",
        "high_vol_crisis",
        "low_vol_compression",
    )
    # Regime detection features
    regime_features: tuple[str, ...] = (
        "vix_level",
        "vix_5d_change",
        "nifty_adx",
        "breadth_20d",
        "nifty_autocorr",
        "sector_dispersion",
    )
    regime_detection_method: str = "kmeans"  # kmeans | gmm | threshold
    n_regimes: int = 5
    warmup_years: int = 2  # years of history for regime clustering


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction configuration."""
    min_confidence: float = 0.58
    min_expected_value: float = 0.0
    max_picks: int = 5                    # configurable by user
    max_per_side: int = 4                 # max LONG or SHORT
    sector_penalty: float = 0.15           # penalty for same-sector picks
    correlation_penalty: float = 0.10      # penalty for correlated picks
    correlation_lookback_days: int = 21
    risk_per_trade_pct: float = 0.02       # 2% risk per trade
    max_sector_exposure: float = 0.40      # max 40% in one sector
    position_sizing: str = "equal_risk"    # equal_risk | kelly | fixed


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward backtest configuration."""
    train_years: tuple[tuple[int, ...], ...] = (
        (2015, 2016, 2017, 2018, 2019, 2020),
        (2015, 2016, 2017, 2018, 2019, 2020, 2021),
        (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
        (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023),
    )
    test_years: tuple[tuple[int, ...], ...] = (
        (2021,),
        (2022,),
        (2023,),
        (2024, 2025),
    )
    retrain_embedding_yearly: bool = True
    min_volume_filter: int = 100000       # min daily volume in shares
    min_price: float = 20.0               # min closing price
    max_price: float = 10000.0            # max closing price


@dataclass(frozen=True)
class SentimentConfig:
    """Per-stock sentiment configuration."""
    news_lookback_days: int = 3            # how many days of news to aggregate
    sentiment_momentum_days: int = 5       # sentiment change lookback
    min_articles_for_score: int = 2        # minimum articles for valid score
    use_live_yfinance: bool = True
    fallback_to_historical: bool = True
    sentiment_features: tuple[str, ...] = (
        "sentiment_score_1d",
        "sentiment_score_3d",
        "sentiment_momentum_5d",
        "article_count_1d",
        "article_count_3d",
        "sentiment_volatility_5d",
        "headline_sentiment_bias",
        "news_to_price_ratio",
    )


@dataclass(frozen=True)
class DailyFeaturesConfig:
    """Engineered daily features (Stream B)."""
    price_action_features: tuple[str, ...] = (
        "return_1d", "return_5d", "return_10d", "return_21d", "return_63d",
        "parkinson_vol_5d", "parkinson_vol_21d",
        "gk_vol_5d", "gk_vol_21d",
        "rel_volume_20d", "volume_trend_5d",
        "price_position_20d", "price_position_63d",
        "gap_size", "overnight_return",
        "rs_vs_sector_5d", "rs_vs_sector_21d",
        "high_low_range_pct", "close_vs_vwap", "afternoon_vs_morning",
    )
    context_features: tuple[str, ...] = (
        "vix_level", "vix_trend_5d",
        "nifty_vs_50dma", "nifty_vs_200dma",
        "breadth_pct_above_20dma",
        "sector_return_1d", "sector_return_5d",
        "sp500_overnight", "usdinr_change", "crude_change",
        "day_of_week", "month", "expiry_week", "budget_day",
    )


@dataclass(frozen=True)
class V8Config:
    """Master configuration for V8 pipeline."""
    target: TargetConfig = field(default_factory=TargetConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    signal_models: tuple[SignalModelConfig, ...] = field(
        default_factory=lambda: _default_signal_models()
    )
    meta_ensemble: MetaEnsembleConfig = field(default_factory=MetaEnsembleConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    daily_features: DailyFeaturesConfig = field(default_factory=DailyFeaturesConfig)

    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models/v8")
    cache_dir: Path = Path("cache/v8")
    output_dir: Path = Path("outputs/v8")

    # Runtime
    seed: int = 42
    n_jobs: int = -1
    device: str = "cpu"
    verbose: bool = True

    @classmethod
    def default(cls) -> V8Config:
        return cls()

    def save(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = _dataclass_to_dict(self)
        path.write_text(json.dumps(config_dict, indent=2, default=str), encoding="utf-8")


def _default_signal_models() -> tuple[SignalModelConfig, ...]:
    return (
        SignalModelConfig(
            name="momentum",
            feature_group_weights={"returns": 1.0, "curve_embedding": 0.8, "relative_strength": 0.6},
        ),
        SignalModelConfig(
            name="reversal",
            feature_group_weights={"price_position": 1.0, "gap": 0.8, "volatility": 0.6, "curve_embedding": 0.5},
        ),
        SignalModelConfig(
            name="breakout",
            feature_group_weights={"volatility_pct": 1.0, "volume_dryup": 0.8, "curve_embedding": 0.7},
        ),
        SignalModelConfig(
            name="sentiment",
            feature_group_weights={"sentiment": 1.0, "headline": 0.5, "news_volume": 0.5},
        ),
        SignalModelConfig(
            name="macro",
            feature_group_weights={"vix": 1.0, "breadth": 0.8, "global": 0.6},
        ),
    )


def _dataclass_to_dict(obj) -> dict:
    """Recursively convert dataclass to dict."""
    from dataclasses import fields, is_dataclass

    if isinstance(obj, tuple):
        return tuple(_dataclass_to_dict(x) for x in obj)
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in fields(obj)
        }
    return obj
