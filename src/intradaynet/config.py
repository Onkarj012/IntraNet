"""
Configuration loading and validation for IntradayNet.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TCNConfig:
    """TCN architecture configuration."""
    channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64])
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.2


@dataclass
class ModelConfig:
    """Model configuration."""
    type: str = "tcn_attention"
    hidden_dim: int = 128
    attn_heads: int = 4
    dropout: float = 0.2
    tcn: TCNConfig = field(default_factory=TCNConfig)
    num_per_bar_features: int = 25
    num_context_features: int = 20
    num_sentiment_features: int = 24
    sequence_length: int = 120


@dataclass
class DataConfig:
    """Data paths and settings."""
    minute_data_dir: str = "nifty500"
    sentiment_csv: str = "sentiment/combined_sentiment_2015_2025.csv"
    index_data_dir: str = ""
    symbols: List[str] = field(default_factory=list)
    market_open: str = "09:15"
    market_close: str = "15:30"
    bars_per_session: int = 375


@dataclass
class SplitsConfig:
    """Train/val/test split dates."""
    train_start: str = "2015-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"


@dataclass
class LossConfig:
    """Loss function weights."""
    direction_weight: float = 0.5
    magnitude_weight: float = 0.3
    confidence_weight: float = 0.2
    gamma_pos: float = 2.0
    gamma_neg: float = 2.0
    downside_weight: float = 1.2
    time_weight_open: float = 1.5  # extra weight for open-period predictions


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    warmup_epochs: int = 3
    sample_interval: int = 15  # sample every N bars within session
    loss: LossConfig = field(default_factory=LossConfig)
    num_workers: int = 0


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    position_size: float = 100000.0  # ₹1L per trade
    max_concurrent: int = 3
    brokerage_per_order: float = 20.0
    stt_rate: float = 0.00025  # 0.025%
    slippage_rate: float = 0.0005  # 0.05%
    stop_loss: float = 0.005  # 0.5%


@dataclass
class IntradayConfig:
    """Main IntradayNet configuration container."""
    horizons: List[int] = field(default_factory=lambda: [15, 30, 60, 375])
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    seed: int = 42
    device: str = "cpu"
    out_dir: str = "runs/intraday"


def load_config(config_path: str) -> IntradayConfig:
    """Load and parse YAML configuration file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Parse nested configurations
    tcn_cfg = TCNConfig(**data.get('model', {}).get('tcn', {}))

    model_data = data.get('model', {})
    model_data.pop('tcn', None)
    model_cfg = ModelConfig(**model_data, tcn=tcn_cfg)

    data_cfg = DataConfig(**data.get('data', {}))
    splits_cfg = SplitsConfig(**data.get('splits', {}))

    loss_cfg = LossConfig(**data.get('train', {}).get('loss', {}))
    train_data = data.get('train', {})
    train_data.pop('loss', None)
    train_cfg = TrainConfig(**train_data, loss=loss_cfg)

    backtest_cfg = BacktestConfig(**data.get('backtest', {}))

    return IntradayConfig(
        horizons=data.get('horizons', [15, 30, 60, 375]),
        data=data_cfg,
        splits=splits_cfg,
        model=model_cfg,
        train=train_cfg,
        backtest=backtest_cfg,
        seed=data.get('seed', 42),
        device=data.get('device', 'cpu'),
        out_dir=data.get('out_dir', 'runs/intraday'),
    )
