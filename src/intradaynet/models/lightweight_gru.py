"""
Lightweight GRU model for intraday prediction.

Single-layer unidirectional GRU — the fastest neural network option.
Unidirectional matches the causal nature of live trading (future bars
don't exist at inference time, so bidirectional wastes compute).

Architecture:
    Per-bar: (B, seq_len, 25) → Linear(25→48) → GRU(48, 48) → last hidden → (B, 48)
    Context + Sentiment → fuse → Prediction Heads

~25K params. Trains ~2× faster than ResNLS.
"""

import torch
import torch.nn as nn
from typing import Dict


class LightweightGRU(nn.Module):
    """
    Lightweight GRU for intraday prediction.

    ~25K params. Single unidirectional GRU layer processes the sequence
    causally, matching how the model would be used in live trading.
    Fewer gates than LSTM (reset + update vs input + forget + output + cell)
    means faster training and less overfitting on noisy financial data.
    """

    def __init__(
        self,
        num_per_bar_features: int = 25,
        num_context_features: int = 20,
        num_sentiment_features: int = 24,
        hidden_dim: int = 48,
        num_horizons: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project input features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(num_per_bar_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Single-layer unidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Context and sentiment projections
        ctx_dim = 24
        sent_dim = 12
        self.context_proj = nn.Sequential(
            nn.Linear(num_context_features, ctx_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.sentiment_proj = nn.Sequential(
            nn.Linear(num_sentiment_features, sent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Fusion MLP
        fused_dim = hidden_dim + ctx_dim + sent_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-horizon prediction heads
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "direction": nn.Linear(hidden_dim, 1),
                "magnitude": nn.Linear(hidden_dim, 1),
                "confidence": nn.Linear(hidden_dim, 1),
            })
            for _ in range(num_horizons)
        ])

    def forward(
        self,
        per_bar: torch.Tensor,
        context: torch.Tensor,
        sentiment: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            per_bar: (B, seq_len, num_per_bar_features)
            context: (B, num_context_features)
            sentiment: (B, num_sentiment_features)

        Returns:
            dict with direction_logits, magnitudes, confidences: each (B, num_horizons)
        """
        # 1. Project + GRU
        x = self.input_proj(per_bar)        # (B, L, hidden_dim)
        _, hidden = self.gru(x)             # hidden: (1, B, hidden_dim)
        seq_repr = self.gru_norm(hidden.squeeze(0))  # (B, hidden_dim)

        # 2. Fuse with context + sentiment
        ctx = self.context_proj(context)
        sent = self.sentiment_proj(sentiment)
        fused = self.fusion(torch.cat([seq_repr, ctx, sent], dim=1))

        # 3. Per-horizon predictions
        dir_logits, mags, confs = [], [], []
        for head in self.heads:
            dir_logits.append(head["direction"](fused))
            mags.append(head["magnitude"](fused))
            confs.append(torch.sigmoid(head["confidence"](fused)))

        return {
            "direction_logits": torch.cat(dir_logits, dim=1),
            "magnitudes": torch.cat(mags, dim=1),
            "confidences": torch.cat(confs, dim=1),
        }
