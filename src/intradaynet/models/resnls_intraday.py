"""
ResNLS (ResNet + LSTM) model for intraday prediction.

Adapted from StockXpert's ResNLSEncoder — proven architecture for
short-term return prediction. Lightweight and fast on CPU.

Architecture:
    Per-bar: (B, seq_len, 25) → Projection → 2× ResBlock → BiLSTM → (B, hidden)
    Context: (B, 20) → Linear → (B, ctx_dim)
    Sentiment: (B, 14) → Linear → (B, sent_dim)
    Concat → MLP → 4 Prediction Heads
"""

import torch
import torch.nn as nn
from typing import Dict


class ResBlock(nn.Module):
    """Point-wise residual block: Linear → ReLU → Dropout → Linear + skip."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return torch.relu(self.net(x) + x)


class IntradayResNLS(nn.Module):
    """
    ResNLS (ResNet + LSTM) for intraday prediction.

    ~80K params with default settings. Processes sequences through:
    1. Linear projection to hidden dim
    2. 2 residual blocks (point-wise, preserves time dimension)
    3. BiLSTM for temporal mixing → final hidden state
    4. Fusion with context + sentiment
    5. Per-horizon prediction heads
    """

    def __init__(
        self,
        num_per_bar_features: int = 25,
        num_context_features: int = 20,
        num_sentiment_features: int = 24,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        num_horizons: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Per-bar encoder
        self.input_proj = nn.Linear(num_per_bar_features, hidden_dim)
        self.res_block1 = ResBlock(hidden_dim, dropout)
        self.res_block2 = ResBlock(hidden_dim, dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # BiLSTM → hidden_dim total
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Context and sentiment projections
        ctx_dim = 32
        sent_dim = 16
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
        # 1. Project + ResBlocks
        x = self.input_proj(per_bar)  # (B, L, hidden_dim)
        x = self.res_block1(x)
        x = self.res_block2(x)

        # 2. BiLSTM → final hidden state
        _, (hidden, _) = self.lstm(x)
        # hidden: (num_layers*2, B, hidden_dim//2) → take last layer fwd + bwd
        h_fwd = hidden[-2]  # (B, hidden_dim//2)
        h_bwd = hidden[-1]  # (B, hidden_dim//2)
        seq_repr = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden_dim)

        # 3. Fuse with context + sentiment
        ctx = self.context_proj(context)
        sent = self.sentiment_proj(sentiment)
        fused = self.fusion(torch.cat([seq_repr, ctx, sent], dim=1))  # (B, hidden_dim)

        # 4. Per-horizon predictions
        dir_logits, mags, confs = [], [], []
        for head in self.heads:
            dir_logits.append(head["direction"](fused))
            mags.append(head["magnitude"](fused))
            confs.append(torch.sigmoid(head["confidence"](fused)))

        return {
            "direction_logits": torch.cat(dir_logits, dim=1),   # (B, H)
            "magnitudes": torch.cat(mags, dim=1),                # (B, H)
            "confidences": torch.cat(confs, dim=1),              # (B, H)
        }
