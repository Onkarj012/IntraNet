"""
Compact CNN model for intraday prediction.

Pure Conv1D with aggressive pooling — no sequential ops (LSTM/attention),
fully parallelizable on CPU. Fastest neural network option.

Architecture:
    Per-bar: (B, seq_len, 25) → 3× [Conv1D + ReLU + MaxPool] → AvgPool → (B, 64)
    Context + Sentiment → fuse → Prediction Heads
"""

import torch
import torch.nn as nn
from typing import Dict


class CompactCNN(nn.Module):
    """
    Compact 1D CNN for intraday prediction.

    ~60K params. Three Conv1D blocks with aggressive pooling reduce
    the 120-step sequence to a fixed-size vector in 3 operations.
    No LSTM or attention — pure convolutions for maximum CPU speed.
    """

    def __init__(
        self,
        num_per_bar_features: int = 25,
        num_context_features: int = 20,
        num_sentiment_features: int = 24,
        channels: tuple = (64, 128, 64),
        kernel_sizes: tuple = (5, 3, 3),
        pool_sizes: tuple = (4, 4, 0),  # 0 = no pool (use adaptive at end)
        num_horizons: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()

        # Conv blocks
        in_ch = num_per_bar_features
        layers = []
        for i, (out_ch, ks, ps) in enumerate(zip(channels, kernel_sizes, pool_sizes)):
            layers.append(nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            if ps > 0:
                layers.append(nn.MaxPool1d(ps))
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool1d(1))  # → (B, last_ch, 1)
        self.encoder = nn.Sequential(*layers)

        # Context and sentiment
        cnn_out = channels[-1]
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

        # Fusion
        fused_dim = cnn_out + ctx_dim + sent_dim
        hidden_dim = 64
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-horizon heads
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
            per_bar: (B, seq_len, features) — note: seq_len first, Conv1d needs (B, C, L)
            context: (B, num_context_features)
            sentiment: (B, num_sentiment_features)
        Returns:
            dict with direction_logits, magnitudes, confidences: each (B, num_horizons)
        """
        # Conv expects (B, C, L) — transpose from (B, L, C)
        x = per_bar.transpose(1, 2)  # (B, features, seq_len)
        x = self.encoder(x)          # (B, last_ch, 1)
        x = x.squeeze(-1)            # (B, last_ch)

        # Fuse
        ctx = self.context_proj(context)
        sent = self.sentiment_proj(sentiment)
        fused = self.fusion(torch.cat([x, ctx, sent], dim=1))

        # Heads
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
