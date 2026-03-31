"""
TCN + Attention model for intraday predictions.

Architecture:
    Minute Bars → [Input Proj] → [TCN: 5 dilated conv blocks] →
    [1-layer Multi-Head Attention] → [Context/Sentiment Fusion MLP] →
    [Per-horizon Prediction Heads]

The TCN uses dilated causal convolutions with dilation rates [1,2,4,8,16]
giving a receptive field of ~124 bars (~2 hours of trading).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C_out, L)"""
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Residual TCN block with gated activation.

    Structure: CausalConv → BatchNorm → GLU → Dropout →
               CausalConv → BatchNorm → GLU → Dropout → Residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        # Two causal conv layers with gated activation
        self.conv1 = CausalConv1d(in_channels, out_channels * 2, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels * 2)

        self.conv2 = CausalConv1d(out_channels, out_channels * 2, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels * 2)

        self.dropout = nn.Dropout(dropout)

        # Residual connection (1x1 conv if dimensions differ)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C_out, L)"""
        residual = self.residual(x)

        # Layer 1 with GLU activation
        out = self.bn1(self.conv1(x))
        out_a, out_b = out.chunk(2, dim=1)
        out = out_a * torch.sigmoid(out_b)  # Gated Linear Unit
        out = self.dropout(out)

        # Layer 2 with GLU activation
        out = self.bn2(self.conv2(out))
        out_a, out_b = out.chunk(2, dim=1)
        out = out_a * torch.sigmoid(out_b)
        out = self.dropout(out)

        return F.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Stack of TCN blocks with increasing dilation rates.

    Dilation rates: [1, 2, 4, 8, 16] with kernel_size=3 gives
    receptive field = sum(2 * (k-1) * d) for each layer ≈ 124 bars.
    """

    def __init__(self, input_dim: int, channels: List[int],
                 kernel_size: int = 3, dilation_base: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        num_layers = len(channels)

        for i in range(num_layers):
            in_ch = input_dim if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation = dilation_base ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) sequence
        Returns:
            (B, L, C_last) TCN encoded sequence
        """
        # Conv1d expects (B, C, L)
        out = x.transpose(1, 2)
        for layer in self.network:
            out = layer(out)
        # Back to (B, L, C)
        return out.transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Single-layer multi-head self-attention for session-spanning patterns."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D) attended sequence
        """
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class PredictionHead(nn.Module):
    """Per-horizon prediction head: direction + magnitude + confidence."""

    def __init__(self, input_dim: int, dropout: float = 0.15):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.direction = nn.Linear(input_dim // 2, 1)
        self.magnitude = nn.Linear(input_dim // 2, 1)
        self.confidence = nn.Linear(input_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, input_dim)
        Returns:
            dict with direction_logit (B,1), magnitude (B,1), confidence (B,1)
        """
        h = self.shared(x)
        return {
            "direction_logit": self.direction(h),
            "magnitude": self.magnitude(h),
            "confidence": torch.sigmoid(self.confidence(h)),
        }


class IntradayTCNAttention(nn.Module):
    """
    Complete IntradayNet model: TCN + Attention with context/sentiment fusion.

    Pipeline:
        1. Project per-bar features → hidden_dim
        2. TCN encoder (5 dilated blocks)
        3. Multi-head self-attention (1 layer)
        4. Global average pooling → sequence representation
        5. Concatenate with context + sentiment features
        6. Fusion MLP
        7. Per-horizon prediction heads
    """

    def __init__(
        self,
        num_per_bar_features: int = 25,
        num_context_features: int = 20,
        num_sentiment_features: int = 24,
        hidden_dim: int = 128,
        tcn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dilation_base: int = 2,
        attn_heads: int = 4,
        num_horizons: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [64, 64, 64, 64, 64]

        self.num_horizons = num_horizons
        tcn_out_dim = tcn_channels[-1]

        # 1. Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_per_bar_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. TCN encoder
        self.tcn = TCNEncoder(
            input_dim=hidden_dim,
            channels=tcn_channels,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
        )

        # 3. Self-attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=tcn_out_dim,
            num_heads=attn_heads,
            dropout=dropout,
        )

        # 4. Context/Sentiment fusion
        fusion_input_dim = tcn_out_dim + num_context_features + num_sentiment_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 5. Per-horizon prediction heads
        self.heads = nn.ModuleList([
            PredictionHead(hidden_dim, dropout) for _ in range(num_horizons)
        ])

    def forward(
        self,
        per_bar: torch.Tensor,
        context: torch.Tensor,
        sentiment: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            per_bar: (B, seq_len, num_per_bar_features) — minute bar features
            context: (B, num_context_features) — session-level context
            sentiment: (B, num_sentiment_features) — sentiment features

        Returns:
            Dict with keys per horizon (h0, h1, h2, h3):
                direction_logits: (B, num_horizons)
                magnitudes: (B, num_horizons)
                confidences: (B, num_horizons)
        """
        # 1. Project input features
        x = self.input_proj(per_bar)  # (B, L, hidden_dim)

        # 2. TCN encoding
        x = self.tcn(x)  # (B, L, tcn_out)

        # 3. Self-attention
        x = self.attention(x)  # (B, L, tcn_out)

        # 4. Global average pooling
        x = x.mean(dim=1)  # (B, tcn_out)

        # 5. Fuse with context and sentiment
        fused = torch.cat([x, context, sentiment], dim=1)
        fused = self.fusion(fused)  # (B, hidden_dim)

        # 6. Per-horizon predictions
        direction_logits = []
        magnitudes = []
        confidences = []

        for head in self.heads:
            pred = head(fused)
            direction_logits.append(pred["direction_logit"])
            magnitudes.append(pred["magnitude"])
            confidences.append(pred["confidence"])

        return {
            "direction_logits": torch.cat(direction_logits, dim=1),  # (B, H)
            "magnitudes": torch.cat(magnitudes, dim=1),              # (B, H)
            "confidences": torch.cat(confidences, dim=1),            # (B, H)
        }
