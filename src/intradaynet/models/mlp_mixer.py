"""
MLP-Mixer model for intraday prediction.

Inspired by Google's MLP-Mixer — uses alternating token-mixing and
channel-mixing MLPs instead of attention or recurrence. Fully
parallelizable with no sequential bottleneck.

Architecture:
    Per-bar: (B, 120, 25) → patch into 8×(15×25) → Linear(375→64) per patch
    → 3× MixerBlock(token_mix + channel_mix) → mean pool → (B, 64)
    Context + Sentiment → fuse → Prediction Heads

~40K params. Comparable speed to CompactCNN.
"""

import torch
import torch.nn as nn
from typing import Dict


class MixerBlock(nn.Module):
    """
    Single Mixer block: token mixing + channel mixing.

    Token mixing: transpose → MLP across patches → transpose back
    Channel mixing: MLP across features within each patch
    """

    def __init__(self, num_patches: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()

        # Token mixing (mix across time patches)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches * 2, num_patches),
            nn.Dropout(dropout),
        )

        # Channel mixing (mix across features)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, hidden_dim)
        Returns:
            (B, num_patches, hidden_dim)
        """
        # Token mixing: (B, P, D) → transpose → (B, D, P) → MLP → transpose back
        residual = x
        x = self.token_norm(x)
        x = x.transpose(1, 2)                # (B, D, P)
        x = self.token_mix(x)                 # (B, D, P)
        x = x.transpose(1, 2) + residual     # (B, P, D)

        # Channel mixing: (B, P, D) → MLP → (B, P, D)
        residual = x
        x = self.channel_norm(x)
        x = self.channel_mix(x) + residual    # (B, P, D)

        return x


class IntradayMLPMixer(nn.Module):
    """
    MLP-Mixer for intraday prediction.

    ~40K params. Patches the 120-bar sequence into 8 patches of 15 bars,
    then applies 3 Mixer blocks for time/feature mixing.

    No attention, no recurrence — fully parallelizable on CPU.
    """

    def __init__(
        self,
        num_per_bar_features: int = 25,
        num_context_features: int = 20,
        num_sentiment_features: int = 24,
        patch_size: int = 15,
        hidden_dim: int = 64,
        num_mixer_blocks: int = 3,
        num_horizons: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        patch_input_dim = patch_size * num_per_bar_features  # 15 × 25 = 375

        # Patch embedding: flatten each patch → project to hidden_dim
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Calculate num_patches (for 120 bars / 15 patch_size = 8 patches)
        # This is set dynamically in forward, but we need it for MixerBlocks
        # Default: 120 / 15 = 8
        default_num_patches = 120 // patch_size
        self.default_num_patches = default_num_patches

        # Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(default_num_patches, hidden_dim, dropout)
            for _ in range(num_mixer_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

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
        fusion_hidden = 48
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-horizon prediction heads
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "direction": nn.Linear(fusion_hidden, 1),
                "magnitude": nn.Linear(fusion_hidden, 1),
                "confidence": nn.Linear(fusion_hidden, 1),
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
        B, L, F = per_bar.shape

        # 1. Always use exactly default_num_patches patches from the end
        required_bars = self.default_num_patches * self.patch_size
        # Take last `required_bars` bars (or all if shorter)
        if L >= required_bars:
            trimmed = per_bar[:, -required_bars:, :]
        else:
            # Pad with zeros if too short (shouldn't happen with seq_len=120)
            pad = torch.zeros(B, required_bars - L, F, device=per_bar.device, dtype=per_bar.dtype)
            trimmed = torch.cat([pad, per_bar], dim=1)

        patches = trimmed.reshape(B, self.default_num_patches, self.patch_size * F)

        # 2. Patch embedding
        x = self.patch_embed(patches)  # (B, num_patches, hidden_dim)

        # 3. Mixer blocks
        for block in self.mixer_blocks:
            x = block(x)

        # 4. Pool across patches
        x = self.final_norm(x)
        x = x.mean(dim=1)  # (B, hidden_dim)

        # 5. Fuse with context + sentiment
        ctx = self.context_proj(context)
        sent = self.sentiment_proj(sentiment)
        fused = self.fusion(torch.cat([x, ctx, sent], dim=1))

        # 6. Per-horizon predictions
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
