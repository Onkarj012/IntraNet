"""OptiNet v3 ResNLS: ResNet conv stack + BiLSTM for intraday spot direction.

Architecture (~80K params target, matches IntradayNet's published ResNLS):
  Input  : [batch, seq_len=60, n_features=5]
  → permute to [batch, n_features, seq_len] for Conv1d
  → Conv1d(5 → 32, kernel=3, padding=1) + BN + ReLU
  → ResBlock(32 → 32, kernel=3) ×2
  → ResBlock(32 → 64, kernel=3) ×2  (downsample)
  → permute to [batch, seq_len, channels]
  → BiLSTM(64 → 32 hidden, bidirectional → 64 features)
  → mean-pool over time → [batch, 64]
  → Linear(64 → 32) + ReLU + Dropout
  → 2 heads:
       - long_logit  → P(label_long_1h=1)
       - short_logit → P(label_short_1h=1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNLS(nn.Module):
    """Residual conv stack + BiLSTM with two binary classification heads."""

    def __init__(
        self,
        n_features: int = 5,
        seq_len: int = 60,
        conv_channels: tuple[int, int] = (32, 64),
        lstm_hidden: int = 32,
        head_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        c1, c2 = conv_channels
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock1d(c1, c1, dropout=dropout * 0.5)
        self.res2 = ResBlock1d(c1, c2, stride=1, dropout=dropout * 0.5)
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        feat_dim = lstm_hidden * 2
        self.head_shared = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head_long = nn.Linear(head_hidden, 1)
        self.head_short = nn.Linear(head_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, n_features]
        x = x.permute(0, 2, 1)              # → [batch, n_features, seq_len]
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = x.permute(0, 2, 1)              # → [batch, seq_len, channels]
        x, _ = self.lstm(x)
        x = x.mean(dim=1)                   # mean-pool over time
        x = self.head_shared(x)
        long_logit = self.head_long(x).squeeze(-1)
        short_logit = self.head_short(x).squeeze(-1)
        return long_logit, short_logit


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
