"""
Tests for IntradayNet model shapes and forward pass.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intradaynet.models.tcn_attention import (
    CausalConv1d,
    TCNBlock,
    TCNEncoder,
    MultiHeadSelfAttention,
    IntradayTCNAttention,
    PredictionHead,
)
from intradaynet.models.intraday_loss import IntradayLoss


class TestCausalConv1d:
    def test_shape_preserved(self):
        conv = CausalConv1d(32, 64, kernel_size=3, dilation=1)
        x = torch.randn(4, 32, 120)  # (B, C, L)
        out = conv(x)
        assert out.shape == (4, 64, 120), f"Expected (4, 64, 120), got {out.shape}"

    def test_dilated_shape(self):
        conv = CausalConv1d(32, 64, kernel_size=3, dilation=4)
        x = torch.randn(4, 32, 120)
        out = conv(x)
        assert out.shape == (4, 64, 120), f"Dilation should not change length"


class TestTCNBlock:
    def test_same_dim(self):
        block = TCNBlock(64, 64, kernel_size=3, dilation=1)
        x = torch.randn(4, 64, 120)
        out = block(x)
        assert out.shape == (4, 64, 120)

    def test_dim_change(self):
        block = TCNBlock(32, 64, kernel_size=3, dilation=2)
        x = torch.randn(4, 32, 120)
        out = block(x)
        assert out.shape == (4, 64, 120)


class TestTCNEncoder:
    def test_output_shape(self):
        enc = TCNEncoder(input_dim=25, channels=[64, 64, 64, 64, 64])
        x = torch.randn(4, 120, 25)  # (B, L, D)
        out = enc(x)
        assert out.shape == (4, 120, 64), f"Expected (4, 120, 64), got {out.shape}"


class TestMultiHeadSelfAttention:
    def test_shape(self):
        attn = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
        x = torch.randn(4, 120, 64)
        out = attn(x)
        assert out.shape == (4, 120, 64)


class TestPredictionHead:
    def test_output_keys(self):
        head = PredictionHead(128)
        x = torch.randn(4, 128)
        out = head(x)
        assert set(out.keys()) == {"direction_logit", "magnitude", "confidence"}

    def test_output_shapes(self):
        head = PredictionHead(128)
        x = torch.randn(4, 128)
        out = head(x)
        assert out["direction_logit"].shape == (4, 1)
        assert out["magnitude"].shape == (4, 1)
        assert out["confidence"].shape == (4, 1)


class TestIntradayTCNAttention:
    def setup_method(self):
        self.model = IntradayTCNAttention(
            num_per_bar_features=25,
            num_context_features=20,
            num_sentiment_features=24,
            hidden_dim=128,
            tcn_channels=[64, 64, 64, 64, 64],
            attn_heads=4,
            num_horizons=4,
            dropout=0.1,
        )

    def test_forward_shape(self):
        per_bar = torch.randn(8, 120, 25)
        context = torch.randn(8, 20)
        sentiment = torch.randn(8, 24)

        out = self.model(per_bar, context, sentiment)

        assert out["direction_logits"].shape == (8, 4), \
            f"Direction logits: expected (8,4), got {out['direction_logits'].shape}"
        assert out["magnitudes"].shape == (8, 4), \
            f"Magnitudes: expected (8,4), got {out['magnitudes'].shape}"
        assert out["confidences"].shape == (8, 4), \
            f"Confidences: expected (8,4), got {out['confidences'].shape}"

    def test_gradient_flow(self):
        per_bar = torch.randn(4, 120, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)

        out = self.model(per_bar, context, sentiment)
        loss = out["direction_logits"].sum() + out["magnitudes"].sum() + out["confidences"].sum()
        loss.backward()

        # Check gradients exist and are finite
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradients for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite gradients for {name}"

    def test_param_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 100_000, f"Model too small: {total:,} params"
        assert total < 5_000_000, f"Model too large: {total:,} params"


class TestIntradayLoss:
    def setup_method(self):
        self.criterion = IntradayLoss()

    def test_loss_computation(self):
        predictions = {
            "direction_logits": torch.randn(8, 4),
            "magnitudes": torch.randn(8, 4),
            "confidences": torch.sigmoid(torch.randn(8, 4)),
        }
        targets = {
            "direction": torch.randint(0, 2, (8, 4)).float(),
            "magnitude": torch.randn(8, 4) * 0.01,
        }

        losses = self.criterion(predictions, targets)

        assert "total_loss" in losses
        assert losses["total_loss"].item() > 0
        assert torch.isfinite(losses["total_loss"])

    def test_time_weighting(self):
        predictions = {
            "direction_logits": torch.randn(8, 4),
            "magnitudes": torch.randn(8, 4),
            "confidences": torch.sigmoid(torch.randn(8, 4)),
        }
        targets = {
            "direction": torch.randint(0, 2, (8, 4)).float(),
            "magnitude": torch.randn(8, 4) * 0.01,
        }
        time_norm = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9])

        losses = self.criterion(predictions, targets, time_norm)
        assert torch.isfinite(losses["total_loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
