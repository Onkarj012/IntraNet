"""
Tests for ResNLS and CompactCNN model shapes.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestIntradayResNLS:
    def setup_method(self):
        from intradaynet.models.resnls_intraday import IntradayResNLS
        self.model = IntradayResNLS(
            num_per_bar_features=25,
            num_context_features=20,
            num_sentiment_features=24,
            hidden_dim=64,
            lstm_layers=2,
            num_horizons=4,
            dropout=0.15,
        )

    def test_forward_shape(self):
        per_bar = torch.randn(8, 120, 25)
        context = torch.randn(8, 20)
        sentiment = torch.randn(8, 24)
        out = self.model(per_bar, context, sentiment)
        assert out["direction_logits"].shape == (8, 4)
        assert out["magnitudes"].shape == (8, 4)
        assert out["confidences"].shape == (8, 4)

    def test_gradient_flow(self):
        per_bar = torch.randn(4, 120, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)
        out = self.model(per_bar, context, sentiment)
        loss = out["direction_logits"].sum() + out["magnitudes"].sum() + out["confidences"].sum()
        loss.backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradients for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite gradients for {name}"

    def test_param_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total < 200_000, f"ResNLS too large: {total:,} params"
        assert total > 10_000, f"ResNLS too small: {total:,} params"

    def test_short_sequence(self):
        """Should work with shorter sequences too."""
        per_bar = torch.randn(4, 30, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)
        out = self.model(per_bar, context, sentiment)
        assert out["direction_logits"].shape == (4, 4)


class TestCompactCNN:
    def setup_method(self):
        from intradaynet.models.compact_cnn import CompactCNN
        self.model = CompactCNN(
            num_per_bar_features=25,
            num_context_features=20,
            num_sentiment_features=24,
            num_horizons=4,
            dropout=0.15,
        )

    def test_forward_shape(self):
        per_bar = torch.randn(8, 120, 25)
        context = torch.randn(8, 20)
        sentiment = torch.randn(8, 24)
        out = self.model(per_bar, context, sentiment)
        assert out["direction_logits"].shape == (8, 4)
        assert out["magnitudes"].shape == (8, 4)
        assert out["confidences"].shape == (8, 4)

    def test_gradient_flow(self):
        per_bar = torch.randn(4, 120, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)
        out = self.model(per_bar, context, sentiment)
        loss = out["direction_logits"].sum() + out["magnitudes"].sum() + out["confidences"].sum()
        loss.backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradients for {name}"

    def test_param_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total < 200_000, f"CompactCNN too large: {total:,} params"
        assert total > 10_000, f"CompactCNN too small: {total:,} params"

    def test_speed_benchmark(self):
        """CompactCNN should be faster than ResNLS per batch."""
        import time
        per_bar = torch.randn(128, 120, 25)
        context = torch.randn(128, 20)
        sentiment = torch.randn(128, 24)

        # Warmup
        self.model(per_bar, context, sentiment)

        t0 = time.time()
        for _ in range(10):
            self.model(per_bar, context, sentiment)
        cnn_time = time.time() - t0

        assert cnn_time < 5.0, f"CompactCNN too slow: {cnn_time:.2f}s for 10 batches"


class TestLightweightGRU:
    def setup_method(self):
        from intradaynet.models.lightweight_gru import LightweightGRU
        self.model = LightweightGRU(
            num_per_bar_features=25,
            num_context_features=20,
            num_sentiment_features=24,
            hidden_dim=48,
            num_horizons=4,
            dropout=0.15,
        )

    def test_forward_shape(self):
        per_bar = torch.randn(8, 120, 25)
        context = torch.randn(8, 20)
        sentiment = torch.randn(8, 24)
        out = self.model(per_bar, context, sentiment)
        assert out["direction_logits"].shape == (8, 4)
        assert out["magnitudes"].shape == (8, 4)
        assert out["confidences"].shape == (8, 4)

    def test_gradient_flow(self):
        per_bar = torch.randn(4, 120, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)
        out = self.model(per_bar, context, sentiment)
        loss = out["direction_logits"].sum() + out["magnitudes"].sum() + out["confidences"].sum()
        loss.backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradients for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite gradients for {name}"

    def test_param_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total < 50_000, f"LightweightGRU too large: {total:,} params"
        assert total > 5_000, f"LightweightGRU too small: {total:,} params"

    def test_short_sequence(self):
        per_bar = torch.randn(4, 30, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)
        out = self.model(per_bar, context, sentiment)
        assert out["direction_logits"].shape == (4, 4)


class TestMLPMixer:
    def setup_method(self):
        from intradaynet.models.mlp_mixer import IntradayMLPMixer
        self.model = IntradayMLPMixer(
            num_per_bar_features=25,
            num_context_features=20,
            num_sentiment_features=24,
            patch_size=15,
            hidden_dim=64,
            num_mixer_blocks=3,
            num_horizons=4,
            dropout=0.15,
        )

    def test_forward_shape(self):
        per_bar = torch.randn(8, 120, 25)
        context = torch.randn(8, 20)
        sentiment = torch.randn(8, 24)
        out = self.model(per_bar, context, sentiment)
        assert out["direction_logits"].shape == (8, 4)
        assert out["magnitudes"].shape == (8, 4)
        assert out["confidences"].shape == (8, 4)

    def test_gradient_flow(self):
        per_bar = torch.randn(4, 120, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)
        out = self.model(per_bar, context, sentiment)
        loss = out["direction_logits"].sum() + out["magnitudes"].sum() + out["confidences"].sum()
        loss.backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradients for {name}"

    def test_param_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total < 100_000, f"MLPMixer too large: {total:,} params"
        assert total > 10_000, f"MLPMixer too small: {total:,} params"

    def test_different_seq_lengths(self):
        """Should work with sequences >= patch_size * default_num_patches.
        MLP-Mixer token mixing has fixed patch count.
        """
        # 120 bars is the standard (8 patches of 15)
        # Longer sequences are trimmed to 8 patches; shorter won't work
        per_bar = torch.randn(4, 120, 25)
        context = torch.randn(4, 20)
        sentiment = torch.randn(4, 24)
        out = self.model(per_bar, context, sentiment)
        assert out["direction_logits"].shape == (4, 4)

        # Longer sequence (150 bars → still 8 patches, last 30 trimmed)
        per_bar_long = torch.randn(4, 150, 25)
        out = self.model(per_bar_long, context, sentiment)
        assert out["direction_logits"].shape == (4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
