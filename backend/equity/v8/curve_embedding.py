"""
Curve Embedding Model — Masked Autoencoder for Intraday Price Curves.

Trains a Transformer encoder to reconstruct masked minute bars of OHLC data.
After pre-training, the encoder produces a 128-dim embedding per stock-day that
captures intraday "shape" patterns — V-recoveries, slow grinds, gap-and-traps,
compression breakouts — without needing labels.

Architecture:
    Input (375×4) → Mask 60% → Embed + PosEncode → Transformer Encoder
    → Linear Decoder → Reconstruct masked bars → MSE Loss

The encoder is trained once on ALL Tier 1 stocks' full history and is reusable.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import EmbeddingConfig


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ---------------------------------------------------------------------------
# Masked Autoencoder for Curves
# ---------------------------------------------------------------------------

class CurveMaskedEncoder(nn.Module):
    """
    Masked autoencoder for intraday price curves.

    Encodes (B, T, 4) OHLC sequences with 60% random masking, producing
    a 128-dim embedding per stock-day after pre-training.

    Parameters
    ----------
    input_dim : int
        Number of input channels (4 for OHLC).
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    embedding_dim : int
        Output embedding dimension (pooled from encoder).
    dropout : float
        Dropout rate.
    mask_ratio : float
        Fraction of bars to mask (0.0 to 1.0).
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        mask_ratio: float = 0.6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio

        # Input projection: (B, T, 4) → (B, T, d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=500, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder: lightweight linear layer to reconstruct masked bars
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

        # Embedding projection: mean pool → [CLS-style] → embedding_dim
        self.embedding_proj = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask bars in the input sequence.

        Parameters
        ----------
        x : (B, T, C) tensor
        mask_ratio : float, optional

        Returns
        -------
        masked_x : (B, T, d_model) — masked input projected to d_model
        mask : (B, T) — boolean mask (True = unmasked)
        ids_restore : (B, T) — indices to restore original order
        """
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        B, T, C = x.shape
        len_keep = int(T * (1 - mask_ratio))
        len_keep = max(1, len_keep)

        noise = torch.rand(B, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.zeros([B, T], device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, True)

        x_proj = self.input_proj(x)
        mask_tokens = self.mask_token.expand(B, T, -1)
        masked_proj = torch.where(mask.unsqueeze(-1), x_proj, mask_tokens)

        return masked_proj, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode masked sequence through Transformer."""
        masked_proj, mask, ids_restore = self.random_masking(x, mask_ratio)
        masked_proj = self.pos_encoding(masked_proj)
        encoded = self.encoder(masked_proj)
        return encoded, mask, ids_restore

    def forward_decoder(
        self,
        encoded: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decode encoded representations back to OHLC."""
        decoded = self.decoder(encoded)
        decoded = torch.gather(
            decoded, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.input_dim),
        )
        return decoded

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: mask → encode → decode.

        Returns
        -------
        pred : (B, T, C) — reconstructed bars
        target : (B, T, C) — original bars
        mask : (B, T) — which positions were masked
        """
        encoded, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(encoded, ids_restore)
        return pred, x, mask

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the 128-dim embedding for a stock-day (unmasked).
        Used at inference time.

        Parameters
        ----------
        x : (B, T, 4) — full, unmasked OHLC sequence

        Returns
        -------
        (B, embedding_dim) — per-day embedding vector
        """
        x_proj = self.input_proj(x)
        x_proj = self.pos_encoding(x_proj)
        encoded = self.encoder(x_proj)
        pooled = encoded.mean(dim=1)
        return self.embedding_proj(pooled)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """
    MSE loss computed only on masked positions.

    Parameters
    ----------
    pred : (B, T, C) — reconstructed bars
    target : (B, T, C) — original bars
    mask : (B, T) — True for unmasked, False for masked
    normalize : bool — if True, normalize loss by number of masked elements
    """
    loss_mask = ~mask  # compute loss on masked positions
    if not loss_mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    diff = (pred - target) ** 2
    loss = diff[loss_mask.unsqueeze(-1).expand_as(pred)]

    if normalize:
        return loss.mean()
    return loss.sum() / loss_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CurveDataset(Dataset):
    """
    PyTorch Dataset for intraday OHLC curves.

    Each sample is a (375, 4) tensor representing one stock-day.
    Data is loaded lazily from pre-processed numpy arrays.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        normalize: bool = True,
        normalize_method: str = "zscore_per_day",  # zscore_per_day | minmax_per_day | global_zscore
        clip_outliers: float = 5.0,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_timesteps, 4) — [O, H, L, C].
        normalize : bool
            Whether to normalize data.
        normalize_method : str
            How to normalize: 'zscore_per_day' normalizes each day's OHLC
            using its own statistics (recommended for premarket).
        clip_outliers : float
            Number of standard deviations to clip.
        """
        self.data = torch.from_numpy(data.astype(np.float32))
        self.n_samples = len(data)
        self.normalize = normalize
        self.normalize_method = normalize_method

        if normalize:
            self.data = self._normalize(self.data, normalize_method, clip_outliers)

    @staticmethod
    def _normalize(data: torch.Tensor, method: str, clip: float) -> torch.Tensor:
        if method == "zscore_per_day":
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True).clamp(min=1e-8)
            data = (data - mean) / std
            data = torch.clamp(data, -clip, clip)
        elif method == "minmax_per_day":
            lo = data.amin(dim=1, keepdim=True)
            hi = data.amax(dim=1, keepdim=True)
            data = (data - lo) / (hi - lo).clamp(min=1e-8)
            data = data * 2 - 1  # scale to [-1, 1]
        elif method == "global_zscore":
            mean = data.mean()
            std = data.std().clamp(min=1e-8)
            data = (data - mean) / std
            data = torch.clamp(data, -clip, clip)
        return data

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def prepare_curve_data(
    sessions: dict[pd.Timestamp, pd.DataFrame],
    sequence_length: int = 75,
    min_bars: int = 200,
    pad_value: float = 0.0,
    downsample_minutes: int = 5,
) -> np.ndarray:
    """
    Convert session DataFrames to a numpy array of shape (n_sessions, T, 4).

    Parameters
    ----------
    sessions : dict
        Date → minute DataFrame mapping.
    sequence_length : int
        Number of timesteps per sample (after downsampling).
    min_bars : int
        Minimum raw 1-min bars required for a valid session.
    pad_value : float
        Value to pad shorter sessions.
    downsample_minutes : int
        Aggregate raw 1-min bars to this resolution (1 = no downsampling).

    Returns
    -------
    np.ndarray
        Shape (n_valid_sessions, sequence_length, 4).
    """
    from .data_pipeline import downsample_curve_ohlc

    valid_dates = sorted(
        date for date, df in sessions.items() if len(df) >= min_bars
    )
    if not valid_dates:
        return np.zeros((0, sequence_length, 4), dtype=np.float32)

    data = np.full((len(valid_dates), sequence_length, 4), pad_value, dtype=np.float32)

    for i, date in enumerate(valid_dates):
        df = sessions[date]
        n_raw_bars = len(df)
        raw_curve = np.zeros((n_raw_bars, 4), dtype=np.float32)
        for j, col in enumerate(["open", "high", "low", "close"]):
            raw_curve[:, j] = df[col].values.astype(np.float32)

        if downsample_minutes > 1:
            curve = downsample_curve_ohlc(raw_curve, target_minutes=downsample_minutes)
        else:
            curve = raw_curve

        n_bars = min(len(curve), sequence_length)
        data[i, :n_bars] = curve[:n_bars]

    return data


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class CurveTrainer:
    """
    Training loop for the curve masked autoencoder.

    Handles:
    - Training/validation loop
    - Learning rate warmup + cosine decay
    - Early stopping
    - Checkpoint management
    - MPS/CPU device handling
    """

    model: CurveMaskedEncoder
    config: EmbeddingConfig
    device: str = "cpu"
    output_dir: str | Path = "models/v8/embedding"
    log_interval: int = 10

    # Internal state
    _train_losses: list[float] = field(default_factory=list)
    _val_losses: list[float] = field(default_factory=list)
    _best_val_loss: float = float("inf")
    _epochs_without_improvement: int = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int | None = None,
    ) -> dict[str, float]:
        """Run full training loop."""
        num_epochs = num_epochs or self.config.max_epochs
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

        self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, epoch)
            val_loss = self._validate(val_loader)

            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self._epochs_without_improvement += 1

            if (epoch + 1) % self.log_interval == 0:
                print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
                      f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

            if self._epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "best_val_loss": self._best_val_loss,
            "epochs_trained": len(self._train_losses),
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)
            pred, target, mask = self.model(batch, self.config.mask_ratio)
            loss = masked_mse_loss(pred, target, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                avg_so_far = total_loss / (batch_idx + 1)
                print(f"  epoch {epoch + 1:3d} | batch {batch_idx + 1:4d}/{n_batches} | "
                      f"avg_loss={avg_so_far:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = len(loader)

        for batch in loader:
            batch = batch.to(self.device)
            pred, target, mask = self.model(batch, self.config.mask_ratio)
            loss = masked_mse_loss(pred, target, mask)
            total_loss += loss.item()

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "config": {
                "input_dim": self.config.input_dim,
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "embedding_dim": self.config.embedding_dim,
                "mask_ratio": self.config.mask_ratio,
                "dropout": self.config.dropout,
            },
        }
        torch.save(checkpoint, output_dir / "best_model.pt")
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:03d}.pt")


def load_curve_encoder(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> CurveMaskedEncoder:
    """
    Load a trained curve encoder from checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to .pt checkpoint.
    device : str
        Device to load model on.

    Returns
    -------
    CurveMaskedEncoder
        Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = CurveMaskedEncoder(
        input_dim=config.get("input_dim", 4),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 4),
        embedding_dim=config.get("embedding_dim", 128),
        dropout=config.get("dropout", 0.1),
        mask_ratio=config.get("mask_ratio", 0.6),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_embeddings(
    encoder: CurveMaskedEncoder,
    dataset: CurveDataset | DataLoader,
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate 128-dim embeddings for all samples in a dataset.

    Parameters
    ----------
    encoder : CurveMaskedEncoder
        Trained encoder in eval mode.
    dataset : CurveDataset or DataLoader
        Data to embed.
    batch_size : int
        Batch size for embedding generation.
    device : str
        Device to run inference on.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        Shape (n_samples, embedding_dim).
    """
    encoder.to(device)
    encoder.eval()

    if isinstance(dataset, CurveDataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        loader = dataset

    embeddings = []
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        emb = encoder.get_embedding(batch)
        embeddings.append(emb.cpu().numpy())

        if verbose and (batch_idx + 1) % 50 == 0:
            print(f"  Embedded {batch_idx * batch_size + len(batch)} / {len(loader.dataset)} samples")

    result = np.concatenate(embeddings, axis=0)
    if verbose:
        print(f"Generated {result.shape[0]} embeddings of dim {result.shape[1]}")

    return result


def save_embeddings(
    embeddings: np.ndarray,
    dates: list[pd.Timestamp],
    symbols: list[str],
    output_path: str | Path,
) -> None:
    """
    Save generated embeddings with metadata.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_samples, embedding_dim).
    dates : list[pd.Timestamp]
        Date for each sample.
    symbols : list[str]
        Symbol for each sample.
    output_path : str | Path
        Path to save .npz file.
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        dates=np.array([str(d.date()) for d in dates]),
        symbols=np.array(symbols),
    )

    meta = {
        "n_samples": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "date_range": [str(min(dates).date()), str(max(dates).date())],
        "n_symbols": len(set(symbols)),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
