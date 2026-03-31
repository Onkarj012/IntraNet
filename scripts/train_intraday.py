#!/usr/bin/env python3
"""
IntradayNet Training Script — Multi-Architecture.

Supports TCN+Attention, ResNLS, and CompactCNN architectures.
Uses pre-batched data for fast loading. Rich CLI for progress display.

Usage:
    python scripts/train_intraday.py --model-type resnls --prebatched prebatched/
    python scripts/train_intraday.py --model-type compact_cnn --prebatched prebatched/
    python scripts/train_intraday.py --model-type tcn_attention --prebatched prebatched/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

from intradaynet.config import load_config, IntradayConfig
from intradaynet.models.intraday_loss import IntradayLoss

console = Console()


# ── Pre-batched Dataset (fast) ───────────────────────────────────────────────

class PreBatchedDataset(Dataset):
    """Dataset that loads pre-batched numpy arrays. Near-instant __getitem__."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.per_bar = torch.from_numpy(data["X_per_bar"])
        self.context = torch.from_numpy(data["X_context"])
        self.sentiment = torch.from_numpy(data["X_sentiment"])
        self.direction = torch.from_numpy(data["Y_direction"])
        self.magnitude = torch.from_numpy(data["Y_magnitude"])
        self.time_norm = torch.from_numpy(data["T_time_norm"])

    def __len__(self):
        return len(self.per_bar)

    def __getitem__(self, idx):
        return {
            "per_bar": self.per_bar[idx],
            "context": self.context[idx],
            "sentiment": self.sentiment[idx],
            "targets": {
                "direction": self.direction[idx],
                "magnitude": self.magnitude[idx],
            },
            "time_normalized": self.time_norm[idx],
        }


def collate_prebatched(batch):
    return {
        "per_bar": torch.stack([b["per_bar"] for b in batch]),
        "context": torch.stack([b["context"] for b in batch]),
        "sentiment": torch.stack([b["sentiment"] for b in batch]),
        "targets": {
            "direction": torch.stack([b["targets"]["direction"] for b in batch]),
            "magnitude": torch.stack([b["targets"]["magnitude"] for b in batch]),
        },
        "time_normalized": torch.stack([b["time_normalized"] for b in batch]),
    }


# ── Model builders ──────────────────────────────────────────────────────────

def build_model(model_type: str, cfg: IntradayConfig) -> nn.Module:
    """Build a model by type."""
    if model_type == "resnls":
        from intradaynet.models.resnls_intraday import IntradayResNLS
        return IntradayResNLS(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            hidden_dim=64,
            lstm_layers=2,
            num_horizons=len(cfg.horizons),
            dropout=cfg.model.dropout,
        )
    elif model_type == "compact_cnn":
        from intradaynet.models.compact_cnn import CompactCNN
        return CompactCNN(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            num_horizons=len(cfg.horizons),
            dropout=cfg.model.dropout,
        )
    elif model_type == "tcn_attention":
        from intradaynet.models.tcn_attention import IntradayTCNAttention
        return IntradayTCNAttention(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            hidden_dim=cfg.model.hidden_dim,
            tcn_channels=cfg.model.tcn.channels,
            kernel_size=cfg.model.tcn.kernel_size,
            dilation_base=cfg.model.tcn.dilation_base,
            attn_heads=cfg.model.attn_heads,
            num_horizons=len(cfg.horizons),
            dropout=cfg.model.dropout,
        )
    elif model_type == "lightweight_gru":
        from intradaynet.models.lightweight_gru import LightweightGRU
        return LightweightGRU(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            hidden_dim=48,
            num_horizons=len(cfg.horizons),
            dropout=cfg.model.dropout,
        )
    elif model_type == "mlp_mixer":
        from intradaynet.models.mlp_mixer import IntradayMLPMixer
        return IntradayMLPMixer(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            patch_size=15,
            hidden_dim=64,
            num_mixer_blocks=3,
            num_horizons=len(cfg.horizons),
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip, progress, task_id):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        per_bar = batch["per_bar"].to(device)
        context = batch["context"].to(device)
        sentiment = batch["sentiment"].to(device)
        targets = {
            "direction": batch["targets"]["direction"].to(device),
            "magnitude": batch["targets"]["magnitude"].to(device),
        }
        time_norm = batch["time_normalized"].to(device)

        predictions = model(per_bar, context, sentiment)
        losses = criterion(predictions, targets, time_norm)
        loss = losses["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = per_bar.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        pred_dir = (torch.sigmoid(predictions["direction_logits"]) > 0.5).float()
        total_correct += (pred_dir == targets["direction"]).float().sum().item()

        progress.update(task_id, advance=bs)

    n_horizons = targets["direction"].size(1)
    return total_loss / max(total_samples, 1), total_correct / max(total_samples * n_horizons, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, horizons):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_horizon_correct = None

    for batch in dataloader:
        per_bar = batch["per_bar"].to(device)
        context = batch["context"].to(device)
        sentiment = batch["sentiment"].to(device)
        targets = {
            "direction": batch["targets"]["direction"].to(device),
            "magnitude": batch["targets"]["magnitude"].to(device),
        }
        time_norm = batch["time_normalized"].to(device)

        predictions = model(per_bar, context, sentiment)
        losses = criterion(predictions, targets, time_norm)

        bs = per_bar.size(0)
        total_loss += losses["total_loss"].item() * bs
        total_samples += bs

        pred_dir = (torch.sigmoid(predictions["direction_logits"]) > 0.5).float()
        correct = (pred_dir == targets["direction"]).float()
        total_correct += correct.sum().item()

        if per_horizon_correct is None:
            per_horizon_correct = correct.sum(dim=0)
        else:
            per_horizon_correct += correct.sum(dim=0)

    n_horizons = per_horizon_correct.size(0)
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples * n_horizons, 1)
    per_h = (per_horizon_correct / max(total_samples, 1)).tolist()

    return avg_loss, avg_acc, per_h


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train IntradayNet")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--model-type", type=str, default="resnls",
                        choices=["resnls", "compact_cnn", "tcn_attention",
                                 "lightweight_gru", "mlp_mixer"],
                        help="Model architecture to train")
    parser.add_argument("--prebatched", type=str, default="prebatched",
                        help="Directory with pre-batched .npz files")
    parser.add_argument("--max-epochs", type=int, default=0,
                        help="Override max epochs (0 = use config)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Override batch size (0 = use config)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    epochs = args.max_epochs if args.max_epochs > 0 else cfg.train.epochs
    batch_size = args.batch_size if args.batch_size > 0 else cfg.train.batch_size
    device = torch.device(cfg.device)
    prebatched = Path(args.prebatched)
    out_dir = Path(cfg.out_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Header
    console.print(Panel.fit(
        f"[bold cyan]IntradayNet Trainer — {args.model_type.upper()}[/bold cyan]",
        border_style="cyan",
    ))

    # Load data
    console.print("[dim]Loading pre-batched data...[/dim]")
    t0 = time.time()
    train_ds = PreBatchedDataset(str(prebatched / "train.npz"))
    val_ds = PreBatchedDataset(str(prebatched / "val.npz"))
    console.print(f"  Train: [green]{len(train_ds):,}[/green] samples")
    console.print(f"  Val:   [green]{len(val_ds):,}[/green] samples")
    console.print(f"  Loaded in {time.time() - t0:.1f}s")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_prebatched, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_prebatched, num_workers=0,
    )

    # Build model
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_model(args.model_type, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Model: [green]{total_params:,}[/green] params ({trainable_params:,} trainable)")

    # Loss, optimizer, scheduler
    criterion = IntradayLoss(
        direction_weight=cfg.train.loss.direction_weight,
        magnitude_weight=cfg.train.loss.magnitude_weight,
        confidence_weight=cfg.train.loss.confidence_weight,
        gamma_pos=cfg.train.loss.gamma_pos,
        gamma_neg=cfg.train.loss.gamma_neg,
        downside_weight=cfg.train.loss.downside_weight,
        time_weight_open=cfg.train.loss.time_weight_open,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5,
    )

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    console.print(f"\n[bold]Starting training for {epochs} epochs...[/bold]\n")

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # Train
        with Progress(
            SpinnerColumn(),
            TextColumn(f"Epoch {epoch}/{epochs}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Training", total=len(train_ds))
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                cfg.train.grad_clip, progress, task_id,
            )

        # Validate
        val_loss, val_acc, per_h_acc = validate(
            model, val_loader, criterion, device, cfg.horizons,
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t_epoch
        lr = optimizer.param_groups[0]['lr']

        # Per-horizon accuracy string
        h_acc_parts = [f"H{cfg.horizons[i]}={per_h_acc[i]:.3f}"
                       for i in range(min(len(per_h_acc), len(cfg.horizons)))]

        # Check for improvement
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_type": args.model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": cfg,
            }
            torch.save(checkpoint, out_dir / "best_model.pt")

        star = " [bold green]★[/bold green]" if improved else ""

        console.print(
            f"  Epoch {epoch:3d}/{epochs} │ "
            f"Loss [cyan]{train_loss:.4f}[/cyan]→[cyan]{val_loss:.4f}[/cyan] │ "
            f"Acc [green]{train_acc:.3f}[/green]→[green]{val_acc:.3f}[/green] │ "
            f"{' | '.join(h_acc_parts)} │ "
            f"LR={lr:.0e} │ {elapsed:.0f}s{star}"
        )

        if not improved:
            patience_counter += 1

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc, "per_h_acc": per_h_acc,
        })

        if patience_counter >= cfg.train.early_stopping_patience:
            console.print(f"\n[yellow]Early stopping at epoch {epoch}[/yellow]")
            break

    # Summary
    console.print(f"\n{'═' * 80}")
    console.print(
        f"[bold green]✓ Training complete![/bold green] "
        f"Best val_loss=[cyan]{best_val_loss:.4f}[/cyan], "
        f"acc=[green]{best_val_acc:.3f}[/green]"
    )
    console.print(f"  Model: [dim]{out_dir / 'best_model.pt'}[/dim]\n")


if __name__ == "__main__":
    main()
