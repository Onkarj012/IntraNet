#!/usr/bin/env python3
"""
IntradayNet Hit Rate Calculator — Prediction Validation.

Runs model inference on test data and compares predictions vs actuals.
Pure accuracy analysis — no trading simulation.

Supports both PyTorch models (.pt) and LightGBM models (directory with .lgb files).

Usage:
    python scripts/hit_rate.py --model runs/intraday/resnls/best_model.pt
    python scripts/hit_rate.py --model runs/lgbm/                           # LightGBM
    python scripts/hit_rate.py --model runs/lgbm/ --prebatched prebatched/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from intradaynet.config import load_config

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Validate model predictions")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt file (PyTorch) or directory with .lgb files (LightGBM)")
    parser.add_argument("--prebatched", type=str, default="prebatched",
                        help="Directory with pre-batched .npz files")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all). Subsample for speed.")
    return parser.parse_args()


def is_lgbm_model(model_path):
    """Check if the model path is a LightGBM model directory."""
    p = Path(model_path)
    if p.is_dir():
        return any(p.glob("*.lgb"))
    return str(model_path).endswith(".lgb")


# ── PyTorch model loading ────────────────────────────────────────────────────

def build_model_from_checkpoint(checkpoint_path, cfg):
    """Load model from checkpoint, auto-detecting model type."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "tcn_attention")

    # Detect sentiment feature count from checkpoint (backward compat)
    state_dict = checkpoint["model_state_dict"]
    ckpt_sent = cfg.model.num_sentiment_features
    for key, val in state_dict.items():
        if "sentiment_proj" in key and "weight" in key and val.dim() == 2:
            ckpt_sent = val.shape[1]
            break

    if model_type == "resnls":
        from intradaynet.models.resnls_intraday import IntradayResNLS
        model = IntradayResNLS(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent,
            hidden_dim=64, lstm_layers=2,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "compact_cnn":
        from intradaynet.models.compact_cnn import CompactCNN
        model = CompactCNN(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "tcn_attention":
        from intradaynet.models.tcn_attention import IntradayTCNAttention
        model = IntradayTCNAttention(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent,
            hidden_dim=cfg.model.hidden_dim,
            tcn_channels=cfg.model.tcn.channels,
            kernel_size=cfg.model.tcn.kernel_size,
            dilation_base=cfg.model.tcn.dilation_base,
            attn_heads=cfg.model.attn_heads,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "lightweight_gru":
        from intradaynet.models.lightweight_gru import LightweightGRU
        model = LightweightGRU(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent,
            hidden_dim=48,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "mlp_mixer":
        from intradaynet.models.mlp_mixer import IntradayMLPMixer
        model = IntradayMLPMixer(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent,
            patch_size=15, hidden_dim=64, num_mixer_blocks=3,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict)
    model.eval()

    if ckpt_sent != cfg.model.num_sentiment_features:
        console.print(f"  [yellow]Note: checkpoint uses {ckpt_sent} sentiment features "
                       f"(config: {cfg.model.num_sentiment_features})[/yellow]")

    return model, model_type, checkpoint.get("epoch", "?"), ckpt_sent


@torch.no_grad()
def compute_predictions_pytorch(model, data, ckpt_sent, batch_size=1024):
    """Run inference on entire dataset in batches."""
    N = len(data["X_per_bar"])
    all_probs = []
    all_mags = []
    all_confs = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        per_bar = torch.from_numpy(data["X_per_bar"][start:end])
        context = torch.from_numpy(data["X_context"][start:end])
        sentiment = torch.from_numpy(data["X_sentiment"][start:end])

        # Truncate sentiment for old checkpoints
        if sentiment.shape[-1] > ckpt_sent:
            sentiment = sentiment[:, :ckpt_sent]

        preds = model(per_bar, context, sentiment)

        probs = torch.sigmoid(preds["direction_logits"]).numpy()
        mags = preds["magnitudes"].numpy()
        confs = preds["confidences"].numpy()

        all_probs.append(probs)
        all_mags.append(mags)
        all_confs.append(confs)

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "magnitudes": np.concatenate(all_mags, axis=0),
        "confidences": np.concatenate(all_confs, axis=0),
    }


# ── LightGBM model loading ──────────────────────────────────────────────────

def flatten_features(X_per_bar, X_context, X_sentiment):
    """Flatten 120-bar window into aggregate features for LightGBM."""
    N, L, F = X_per_bar.shape
    agg_features = []

    for w in [5, 30, min(L, 120)]:
        window = X_per_bar[:, -w:, :]
        agg_features.append(np.nanmean(window, axis=1))
        agg_features.append(np.nanstd(window, axis=1))
        if w > 1:
            slope = (window[:, -1, :] - window[:, 0, :]) / w
            agg_features.append(slope)

    agg_features.append(X_per_bar[:, -1, :])  # last bar
    agg_features.append(np.nanmin(X_per_bar, axis=1))
    agg_features.append(np.nanmax(X_per_bar, axis=1))

    flat = np.concatenate(agg_features + [X_context, X_sentiment], axis=1)
    return np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def load_lgbm_models(model_dir, horizon_names):
    """Load LightGBM booster models from directory."""
    import lightgbm as lgb
    model_dir = Path(model_dir)

    dir_models = {}
    mag_models = {}

    for hname in horizon_names:
        dir_path = model_dir / f"dir_{hname}.lgb"
        mag_path = model_dir / f"mag_{hname}.lgb"

        if dir_path.exists():
            dir_models[hname] = lgb.Booster(model_file=str(dir_path))
        if mag_path.exists():
            mag_models[hname] = lgb.Booster(model_file=str(mag_path))

    return dir_models, mag_models


def compute_predictions_lgbm(dir_models, mag_models, data, horizon_names,
                              max_samples=0, chunk_size=50_000):
    """Run LightGBM inference with chunked flattening."""
    N = len(data["X_per_bar"])
    if max_samples > 0 and N > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(N, max_samples, replace=False)
        indices.sort()
    else:
        indices = np.arange(N)

    n_horizons = len(horizon_names)
    all_probs = np.zeros((len(indices), n_horizons), dtype=np.float32)
    all_mags = np.zeros((len(indices), n_horizons), dtype=np.float32)

    n_chunks = (len(indices) + chunk_size - 1) // chunk_size

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Running LightGBM inference...", total=n_chunks)

        row_offset = 0
        for c in range(n_chunks):
            chunk_idx = indices[c * chunk_size: (c + 1) * chunk_size]
            chunk_len = len(chunk_idx)

            per_bar = np.array(data['X_per_bar'][chunk_idx])
            context = np.array(data['X_context'][chunk_idx])
            sentiment = np.array(data['X_sentiment'][chunk_idx])
            X_flat = flatten_features(per_bar, context, sentiment)
            del per_bar, context, sentiment

            for hi, hname in enumerate(horizon_names):
                if hname in dir_models:
                    all_probs[row_offset:row_offset + chunk_len, hi] = \
                        dir_models[hname].predict(X_flat)
                if hname in mag_models:
                    all_mags[row_offset:row_offset + chunk_len, hi] = \
                        mag_models[hname].predict(X_flat)

            row_offset += chunk_len
            progress.update(task, advance=1)

    # Extract targets for the same indices
    Y_direction = np.array(data["Y_direction"][indices])
    Y_magnitude = np.array(data["Y_magnitude"][indices])
    T_time_norm = np.array(data["T_time_norm"][indices])

    # LightGBM direction classifier outputs probabilities directly
    # Magnitude regressor outputs raw values
    # Confidence: use abs(prob - 0.5) * 2 as proxy
    confs = np.abs(all_probs - 0.5) * 2

    # Extract Gap features (Index 10: gap_size, Index 11: gap_direction in X_context)
    X_context = np.array(data["X_context"][indices])
    gap_size = X_context[:, 10]
    gap_dir = X_context[:, 11]

    return {
        "probs": all_probs,
        "magnitudes": all_mags,
        "confidences": confs,
        "gap_size": gap_size,
        "gap_dir": gap_dir,
    }, Y_direction, Y_magnitude, T_time_norm


# ── Analysis functions ───────────────────────────────────────────────────────

def analyze_horizon(probs, pred_mags, confs, actual_dir, actual_mag, horizon_name):
    """Compute metrics for a single horizon."""
    N = len(probs)

    pred_up = probs > 0.5
    actual_up = actual_dir > 0.5
    hit_rate = np.mean(pred_up == actual_up)

    conf_weighted_correct = np.mean((pred_up == actual_up).astype(float) * confs)

    long_mask = probs >= 0.6
    short_mask = probs <= 0.4
    neutral_mask = ~long_mask & ~short_mask

    long_precision = np.mean(actual_up[long_mask]) if long_mask.sum() > 0 else 0
    short_precision = np.mean(~actual_up[short_mask]) if short_mask.sum() > 0 else 0

    avg_pred_mag = np.mean(np.abs(pred_mags))
    avg_actual_mag = np.mean(np.abs(actual_mag))

    if len(np.unique(actual_mag)) > 1:
        corr = np.corrcoef(pred_mags, actual_mag)[0, 1]
    else:
        corr = 0.0

    long_returns = actual_mag[long_mask]
    short_returns = -actual_mag[short_mask]
    all_signal_returns = np.concatenate([long_returns, short_returns]) if (long_mask.sum() + short_mask.sum()) > 0 else np.array([0])
    gross_profit = np.sum(all_signal_returns[all_signal_returns > 0])
    gross_loss = abs(np.sum(all_signal_returns[all_signal_returns < 0]))
    profit_factor = gross_profit / max(gross_loss, 1e-10)

    return {
        "horizon": horizon_name,
        "samples": N,
        "hit_rate": hit_rate,
        "long_signals": int(long_mask.sum()),
        "short_signals": int(short_mask.sum()),
        "neutral": int(neutral_mask.sum()),
        "long_precision": long_precision,
        "short_precision": short_precision,
        "avg_pred_mag": avg_pred_mag,
        "avg_actual_mag": avg_actual_mag,
        "mag_correlation": corr,
        "profit_factor": profit_factor,
        "avg_confidence": np.mean(confs),
    }


def analyze_by_confidence(probs, confs, actual_dir, actual_mag, horizon_name):
    """Break down metrics by confidence bucket."""
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    results = []

    for lo, hi in buckets:
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            results.append({
                "bucket": f"{lo:.1f}-{hi:.1f}", "count": 0,
                "hit_rate": 0, "avg_return_if_followed": 0,
            })
            continue

        pred_up = probs[mask] > 0.5
        actual_up = actual_dir[mask] > 0.5
        returns = np.where(pred_up, actual_mag[mask], -actual_mag[mask])

        results.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "count": int(mask.sum()),
            "hit_rate": float(np.mean(pred_up == actual_up)),
            "avg_return_if_followed": float(np.mean(returns)),
        })

    return results


def analyze_by_time(probs, confs, actual_dir, time_norm):
    """Break down by time of day (using time_normalized)."""
    periods = [
        ("Morning (9:15-11:00)", 0.0, 0.28),
        ("Midday (11:00-13:00)", 0.28, 0.60),
        ("Afternoon (13:00-15:30)", 0.60, 1.01),
    ]
    results = []
    for name, lo, hi in periods:
        mask = (time_norm >= lo) & (time_norm < hi)
        if mask.sum() == 0:
            results.append({"period": name, "count": 0, "hit_rate": 0})
            continue

        pred_up = probs[mask] > 0.5
        actual_up = actual_dir[mask] > 0.5
        results.append({
            "period": name,
            "count": int(mask.sum()),
            "hit_rate": float(np.mean(pred_up == actual_up)),
        })

    return results

def analyze_by_gap(probs, actual_dir, gap_size, gap_dir):
    """Break down by Gap Up, Gap Down, and Flat."""
    results = []
    
    # Define conditions
    masks = {
        "Heavy Gap Up (>0.5%)": (gap_dir > 0) & (gap_size >= 0.005),
        "Light Gap Up (<0.5%)": (gap_dir > 0) & (gap_size > 0) & (gap_size < 0.005),
        "Heavy Gap Down (>0.5%)": (gap_dir < 0) & (gap_size >= 0.005),
        "Light Gap Down (<0.5%)": (gap_dir < 0) & (gap_size > 0) & (gap_size < 0.005),
        "Flat (No Gap)": (gap_size == 0)
    }

    for name, mask in masks.items():
        if mask.sum() == 0:
            results.append({"category": name, "count": 0, "hit_rate": 0})
            continue

        pred_up = probs[mask] > 0.5
        actual_up = actual_dir[mask] > 0.5
        results.append({
            "category": name,
            "count": int(mask.sum()),
            "hit_rate": float(np.mean(pred_up == actual_up)),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    prebatched = Path(args.prebatched)
    horizon_names = [f"H{h}" for h in cfg.horizons]

    use_lgbm = is_lgbm_model(args.model)

    console.print(Panel.fit(
        f"[bold cyan]IntradayNet — Hit Rate Calculator[/bold cyan]\n"
        f"[dim]Model type: {'LightGBM' if use_lgbm else 'PyTorch'}[/dim]",
        border_style="cyan",
    ))

    # Load data
    console.print(f"[dim]Loading {args.split} data (memory-mapped)...[/dim]")
    data = np.load(prebatched / f"{args.split}.npz", mmap_mode='r')
    N = len(data["Y_direction"])
    console.print(f"  Samples: [green]{N:,}[/green]")

    # Run inference
    t0 = time.time()

    if use_lgbm:
        console.print(f"[dim]Loading LightGBM models from {args.model}...[/dim]")
        dir_models, mag_models = load_lgbm_models(args.model, horizon_names)
        console.print(f"  Direction models: [green]{len(dir_models)}[/green], "
                       f"Magnitude models: [green]{len(mag_models)}[/green]")

        console.print("[dim]Running LightGBM inference (chunked)...[/dim]")
        preds, Y_dir, Y_mag, T_time = compute_predictions_lgbm(
            dir_models, mag_models, data, horizon_names,
            max_samples=args.max_samples,
        )
    else:
        console.print(f"[dim]Loading PyTorch model from {args.model}...[/dim]")
        model, model_type, epoch, ckpt_sent = build_model_from_checkpoint(args.model, cfg)
        console.print(f"  Model: [green]{model_type}[/green] (epoch {epoch})")

        console.print("[dim]Running PyTorch inference...[/dim]")
        preds = compute_predictions_pytorch(model, data, ckpt_sent)
        
        # Extract Gap features (Index 10: gap_size, Index 11: gap_direction)
        X_context = np.array(data["X_context"])
        preds["gap_size"] = X_context[:, 10]
        preds["gap_dir"] = X_context[:, 11]

        Y_dir = np.array(data["Y_direction"])
        Y_mag = np.array(data["Y_magnitude"])
        T_time = np.array(data["T_time_norm"])

    console.print(f"  Inference time: {time.time() - t0:.1f}s\n")

    # ── Per-Horizon Metrics ──
    table = Table(title="Per-Horizon Hit Rate Analysis")
    table.add_column("Horizon", style="cyan")
    table.add_column("Hit Rate", style="green", justify="right")
    table.add_column("Long ↑", style="green", justify="right")
    table.add_column("Short ↓", style="red", justify="right")
    table.add_column("Long Prec", style="green", justify="right")
    table.add_column("Short Prec", style="red", justify="right")
    table.add_column("Mag Corr", style="yellow", justify="right")
    table.add_column("Profit Factor", style="bold", justify="right")

    all_horizon_metrics = []
    n_eval = len(preds["probs"])
    for hi, hname in enumerate(horizon_names):
        metrics = analyze_horizon(
            preds["probs"][:, hi], preds["magnitudes"][:, hi],
            preds["confidences"][:, hi],
            Y_dir[:, hi], Y_mag[:, hi], hname,
        )
        all_horizon_metrics.append(metrics)

        pf_style = "bold green" if metrics["profit_factor"] > 1.0 else "bold red"
        table.add_row(
            hname,
            f"{metrics['hit_rate']:.1%}",
            f"{metrics['long_signals']:,}",
            f"{metrics['short_signals']:,}",
            f"{metrics['long_precision']:.1%}",
            f"{metrics['short_precision']:.1%}",
            f"{metrics['mag_correlation']:.3f}",
            f"[{pf_style}]{metrics['profit_factor']:.2f}[/{pf_style}]",
        )

    console.print(table)

    # ── Confidence Breakdown (best horizon) ──
    best_hi = max(range(len(all_horizon_metrics)),
                  key=lambda i: all_horizon_metrics[i]["hit_rate"])
    best_hname = horizon_names[best_hi]

    conf_table = Table(title=f"Confidence Breakdown — {best_hname}")
    conf_table.add_column("Confidence", style="cyan")
    conf_table.add_column("Count", justify="right")
    conf_table.add_column("Hit Rate", style="green", justify="right")
    conf_table.add_column("Avg Return (if followed)", style="yellow", justify="right")

    conf_buckets = analyze_by_confidence(
        preds["probs"][:, best_hi], preds["confidences"][:, best_hi],
        Y_dir[:, best_hi], Y_mag[:, best_hi], best_hname,
    )

    for bucket in conf_buckets:
        ret_style = "green" if bucket["avg_return_if_followed"] > 0 else "red"
        conf_table.add_row(
            bucket["bucket"],
            f"{bucket['count']:,}",
            f"{bucket['hit_rate']:.1%}" if bucket["count"] > 0 else "N/A",
            f"[{ret_style}]{bucket['avg_return_if_followed']:.4%}[/{ret_style}]" if bucket["count"] > 0 else "N/A",
        )

    console.print()
    console.print(conf_table)

    # ── Time-of-Day Breakdown ──
    time_table = Table(title=f"Time-of-Day Breakdown — {best_hname}")
    time_table.add_column("Period", style="cyan")
    time_table.add_column("Samples", justify="right")
    time_table.add_column("Hit Rate", style="green", justify="right")

    time_buckets = analyze_by_time(
        preds["probs"][:, best_hi], preds["confidences"][:, best_hi],
        Y_dir[:, best_hi], T_time,
    )

    for bucket in time_buckets:
        time_table.add_row(
            bucket["period"],
            f"{bucket['count']:,}",
            f"{bucket['hit_rate']:.1%}" if bucket["count"] > 0 else "N/A",
        )

    console.print()
    console.print(time_table)

    # ── Gap Analysis Breakdown ──
    gap_table = Table(title=f"Gap Accuracy Breakdown — {best_hname}")
    gap_table.add_column("Gap Type", style="cyan")
    gap_table.add_column("Samples", justify="right")
    gap_table.add_column("Hit Rate", style="green", justify="right")

    gap_results = analyze_by_gap(
        preds["probs"][:, best_hi], 
        Y_dir[:, best_hi], 
        preds["gap_size"], 
        preds["gap_dir"]
    )

    for res in gap_results:
        gap_table.add_row(
            res["category"],
            f"{res['count']:,}",
            f"{res['hit_rate']:.1%}" if res["count"] > 0 else "N/A",
        )

    console.print()
    console.print(gap_table)

    # ── Summary ──
    best_m = all_horizon_metrics[best_hi]
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Model: [cyan]{'LightGBM' if use_lgbm else 'PyTorch'}[/cyan]")
    console.print(f"  Evaluated: [green]{n_eval:,}[/green] samples ({args.split} set)")
    console.print(f"  Best horizon: [cyan]{best_hname}[/cyan] — "
                  f"hit rate [green]{best_m['hit_rate']:.1%}[/green], "
                  f"profit factor [{'green' if best_m['profit_factor'] > 1 else 'red'}]"
                  f"{best_m['profit_factor']:.2f}[/{'green' if best_m['profit_factor'] > 1 else 'red'}]")
    console.print(f"  Avg confidence: {best_m['avg_confidence']:.3f}")
    console.print(f"  Magnitude correlation: {best_m['mag_correlation']:.3f}\n")


if __name__ == "__main__":
    main()
