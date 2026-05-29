#!/usr/bin/env python3
"""Train OptiNet v3 ResNLS on the minute-sequence dataset.

70/15/15 chronological split. AdamW, BCEWithLogits, multi-task long+short heads.
Early-stops on val long-AUC. Saves best model + per-epoch metrics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.resnls import ResNLS, count_parameters
from optinet.sequence_dataset import load_dataset

DATA_PATH = PROJECT_ROOT / "cache/optinet_v3/sequence_dataset.npz"
MODEL_OUT = PROJECT_ROOT / "models/optinet_v3/resnls_best.pt"
METRICS_OUT = PROJECT_ROOT / "results/optinet_v3/resnls_training.json"

EPOCHS = 25
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
SEED = 42


def _split_indices(meta: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split by trade_date so val/test are strictly future of train."""
    days = sorted(meta["trade_date"].unique())
    n = len(days)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_days = set(days[:train_end])
    val_days = set(days[train_end:val_end])
    test_days = set(days[val_end:])
    train_idx = meta.index[meta["trade_date"].isin(train_days)].to_numpy()
    val_idx = meta.index[meta["trade_date"].isin(val_days)].to_numpy()
    test_idx = meta.index[meta["trade_date"].isin(test_days)].to_numpy()
    return train_idx, val_idx, test_idx


def _eval_split(model, loader, device) -> dict:
    from sklearn.metrics import roc_auc_score
    model.eval()
    long_logits, short_logits, long_y, short_y = [], [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            lo, sh = model(X)
            long_logits.append(lo.cpu().numpy())
            short_logits.append(sh.cpu().numpy())
            long_y.append(y[:, 0].numpy())
            short_y.append(y[:, 1].numpy())
    long_logits = np.concatenate(long_logits)
    short_logits = np.concatenate(short_logits)
    long_y = np.concatenate(long_y)
    short_y = np.concatenate(short_y)
    long_p = 1.0 / (1.0 + np.exp(-long_logits))
    short_p = 1.0 / (1.0 + np.exp(-short_logits))
    return {
        "long_auc": float(roc_auc_score(long_y, long_p)) if long_y.sum() and long_y.sum() != len(long_y) else 0.5,
        "short_auc": float(roc_auc_score(short_y, short_p)) if short_y.sum() and short_y.sum() != len(short_y) else 0.5,
        "long_p": long_p,
        "short_p": short_p,
        "long_y": long_y,
        "short_y": short_y,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print(f"\nLoading {DATA_PATH} …")
    X, y, meta = load_dataset(DATA_PATH)
    print(f"X: {X.shape}  y: {y.shape}  meta: {len(meta)} rows")

    train_idx, val_idx, test_idx = _split_indices(meta)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"Train days: {meta.iloc[train_idx]['trade_date'].min().date()} → "
          f"{meta.iloc[train_idx]['trade_date'].max().date()}")
    print(f"Val days  : {meta.iloc[val_idx]['trade_date'].min().date()} → "
          f"{meta.iloc[val_idx]['trade_date'].max().date()}")
    print(f"Test days : {meta.iloc[test_idx]['trade_date'].min().date()} → "
          f"{meta.iloc[test_idx]['trade_date'].max().date()}")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).float()

    def _loader(idx, shuffle):
        ds = TensorDataset(X_t[idx], y_t[idx])
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = _loader(train_idx, shuffle=True)
    val_loader = _loader(val_idx, shuffle=False)
    test_loader = _loader(test_idx, shuffle=False)

    model = ResNLS(n_features=X.shape[2], seq_len=X.shape[1]).to(device)
    print(f"Model: {count_parameters(model):,} params")

    # pos_weight = N_neg / N_pos for each task (handles class imbalance)
    long_pos = float(y[train_idx, 0].sum())
    long_neg = float(len(train_idx) - long_pos)
    short_pos = float(y[train_idx, 1].sum())
    short_neg = float(len(train_idx) - short_pos)
    pos_weight_long = torch.tensor(long_neg / max(long_pos, 1.0), device=device)
    pos_weight_short = torch.tensor(short_neg / max(short_pos, 1.0), device=device)
    loss_long = nn.BCEWithLogitsLoss(pos_weight=pos_weight_long)
    loss_short = nn.BCEWithLogitsLoss(pos_weight=pos_weight_short)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            lo, sh = model(Xb)
            l = loss_long(lo, yb[:, 0]) + loss_short(sh, yb[:, 1])
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(l.item()))
        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val = _eval_split(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_long_auc": val["long_auc"],
            "val_short_auc": val["short_auc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": round(time.time() - t0, 1),
        })
        print(f"epoch {epoch:2d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"val_L_auc={val['long_auc']:.4f}  val_S_auc={val['short_auc']:.4f}  "
              f"({history[-1]['epoch_seconds']}s)")

        avg_val = (val["long_auc"] + val["short_auc"]) / 2
        if avg_val > best_val_auc:
            best_val_auc = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save({
                "model_state": best_state,
                "config": {
                    "n_features": X.shape[2],
                    "seq_len": X.shape[1],
                    "epoch": epoch,
                    "val_long_auc": val["long_auc"],
                    "val_short_auc": val["short_auc"],
                },
            }, MODEL_OUT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
                break

    # Reload best for test eval
    if best_state is not None:
        model.load_state_dict(best_state)
    test = _eval_split(model, test_loader, device)
    print(f"\nTEST  long_auc={test['long_auc']:.4f}  short_auc={test['short_auc']:.4f}")

    # Per (decision_time × index) breakdown on test set
    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    test_meta["long_p"] = test["long_p"]
    test_meta["short_p"] = test["short_p"]
    test_meta["picked_long"] = test_meta["long_p"] >= test_meta["short_p"]
    test_meta["conf"] = np.maximum(test_meta["long_p"], test_meta["short_p"])
    test_meta["signed_ret"] = np.where(test_meta["picked_long"], test_meta["ret_1h"], -test_meta["ret_1h"])

    print("\n=== Threshold sweep on test (signed-direction win rate vs avg ret) ===")
    print(f'{"thr":>6s} {"picks":>7s} {"avg/day":>8s} {"win":>7s} {"avg_ret":>9s} {"L_win":>8s} {"L_count":>8s}')
    n_test_days = test_meta["trade_date"].nunique()
    rows_summary = []
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
        picks = test_meta[test_meta["conf"] >= thr]
        if picks.empty:
            print(f"{thr:>6.2f}  no picks")
            continue
        wins = ((picks["picked_long"] & (picks["ret_1h"] > 0)) |
                (~picks["picked_long"] & (picks["ret_1h"] < 0)))
        longs = picks[picks["picked_long"]]
        shorts = picks[~picks["picked_long"]]
        l_win = ((longs["ret_1h"] > 0).mean()) if len(longs) else 0
        rows_summary.append({
            "threshold": thr,
            "picks": int(len(picks)),
            "avg_per_day": round(len(picks)/n_test_days, 2),
            "win_rate": round(float(wins.mean()), 4),
            "avg_signed_ret_pct": round(float(picks["signed_ret"].mean()*100), 4),
            "long_count": int(len(longs)),
            "long_win_rate": round(float(l_win), 4),
            "long_avg_ret_pct": round(float(longs["signed_ret"].mean()*100), 4) if len(longs) else 0,
            "short_count": int(len(shorts)),
            "short_win_rate": round(float(((shorts["ret_1h"] < 0).mean()) if len(shorts) else 0), 4),
            "short_avg_ret_pct": round(float(shorts["signed_ret"].mean()*100), 4) if len(shorts) else 0,
        })
        print(f"{thr:>6.2f} {len(picks):>7d} {len(picks)/n_test_days:>8.1f} "
              f"{wins.mean():>6.2%} {picks['signed_ret'].mean()*100:>8.4f}% "
              f"{l_win:>7.2%} {len(longs):>8d}")

    print("\n=== Per (decision_time × index) at threshold 0.50 ===")
    cells = test_meta[test_meta["conf"] >= 0.50]
    cell_break = cells.groupby(["decision_time", "index"]).apply(
        lambda g: pd.Series({
            "picks": int(len(g)),
            "win_rate": float(((g["picked_long"] & (g["ret_1h"] > 0))
                                | (~g["picked_long"] & (g["ret_1h"] < 0))).mean()),
            "avg_signed_ret_pct": float(g["signed_ret"].mean()*100),
        }), include_groups=False).reset_index()
    cell_break = cell_break.sort_values("avg_signed_ret_pct", ascending=False)
    print(cell_break.to_string(index=False))

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_OUT.open("w") as f:
        json.dump({
            "model_params": count_parameters(model),
            "best_val_long_auc": max((h["val_long_auc"] for h in history), default=0),
            "best_val_short_auc": max((h["val_short_auc"] for h in history), default=0),
            "test_long_auc": test["long_auc"],
            "test_short_auc": test["short_auc"],
            "history": history,
            "threshold_sweep_test": rows_summary,
            "cell_breakdown_test_thr_0.5": cell_break.to_dict(orient="records"),
            "splits": {
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "test_size": int(len(test_idx)),
                "train_days": int(meta.iloc[train_idx]["trade_date"].nunique()),
                "val_days": int(meta.iloc[val_idx]["trade_date"].nunique()),
                "test_days": int(meta.iloc[test_idx]["trade_date"].nunique()),
            },
        }, f, indent=2, default=str)
    print(f"\nMetrics → {METRICS_OUT}")
    print(f"Best model → {MODEL_OUT}")


if __name__ == "__main__":
    main()
