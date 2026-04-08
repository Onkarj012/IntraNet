#!/usr/bin/env python3
"""
Walk-forward analysis for IntradayNet LightGBM.

Tests:
1. Existing models: evaluate on train/val/test splits from prebatched data
2. Train fresh models for each walk-forward window
3. Backtest each window's models on the corresponding test period

Usage:
    python scripts/walkforward_analysis.py
    python scripts/walkforward_analysis.py --quick  # skip retraining, just eval existing
    python scripts/walkforward_analysis.py --windows 2024  # train window for 2024
    python scripts/walkforward_analysis.py --max-stocks 50 --max-train 80000
"""

import argparse
import sys
import time
import json
import gc
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from intradaynet.costs import IndianMarketCosts
from intradaynet.regime import MarketRegime
from intradaynet.sampling import smart_subsample

console = Console()

HORIZONS = ["H15", "H30", "H60"]
FLAT_DIM = 625
SEQ_LENGTH = 120


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Skip retraining, just evaluate existing models on splits")
    parser.add_argument("--windows", type=str, default="",
                        help="Comma-separated windows to train: '2024,Q1_2025' or 'all'")
    parser.add_argument("--prebatched-dir", type=str, default="prebatched_v2")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    parser.add_argument("--output-dir", type=str, default="runs/lgbm_wf")
    parser.add_argument("--max-stocks", type=int, default=100,
                        help="Max stocks per window for training (0=all)")
    parser.add_argument("--max-train", type=int, default=100_000,
                        help="Max training samples per window")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _infer_regime_from_ohlcv(ohlcv: np.ndarray) -> tuple:
    if len(ohlcv) < 60:
        return MarketRegime.CALM_BULL, True

    c, h, l, v = ohlcv[:, 3], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 4]
    log_ret = np.diff(np.concatenate([[c[0]], c]))
    log_ret = np.log(np.maximum(c[1:] / np.maximum(c[:-1], 1e-10), 1e-10))
    log_ret = np.concatenate([[0], log_ret])

    vol_recent = np.std(log_ret[-30:]) if len(log_ret) >= 30 else np.std(log_ret)
    vol_mean = np.std(log_ret[-100:]) if len(log_ret) >= 100 else vol_recent
    vol_ratio = vol_recent / max(vol_mean, 1e-8)
    trend = np.mean(log_ret[-20:]) if len(log_ret) >= 20 else 0.0

    vix_proxy = 15.0 * max(vol_ratio, 0.5)
    vix_change = max(0, (vol_ratio - 1.0)) * 0.5

    if vix_proxy > 28.0 or vix_change > 0.20:
        return MarketRegime.EXTREME, False
    gap = abs(c[-1] - c[0]) / max(c[0], 1) if len(c) > 0 else 0.0
    if gap > 0.015:
        return MarketRegime.EXTREME, False

    is_volatile = vix_proxy > 22.0
    regime = MarketRegime.VOLATILE_BULL if (is_volatile and trend >= 0) else \
             MarketRegime.VOLATILE_BEAR if is_volatile else \
             MarketRegime.CALM_BULL if trend >= 0 else MarketRegime.CALM_BEAR
    return regime, True


def _ewm_numpy(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _rolling_mean(arr, window):
    out = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.mean(arr[i - window + 1:i + 1])
    return out


def _rolling_std(arr, window):
    out = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.std(arr[i - window + 1:i + 1])
    return out


def _rsi_numpy(close, period=14):
    delta = np.diff(np.concatenate([[close[0]], close]))
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    alpha = 1.0 / period
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(close)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_numpy(high, low, close, period=14):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return np.clip(atr / np.where(close == 0, 1, close), 0, 0.5)


def _expanding_max(arr):
    out = np.empty_like(arr, dtype=np.float64)
    cur = arr[0]
    for i in range(len(arr)):
        cur = max(cur, arr[i])
        out[i] = cur
    return out


def _expanding_min(arr):
    out = np.empty_like(arr, dtype=np.float64)
    cur = arr[0]
    for i in range(len(arr)):
        cur = min(cur, arr[i])
        out[i] = cur
    return out


def _obv_slope_numpy(close, volume, window=20):
    sign = np.sign(np.diff(np.concatenate([[close[0]], close])))
    sign[0] = 0
    obv = np.cumsum(sign * volume)
    x = np.arange(len(obv), dtype=np.float64)
    out = np.zeros_like(obv)
    for i in range(window - 1, len(obv)):
        xi = x[i - window + 1:i + 1]
        yi = obv[i - window + 1:i + 1]
        xm, ym = np.mean(xi), np.mean(yi)
        cov = np.mean((xi - xm) * (yi - ym))
        var = np.mean((xi - xm) ** 2)
        slope = cov / max(var, 1e-10)
        out[i] = slope
    slope_rolling_std = _rolling_std(out, 100)
    slope_rolling_std = np.where(slope_rolling_std == 0, 1, slope_rolling_std)
    return np.clip(out / slope_rolling_std, -5, 5)


def compute_per_bar_features(ohlcv: np.ndarray) -> np.ndarray:
    n = len(ohlcv)
    o, h, l, c, v = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
    f = np.full((n, 25), np.nan, dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    log_ret = np.log(np.maximum(c / prev_c, 1e-10))
    log_ret[0] = 0.0
    f[:, 0] = log_ret

    vol_ema20 = _ewm_numpy(v, span=20)
    vol_ema20 = np.where(vol_ema20 == 0, 1, vol_ema20)
    f[:, 1] = np.clip(v / vol_ema20, 0, 50)

    tp = (h + l + c) / 3.0
    tpv = tp * v
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(v)
    cum_vol = np.where(cum_vol == 0, 1, cum_vol)
    vwap = cum_tpv / cum_vol
    f[:, 2] = np.clip((c - vwap) / np.where(vwap == 0, 1, vwap), -5, 5)

    ema_9 = _ewm_numpy(c, span=9)
    ema_20 = _ewm_numpy(c, span=20)
    f[:, 3] = np.clip((c - ema_9) / np.where(ema_9 == 0, 1, ema_9), -5, 5)
    f[:, 4] = np.clip((c - ema_20) / np.where(ema_20 == 0, 1, ema_20), -5, 5)

    f[:, 5] = _rsi_numpy(c, period=14) / 100.0

    bb_mid = _rolling_mean(c, 20)
    bb_std = _rolling_std(c, 20)
    bb_std = np.where(bb_std == 0, 1e-10, bb_std)
    bb_half = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    f[:, 6] = np.clip((c - bb_mid) / np.where(bb_half - bb_mid == 0, 1e-10, bb_half - bb_mid), -5, 5)
    f[:, 7] = np.clip((bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / np.where(bb_mid == 0, 1, bb_mid), 0, 5)

    hl_range = np.where(h - l == 0, 1e-10, h - l)
    f[:, 8] = np.clip(np.abs(c - o) / hl_range, 0, 1)
    f[:, 9] = np.clip((h - np.maximum(o, c)) / hl_range, 0, 1)
    f[:, 10] = np.clip((np.minimum(o, c) - l) / hl_range, 0, 1)

    f[:, 11] = np.clip((h - l) / np.where(c == 0, 1, c), 0, 0.5)

    vol_roll_avg = _rolling_mean(v, 30)
    vol_roll_avg = np.where(vol_roll_avg == 0, 1, vol_roll_avg)
    f[:, 12] = np.clip(v / vol_roll_avg, 0, 50)

    bar_in_sess = np.arange(n, dtype=np.float64)
    f[:, 13] = np.clip(bar_in_sess / max(n, 1), 0, 1)

    orb_high = np.full(n, h[:15].max() if n >= 15 else h.max())
    orb_low = np.full(n, l[:15].min() if n >= 15 else l.min())
    orb_range = orb_high - orb_low
    orb_range = np.where(orb_range == 0, 1e-10, orb_range)
    f[:, 14] = np.clip((c - orb_high) / orb_range, -10, 10)
    f[:, 15] = np.clip((c - orb_low) / orb_range, -10, 10)

    day_open = o[0]
    day_open = day_open if day_open != 0 else 1
    f[:, 16] = np.clip((c - day_open) / day_open, -0.2, 0.2)

    c_shifted_5 = np.roll(c, 5)
    c_shifted_5[:5] = c[0]
    f[:, 17] = np.clip(c / np.where(c_shifted_5 == 0, 1, c_shifted_5) - 1, -0.1, 0.1)
    c_shifted_20 = np.roll(c, 20)
    c_shifted_20[:20] = c[0]
    f[:, 18] = np.clip(c / np.where(c_shifted_20 == 0, 1, c_shifted_20) - 1, -0.2, 0.2)

    close_diff = np.diff(np.concatenate([[c[0]], c]))
    up_vol = np.where(close_diff > 0, v, 0)
    dn_vol = np.where(close_diff < 0, v, 0)
    up_vol_20 = _rolling_mean(up_vol, 20) * 20
    dn_vol_20 = _rolling_mean(dn_vol, 20) * 20
    total_vol_20 = _rolling_mean(v, 20) * 20
    total_vol_20 = np.where(total_vol_20 == 0, 1, total_vol_20)
    f[:, 19] = np.clip((up_vol_20 - dn_vol_20) / total_vol_20, -1, 1)

    f[:, 20] = _atr_numpy(h, l, c, period=14)

    day_high = _expanding_max(h)
    day_low = _expanding_min(l)
    day_range = day_high - day_low
    day_range = np.where(day_range == 0, 1e-10, day_range)
    f[:, 21] = np.clip((c - day_low) / day_range, 0, 1)

    f[:, 22] = np.clip(_rolling_std(log_ret, 20), 0, 0.1)

    f[:, 23] = _obv_slope_numpy(c, v, window=20)

    trade_int = v * (h - l)
    trade_int_mean = _rolling_mean(trade_int, 20)
    trade_int_mean = np.where(trade_int_mean == 0, 1, trade_int_mean)
    f[:, 24] = np.clip(trade_int / trade_int_mean, 0, 50)

    return np.nan_to_num(f, nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)


def flatten_features(per_bar_vals: np.ndarray) -> np.ndarray:
    L, F = per_bar_vals.shape
    w = per_bar_vals.reshape(1, L, F)
    parts = []
    for win in [5, 15, 30, 60, 120]:
        if win > L:
            continue
        window = w[:, -win:, :]
        parts.append(np.nanmean(window, axis=1))
        parts.append(np.nanstd(window, axis=1))
        parts.append(np.nanmin(window, axis=1))
        parts.append(np.nanmax(window, axis=1))
    parts.append(w[:, -1, :])
    parts.append(w[:, 0, :])
    parts.append(w[:, -1, :] - w[:, 0, :])
    if L >= 30:
        parts.append(np.nanmean(w[:, -5:, :], axis=1) - np.nanmean(w[:, -30:, :], axis=1))
    if L >= 60:
        parts.append(np.nanmean(w[:, -15:, :], axis=1) - np.nanmean(w[:, -60:, :], axis=1))
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=5.0, neginf=-5.0).squeeze().astype(np.float32)


class FlattenedDataset:
    def __init__(self, X, targets, valids):
        self.X = X
        self.targets = targets
        self.valids = valids


def extract_from_csv(symbol, csv_path, start_date, end_date, sample_interval=15):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.set_index("datetime")
    df.columns = df.columns.str.lower()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    if len(df) < SEQ_LENGTH + 50:
        return None

    close = df["close"].values
    dates = np.array([d.date() if hasattr(d, "date") else d for d in df.index])
    unique_dates = np.unique(dates)

    all_windows = []
    all_targets = {h: [] for h in HORIZONS}
    all_valid = {h: [] for h in HORIZONS}

    per_bar = compute_per_bar_features(
        np.column_stack([df["open"].values, df["high"].values,
                         df["low"].values, df["close"].values, df["volume"].values])
    )

    min_bars = 150
    for date in unique_dates:
        session_mask = dates == date
        session_indices = np.where(session_mask)[0]
        n_bars = len(session_indices)
        if n_bars < min_bars:
            continue

        sess_start = session_indices[0]
        offsets = np.arange(SEQ_LENGTH, n_bars - 60, sample_interval)
        if len(offsets) == 0:
            continue

        start_indices = sess_start + offsets - SEQ_LENGTH
        end_indices = sess_start + offsets

        for si, ei in zip(start_indices, end_indices):
            all_windows.append(per_bar[si:ei])

        anchor_close = close[sess_start + offsets]

        for h_name, h_bars in [("H15", 15), ("H30", 30), ("H60", 60)]:
            future_idx = sess_start + offsets + h_bars
            session_end = sess_start + n_bars
            future_close = np.where(
                future_idx < session_end,
                close[future_idx],
                close[session_end - 1]
            )
            valid_mask = future_idx < session_end
            raw_ret = np.where(
                valid_mask & (anchor_close > 1e-10),
                (future_close - anchor_close) / anchor_close,
                0.0
            )
            raw_ret = np.clip(raw_ret, -0.05, 0.05)
            cost_adj = np.where(raw_ret > 0, raw_ret - 0.001, raw_ret + 0.001)

            all_targets[h_name].append(raw_ret)
            dir_signal = np.where(cost_adj > 0.003, 1.0, np.where(cost_adj < -0.003, 0.0, np.nan))
            all_valid[h_name].append(dir_signal)

    if not all_windows:
        return None

    all_windows = np.stack(all_windows).astype(np.float32)
    nan_mask = ~np.isnan(all_windows).any(axis=(1, 2))
    if nan_mask.sum() == 0:
        return None

    result = {"X": all_windows[nan_mask]}
    for h_name in HORIZONS:
        t = np.concatenate(all_valid[h_name], axis=0)[nan_mask].astype(np.float32)
        result[f"valid_{h_name}"] = t
    return result


def load_stock_data(symbol, data_dir):
    csv_path = data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.set_index("datetime")
    df.columns = df.columns.str.lower()
    return df


def extract_all_stocks(symbols, data_dir, start_date, end_date, max_stocks, progress=None, task_id=None):
    results = []
    for i, symbol in enumerate(symbols):
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        df = load_stock_data(symbol, data_dir)
        if df is None:
            continue
        data = extract_from_csv(symbol, data_dir / f"{symbol}_minute.csv", start_date, end_date)
        if data is not None:
            results.append(data)
        if max_stocks > 0 and len(results) >= max_stocks:
            break
    return results


def build_arrays(stock_results):
    if not stock_results:
        return None

    all_X = np.concatenate([r["X"] for r in stock_results], axis=0)
    N = len(all_X)

    X_flat = []
    for w in all_X:
        flat = flatten_features(w)
        X_flat.append(flat)
    X_flat = np.stack(X_flat, axis=0).astype(np.float32)

    valids = {}
    for h in HORIZONS:
        v = np.concatenate([r[f"valid_{h}"] for r in stock_results], axis=0).astype(np.float32)
        valids[h] = v

    return FlattenedDataset(X_flat, None, valids)


def train_models(dataset, output_dir, max_train, n_estimators, seed, progress=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    X = dataset.X
    valids = dataset.valids

    lgb_params_base = {
        "n_estimators": n_estimators,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbosity": -1,
        "random_state": seed,
    }
    clf_params = {**lgb_params_base, "objective": "binary", "metric": "binary_logloss"}

    metrics = {}
    models = {}

    t0 = time.time()

    for horizon in HORIZONS:
        y = valids[horizon]
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 100:
            console.print(f"  [yellow]Skipping {horizon}: only {valid_mask.sum()} valid samples[/yellow]")
            continue

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if max_train > 0 and len(X_valid) > max_train:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_valid), max_train, replace=False)
            idx.sort()
            X_tr = X_valid[idx]
            y_tr = y_valid[idx]
        else:
            X_tr = X_valid
            y_tr = y_valid

        split = int(len(X_tr) * 0.85)
        X_train, X_val = X_tr[:split], X_tr[split:]
        y_train, y_val = y_tr[:split], y_tr[split:]

        model = lgb.LGBMClassifier(**clf_params)
        model.fit(X_train, y_train.astype(int),
                  eval_set=[(X_val, y_val.astype(int))],
                  callbacks=[lgb.log_evaluation(0)])

        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, val_pred)
        try:
            auc = roc_auc_score(y_val, val_prob)
        except ValueError:
            auc = 0.5

        model.booster_.save_model(str(output_dir / f"dir_{horizon}.lgb"))
        metrics[horizon] = {"accuracy": float(acc), "auc": float(auc)}
        models[horizon] = model

    elapsed = time.time() - t0
    return models, metrics, elapsed


def evaluate_on_prebatched(prebatched_dir, model_dir, output_dir):
    console.print("\n[bold cyan]Step 1: Evaluating existing models on prebatched splits[/bold cyan]")

    results = {}
    for split_name, npz_name in [("train", "train.npz"), ("val", "val.npz"), ("test", "test.npz")]:
        npz_path = Path(prebatched_dir) / npz_name
        if not npz_path.exists():
            console.print(f"  [yellow]Skipping {split_name}: {npz_path} not found[/yellow]")
            continue

        data = np.load(npz_path, mmap_mode="r")
        X = data["X_flat"][:]
        valids = {h: data[f"valid_{h}"][:] for h in HORIZONS}

        split_results = {}
        for horizon in HORIZONS:
            model_path = Path(model_dir) / f"dir_{horizon}.lgb"
            if not model_path.exists():
                continue
            model = lgb.Booster(model_file=str(model_path))

            y = valids[horizon]
            valid_mask = ~np.isnan(y)
            X_v = X[valid_mask]
            y_v = y[valid_mask]

            pred = model.predict(X_v)
            acc = accuracy_score(y_v.astype(int), (pred > 0.5).astype(int))
            try:
                auc = roc_auc_score(y_v, pred)
            except ValueError:
                auc = 0.5

            n_up = (y_v == 1).sum()
            n_down = (y_v == 0).sum()
            split_results[horizon] = {
                "accuracy": float(acc),
                "auc": float(auc),
                "n_samples": int(valid_mask.sum()),
                "n_up": int(n_up),
                "n_down": int(n_down),
                "class_balance": float(n_up / (n_up + n_down)),
            }

        results[split_name] = split_results
        console.print(f"  {split_name}: {split_results}")

    output_path = Path(output_dir) / "prebatched_eval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [green]✓ Saved to {output_path}[/green]")
    return results


def run_backtest_single(args_tuple):
    (symbol, csv_path, model_dir, horizon, start_str, end_str,
     entry_threshold, stop_loss_pct, take_profit_pct) = args_tuple

    try:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.set_index("datetime")
        df.columns = df.columns.str.lower()
        df = df[(df.index >= start_str) & (df.index <= end_str)]

        if len(df) < SEQ_LENGTH + 50:
            return None

        close_all = df["close"].values
        high_all = df["high"].values
        low_all = df["low"].values
        open_all = df["open"].values
        vol_all = df["volume"].values
        dates = np.array([d.date() if hasattr(d, "date") else d for d in df.index])
        unique_dates = np.unique(dates)

        model_paths = {
            h: Path(model_dir) / f"dir_{h}.lgb" for h in HORIZONS
        }
        models = {h: lgb.Booster(model_file=str(p)) for h, p in model_paths.items() if p.exists()}
        if not models:
            return None

        costs = IndianMarketCosts()
        qty = 100

        total_trades, wins, losses = 0, 0, 0
        total_pnl, total_costs = 0.0, 0.0

        for date in unique_dates:
            session_mask = dates == date
            session_indices = np.where(session_mask)[0]
            n_bars = len(session_indices)
            if n_bars < SEQ_LENGTH + 10:
                continue

            ohlcv = np.column_stack([
                open_all[session_indices],
                high_all[session_indices],
                low_all[session_indices],
                close_all[session_indices],
                vol_all[session_indices],
            ])

            _, should_trade = _infer_regime_from_ohlcv(ohlcv)
            if not should_trade:
                continue

            sess_features = compute_per_bar_features(ohlcv)
            sess_close = close_all[session_indices]

            active = None
            for local_bar in range(SEQ_LENGTH, n_bars - 5):
                if local_bar % 5 != 0 and active is None:
                    continue

                window_feats = sess_features[local_bar - SEQ_LENGTH:local_bar]
                flat = flatten_features(window_feats)

                probs = {h: float(m.predict(flat.reshape(1, -1))[0]) for h, m in models.items()}

                if active is None:
                    dir_ = None
                    best_horizon = None
                    best_prob = entry_threshold

                    for h in HORIZONS:
                        if probs[h] > best_prob:
                            dir_ = "LONG"
                            best_horizon = h
                            best_prob = probs[h]
                        elif probs[h] < (1 - entry_threshold):
                            dir_ = "SHORT"
                            best_horizon = h
                            best_prob = probs[h]
                            break

                    if dir_ is not None:
                        entry = sess_close[local_bar]
                        qty_stock = int(200000 / entry / 100) * 100
                        if qty_stock <= 0:
                            continue
                        active = {
                            "direction": dir_,
                            "entry_bar": local_bar,
                            "entry_price": entry,
                            "qty": qty_stock,
                            "prob": best_prob,
                            "horizon": best_horizon,
                        }

                if active is not None:
                    d = 1 if active["direction"] == "LONG" else -1
                    entry = active["entry_price"]
                    qty_stock = active["qty"]
                    pnl_pct = d * (sess_close[local_bar] - entry) / entry
                    exit_reason = None

                    if pnl_pct <= -stop_loss_pct:
                        exit_reason = "stop_loss"
                        exit_price = entry * (1 - d * stop_loss_pct)
                    elif pnl_pct >= take_profit_pct:
                        exit_reason = "take_profit"
                        exit_price = entry * (1 + d * take_profit_pct)
                    elif local_bar - active["entry_bar"] >= 60:
                        exit_reason = "time_exit"
                        exit_price = sess_close[local_bar]

                    if exit_reason is not None or local_bar == n_bars - 2:
                        if exit_reason is None:
                            exit_reason = "eod"
                            exit_price = sess_close[min(local_bar + 1, n_bars - 1)]

                        gross = d * (exit_price - entry) * qty_stock
                        cost = float(costs.total_cost(entry, qty_stock))
                        net = gross - cost

                        total_trades += 1
                        total_pnl += net
                        total_costs += cost
                        if net > 0:
                            wins += 1
                        else:
                            losses += 1
                        active = None

        return {
            "symbol": symbol,
            "trades": total_trades,
            "wins": wins,
            "losses": losses,
            "pnl": total_pnl,
            "costs": total_costs,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def backtest_window(model_dir, data_dir, symbols, start_str, end_str, horizon="H60",
                    entry_threshold=0.52, stop_loss=0.005, take_profit=0.01,
                    max_workers=16):
    work_items = []
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        work_items.append((
            symbol, str(csv_path), str(model_dir), horizon,
            start_str, end_str, entry_threshold, stop_loss, take_profit
        ))

    results = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                 BarColumn(), TextColumn("{task.completed}/{task.total}"),
                 TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Backtesting...", total=len(work_items))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_backtest_single, w): w[0] for w in work_items}
            for future in as_completed(futures):
                progress.advance(task)
                r = future.result()
                if r is not None:
                    results.append(r)

    if not results:
        return None

    total_trades = sum(r["trades"] for r in results)
    total_pnl = sum(r["pnl"] for r in results)
    total_costs = sum(r["costs"] for r in results)
    wins = sum(r["wins"] for r in results)
    losses = sum(r["losses"] for r in results)

    return {
        "start": start_str,
        "end": end_str,
        "stocks": len(results),
        "total_trades": total_trades,
        "win_rate": wins / max(total_trades, 1),
        "total_pnl": float(total_pnl),
        "total_costs": float(total_costs),
        "per_stock": results,
    }


def main():
    args = parse_args()
    prebatched_dir = Path(args.prebatched_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]Walk-Forward Analysis[/bold cyan]",
        subtitle=f"Output: {output_dir}",
        border_style="cyan",
    ))

    eval_results = evaluate_on_prebatched(prebatched_dir, "runs/lgbm_v2", output_dir)

    if args.quick:
        console.print("\n[yellow]Quick mode: skipping retraining[/yellow]")
        return

    available_windows = {
        "W1_2022_2023": {
            "train": ("2022-01-01", "2023-12-31"),
            "test": ("2025-01-01", "2025-03-31"),
            "label": "Existing models (2022-2023 train → Q1 2025 test)",
        },
        "W2_2024": {
            "train": ("2024-01-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-03-31"),
            "label": "2024 train → Q1 2025 test",
        },
        "W3_2024_H1": {
            "train": ("2024-01-01", "2024-06-30"),
            "test": ("2024-07-01", "2024-12-31"),
            "label": "2024 H1 train → 2024 H2 test",
        },
        "W4_2024_H2": {
            "train": ("2024-07-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-03-31"),
            "label": "2024 H2 train → Q1 2025 test",
        },
    }

    windows_to_run = []
    if not args.windows or args.windows == "all":
        windows_to_run = list(available_windows.keys())
    else:
        for w in args.windows.split(","):
            w = w.strip()
            if w in available_windows:
                windows_to_run.append(w)

    all_results = {"prebatched_eval": eval_results, "windows": {}}

    if not windows_to_run:
        console.print("[yellow]No windows to run. Use --windows to specify.[/yellow]")
        console.print(f"  Available: {list(available_windows.keys())}")
        return

    all_symbols = sorted([
        p.stem.replace("_minute", "") for p in data_dir.glob("*_minute.csv")
    ])
    console.print(f"\n[bold]Available symbols: {len(all_symbols)}[/bold]")

    for window_name in windows_to_run:
        cfg = available_windows[window_name]
        train_start, train_end = cfg["train"]
        test_start, test_end = cfg["test"]

        console.print(f"\n[bold cyan]Window: {window_name}[/bold cyan]")
        console.print(f"  {cfg['label']}")
        console.print(f"  Train: {train_start} → {train_end}")
        console.print(f"  Test:  {test_start} → {test_end}")

        window_dir = output_dir / window_name
        window_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n  [bold]Extracting training data (max {args.max_stocks} stocks)...[/bold]")
        t0 = time.time()

        stock_results = []
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TimeElapsedColumn(), console=console,
        ) as progress:
            task = progress.add_task(f"Loading stocks...", total=len(all_symbols))
            for i, symbol in enumerate(all_symbols):
                progress.update(task, description=f"Loading {symbol}...")
                if args.max_stocks > 0 and len(stock_results) >= args.max_stocks:
                    break
                df = load_stock_data(symbol, data_dir)
                if df is None:
                    progress.advance(task)
                    continue
                data = extract_from_csv(symbol, data_dir / f"{symbol}_minute.csv",
                                        train_start, train_end)
                if data is not None:
                    stock_results.append(data)
                progress.advance(task)

        console.print(f"  Loaded {len(stock_results)} stocks in {time.time()-t0:.1f}s")

        if not stock_results:
            console.print(f"  [red]No data extracted. Skipping window.[/red]")
            continue

        console.print("  Building feature arrays...")
        dataset = build_arrays(stock_results)
        if dataset is None:
            console.print("  [red]Failed to build arrays. Skipping.[/red]")
            continue

        console.print(f"  Dataset: {dataset.X.shape[0]:,} samples × {dataset.X.shape[1]} features")

        console.print(f"  Training models (max_train={args.max_train}, n_est={args.n_estimators})...")
        models, metrics, train_time = train_models(
            dataset, window_dir, args.max_train, args.n_estimators, args.seed
        )
        console.print(f"  Training time: {train_time:.1f}s")
        for h, m in metrics.items():
            console.print(f"    {h}: acc={m['accuracy']:.3f}, auc={m['auc']:.3f}")

        del dataset, stock_results
        gc.collect()

        console.print(f"\n  Running backtest on {test_start} to {test_end}...")
        bt_start = time.time()
        test_symbols = all_symbols[:min(100, len(all_symbols))]
        backtest_result = backtest_window(
            window_dir, data_dir, test_symbols,
            test_start, test_end,
            max_workers=16,
        )
        if backtest_result:
            backtest_result["backtest_time"] = time.time() - bt_start
            console.print(f"  Backtest: {backtest_result['total_trades']} trades, "
                          f"win_rate={backtest_result['win_rate']:.1%}, "
                          f"P&L=₹{backtest_result['total_pnl']:,.0f}")
        else:
            console.print("  [yellow]Backtest returned no results[/yellow]")

        window_summary = {
            "window_name": window_name,
            "label": cfg["label"],
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_time_s": train_time,
            "metrics": metrics,
            "backtest": backtest_result,
        }

        with open(window_dir / "summary.json", "w") as f:
            json.dump(window_summary, f, indent=2)

        all_results["windows"][window_name] = window_summary
        gc.collect()

    console.print("\n" + "="*70)
    console.print("[bold cyan]Walk-Forward Summary[/bold cyan]")
    console.print("="*70)

    table = Table()
    table.add_column("Window", style="cyan")
    table.add_column("Train Period", style="dim")
    table.add_column("Test Period", style="dim")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("P&L", style="yellow", justify="right")
    table.add_column("AUC (H60)", justify="right")

    for wname, wdata in all_results["windows"].items():
        bt = wdata.get("backtest")
        metrics_str = ""
        if "H60" in wdata.get("metrics", {}):
            metrics_str = f"{wdata['metrics']['H60']['auc']:.3f}"

        row = [
            wname,
            f"{wdata['train_start'][:4]}",
            f"{wdata['test_start'][:4]}",
            str(bt["total_trades"]) if bt else "N/A",
            f"{bt['win_rate']:.1%}" if bt else "N/A",
            f"₹{bt['total_pnl']:,.0f}" if bt else "N/A",
            metrics_str,
        ]
        table.add_row(*row)

    console.print(table)

    if all_results["prebatched_eval"]:
        eval_table = Table(title="Existing Models: AUC by Split")
        eval_table.add_column("Split", style="cyan")
        eval_table.add_column("H15 AUC", justify="right")
        eval_table.add_column("H30 AUC", justify="right")
        eval_table.add_column("H60 AUC", justify="right")

        for split, data in all_results["prebatched_eval"].items():
            eval_table.add_row(
                split,
                f"{data.get('H15', {}).get('auc', 'N/A'):.3f}" if isinstance(data.get('H15', {}).get('auc'), float) else "N/A",
                f"{data.get('H30', {}).get('auc', 'N/A'):.3f}" if isinstance(data.get('H30', {}).get('auc'), float) else "N/A",
                f"{data.get('H60', {}).get('auc', 'N/A'):.3f}" if isinstance(data.get('H60', {}).get('auc'), float) else "N/A",
            )
        console.print("\n")
        console.print(eval_table)

    with open(output_dir / "walkforward_results.json", "w") as f:
        json.dump({k: v for k, v in all_results.items() if k != "prebatched_eval"},
                  f, indent=2, default=str)

    console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()
