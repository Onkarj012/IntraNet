"""Build minute-sequence dataset for ResNLS training.

For each (index, trade_date, decision_time) cell, extract the 60 minute bars
ending at the decision minute and compute 5 per-bar features:
  log_return, range_pct, body_ratio, upper_shadow_ratio, rsi_14_norm

Persists as .npz with arrays:
  X    : float32 [n_samples, 60, 5]
  y    : int8    [n_samples, 2]   (long_label, short_label)
  meta : structured array of (index, trade_date, decision_time, ret_1h, ret_eod)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from optinet.intraday_v3 import (
    DECISION_TIMES,
    INDEX_FILES,
    LONG_THRESHOLD_1H,
    SHORT_THRESHOLD_1H,
    _load_minute,
)

SEQ_LEN = 60
N_FEATURES = 5
LABEL_THRESHOLD_LONG = LONG_THRESHOLD_1H
LABEL_THRESHOLD_SHORT = SHORT_THRESHOLD_1H


def _per_bar_features(window: pd.DataFrame) -> np.ndarray:
    """Compute 5 features per minute bar over a 60-bar window. Returns [60, 5] array."""
    o = window["open"].to_numpy(dtype=np.float64)
    h = window["high"].to_numpy(dtype=np.float64)
    l = window["low"].to_numpy(dtype=np.float64)
    c = window["close"].to_numpy(dtype=np.float64)

    log_ret = np.zeros(len(c))
    log_ret[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-9))

    range_pct = (h - l) / np.maximum(c, 1e-9)
    body_ratio = (c - o) / np.maximum(h - l, 1e-9)
    upper_shadow = (h - np.maximum(o, c)) / np.maximum(h - l, 1e-9)

    # RSI 14, then normalize to [-1, 1]
    diff = np.diff(c, prepend=c[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=7).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(14, min_periods=7).mean().to_numpy()
    rs = avg_gain / np.maximum(avg_loss, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.where(np.isnan(rsi), 50, rsi)
    rsi_norm = (rsi - 50) / 50.0  # → [-1, 1]

    feat = np.column_stack([log_ret, range_pct, body_ratio, upper_shadow, rsi_norm])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32)


def build_sequence_dataset(
    index_files: dict[str, str | Path] = INDEX_FILES,
    decision_times: list[str] | None = None,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    decision_times = list(decision_times or DECISION_TIMES)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metas: list[dict] = []

    for index_name, path in index_files.items():
        path = Path(path)
        if not path.exists():
            print(f"[skip] {index_name}: {path} not found")
            continue
        df = _load_minute(path)
        print(f"  {index_name}: {len(df):,} bars; {df['trade_date'].nunique()} days")

        for trade_date, day_df in df.groupby("trade_date"):
            day_df = day_df.reset_index(drop=True)
            close_series = day_df["close"]
            for dt in decision_times:
                idxs = day_df.index[day_df["minute"] == dt].tolist()
                if not idxs:
                    continue
                cut = idxs[0]
                if cut < seq_len:
                    continue
                window = day_df.iloc[cut - seq_len + 1 : cut + 1]
                if len(window) < seq_len:
                    continue
                feat = _per_bar_features(window)
                if feat.shape != (seq_len, N_FEATURES):
                    continue

                # 1H horizon = next 60 minutes
                end_1h = cut + 60
                if end_1h >= len(day_df):
                    end_1h = len(day_df) - 1
                if end_1h <= cut:
                    continue
                decision_close = float(close_series.iloc[cut])
                ret_1h = float(close_series.iloc[end_1h]) / decision_close - 1.0
                ret_eod = float(close_series.iloc[-1]) / decision_close - 1.0
                long_lab = int(ret_1h >= LABEL_THRESHOLD_LONG)
                short_lab = int(ret_1h <= LABEL_THRESHOLD_SHORT)

                Xs.append(feat)
                ys.append([long_lab, short_lab])
                metas.append({
                    "index": index_name,
                    "trade_date": pd.Timestamp(trade_date).normalize(),
                    "decision_time": dt,
                    "ret_1h": ret_1h,
                    "ret_eod": ret_eod,
                })

    if not Xs:
        return np.empty((0, seq_len, N_FEATURES), dtype=np.float32), \
               np.empty((0, 2), dtype=np.int8), \
               pd.DataFrame()

    X = np.stack(Xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int8)
    meta = pd.DataFrame(metas).reset_index(drop=True)
    return X, y, meta


def save_dataset(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_records = meta.to_dict(orient="list")
    np.savez_compressed(
        out,
        X=X, y=y,
        index=np.array(meta_records["index"]),
        trade_date=np.array([str(d.date()) for d in meta_records["trade_date"]]),
        decision_time=np.array(meta_records["decision_time"]),
        ret_1h=np.array(meta_records["ret_1h"], dtype=np.float32),
        ret_eod=np.array(meta_records["ret_eod"], dtype=np.float32),
    )


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data = np.load(path, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    meta = pd.DataFrame({
        "index": data["index"],
        "trade_date": pd.to_datetime(data["trade_date"]),
        "decision_time": data["decision_time"],
        "ret_1h": data["ret_1h"],
        "ret_eod": data["ret_eod"],
    })
    return X, y, meta
