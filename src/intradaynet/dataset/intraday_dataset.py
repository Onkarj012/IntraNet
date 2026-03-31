"""
PyTorch Dataset for intraday minute-bar data.

Two modes:
  1. **Cached mode** (recommended): loads pre-computed .npz files from
     `features_cache/`. Run `scripts/precompute_features.py` first.
  2. **Live mode**: computes features on-the-fly (slow for many stocks).

Uses a StockGroupedBatchSampler to ensure all samples from one stock
are yielded before moving to the next, avoiding cache thrashing.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from intradaynet.features.per_bar_features import (
    compute_per_bar_features,
    PER_BAR_FEATURE_NAMES,
)
from intradaynet.features.session_features import (
    compute_session_features,
    SESSION_FEATURE_NAMES,
)
from intradaynet.features.sentiment_features import (
    SentimentFeatureBuilder,
    SENTIMENT_FEATURE_NAMES,
)

logger = logging.getLogger("intradaynet.dataset")


class IntradayDataset(Dataset):
    """
    PyTorch Dataset that produces intraday samples.

    Each sample contains:
        - per_bar: (seq_len, 25) per-bar features
        - context: (20,) session-level features
        - sentiment: (14,) sentiment features
        - targets: dict with direction (H,) and magnitude (H,)
        - time_normalized: float, position of sample in session
    """

    def __init__(
        self,
        minute_data_dir: str,
        sentiment_csv: str = "",
        symbols: Optional[List[str]] = None,
        date_start: str = "2015-01-01",
        date_end: str = "2023-12-31",
        sequence_length: int = 120,
        horizons: Optional[List[int]] = None,
        sample_interval: int = 15,
        market_open: str = "09:15",
        market_close: str = "15:30",
        features_cache_dir: str = "features_cache",
    ):
        super().__init__()
        self.minute_data_dir = Path(minute_data_dir)
        self.sequence_length = sequence_length
        self.horizons = horizons or [15, 30, 60, 375]
        self.sample_interval = sample_interval
        self.market_open = market_open
        self.market_close = market_close
        self.date_start = pd.Timestamp(date_start)
        self.date_end = pd.Timestamp(date_end)
        self.features_cache_dir = Path(features_cache_dir)

        # Check if cached features exist
        self.use_cache = self.features_cache_dir.exists() and any(
            self.features_cache_dir.glob("*.npz")
        )
        if self.use_cache:
            logger.info(f"Using pre-computed features from {features_cache_dir}")
        else:
            logger.info("No feature cache found — computing features on-the-fly (slow)")

        # Discover available symbols
        if symbols:
            self.symbols = symbols
        else:
            if self.use_cache:
                self.symbols = sorted([
                    f.stem for f in self.features_cache_dir.glob("*.npz")
                ])
            else:
                self.symbols = sorted([
                    f.stem.replace("_minute", "")
                    for f in self.minute_data_dir.glob("*_minute.csv")
                ])

        logger.info(f"Found {len(self.symbols)} symbols")

        # Sentiment builder (only needed for live mode)
        self.sentiment_builder = (
            SentimentFeatureBuilder(sentiment_csv)
            if sentiment_csv and not self.use_cache
            else None
        )

        # Cache for loaded stock data
        self._cache_symbol: Optional[str] = None
        self._cache_per_bar: Optional[np.ndarray] = None      # (N, 25)
        self._cache_close: Optional[np.ndarray] = None         # (N,)
        self._cache_dates: Optional[np.ndarray] = None         # (N,) str
        self._cache_session_feat: Optional[Dict] = None        # date_str → (20,)
        self._cache_sentiment_feat: Optional[Dict] = None      # date_str → (14,)
        self._cache_session_slices: Optional[Dict] = None      # date_str → (start, end)

        # Build index
        self._index: List[Tuple[int, str, int]] = []
        self._build_index()

    def _build_index(self):
        """Build a flat index of valid sample positions."""
        logger.info("Building dataset index...")

        non_eod_horizons = [h for h in self.horizons if h < 375]
        max_non_eod = max(non_eod_horizons) if non_eod_horizons else 60
        effective_forward = max(max_non_eod, 60)

        for sym_idx, symbol in enumerate(self.symbols):
            try:
                session_bar_counts = self._get_session_bar_counts(symbol)
            except Exception as e:
                logger.debug(f"Skipping {symbol}: {e}")
                continue

            for session_date, n_bars in session_bar_counts.items():
                if n_bars < self.sequence_length + effective_forward:
                    continue

                start = self.sequence_length
                end = n_bars - effective_forward

                if start >= end:
                    continue

                for bar_offset in range(start, end, self.sample_interval):
                    self._index.append((sym_idx, session_date, bar_offset))

        logger.info(f"Built index with {len(self._index)} samples from {len(self.symbols)} stocks")

    def _get_session_bar_counts(self, symbol: str) -> Dict[str, int]:
        """Get bar counts per session for a symbol (fast, no feature computation)."""
        if self.use_cache:
            npz_path = self.features_cache_dir / f"{symbol}.npz"
            if not npz_path.exists():
                return {}
            data = np.load(npz_path, allow_pickle=True)
            dates = data["per_bar_dates"]
            # Filter by date range using string comparison (dates are 'YYYY-MM-DD')
            date_start_str = str(self.date_start.date())
            date_end_str = str(self.date_end.date())
            mask = (dates >= date_start_str) & (dates <= date_end_str)
            filtered_dates = dates[mask]
            unique, counts = np.unique(filtered_dates, return_counts=True)
            return dict(zip(unique, counts))
        else:
            csv_path = self.minute_data_dir / f"{symbol}_minute.csv"
            if not csv_path.exists():
                return {}
            df = pd.read_csv(csv_path, usecols=["date"], parse_dates=["date"])
            df = df[(df["date"] >= self.date_start) & (df["date"] <= self.date_end)]
            df["time"] = df["date"].dt.strftime("%H:%M")
            df = df[(df["time"] >= self.market_open) & (df["time"] < self.market_close)]
            df["session_date"] = df["date"].dt.date.astype(str)
            return df.groupby("session_date").size().to_dict()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sym_idx, session_date, bar_offset = self._index[idx]
        symbol = self.symbols[sym_idx]

        self._ensure_loaded(symbol)

        # Get session slice
        session_slice = self._cache_session_slices.get(session_date)
        if session_slice is None:
            return self._empty_sample()

        sess_start, sess_end = session_slice
        n_bars = sess_end - sess_start

        if n_bars < bar_offset + 1:
            return self._empty_sample()

        # Global indices for this session
        global_start = sess_start + bar_offset - self.sequence_length
        global_end = sess_start + bar_offset

        # Per-bar features (seq_len, 25)
        per_bar_slice = self._cache_per_bar[global_start:global_end]
        if len(per_bar_slice) < self.sequence_length:
            return self._empty_sample()

        per_bar_tensor = torch.from_numpy(per_bar_slice.copy())

        # Context features (20,)
        context = self._cache_session_feat.get(session_date)
        if context is not None:
            context_tensor = torch.from_numpy(context.copy())
        else:
            context_tensor = torch.zeros(len(SESSION_FEATURE_NAMES), dtype=torch.float32)

        # Sentiment features (14,)
        sentiment = self._cache_sentiment_feat.get(session_date)
        if sentiment is not None:
            sentiment_tensor = torch.from_numpy(sentiment.copy())
        else:
            sentiment_tensor = torch.zeros(len(SENTIMENT_FEATURE_NAMES), dtype=torch.float32)

        # Targets
        anchor_close = self._cache_close[sess_start + bar_offset]
        directions = []
        magnitudes = []

        for h in self.horizons:
            target_global = sess_start + bar_offset + h
            if h >= 375 or target_global >= sess_end:
                target_close = self._cache_close[sess_end - 1]
            else:
                target_close = self._cache_close[target_global]

            ret = float((target_close - anchor_close) / max(anchor_close, 1e-10))
            directions.append(1.0 if ret > 0 else 0.0)
            magnitudes.append(ret)

        time_norm = bar_offset / max(n_bars, 1)

        return {
            "per_bar": per_bar_tensor,
            "context": context_tensor,
            "sentiment": sentiment_tensor,
            "targets": {
                "direction": torch.tensor(directions, dtype=torch.float32),
                "magnitude": torch.tensor(magnitudes, dtype=torch.float32),
            },
            "time_normalized": torch.tensor(time_norm, dtype=torch.float32),
        }

    def _ensure_loaded(self, symbol: str):
        """Load stock data into cache (from .npz or raw CSV)."""
        if self._cache_symbol == symbol:
            return

        if self.use_cache:
            self._load_from_cache(symbol)
        else:
            self._load_from_csv(symbol)

        self._cache_symbol = symbol

    def _load_from_cache(self, symbol: str):
        """Load pre-computed features from .npz file (~instant)."""
        npz_path = self.features_cache_dir / f"{symbol}.npz"
        data = np.load(npz_path, allow_pickle=True)

        per_bar = data["per_bar_features"]           # (N, 25) float32
        close = data["close"]                         # (N,) float32
        dates = data["per_bar_dates"]                 # (N,) str
        session_feats = data["session_features"]      # (S, 20) float32
        session_dates = data["session_dates"]          # (S,) str
        sentiment_feats = data["sentiment_features"]   # (S, 14) float32

        # Filter to date range using string comparison
        date_start_str = str(self.date_start.date())
        date_end_str = str(self.date_end.date())
        mask = (dates >= date_start_str) & (dates <= date_end_str)
        per_bar = per_bar[mask]
        close = close[mask]
        dates = dates[mask]

        self._cache_per_bar = per_bar
        self._cache_close = close
        self._cache_dates = dates

        # Build session slices
        self._cache_session_slices = {}
        if len(dates) > 0:
            unique_dates = np.unique(dates)
            for d in unique_dates:
                indices = np.where(dates == d)[0]
                self._cache_session_slices[d] = (indices[0], indices[-1] + 1)

        # Session features dict
        self._cache_session_feat = {}
        for i, d in enumerate(session_dates):
            if d in self._cache_session_slices:
                self._cache_session_feat[d] = session_feats[i]

        # Sentiment features dict
        self._cache_sentiment_feat = {}
        for i, d in enumerate(session_dates):
            if d in self._cache_session_slices:
                self._cache_sentiment_feat[d] = sentiment_feats[i]

    def _load_from_csv(self, symbol: str):
        """Load and compute features from raw CSV (slow, fallback)."""
        csv_path = self.minute_data_dir / f"{symbol}_minute.csv"
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = df.columns.str.lower()
        df = df[(df.index >= self.date_start) & (df.index <= self.date_end)]
        df["_time"] = df.index.strftime("%H:%M")
        df = df[(df["_time"] >= self.market_open) & (df["_time"] < self.market_close)]
        df = df.drop(columns=["_time"])

        features = compute_per_bar_features(df)
        session = compute_session_features(df)

        self._cache_per_bar = features[PER_BAR_FEATURE_NAMES].values.astype(np.float32)
        self._cache_close = df["close"].values.astype(np.float32)

        dates_str = np.array([str(d) for d in df.index.date])
        self._cache_dates = dates_str

        # Build session slices
        self._cache_session_slices = {}
        unique_dates = np.unique(dates_str)
        for d in unique_dates:
            indices = np.where(dates_str == d)[0]
            self._cache_session_slices[d] = (indices[0], indices[-1] + 1)

        # Session features
        self._cache_session_feat = {}
        for i, d in enumerate(session.index):
            ds = str(d.date()) if hasattr(d, 'date') else str(d)
            if ds in self._cache_session_slices:
                self._cache_session_feat[ds] = session.iloc[i].values.astype(np.float32)

        # Sentiment (optional)
        self._cache_sentiment_feat = {}
        if self.sentiment_builder:
            sent = self.sentiment_builder.get_features(symbol, session.index)
            for i, d in enumerate(sent.index):
                ds = str(d.date()) if hasattr(d, 'date') else str(d)
                if ds in self._cache_session_slices:
                    self._cache_sentiment_feat[ds] = sent.iloc[i].values.astype(np.float32)

        logger.debug(f"Loaded {symbol}: {len(df)} bars, {len(self._cache_session_slices)} sessions")

    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return a zero-filled sample (fallback for edge cases)."""
        n_horizons = len(self.horizons)
        return {
            "per_bar": torch.zeros(self.sequence_length, len(PER_BAR_FEATURE_NAMES)),
            "context": torch.zeros(len(SESSION_FEATURE_NAMES)),
            "sentiment": torch.zeros(len(SENTIMENT_FEATURE_NAMES)),
            "targets": {
                "direction": torch.zeros(n_horizons),
                "magnitude": torch.zeros(n_horizons),
            },
            "time_normalized": torch.tensor(0.0),
        }


def collate_intraday(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for IntradayDataset batches."""
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


class StockGroupedBatchSampler:
    """
    Batch sampler that groups samples by stock to minimize cache thrashing.

    All samples from one stock are yielded (in shuffled order) before
    moving to the next stock. Stock order is shuffled each epoch.
    """

    def __init__(self, dataset: IntradayDataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group sample indices by stock
        self._groups: Dict[int, List[int]] = {}
        for idx, (sym_idx, _, _) in enumerate(dataset._index):
            self._groups.setdefault(sym_idx, []).append(idx)

        self._stock_ids = list(self._groups.keys())

    def __iter__(self):
        stock_order = list(self._stock_ids)
        if self.shuffle:
            np.random.shuffle(stock_order)

        for stock_id in stock_order:
            indices = list(self._groups[stock_id])
            if self.shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        total = sum(len(v) for v in self._groups.values())
        return (total + self.batch_size - 1) // self.batch_size
