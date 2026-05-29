"""
Regime Detection for V8 Meta-Ensemble.

Clusters each trading day into one of 5 regimes using:
- VIX level and VIX 5-day change
- Nifty 50 trend strength (ADX or return/vol ratio)
- % stocks above 20-day MA (breadth)
- Nifty correlation with previous day (1 = trending, 0 = choppy)
- Sector dispersion (high = stock-picker's market, low = macro-driven)

Regimes:
1. Strong Trend Up — weight momentum high, macro long
2. Strong Trend Down — weight momentum high, macro short
3. Choppy/Mean-Reverting — weight reversal high, breakout
4. High Vol / Crisis — reduce all positions, wider stops
5. Low Vol / Compression — weight breakout high
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REGIME_LABELS = [
    "strong_trend_up",
    "strong_trend_down",
    "choppy_reverting",
    "high_vol_crisis",
    "low_vol_compression",
]

# Signal model weights per regime (row = regime, col = signal)
DEFAULT_REGIME_WEIGHTS = np.array([
    # momentum  reversal  breakout  sentiment  macro
    [0.40,      0.10,     0.15,     0.15,      0.20],   # strong_trend_up
    [0.40,      0.10,     0.15,     0.15,      0.20],   # strong_trend_down
    [0.10,      0.40,     0.25,     0.15,      0.10],   # choppy_reverting
    [0.10,      0.15,     0.10,     0.20,      0.45],   # high_vol_crisis
    [0.15,      0.10,     0.45,     0.15,      0.15],   # low_vol_compression
])

# Regime scaling factors (multiplied to confidence thresholds)
REGIME_CONFIDENCE_SCALE = np.array([0.95, 0.95, 1.05, 1.20, 0.90])

# Maximum LONG/SHORT per regime
REGIME_MAX_LONG = [5, 1, 3, 2, 4]
REGIME_MAX_SHORT = [1, 5, 3, 2, 2]


@dataclass
class RegimeAssignment:
    """Single day regime classification."""
    date: pd.Timestamp
    regime_id: int
    regime_label: str
    confidence: float  # cluster assignment confidence
    features: dict[str, float]


class RegimeDetector:
    """
    K-means based market regime detector.

    Clusters market conditions into 5 interpretable regimes.
    Uses unsupervised learning on market-level features.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        warmup_years: int = 2,
        seed: int = 42,
    ):
        self.n_regimes = n_regimes
        self.warmup_years = warmup_years
        self.seed = seed

        self.scaler = StandardScaler()
        self.clusterer = KMeans(
            n_clusters=n_regimes,
            random_state=seed,
            n_init=10,
            max_iter=300,
        )
        self._fitted = False
        self._regime_map: dict[int, str] = {}

    def fit(self, market_data: pd.DataFrame) -> RegimeDetector:
        """
        Fit regime clusters on market data.

        Parameters
        ----------
        market_data : pd.DataFrame
            Must contain columns: vix_level, vix_5d_change, nifty_adx,
            breadth_20d, nifty_autocorr, sector_dispersion.
            Index must be dates.
        """
        features = self._extract_features(market_data)
        if len(features) < self.n_regimes * 10:
            raise ValueError(
                f"Need at least {self.n_regimes * 10} data points for clustering, "
                f"got {len(features)}"
            )

        scaled = self.scaler.fit_transform(features.values)
        self.clusterer.fit(scaled)

        self._map_regimes(scaled, market_data)
        self._fitted = True
        return self

    def predict(self, market_data: pd.DataFrame) -> list[RegimeAssignment]:
        """Predict regime for each row in market_data."""
        if not self._fitted:
            raise RuntimeError("RegimeDetector not fitted. Call fit() first.")

        features = self._extract_features(market_data)
        scaled = self.scaler.transform(features.values)

        distances = self.clusterer.transform(scaled)
        labels = self.clusterer.predict(scaled)

        # Confidence = 1 - (distance to assigned cluster) / (sum of all distances)
        confidences = 1.0 - distances[np.arange(len(labels)), labels] / distances.sum(axis=1)

        assignments = []
        for i, (date, label) in enumerate(zip(market_data.index, labels)):
            assignments.append(RegimeAssignment(
                date=pd.Timestamp(date),
                regime_id=int(label),
                regime_label=self._regime_map.get(int(label), "unknown"),
                confidence=float(confidences[i]),
                features={
                    col: float(features.iloc[i][col])
                    for col in features.columns
                },
            ))

        return assignments

    def get_weights(self, regime_id: int) -> np.ndarray:
        """Get signal model weights for a regime."""
        if 0 <= regime_id < len(DEFAULT_REGIME_WEIGHTS):
            return DEFAULT_REGIME_WEIGHTS[regime_id]
        return np.ones(5) / 5  # uniform fallback

    def get_confidence_scale(self, regime_id: int) -> float:
        """Get confidence scale factor for a regime."""
        if 0 <= regime_id < len(REGIME_CONFIDENCE_SCALE):
            return float(REGIME_CONFIDENCE_SCALE[regime_id])
        return 1.0

    def save(self, path: str | Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "n_regimes": self.n_regimes,
                "warmup_years": self.warmup_years,
                "seed": self.seed,
                "scaler": self.scaler,
                "clusterer": self.clusterer,
                "_fitted": self._fitted,
                "_regime_map": self._regime_map,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> RegimeDetector:
        import pickle
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)

        detector = cls(
            n_regimes=data["n_regimes"],
            warmup_years=data["warmup_years"],
            seed=data["seed"],
        )
        detector.scaler = data["scaler"]
        detector.clusterer = data["clusterer"]
        detector._fitted = data["_fitted"]
        detector._regime_map = data["_regime_map"]
        return detector

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regime detection features from market data."""
        required = [
            "vix_level", "vix_5d_change", "nifty_adx",
            "breadth_20d", "nifty_autocorr", "sector_dispersion",
        ]
        available = [c for c in required if c in df.columns]

        if not available:
            raise ValueError(
                f"None of the required regime features found: {required}"
            )

        features = df[available].copy()
        features = features.fillna(features.median())
        return features

    def _map_regimes(self, scaled: np.ndarray, market_data: pd.DataFrame) -> None:
        """Map cluster IDs to interpretable regime labels."""
        features = self._extract_features(market_data)
        cluster_centers = self.clusterer.cluster_centers_

        nifty_adx_idx = features.columns.get_loc("nifty_adx") if "nifty_adx" in features.columns else None
        vix_idx = features.columns.get_loc("vix_level") if "vix_level" in features.columns else None
        breadth_idx = features.columns.get_loc("breadth_20d") if "breadth_20d" in features.columns else None
        autocorr_idx = features.columns.get_loc("nifty_autocorr") if "nifty_autocorr" in features.columns else None

        assignments = {}
        for i in range(self.n_regimes):
            center = cluster_centers[i]

            # Heuristic regime classification
            is_high_vol = vix_idx is not None and center[vix_idx] > 0.5
            is_trending = (nifty_adx_idx is not None and center[nifty_adx_idx] > 0) or \
                          (autocorr_idx is not None and center[autocorr_idx] > 0.5)
            is_bullish = breadth_idx is not None and center[breadth_idx] > 0
            is_compression = (vix_idx is not None and center[vix_idx] < -0.5)

            if is_high_vol and abs(center[vix_idx or 0]) > 1.0:
                assignments[i] = "high_vol_crisis"
            elif is_compression:
                assignments[i] = "low_vol_compression"
            elif is_trending and is_bullish:
                assignments[i] = "strong_trend_up"
            elif is_trending and not is_bullish:
                assignments[i] = "strong_trend_down"
            elif not is_trending:
                assignments[i] = "choppy_reverting"
            else:
                assignments[i] = "choppy_reverting"

        self._regime_map = assignments
