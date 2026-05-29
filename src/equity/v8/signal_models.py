"""
Signal Models — 5 Specialist LightGBM Classifiers for V8.

Each model is a calibrated binary classifier predicting the barrier target
for a specific market inefficiency:

| Signal    | Predicts                  | Features Emphasized         | Works In            |
|-----------|--------------------------|-----------------------------|---------------------|
| Momentum  | Trend continuation       | Returns, embeddings, RS     | Trending markets    |
| Reversal  | Mean reversion           | Overbought/oversold, gap, vol| Choppy/sideways    |
| Breakout  | Range expansion          | Vol contraction, vol spikes  | Low-vol regimes     |
| Sentiment | News-driven moves        | Sentiment scores, counts     | High-news days      |
| Macro     | Market-direction bets    | VIX, breadth, global indices | Regime transitions  |

Each model:
1. Trains a LightGBM binary classifier on its specialized feature subset
2. Calibrates probabilities via isotonic regression
3. Outputs a per-stock probability each day
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from .config import SignalModelConfig


# ---------------------------------------------------------------------------
# Feature groups for each specialist
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "momentum": [
        "return_1d", "return_5d", "return_10d", "return_21d", "return_63d",
        "momentum_5d", "momentum_10d", "momentum_21d",
        "rs_vs_sector_5d", "rs_vs_sector_21d",
        "close_vs_sma_21d", "close_vs_sma_63d",
    ],
    "reversal": [
        "price_position_20d", "price_position_63d",
        "gap_size", "overnight_return",
        "rsi_14d",
        "bollinger_position",
        "prev_day_range_pct",
        "parkinson_vol_5d", "gk_vol_5d",
    ],
    "breakout": [
        "vol_contraction_5d", "vol_contraction_21d",
        "volume_dryup_ratio",
        "high_low_range_pct", "atr_percentile",
        "avg_true_range_14d",
        "inside_day_count_5d",
        "close_vs_narrow_range",
    ],
    "sentiment": [
        "sentiment_score_1d", "sentiment_score_3d",
        "sentiment_momentum_5d", "sentiment_volatility_5d",
        "article_count_1d", "article_count_3d",
        "headline_sentiment_bias",
        "news_to_price_ratio",
    ],
    "macro": [
        "vix_level", "vix_trend_5d",
        "nifty_vs_50dma", "nifty_vs_200dma",
        "breadth_pct_above_20dma",
        "sector_return_1d", "sector_return_5d",
        "sp500_overnight", "usdinr_change", "crude_change",
        "crude_oil_return", "gold_return",
        "dxy_change", "asia_sentiment",
        "risk_on_signal",
    ],
    "market_structure": [
        "day_of_week", "month",
        "expiry_week", "budget_day",
        "market_cap_category",
    ],
}


# ---------------------------------------------------------------------------
# Signal Model
# ---------------------------------------------------------------------------

@dataclass
class SignalModel:
    """
    A single specialist signal model (LightGBM + calibration).

    Handles training, inference, calibration, and persistence.
    """

    name: str
    config: SignalModelConfig
    model: Optional[object] = None  # lgb.Booster
    calibrator: Optional[object] = None  # CalibratedClassifierCV
    feature_names: list[str] = field(default_factory=list)
    _trained: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        *,
        sample_weight: np.ndarray | None = None,
        categorical_features: list[str] | None = None,
        verbose: bool = True,
    ) -> SignalModel:
        """
        Train the LightGBM classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features. Only columns in this model's feature groups are used.
        y_train : np.ndarray
            Binary target labels (0/1).
        X_val : pd.DataFrame, optional
            Validation features for early stopping.
        y_val : np.ndarray, optional
            Validation targets.
        sample_weight : np.ndarray, optional
            Sample weights.
        categorical_features : list[str], optional
            Categorical feature names.
        verbose : bool
            Print training progress.
        """
        if lgb is None:
            raise ImportError("lightgbm is required for SignalModel")

        self.feature_names = self._select_features(X_train)

        if not self.feature_names:
            raise ValueError(f"No matching features found for signal '{self.name}'")

        X = X_train[self.feature_names].copy()
        self._check_nan_inf(X)

        cfg = self.config

        scale_pos_weight = cfg.lgb_scale_pos_weight
        if scale_pos_weight is None:
            pos = float(np.sum(y_train == 1))
            neg = float(np.sum(y_train == 0))
            scale_pos_weight = neg / max(pos, 1.0)

        params = {
            "objective": cfg.lgb_objective,
            "metric": cfg.lgb_metric,
            "num_leaves": cfg.lgb_num_leaves,
            "max_bin": cfg.lgb_max_bin,
            "min_child_samples": cfg.lgb_min_child_samples,
            "subsample": cfg.lgb_subsample,
            "colsample_bytree": cfg.lgb_colsample_bytree,
            "reg_alpha": cfg.lgb_reg_alpha,
            "reg_lambda": cfg.lgb_reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "verbosity": -1,
            "seed": 42,
            "n_jobs": -1,
        }
        if not cfg.use_is_unbalance:
            params.pop("is_unbalance", None)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_v = X_val[self.feature_names].copy()
            self._check_nan_inf(X_v)

            evals_result = {}
            self.model = lgb.train(
                params,
                lgb.Dataset(X, label=y_train, weight=sample_weight, categorical_feature=categorical_features or "auto"),
                num_boost_round=cfg.lgb_n_estimators,
                valid_sets=[lgb.Dataset(X_v, label=y_val)],
                valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(cfg.lgb_early_stopping_rounds, verbose=False),
                    lgb.record_evaluation(evals_result),
                ],
            )
        else:
            self.model = lgb.train(
                params,
                lgb.Dataset(X, label=y_train, weight=sample_weight, categorical_feature=categorical_features or "auto"),
                num_boost_round=cfg.lgb_n_estimators,
            )

        self._trained = True

        if verbose:
            boost_rounds = self.model.current_iteration()
            print(f"  [{self.name}] Trained with {boost_rounds} rounds, "
                  f"{len(self.feature_names)} features")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of positive class.

        Returns calibrated probability if calibrator is fitted.
        """
        if not self._trained or self.model is None:
            raise RuntimeError(f"Signal model '{self.name}' not trained")

        features = [f for f in self.feature_names if f in X.columns]
        if not features:
            raise ValueError(f"No matching features for signal '{self.name}'")

        X_sub = X[features].fillna(0).copy()
        raw_prob = self.model.predict(X_sub)

        if self.calibrator is not None:
            try:
                raw_prob = self.calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
            except AttributeError:
                raw_prob = self.calibrator.predict(raw_prob)

        return np.clip(raw_prob, 0.0, 1.0)

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        *,
        method: str = "isotonic",
    ) -> SignalModel:
        """
        Calibrate probabilities using isotonic regression or Platt scaling.

        Parameters
        ----------
        X_cal : pd.DataFrame
            Calibration features.
        y_cal : np.ndarray
            Calibration targets (0/1).
        method : str
            'isotonic' or 'sigmoid'.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        features = [f for f in self.feature_names if f in X_cal.columns]
        X_sub = X_cal[features].fillna(0).copy()

        raw_scores = self.model.predict(X_sub)

        if method == "sigmoid":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw_scores.reshape(-1, 1), y_cal)
        else:
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_scores, y_cal)

        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self._trained or self.model is None:
            return pd.DataFrame(columns=["feature", "importance"])

        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.model.feature_name(),
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def save(self, path: str | Path) -> None:
        """Save model to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "config": self.config,
            "model": self.model,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "_trained": self._trained,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> SignalModel:
        """Load model from pickle."""
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)

        instance = cls(name=state["name"], config=state["config"])
        instance.model = state["model"]
        instance.calibrator = state["calibrator"]
        instance.feature_names = state["feature_names"]
        instance._trained = state["_trained"]
        return instance

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        """Select features from X that belong to this model's domain."""
        relevant = FEATURE_GROUPS.get(self.name, [])
        available = [f for f in relevant if f in X.columns]
        if not available:
            available = list(X.select_dtypes(include=np.number).columns)
        return available

    @staticmethod
    def _check_nan_inf(df: pd.DataFrame) -> None:
        """Check DataFrame for NaN or Inf values."""
        nan_cols = df.columns[df.isna().any()].tolist()
        num_df = df.select_dtypes(include=np.number)
        inf_cols = [c for c in num_df.columns if np.isinf(num_df[c]).any()]
        if nan_cols or inf_cols:
            raise ValueError(
                f"Data contains NaN in columns: {nan_cols}, "
                f"Inf in columns: {inf_cols}"
            )


# ---------------------------------------------------------------------------
# Meta-Ensemble
# ---------------------------------------------------------------------------

@dataclass
class MetaEnsemble:
    """
    Meta-ensemble that combines 5 specialist signal models with
    regime-weighted averaging.

    For each stock on each day:
        final_score = Σ(w_i × P_i(specialist))
    Where w_i are regime-dependent weights.
    """

    models: dict[str, SignalModel]
    regime_weights: np.ndarray  # (n_regimes, n_models)
    model_order: list[str] = field(default_factory=lambda: [
        "momentum", "reversal", "breakout", "sentiment", "macro",
    ])

    def predict(
        self,
        X: pd.DataFrame,
        regime_id: int = 0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Generate ensemble predictions for one regime.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        regime_id : int
            Current market regime ID.

        Returns
        -------
        ensemble_prob : np.ndarray
            Weighted ensemble probability.
        model_probs : dict
            Individual model probabilities.
        """
        model_probs = {}
        weighted_sum = np.zeros(len(X))
        weight_sum = 0.0

        if regime_id < len(self.regime_weights):
            weights = self.regime_weights[regime_id]
        else:
            weights = np.ones(len(self.model_order)) / len(self.model_order)

        for i, name in enumerate(self.model_order):
            if name not in self.models:
                continue
            try:
                prob = self.models[name].predict_proba(X)
                model_probs[name] = prob
                weighted_sum += weights[i] * prob
                weight_sum += weights[i]
            except Exception:
                model_probs[name] = np.full(len(X), np.nan)

        if weight_sum > 0:
            ensemble_prob = weighted_sum / weight_sum
        else:
            ensemble_prob = np.full(len(X), 0.5)

        return ensemble_prob, model_probs

    def predict_multi_regime(
        self,
        X: pd.DataFrame,
        regime_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Generate ensemble predictions with per-sample regime weights.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        regime_ids : np.ndarray
            Regime ID for each sample.

        Returns
        -------
        np.ndarray
            Ensemble probabilities (one per sample).
        """
        probs = np.zeros(len(X))
        unique_regimes = np.unique(regime_ids)

        for regime_id in unique_regimes:
            mask = regime_ids == regime_id
            if not mask.any():
                continue
            ensemble_prob, _ = self.predict(
                X.iloc[mask], regime_id=int(regime_id),
            )
            probs[mask] = ensemble_prob

        return probs

    def save(self, dir_path: str | Path) -> None:
        """Save all models and ensemble metadata."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model.save(dir_path / f"{name}_signal.pkl")

        np.save(dir_path / "regime_weights.npy", self.regime_weights)

    @classmethod
    def load(cls, dir_path: str | Path) -> MetaEnsemble:
        """Load all models and ensemble metadata."""
        dir_path = Path(dir_path)

        models = {}
        for name in ["momentum", "reversal", "breakout", "sentiment", "macro"]:
            model_path = dir_path / f"{name}_signal.pkl"
            if model_path.exists():
                models[name] = SignalModel.load(model_path)

        weights_path = dir_path / "regime_weights.npy"
        if weights_path.exists():
            regime_weights = np.load(weights_path)
        else:
            regime_weights = DEFAULT_REGIME_WEIGHTS

        return cls(models=models, regime_weights=regime_weights)


# Re-export regime weights for convenience
from .regime_detector import DEFAULT_REGIME_WEIGHTS as _DEFAULT_REGIME_WEIGHTS
DEFAULT_REGIME_WEIGHTS = _DEFAULT_REGIME_WEIGHTS
