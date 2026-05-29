from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


LABEL_COLUMNS = {
    "conservative": ("conservative_long_label", "conservative_short_label"),
    "balanced": ("balanced_long_label", "balanced_short_label"),
    "aggressive": ("aggressive_long_label", "aggressive_short_label"),
}

NON_FEATURE_COLUMNS = {
    "index",
    "date",
    "nearest_expiry",
    "next_open",
    "next_close",
    "next_close_return",
    "up_magnitude",
    "down_magnitude",
    "regime_label",
    "regime_block",
    *[col for pair in LABEL_COLUMNS.values() for col in pair],
    "conservative_trade_label",
    "balanced_trade_label",
    "aggressive_trade_label",
}


@dataclass
class OptiNetModelBundle:
    profile: str
    feature_columns: list[str]
    long_classifier: Any
    short_classifier: Any
    up_regressor: Any
    down_regressor: Any
    metrics: dict[str, float]
    long_calibrator: Any = None   # IsotonicRegression fitted on val probabilities
    short_calibrator: Any = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        meta = {
            "profile": self.profile,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "OptiNetModelBundle":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _make_lgbm(is_classifier: bool):
    try:
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if is_classifier else lgb.LGBMRegressor
        params = {
            "n_estimators": 300,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if is_classifier:
            # OptiNet v2.1: handle class imbalance (~12% positive rate for LONG, ~16% for SHORT)
            params["class_weight"] = "balanced"
        else:
            params["objective"] = "regression"
        return cls(**params)
    except Exception:
        if is_classifier:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, random_state=42)
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, random_state=42)


def _make_xgb(is_classifier: bool, scale_pos_weight: float = 1.0):
    """XGBoost factory with class imbalance handling for classifiers."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    cls = xgb.XGBClassifier if is_classifier else xgb.XGBRegressor
    params = {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": 43,            # different seed for ensemble diversity
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }
    if is_classifier:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "auc"
        # XGB doesn't support 'balanced' string — pass numeric scale_pos_weight
        params["scale_pos_weight"] = float(scale_pos_weight)
    else:
        params["objective"] = "reg:squarederror"
    return cls(**params)


def _make_cat(is_classifier: bool, scale_pos_weight: float = 1.0):
    """CatBoost factory with class imbalance handling for classifiers."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        return None
    cls = CatBoostClassifier if is_classifier else CatBoostRegressor
    params = {
        "iterations": 300,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 0.85,
        "random_seed": 44,             # different seed for ensemble diversity
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
    }
    if is_classifier:
        params["loss_function"] = "Logloss"
        params["eval_metric"] = "AUC"
        # auto_class_weights survives sklearn's clone() unlike a class_weights list
        params["auto_class_weights"] = "Balanced"
        # CatBoost's subsample requires bootstrap_type='Bernoulli'
        params["bootstrap_type"] = "Bernoulli"
    else:
        params["loss_function"] = "RMSE"
        params["bootstrap_type"] = "Bernoulli"
    return cls(**params)


class EnsembleClassifier:
    """Average predict_proba across multiple fitted classifiers."""

    def __init__(self, models: list[Any]):
        self.models = [m for m in models if m is not None]
        if not self.models:
            raise ValueError("EnsembleClassifier requires at least one base model")
        # Sklearn-compatible attribute used elsewhere in the code
        self.classes_ = self.models[0].classes_ if hasattr(self.models[0], "classes_") else np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        probs = np.stack([_predict_positive(m, X) for m in self.models], axis=0)
        avg = probs.mean(axis=0)
        # Return shape (n, 2) to match sklearn convention
        return np.column_stack([1.0 - avg, avg])

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class EnsembleRegressor:
    """Average predict across multiple fitted regressors."""

    def __init__(self, models: list[Any]):
        self.models = [m for m in models if m is not None]
        if not self.models:
            raise ValueError("EnsembleRegressor requires at least one base model")

    def predict(self, X) -> np.ndarray:
        preds = np.stack([np.asarray(m.predict(X), dtype=float) for m in self.models], axis=0)
        return preds.mean(axis=0)


def _fit_ensemble_classifier(X, y, scale_pos_weight: float, n_splits: int = 3) -> Any:
    """Fit LGBM + XGBoost + CatBoost classifiers, each calibrated independently."""
    members: list[Any] = []
    for factory in (
        lambda: _make_lgbm(True),
        lambda: _make_xgb(True, scale_pos_weight=scale_pos_weight),
        lambda: _make_cat(True, scale_pos_weight=scale_pos_weight),
    ):
        base = factory()
        if base is None:
            continue
        try:
            calibrated = _wrap_calibrated(base, X, y, n_splits=n_splits)
            members.append(calibrated)
        except Exception as exc:
            print(f"  [warn] ensemble member failed: {type(base).__name__}: {exc}")
    if not members:
        # Fall back to a single LGBM if everything else failed
        base = _make_lgbm(True)
        members.append(_wrap_calibrated(base, X, y, n_splits=n_splits))
    return EnsembleClassifier(members) if len(members) > 1 else members[0]


def _fit_ensemble_regressor(X, y) -> Any:
    """Fit LGBM + XGBoost + CatBoost regressors and average their predictions."""
    members: list[Any] = []
    for factory in (
        lambda: _make_lgbm(False),
        lambda: _make_xgb(False),
        lambda: _make_cat(False),
    ):
        model = factory()
        if model is None:
            continue
        try:
            model.fit(X, y)
            members.append(model)
        except Exception as exc:
            print(f"  [warn] regressor member failed: {type(model).__name__}: {exc}")
    if not members:
        base = _make_lgbm(False)
        base.fit(X, y)
        members.append(base)
    return EnsembleRegressor(members) if len(members) > 1 else members[0]


def _predict_positive(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.full(len(X), float(model.classes_[0]))
        return proba[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def _fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(y_prob, y_true)
    return cal


def _calibrate(calibrator: Any | None, prob: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return prob
    return np.asarray(calibrator.predict(prob), dtype=float)


def _wrap_calibrated(base_estimator, X, y, n_splits: int = 3):
    """Wrap a classifier in Platt-scaled CalibratedClassifierCV using TimeSeriesSplit.

    Falls back to prefit + isotonic on a small validation set when there are too
    few positive samples for cross-validated calibration.
    """
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos < n_splits * 5 or n_neg < n_splits * 5:
        # Insufficient positive class for CV — fit once and skip calibration
        base_estimator.fit(X, y)
        return base_estimator
    cv = TimeSeriesSplit(n_splits=n_splits)
    calibrated = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=cv)
    calibrated.fit(X, y)
    return calibrated


def train_model_stack(
    frame: pd.DataFrame,
    *,
    profile: str = "balanced",
    validation_fraction: float = 0.2,
    use_ensemble: bool = True,
) -> OptiNetModelBundle:
    if profile not in LABEL_COLUMNS:
        raise ValueError(f"Unknown profile: {profile}")
    long_col, short_col = LABEL_COLUMNS[profile]
    train = frame.dropna(subset=[long_col, short_col, "up_magnitude", "down_magnitude"]).sort_values(["date", "index"])
    if len(train) < 30:
        raise ValueError("Need at least 30 labeled rows to train OptiNet")
    cols = feature_columns(train)
    split = max(1, int(len(train) * (1.0 - validation_fraction)))
    split = min(split, len(train) - 1)
    train_df = train.iloc[:split]
    val_df = train.iloc[split:]
    X_train = train_df[cols].fillna(0.0)
    X_val = val_df[cols].fillna(0.0)

    y_long = train_df[long_col].astype(int).to_numpy()
    y_short = train_df[short_col].astype(int).to_numpy()

    if use_ensemble:
        # Compute scale_pos_weight per target = N_neg / N_pos for XGBoost / CatBoost
        spw_long = max(1.0, (y_long == 0).sum() / max((y_long == 1).sum(), 1))
        spw_short = max(1.0, (y_short == 0).sum() / max((y_short == 1).sum(), 1))
        long_clf = _fit_ensemble_classifier(X_train, y_long, scale_pos_weight=spw_long)
        short_clf = _fit_ensemble_classifier(X_train, y_short, scale_pos_weight=spw_short)
        up_reg = _fit_ensemble_regressor(X_train, train_df["up_magnitude"].fillna(0.0))
        down_reg = _fit_ensemble_regressor(X_train, train_df["down_magnitude"].fillna(0.0))
    else:
        long_clf = _wrap_calibrated(_make_lgbm(True), X_train, y_long)
        short_clf = _wrap_calibrated(_make_lgbm(True), X_train, y_short)
        up_reg = _make_lgbm(False); up_reg.fit(X_train, train_df["up_magnitude"].fillna(0.0))
        down_reg = _make_lgbm(False); down_reg.fit(X_train, train_df["down_magnitude"].fillna(0.0))

    long_prob = _predict_positive(long_clf, X_val)
    short_prob = _predict_positive(short_clf, X_val)
    up_pred = up_reg.predict(X_val)
    down_pred = down_reg.predict(X_val)

    long_cal = None
    short_cal = None
    long_prob_cal = long_prob
    short_prob_cal = short_prob

    metrics: dict[str, float] = {
        "rows": float(len(train)),
        "train_rows": float(len(train_df)),
        "validation_rows": float(len(val_df)),
        "long_positive_rate": float(train_df[long_col].mean()),
        "short_positive_rate": float(train_df[short_col].mean()),
        "up_mae": float(mean_absolute_error(val_df["up_magnitude"], up_pred)),
        "down_mae": float(mean_absolute_error(val_df["down_magnitude"], down_pred)),
        "ensemble": bool(use_ensemble),
    }
    for name, y, raw_prob, cal_prob in [
        ("long", val_df[long_col].astype(int).to_numpy(), long_prob, long_prob_cal),
        ("short", val_df[short_col].astype(int).to_numpy(), short_prob, short_prob_cal),
    ]:
        try:
            metrics[f"{name}_auc"] = float(roc_auc_score(y, raw_prob))
        except ValueError:
            metrics[f"{name}_auc"] = 0.5
        try:
            metrics[f"{name}_brier_raw"] = float(brier_score_loss(y, raw_prob))
            metrics[f"{name}_brier_cal"] = float(brier_score_loss(y, cal_prob))
        except Exception:
            pass

    return OptiNetModelBundle(profile, cols, long_clf, short_clf, up_reg, down_reg, metrics, long_cal, short_cal)


def score_frame(bundle: OptiNetModelBundle, frame: pd.DataFrame,
                 *, apply_regime_filter: bool = True) -> pd.DataFrame:
    from index_options.config import PROFILE_SPECS
    cost_buffer = PROFILE_SPECS.get(bundle.profile, PROFILE_SPECS["balanced"]).cost_buffer_pct

    X = frame.reindex(columns=bundle.feature_columns).fillna(0.0)
    out = frame[["index", "date"]].copy()
    raw_long = _predict_positive(bundle.long_classifier, X)
    raw_short = _predict_positive(bundle.short_classifier, X)
    out["long_probability"] = _calibrate(bundle.long_calibrator, raw_long)
    out["short_probability"] = _calibrate(bundle.short_calibrator, raw_short)
    out["up_magnitude"] = np.maximum(bundle.up_regressor.predict(X), 0.0)
    out["down_magnitude"] = np.maximum(bundle.down_regressor.predict(X), 0.0)
    out["direction"] = np.where(out["long_probability"] >= out["short_probability"], "LONG", "SHORT")
    out["confidence"] = np.maximum(out["long_probability"], out["short_probability"])
    out["magnitude"] = np.where(out["direction"] == "LONG", out["up_magnitude"], out["down_magnitude"])
    out["expected_edge"] = out["confidence"] * out["magnitude"]
    out["executable_edge"] = np.maximum(out["expected_edge"] - cost_buffer, 0.0)
    out["score"] = out["executable_edge"]

    # OptiNet v2.1: regime hard filter — skip volatile / crash days
    if apply_regime_filter and "regime_block" in frame.columns:
        block_mask = frame["regime_block"].fillna(False).astype(bool).to_numpy()
        out.loc[block_mask, "direction"] = "NO_TRADE"
        out.loc[block_mask, "confidence"] = 0.0
        out.loc[block_mask, "score"] = 0.0
        out.loc[block_mask, "executable_edge"] = 0.0
    return out
