from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score


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
        if not is_classifier:
            params["objective"] = "regression"
        return cls(**params)
    except Exception:
        if is_classifier:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, random_state=42)
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, random_state=42)


def _predict_positive(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.full(len(X), float(model.classes_[0]))
        return proba[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def train_model_stack(
    frame: pd.DataFrame,
    *,
    profile: str = "balanced",
    validation_fraction: float = 0.2,
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

    long_clf = _make_lgbm(True)
    short_clf = _make_lgbm(True)
    up_reg = _make_lgbm(False)
    down_reg = _make_lgbm(False)

    long_clf.fit(X_train, train_df[long_col].astype(int))
    short_clf.fit(X_train, train_df[short_col].astype(int))
    up_reg.fit(X_train, train_df["up_magnitude"].fillna(0.0))
    down_reg.fit(X_train, train_df["down_magnitude"].fillna(0.0))

    long_prob = _predict_positive(long_clf, X_val)
    short_prob = _predict_positive(short_clf, X_val)
    up_pred = up_reg.predict(X_val)
    down_pred = down_reg.predict(X_val)
    metrics: dict[str, float] = {
        "rows": float(len(train)),
        "train_rows": float(len(train_df)),
        "validation_rows": float(len(val_df)),
        "long_positive_rate": float(train_df[long_col].mean()),
        "short_positive_rate": float(train_df[short_col].mean()),
        "up_mae": float(mean_absolute_error(val_df["up_magnitude"], up_pred)),
        "down_mae": float(mean_absolute_error(val_df["down_magnitude"], down_pred)),
    }
    for name, y, prob in [
        ("long_auc", val_df[long_col].astype(int), long_prob),
        ("short_auc", val_df[short_col].astype(int), short_prob),
    ]:
        try:
            auc = float(roc_auc_score(y, prob))
            metrics[name] = auc if np.isfinite(auc) else 0.5
        except ValueError:
            metrics[name] = 0.5

    return OptiNetModelBundle(profile, cols, long_clf, short_clf, up_reg, down_reg, metrics)


def score_frame(bundle: OptiNetModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.reindex(columns=bundle.feature_columns).fillna(0.0)
    out = frame[["index", "date"]].copy()
    out["long_probability"] = _predict_positive(bundle.long_classifier, X)
    out["short_probability"] = _predict_positive(bundle.short_classifier, X)
    out["up_magnitude"] = np.maximum(bundle.up_regressor.predict(X), 0.0)
    out["down_magnitude"] = np.maximum(bundle.down_regressor.predict(X), 0.0)
    out["direction"] = np.where(out["long_probability"] >= out["short_probability"], "LONG", "SHORT")
    out["confidence"] = np.maximum(out["long_probability"], out["short_probability"])
    out["magnitude"] = np.where(out["direction"] == "LONG", out["up_magnitude"], out["down_magnitude"])
    out["score"] = out["confidence"] * out["magnitude"]
    return out
