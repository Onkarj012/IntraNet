#!/usr/bin/env python3
"""
Adversarial Validation — detect distribution shift between training and recent data.

High AUC means the model can't tell training vs recent data apart — bad.
If AUC > 0.70, the market regime has shifted and the model may underperform.

Usage:
    python scripts/adversarial_validation.py --train-data prebatched_v2/lgbm_dataset.npz
    python scripts/adversarial_validation.py --train-data prebatched_v2/lgbm_dataset.npz --recent-days 60
"""

import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from intradaynet.feature_names import FEATURE_NAMES


def flatten_windows_vectorized(windows: np.ndarray) -> np.ndarray:
    """
    Match the exact flattening used in prebatch_lgbm_v2.py.
    Same as the function in scripts/prebatch_lgbm_v2.py.
    """
    N, L, F = windows.shape
    parts = []

    for w in [5, 15, 30, 60, 120]:
        if w > L:
            continue
        window = windows[:, -w:, :]
        parts.append(np.nanmean(window, axis=1))
        parts.append(np.nanstd(window, axis=1))
        parts.append(np.nanmin(window, axis=1))
        parts.append(np.nanmax(window, axis=1))

    parts.append(windows[:, -1, :])
    parts.append(windows[:, 0, :])
    diff = windows[:, -1, :] - windows[:, 0, :]
    parts.append(diff)

    if L >= 30:
        parts.append(np.nanmean(windows[:, -5:, :], axis=1) - np.nanmean(windows[:, -30:, :], axis=1))
    if L >= 60:
        parts.append(np.nanmean(windows[:, -15:, :], axis=1) - np.nanmean(windows[:, -60:, :], axis=1))

    flat = np.concatenate(parts, axis=1)
    return np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def extract_recent_features(data_dir: Path, n_days: int, n_stocks: int = 50) -> np.ndarray:
    """
    Extract features for the most recent N days from CSV data.
    Uses the same flattening logic as prebatch_lgbm_v2.py.
    """
    import pandas as pd
    from intradaynet.features.per_bar_features import compute_per_bar_features

    cutoff = datetime.now() - timedelta(days=n_days)
    stock_files = sorted(data_dir.glob("*_minute.csv"))[:n_stocks]

    all_features = []

    for sf in stock_files:
        try:
            symbol = sf.stem.replace("_minute", "")
            df = pd.read_csv(sf)
            df["datetime"] = pd.to_datetime(df["date"])
            df = df.set_index("datetime")
            df.columns = df.columns.str.lower()
            df = df[df.index >= cutoff]

            if len(df) < 200:
                continue

            feats = compute_per_bar_features(df)

            dates = feats.index.date
            unique_dates = sorted(set(dates))
            if not unique_dates:
                continue

            last_date = unique_dates[-1]
            mask = feats.index.date == last_date
            last_feats = feats[mask].iloc[-120:]
            window = last_feats.values.astype(np.float32)

            if window.shape[0] < 120 or np.isnan(window).mean() > 0.3:
                continue

            flat = flatten_windows_vectorized(window[np.newaxis])[0]
            all_features.append(flat)

        except Exception:
            continue

    if not all_features:
        return None
    return np.stack(all_features)


def adversarial_validation(
    X_train: np.ndarray,
    X_recent: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train a classifier to distinguish train vs recent data.
    High AUC = distributions differ = model may not generalize.
    """
    y = np.concatenate([
        np.zeros(len(X_train)),
        np.ones(len(X_recent)),
    ])
    X = np.concatenate([X_train, X_recent])

    max_recent = len(X_recent) * 5
    if len(X_train) > max_recent:
        rng = np.random.RandomState(seed)
        keep_idx = rng.choice(len(X_train), size=max_recent, replace=False)
        X_sub = np.concatenate([X_train[keep_idx], X_recent])
        y_sub = np.concatenate([np.zeros(len(keep_idx)), np.ones(len(X_recent))])
    else:
        X_sub, y_sub = X, y

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )

    scores = cross_val_score(clf, X_sub, y_sub, cv=n_folds, scoring="roc_auc")
    auc = float(scores.mean())
    auc_std = float(scores.std())

    print(f"\nAdversarial Validation AUC: {auc:.4f} ± {auc_std:.4f}")
    if auc > 0.80:
        print("  SEVERE distribution shift — retrain immediately")
        verdict = "SEVERE"
    elif auc > 0.70:
        print("  Moderate shift — monitor closely, consider retraining")
        verdict = "MODERATE"
    elif auc > 0.60:
        print("  Mild shift — normal market evolution")
        verdict = "MILD"
    else:
        print("  Distributions are similar — model should generalize well")
        verdict = "NONE"

    clf.fit(X_sub, y_sub)
    importance = clf.feature_importances_
    top_idx = np.argsort(importance)[-15:]

    print("\nTop 15 features distinguishing train vs recent:")
    top_features = []
    for i in reversed(top_idx):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        print(f"  {name}: {importance[i]}")
        top_features.append({"name": name, "importance": int(importance[i])})

    return {
        "auc": auc,
        "auc_std": auc_std,
        "verdict": verdict,
        "n_train": len(X_train),
        "n_recent": len(X_recent),
        "n_folds": n_folds,
        "top_features": top_features,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial validation for distribution shift")
    parser.add_argument("--train-data", default="prebatched_v2/lgbm_dataset.npz",
                        help="Path to prebatched training data")
    parser.add_argument("--data-dir", default="nifty500",
                        help="Directory with stock minute CSVs")
    parser.add_argument("--recent-days", type=int, default=30,
                        help="Days of recent data to extract")
    parser.add_argument("--n-stocks", type=int, default=100,
                        help="Max stocks to sample for recent data")
    parser.add_argument("--output", default="",
                        help="Save results to JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train_data)
    data_dir = Path(args.data_dir)

    if train_path.is_dir():
        train_path = train_path / "train.npz"

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        return

    print(f"Loading training data from {train_path}...")
    data = np.load(train_path, allow_pickle=True)
    X_train = data["X_flat"]
    print(f"  Train shape: {X_train.shape}")

    print(f"\nExtracting recent features (last {args.recent_days} days, {args.n_stocks} stocks)...")
    X_recent = extract_recent_features(data_dir, args.recent_days, args.n_stocks)
    if X_recent is None or len(X_recent) < 10:
        print("ERROR: Could not extract recent features. Check data availability.")
        return
    print(f"  Recent shape: {X_recent.shape}")

    result = adversarial_validation(X_train, X_recent, n_folds=5)

    print("\n[dim]Note: AUC=1.0 often driven by time_normalized features")
    print("  (uses non-causal transform('count') = total session bars)")
    print("  This is a known feature artifact, not necessarily harmful.[/dim]")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
