#!/usr/bin/env python3
"""
Feature Pruning — identify and remove dead-weight features after training.
Retrain with pruned feature set for speed + regularization.

Usage:
    python scripts/prune_features.py --model runs/lgbm_v2/dir_H60.lgb --data prebatched_v2/lgbm_dataset.npz
    python scripts/prune_features.py --model runs/lgbm_v2/ --data prebatched_v2/lgbm_dataset.npz --threshold 0.0005
"""

import argparse
import json
import lightgbm as lgb
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.feature_names import FEATURE_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Prune zero/low-importance features")
    parser.add_argument("--model", required=True, help="Path to .lgb file or model directory")
    parser.add_argument("--data", required=True, help="Path to lgbm_dataset.npz")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Drop features below this fraction of total gain")
    parser.add_argument("--output", default="",
                        help="Save selection to JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)

    if model_path.is_dir():
        dir_model_path = model_path / "dir_H60.lgb"
        mag_model_path = model_path / "mag_H60.lgb"
    else:
        dir_model_path = model_path
        mag_model_path = model_path.parent / f"mag_{model_path.stem[4:]}"

    data = np.load(data_path, allow_pickle=True)
    feature_names = FEATURE_NAMES

    print(f"Original features: {len(feature_names)}")

    results = {}
    for model_file, model_type in [(dir_model_path, "direction"), (mag_model_path, "mag_H60")]:
        if not model_file.exists():
            print(f"  Skipping {model_type} model: {model_file} not found")
            continue

        model = lgb.Booster(model_file=str(model_file))
        importance = model.feature_importance(importance_type="gain")
        total_gain = importance.sum()
        thr = total_gain * args.threshold

        keep_mask = importance > thr
        kept = [(n, float(imp)) for n, imp, k in zip(feature_names, importance, keep_mask) if k]
        dropped = [(n, float(imp)) for n, imp, k in zip(feature_names, importance, keep_mask) if not k]

        kept_by_gain = sorted(kept, key=lambda x: x[1], reverse=True)
        dropped_by_gain = sorted(dropped, key=lambda x: x[1])

        print(f"\n--- {model_type.upper()} MODEL ---")
        print(f"Total gain: {total_gain:.2f}")
        print(f"Threshold: {thr:.4f} ({args.threshold * 100:.3f}% of total)")
        print(f"Keeping: {len(kept)} features")
        print(f"Dropping: {len(dropped)} features")

        print(f"\nTop 20 features by gain:")
        for name, imp in kept_by_gain[:20]:
            print(f"  {imp:12.2f}  {name}")

        print(f"\nDropped features ({len(dropped_by_gain)}):")
        for name, imp in dropped_by_gain[:20]:
            print(f"  {imp:8.4f}  {name}")
        if len(dropped_by_gain) > 20:
            print(f"  ... and {len(dropped_by_gain) - 20} more")

        keep_names = [n for n, _ in kept]
        keep_indices = [i for i, k in enumerate(keep_mask) if k]
        results[model_type] = {
            "keep_names": keep_names,
            "keep_indices": keep_indices,
            "n_original": len(feature_names),
            "n_kept": len(keep_names),
            "n_dropped": len(dropped),
        }

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved feature selection to {out_path}")
        print("Use keep_indices to slice X[:, keep_indices] before training/inference.")
    else:
        out_path = model_path.parent / "feature_selection.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved feature selection to {out_path}")


if __name__ == "__main__":
    main()
