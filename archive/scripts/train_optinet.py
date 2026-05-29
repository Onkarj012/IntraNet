#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.models import train_model_stack
from optinet.recommender import build_dataset


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["date"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train OptiNet v1 LightGBM model stack.")
    parser.add_argument("--dataset", help="Prepared dataset CSV/parquet. If omitted, --index and --options are used.")
    parser.add_argument("--index", nargs="+", help="Index OHLCV files.")
    parser.add_argument("--options", nargs="+", help="Option chain/bhavcopy files.")
    parser.add_argument("--profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--output", default="results/models/optinet/optinet_balanced.pkl")
    args = parser.parse_args()

    if args.dataset:
        frame = _read_dataset(PROJECT_ROOT / args.dataset)
    else:
        if not args.index or not args.options:
            parser.error("Provide either --dataset or both --index and --options")
        frame, _ = build_dataset(args.index, args.options)
    bundle = train_model_stack(frame, profile=args.profile)
    output = PROJECT_ROOT / args.output
    bundle.save(output)
    print(json.dumps({"model": str(output), "metrics": bundle.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
