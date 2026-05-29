#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.recommender import build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OptiNet training dataset from index and F&O files.")
    parser.add_argument("--index", nargs="+", required=True, help="Index OHLCV CSV/parquet files.")
    parser.add_argument("--options", nargs="+", required=True, help="F&O option chain/bhavcopy CSV/parquet files.")
    parser.add_argument("--output", default="cache/optinet/training_dataset.parquet")
    args = parser.parse_args()

    dataset, _ = build_dataset(args.index, args.options)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".csv"}:
        dataset.to_csv(output, index=False)
    else:
        dataset.to_parquet(output, index=False)
    print(f"Wrote {len(dataset):,} rows and {len(dataset.columns):,} columns to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
