#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.recommender import backtest_from_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OptiNet daily-resolution options backtest.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", nargs="+", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--output-dir", default="results/optinet/backtest")
    args = parser.parse_args()

    result = backtest_from_files(
        PROJECT_ROOT / args.model,
        args.index,
        args.options,
        profile=args.profile,
        output_dir=PROJECT_ROOT / args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
