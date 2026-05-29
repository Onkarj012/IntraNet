#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.robustness import json_ready
from optinet.evaluation import evaluate_optinet


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OptiNet with train, walk-forward, blind, and readiness gates.")
    parser.add_argument("--index", nargs="+", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--blind-start", required=True)
    parser.add_argument("--blind-end", required=True)
    parser.add_argument("--output-dir", default="results/optinet/evaluation")
    parser.add_argument("--model-output")
    parser.add_argument("--dataset-output")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    args = parser.parse_args()

    result = evaluate_optinet(
        index_paths=args.index,
        option_paths=args.options,
        profile=args.profile,
        train_start=args.train_start,
        train_end=args.train_end,
        blind_start=args.blind_start,
        blind_end=args.blind_end,
        output_dir=PROJECT_ROOT / args.output_dir,
        model_output=PROJECT_ROOT / args.model_output if args.model_output else None,
        dataset_path=PROJECT_ROOT / args.dataset_output if args.dataset_output else None,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(json_ready(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
