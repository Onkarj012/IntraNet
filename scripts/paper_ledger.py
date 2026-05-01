#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.paper import create_optinet_paper_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a paper-trading ledger from the latest model picks.")
    parser.add_argument("--system", choices=["optinet"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", nargs="+", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--readiness")
    parser.add_argument("--output", default="outputs/paper/optinet_paper_ledger.csv")
    args = parser.parse_args()

    frame = create_optinet_paper_ledger(
        model_path=PROJECT_ROOT / args.model,
        index_paths=args.index,
        option_paths=args.options,
        output_path=PROJECT_ROOT / args.output,
        profile=args.profile,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
        readiness_path=PROJECT_ROOT / args.readiness if args.readiness else None,
    )
    print(f"Wrote {len(frame):,} paper-ledger rows to {PROJECT_ROOT / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
