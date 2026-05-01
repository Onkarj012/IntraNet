#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.models import OptiNetModelBundle
from optinet.recommender import build_dataset, recommend_latest, write_json_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate latest OptiNet index option recommendations.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", nargs="+", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--output", default="outputs/optinet/latest_recommendations.json")
    args = parser.parse_args()

    features, option_chain = build_dataset(args.index, args.options)
    bundle = OptiNetModelBundle.load(PROJECT_ROOT / args.model)
    payload = recommend_latest(
        bundle,
        features,
        option_chain,
        profile=args.profile,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
    )
    write_json_report(payload, PROJECT_ROOT / args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
