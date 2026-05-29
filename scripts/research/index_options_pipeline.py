#!/usr/bin/env python3
"""OptiNet v2 production pipeline.

Subcommands:
  train         Train 4-LGBM stack on the full parquet lake up to --cutoff.
  recommend     Score latest data and emit ranked option picks.
  daily-update  Download today's bhavcopy, parse, then recommend.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from index_options.data import align_spot_to_chain
from index_options.features import build_training_frame
from index_options.labels import build_labels, merge_features_labels
from index_options.models import OptiNetModelBundle, train_model_stack
from index_options.parquet_loader import load_index_lake, load_options_lake
from index_options.readiness import evaluate_readiness
from index_options.recommender import write_json_report
from index_options.translator import translate_ranked_signals
from index_options.models import score_frame


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ── train ──────────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> int:
    cutoff = args.cutoff or date.today().isoformat()
    print(f"Loading parquet lake up to {cutoff} …")
    option_chain = load_options_lake(end=cutoff)
    index_bars = load_index_lake()

    chain_with_spot = align_spot_to_chain(option_chain, index_bars)
    features = build_training_frame(index_bars, chain_with_spot)
    labels = build_labels(index_bars)
    dataset = merge_features_labels(features, labels)

    print(f"Dataset: {len(dataset):,} rows × {len(dataset.columns)} cols")
    bundle = train_model_stack(dataset, profile=args.profile)

    out = PROJECT_ROOT / args.output
    bundle.save(out)
    print(json.dumps({"model": str(out), "metrics": bundle.metrics}, indent=2))
    return 0


# ── recommend ──────────────────────────────────────────────────────────────────

def cmd_recommend(args: argparse.Namespace) -> int:
    bundle = OptiNetModelBundle.load(PROJECT_ROOT / args.model)

    option_chain = load_options_lake()
    index_bars = load_index_lake()
    chain_with_spot = align_spot_to_chain(option_chain, index_bars)
    features = build_training_frame(index_bars, chain_with_spot)

    latest_date = features["date"].max()
    latest_features = features[features["date"] == latest_date].copy()

    scores = score_frame(bundle, latest_features)
    scores = scores[scores["confidence"] >= args.min_confidence]

    trades = translate_ranked_signals(
        scores, chain_with_spot, latest_features,
        profile=args.profile, top_k=args.top_k,
    )
    readiness = evaluate_readiness(features, chain_with_spot)

    payload = {
        "status": readiness.status,
        "as_of": str(latest_date.date()),
        "profile": args.profile,
        "picks": [t.as_dict() for t in trades],
        "readiness": {"checks": readiness.checks, "warnings": readiness.warnings},
    }

    out = PROJECT_ROOT / args.output.replace("YYYYMMDD", latest_date.strftime("%Y%m%d"))
    write_json_report(payload, out)
    print(json.dumps(payload, indent=2))
    return 0


# ── daily-update ───────────────────────────────────────────────────────────────

def cmd_daily_update(args: argparse.Namespace) -> int:
    today = date.today().isoformat()
    lake_cli = str(PROJECT_ROOT / "scripts" / "optinet_data_lake.py")
    python = str(PROJECT_ROOT / ".venv" / "bin" / "python")

    print(f"Downloading bhavcopy for {today} …")
    subprocess.run([python, lake_cli, "--data-root", "data", "download",
                    "--start", today, "--end", today], check=True)

    print("Parsing …")
    subprocess.run([python, lake_cli, "--data-root", "data", "parse",
                    "--start", today, "--end", today, "--overwrite"], check=True)

    print("Generating recommendations …")
    return cmd_recommend(args)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="OptiNet v2 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train on full parquet lake")
    p_train.add_argument("--profile", default="balanced",
                         choices=["conservative", "balanced", "aggressive"])
    p_train.add_argument("--cutoff", default=None, help="YYYY-MM-DD upper date bound")
    p_train.add_argument("--output", default="models/optinet/optinet_balanced.pkl")
    p_train.set_defaults(func=cmd_train)

    # recommend
    p_rec = sub.add_parser("recommend", help="Score latest data and emit picks")
    p_rec.add_argument("--model", default="models/optinet/optinet_balanced.pkl")
    p_rec.add_argument("--profile", default="balanced",
                       choices=["conservative", "balanced", "aggressive"])
    p_rec.add_argument("--top-k", type=int, default=4)
    p_rec.add_argument("--min-confidence", type=float, default=0.40)
    p_rec.add_argument("--output", default="recommendations/optinet_picks_YYYYMMDD.json")
    p_rec.set_defaults(func=cmd_recommend)

    # daily-update
    p_daily = sub.add_parser("daily-update",
                              help="Download today's bhavcopy, parse, then recommend")
    p_daily.add_argument("--model", default="models/optinet/optinet_balanced.pkl")
    p_daily.add_argument("--profile", default="balanced",
                         choices=["conservative", "balanced", "aggressive"])
    p_daily.add_argument("--top-k", type=int, default=4)
    p_daily.add_argument("--min-confidence", type=float, default=0.40)
    p_daily.add_argument("--output", default="recommendations/optinet_picks_YYYYMMDD.json")
    p_daily.set_defaults(func=cmd_daily_update)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
