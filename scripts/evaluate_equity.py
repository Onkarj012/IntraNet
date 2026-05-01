#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.robustness import PromotionGateConfig, evaluate_promotion_gates, write_json


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrap an existing equity backtest summary in the shared readiness gate.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary", required=True, help="Existing IntradayNet backtest summary JSON.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", default="results/equity/evaluation")
    args = parser.parse_args()

    source = _load_summary(PROJECT_ROOT / args.summary)
    metrics = {
        "trades": float(source.get("total_trades", source.get("trades", 0.0))),
        "net_pnl": float(source.get("total_net_pnl", source.get("net_pnl", 0.0))),
        "sharpe": float(source.get("sharpe", source.get("sharpe_ratio", 0.0))),
        "stop_exit_rate": float(source.get("stop_exit_rate", 0.0)),
    }
    confidence = {"buckets": [], "has_inversion": False}
    verdict = evaluate_promotion_gates(metrics, confidence, PromotionGateConfig(min_blind_trades=100))
    output = PROJECT_ROOT / args.output_dir
    write_json(output / "readiness.json", {
        "system": "intradaynet",
        "model": str(PROJECT_ROOT / args.model),
        "source_summary": str(PROJECT_ROOT / args.summary),
        "start": args.start,
        "end": args.end,
        "status": verdict.status,
        "reasons": verdict.reasons,
        "metrics": verdict.metrics,
    })
    print(json.dumps({"status": verdict.status, "reasons": verdict.reasons, "metrics": verdict.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
