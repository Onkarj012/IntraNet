#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.robustness import RiskPolicy, write_json


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _equity_verdict(locked: dict[str, Any] | None, forward: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    if not locked:
        reasons.append("Missing locked equity backtest.")
    if not forward:
        reasons.append("Missing forward equity blind test.")
    locked_net = float((locked or {}).get("total_net_pnl", 0.0))
    forward_net = float((forward or {}).get("total_net_pnl", 0.0))
    forward_sharpe = float((forward or {}).get("sharpe_like_daily", 0.0))
    forward_trades = int((forward or {}).get("total_trades", 0))
    if locked and locked_net <= 0:
        reasons.append("Locked equity backtest is not positive.")
    if forward and forward_net <= 0:
        reasons.append("Forward equity blind P&L is not positive.")
    if forward and forward_sharpe <= 0:
        reasons.append("Forward equity blind Sharpe-like metric is not positive.")
    if forward and forward_trades < 100:
        reasons.append("Forward equity blind trade count is below 100.")
    status = "PAPER_ONLY" if not reasons else "BLOCKED"
    return {
        "system": "intradaynet",
        "status": status,
        "reasons": reasons,
        "metrics": {
            "locked_net_pnl": locked_net,
            "forward_net_pnl": forward_net,
            "forward_sharpe_like_daily": forward_sharpe,
            "forward_trades": forward_trades,
        },
    }


def _optinet_verdict(readiness: dict[str, Any] | None) -> dict[str, Any]:
    if not readiness:
        return {"system": "optinet", "status": "BLOCKED", "reasons": ["Missing OptiNet readiness file."], "metrics": {}}
    return {
        "system": "optinet",
        "status": readiness.get("status", "BLOCKED"),
        "reasons": readiness.get("reasons", []),
        "metrics": readiness.get("metrics", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one trading-system readiness snapshot for equity and options.")
    parser.add_argument(
        "--equity-locked",
        default="results/backtests/backtest_results_nifty500_universe_v7_relaxed/summary_intraday_model_balanced_2025-01-01_2025-12-31.json",
    )
    parser.add_argument(
        "--equity-forward",
        default="results/backtests/backtest_results_q1_2026_v7_relaxed/summary_intraday_model_balanced_2026-01-01_2026-03-31.json",
    )
    parser.add_argument("--optinet-readiness", default="results/optinet/robust_eval_2021_2026/readiness.json")
    parser.add_argument("--output", default="outputs/system/trading_system_status.json")
    args = parser.parse_args()

    equity = _equity_verdict(_load(PROJECT_ROOT / args.equity_locked), _load(PROJECT_ROOT / args.equity_forward))
    optinet = _optinet_verdict(_load(PROJECT_ROOT / args.optinet_readiness))
    overall = "PAPER_ONLY" if equity["status"] == "PAPER_ONLY" else "BLOCKED"
    payload = {
        "overall_status": overall,
        "live_trading_enabled": False,
        "live_trading_reason": "Broker execution is intentionally disabled until paper ledgers pass.",
        "risk_policy": RiskPolicy().__dict__,
        "systems": {
            "equity": equity,
            "options": optinet,
        },
    }
    write_json(PROJECT_ROOT / args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
