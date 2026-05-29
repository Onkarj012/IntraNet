#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.equity_paper import create_equity_paper_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an IntradayNet equity paper-trading ledger from recommendations JSON.")
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--output", default="outputs/paper/equity_paper_ledger.csv")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.005)
    parser.add_argument("--max-position-pct", type=float, default=0.20)
    args = parser.parse_args()

    frame = create_equity_paper_ledger(
        recommendations_path=PROJECT_ROOT / args.recommendations,
        output_path=PROJECT_ROOT / args.output,
        capital=args.capital,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_position_pct=args.max_position_pct,
    )
    print(f"Wrote {len(frame):,} equity paper-ledger rows to {PROJECT_ROOT / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
