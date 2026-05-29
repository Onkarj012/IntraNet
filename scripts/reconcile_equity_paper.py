#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.equity_paper import reconcile_equity_paper_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile an IntradayNet equity paper ledger against minute bars.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--output", default="outputs/paper/equity_paper_ledger_reconciled.csv")
    parser.add_argument("--brokerage-per-trade", type=float, default=40.0)
    args = parser.parse_args()

    frame = reconcile_equity_paper_ledger(
        ledger_path=PROJECT_ROOT / args.ledger,
        data_dir=PROJECT_ROOT / args.data_dir,
        output_path=PROJECT_ROOT / args.output,
        brokerage_per_trade=args.brokerage_per_trade,
    )
    print(f"Wrote {len(frame):,} reconciled equity rows to {PROJECT_ROOT / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
