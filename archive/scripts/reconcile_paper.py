#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.paper import reconcile_optinet_paper_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile an OptiNet paper-trading ledger against actual option-chain bars.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--index", nargs="+", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/paper/optinet_paper_ledger_reconciled.csv")
    args = parser.parse_args()

    frame = reconcile_optinet_paper_ledger(
        ledger_path=PROJECT_ROOT / args.ledger,
        index_paths=args.index,
        option_paths=args.options,
        output_path=PROJECT_ROOT / args.output,
    )
    print(f"Wrote {len(frame):,} reconciled rows to {PROJECT_ROOT / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
