#!/usr/bin/env python3
"""Reconcile paper ledger rows against dry-run/live order tickets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine.orders import ORDER_LEDGER
from engine.reconcile import reconcile_order_tickets


DEFAULT_PAPER_LEDGER = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_PAPER_LEDGER))
    parser.add_argument("--order-ledger", type=str, default=str(ORDER_LEDGER))
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--variants", type=str, default="A,C")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    variant_to_source = {"A": "paper", "C": "paper_c"}
    sources = {
        variant_to_source[v.strip().upper()]
        for v in args.variants.split(",")
        if v.strip().upper() in variant_to_source
    }
    report = reconcile_order_tickets(
        Path(args.ledger),
        Path(args.order_ledger),
        trade_date=args.date,
        sources=sources,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("Order-ticket reconciliation")
        print(f"  ok: {report.ok}")
        print(f"  ledger rows: {report.ledger_rows}")
        print(f"  order records: {report.order_records}")
        print(f"  missing: {report.missing_tickets}")
        print(f"  duplicates: {report.duplicate_tickets}")
        print(f"  unknown: {report.unknown_tickets}")
        print(f"  variant mismatches: {report.variant_mismatches}")
    return 0 if report.ok else 7


if __name__ == "__main__":
    sys.exit(main())
