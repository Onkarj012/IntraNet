#!/usr/bin/env python3
"""Live-execution scaffolding for the OptiNet futures engine.

Reads a paper_trading_ledger.csv row (or all rows for a date) and emits
order tickets through the OrderClient.  Default mode is dry-run:
tickets are written to results/router_v0/order_tickets.jsonl but no
orders are placed.

Real broker placement requires the FULL triple-key gate to pass:
  --live                                    (CLI flag)
  OPTINET_LIVE=1                            (env var)
  --confirm-token /path/to/token.txt        (file content matches
                                              results/router_v0/LIVE_TOKEN)
PLUS the kill-switch file must NOT exist.

Even with all gates clear, UpstoxOrderClient.place_order() raises
NotImplementedError until the SDK is wired.  The user must intentionally
edit src/optinet/v5_runtime/orders.py to wire real execution.

Usage:
  # Default: dry-run for the most recent paper-trading day, Variant A only
  scripts/live_execute.py --auto

  # Specific date, both variants
  scripts/live_execute.py --date 2026-04-15 --variants A,C

  # Live (requires ALL gates clear; will FAIL until SDK wired)
  scripts/live_execute.py --date 2026-04-15 --variant A --live \\
                          --confirm-token /tmp/my_token.txt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine.orders import (
    OrderTicket, LiveExecutionGate, make_order_client, ORDER_LEDGER,
)
from engine.reconcile import reconcile_order_tickets
from engine.risk import BrokerState, RiskLimits, evaluate_ticket_risk

LEDGER_PATH = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"


def ticket_from_ledger_row(row: pd.Series, intended_for_live: bool) -> OrderTicket:
    """Convert one paper_trading_ledger row into a bracket-order ticket."""
    src = str(row["source"])
    variant = "C" if src == "paper_c" else "A"
    return OrderTicket(
        ticket_id=f"{variant}-{row['paper_trade_id']}",
        paper_trade_id=str(row["paper_trade_id"]),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        symbol="NIFTY",
        expiry=None,                              # front-month futures, broker resolves
        side="BUY" if row["side"] == "LONG" else "SELL",
        qty_lots=int(row["size_mult"] * 1),       # 1 lot * size_mult; round up below
        order_type="MARKET",
        limit_price=None,
        target_price=float(row["target_px"]),
        stop_price=float(row["stop_px"]),
        horizon_minutes=60,
        size_mult=float(row["size_mult"]),
        variant=variant,
        intended_for_live=intended_for_live,
        notes=f"src={src}; entry_px={row['entry_px']}; score={row['long_score']}",
        trade_date=str(row["trade_date"]),
    )


def get_most_recent_paper_date(df: pd.DataFrame, sources: list[str]) -> Optional[str]:
    sub = df[df["source"].isin(sources)]
    if sub.empty:
        return None
    return sub["trade_date"].max()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--auto", action="store_true",
                   help="Use most recent paper-trading date in the ledger")
    p.add_argument("--variants", type=str, default="A",
                   help="Comma-separated variants to execute (A, C, or A,C)")
    p.add_argument("--live", action="store_true",
                   help="Attempt real broker execution (requires triple-key gate)")
    p.add_argument("--confirm-token", type=str, default=None,
                   help="Path to a file whose content matches the LIVE_TOKEN reference")
    p.add_argument("--ledger", type=str, default=str(LEDGER_PATH))
    p.add_argument("--order-ledger", type=str, default=str(ORDER_LEDGER),
                   help="Dry-run ticket JSONL sink")
    p.add_argument("--max-live-lots", type=int, default=1,
                   help="Maximum lots per order allowed by risk governor")
    p.add_argument("--broker-open-positions", type=int, default=0,
                   help="Manual broker-state input for risk checks")
    p.add_argument("--broker-day-pnl", type=float, default=0.0,
                   help="Manual broker realized day PnL input for risk checks")
    p.add_argument("--skip-reconcile", action="store_true",
                   help="Do not print post-run ticket reconciliation")
    args = p.parse_args()

    print("╔" + "═" * 88 + "╗")
    print("║  Live execution scaffolding — order ticket emitter".ljust(89) + "║")
    print("╚" + "═" * 88 + "╝")

    # 1) Evaluate the triple-key gate up-front, every time
    gate = LiveExecutionGate.evaluate(
        cli_live=bool(args.live),
        confirm_token_path=Path(args.confirm_token) if args.confirm_token else None,
    )
    print(f"\n  ── live-execution gate ──")
    for line in gate.detail:
        print(f"    {line}")
    cleared = gate.is_clear()
    print(f"  → gate {'CLEARED → live mode' if cleared else 'NOT cleared → dry-run mode'}")

    if args.live and not cleared:
        print("\n  ⚠️  --live was passed but the gate did not clear.")
        print("     Falling back to dry-run.  No real orders will be placed.")

    # 2) Load ledger
    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        print(f"\n  ledger not found: {ledger_path}")
        return 1
    df = pd.read_csv(ledger_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

    # 3) Resolve target variants and date
    requested_variants = {v.strip().upper() for v in args.variants.split(",")}
    src_map = {"A": "paper", "C": "paper_c"}
    target_sources = [src_map[v] for v in requested_variants if v in src_map]
    if not target_sources:
        print(f"  no valid variants in {args.variants!r} (allowed: A, C)")
        return 1

    if args.date:
        target_date = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    elif args.auto:
        target_date = get_most_recent_paper_date(df, target_sources)
        if target_date is None:
            print(f"  --auto: no rows in ledger for variants={target_sources}")
            return 0
        print(f"  --auto resolved to {target_date}")
    else:
        print("  must pass --date or --auto")
        return 1

    rows = df[(df["trade_date"] == target_date) &
                df["source"].isin(target_sources)]
    if rows.empty:
        print(f"  no ledger rows for {target_date} variants={requested_variants}")
        return 0

    print(f"\n  {len(rows)} rows for {target_date} "
          f"variants={sorted(requested_variants)}")

    # 4) Make order client
    order_ledger_path = Path(args.order_ledger)
    client = make_order_client(cleared, dry_run_ledger_path=order_ledger_path)
    print(f"  order client: {client.name()}")
    print(f"  ticket sink: {order_ledger_path if not cleared else ORDER_LEDGER}")
    limits = RiskLimits(max_qty_lots_per_order=int(args.max_live_lots))
    broker_state = BrokerState(
        open_positions=int(args.broker_open_positions),
        day_realized_pnl_inr=float(args.broker_day_pnl),
    )

    # 5) Emit tickets
    print(f"\n  ── tickets ──")
    n_emitted = n_skipped = n_failed = 0
    for _, row in rows.iterrows():
        ticket = ticket_from_ledger_row(row, intended_for_live=cleared)
        print(f"\n  ticket {ticket.ticket_id}")
        print(f"    paper_trade_id  {ticket.paper_trade_id}")
        print(f"    variant         {ticket.variant}")
        print(f"    side / size     {ticket.side}  {ticket.qty_lots} lot(s)  "
              f"(size_mult={ticket.size_mult})")
        print(f"    order_type      {ticket.order_type}")
        print(f"    target / stop   ₹{ticket.target_price:,.2f}  /  "
              f"₹{ticket.stop_price:,.2f}")
        print(f"    intended_live   {ticket.intended_for_live}")
        risk = evaluate_ticket_risk(
            ticket,
            trade_date=target_date,
            order_ledger=order_ledger_path,
            limits=limits,
            broker_state=broker_state,
        )
        print(f"    risk            {'OK' if risk.ok else 'BLOCKED'}")
        for reason in risk.reasons:
            print(f"      - {reason}")
        if cleared and not risk.ok:
            print("    response:       {'status': 'RISK_BLOCKED'}")
            n_failed += 1
            continue
        try:
            resp = client.place_order(ticket)
            print(f"    response:       {resp}")
            if resp.get("status") == "DRYRUN_DUPLICATE_SKIPPED":
                n_skipped += 1
            else:
                n_emitted += 1
        except NotImplementedError as e:
            print(f"    ⚠️  place_order raised NotImplementedError")
            print(f"       {str(e)[:100]}")
            n_failed += 1

    print(f"\n  emitted: {n_emitted}    skipped_duplicates: {n_skipped}    failed: {n_failed}")
    if not args.skip_reconcile:
        report = reconcile_order_tickets(
            ledger_path,
            order_ledger_path,
            trade_date=target_date,
            sources=set(target_sources),
        )
        print("\n  -- ticket reconciliation --")
        print(f"  ok: {report.ok}")
        print(f"  ledger_rows: {report.ledger_rows}  order_records: {report.order_records}")
        if report.missing_tickets:
            print(f"  missing: {report.missing_tickets}")
        if report.duplicate_tickets:
            print(f"  duplicates: {report.duplicate_tickets}")
        if report.unknown_tickets:
            print(f"  unknown: {report.unknown_tickets}")
        if report.variant_mismatches:
            print(f"  variant mismatches: {report.variant_mismatches}")
    return 0 if n_failed == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
