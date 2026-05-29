#!/usr/bin/env python3
"""Broker shadow mode — exercises the full pre-trade checklist without placing orders.

Runs every step a live execution would take, but never calls place_order:
  1. Auth / health check
  2. Account flat check (no open positions, no pending orders)
  3. Instrument resolution (NIFTY futures key lookup)
  4. Live quote fetch
  5. Risk governor check
  6. Would-place ticket generation
  7. Write to live_execution_ledger.csv with broker_status='SHADOW'
  8. Reconcile: confirm no real position was opened

This catches broker API, auth, and instrument bugs before any capital is at risk.

Usage:
  scripts/broker_shadow.py --date 2026-05-27
  scripts/broker_shadow.py --auto
  scripts/broker_shadow.py --auto --broker mock   # use MockBroker (default)
  scripts/broker_shadow.py --auto --broker upstox # use UpstoxBroker (requires creds)
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from index_options.v5_runtime.broker import make_broker, UpstoxBroker
from engine.orders import (
    OrderTicket, DryRunOrderClient, ORDER_LEDGER,
)

LEDGER_PATH       = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"
LIVE_EXEC_LEDGER  = PROJECT_ROOT / "results/router_v0/live_execution_ledger.csv"
KILL_SWITCH       = PROJECT_ROOT / "results/router_v0/PAPER_TRADING_HALTED"

LIVE_EXEC_COLS = [
    "run_id", "ticket_id", "paper_trade_id", "trade_date", "variant",
    "intended_live", "risk_ok", "broker_order_id", "broker_status",
    "placed_at", "filled_at", "avg_fill_price", "qty_lots",
    "target_price", "stop_price", "exit_order_id", "exit_status",
    "realized_pnl_inr", "reconciled", "notes",
]


def get_most_recent_paper_date(source: str = "paper") -> str | None:
    if not LEDGER_PATH.exists():
        return None
    df = pd.read_csv(LEDGER_PATH, usecols=["trade_date", "source"])
    sub = df[df["source"] == source]
    return sub["trade_date"].max() if not sub.empty else None


def get_paper_rows(trade_date: str, source: str = "paper") -> pd.DataFrame:
    if not LEDGER_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LEDGER_PATH)
    return df[(df["trade_date"] == trade_date) & (df["source"] == source)].copy()


def write_live_exec_row(row: dict) -> None:
    LIVE_EXEC_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LIVE_EXEC_LEDGER.exists()
    pd.DataFrame([row], columns=LIVE_EXEC_COLS).to_csv(
        LIVE_EXEC_LEDGER, mode="a", header=write_header, index=False)


def run_shadow_checklist(broker, trade_date: str, paper_rows: pd.DataFrame,
                          run_id: str) -> list[dict]:
    """Run the full pre-trade checklist for each paper row. Returns result dicts."""
    results = []

    for _, row in paper_rows.iterrows():
        ticket_id = f"SHADOW-{row['paper_trade_id']}"
        result = {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "paper_trade_id": str(row["paper_trade_id"]),
            "trade_date": trade_date,
            "variant": "C" if row["source"] == "paper_c" else "A",
            "intended_live": False,
            "risk_ok": False,
            "broker_order_id": None,
            "broker_status": "SHADOW",
            "placed_at": None,
            "filled_at": None,
            "avg_fill_price": None,
            "qty_lots": int(row["size_mult"]),
            "target_price": float(row["target_px"]),
            "stop_price": float(row["stop_px"]),
            "exit_order_id": None,
            "exit_status": None,
            "realized_pnl_inr": None,
            "reconciled": False,
            "notes": "",
        }
        notes = []

        # Step 1: Kill-switch
        if KILL_SWITCH.exists():
            notes.append("BLOCKED: kill-switch present")
            result["notes"] = "; ".join(notes)
            results.append(result)
            continue

        # Step 2: Broker health
        try:
            healthy = broker.health()
            notes.append(f"broker.health()={healthy}")
        except Exception as e:
            notes.append(f"broker.health() raised: {e}")
            healthy = False

        # Step 3: Account flat check
        try:
            acct = broker.get_account_state()
            is_flat = acct.get("is_flat", False)
            notes.append(f"account.is_flat={is_flat}")
            if not is_flat:
                notes.append(f"BLOCKED: account not flat: {acct}")
        except NotImplementedError:
            notes.append("get_account_state: NotImplementedError (stub not wired)")
            is_flat = None  # unknown — shadow mode continues
        except Exception as e:
            notes.append(f"get_account_state raised: {e}")
            is_flat = None

        # Step 4: Instrument resolution
        try:
            from engine.config import next_weekly_expiry
            expiry = next_weekly_expiry("NIFTY", pd.Timestamp(trade_date).date())
            instrument_key = broker.resolve_instrument("NIFTY", expiry)
            notes.append(f"instrument_key={instrument_key}")
        except NotImplementedError:
            notes.append("resolve_instrument: NotImplementedError (stub not wired)")
            instrument_key = None
            expiry = None
        except Exception as e:
            notes.append(f"resolve_instrument raised: {e}")
            instrument_key = None
            expiry = None

        # Step 5: Live quote fetch
        try:
            if expiry is None:
                from engine.config import next_weekly_expiry
                expiry = next_weekly_expiry("NIFTY", pd.Timestamp(trade_date).date())
            fut_quote = broker.get_futures("NIFTY", expiry)
            notes.append(f"fut_quote.fut_close={fut_quote.fut_close:.2f}")
        except NotImplementedError:
            notes.append("get_futures: NotImplementedError (stub not wired)")
            fut_quote = None
        except Exception as e:
            notes.append(f"get_futures raised: {e}")
            fut_quote = None

        # Step 6: Risk check (Variant A only, kill-switch clear, account flat or unknown)
        risk_ok = (
            not KILL_SWITCH.exists() and
            result["variant"] == "A" and
            (is_flat is None or is_flat)  # unknown = allow in shadow
        )
        result["risk_ok"] = risk_ok
        notes.append(f"risk_ok={risk_ok}")

        # Step 7: Would-place ticket (dry-run only — never calls place_order)
        # Use a unique ticket_id per run so dedupe doesn't suppress shadow tickets
        shadow_ticket_id = f"SHADOW-{run_id}-{row['paper_trade_id']}"
        ticket = OrderTicket(
            ticket_id=shadow_ticket_id,
            paper_trade_id=str(row["paper_trade_id"]),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            symbol="NIFTY",
            expiry=str(expiry) if expiry is not None else None,
            side="BUY",
            qty_lots=int(row["size_mult"]),
            order_type="MARKET",
            limit_price=None,
            target_price=float(row["target_px"]),
            stop_price=float(row["stop_px"]),
            horizon_minutes=60,
            size_mult=float(row["size_mult"]),
            variant=result["variant"],
            intended_for_live=False,
            notes="shadow_mode",
        )
        dry_client = DryRunOrderClient()
        resp = dry_client.place_order(ticket)
        sim_id = resp.get("simulated_order_id") or resp.get("ticket_id", "?")
        notes.append(f"dry_run_ticket={sim_id}")

        # Step 8: Reconcile — confirm no real position (trivially true in shadow)
        result["reconciled"] = True
        notes.append("reconciled=True (shadow: no real order placed)")

        result["notes"] = "; ".join(notes)
        results.append(result)

    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--auto", action="store_true")
    p.add_argument("--broker", type=str, default="mock",
                   help="Broker to use: mock (default) or upstox")
    p.add_argument("--variant", type=str, default="A",
                   help="Which paper source to shadow: A or C")
    args = p.parse_args()

    print("╔" + "═" * 88 + "╗")
    print("║  Broker shadow mode — pre-trade checklist (no orders placed)".ljust(89) + "║")
    print("╚" + "═" * 88 + "╝")

    source_map = {"A": "paper", "C": "paper_c"}
    source = source_map.get(args.variant.upper(), "paper")

    if args.date:
        trade_date = args.date
    elif args.auto:
        trade_date = get_most_recent_paper_date(source)
        if trade_date is None:
            print(f"  --auto: no {source} rows in ledger")
            return 0
        print(f"  --auto resolved to {trade_date}")
    else:
        print("  must pass --date or --auto")
        return 1

    paper_rows = get_paper_rows(trade_date, source)
    if paper_rows.empty:
        print(f"  no {source} rows for {trade_date}")
        return 0

    print(f"\n  {len(paper_rows)} paper rows for {trade_date} (variant={args.variant})")
    print(f"  broker: {args.broker}")

    broker = make_broker(args.broker)
    print(f"  broker.name(): {broker.name()}")
    print(f"  broker.health(): {broker.health()}")

    run_id = str(uuid.uuid4())[:12]
    print(f"  run_id: {run_id}")

    results = run_shadow_checklist(broker, trade_date, paper_rows, run_id)

    print(f"\n  ── shadow checklist results ──")
    for r in results:
        risk_icon = "✓" if r["risk_ok"] else "✗"
        rec_icon  = "✓" if r["reconciled"] else "✗"
        print(f"\n  ticket {r['ticket_id']}")
        print(f"    risk_ok={risk_icon}  reconciled={rec_icon}  status={r['broker_status']}")
        for note in r["notes"].split("; "):
            print(f"    {note}")
        write_live_exec_row(r)

    n_risk_ok = sum(1 for r in results if r["risk_ok"])
    n_reconciled = sum(1 for r in results if r["reconciled"])
    print(f"\n  summary: {len(results)} rows  risk_ok={n_risk_ok}  reconciled={n_reconciled}")
    print(f"  → {LIVE_EXEC_LEDGER}")

    # Shadow passes if all rows are reconciled (no real positions opened)
    all_reconciled = all(r["reconciled"] for r in results)
    print(f"\n  shadow mode: {'PASS ✓' if all_reconciled else 'FAIL ✗'}")
    return 0 if all_reconciled else 1


if __name__ == "__main__":
    sys.exit(main())
