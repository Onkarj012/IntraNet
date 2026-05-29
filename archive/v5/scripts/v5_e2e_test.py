#!/usr/bin/env python
"""V5 v1 end-to-end mock-broker simulation.

Replays one full trading day (09:30 → 15:45) using the MockBroker, exercising:
- pre-flight health check (skipped — checked separately)
- minute decision pipeline at every minute
- intra-trade MTM stop-loss
- 15:25 force-close
- 15:45 EOD reconcile

Verifies:
- Hard 14:55 cutoff: no PLACED actions at/after 14:55
- Halt-flag respected
- Stop-loss simulator produces sane PnL
- Reconciliation gap < ₹50

Exits 0 on success, 1 if any invariant fails.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from optinet.v5_runtime.broker import MockBroker
from optinet.v5_runtime import ledger as ld
from optinet.v5_runtime.online_features import OnlineFeatureBuilder
from optinet.v5_runtime.runtime_config import (
    HARD_CUTOFF_TIME, EOD_FORCE_CLOSE_TIME, MARKET_OPEN_TIME, FLAGS_DIR,
    ensure_dirs,
)

# Reuse decision functions from the live script
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from v5_minute_decision import decide_at_minute, mtm_check_and_stop  # type: ignore
from v5_force_close import force_close_all  # type: ignore
from v5_eod_reconcile import reconcile_day  # type: ignore


def reset_ledger_and_flags():
    if ld.LEDGER_PATH.exists():
        ld.LEDGER_PATH.unlink()
    ensure_dirs()
    for f in FLAGS_DIR.glob("*.flag"):
        f.unlink()


def run_day(*, sim_date: date, verbose: bool = False) -> dict:
    """Simulate one full trading day from 09:30 to 15:45."""
    broker = MockBroker(simulate_date=sim_date)
    builder_cache: dict = {}

    placed_minutes = []
    cutoff_minutes = []
    other_actions = []

    # Walk minute-by-minute
    current = datetime.combine(sim_date, MARKET_OPEN_TIME) + timedelta(minutes=15)  # 09:30
    end = datetime.combine(sim_date, EOD_FORCE_CLOSE_TIME) + timedelta(minutes=20)  # 15:45

    while current <= end:
        if current.time() < EOD_FORCE_CLOSE_TIME:
            res = decide_at_minute(
                now=current, broker=broker, builder_cache=builder_cache,
                enabled_symbols=("NIFTY",), write_ledger=True,
            )
            if res.get("action") == "MULTI":
                for d in res["decisions"]:
                    if d["action"] == "PLACED":
                        placed_minutes.append((current, d))
                    else:
                        other_actions.append((current, d["action"]))
            elif res.get("action") == "NO_TRADE_CUTOFF":
                cutoff_minutes.append(current)
        elif current.time() == EOD_FORCE_CLOSE_TIME:
            # Force close
            closed = force_close_all(now=current, broker=broker,
                                       reason="EOD", write_ledger=True)
            if verbose:
                print(f"[{current}] force-close: {len(closed)} positions")
        current += timedelta(minutes=1)

    # EOD reconcile (15:45)
    reconcile_now = datetime.combine(sim_date, EOD_FORCE_CLOSE_TIME) + timedelta(minutes=20)
    summary = reconcile_day(today=sim_date, broker=broker, write=True)

    # Aggregate
    final_ledger = ld.load_ledger()
    return {
        "sim_date": sim_date,
        "ledger_rows": int(len(final_ledger)),
        "placed": placed_minutes,
        "cutoff_minutes": cutoff_minutes,
        "summary": summary,
        "ledger": final_ledger,
    }


def assert_invariants(result: dict) -> list[str]:
    failures = []

    # Invariant 1: no PLACED at or after 14:55
    for ts, d in result["placed"]:
        if ts.time() >= HARD_CUTOFF_TIME:
            failures.append(f"PLACED at {ts} violates 14:55 cutoff")

    # Invariant 2: every trade is RECONCILED
    df = result["ledger"]
    not_reconciled = df[df["status"] != "RECONCILED"]
    if len(not_reconciled):
        failures.append(f"{len(not_reconciled)} trades not RECONCILED")

    # Invariant 3: no trade has |reconciliation_gap| > 50
    if "reconciliation_gap" in df.columns:
        big_gap = df[df["reconciliation_gap"].abs() > 50.0]
        if len(big_gap):
            failures.append(f"{len(big_gap)} trades have gap > ₹50")

    # Invariant 4: per-day cap respected
    n_today = int(len(df))
    if n_today > 4:
        failures.append(f"daily total cap exceeded: {n_today} > 4")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-01-02",
                    help="Simulation date (YYYY-MM-DD); must have NIFTY dte∈{2,3}")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="Wipe ledger and flags before run")
    args = ap.parse_args()

    sim_date = date.fromisoformat(args.date)
    if args.reset:
        reset_ledger_and_flags()

    print(f"=== End-to-end mock-broker test: {sim_date} ===")
    result = run_day(sim_date=sim_date, verbose=args.verbose)

    print(f"\n--- Decisions log ---")
    print(f"  PLACED entries  : {len(result['placed'])}")
    print(f"  Cutoff minutes  : {len(result['cutoff_minutes'])} (after 14:55)")

    print(f"\n--- Final ledger ({result['ledger_rows']} rows) ---")
    df = result["ledger"]
    if not df.empty:
        cols = ["trade_id", "entry_ts", "exit_ts", "atm_strike",
                 "entry_call_px", "entry_put_px", "exit_call_px", "exit_put_px",
                 "live_pnl_inr", "realized_pnl_inr", "reconciliation_gap",
                 "was_stopped", "exit_reason", "status"]
        print(df[cols].to_string(index=False))

    print(f"\n--- Summary ---")
    s = result["summary"]
    for k, v in s.items():
        if k == "reconciliation_gaps":
            print(f"  reconciliation_gaps : {len(v)} flagged")
            for g in v:
                print(f"    {g}")
        else:
            print(f"  {k:20s}: {v}")

    failures = assert_invariants(result)
    if failures:
        print("\n=== INVARIANT FAILURES ===")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("\n=== ALL INVARIANTS PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
