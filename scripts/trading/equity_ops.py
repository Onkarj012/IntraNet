#!/usr/bin/env python
"""Equity momentum daily ops orchestrator.

Chains in order:
  1. Update adjusted panel (yfinance --update, incremental)
  2. Generate today's picks JSON
  3. Append live ledger (equity_paper_trade.py)
  4. Status dashboard + halt checks (equity_paper_status.py)

Exit codes mirror paper_ops.py:
  0  all OK
  2  soft halt triggered (alert only, run continues)
  3  hard halt — kill-switch written
  non-zero from step 1/2 → abort

Usage:
  python scripts/trading/equity_ops.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parents[2]
PYTHON = str(_root / ".venv/bin/python") if (_root / ".venv/bin/python").exists() else sys.executable


def run(label: str, cmd: list[str], abort_on_fail: bool = True) -> int:
    print(f"\n{'='*70}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'='*70}", flush=True)
    rc = subprocess.run(cmd, cwd=_root).returncode
    print(f"  exit code: {rc}", flush=True)
    if rc != 0 and abort_on_fail:
        print(f"\n  ABORT: {label} failed (rc={rc})")
    return rc


def main() -> int:
    print(f"\n  equity_ops  {pd.Timestamp.now(tz='Asia/Kolkata').isoformat(timespec='seconds')}")

    # 1. Update panel (incremental yfinance fetch — non-fatal if partial)
    rc = run("Step 1: Update adjusted EOD panel",
             [PYTHON, "scripts/data/equity_eod_panel.py", "--universe", "nifty500", "--update"],
             abort_on_fail=False)
    if rc != 0:
        print("  WARNING: panel update had errors — continuing with existing data")

    # 2. Generate today's picks
    rc = run("Step 2: Generate equity picks",
             [PYTHON, "scripts/trading/equity_picks.py"])
    if rc != 0:
        return rc

    # 3. Append live ledger
    rc = run("Step 3: Append live paper ledger",
             [PYTHON, "scripts/trading/equity_paper_trade.py"],
             abort_on_fail=False)
    if rc == 3:
        return 3  # hard halt — stop here

    # 4. Status + halt checks
    rc_status = run("Step 4: Status dashboard + halt checks",
                    [PYTHON, "scripts/trading/equity_paper_status.py", "--write-halt"],
                    abort_on_fail=False)
    return rc_status


if __name__ == "__main__":
    sys.exit(main())
