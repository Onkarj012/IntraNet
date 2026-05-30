#!/usr/bin/env python3
"""Daily unattended run — single cron entrypoint.

Chains in order:
  1. EOD data cache  — appends today's NIFTY bars + VIX (requires valid KITE_ACCESS_TOKEN)
  2. Paper trading ops — Variant A + C replay, status dashboard, halt checks

The Kite access token must be refreshed manually each morning via
`scripts/data/kite_login.py` before this script runs.

Exit codes:
  0  all steps succeeded, no halts
  2  paper run succeeded, soft halt(s) triggered (alerts only — run continues)
  3  HARD HALT triggered — kill-switch written, investigate before resuming
  non-zero from step 1 → EOD cache failed, paper run skipped

Usage (manual):
    .venv/bin/python scripts/trading/daily_run.py

Cron (18:00 IST Mon-Fri, after NSE publishes EOD data):
    0 18 * * 1-5  cd /path/to/intranet_optinet && \
                  .venv/bin/python scripts/trading/daily_run.py \
                  >> logs/daily_run.log 2>&1
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv/bin/python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run(label: str, cmd: list[str]) -> int:
    print(f"\n{'='*70}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'='*70}", flush=True)
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    print(f"  exit code: {rc}", flush=True)
    return rc


def main() -> int:
    print(f"\n  daily_run  {pd.Timestamp.now(tz='Asia/Kolkata').isoformat(timespec='seconds')}")

    # 1. EOD data cache (needs KITE_ACCESS_TOKEN set in .env)
    rc = run("Step 1: EOD data cache (NIFTY bars + VIX)",
             [PYTHON, "scripts/data/kite_eod_cache.py"])
    if rc != 0:
        print("\n  ABORT: EOD cache failed — skipping paper run")
        return rc

    # 2. Paper trading ops (Variant A + C + status + halt checks)
    rc = run("Step 2: Paper trading ops",
             [PYTHON, "scripts/trading/paper_ops.py", "--auto", "--write-halt"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
