#!/usr/bin/env python3
"""Daily unattended run — single cron entrypoint.

Chains in order:
  1. EOD data cache  — appends today's NIFTY bars + VIX (requires valid KITE_ACCESS_TOKEN)
  2. Futures paper trading ops — Variant A + C replay, status dashboard, halt checks
  3. Equity momentum ops — update panel, generate picks, append ledger, status + halt checks

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

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv/bin/python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

STATUS_PATH = PROJECT_ROOT / "results/daily_run_status.json"
FUT_HALT = PROJECT_ROOT / "results/router_v0/PAPER_TRADING_HALTED"
EQ_HALT = PROJECT_ROOT / "results/equity/EQUITY_PAPER_HALTED"


def write_status(steps: list[dict], exit_code: int) -> None:
    """Consolidated machine-readable status for the monitoring dashboard."""
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({
        "run_timestamp": now.isoformat(timespec="seconds"),
        "date": str(now.date()),
        "steps": steps,
        "exit_code": exit_code,
        "ok": exit_code == 0,
        "futures_halted": FUT_HALT.exists(),
        "equity_halted": EQ_HALT.exists(),
    }, indent=2))
    print(f"\n  status written → {STATUS_PATH}")


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
    steps: list[dict] = []

    def record(label: str, rc: int) -> int:
        steps.append({"label": label, "return_code": rc, "ok": rc == 0})
        return rc

    # 1. EOD data cache (needs KITE_ACCESS_TOKEN set in .env)
    rc = record("EOD data cache", run("Step 1: EOD data cache (NIFTY bars + VIX)",
                                       [PYTHON, "scripts/data/kite_eod_cache.py"]))
    if rc != 0:
        print("\n  ABORT: EOD cache failed — skipping paper run")
        write_status(steps, rc)
        return rc

    # 2. Paper trading ops (Variant A + C + status + halt checks)
    # paper_ops.py already passes --write-halt to paper_status.py internally
    futures_rc = record("Futures paper ops", run("Step 2: Futures paper trading ops",
                                                 [PYTHON, "scripts/trading/paper_ops.py", "--auto"]))

    # 3. Equity momentum ops (panel update → picks → ledger → status)
    # futures halt is non-blocking for equity — equity runs independently
    equity_rc = record("Equity momentum ops", run("Step 3: Equity momentum ops",
                                                  [PYTHON, "scripts/trading/equity_ops.py"]))

    # Worst exit code (3 hard halt > 2 soft halt > 0 ok)
    exit_code = max(futures_rc, equity_rc)
    write_status(steps, exit_code)

    # 4. Push artifacts to the hosted dashboard (no-op if CONVEX_HTTP_URL unset).
    #    Best-effort: a push failure never changes the trading exit code.
    run("Step 4: Push dashboard snapshot", [PYTHON, "scripts/trading/push_dashboard.py"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
