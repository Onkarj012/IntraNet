#!/usr/bin/env python3
"""Intraday equity — single daily orchestrator (full automated chain).

Modes:
  --mode preopen   (08:30 IST): update panel → picks → score (pre-open) → push
  --mode postopen  (09:22 IST): re-score with real 5-min ORB bars → push
  --mode eod       (15:35 IST): reconcile paper ledger → drift monitor → push

Writes results/intraday_daily_status.json after each run for dashboard/ops.
yfinance only — no Kite token required.

Cron:
  30 8  * * 1-5  .venv/bin/python scripts/trading/intraday_daily.py --mode preopen
  22 9  * * 1-5  .venv/bin/python scripts/trading/intraday_daily.py --mode postopen
  35 15 * * 1-5  .venv/bin/python scripts/trading/intraday_daily.py --mode eod
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
STATUS = ROOT / "results/intraday_daily_status.json"


def step(label: str, cmd: list[str], *, fatal: bool = False) -> dict:
    print(f"\n{'='*70}\n  {label}\n  {' '.join(cmd)}\n{'='*70}", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"  exit code: {rc}", flush=True)
    return {"label": label, "return_code": rc, "ok": rc == 0, "fatal": fatal and rc != 0}


def run_chain(mode: str) -> list[dict]:
    steps: list[dict] = []

    if mode == "preopen":
        steps.append(step("Update OHLCV panel (yfinance)",
                          [PYTHON, "scripts/data/equity_eod_panel.py", "--universe", "nifty500", "--update"]))
        steps.append(step("Fit regime detector",
                          [PYTHON, "scripts/trading/fit_regime_detector.py"]))
        steps.append(step("Generate equity picks",
                          [PYTHON, "scripts/trading/equity_picks.py"], fatal=True))
        steps.append(step("Morning run (pre-open scoring)",
                          [PYTHON, "scripts/trading/morning_run.py"]))

    elif mode == "postopen":
        steps.append(step("Morning run (post-open 09:20 ORB re-score)",
                          [PYTHON, "scripts/trading/morning_run.py", "--post-open"]))

    elif mode == "eod":
        steps.append(step("Reconcile paper ledger",
                          [PYTHON, "scripts/trading/equity_intraday_paper.py"]))
        steps.append(step("Drift monitor",
                          [PYTHON, "scripts/trading/equity_drift_monitor.py"]))
        steps.append(step("Push dashboard",
                          [PYTHON, "scripts/trading/push_dashboard.py"]))

    else:
        raise ValueError(f"unknown mode: {mode}")

    return steps


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["preopen", "postopen", "eod"])
    args = p.parse_args()

    ts = pd.Timestamp.now(tz="Asia/Kolkata")
    print(f"\n  intraday_daily [{args.mode}]  {ts.isoformat(timespec='seconds')}")

    steps = run_chain(args.mode)
    exit_code = next((s["return_code"] for s in steps if s["fatal"]), 0)

    status = {
        "mode": args.mode,
        "run_timestamp": ts.isoformat(timespec="seconds"),
        "date": str(ts.date()),
        "steps": steps,
        "exit_code": exit_code,
        "ok": all(s["ok"] for s in steps),
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    # Merge with prior status so each mode's result is preserved
    prior = {}
    if STATUS.exists():
        try:
            prior = json.loads(STATUS.read_text())
            if not isinstance(prior.get("runs"), dict):
                prior = {"runs": {}}
        except Exception:
            prior = {"runs": {}}
    runs = prior.get("runs", {})
    runs[args.mode] = status
    STATUS.write_text(json.dumps({"updated_at": ts.isoformat(timespec="seconds"), "runs": runs}, indent=2))
    print(f"\n  status → {STATUS}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
