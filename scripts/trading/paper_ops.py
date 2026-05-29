#!/usr/bin/env python3
"""Daily paper-trading operations wrapper.

Runs the measurement stack in order:
  1. Variant A paper runner
  2. Variant C observation runner
  3. Status dashboard with halt checks

The wrapper does not change model/config artifacts. It returns nonzero only
when a runner/status command reports a real operational failure or halt state.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine.ops_report import (
    OpsRunReport,
    OpsStep,
    next_report_path,
    summarize_paper_ledger,
)

DEFAULT_LEDGER = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def run_step(label: str, cmd: list[str]) -> int:
    print("\n" + "=" * 90, flush=True)
    print(f"{label}", flush=True)
    print("=" * 90, flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f"\n{label} exit code: {result.returncode}", flush=True)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--auto", action="store_true",
                       help="Use the latest date in the source minute data")
    group.add_argument("--date", type=str, default=None,
                       help="Run an explicit trade date YYYY-MM-DD")
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER))
    parser.add_argument("--allow-stale-data", action="store_true",
                        help="Pass through to paper runners for intentional historical replays")
    parser.add_argument("--allow-incomplete-session", action="store_true",
                        help="Pass through to paper runners for intentional historical replays")
    parser.add_argument("--ignore-kill-switch", action="store_true",
                        help="Pass through to paper runners")
    parser.add_argument("--include-bootstrap", action="store_true",
                        help="Include bootstrap reference in final status output")
    parser.add_argument("--by-month", action="store_true",
                        help="Show monthly breakdowns in final status output")
    parser.add_argument("--report-path", type=str, default=None,
                        help="Write a JSON ops report to this path")
    args = parser.parse_args()

    date_args = ["--auto"] if args.auto or not args.date else ["--date", args.date]
    common_runner_args = [*date_args, "--ledger", args.ledger]
    if args.allow_stale_data:
        common_runner_args.append("--allow-stale-data")
    if args.allow_incomplete_session:
        common_runner_args.append("--allow-incomplete-session")
    if args.ignore_kill_switch:
        common_runner_args.append("--ignore-kill-switch")

    steps = [
        ("Variant A paper run", [sys.executable, "scripts/paper_trade_daily.py", *common_runner_args]),
        ("Variant C paper run", [sys.executable, "scripts/paper_trade_variant_c.py", *common_runner_args]),
    ]

    report = OpsRunReport(run_timestamp=str(pd.Timestamp.now(tz="Asia/Kolkata")))
    report_path = Path(args.report_path) if args.report_path else next_report_path(DEFAULT_LOG_DIR)

    for label, cmd in steps:
        rc = run_step(label, cmd)
        report.steps.append(OpsStep(label=label, command=cmd, return_code=rc))
        report.artifacts["paper_ledger"] = summarize_paper_ledger(Path(args.ledger))
        report.write(report_path)
        if rc != 0:
            print(f"\nops report: {report_path}")
            return rc

    status_cmd = [
        sys.executable,
        "scripts/paper_trade_status.py",
        "--ledger",
        args.ledger,
        "--write-halt",
    ]
    if args.include_bootstrap:
        status_cmd.append("--include-bootstrap")
    if args.by_month:
        status_cmd.append("--by-month")

    rc = run_step("Paper-trading status + halt checks", status_cmd)
    report.steps.append(OpsStep(label="Paper-trading status + halt checks", command=status_cmd, return_code=rc))
    report.artifacts["paper_ledger"] = summarize_paper_ledger(Path(args.ledger))
    report.write(report_path)
    print(f"\nops report: {report_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
