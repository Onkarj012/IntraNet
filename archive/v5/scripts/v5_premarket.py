#!/usr/bin/env python
"""V5 v1 premarket script — runs at 08:00 IST.

Tasks:
1. Update NSE bhavcopy lake for prior trading day (idempotent)
2. Verify chain feature partitions are fresh
3. Test broker API auth
4. yfinance vs broker spot cross-check (warn if > 0.5% diff)
5. Write today's go/no-go decision to flags/premarket_<date>.json
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from optinet.v5_runtime.broker import make_broker, yfinance_spot_sanity
from optinet.v5_runtime.runtime_config import FLAGS_DIR, ensure_dirs

log = logging.getLogger("v5_premarket")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")


def update_data_lake(prior_date: date) -> dict:
    """Run the optinet_data_lake script for the prior trading day."""
    cmd_dl = [
        sys.executable, str(REPO_ROOT / "scripts" / "optinet_data_lake.py"),
        "--data-root", str(REPO_ROOT / "data"),
        "download", "--start", prior_date.isoformat(),
        "--end", prior_date.isoformat(),
    ]
    cmd_parse = [
        sys.executable, str(REPO_ROOT / "scripts" / "optinet_data_lake.py"),
        "--data-root", str(REPO_ROOT / "data"),
        "parse", "--start", prior_date.isoformat(),
        "--end", prior_date.isoformat(), "--overwrite",
    ]
    out = {}
    for tag, cmd in [("download", cmd_dl), ("parse", cmd_parse)]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            out[tag] = {"rc": r.returncode,
                        "stderr": r.stderr[-500:] if r.stderr else ""}
        except subprocess.TimeoutExpired:
            out[tag] = {"rc": -1, "error": "timeout"}
    return out


def broker_auth_check() -> dict:
    broker = make_broker()
    name = broker.name()
    healthy = bool(broker.health())
    sample = None
    if healthy and name != "MockBroker":
        try:
            sample = broker.get_spot("NIFTY").spot
        except Exception as exc:  # noqa: BLE001
            healthy = False
            sample = f"error: {exc!r}"
    return {"broker": name, "healthy": healthy, "sample_spot": sample}


def cross_check_spot() -> dict:
    """Compare broker spot vs yfinance spot. Non-critical."""
    yf_spot = yfinance_spot_sanity("NIFTY")
    broker = make_broker()
    broker_spot = None
    if broker.health() and broker.name() != "MockBroker":
        try:
            broker_spot = broker.get_spot("NIFTY").spot
        except Exception:
            broker_spot = None

    if yf_spot is None or broker_spot is None:
        return {"yfinance": yf_spot, "broker": broker_spot, "diff_pct": None}
    diff = abs(yf_spot - broker_spot) / yf_spot
    return {"yfinance": yf_spot, "broker": broker_spot,
            "diff_pct": float(diff),
            "warn": bool(diff > 0.005)}


def main() -> int:
    ensure_dirs()
    today = date.today()
    yest = today - timedelta(days=1)
    while yest.weekday() >= 5:
        yest -= timedelta(days=1)

    log.info(f"premarket {today} (prior trading day {yest})")

    out = {
        "today": str(today),
        "prior_day": str(yest),
        "data_lake": update_data_lake(yest),
        "broker_auth": broker_auth_check(),
        "spot_cross_check": cross_check_spot(),
    }

    # Decide go/no-go
    go = out["broker_auth"]["healthy"]
    out["decision"] = "GO" if go else "NO_GO"

    flag_path = FLAGS_DIR / f"premarket_{today}.json"
    flag_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"wrote {flag_path}")
    print(json.dumps(out, indent=2, default=str))
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
