#!/usr/bin/env python
"""V5 v1 health check — pre-flight gate before any trading.

Exits 0 if all checks pass, non-zero otherwise. With --strict, prints a
detailed report and creates flags/halt_today.flag on failure.

Checks:
 1. Required model files exist and load
 2. Chain feature partition is recent
 3. Broker API health (auth + sample quote)
 4. yfinance ^NSEI sanity (non-critical)
 5. Disk space
 6. Today's NIFTY weekly dte is in {2, 3} (preferred-launch hint)
 7. India VIX in [8, 50] (yfinance, non-critical)
 8. No halt flags present
 9. Yesterday's ledger reconciled (or first-day exemption)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure repo src is importable when run via cron
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import lightgbm as lgb
import pandas as pd

from optinet.v5_runtime.broker import make_broker, yfinance_spot_sanity
from optinet.v5_runtime import ledger as ld
from optinet.v5_runtime.runtime_config import (
    GATE_DIR, VOL_MODEL_PATH, FLAGS_DIR, ensure_dirs,
    any_halt_active, next_weekly_expiry, days_to_expiry,
)

CHAIN_CACHE = REPO_ROOT / "cache" / "optinet_v4" / "chain_features" / "index=NIFTY"
DISK_MIN_GB = 5.0


class CheckResult:
    def __init__(self, name: str, passed: bool, message: str,
                 critical: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.critical = critical

    def line(self) -> str:
        sym = "✓" if self.passed else ("✗" if self.critical else "⚠")
        crit = "" if self.critical else "  (non-critical)"
        return f"  [{sym}] {self.name}: {self.message}{crit}"


def check_models() -> CheckResult:
    g2 = GATE_DIR / "gate_dte2.lgb"
    g3 = GATE_DIR / "gate_dte3.lgb"
    missing = [str(p) for p in (g2, g3, VOL_MODEL_PATH) if not p.exists()]
    if missing:
        return CheckResult("models", False, f"missing: {missing}")
    try:
        for p in (g2, g3, VOL_MODEL_PATH):
            lgb.Booster(model_file=str(p))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("models", False, f"load failed: {exc!r}")
    return CheckResult("models", True, "gate_dte2, gate_dte3, vol model load OK")


def check_chain_cache() -> CheckResult:
    if not CHAIN_CACHE.exists():
        return CheckResult("chain_cache", False, f"missing: {CHAIN_CACHE}")
    parquets = sorted(CHAIN_CACHE.rglob("*.parquet"))
    if not parquets:
        return CheckResult("chain_cache", False, "no parquet files found")
    newest = max(parquets, key=lambda p: p.stat().st_mtime)
    age_days = (datetime.now().timestamp() - newest.stat().st_mtime) / 86400
    if age_days > 14:
        return CheckResult("chain_cache", False,
                            f"newest partition is {age_days:.1f} days old")
    return CheckResult("chain_cache", True,
                        f"newest partition {newest.name}, {age_days:.1f}d old")


def check_broker() -> CheckResult:
    broker = make_broker()
    if not broker.health():
        critical = (broker.name() != "MockBroker")
        return CheckResult("broker", False,
                            f"{broker.name()}: health() returned False",
                            critical=critical)
    # Try a sample quote — for MockBroker this needs simulate_date
    if broker.name() == "MockBroker":
        return CheckResult("broker", True,
                            "MockBroker connected (offline-test mode)",
                            critical=False)
    try:
        # For live brokers a real quote attempt
        broker.get_spot("NIFTY")
        return CheckResult("broker", True, f"{broker.name()} sample quote OK")
    except NotImplementedError as exc:
        return CheckResult("broker", False,
                            f"{broker.name()} not yet implemented: {exc}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("broker", False, f"sample quote failed: {exc!r}")


def check_yfinance_sanity() -> CheckResult:
    spot = yfinance_spot_sanity("NIFTY")
    if spot is None:
        return CheckResult("yfinance_sanity", False,
                            "yfinance unreachable (non-critical)",
                            critical=False)
    if not (5000 < spot < 50000):
        return CheckResult("yfinance_sanity", False,
                            f"NIFTY spot looks bogus: {spot}",
                            critical=False)
    return CheckResult("yfinance_sanity", True,
                        f"NIFTY ~ {spot:.1f} (sanity)",
                        critical=False)


def check_disk() -> CheckResult:
    free_gb = shutil.disk_usage(str(REPO_ROOT)).free / 1e9
    if free_gb < DISK_MIN_GB:
        return CheckResult("disk_space", False, f"only {free_gb:.1f} GB free")
    return CheckResult("disk_space", True, f"{free_gb:.1f} GB free")


def check_dte(today: date) -> CheckResult:
    expiry = next_weekly_expiry("NIFTY", today)
    dte = days_to_expiry(today, expiry)
    if dte in (2, 3):
        return CheckResult("dte_today", True, f"NIFTY dte={dte} (eligible)")
    return CheckResult("dte_today", False,
                        f"NIFTY dte={dte} → no eligible bucket today",
                        critical=False)


def check_vix() -> CheckResult:
    try:
        import yfinance as yf
        h = yf.Ticker("^INDIAVIX").history(period="2d", interval="1d")
        if h.empty:
            return CheckResult("vix_sanity", False,
                                "no VIX data (non-critical)",
                                critical=False)
        vix = float(h["Close"].iloc[-1])
        if 8 <= vix <= 50:
            return CheckResult("vix_sanity", True, f"VIX={vix:.2f}",
                                critical=False)
        return CheckResult("vix_sanity", False,
                            f"VIX={vix:.2f} outside [8, 50]",
                            critical=False)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("vix_sanity", False, f"yfinance failed: {exc!r}",
                            critical=False)


def check_halt_flags() -> CheckResult:
    msg = any_halt_active()
    if msg:
        return CheckResult("halt_flags", False, msg)
    return CheckResult("halt_flags", True, "no halt flags")


def check_yesterday_reconciled(today: date) -> CheckResult:
    df = ld.load_ledger()
    if df.empty:
        return CheckResult("ledger_reconciled", True, "first-day, ledger empty",
                            critical=False)
    yest = today - timedelta(days=1)
    while yest.weekday() >= 5:  # back to last weekday
        yest -= timedelta(days=1)
    yt = ld.trades_on_date(yest)
    if yt.empty:
        return CheckResult("ledger_reconciled", True,
                            f"no trades on {yest} (skip)",
                            critical=False)
    open_or_closed = yt[~yt["status"].isin(["RECONCILED"])]
    if len(open_or_closed):
        return CheckResult("ledger_reconciled", False,
                            f"{len(open_or_closed)} unreconciled trades from {yest}")
    return CheckResult("ledger_reconciled", True,
                        f"all {len(yt)} trades from {yest} reconciled")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Create halt_today.flag on critical failure")
    ap.add_argument("--today", default=None, help="Override 'today' (YYYY-MM-DD)")
    args = ap.parse_args()
    today = (date.fromisoformat(args.today) if args.today else date.today())

    ensure_dirs()

    print("=" * 72)
    print(f"V5 v1 health check — {today}")
    print("=" * 72)

    checks = [
        check_models(),
        check_chain_cache(),
        check_broker(),
        check_disk(),
        check_dte(today),
        check_vix(),
        check_yfinance_sanity(),
        check_halt_flags(),
        check_yesterday_reconciled(today),
    ]

    for c in checks:
        print(c.line())

    failed_critical = [c for c in checks if not c.passed and c.critical]
    failed_noncritical = [c for c in checks if not c.passed and not c.critical]

    print("-" * 72)
    print(f"Critical failures: {len(failed_critical)}  "
          f"Non-critical warnings: {len(failed_noncritical)}")

    if failed_critical:
        if args.strict:
            (FLAGS_DIR / "halt_today.flag").touch()
            print(f"  → wrote {FLAGS_DIR / 'halt_today.flag'}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
