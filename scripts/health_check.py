#!/usr/bin/env python3
"""Daily health check — run anytime to verify the pipeline is healthy.

Checks:
  1. NIFTY FUT data freshness (today's file exists)
  2. NIFTY spot data freshness
  3. NIFTY 50 minute CSV freshness
  4. Paper trading ledger — today's run present
  5. Model file integrity
  6. Kite access token validity

Exit codes:
  0 = all green
  1 = one or more checks failed

Usage:
    .venv/bin/python scripts/health_check.py
    .venv/bin/python scripts/health_check.py --json   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NIFTY_MIN_PATH  = PROJECT_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"
FUT_ROOT        = PROJECT_ROOT / "data/option_data/nifty_data/nifty_fut"
SPOT_ROOT       = PROJECT_ROOT / "data/option_data/nifty_data/nifty_spot"
MODEL_PATH      = PROJECT_ROOT / "models/router_v0/futures/final_long.lgb"
LEDGER_PATH     = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"
LOG_DIR         = PROJECT_ROOT / "logs"


def _latest_weekday(d: date | None = None) -> date:
    from datetime import timedelta
    t = datetime.now().date()
    while t.weekday() >= 5:
        t -= timedelta(days=1)
    return t


def check_nifty_minute() -> dict:
    if not NIFTY_MIN_PATH.exists():
        return {"name": "nifty_minute", "ok": False, "msg": "file missing"}
    df = pd.read_csv(NIFTY_MIN_PATH, usecols=["date"])
    last = pd.to_datetime(df["date"]).max()
    expected = _latest_weekday()
    ok = last.date() >= expected
    return {"name": "nifty_minute", "ok": ok,
            "last": str(last.date()), "expected": str(expected),
            "msg": "OK" if ok else f"stale: last={last.date()} expected>={expected}"}


def check_fut_today() -> dict:
    today = _latest_weekday()
    p = FUT_ROOT / str(today.year) / str(today.month) / \
        f"nifty_fut_{today.day:02d}_{today.month:02d}_{today.year}.csv"
    ok = p.exists()
    return {"name": "fut_today", "ok": ok,
            "path": str(p.name),
            "msg": "OK" if ok else f"missing: {p.name}"}


def check_spot_today() -> dict:
    today = _latest_weekday()
    p = SPOT_ROOT / str(today.year) / str(today.month) / \
        f"nifty_spot{today.day:02d}_{today.month:02d}_{today.year}.csv"
    ok = p.exists()
    return {"name": "spot_today", "ok": ok,
            "path": str(p.name),
            "msg": "OK" if ok else f"missing: {p.name}"}


def check_model() -> dict:
    if not MODEL_PATH.exists():
        return {"name": "model", "ok": False, "msg": "model file missing"}
    size_kb = MODEL_PATH.stat().st_size / 1e3
    ok = size_kb > 5
    return {"name": "model", "ok": ok,
            "size_kb": round(size_kb, 1),
            "msg": f"OK ({size_kb:.0f} KB)" if ok else "model file too small"}


def check_paper_run() -> dict:
    if not LEDGER_PATH.exists():
        return {"name": "paper_run", "ok": False, "msg": "ledger missing"}
    df = pd.read_csv(LEDGER_PATH, usecols=["trade_date", "source"])
    today = str(_latest_weekday())
    live_rows = df[(df["trade_date"] == today) & (df["source"] == "paper")]
    ok = len(live_rows) > 0
    last_date = df[df["source"] == "paper"]["trade_date"].max()
    return {"name": "paper_run", "ok": ok,
            "today": today, "last_run": str(last_date),
            "trades_today": len(live_rows),
            "msg": f"OK ({len(live_rows)} trades)" if ok else f"no paper run for {today} (last: {last_date})"}


def check_kite_token() -> dict:
    from broker.kite_client import _load_env
    _load_env()
    token = os.environ.get("KITE_ACCESS_TOKEN", "")
    if not token:
        return {"name": "kite_token", "ok": False, "msg": "no token in .env"}
    try:
        from kiteconnect import KiteConnect
        api_key = os.environ.get("KITE_API_KEY", "")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        profile = kite.profile()
        return {"name": "kite_token", "ok": True,
                "user": profile.get("user_name", "?"),
                "msg": f"OK (logged in as {profile.get('user_name', '?')})"}
    except Exception as e:
        return {"name": "kite_token", "ok": False, "msg": f"token invalid: {e}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--skip-kite", action="store_true", help="Skip Kite token check")
    args = p.parse_args()

    checks = [
        check_nifty_minute(),
        check_fut_today(),
        check_spot_today(),
        check_model(),
        check_paper_run(),
    ]
    if not args.skip_kite:
        checks.append(check_kite_token())

    all_ok = all(c["ok"] for c in checks)
    timestamp = datetime.now().isoformat(timespec="seconds")

    result = {"timestamp": timestamp, "all_ok": all_ok, "checks": checks}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  Health Check — {timestamp}")
        print(f"{'='*60}")
        for c in checks:
            icon = "✅" if c["ok"] else "❌"
            print(f"  {icon}  {c['name']:20s}  {c['msg']}")
        print(f"{'='*60}")
        print(f"  {'ALL GREEN' if all_ok else 'FAILURES DETECTED'}")
        print(f"{'='*60}\n")

    # Write timestamped log
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"health_check_{datetime.now().strftime('%Y%m%d')}.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
