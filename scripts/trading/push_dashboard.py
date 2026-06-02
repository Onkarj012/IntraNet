#!/usr/bin/env python3
"""Push dashboard artifacts to Convex so the hosted dashboard reads live data.

Reads from repo .env, dashboard/.env.local, or environment:
  CONVEX_HTTP_URL              https://<deployment>.convex.site
  NEXT_PUBLIC_CONVEX_SITE_URL  same URL (fallback from dashboard/.env.local)
  DASHBOARD_PUSH_SECRET        shared secret (must equal Convex PUSH_SECRET)

Safe no-op (exit 0) when CONVEX_HTTP_URL is unset, so it can sit at the tail of
the daily cron chain without affecting local-only setups.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STATIC = [
    "results/router_v0/paper_trading_ledger.csv",
    "results/router_v0/tier1_validation.json",
    "results/router_v0/forward_walk_summary.json",
    "results/router_v0/feature_importance_long.json",
    "results/equity/paper_ledger.csv",
    "results/daily_run_status.json",
    "results/cron_status.json",
    "results/recommendations.json",
]


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            key = k.strip()
            if override:
                os.environ[key] = v
            else:
                os.environ.setdefault(key, v)


def load_env() -> None:
    """Repo .env first, then dashboard/.env.local (overrides Convex/push vars)."""
    _load_env_file(ROOT / ".env")
    _load_env_file(ROOT / "dashboard" / ".env.local", override=True)
    # Next/Vercel name; push + source.ts use CONVEX_HTTP_URL (.convex.site)
    site = os.environ.get("NEXT_PUBLIC_CONVEX_SITE_URL", "").rstrip("/")
    if site:
        os.environ.setdefault("CONVEX_HTTP_URL", site)


def latest(pattern: str) -> str | None:
    files = sorted(glob.glob(str(ROOT / pattern)))
    return Path(files[-1]).read_text() if files else None


def main() -> int:
    load_env()
    base = os.environ.get("CONVEX_HTTP_URL", "").rstrip("/")
    secret = (os.environ.get("DASHBOARD_PUSH_SECRET") or os.environ.get("PUSH_SECRET") or "").strip()
    if not base:
        print("  CONVEX_HTTP_URL unset — skipping dashboard push")
        return 0
    if not secret:
        print("  DASHBOARD_PUSH_SECRET unset — cannot push", file=sys.stderr)
        return 1

    items: dict[str, str] = {}
    for rel in STATIC:
        f = ROOT / rel
        if f.exists():
            items[rel] = f.read_text()
    for key, pattern in [
        ("@health_latest", "logs/health_check_*.json"),
        ("@picks_latest", "results/equity/picks/picks_*.json"),
        ("@ops_latest", "logs/paper_trade_ops_*.json"),
    ]:
        text = latest(pattern)
        if text:
            items[key] = text
    items["@halts"] = json.dumps({
        "futures": (ROOT / "results/router_v0/PAPER_TRADING_HALTED").exists(),
        "equity": (ROOT / "results/equity/EQUITY_PAPER_HALTED").exists(),
    })

    ok = 0
    for key, content in items.items():
        body = json.dumps({"key": key, "content": content}).encode()
        req = urllib.request.Request(
            f"{base}/push", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {secret}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
            ok += 1
        except Exception as e:  # noqa: BLE001 — push is best-effort
            print(f"  push failed [{key}]: {e}", file=sys.stderr)
    print(f"  pushed {ok}/{len(items)} artifacts → {base}")
    return 0 if ok == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
