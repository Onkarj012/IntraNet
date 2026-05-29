#!/usr/bin/env python3
"""Daily Zerodha Kite login — run at 08:50 IST before market open.

Reads credentials from .env, performs headless TOTP login,
writes fresh access_token back to .env.

Usage:
    .venv/bin/python scripts/kite_login.py
    .venv/bin/python scripts/kite_login.py --check   # verify token only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from broker.kite_client import KiteClient


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="Verify existing token only")
    args = p.parse_args()

    kite = KiteClient.from_env()

    if args.check:
        import os
        token = os.environ.get("KITE_ACCESS_TOKEN", "")
        if not token:
            print("  no KITE_ACCESS_TOKEN in .env")
            return 1
        kite.set_access_token(token)
        try:
            profile = kite.kite.profile()
            print(f"  token valid — logged in as {profile['user_name']}")
            return 0
        except Exception as e:
            print(f"  token invalid: {e}")
            return 1

    token = kite.login()
    print(f"  access_token: {token[:8]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
