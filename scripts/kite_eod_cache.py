#!/usr/bin/env python3
"""Daily EOD data cache — run at 15:35 IST after market close.

Fetches today's complete session (09:15–15:30) for:
  - NIFTY front-month FUT (1-min + OI)
  - NIFTY spot index (1-min)
  - INDIA VIX (daily)

Saves per-day files in the same schema as existing historical data.
Also handles pre-expiry full-contract fetch (run on expiry day to
capture the full 3-month history before the token dies).

Usage:
    # Cache today
    .venv/bin/python scripts/kite_eod_cache.py

    # Cache a specific past date
    .venv/bin/python scripts/kite_eod_cache.py --date 2026-05-28

    # Pre-expiry: fetch full contract history before token expires
    .venv/bin/python scripts/kite_eod_cache.py --full-contract
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from broker.kite_client import KiteClient

REPO_ROOT = Path(__file__).resolve().parents[1]
FUT_ROOT  = REPO_ROOT / "data/option_data/nifty_data/nifty_fut"
SPOT_ROOT = REPO_ROOT / "data/option_data/nifty_data/nifty_spot"
VIX_PATH  = REPO_ROOT / "data/nifty_intraday/INDIA VIX_day.csv"
NIFTY_MIN = REPO_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"


def _fut_path(d: date) -> Path:
    return FUT_ROOT / str(d.year) / str(d.month) / f"nifty_fut_{d.day:02d}_{d.month:02d}_{d.year}.csv"


def _spot_path(d: date) -> Path:
    return SPOT_ROOT / str(d.year) / str(d.month) / f"nifty_spot{d.day:02d}_{d.month:02d}_{d.year}.csv"


def _save_fut(df: pd.DataFrame, target_date: date) -> None:
    p = _fut_path(target_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "time": df["date"].dt.strftime("%H:%M:%S"),
        "symbol": "NIFTY-I",
        "open": df["open"].round(4),
        "high": df["high"].round(4),
        "low": df["low"].round(4),
        "close": df["close"].round(4),
        "oi": df["oi"].fillna(0).astype(int),
        "volume": df["volume"].fillna(0).astype(int),
    })
    out.to_csv(p, index=False)
    print(f"  saved FUT {target_date}: {len(out)} bars → {p.name}")


def _save_spot(df: pd.DataFrame, target_date: date) -> None:
    p = _spot_path(target_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "time": df["date"].dt.strftime("%H:%M:%S"),
        "symbol": "NIFTY",
        "open": df["open"].round(2),
        "high": df["high"].round(2),
        "low": df["low"].round(2),
        "close": df["close"].round(2),
    })
    out.to_csv(p, index=False)
    print(f"  saved SPOT {target_date}: {len(out)} bars → {p.name}")


def _append_nifty_minute(df: pd.DataFrame) -> int:
    """Also append to the main NIFTY 50_minute.csv used by paper trading."""
    if not NIFTY_MIN.exists():
        return 0
    existing = pd.read_csv(NIFTY_MIN, usecols=["date"])
    existing["dt"] = pd.to_datetime(existing["date"])
    last_dt = existing["dt"].max()

    new = df[df["date"] > last_dt].copy()
    if new.empty:
        return 0
    out = pd.DataFrame({
        "date": new["date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": new["open"].round(2),
        "high": new["high"].round(2),
        "low": new["low"].round(2),
        "close": new["close"].round(2),
        "volume": new["volume"].fillna(0).astype(int),
    })
    out.to_csv(NIFTY_MIN, mode="a", header=False, index=False)
    print(f"  appended {len(out)} rows to NIFTY 50_minute.csv")
    return len(out)


def _update_vix(kite: KiteClient) -> None:
    """Append today's VIX daily bar."""
    try:
        vix_token = kite.get_spot_token("INDIA VIX")
        today = date.today()
        from_dt = datetime.combine(today - timedelta(days=3), datetime.min.time())
        to_dt   = datetime.combine(today, datetime.max.time())
        df = kite.historical_minute(vix_token, from_dt, to_dt, oi=False, interval="day")
        if df.empty:
            return
        existing = pd.read_csv(VIX_PATH) if VIX_PATH.exists() else pd.DataFrame()
        if not existing.empty:
            last = pd.to_datetime(existing.iloc[-1, 0]).date()
            df = df[df["date"].dt.date > last]
        if df.empty:
            return
        out = pd.DataFrame({
            "date": df["date"].dt.strftime("%Y-%m-%d 00:00:00"),
            "open": df["open"].round(2),
            "high": df["high"].round(2),
            "low": df["low"].round(2),
            "close": df["close"].round(2),
            "volume": 0,
        })
        out.to_csv(VIX_PATH, mode="a", header=False, index=False)
        print(f"  appended {len(out)} VIX daily rows")
    except Exception as e:
        print(f"  VIX update skipped: {e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--full-contract", action="store_true",
                   help="Fetch full 3-month history of current contract (run on expiry day)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    target_date = (datetime.strptime(args.date, "%Y-%m-%d").date()
                   if args.date else date.today())

    print(f"Kite EOD cache: {target_date}")

    kite = KiteClient.from_env()
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if token:
        kite.set_access_token(token)
    else:
        kite.login()

    fut_token, expiry = kite.get_fut_token("NIFTY")
    spot_token = kite.get_spot_token("NIFTY")

    if args.full_contract:
        # Fetch full contract history (up to 60 days back from today)
        from_dt = datetime.combine(target_date - timedelta(days=60),
                                    datetime.min.time()).replace(hour=9, minute=15)
        to_dt   = datetime.combine(target_date, datetime.min.time()).replace(hour=15, minute=30)
        print(f"  full-contract mode: {from_dt.date()} → {to_dt.date()}")
    else:
        from_dt = datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=15)
        to_dt   = datetime.combine(target_date, datetime.min.time()).replace(hour=15, minute=30)

    # Fetch futures
    df_fut = kite.historical_minute(fut_token, from_dt, to_dt, oi=True)
    if not df_fut.empty:
        t = df_fut["date"].dt.time
        df_fut = df_fut[
            (t >= pd.Timestamp("09:15:00").time()) &
            (t <= pd.Timestamp("15:30:00").time())
        ]
        if args.full_contract:
            # Save each day separately
            for day, grp in df_fut.groupby(df_fut["date"].dt.normalize()):
                d = day.date()
                if not _fut_path(d).exists() or args.overwrite:
                    _save_fut(grp, d)
        else:
            if not _fut_path(target_date).exists() or args.overwrite:
                _save_fut(df_fut, target_date)

    # Fetch spot
    df_spot = kite.historical_minute(spot_token, from_dt, to_dt, oi=False)
    if not df_spot.empty:
        t = df_spot["date"].dt.time
        df_spot = df_spot[
            (t >= pd.Timestamp("09:15:00").time()) &
            (t <= pd.Timestamp("15:30:00").time())
        ]
        if args.full_contract:
            for day, grp in df_spot.groupby(df_spot["date"].dt.normalize()):
                d = day.date()
                if not _spot_path(d).exists() or args.overwrite:
                    _save_spot(grp, d)
        else:
            if not _spot_path(target_date).exists() or args.overwrite:
                _save_spot(df_spot, target_date)
        # Also keep the main NIFTY 50_minute.csv up to date
        _append_nifty_minute(df_spot)

    # Update VIX daily
    _update_vix(kite)

    print("  done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
