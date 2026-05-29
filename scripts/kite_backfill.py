#!/usr/bin/env python3
"""Backfill 3 months of NIFTY FUT + spot minute data from Zerodha.

Fetches the current front-month futures contract (and optionally the
next-month contract) going back up to 60 days per request, chunked
automatically. Saves in the same schema as existing nifty_fut CSVs:

  data/option_data/nifty_data/nifty_fut/YYYY/M/nifty_fut_DD_MM_YYYY.csv
  data/option_data/nifty_data/nifty_spot/YYYY/M/nifty_spotDD_MM_YYYY.csv

Usage:
    # Backfill last 60 days (one contract window)
    .venv/bin/python scripts/kite_backfill.py

    # Backfill specific date range
    .venv/bin/python scripts/kite_backfill.py --from 2026-03-01 --to 2026-05-29

    # Dry-run: show what would be fetched
    .venv/bin/python scripts/kite_backfill.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from broker.kite_client import KiteClient

REPO_ROOT = Path(__file__).resolve().parents[1]
FUT_ROOT  = REPO_ROOT / "data/option_data/nifty_data/nifty_fut"
SPOT_ROOT = REPO_ROOT / "data/option_data/nifty_data/nifty_spot"
MARKET_OPEN  = "09:15:00"
MARKET_CLOSE = "15:30:00"


def _fut_path(d: datetime) -> Path:
    return FUT_ROOT / str(d.year) / str(d.month) / f"nifty_fut_{d.day:02d}_{d.month:02d}_{d.year}.csv"


def _spot_path(d: datetime) -> Path:
    return SPOT_ROOT / str(d.year) / str(d.month) / f"nifty_spot{d.day:02d}_{d.month:02d}_{d.year}.csv"


def _to_fut_csv(df: pd.DataFrame, symbol: str = "NIFTY-I") -> pd.DataFrame:
    """Convert Kite candle DataFrame to nifty_fut CSV schema."""
    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "time": df["date"].dt.strftime("%H:%M:%S"),
        "symbol": symbol,
        "open": df["open"].round(4),
        "high": df["high"].round(4),
        "low": df["low"].round(4),
        "close": df["close"].round(4),
        "oi": df["oi"].fillna(0).astype(int),
        "volume": df["volume"].fillna(0).astype(int),
    })
    return out


def _to_spot_csv(df: pd.DataFrame, symbol: str = "NIFTY") -> pd.DataFrame:
    """Convert Kite candle DataFrame to nifty_spot CSV schema."""
    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "time": df["date"].dt.strftime("%H:%M:%S"),
        "symbol": symbol,
        "open": df["open"].round(2),
        "high": df["high"].round(2),
        "low": df["low"].round(2),
        "close": df["close"].round(2),
    })
    return out


def save_by_day(df: pd.DataFrame, path_fn, to_schema_fn, label: str,
                dry_run: bool, overwrite: bool) -> int:
    """Split a multi-day DataFrame into per-day files."""
    df["_date"] = df["date"].dt.normalize()
    saved = 0
    for day, group in df.groupby("_date"):
        p = path_fn(day)
        if p.exists() and not overwrite:
            print(f"  skip {label} {day.date()} (exists, use --overwrite)")
            continue
        if dry_run:
            print(f"  [dry-run] would save {len(group)} rows → {p}")
            saved += 1
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        to_schema_fn(group.drop(columns=["_date"])).to_csv(p, index=False)
        print(f"  saved {label} {day.date()}: {len(group)} bars → {p.name}")
        saved += 1
    return saved


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", default=None,
                   help="Start date YYYY-MM-DD (default: 60 days ago)")
    p.add_argument("--to", dest="to_date", default=None,
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-day files")
    p.add_argument("--spot-only", action="store_true")
    p.add_argument("--fut-only", action="store_true")
    args = p.parse_args()

    to_dt   = datetime.strptime(args.to_date, "%Y-%m-%d") if args.to_date \
              else datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date \
              else to_dt - timedelta(days=60)
    from_dt = from_dt.replace(hour=9, minute=15, second=0, microsecond=0)

    print(f"Kite backfill: {from_dt.date()} → {to_dt.date()}")
    print(f"  dry_run={args.dry_run}  overwrite={args.overwrite}")

    kite = KiteClient.from_env()
    # Use existing access token if available, else login
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if token:
        kite.set_access_token(token)
    else:
        kite.login()

    total_fut = total_spot = 0

    # ── Futures ────────────────────────────────────────────────────────────
    if not args.spot_only:
        fut_token, expiry = kite.get_fut_token("NIFTY")
        print(f"\n  NIFTY FUT token={fut_token} expiry={expiry}")
        df_fut = kite.historical_minute(fut_token, from_dt, to_dt, oi=True)
        if df_fut.empty:
            print("  no futures data returned")
        else:
            # Filter to market hours
            t = df_fut["date"].dt.time
            df_fut = df_fut[
                (t >= pd.Timestamp(MARKET_OPEN).time()) &
                (t <= pd.Timestamp(MARKET_CLOSE).time())
            ]
            print(f"  fetched {len(df_fut)} fut bars across "
                  f"{df_fut['date'].dt.normalize().nunique()} days")
            total_fut = save_by_day(df_fut, _fut_path, _to_fut_csv,
                                     "FUT", args.dry_run, args.overwrite)

    # ── Spot ───────────────────────────────────────────────────────────────
    if not args.fut_only:
        spot_token = kite.get_spot_token("NIFTY")
        print(f"\n  NIFTY spot token={spot_token}")
        df_spot = kite.historical_minute(spot_token, from_dt, to_dt, oi=False)
        if df_spot.empty:
            print("  no spot data returned")
        else:
            t = df_spot["date"].dt.time
            df_spot = df_spot[
                (t >= pd.Timestamp(MARKET_OPEN).time()) &
                (t <= pd.Timestamp(MARKET_CLOSE).time())
            ]
            print(f"  fetched {len(df_spot)} spot bars across "
                  f"{df_spot['date'].dt.normalize().nunique()} days")
            total_spot = save_by_day(df_spot, _spot_path, _to_spot_csv,
                                      "SPOT", args.dry_run, args.overwrite)

    print(f"\n  done: {total_fut} fut files, {total_spot} spot files saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
