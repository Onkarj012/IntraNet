#!/usr/bin/env python3
"""Repair / update the NIFTY 50 minute CSV from yfinance.

Fetches any missing bars since the last date in the file and appends them.
Also updates NIFTY daily and INDIA VIX daily from yfinance.

Usage:
  scripts/update_nifty_data.py              # append missing bars
  scripts/update_nifty_data.py --dry-run    # show what would be appended
  scripts/update_nifty_data.py --force-full # re-fetch last 60 days (dedup safe)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NIFTY_MIN_PATH   = PROJECT_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"
NIFTY_DAILY_PATH = PROJECT_ROOT / "data/indices/nifty_daily.csv"
VIX_DAILY_PATH   = PROJECT_ROOT / "data/nifty_intraday/INDIA VIX_day.csv"

NIFTY_TICKER = "^NSEI"
VIX_TICKER   = "^INDIAVIX"

# yfinance 1m data is only available for the last 30 days
YFINANCE_1M_LOOKBACK_DAYS = 29
YFINANCE_1M_CHUNK_DAYS = 7   # max safe chunk size for 1m requests


def fetch_nifty_minute(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch NIFTY 1-minute bars from yfinance, return in CSV schema."""
    ticker = yf.Ticker(NIFTY_TICKER)
    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=YFINANCE_1M_CHUNK_DAYS), end)
        try:
            hist = ticker.history(
                start=chunk_start.strftime("%Y-%m-%d"),
                end=(chunk_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1m",
            )
            if not hist.empty:
                frames.append(hist)
        except Exception as e:
            print(f"  warning: chunk {chunk_start.date()}→{chunk_end.date()}: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)

    # Filter to market hours 09:15-15:30
    t = df.index.time
    df = df[(t >= pd.Timestamp("09:15").time()) &
              (t <= pd.Timestamp("15:30").time())]

    out = pd.DataFrame({
        "date": df.index.strftime("%Y-%m-%d %H:%M:%S"),
        "open": df["Open"].round(2),
        "high": df["High"].round(2),
        "low": df["Low"].round(2),
        "close": df["Close"].round(2),
        "volume": df["Volume"].fillna(0).astype(int),
    })
    return out.reset_index(drop=True)


def fetch_nifty_daily(start: datetime) -> pd.DataFrame:
    """Fetch NIFTY daily bars, return in nifty_daily.csv schema."""
    hist = yf.Ticker(NIFTY_TICKER).history(
        start=start.strftime("%Y-%m-%d"), interval="1d")
    if hist.empty:
        return pd.DataFrame()
    hist.index = hist.index.tz_localize(None) if hist.index.tz is None else \
        hist.index.tz_convert("Asia/Kolkata").tz_localize(None)
    out = pd.DataFrame({
        "date": hist.index.strftime("%Y-%m-%d 00:00:00+05:30"),
        "open": hist["Open"].round(2),
        "high": hist["High"].round(2),
        "low": hist["Low"].round(2),
        "close": hist["Close"].round(2),
        "volume": hist["Volume"].fillna(0).astype(int),
        "symbol": "NIFTY",
    })
    return out.reset_index(drop=True)


def fetch_vix_daily(start: datetime) -> pd.DataFrame:
    """Fetch INDIA VIX daily bars, return in INDIA VIX_day.csv schema."""
    hist = yf.Ticker(VIX_TICKER).history(
        start=start.strftime("%Y-%m-%d"), interval="1d")
    if hist.empty:
        return pd.DataFrame()
    hist.index = hist.index.tz_localize(None) if hist.index.tz is None else \
        hist.index.tz_convert("Asia/Kolkata").tz_localize(None)
    out = pd.DataFrame({
        "date": hist.index.strftime("%Y-%m-%d 00:00:00"),
        "open": hist["Open"].round(2),
        "high": hist["High"].round(2),
        "low": hist["Low"].round(2),
        "close": hist["Close"].round(2),
        "volume": 0,
    })
    return out.reset_index(drop=True)


def update_minute_csv(dry_run: bool, force_full: bool) -> int:
    """Append missing minute bars. Returns count of new rows added."""
    existing = pd.read_csv(NIFTY_MIN_PATH)
    existing["datetime"] = pd.to_datetime(existing["date"])
    last_dt = existing["datetime"].max()
    print(f"  NIFTY minute: last bar = {last_dt}")

    if force_full:
        fetch_start = datetime.now() - timedelta(days=YFINANCE_1M_LOOKBACK_DAYS)
    else:
        fetch_start = last_dt.to_pydatetime()

    fetch_end = datetime.now()
    print(f"  fetching {fetch_start.date()} → {fetch_end.date()} …")

    new_bars = fetch_nifty_minute(fetch_start, fetch_end)
    if new_bars.empty:
        print("  no new bars returned from yfinance")
        return 0

    new_bars["datetime"] = pd.to_datetime(new_bars["date"])
    # Only keep bars strictly after the last existing bar
    new_bars = new_bars[new_bars["datetime"] > last_dt].copy()
    if new_bars.empty:
        print("  no new bars after last existing timestamp")
        return 0

    # Validate: check last day completeness
    last_new_day = new_bars["datetime"].dt.normalize().max()
    last_day_bars = new_bars[new_bars["datetime"].dt.normalize() == last_new_day]
    last_bar_time = last_day_bars["datetime"].max().time()
    n_bars = len(last_day_bars)
    complete = last_bar_time >= pd.Timestamp("15:29").time() and n_bars >= 370
    print(f"  new bars: {len(new_bars)} rows, latest day={last_new_day.date()}, "
          f"bars={n_bars}, last={last_bar_time}, complete={complete}")

    if dry_run:
        print(f"  [dry-run] would append {len(new_bars)} rows")
        print(new_bars.tail(3).to_string(index=False))
        return len(new_bars)

    # Append (drop datetime helper col)
    new_bars.drop(columns=["datetime"]).to_csv(
        NIFTY_MIN_PATH, mode="a", header=False, index=False)
    print(f"  ✓ appended {len(new_bars)} rows → {NIFTY_MIN_PATH}")
    return len(new_bars)


def update_daily_csv(path: Path, fetch_fn, label: str, dry_run: bool) -> int:
    """Generic daily CSV updater."""
    existing = pd.read_csv(path)
    # Normalize date column (handle timezone suffixes)
    date_col = existing.columns[0]
    existing["_dt"] = pd.to_datetime(existing[date_col], utc=True).dt.tz_convert(None)
    last_dt = existing["_dt"].max()
    print(f"  {label}: last bar = {last_dt.date()}")

    fetch_start = last_dt.to_pydatetime() - timedelta(days=3)  # small overlap for safety
    new_rows = fetch_fn(fetch_start)
    if new_rows.empty:
        print(f"  {label}: no new rows from yfinance")
        return 0

    new_rows["_dt"] = pd.to_datetime(new_rows[new_rows.columns[0]], utc=True,
                                       errors="coerce").dt.tz_convert(None)
    new_rows = new_rows[new_rows["_dt"] > last_dt].copy()
    if new_rows.empty:
        print(f"  {label}: no new rows after {last_dt.date()}")
        return 0

    print(f"  {label}: {len(new_rows)} new rows, latest={new_rows['_dt'].max().date()}")
    if dry_run:
        print(f"  [dry-run] would append {len(new_rows)} rows")
        return len(new_rows)

    new_rows.drop(columns=["_dt"]).to_csv(path, mode="a", header=False, index=False)
    print(f"  ✓ appended {len(new_rows)} rows → {path}")
    return len(new_rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-full", action="store_true",
                   help="Re-fetch last 29 days of minute data (dedup-safe)")
    p.add_argument("--minute-only", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print("  NIFTY data update")
    print("=" * 70)

    n_min = update_minute_csv(args.dry_run, args.force_full)

    if not args.minute_only:
        n_daily = update_daily_csv(
            NIFTY_DAILY_PATH, fetch_nifty_daily, "NIFTY daily", args.dry_run)
        n_vix = update_daily_csv(
            VIX_DAILY_PATH, fetch_vix_daily, "INDIA VIX daily", args.dry_run)
    else:
        n_daily = n_vix = 0

    print(f"\n  summary: minute={n_min} new rows, "
          f"daily={n_daily} new rows, vix={n_vix} new rows")

    if not args.dry_run:
        # Quick verification
        nifty = pd.read_csv(NIFTY_MIN_PATH)
        nifty["datetime"] = pd.to_datetime(nifty["date"])
        last = nifty["datetime"].max()
        last_day_bars = nifty[nifty["datetime"].dt.normalize() == last.normalize()]
        last_time = last_day_bars["datetime"].max().time()
        n = len(last_day_bars)
        complete = last_time >= pd.Timestamp("15:29").time() and n >= 370
        print(f"\n  post-update: latest={last.date()}, "
              f"last_day_bars={n}, last_bar={last_time}, "
              f"session_complete={'YES' if complete else 'NO'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
