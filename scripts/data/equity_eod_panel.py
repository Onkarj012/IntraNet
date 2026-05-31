#!/usr/bin/env python
"""Corporate-action-adjusted daily EOD panel for the equity universe.

Source priority: yfinance (auto_adjust=True → split+dividend adjusted) FIRST,
Kite as fallback only for symbols yfinance misses. Output is a wide MultiIndex
parquet {close, volume} compatible with scripts/research/factor_portfolio.py.

  build:  python scripts/data/equity_eod_panel.py --universe nifty500
  daily:  python scripts/data/equity_eod_panel.py --universe nifty500 --update
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))

from equity.universe import get_universe


def _clean(df: pd.DataFrame | None, min_rows: int) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df[(df["volume"] > 0) & (df["close"] > 0)]
    return df if len(df) >= min_rows else None


def fetch_yf(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    import yfinance as yf
    raw = yf.download(f"{symbol}.NS", start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    idx = pd.to_datetime(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw.index = idx.normalize()
    return raw[["close", "volume"]]


def fetch_kite(client, symbol: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        key = f"NSE:{symbol}"
        tok = int(client.kite.ltp([key])[key]["instrument_token"])
        df = client.historical_minute(
            tok, dt.datetime.combine(pd.Timestamp(start).date(), dt.time()),
            dt.datetime.combine(pd.Timestamp(end).date(), dt.time()),
            oi=False, interval="day",
        )
        if df.empty:
            return None
        df.index = pd.to_datetime(df["date"]).dt.tz_localize(None).normalize()
        return df[["close", "volume"]]
    except Exception as e:
        print(f"  kite fail {symbol}: {e}")
        return None


def _try_kite():
    try:
        from broker.kite_client import KiteClient
        import os
        c = KiteClient.from_env()
        tok = os.environ.get("KITE_ACCESS_TOKEN")
        if not tok:
            print("  kite fallback unavailable: no KITE_ACCESS_TOKEN")
            return None
        c.set_access_token(tok)
        c.kite.profile()  # validate
        return c
    except Exception as e:
        print(f"  kite fallback unavailable: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="nifty500")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--update", action="store_true", help="append recent bars to existing panel")
    p.add_argument("--kite-fallback", action="store_true", help="use Kite for symbols yfinance misses")
    p.add_argument("--limit", type=int, default=0, help="cap symbols (testing)")
    args = p.parse_args()

    out = Path(args.out) if args.out else _root / "cache/v8" / f"daily_panel_{args.universe}_adj.parquet"
    end = args.end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    syms = get_universe(args.universe)
    if args.limit:
        syms = syms[:args.limit]

    existing = None
    start = args.start
    min_rows = 260
    if args.update and out.exists():
        existing = pd.read_parquet(out)
        last = existing.index.max()
        start = (last - pd.Timedelta(days=10)).date().isoformat()
        min_rows = 1
        print(f"Update mode: existing through {last.date()}, refetching from {start}")

    closes, vols, yf_fail, kite = {}, {}, [], None
    for i, s in enumerate(syms):
        d = _clean(fetch_yf(s, start, end), min_rows)
        if d is None:
            yf_fail.append(s)
            if args.kite_fallback:
                kite = kite or _try_kite()
                if kite is not None:
                    d = _clean(fetch_kite(kite, s, start, end), min_rows)
        if d is not None:
            closes[s], vols[s] = d["close"], d["volume"]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(syms)} fetched ({len(closes)} ok, {len(yf_fail)} yf-miss)")
        time.sleep(0.05)

    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).reindex_like(close)
    panel = pd.concat({"close": close, "volume": vol}, axis=1)

    if existing is not None:
        if panel.empty:
            print("  Update mode: no new rows fetched — keeping existing panel unchanged.")
            panel = existing
        else:
            panel = pd.concat({"close": panel.xs("close", axis=1, level=0).combine_first(existing.xs("close", axis=1, level=0)),
                               "volume": panel.xs("volume", axis=1, level=0).combine_first(existing.xs("volume", axis=1, level=0))}, axis=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.sort_index().to_parquet(out)
    cl = panel.xs("close", axis=1, level=0)
    print(f"\nPanel: {cl.shape[1]} symbols, {cl.index.min().date()} → {cl.index.max().date()}, saved {out}")
    if yf_fail:
        print(f"yfinance missed {len(yf_fail)} symbols"
              f"{' (Kite filled where possible)' if args.kite_fallback else ' — rerun with --kite-fallback'}: "
              f"{yf_fail[:15]}{'...' if len(yf_fail) > 15 else ''}")


if __name__ == "__main__":
    main()
