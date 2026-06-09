#!/usr/bin/env python
"""Build dense historical sentiment cache from all available sources.

Sources (combined + deduplicated):
  1. data/sentiment/combined_sentiment_latest.csv  (Google News 2015-2026)
  2. data/sentiment/gdelt_india_2022_2026.csv
  3. Live Google News RSS backfill for symbols with sparse coverage

Output: data/sentiment/sentiment_cache_full.parquet
Schema:  symbol, timestamp, headline, source, score, industry, premarket_trade_date

Usage:
    python scripts/data/build_sentiment_cache.py
    python scripts/data/build_sentiment_cache.py --min-articles 5  # only backfill thin symbols
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data/sentiment/sentiment_cache_full.parquet"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-articles", type=int, default=10,
                   help="Backfill symbols with fewer than this many articles")
    p.add_argument("--backfill-n", type=int, default=50,
                   help="Max symbols to backfill via Google News RSS")
    args = p.parse_args()

    from equity.live_news import normalize_historical_sentiment_csv

    print("=== Building sentiment cache ===")
    frames = []

    # 1. Load existing CSVs
    csv_files = [
        ROOT / "data/sentiment/combined_sentiment_latest.csv",
        ROOT / "data/sentiment/combined_sentiment_with_mar_apr_2026.csv",
    ]
    for f in csv_files:
        if f.exists():
            try:
                df = normalize_historical_sentiment_csv(
                    f, universe_metadata_csv=str(ROOT/"data/sentiment/ind_nifty500list.csv"))
                frames.append(df)
                print(f"  Loaded {len(df):,} rows from {f.name}")
            except Exception as e:
                print(f"  Warning: {f.name} failed: {e}")

    # 2. Load GDELT
    gdelt_path = ROOT / "data/sentiment/gdelt_india_2022_2026.csv"
    if gdelt_path.exists():
        try:
            gd = pd.read_csv(gdelt_path, nrows=10)  # probe columns
            # GDELT has different schema — try to normalize
            col_map = {}
            for col in gd.columns:
                cl = col.lower()
                if "symbol" in cl or "ticker" in cl: col_map[col] = "symbol"
                elif "title" in cl or "headline" in cl: col_map[col] = "headline"
                elif "date" in cl or "time" in cl: col_map[col] = "timestamp"
                elif "score" in cl or "tone" in cl or "sentiment" in cl: col_map[col] = "score"
            if "symbol" in col_map.values() and "timestamp" in col_map.values():
                gd_full = pd.read_csv(gdelt_path).rename(columns=col_map)
                gd_full["source"] = "gdelt"
                if "headline" not in gd_full.columns: gd_full["headline"] = ""
                if "score" not in gd_full.columns: gd_full["score"] = 0.0
                gd_full["score"] = pd.to_numeric(gd_full["score"], errors="coerce").fillna(0)
                gd_full["timestamp"] = pd.to_datetime(gd_full["timestamp"], errors="coerce")
                gd_full["symbol"] = gd_full["symbol"].astype(str).str.upper().str.replace(".NS","",regex=False)
                gd_full["industry"] = ""
                frames.append(gd_full[["symbol","timestamp","headline","source","score","industry"]])
                print(f"  Loaded {len(gd_full):,} GDELT rows")
        except Exception as e:
            print(f"  GDELT load failed: {e}")

    if not frames:
        print("  No source data found."); return

    # Combine + deduplicate
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["timestamp","symbol"])
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.drop_duplicates(subset=["symbol","timestamp","headline"], keep="last")
    combined = combined.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"\n  Combined: {len(combined):,} articles, {combined['symbol'].nunique()} symbols")
    print(f"  Date range: {combined['timestamp'].min().date()} → {combined['timestamp'].max().date()}")

    # 3. Identify thin symbols and backfill via Google News RSS
    sym_counts = combined.groupby("symbol").size()
    thin_syms = list(sym_counts[sym_counts < args.min_articles].index)
    from equity.universe import get_universe
    universe = set(get_universe("nifty500"))
    thin_universe = [s for s in thin_syms if s in universe][:args.backfill_n]

    if thin_universe:
        print(f"\n  Backfilling {len(thin_universe)} thin symbols via Google News RSS...")
        from equity.google_news import fetch_overnight_news
        news_df = fetch_overnight_news(
            thin_universe, hours_back=8760,  # 1 year — no strict cutoff
            universe_meta_csv=str(ROOT/"data/sentiment/ind_nifty500list.csv"))
        if not news_df.empty:
            news_df["source"] = "google_news_rss_backfill"
            news_df["industry"] = news_df.get("industry", "")
            extra = news_df[["symbol","timestamp","headline","source","score","industry"]].copy()
            combined = pd.concat([combined, extra], ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol","timestamp","headline"], keep="last")
            print(f"  Added {len(extra):,} RSS articles")

    # Add premarket_trade_date
    ts_ist = combined["timestamp"] + pd.Timedelta(hours=5, minutes=30)
    same_day = ts_ist.dt.time < pd.Timestamp("09:15").time()
    combined["premarket_trade_date"] = pd.to_datetime(
        ts_ist.dt.normalize().where(same_day, ts_ist.dt.normalize() + pd.Timedelta(days=1))
    ).dt.date

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print(f"  Final: {len(combined):,} articles, {combined['symbol'].nunique()} symbols")


if __name__ == "__main__":
    main()
