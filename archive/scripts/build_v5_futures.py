#!/usr/bin/env python3
"""V5 Phase 1 — futures feature builder.

Walks data/option_data/, computes 11 minute-level futures features per
(index, day), partitions by (index, year). Resumable.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.v5_futures import build_fut_day, discover_fut_days

DATA_ROOT = PROJECT_ROOT / "data/option_data"
OUT_ROOT = PROJECT_ROOT / "cache/optinet_v5/futures_features"

# Same naming convention for spot files as in V4-A
SPOT_DIRS = {"NIFTY": "nifty_data/nifty_spot", "BANKNIFTY": "banknifty_data/banknifty_spot"}
SPOT_PREFIX = {"NIFTY": "nifty_spot", "BANKNIFTY": "banknifty_spot"}


def _spot_path(d, index_name: str) -> Path:
    sd = SPOT_DIRS[index_name]
    pref = SPOT_PREFIX[index_name]
    return (DATA_ROOT / sd / str(d.year) / str(d.month) /
            f"{pref}{d.day:02d}_{d.month:02d}_{d.year}.csv")


def _process_one(args):
    d, fut_path, index_name = args
    spot_path = _spot_path(d, index_name)
    if not spot_path.exists():
        return d, 0, pd.DataFrame(), f"missing spot: {spot_path}"
    try:
        df = build_fut_day(fut_path, spot_path, index_name)
        return d, len(df), df, None
    except Exception as exc:  # noqa: BLE001
        return d, 0, pd.DataFrame(), str(exc)


def build_index_year(index_name: str, year: int, day_specs: list, workers: int) -> Path | None:
    out_dir = OUT_ROOT / f"index={index_name}" / f"year={year}"
    out_path = out_dir / "data.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        print(f"  [{index_name} {year}] skip — {len(existing)} rows already")
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    work = [(d, p, index_name) for (d, p) in day_specs]
    print(f"  [{index_name} {year}] processing {len(work)} days, {workers} workers …")

    chunks: list[pd.DataFrame] = []
    errors = 0
    completed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, w): w[0] for w in work}
        for fut in as_completed(futures):
            d, n_rows, df, err = fut.result()
            completed += 1
            if err is not None:
                errors += 1
                continue
            if n_rows:
                chunks.append(df)
            if completed % 50 == 0 or completed == len(work):
                rate = completed / max(time.time() - t0, 1e-3)
                print(f"    {completed}/{len(work)} ({rate:.1f} d/s)")

    if not chunks:
        return None
    out = pd.concat(chunks, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    print(f"  [{index_name} {year}] wrote {len(out):,} rows  "
          f"({(time.time()-t0):.1f}s, {errors} errors)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", default=["NIFTY", "BANKNIFTY"])
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grand_t0 = time.time()
    for idx in args.indices:
        all_days = discover_fut_days(DATA_ROOT, idx)
        print(f"\n=== {idx}: {len(all_days)} fut days ===")
        by_year = {}
        for d, p in all_days:
            by_year.setdefault(d.year, []).append((d, p))
        for year in args.years:
            if year not in by_year:
                continue
            build_index_year(idx, year, sorted(by_year[year]), args.workers)
    print(f"\n=== Done in {(time.time() - grand_t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
