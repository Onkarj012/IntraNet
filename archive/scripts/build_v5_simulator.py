#!/usr/bin/env python3
"""V5 Phase 1.2 — Strategy simulator orchestrator.

For every (index, date) day, simulate 13 strategies at every minute from 09:30
to 14:55, using V4-A chain features for ATM strike / ATM IV / T_years, and the
raw intraday option chain CSVs for real prices (with BS fallback).

Output: cache/optinet_v5/strategy_labels/index={NIFTY,BANKNIFTY}/year=YYYY/data.parquet
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

from optinet.v4_chain import discover_days, load_options_day, load_spot
from optinet.v5_simulator import simulate_day_with_features

DATA_ROOT = PROJECT_ROOT / "data/option_data"
CHAIN_DIR = PROJECT_ROOT / "cache/optinet_v4/chain_features"
OUT_ROOT  = PROJECT_ROOT / "cache/optinet_v5/strategy_labels"


def _process_one(args):
    d, opt_path, spot_path, index_name = args
    try:
        # Load options + spot for the day
        options_day = load_options_day(opt_path, index_name)
        spot_day = load_spot(spot_path, index_name)

        # Load V4-A chain features for this day from the parquet partition
        year = d.year
        chain_path = CHAIN_DIR / f"index={index_name}" / f"year={year}" / "data.parquet"
        if not chain_path.exists():
            return d, 0, pd.DataFrame(), f"missing chain features: {chain_path}"
        chain_full = pd.read_parquet(chain_path)
        chain_full["datetime"] = pd.to_datetime(chain_full["datetime"])
        chain_full = chain_full[chain_full["datetime"].dt.date == d]
        if chain_full.empty:
            return d, 0, pd.DataFrame(), "no chain features for date"

        df = simulate_day_with_features(chain_full, options_day, spot_day, index_name)
        return d, len(df), df, None
    except Exception as exc:  # noqa: BLE001
        import traceback
        return d, 0, pd.DataFrame(), f"{exc}\n{traceback.format_exc()[:500]}"


def build_index_year(index_name: str, year: int, day_specs: list, workers: int) -> Path | None:
    out_dir = OUT_ROOT / f"index={index_name}" / f"year={year}"
    out_path = out_dir / "data.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        print(f"  [{index_name} {year}] skip — {len(existing):,} rows already")
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    work = [(d, opt, spot, index_name) for (d, opt, spot) in day_specs]
    print(f"  [{index_name} {year}] simulating {len(work)} days, {workers} workers …")

    chunks: list[pd.DataFrame] = []
    errors = 0
    error_samples = []
    completed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, w): w[0] for w in work}
        for fut in as_completed(futures):
            d, n_rows, df, err = fut.result()
            completed += 1
            if err is not None:
                errors += 1
                if len(error_samples) < 5:
                    error_samples.append(f"{d}: {err[:200]}")
                continue
            if n_rows:
                chunks.append(df)
            if completed % 20 == 0 or completed == len(work):
                rate = completed / max(time.time() - t0, 1e-3)
                print(f"    {completed}/{len(work)} ({rate:.1f} d/s, {errors} err)")

    if not chunks:
        print(f"  [{index_name} {year}] NO ROWS  errors_sample={error_samples}")
        return None
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["datetime", "strategy_id"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    print(f"  [{index_name} {year}] wrote {len(out):,} rows  "
          f"({(time.time()-t0)/60:.1f} min, {errors} errors)")
    if error_samples:
        for e in error_samples[:3]:
            print(f"    err sample: {e}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", default=["NIFTY", "BANKNIFTY"])
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--workers", type=int, default=4)  # heavier workload than V4-A
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grand_t0 = time.time()
    for idx in args.indices:
        all_days = discover_days(DATA_ROOT, idx)
        print(f"\n=== {idx}: {len(all_days)} days ===")
        by_year = {}
        for d, opt, spot in all_days:
            by_year.setdefault(d.year, []).append((d, opt, spot))
        for year in args.years:
            if year not in by_year:
                continue
            build_index_year(idx, year, sorted(by_year[year], key=lambda x: x[0]), args.workers)
    print(f"\n=== Done in {(time.time() - grand_t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
