#!/usr/bin/env python3
"""Orchestrator for V4-A: build intraday chain features for all days × both indices.

Outputs partitioned parquet at:
    cache/optinet_v4/chain_features/index={NIFTY,BANKNIFTY}/year=YYYY/data.parquet

Resumable: skips year partitions that already exist (delete to force rebuild).
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

from optinet.v4_chain import build_day, discover_days

DATA_ROOT = PROJECT_ROOT / "data/option_data"
OUT_ROOT = PROJECT_ROOT / "cache/optinet_v4/chain_features"


def _process_one(args):
    d, opt_path, spot_path, index_name = args
    try:
        df = build_day(opt_path, spot_path, index_name)
        return d, len(df), df, None
    except Exception as exc:  # noqa: BLE001
        return d, 0, pd.DataFrame(), str(exc)


def build_index_year(index_name: str, year: int, day_specs: list, workers: int) -> Path | None:
    out_dir = OUT_ROOT / f"index={index_name}" / f"year={year}"
    out_path = out_dir / "data.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        print(f"  [{index_name} {year}] skip — {len(existing)} rows already at {out_path}")
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    work = [(d, opt, spot, index_name) for (d, opt, spot) in day_specs]
    print(f"  [{index_name} {year}] processing {len(work)} days with {workers} workers …")

    t0 = time.time()
    chunks: list[pd.DataFrame] = []
    errors = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, w): w[0] for w in work}
        for fut in as_completed(futures):
            d, n_rows, df, err = fut.result()
            completed += 1
            if err is not None:
                errors += 1
                print(f"    !! {d}: {err}")
                continue
            if n_rows:
                chunks.append(df)
            if completed % 25 == 0 or completed == len(work):
                rate = completed / max(time.time() - t0, 1e-3)
                print(f"    {completed}/{len(work)} done ({rate:.1f} days/s)")

    if not chunks:
        print(f"  [{index_name} {year}] NO ROWS — skipping write")
        return None
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["datetime"]).reset_index(drop=True)
    out["index"] = index_name  # ensure column
    out.to_parquet(out_path, index=False)
    print(f"  [{index_name} {year}] wrote {len(out):,} rows to {out_path}  "
          f"({(time.time()-t0):.1f}s, {errors} errors)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build V4-A chain feature store")
    parser.add_argument("--indices", nargs="+", default=["NIFTY", "BANKNIFTY"])
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grand_t0 = time.time()
    summary = []

    for index_name in args.indices:
        all_days = discover_days(DATA_ROOT, index_name)
        print(f"\n=== {index_name}: {len(all_days)} total days ===")
        # Group by year
        by_year: dict[int, list] = {}
        for d, opt, spot in all_days:
            by_year.setdefault(d.year, []).append((d, opt, spot))
        for year in args.years:
            if year not in by_year:
                continue
            day_specs = sorted(by_year[year], key=lambda x: x[0])
            path = build_index_year(index_name, year, day_specs, args.workers)
            summary.append({"index": index_name, "year": year,
                             "days": len(day_specs),
                             "path": str(path) if path else "(none)"})

    print(f"\n=== Done in {(time.time() - grand_t0)/60:.1f} min ===")
    for r in summary:
        print(f"  {r['index']} {r['year']}: {r['days']} days → {r['path']}")


if __name__ == "__main__":
    main()
