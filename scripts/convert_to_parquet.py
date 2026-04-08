#!/usr/bin/env python3
"""
CSV to Parquet Converter for IntradayNet.

Converts minute-bar CSV files to Parquet format with:
- Snappy compression (3-4x smaller)
- Optimized dtypes (float32 instead of float64)
- Proper datetime indexing

Usage:
    python scripts/convert_to_parquet.py
    python scripts/convert_to_parquet.py --input nifty500/ --output nifty500_parquet/ --workers 4
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert_file(csv_path: Path, output_dir: Path) -> dict:
    """Convert a single CSV to Parquet."""
    symbol = csv_path.stem.replace("_minute", "")
    parquet_path = output_dir / f"{symbol}.parquet"

    try:
        df = pd.read_csv(csv_path, parse_dates=["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)

        # Optimize dtypes
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        int_cols = df.select_dtypes("int64").columns
        df[int_cols] = df[int_cols].astype("int32")

        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        csv_size = csv_path.stat().st_size
        parquet_size = parquet_path.stat().st_size

        return {
            "symbol": symbol,
            "csv_size_mb": csv_size / 1e6,
            "parquet_size_mb": parquet_size / 1e6,
            "rows": len(df),
            "success": True,
        }
    except Exception as e:
        return {
            "symbol": csv_path.stem,
            "csv_size_mb": csv_path.stat().st_size / 1e6,
            "parquet_size_mb": 0,
            "rows": 0,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Convert CSVs to Parquet")
    parser.add_argument("--input", default="nifty500", help="Input CSV directory")
    parser.add_argument("--output", default="nifty500_parquet", help="Output Parquet directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (1=sequential)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0=all)")
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(exist_ok=True, parents=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if args.limit > 0:
        csv_files = csv_files[:args.limit]

    print(f"Converting {len(csv_files)} CSV files to Parquet...")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    total_csv = 0
    total_parquet = 0
    total_rows = 0
    results = []

    t0 = time.time()

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(convert_file, f, output_dir): f for f in csv_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                result = convert_file(future.result())
                pass
    else:
        for csv_file in tqdm(csv_files, desc="Converting"):
            result = convert_file(csv_file, output_dir)
            results.append(result)
            total_csv += result["csv_size_mb"]
            total_parquet += result["parquet_size_mb"]
            total_rows += result["rows"]

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'=' * 60}")
    print("CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files converted: {len(csv_files)}")
    print(f"Total rows:     {total_rows:,}")
    print(f"CSV total:      {total_csv:.2f} MB")
    print(f"Parquet total:  {total_parquet:.2f} MB")
    print(f"Compression:    {total_csv / max(total_parquet, 1):.1f}x")
    print(f"Space saved:    {total_csv - total_parquet:.2f} MB ({(1 - total_parquet/max(total_csv, 1))*100:.1f}%)")
    print(f"Time elapsed:   {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Save conversion log
    log_path = output_dir / "conversion_log.csv"
    log_df = pd.DataFrame(results)
    log_df.to_csv(log_path, index=False)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
