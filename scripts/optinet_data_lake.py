#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.data_lake import (  # noqa: E402
    DataLakeConfig,
    build_expiry_calendar,
    download_bhavcopies,
    ensure_data_lake_dirs,
    ingest_raw_file,
    normalize_index_ohlc,
    parse_bhavcopy_file,
    raw_bhavcopy_path,
    source_for_date,
    validate_options_frame,
    write_partitioned_parquet,
    write_validation_artifacts,
)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _config(args: argparse.Namespace) -> DataLakeConfig:
    return DataLakeConfig(
        data_root=PROJECT_ROOT / args.data_root,
        udiff_cutoff=_parse_date(args.udiff_cutoff),
        retries=args.retries,
        retry_sleep_seconds=args.retry_sleep,
    )


def _download(args: argparse.Namespace) -> int:
    config = _config(args)
    results = download_bhavcopies(_parse_date(args.start), _parse_date(args.end), config)
    for result in results:
        print(f"{result.trade_date} {result.source_format:6s} {result.status:16s} {result.path}")
    counts = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"Download summary: {counts}")
    return 0


def _parse(args: argparse.Namespace) -> int:
    config = _config(args)
    ensure_data_lake_dirs(config)
    paths = [Path(path) for path in args.files]
    if not paths and args.start and args.end:
        current = _parse_date(args.start)
        end = _parse_date(args.end)
        while current <= end:
            if current.weekday() < 5:
                path = raw_bhavcopy_path(current, config)
                if path.exists():
                    paths.append(path)
            current = date.fromordinal(current.toordinal() + 1)
    if not paths:
        print("No raw files found to parse.")
        return 1

    total_rows = 0
    total_written = 0
    for path in paths:
        frame, report, written = ingest_raw_file(path, config, overwrite=args.overwrite)
        total_rows += len(frame)
        total_written += len(written)
        print(f"{path}: valid={report.valid_row_count:,} bad={report.bad_row_count:,} parquet_files={len(written)}")
    print(f"Parsed {len(paths):,} raw files, {total_rows:,} valid rows, wrote {total_written:,} parquet files.")
    return 0


def _normalize_index(args: argparse.Namespace) -> int:
    config = _config(args)
    ensure_data_lake_dirs(config)
    for source in args.files:
        input_path = Path(source)
        frame = normalize_index_ohlc(pd.read_csv(input_path), default_symbol=args.default_symbol)
        out_path = config.raw_root / "index" / f"{input_path.stem}_normalized.parquet"
        frame.to_parquet(out_path, index=False)
        print(f"Wrote {len(frame):,} index rows to {out_path}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    config = _config(args)
    frames = [parse_bhavcopy_file(path) for path in args.files]
    if not frames:
        print("No files supplied.")
        return 1
    frame = pd.concat(frames, ignore_index=True)
    good, bad, report = validate_options_frame(frame)
    write_validation_artifacts(bad, report, config.normalized_root / "validation", args.name)
    print(report.to_json())
    return 0 if len(good) else 1


def _calendar(args: argparse.Namespace) -> int:
    config = _config(args)
    frames = [parse_bhavcopy_file(path) for path in args.files]
    if not frames:
        print("No files supplied.")
        return 1
    calendar = build_expiry_calendar(pd.concat(frames, ignore_index=True))
    out_path = config.metadata_root / "expiry" / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(out_path, index=False)
    print(f"Wrote {len(calendar):,} expiry calendar rows to {out_path}")
    return 0


def _route(args: argparse.Namespace) -> int:
    config = _config(args)
    trade_date = _parse_date(args.date)
    source = source_for_date(trade_date, config.udiff_cutoff)
    print(f"{trade_date}: {source} -> {raw_bhavcopy_path(trade_date, config)}")
    return 0


def _write_parquet(args: argparse.Namespace) -> int:
    config = _config(args)
    frames = [parse_bhavcopy_file(path) for path in args.files]
    if not frames:
        print("No files supplied.")
        return 1
    frame = pd.concat(frames, ignore_index=True)
    good, bad, report = validate_options_frame(frame)
    write_validation_artifacts(bad, report, config.normalized_root / "validation", args.name)
    written = write_partitioned_parquet(good, config.parquet_root, overwrite=args.overwrite)
    print(f"Wrote {len(written):,} parquet files from {report.valid_row_count:,} valid rows.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the OptiNet NSE options data lake.")
    parser.add_argument("--data-root", default="data", help="Data lake root. Defaults to repo/data.")
    parser.add_argument("--udiff-cutoff", default="2024-07-08", help="First UDiFF trading date, YYYY-MM-DD.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download immutable raw NSE F&O bhavcopy files.")
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.set_defaults(func=_download)

    parse = sub.add_parser("parse", help="Parse raw files, validate rows, and write partitioned parquet.")
    parse.add_argument("files", nargs="*", help="Raw CSV/ZIP files. If omitted, use --start/--end raw paths.")
    parse.add_argument("--start")
    parse.add_argument("--end")
    parse.add_argument("--overwrite", action="store_true")
    parse.set_defaults(func=_parse)

    index = sub.add_parser("normalize-index", help="Normalize index spot OHLC CSVs into raw/index parquet.")
    index.add_argument("files", nargs="+")
    index.add_argument("--default-symbol")
    index.set_defaults(func=_normalize_index)

    validate = sub.add_parser("validate", help="Validate raw bhavcopy files and write validation artifacts.")
    validate.add_argument("files", nargs="+")
    validate.add_argument("--name", default="manual_validation")
    validate.set_defaults(func=_validate)

    calendar = sub.add_parser("expiry-calendar", help="Build expiry metadata from raw bhavcopy files.")
    calendar.add_argument("files", nargs="+")
    calendar.add_argument("--output", default="expiry_calendar.csv")
    calendar.set_defaults(func=_calendar)

    route = sub.add_parser("route", help="Show whether a date uses legacy or UDiFF.")
    route.add_argument("date")
    route.set_defaults(func=_route)

    parquet = sub.add_parser("write-parquet", help="Parse supplied raw files and write parquet partitions.")
    parquet.add_argument("files", nargs="+")
    parquet.add_argument("--name", default="manual_parquet")
    parquet.add_argument("--overwrite", action="store_true")
    parquet.set_defaults(func=_write_parquet)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
