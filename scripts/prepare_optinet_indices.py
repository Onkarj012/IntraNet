#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


OPTION_RE = re.compile(r"^(BANKNIFTY|NIFTY)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _aggregate_spot_file(path: Path, index_name: str) -> dict[str, object] | None:
    df = _read_csv(path)
    if df.empty:
        return None
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    if df.empty:
        return None
    return {
        "date": pd.Timestamp(df["datetime"].dt.date.iloc[0]),
        "index": index_name,
        "open": float(df["open"].iloc[0]),
        "high": float(df["high"].max()),
        "low": float(df["low"].min()),
        "close": float(df["close"].iloc[-1]),
        "volume": float(df["volume"].sum()) if "volume" in df.columns else 0.0,
    }


def _aggregate_option_file(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    parsed = df["symbol"].astype(str).str.upper().str.extract(OPTION_RE)
    parsed.columns = ["index", "expiry_raw", "strike", "option_type"]
    keep = parsed.notna().all(axis=1)
    if not keep.any():
        return pd.DataFrame()
    df = df.loc[keep].copy()
    parsed_values = parsed.loc[keep].copy()
    parsed_values["expiry"] = pd.to_datetime(parsed_values["expiry_raw"], format="%d%b%y", errors="coerce").dt.normalize()
    parsed_values["strike"] = pd.to_numeric(parsed_values["strike"], errors="coerce")
    parsed_values = parsed_values.drop(columns=["expiry_raw"])
    df = pd.concat([df, parsed_values], axis=1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    for col in ["open", "high", "low", "close", "oi", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "datetime", "index", "expiry", "strike", "option_type", "close"])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["index", "date", "expiry", "strike", "option_type", "datetime"])
    group_cols = ["index", "date", "expiry", "strike", "option_type"]
    out = df.groupby(group_cols, as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        open_interest=("oi", "last"),
        first_oi=("oi", "first"),
    )
    out["change_oi"] = out["open_interest"] - out["first_oi"]
    out = out.drop(columns=["first_oi"])
    return out


def _path_allowed(path: Path, years: set[str] | None, months: set[str] | None) -> bool:
    if years and not any(part in years for part in path.parts):
        return False
    if months and not any(part in months for part in path.parts):
        return False
    if not years and not months:
        return True
    return True


def prepare_indices(
    data_root: Path,
    output_dir: Path,
    years: set[str] | None = None,
    months: set[str] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spot_rows: list[dict[str, object]] = []
    option_frames: list[pd.DataFrame] = []
    configs = [
        ("NIFTY", data_root / "nifty_data" / "nifty_spot", data_root / "nifty_data" / "nifty_options"),
        ("BANKNIFTY", data_root / "banknifty_data" / "banknifty_spot", data_root / "banknifty_data" / "banknifty_options"),
    ]
    for index_name, spot_dir, option_dir in configs:
        for path in sorted(spot_dir.rglob("*.csv")):
            if not _path_allowed(path, years, months):
                continue
            row = _aggregate_spot_file(path, index_name)
            if row is not None:
                spot_rows.append(row)
        for path in sorted(option_dir.rglob("*.csv")):
            if not _path_allowed(path, years, months):
                continue
            options = _aggregate_option_file(path)
            if not options.empty:
                option_frames.append(options)

    index_daily = pd.DataFrame(spot_rows).sort_values(["index", "date"])
    options_daily = pd.concat(option_frames, ignore_index=True).sort_values(["index", "date", "expiry", "strike", "option_type"])

    index_path = output_dir / "index_daily.csv"
    options_path = output_dir / "options_daily.csv"
    index_daily.to_csv(index_path, index=False)
    options_daily.to_csv(options_path, index=False)
    return index_path, options_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare NIFTY/BANKNIFTY minute index option data for OptiNet.")
    parser.add_argument("--data-root", default="data/indices")
    parser.add_argument("--output-dir", default="cache/optinet/prepared_indices")
    parser.add_argument("--years", nargs="*", help="Optional year filter, e.g. --years 2023 2024.")
    parser.add_argument("--months", nargs="*", help="Optional month-number filter, e.g. --months 1 2 3.")
    args = parser.parse_args()
    years = set(args.years) if args.years else None
    months = set(args.months) if args.months else None
    index_path, options_path = prepare_indices(PROJECT_ROOT / args.data_root, PROJECT_ROOT / args.output_dir, years, months)
    print(f"index={index_path}")
    print(f"options={options_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
