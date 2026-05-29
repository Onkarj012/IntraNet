"""Load the OptiNet parquet lake into the same schemas expected by optinet.data."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd


_DEFAULT_LAKE = Path(__file__).resolve().parents[2] / "data" / "parquet"
_DEFAULT_INDEX = Path(__file__).resolve().parents[2] / "data" / "indices"

_SYMBOL_MAP = {"NIFTY": "nifty_daily.csv", "BANKNIFTY": "banknifty_daily.csv"}


def load_options_lake(
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
    lake_root: str | Path = _DEFAULT_LAKE,
) -> pd.DataFrame:
    """Read partitioned parquet files and return a frame matching optinet.data.load_option_chain schema."""
    root = Path(lake_root)
    sym_filter = {s.upper() for s in symbols} if symbols else None

    pattern = str(root / "symbol=*" / "year=*" / "options_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    # Filter by symbol and year before loading to avoid reading everything
    if sym_filter or start or end:
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None
        filtered = []
        for f in files:
            parts = Path(f).parts
            sym_part = next((p for p in parts if p.startswith("symbol=")), "")
            year_part = next((p for p in parts if p.startswith("year=")), "")
            sym = sym_part.replace("symbol=", "")
            year = int(year_part.replace("year=", "")) if year_part else 0
            if sym_filter and sym not in sym_filter:
                continue
            if start_ts and year < start_ts.year:
                continue
            if end_ts and year > end_ts.year:
                continue
            filtered.append(f)
        files = filtered

    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Rename to match optinet.data.load_option_chain output schema
    df = df.rename(columns={
        "symbol": "index",
        "expiry_date": "expiry",
        "strike_price": "strike",
        "change_in_oi": "change_oi",
        "open_interest": "open_interest",
    })

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    df["days_to_expiry"] = (df["expiry"] - df["date"]).dt.days
    df = df[df["days_to_expiry"] >= 0]

    if start_ts := (pd.Timestamp(start) if start else None):
        df = df[df["date"] >= start_ts]
    if end_ts := (pd.Timestamp(end) if end else None):
        df = df[df["date"] <= end_ts]

    return df.sort_values(["index", "date", "expiry", "strike", "option_type"]).reset_index(drop=True)


def load_index_lake(
    symbols: list[str] | None = None,
    index_root: str | Path = _DEFAULT_INDEX,
) -> pd.DataFrame:
    """Read saved daily OHLC CSVs and return a frame matching optinet.data.load_index_bars schema."""
    root = Path(index_root)
    sym_filter = {s.upper() for s in symbols} if symbols else set(_SYMBOL_MAP.keys())

    frames = []
    for sym in sym_filter:
        fname = _SYMBOL_MAP.get(sym)
        if not fname:
            continue
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(f"Index OHLC file not found: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["index"] = sym
        frames.append(df[["index", "date", "open", "high", "low", "close", "volume"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["index", "date"]).reset_index(drop=True)
