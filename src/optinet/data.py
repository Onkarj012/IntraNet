from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from optinet.config import canonical_index


INDEX_COLUMNS = {
    "date": ("date", "timestamp", "datetime", "trade_date", "Date", "TIMESTAMP"),
    "open": ("open", "Open", "OPEN"),
    "high": ("high", "High", "HIGH"),
    "low": ("low", "Low", "LOW"),
    "close": ("close", "Close", "CLOSE", "ltp", "LTP"),
    "volume": ("volume", "Volume", "VOLUME", "tottrdqty", "TOTTRDQTY"),
    "index": ("index", "index_name", "symbol", "SYMBOL", "underlying", "UNDERLYING", "name"),
}

OPTION_COLUMNS = {
    "date": ("date", "timestamp", "trade_date", "TIMESTAMP", "Date"),
    "index": ("index", "index_name", "symbol", "SYMBOL", "underlying", "UNDERLYING", "name"),
    "expiry": ("expiry", "EXPIRY_DT", "expiry_date", "Expiry"),
    "strike": ("strike", "STRIKE_PR", "strike_price", "Strike"),
    "option_type": ("option_type", "OPTION_TYP", "instrument_type", "type", "cp"),
    "open": ("open", "OPEN", "Open"),
    "high": ("high", "HIGH", "High"),
    "low": ("low", "LOW", "Low"),
    "close": ("close", "CLOSE", "Close", "settle_pr", "SETTLE_PR", "ltp"),
    "volume": ("volume", "CONTRACTS", "volume", "VOLUME", "tottrdqty"),
    "open_interest": ("open_interest", "OPEN_INT", "oi", "OI"),
    "change_oi": ("change_oi", "CHG_IN_OI", "change_in_oi", "CHG_OI"),
}


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in available:
            return available[candidate.lower()]
    return None


def _rename_known_columns(df: pd.DataFrame, mapping: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for target, candidates in mapping.items():
        source = _first_existing(df.columns, candidates)
        if source is not None:
            rename[source] = target
    return df.rename(columns=rename)


def load_index_bars(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [_rename_known_columns(_read_table(path), INDEX_COLUMNS) for path in paths]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    missing = {"date", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Index data is missing required columns: {sorted(missing)}")
    if "index" not in df.columns:
        df["index"] = "NIFTY"
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce").dt.normalize()
    df["index"] = df["index"].map(canonical_index)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "index", "open", "high", "low", "close"]).sort_values(["index", "date"])


def load_option_chain(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [_rename_known_columns(_read_table(path), OPTION_COLUMNS) for path in paths]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    missing = {"date", "index", "expiry", "strike", "option_type", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Option chain data is missing required columns: {sorted(missing)}")
    for optional in ["volume", "open_interest", "change_oi"]:
        if optional not in df.columns:
            df[optional] = 0.0
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce").dt.normalize()
    df["expiry"] = pd.to_datetime(df["expiry"], format="mixed", errors="coerce").dt.normalize()
    df["index"] = df["index"].map(canonical_index)
    df["option_type"] = df["option_type"].astype(str).str.upper().str[0].map({"C": "CE", "P": "PE"}).fillna(
        df["option_type"].astype(str).str.upper()
    )
    for col in ["strike", "open", "high", "low", "close", "volume", "open_interest", "change_oi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["days_to_expiry"] = (df["expiry"] - df["date"]).dt.days
    df = df[df["days_to_expiry"] >= 0]
    return df.dropna(subset=["date", "index", "expiry", "strike", "option_type", "close"]).sort_values(
        ["index", "date", "expiry", "strike", "option_type"]
    )


def align_spot_to_chain(option_chain: pd.DataFrame, index_bars: pd.DataFrame) -> pd.DataFrame:
    spot = index_bars[["index", "date", "close"]].rename(columns={"close": "spot"})
    merged = option_chain.merge(spot, on=["index", "date"], how="left")
    if "days_to_expiry" not in merged.columns and {"expiry", "date"}.issubset(merged.columns):
        merged["days_to_expiry"] = (pd.to_datetime(merged["expiry"]) - pd.to_datetime(merged["date"])).dt.days
    return merged[merged["spot"].notna()].copy()


def latest_rows_by_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.groupby("index")["date"].idxmax()
    return df.loc[idx].sort_values("index")
