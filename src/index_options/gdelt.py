"""GDELT 2.0 event-count ingestion for OptiNet.

Pulls daily article volume from the GDELT DOC API for India-tagged macro/financial
themes between 2022-01-01 and today. Falls back to whatever range is available.

Themes captured (one feature column each):
    gdelt_economy            — broad economy / GDP
    gdelt_inflation_rates    — inflation, interest rates, RBI
    gdelt_stockmarket        — NIFTY, sensex, equities
    gdelt_policy_election    — policy, regulation, election
    gdelt_foreign_investment — FII, FDI, foreign investment

The API's `mode=TimelineVolRaw` limits each request to ~90 days, so we chunk.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Map of feature column → GDELT query
GDELT_THEMES: dict[str, str] = {
    "gdelt_economy": "india AND (economy OR GDP OR economic)",
    "gdelt_inflation_rates": "india AND (inflation OR \"interest rate\" OR RBI OR \"reserve bank\")",
    "gdelt_stockmarket": "india AND (NIFTY OR sensex OR \"stock market\" OR equities OR shares)",
    "gdelt_policy_election": "india AND (policy OR regulation OR election OR government)",
    "gdelt_foreign_investment": "india AND (FII OR FDI OR \"foreign investment\" OR \"foreign portfolio\")",
}

DEFAULT_CACHE_PATH = Path("data/sentiment/gdelt_india_2022_2026.csv")
_CHUNK_DAYS = 90
_REQUEST_TIMEOUT = 30
_RETRY_SLEEP = 1.5
_INTER_REQUEST_GAP = 0.5

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


def _format_dt(d: date) -> str:
    return d.strftime("%Y%m%d") + "000000"


def _fetch_chunk(query: str, start: date, end: date, retries: int = 3) -> pd.DataFrame:
    """Fetch a single TimelineVolRaw chunk (≤ 90 days)."""
    params = {
        "query": query,
        "mode": "TimelineVolRaw",
        "format": "csv",
        "startdatetime": _format_dt(start),
        "enddatetime": _format_dt(end + timedelta(days=1)),
        "timezoom": "no",
    }
    last_err = ""
    for attempt in range(retries):
        try:
            r = requests.get(GDELT_DOC_URL, params=params, headers=_HTTP_HEADERS,
                             timeout=_REQUEST_TIMEOUT)
            if r.status_code == 200 and r.text.strip():
                df = pd.read_csv(StringIO(r.text))
                return df
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(_RETRY_SLEEP * (attempt + 1))
    print(f"  [warn] chunk {start}–{end} failed: {last_err}")
    return pd.DataFrame()


def _normalize_chunk(df: pd.DataFrame, theme_col: str) -> pd.DataFrame:
    """Coerce a TimelineVolRaw response into (date, theme_col) daily counts."""
    if df.empty:
        return pd.DataFrame(columns=["date", theme_col])
    # Detect the date column the API uses (varies)
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime"}), df.columns[0])
    val_col = next((c for c in df.columns if c.lower() in {"value", "count", "volume", "rawvolume"}),
                   df.columns[-1])
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        theme_col: pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna(subset=["date"])
    out["date"] = out["date"].dt.normalize()
    out = out.groupby("date", as_index=False)[theme_col].sum()
    return out


def fetch_gdelt_theme(theme_col: str, query: str,
                       start: date, end: date) -> pd.DataFrame:
    """Fetch a single theme's daily volume, chunking the date range."""
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=_CHUNK_DAYS - 1), end)
        df = _fetch_chunk(query, cur, chunk_end)
        if not df.empty:
            chunks.append(_normalize_chunk(df, theme_col))
        cur = chunk_end + timedelta(days=1)
        time.sleep(_INTER_REQUEST_GAP)

    if not chunks:
        return pd.DataFrame(columns=["date", theme_col])
    return pd.concat(chunks, ignore_index=True).drop_duplicates("date").sort_values("date")


def fetch_gdelt_features(start: date, end: date,
                         themes: dict[str, str] | None = None) -> pd.DataFrame:
    """Fetch all configured themes and return one wide frame keyed by date."""
    themes = themes or GDELT_THEMES
    merged: pd.DataFrame | None = None
    for col, q in themes.items():
        print(f"  GDELT {col}: {start} → {end}")
        df = fetch_gdelt_theme(col, q, start, end)
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["date", *themes.keys()])
    merged = merged.sort_values("date").fillna(0.0)
    return merged


def update_gdelt_cache(cache_path: str | Path = DEFAULT_CACHE_PATH,
                       start: date | None = None,
                       end: date | None = None) -> pd.DataFrame:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    end = end or date.today()
    start = start or date(2022, 1, 1)

    new = fetch_gdelt_features(start, end)

    if path.exists() and path.stat().st_size > 4:
        try:
            existing = pd.read_csv(path, parse_dates=["date"])
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    merged = pd.concat([existing, new], ignore_index=True) if not new.empty else existing
    if merged.empty:
        return merged
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def load_gdelt_cache(cache_path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists() or path.stat().st_size < 5:
        cols = ["date"] + list(GDELT_THEMES.keys())
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df
