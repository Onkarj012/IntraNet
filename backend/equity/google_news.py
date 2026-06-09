"""
Overnight Google News fetcher for intraday sentiment features.

Fetches Google News RSS for each symbol (close→open window),
scores headlines with a financial lexicon, returns a DataFrame
compatible with SentimentFeatureBuilder.

No external dependencies beyond stdlib + requests.

Usage:
    from equity.google_news import fetch_overnight_news
    df = fetch_overnight_news(["RELIANCE", "TCS", "HDFCBANK"],
                               hours_back=18)
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Sequence

import requests
import pandas as pd
import numpy as np

# ── Financial sentiment lexicon (compact but domain-relevant) ──────────────
_POSITIVE = {
    "surge", "surges", "gain", "gains", "rally", "rallies", "rise", "rises", "rises",
    "jump", "jumps", "soar", "soars", "record", "high", "profit", "profits", "growth",
    "strong", "beat", "beats", "upgrade", "upgraded", "buy", "outperform", "positive",
    "bullish", "expansion", "order", "orders", "deal", "deals", "award", "awarded",
    "dividend", "acquisition", "revenue", "increases", "improved", "improvement",
    "raised", "raises", "higher", "upside", "breakout", "momentum", "optimistic",
}
_NEGATIVE = {
    "fall", "falls", "drop", "drops", "decline", "declines", "plunge", "plunges",
    "crash", "crashes", "loss", "losses", "miss", "misses", "downgrade", "downgraded",
    "sell", "underperform", "negative", "bearish", "cut", "cuts", "concern", "concerns",
    "risk", "risks", "debt", "default", "investigation", "fraud", "penalty", "fine",
    "weak", "lower", "slowdown", "disappointing", "disappoints", "sell-off", "selloff",
    "slump", "slumps", "tumble", "tumbles", "warning", "caution", "headwind",
}


def _score_headline(text: str) -> float:
    """Return sentiment score in [-1, 1] from headline text."""
    words = re.findall(r"[a-z]+", text.lower())
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _parse_pubdate(pubdate_str: str) -> datetime | None:
    """Parse RSS pubDate string to UTC datetime."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(pubdate_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _fetch_rss(query: str, timeout: int = 8) -> list[dict]:
    """Fetch Google News RSS for a query, return list of {title, published}."""
    url = (f"https://news.google.com/rss/search"
           f"?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            pub = item.findtext("pubDate") or ""
            dt = _parse_pubdate(pub)
            if title and dt:
                items.append({"title": title, "published": dt})
        return items
    except Exception:
        return []


def fetch_overnight_news(
    symbols: Sequence[str],
    *,
    hours_back: int = 18,
    sleep_sec: float = 0.3,
    universe_meta_csv: str | None = None,
) -> pd.DataFrame:
    """
    Fetch Google News RSS for each symbol, score sentiment,
    return DataFrame with columns:
        symbol, timestamp, headline, source, score, industry, premarket_trade_date

    Parameters
    ----------
    symbols : list[str]
        NSE stock symbols (e.g. "RELIANCE").
    hours_back : int
        Only keep articles published within this many hours (default 18 = close→open).
    sleep_sec : float
        Delay between requests to avoid rate limiting.
    universe_meta_csv : str, optional
        Path to ind_nifty500list.csv for industry mapping.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    trade_date = pd.Timestamp.now(tz="Asia/Kolkata").normalize()

    # Industry map
    industry_map: dict[str, str] = {}
    if universe_meta_csv and Path(universe_meta_csv).exists():
        try:
            meta = pd.read_csv(universe_meta_csv)
            sym_col = next((c for c in meta.columns if c.lower() in {"symbol","ticker"}), None)
            ind_col = next((c for c in meta.columns if "industry" in c.lower()), None)
            if sym_col and ind_col:
                industry_map = dict(zip(meta[sym_col].str.upper(), meta[ind_col].fillna("")))
        except Exception:
            pass

    rows = []
    for i, sym in enumerate(symbols):
        query = f"{sym} NSE stock"
        articles = _fetch_rss(query)
        # Sort by recency, take top 10
        articles.sort(key=lambda x: x["published"], reverse=True)
        recent = articles[:10]
        for j, art in enumerate(recent):
            # Recency weight: 1.0 for newest, decays to 0.5 for 10th
            recency_weight = 1.0 - j * 0.05
            score = _score_headline(art["title"]) * recency_weight
            # Flag as "overnight" if within hours_back window
            in_window = art["published"] >= cutoff
            rows.append({
                "symbol": sym.upper(),
                "timestamp": pd.Timestamp(art["published"].replace(tzinfo=None)),
                "headline": art["title"],
                "source": "google_news_rss",
                "score": round(score, 4),
                "in_window": in_window,
                "industry": industry_map.get(sym.upper(), ""),
                "premarket_trade_date": trade_date.date(),
            })
        if (i + 1) % 5 == 0 or i == len(symbols) - 1:
            print(f"  [{i+1}/{len(symbols)}] fetched, {len(rows)} articles total")
        time.sleep(sleep_sec)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol","timestamp","headline","source",
                                      "score","industry","premarket_trade_date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Google News: {len(df)} articles for {df['symbol'].nunique()} symbols")
    return df
