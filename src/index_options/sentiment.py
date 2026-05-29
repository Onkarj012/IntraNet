"""OptiNet index sentiment fetcher (yfinance + RSS).

Pulls news for NIFTY / BANKNIFTY from yfinance and a curated set of Indian markets
RSS feeds, scores each headline with a lightweight lexicon, and aggregates per
(symbol, date) into 6 features used downstream by features.py:
    news_count, sentiment_mean, sentiment_std,
    sentiment_pos_ratio, sentiment_neg_ratio, sentiment_roll_3d
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


# Reuse the lexicon style from equity.live_news but tuned to index-level news.
_POSITIVE = {
    "beat", "beats", "growth", "surge", "surges", "up", "gain", "gains", "strong",
    "bullish", "rally", "rallies", "profit", "profits", "record", "expands",
    "upgrade", "outperform", "soar", "soars", "rise", "rises", "high", "highs",
    "boom", "robust", "rebound", "recovers", "recovery", "milestone",
}

_NEGATIVE = {
    "miss", "misses", "drop", "drops", "falls", "fall", "down", "loss", "losses",
    "weak", "bearish", "slump", "cut", "cuts", "downgrade", "fraud", "warning",
    "plunge", "plunges", "tumble", "tumbles", "crash", "crashes", "decline",
    "declines", "low", "lows", "fears", "panic", "concern", "concerns", "risk",
    "risks", "selloff", "correction",
}


_RSS_FEEDS = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketsnews.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
}

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml; q=0.9, */*; q=0.8",
}

_REQUEST_TIMEOUT = 15

# Symbols mapped to yfinance tickers and RSS keyword filters
_SYMBOL_CONFIG = {
    "NIFTY": {
        "yf_ticker": "^NSEI",
        "keywords": ("nifty", "sensex", "indian market", "indian equities",
                     "indian stock", "nse", "bse", "fii", "dii", "rupee"),
    },
    "BANKNIFTY": {
        "yf_ticker": "^NSEBANK",
        "keywords": ("bank nifty", "banknifty", "nifty bank", "psu bank",
                     "private bank", "rbi", "bank of india", "hdfc bank",
                     "icici bank", "axis bank", "sbi"),
    },
}

DEFAULT_CACHE_PATH = Path("data/sentiment/optinet_index_sentiment.csv")


def score_headline(text: str) -> float:
    """Lexicon score for a single headline; range [-1, 1]."""
    if not text:
        return 0.0
    words = {w.strip(".,:;!?()[]{}'\"-").lower() for w in re.split(r"\W+", str(text)) if w}
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    if pos == 0 and neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / max(pos + neg, 1), -1.0, 1.0))


# ── yfinance source ───────────────────────────────────────────────────────────

def _fetch_yfinance_articles(yf_ticker: str) -> list[dict]:
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        articles = yf.Ticker(yf_ticker).news or []
    except Exception:
        return []

    rows = []
    for art in articles:
        title = art.get("title") or art.get("content", {}).get("title", "")
        ts = art.get("providerPublishTime") or art.get("pubDate")
        if not title or ts is None:
            continue
        if isinstance(ts, (int, float)):
            published = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            published = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.isna(published):
                continue
            published = published.to_pydatetime()
        rows.append({
            "title": str(title),
            "published_at": published,
            "source": art.get("publisher") or "yfinance",
        })
    return rows


# ── RSS sources ───────────────────────────────────────────────────────────────

def _fetch_rss_articles(feed_url: str) -> list[dict]:
    try:
        resp = requests.get(feed_url, headers=_HTTP_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    rows = []
    for item in root.iter("item"):
        title_el = item.find("title")
        date_el = item.find("pubDate")
        if title_el is None or date_el is None:
            continue
        title = (title_el.text or "").strip()
        if not title:
            continue
        published = pd.to_datetime(date_el.text, utc=True, errors="coerce")
        if pd.isna(published):
            continue
        rows.append({
            "title": title,
            "published_at": published.to_pydatetime(),
            "source": feed_url,
        })
    return rows


# ── Aggregation ───────────────────────────────────────────────────────────────

def _filter_by_keywords(articles: list[dict], keywords: tuple[str, ...]) -> list[dict]:
    if not keywords:
        return articles
    pat = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    return [a for a in articles if pat.search(a["title"])]


def _articles_to_daily(articles: list[dict], symbol: str,
                       start: str | None, end: str | None) -> pd.DataFrame:
    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df["score"] = df["title"].map(score_headline)
    # Use IST trading-day stamp: stamp news at its publish-day IST date
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["date"])

    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("date").agg(
        news_count=("title", "size"),
        sentiment_mean=("score", "mean"),
        sentiment_std=("score", lambda s: float(s.std(ddof=0))),
        sentiment_pos_ratio=("score", lambda s: float((s > 0.1).mean())),
        sentiment_neg_ratio=("score", lambda s: float((s < -0.1).mean())),
    ).reset_index()
    grouped["index"] = symbol

    grouped = grouped.sort_values("date")
    grouped["sentiment_roll_3d"] = grouped["sentiment_mean"].rolling(3, min_periods=1).mean()
    return grouped[[
        "index", "date", "news_count", "sentiment_mean", "sentiment_std",
        "sentiment_pos_ratio", "sentiment_neg_ratio", "sentiment_roll_3d",
    ]]


def fetch_index_sentiment(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    use_yfinance: bool = True,
    use_rss: bool = True,
) -> pd.DataFrame:
    """Pull and aggregate news sentiment for a single index symbol."""
    cfg = _SYMBOL_CONFIG.get(symbol.upper())
    if cfg is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    articles: list[dict] = []
    if use_yfinance:
        articles.extend(_fetch_yfinance_articles(cfg["yf_ticker"]))
    if use_rss:
        for url in _RSS_FEEDS.values():
            articles.extend(_fetch_rss_articles(url))
            time.sleep(0.25)
    articles = _filter_by_keywords(articles, cfg["keywords"])

    return _articles_to_daily(articles, symbol.upper(), start, end)


def update_sentiment_cache(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    symbols: Iterable[str] = ("NIFTY", "BANKNIFTY"),
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch latest news, merge with cached daily aggregates, and persist."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_frames = [fetch_index_sentiment(s, start=start, end=end) for s in symbols]
    new_df = pd.concat([f for f in new_frames if not f.empty], ignore_index=True) \
             if any(not f.empty for f in new_frames) else pd.DataFrame()

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"])
    else:
        existing = pd.DataFrame()

    merged = pd.concat([existing, new_df], ignore_index=True) if not new_df.empty else existing
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset=["index", "date"], keep="last")
    merged = merged.sort_values(["index", "date"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def load_sentiment_cache(cache_path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame(columns=[
            "index", "date", "news_count", "sentiment_mean", "sentiment_std",
            "sentiment_pos_ratio", "sentiment_neg_ratio", "sentiment_roll_3d",
        ])
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df
