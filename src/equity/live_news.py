from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from equity.universe import get_symbol_to_industry_map


DEFAULT_TZ = "Asia/Kolkata"


@dataclass
class LiveNewsSummary:
    symbols_attempted: int = 0
    symbols_with_live_news: int = 0
    symbols_with_source_failure: int = 0
    live_article_count_kept: int = 0
    historical_article_count_used: int = 0
    stock_news_coverage: int = 0
    industry_news_coverage: int = 0
    live_source_failure_count: int = 0
    fallback_used_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "symbols_attempted": self.symbols_attempted,
            "symbols_with_live_news": self.symbols_with_live_news,
            "symbols_with_source_failure": self.symbols_with_source_failure,
            "live_article_count_kept": self.live_article_count_kept,
            "historical_article_count_used": self.historical_article_count_used,
            "stock_news_coverage": self.stock_news_coverage,
            "industry_news_coverage": self.industry_news_coverage,
            "live_source_failure_count": self.live_source_failure_count,
            "fallback_used_count": self.fallback_used_count,
        }


def infer_title_sentiment(text: str) -> float:
    positive = {
        "beat", "beats", "growth", "surge", "up", "gain", "gains", "strong",
        "bullish", "profit", "profits", "record", "expands", "upgrade",
    }
    negative = {
        "miss", "misses", "drop", "falls", "down", "loss", "losses", "weak",
        "bearish", "cut", "cuts", "downgrade", "fraud", "slump", "warning",
    }
    words = {w.strip(".,:;!?()[]{}'\"").lower() for w in str(text).split()}
    score = sum(1 for word in words if word in positive) - sum(1 for word in words if word in negative)
    return float(np.clip(score / 4.0, -1.0, 1.0))


def next_business_day(timestamp: pd.Timestamp) -> pd.Timestamp:
    dt = timestamp.normalize() + pd.Timedelta(days=1)
    while dt.weekday() >= 5:
        dt += pd.Timedelta(days=1)
    return dt


def compute_effective_trade_date(timestamp: pd.Timestamp, cutoff: time) -> pd.Timestamp:
    if pd.isna(timestamp):
        return pd.NaT
    timestamp = pd.Timestamp(timestamp)
    if timestamp.time() <= cutoff and timestamp.weekday() < 5:
        return timestamp.normalize()
    return next_business_day(timestamp)


def _with_trade_dates(df: pd.DataFrame, market_open_cutoff: str = "09:15", post_open_cutoff: str = "09:20") -> pd.DataFrame:
    if df.empty:
        return df.copy()

    market_open_time = _parse_time(market_open_cutoff)
    post_open_time = _parse_time(post_open_cutoff)
    enriched = df.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], errors="coerce")
    enriched = enriched[enriched["timestamp"].notna()].copy()
    enriched["premarket_trade_date"] = enriched["timestamp"].apply(lambda ts: compute_effective_trade_date(ts, market_open_time))
    enriched["post_open_trade_date"] = enriched["timestamp"].apply(lambda ts: compute_effective_trade_date(ts, post_open_time))
    enriched["is_premarket_eligible"] = True
    enriched["is_post_open_eligible"] = True
    return enriched


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def normalize_historical_sentiment_csv(
    csv_path: str | Path,
    *,
    universe_metadata_csv: str | None = None,
    market_open_cutoff: str = "09:15",
    post_open_cutoff: str = "09:20",
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=_article_columns())

    df = pd.read_csv(csv_path, parse_dates=["Publish Date"])
    df = df.rename(
        columns={
            "Symbol": "symbol",
            "Publish Date": "timestamp",
            "sentiment_score": "score",
        }
    )
    if "headline" not in df.columns:
        df["headline"] = ""
    df["symbol"] = df["symbol"].fillna("").astype(str).str.upper().str.replace(".NS", "", regex=False)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["source"] = "historical_csv"
    industry_map = get_symbol_to_industry_map(universe_metadata_csv)
    df["industry"] = df["symbol"].map(industry_map).fillna("")
    df["url"] = df.get("url", "")
    base_columns = ["symbol", "timestamp", "headline", "url", "source", "score", "industry"]
    return _with_trade_dates(df[base_columns], market_open_cutoff=market_open_cutoff, post_open_cutoff=post_open_cutoff)


def fetch_live_yfinance_news(
    symbols: Iterable[str],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    universe_metadata_csv: str | None = None,
    market_open_cutoff: str = "09:15",
    post_open_cutoff: str = "09:20",
) -> tuple[pd.DataFrame, LiveNewsSummary]:
    import yfinance as yf

    rows: list[dict[str, object]] = []
    summary = LiveNewsSummary()
    industry_map = get_symbol_to_industry_map(universe_metadata_csv)
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)

    for symbol in symbols:
        summary.symbols_attempted += 1
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            news_items = getattr(ticker, "news", []) or []
        except Exception:
            summary.symbols_with_source_failure += 1
            summary.live_source_failure_count += 1
            continue

        kept_for_symbol = 0
        for item in news_items:
            publish_ts = item.get("providerPublishTime")
            if publish_ts is None:
                continue
            try:
                ts = pd.to_datetime(publish_ts, unit="s", utc=True).tz_convert(DEFAULT_TZ).tz_localize(None)
            except Exception:
                continue
            if ts < start_ts or ts > end_ts:
                continue
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "timestamp": ts,
                    "headline": item.get("title", "") or "",
                    "url": item.get("link", "") or "",
                    "source": "live_yfinance",
                    "score": infer_title_sentiment(item.get("title", "") or ""),
                    "industry": industry_map.get(symbol.upper(), ""),
                }
            )
            kept_for_symbol += 1

        if kept_for_symbol > 0:
            summary.symbols_with_live_news += 1

    if rows:
        articles = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "timestamp", "headline"], keep="last")
        articles = _with_trade_dates(
            articles[["symbol", "timestamp", "headline", "url", "source", "score", "industry"]],
            market_open_cutoff=market_open_cutoff,
            post_open_cutoff=post_open_cutoff,
        )
    else:
        articles = pd.DataFrame(columns=_article_columns())

    summary.live_article_count_kept = int(len(articles))
    return articles, summary


def combine_article_sources(
    historical_articles: pd.DataFrame | None,
    live_articles: pd.DataFrame | None,
) -> pd.DataFrame:
    frames = []
    if historical_articles is not None and not historical_articles.empty:
        frames.append(historical_articles[_article_columns()].copy())
    if live_articles is not None and not live_articles.empty:
        frames.append(live_articles[_article_columns()].copy())
    if not frames:
        return pd.DataFrame(columns=_article_columns())

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined[combined["timestamp"].notna()].copy()
    combined = combined.sort_values(["timestamp", "source"]).drop_duplicates(
        subset=["symbol", "timestamp", "headline", "source"],
        keep="last",
    )
    return combined.reset_index(drop=True)


def summarize_article_coverage(
    articles: pd.DataFrame,
    *,
    symbols: Iterable[str],
    mode: str,
    target_dates: Iterable[pd.Timestamp],
    base_summary: LiveNewsSummary | None = None,
) -> LiveNewsSummary:
    summary = base_summary or LiveNewsSummary()
    eligible_col = "premarket_trade_date" if mode == "premarket" else "post_open_trade_date"
    target_set = {pd.Timestamp(value).normalize() for value in target_dates}

    if articles.empty:
        summary.stock_news_coverage = 0
        summary.industry_news_coverage = 0
        summary.fallback_used_count = len(list(symbols))
        return summary

    eligible_articles = articles[articles[eligible_col].isin(target_set)].copy()
    summary.historical_article_count_used = int((eligible_articles["source"] == "historical_csv").sum())
    summary.stock_news_coverage = int(eligible_articles["symbol"].nunique())
    summary.industry_news_coverage = int(eligible_articles.loc[eligible_articles["industry"] != "", "industry"].nunique())
    requested_symbols = {str(symbol).upper() for symbol in symbols}
    symbols_with_news = set(eligible_articles["symbol"].str.upper().unique().tolist())
    summary.fallback_used_count = int(len(requested_symbols - symbols_with_news))
    return summary


def _article_columns() -> list[str]:
    return [
        "symbol",
        "timestamp",
        "headline",
        "url",
        "source",
        "score",
        "industry",
        "premarket_trade_date",
        "post_open_trade_date",
        "is_premarket_eligible",
        "is_post_open_eligible",
    ]
