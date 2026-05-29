"""Session-level data quality checks for the futures paper runtime."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import time as dtime
from pathlib import Path

import pandas as pd


SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 29)   # yfinance last bar is 15:29, not 15:30
MIN_EXPECTED_BARS = 360
MAX_DUPLICATE_TIMESTAMPS = 0


@dataclass(frozen=True)
class SessionQualityReport:
    path: str
    trade_date: str
    n_bars: int
    first_bar: str | None
    last_bar: str | None
    duplicate_timestamps: int
    missing_close: bool
    price_min: float | None
    price_max: float | None
    ok: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def check_minute_session_quality(
    path: Path,
    trade_date: pd.Timestamp,
    *,
    date_col: str = "date",
    close_col: str = "close",
    min_expected_bars: int = MIN_EXPECTED_BARS,
) -> SessionQualityReport:
    reasons: list[str] = []
    target = pd.Timestamp(trade_date).normalize()
    if not path.exists():
        return SessionQualityReport(
            path=str(path),
            trade_date=str(target.date()),
            n_bars=0,
            first_bar=None,
            last_bar=None,
            duplicate_timestamps=0,
            missing_close=True,
            price_min=None,
            price_max=None,
            ok=False,
            reasons=[f"missing file: {path}"],
        )

    try:
        raw = pd.read_csv(path, usecols=[date_col, close_col])
    except Exception as exc:
        return SessionQualityReport(
            path=str(path),
            trade_date=str(target.date()),
            n_bars=0,
            first_bar=None,
            last_bar=None,
            duplicate_timestamps=0,
            missing_close=True,
            price_min=None,
            price_max=None,
            ok=False,
            reasons=[f"could not read CSV: {exc}"],
        )

    raw["datetime"] = pd.to_datetime(raw[date_col], errors="coerce")
    day = raw[
        (raw["datetime"] >= target) &
        (raw["datetime"] < target + pd.Timedelta(days=1))
    ].dropna(subset=["datetime"]).copy()

    if day.empty:
        return SessionQualityReport(
            path=str(path),
            trade_date=str(target.date()),
            n_bars=0,
            first_bar=None,
            last_bar=None,
            duplicate_timestamps=0,
            missing_close=True,
            price_min=None,
            price_max=None,
            ok=False,
            reasons=["no bars for target date"],
        )

    day = day.sort_values("datetime")
    n_bars = int(len(day))
    dupes = int(day["datetime"].duplicated().sum())
    first = pd.Timestamp(day["datetime"].iloc[0])
    last = pd.Timestamp(day["datetime"].iloc[-1])
    prices = pd.to_numeric(day[close_col], errors="coerce")
    price_min = float(prices.min()) if prices.notna().any() else None
    price_max = float(prices.max()) if prices.notna().any() else None

    if n_bars < min_expected_bars:
        reasons.append(f"only {n_bars} bars, expected >= {min_expected_bars}")
    if dupes > MAX_DUPLICATE_TIMESTAMPS:
        reasons.append(f"{dupes} duplicate timestamps")
    if first.time() > SESSION_OPEN:
        reasons.append(f"first bar {first.time()} after {SESSION_OPEN}")
    missing_close = last.time() < SESSION_CLOSE
    if missing_close:
        reasons.append(f"last bar {last.time()} before {SESSION_CLOSE}")
    if price_min is None or price_max is None or price_min <= 0:
        reasons.append("invalid close prices")

    return SessionQualityReport(
        path=str(path),
        trade_date=str(target.date()),
        n_bars=n_bars,
        first_bar=first.isoformat(),
        last_bar=last.isoformat(),
        duplicate_timestamps=dupes,
        missing_close=missing_close,
        price_min=price_min,
        price_max=price_max,
        ok=not reasons,
        reasons=reasons,
    )


def print_session_quality(report: SessionQualityReport) -> None:
    status = "OK" if report.ok else "FAIL"
    print("\n  -- session quality --")
    print(
        f"  {status}: {report.trade_date} bars={report.n_bars} "
        f"first={report.first_bar} last={report.last_bar}"
    )
    if report.price_min is not None and report.price_max is not None:
        print(f"  close range: {report.price_min:,.2f} -> {report.price_max:,.2f}")
    for reason in report.reasons:
        print(f"  reason: {reason}")
