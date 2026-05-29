"""Data freshness checks for the futures paper-trading runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FreshnessCheck:
    label: str
    path: Path
    latest_date: pd.Timestamp | None
    required_date: pd.Timestamp

    @property
    def ok(self) -> bool:
        return self.latest_date is not None and self.latest_date >= self.required_date

    def message(self) -> str:
        latest = "missing" if self.latest_date is None else str(self.latest_date.date())
        status = "OK" if self.ok else "STALE"
        return (
            f"{status}: {self.label} latest={latest}, "
            f"required>={self.required_date.date()} ({self.path})"
        )


def latest_csv_date(path: Path, date_col: str = "date") -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=[date_col])
    except Exception:
        return None
    if df.empty:
        return None
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    if dates.notna().sum() == 0:
        return None
    return dates.dt.tz_convert(None).dt.normalize().max()


def latest_weekday(as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the latest Monday-Friday date as a pragmatic trading-day proxy."""
    if as_of is None:
        as_of = pd.Timestamp.now(tz="Asia/Kolkata")
    day = pd.Timestamp(as_of).tz_localize(None).normalize()
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def check_file_freshness(
    label: str,
    path: Path,
    required_date: pd.Timestamp,
    date_col: str = "date",
) -> FreshnessCheck:
    return FreshnessCheck(
        label=label,
        path=path,
        latest_date=latest_csv_date(path, date_col=date_col),
        required_date=pd.Timestamp(required_date).normalize(),
    )


def print_freshness(checks: list[FreshnessCheck]) -> None:
    print("\n  -- data freshness --")
    for check in checks:
        print(f"  {check.message()}")


def freshness_failed(checks: list[FreshnessCheck]) -> bool:
    return any(not check.ok for check in checks)
