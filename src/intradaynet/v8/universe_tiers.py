"""
Stock universe tier classification for V8.

Not all 531 Nifty 500 stocks are equal. Tier by data quality:
- Tier 1: Full 2015-2026 data (~334 stocks) — all features + embeddings
- Tier 2: 2023-2026 (2-3 years, ~100 stocks) — reduced features, no embedding
- Tier 3: < 2 years data (~97 stocks) — basic features only, higher thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

from intradaynet.universe import get_universe, get_symbol_metadata


class DataTier(Enum):
    """Stock data quality tier."""
    TIER_1 = "tier_1"  # Full 2015-2026 data
    TIER_2 = "tier_2"  # 2-3 years of data
    TIER_3 = "tier_3"  # < 2 years (IPOs, new listings)
    UNKNOWN = "unknown"  # No data found

    @property
    def uses_embeddings(self) -> bool:
        """Whether curve embeddings can be computed for this tier."""
        return self == DataTier.TIER_1

    @property
    def min_confidence_multiplier(self) -> float:
        """Higher confidence required for lower tiers."""
        return {DataTier.TIER_1: 1.0, DataTier.TIER_2: 1.1, DataTier.TIER_3: 1.25}[self]

    @property
    def slippage_pct(self) -> float:
        """Estimated slippage by tier."""
        return {DataTier.TIER_1: 0.05, DataTier.TIER_2: 0.10, DataTier.TIER_3: 0.20}[self]


@dataclass
class TierAssignment:
    """Complete tier assignment for one stock."""
    symbol: str
    tier: DataTier
    data_start_date: Optional[pd.Timestamp] = None
    data_end_date: Optional[pd.Timestamp] = None
    total_days: int = 0
    total_bars: int = 0
    avg_daily_volume: float = 0.0
    avg_daily_bars: float = 0.0
    has_gaps: bool = False
    industry: str = ""
    company_name: str = ""

    @property
    def years_of_data(self) -> float:
        if self.data_start_date and self.data_end_date:
            return (self.data_end_date - self.data_start_date).days / 365.25
        return 0.0

    @property
    def is_liquid(self) -> bool:
        """Sufficiently liquid for trading."""
        return self.avg_daily_volume >= 100000


@dataclass
class UniverseTierReport:
    """Summary of tier distribution across a universe."""
    symbols: list[str] = field(default_factory=list)
    assignments: dict[str, TierAssignment] = field(default_factory=dict)
    tier_counts: dict[DataTier, int] = field(default_factory=dict)

    @property
    def tier_1_symbols(self) -> list[str]:
        return [s for s, a in self.assignments.items() if a.tier == DataTier.TIER_1]

    @property
    def tier_2_symbols(self) -> list[str]:
        return [s for s, a in self.assignments.items() if a.tier == DataTier.TIER_2]

    @property
    def tier_3_symbols(self) -> list[str]:
        return [s for s, a in self.assignments.items() if a.tier == DataTier.TIER_3]

    def summary(self) -> str:
        lines = ["Universe Tier Report", "=" * 40]
        for tier in (DataTier.TIER_1, DataTier.TIER_2, DataTier.TIER_3, DataTier.UNKNOWN):
            count = self.tier_counts.get(tier, 0)
            lines.append(f"  {tier.value}: {count} symbols")
        return "\n".join(lines)


def classify_tiers(
    data_dir: str | Path,
    universe: str = "nifty500",
    *,
    min_tier_1_years: float = 7.0,  # ~2015-2023
    min_tier_2_years: float = 2.0,
    min_daily_bars: int = 200,
    verbose: bool = False,
) -> UniverseTierReport:
    """
    Classify all stocks in a universe into data quality tiers.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing {SYMBOL}_minute.csv files.
    universe : str
        Universe name (nifty50, nifty100, nifty200, nifty500).
    min_tier_1_years : float
        Minimum years of data for Tier 1.
    min_tier_2_years : float
        Minimum years of data for Tier 2.
    min_daily_bars : int
        Minimum bars per day for a session to be counted.
    verbose : bool
        Print progress.

    Returns
    -------
    UniverseTierReport
        Complete tier classification.
    """
    data_dir = Path(data_dir)
    symbols = get_universe(universe)
    assignments: dict[str, TierAssignment] = {}
    tier_counts: dict[DataTier, int] = {t: 0 for t in DataTier}

    if verbose:
        print(f"Classifying {len(symbols)} symbols from {data_dir}...")

    for i, symbol in enumerate(symbols):
        if verbose and (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(symbols)} symbols classified")

        file_path = data_dir / f"{symbol}_minute.csv"
        assignment = _classify_single_symbol(
            symbol, file_path, min_tier_1_years, min_tier_2_years, min_daily_bars
        )
        assignments[symbol] = assignment
        tier_counts[assignment.tier] = tier_counts.get(assignment.tier, 0) + 1

    if verbose:
        print(f"Done. T1={tier_counts[DataTier.TIER_1]}, "
              f"T2={tier_counts[DataTier.TIER_2]}, "
              f"T3={tier_counts[DataTier.TIER_3]}, "
              f"UNK={tier_counts[DataTier.UNKNOWN]}")

    return UniverseTierReport(
        symbols=symbols,
        assignments=assignments,
        tier_counts=tier_counts,
    )


def _classify_single_symbol(
    symbol: str,
    file_path: Path,
    min_tier_1_years: float,
    min_tier_2_years: float,
    min_daily_bars: int,
) -> TierAssignment:
    """Classify a single symbol based on its minute data file."""
    assignment = TierAssignment(symbol=symbol, tier=DataTier.UNKNOWN)

    metadata = get_symbol_metadata(symbol)
    assignment.industry = metadata.get("industry", "")
    assignment.company_name = metadata.get("company_name", "")

    if not file_path.exists():
        return assignment

    try:
        df = _load_and_parse_minute_csv(file_path)
        if df.empty:
            return assignment

        assignment.data_start_date = df.index.min()
        assignment.data_end_date = df.index.max()
        assignment.total_bars = len(df)

        daily_groups = df.resample("D")
        valid_days = 0
        total_volume = 0.0
        total_bars_count = 0

        for _, day_df in df.groupby(df.index.date):
            if len(day_df) >= min_daily_bars:
                valid_days += 1
                total_volume += day_df["volume"].sum()
                total_bars_count += len(day_df)

        assignment.total_days = valid_days
        assignment.avg_daily_volume = total_volume / max(valid_days, 1)
        assignment.avg_daily_bars = total_bars_count / max(valid_days, 1)
        assignment.has_gaps = _check_data_gaps(df.index)

        years = assignment.years_of_data
        if years >= min_tier_1_years:
            assignment.tier = DataTier.TIER_1
        elif years >= min_tier_2_years:
            assignment.tier = DataTier.TIER_2
        else:
            assignment.tier = DataTier.TIER_3

    except Exception:
        pass

    return assignment


def _load_and_parse_minute_csv(file_path: Path) -> pd.DataFrame:
    """Load and parse a minute CSV file."""
    try:
        df = pd.read_csv(file_path, parse_dates=True)
    except Exception:
        return pd.DataFrame()

    datetime_col = None
    for col in ("datetime", "Datetime", "timestamp", "Timestamp", "date", "Date", "time", "Time"):
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is None:
        return pd.DataFrame()

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(c.lower() for c in df.columns)
    if missing:
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    return df


def _check_data_gaps(index: pd.DatetimeIndex) -> bool:
    """Check if there are significant gaps in data (missing months/quarters)."""
    if len(index) < 2:
        return False

    sorted_idx = index.sort_values()
    diffs = sorted_idx.to_series().diff().dropna()
    large_gaps = diffs[diffs > pd.Timedelta(days=7)]
    return not large_gaps.empty
