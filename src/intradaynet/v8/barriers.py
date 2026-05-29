"""
Path-dependent barrier targets for V8.

Answers the question: "Will this stock hit +X% before -Y% today?"

Barrier target = 1 if price hits open * (1 + target_pct) before hitting
open * (1 - stop_pct) during the day. More actionable than point-to-point 
returns since it maps 1:1 to how trade recommendations are actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarrierTarget:
    """Result of barrier computation for one stock-day."""
    symbol: str
    date: pd.Timestamp
    open_price: float
    high_price: float
    low_price: float
    close_price: float

    # Barrier levels
    long_target_level: float      # open * (1 + target_pct)
    long_stop_level: float        # open * (1 - stop_pct)
    short_target_level: float     # open * (1 - target_pct)
    short_stop_level: float       # open * (1 + stop_pct)

    # Hit timing (in minutes from market open)
    long_target_hit: bool           # did price reach long target?
    long_stop_hit: bool             # did price reach long stop?
    short_target_hit: bool          # did price reach short target?
    short_stop_hit: bool            # did price reach short stop?

    long_target_hit_minute: Optional[int]  # minute bar index when hit
    long_stop_hit_minute: Optional[int]
    short_target_hit_minute: Optional[int]
    short_stop_hit_minute: Optional[int]

    # Final labels
    long_label: int  # 1 = hit target before stop, 0 = hit stop first, -1 = neither
    short_label: int  # 1 = hit target before stop, 0 = hit stop first, -1 = neither
    label: Literal["LONG", "SHORT", "NEUTRAL"]  # primary label: which side won

    # Metadata
    target_pct: float
    stop_pct: float
    total_bars: int                 # total minute bars in this session
    first_touch_bar: Optional[int]  # bar index of first barrier hit (any side)

    def is_tradable_long(self) -> bool:
        """True if LONG target hit before stop."""
        return self.long_label == 1

    def is_tradable_short(self) -> bool:
        """True if SHORT target hit before stop."""
        return self.short_label == 1

    def is_actionable(self) -> bool:
        """True if either side reached target before stop."""
        return self.is_tradable_long() or self.is_tradable_short()

    def time_to_first_target(self) -> Optional[int]:
        """Minutes until first barrier hit (any side)."""
        return self.first_touch_bar

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "date": str(self.date.date()),
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "long_target_level": self.long_target_level,
            "long_stop_level": self.long_stop_level,
            "short_target_level": self.short_target_level,
            "short_stop_level": self.short_stop_level,
            "long_target_hit": self.long_target_hit,
            "long_stop_hit": self.long_stop_hit,
            "short_target_hit": self.short_target_hit,
            "short_stop_hit": self.short_stop_hit,
            "long_target_hit_minute": self.long_target_hit_minute,
            "long_stop_hit_minute": self.long_stop_hit_minute,
            "short_target_hit_minute": self.short_target_hit_minute,
            "short_stop_hit_minute": self.short_stop_hit_minute,
            "long_label": self.long_label,
            "short_label": self.short_label,
            "label": self.label,
            "target_pct": self.target_pct,
            "stop_pct": self.stop_pct,
            "total_bars": self.total_bars,
            "first_touch_bar": self.first_touch_bar,
        }


def compute_barrier_targets(
    session_df: pd.DataFrame,
    symbol: str,
    *,
    target_pct: float = 0.015,
    stop_pct: float = 0.010,
    min_bars: int = 200,
) -> Optional[BarrierTarget]:
    """
    Compute path-dependent barrier targets for one stock-day session.

    Scans minute-by-minute to determine which barrier (target or stop)
    is hit first for both LONG and SHORT directions.

    Parameters
    ----------
    session_df : pd.DataFrame
        Minute-level OHLCV data for one trading session.
        Must have columns: open, high, low, close.
        Index must be datetime.
    symbol : str
        Stock symbol (e.g., "RELIANCE").
    target_pct : float
        Target percentage (e.g., 0.015 for 1.5%).
    stop_pct : float
        Stop-loss percentage (e.g., 0.010 for 1.0%).
    min_bars : int
        Minimum number of minute bars required for a valid session.

    Returns
    -------
    BarrierTarget or None
        Target result or None if session has insufficient data.
    """
    session_df = session_df.loc[session_df.index.min():session_df.index.max()].copy()
    session_df = session_df.sort_index()

    if len(session_df) < min_bars:
        return None

    open_price = session_df["open"].iloc[0]
    if open_price <= 0 or np.isnan(open_price):
        return None

    long_target = open_price * (1.0 + target_pct)
    long_stop = open_price * (1.0 - stop_pct)
    short_target = open_price * (1.0 - target_pct)
    short_stop = open_price * (1.0 + stop_pct)

    highs = session_df["high"].values.astype(np.float64)
    lows = session_df["low"].values.astype(np.float64)
    close = session_df["close"].iloc[-1]
    n_bars = len(session_df)
    date = session_df.index[0].normalize()

    long_target_hit = False
    long_stop_hit = False
    short_target_hit = False
    short_stop_hit = False

    long_target_hit_bar: Optional[int] = None
    long_stop_hit_bar: Optional[int] = None
    short_target_hit_bar: Optional[int] = None
    short_stop_hit_bar: Optional[int] = None

    first_touch_bar: Optional[int] = None

    for i in range(n_bars):
        bar_high = highs[i]
        bar_low = lows[i]

        if bar_high >= long_target and not long_target_hit:
            long_target_hit = True
            long_target_hit_bar = i
            if first_touch_bar is None:
                first_touch_bar = i

        if bar_low <= long_stop and not long_stop_hit:
            long_stop_hit = True
            long_stop_hit_bar = i
            if first_touch_bar is None:
                first_touch_bar = i

        if bar_low <= short_target and not short_target_hit:
            short_target_hit = True
            short_target_hit_bar = i
            if first_touch_bar is None:
                first_touch_bar = i

        if bar_high >= short_stop and not short_stop_hit:
            short_stop_hit = True
            short_stop_hit_bar = i
            if first_touch_bar is None:
                first_touch_bar = i

    long_label = _resolve_label(long_target_hit, long_stop_hit, long_target_hit_bar, long_stop_hit_bar)
    short_label = _resolve_label(short_target_hit, short_stop_hit, short_target_hit_bar, short_stop_hit_bar)

    label = _primary_label(long_label, short_label)

    return BarrierTarget(
        symbol=symbol,
        date=date,
        open_price=open_price,
        high_price=float(session_df["high"].max()),
        low_price=float(session_df["low"].min()),
        close_price=float(close),
        long_target_level=long_target,
        long_stop_level=long_stop,
        short_target_level=short_target,
        short_stop_level=short_stop,
        long_target_hit=long_target_hit,
        long_stop_hit=long_stop_hit,
        short_target_hit=short_target_hit,
        short_stop_hit=short_stop_hit,
        long_target_hit_minute=long_target_hit_bar,
        long_stop_hit_minute=long_stop_hit_bar,
        short_target_hit_minute=short_target_hit_bar,
        short_stop_hit_minute=short_stop_hit_bar,
        long_label=long_label,
        short_label=short_label,
        label=label,
        target_pct=target_pct,
        stop_pct=stop_pct,
        total_bars=n_bars,
        first_touch_bar=first_touch_bar,
    )


def compute_barrier_targets_batch(
    sessions: dict[pd.Timestamp, pd.DataFrame],
    symbol: str,
    *,
    target_pct: float = 0.015,
    stop_pct: float = 0.010,
    min_bars: int = 200,
) -> list[BarrierTarget]:
    """
    Compute barrier targets for all sessions of one stock.

    Parameters
    ----------
    sessions : dict
        Mapping from date to minute DataFrame.
    symbol : str
        Stock symbol.
    target_pct, stop_pct, min_bars : see compute_barrier_targets.

    Returns
    -------
    list[BarrierTarget]
        One result per valid session.
    """
    results = []
    for date, df in sorted(sessions.items()):
        result = compute_barrier_targets(
            df, symbol,
            target_pct=target_pct, stop_pct=stop_pct, min_bars=min_bars,
        )
        if result is not None:
            results.append(result)
    return results


def barrier_targets_to_dataframe(targets: list[BarrierTarget]) -> pd.DataFrame:
    """Convert a list of BarrierTarget to a DataFrame."""
    if not targets:
        return pd.DataFrame()
    return pd.DataFrame([t.to_dict() for t in targets])


def barrier_label_distribution(targets: list[BarrierTarget]) -> dict[str, int]:
    """Count of LONG/SHORT/NEUTRAL labels."""
    counts = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
    for t in targets:
        counts[t.label] = counts.get(t.label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_label(
    target_hit: bool,
    stop_hit: bool,
    target_bar: Optional[int],
    stop_bar: Optional[int],
) -> int:
    """Resolve barrier label: 1 = target first, 0 = stop first, -1 = neither."""
    if target_hit and stop_hit:
        tb = target_bar if target_bar is not None else 99999
        sb = stop_bar if stop_bar is not None else 99999
        return 1 if tb < sb else 0
    elif target_hit:
        return 1
    elif stop_hit:
        return 0
    else:
        return -1


def _primary_label(long_label: int, short_label: int) -> Literal["LONG", "SHORT", "NEUTRAL"]:
    """Determine the primary trade direction based on barrier labels."""
    long_winner = long_label == 1
    short_winner = short_label == 1

    if long_winner and not short_winner:
        return "LONG"
    elif short_winner and not long_winner:
        return "SHORT"
    elif long_winner and short_winner:
        # Both hit target before stop — rare, pick the faster one
        return "LONG"  # bias toward long in ambiguous case
    elif long_label == 0 and short_label != 0:
        return "SHORT"  # long stopped out, short might have won
    elif short_label == 0 and long_label != 0:
        return "LONG"  # short stopped out, long might have won
    else:
        return "NEUTRAL"


def compute_multi_horizon_barriers(
    session_df: pd.DataFrame,
    symbol: str,
    *,
    horizons: tuple[str, ...] = ("H30", "H60", "H375"),
    target_pct: float = 0.015,
    stop_pct: float = 0.010,
    min_bars: int = 200,
) -> dict[str, Optional[BarrierTarget]]:
    """
    Compute barrier targets for multiple horizons within one session.

    Slices the session into progressively shorter windows starting from
    market open. For example, H30 uses the first 30 bars.

    Parameters
    ----------
    session_df : pd.DataFrame
        Full session minute data.
    symbol : str
        Stock symbol.
    horizons : tuple
        Horizon labels (e.g., "H30", "H60", "H375"). Must map to minute counts.
    target_pct, stop_pct, min_bars : see compute_barrier_targets.

    Returns
    -------
    dict[str, Optional[BarrierTarget]]
        Mapping from horizon label to target result.
    """
    horizon_bars = {
        "H15": 15,
        "H30": 30,
        "H60": 60,
        "H120": 120,
        "H180": 180,
        "H240": 240,
        "H375": 375,
    }

    results = {}
    for horizon in horizons:
        bars = horizon_bars.get(horizon, 375)
        session_slice = session_df.iloc[:min(bars, len(session_df))]
        result = compute_barrier_targets(
            session_slice, symbol,
            target_pct=target_pct, stop_pct=stop_pct,
            min_bars=min(min_bars, bars),
        )
        results[horizon] = result

    return results
