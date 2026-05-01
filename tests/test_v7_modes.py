from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intradaynet.v7_modes import compute_post_open_adjustment, extract_post_open_session


def test_extract_post_open_session_keeps_bars_through_cutoff():
    index = pd.to_datetime(
        [
            "2026-04-22 09:14:00",
            "2026-04-22 09:15:00",
            "2026-04-22 09:16:00",
            "2026-04-22 09:30:00",
            "2026-04-22 09:31:00",
        ]
    )
    minute_df = pd.DataFrame(
        {
            "open": [99, 100, 101, 102, 103],
            "high": [99, 101, 102, 103, 104],
            "low": [98, 99, 100, 101, 102],
            "close": [99, 100.5, 101.5, 102.5, 103.5],
            "volume": [1, 10, 10, 10, 10],
        },
        index=index,
    )

    session = extract_post_open_session(minute_df, pd.Timestamp("2026-04-22"), "09:30")

    assert list(session.index.strftime("%H:%M")) == ["09:15", "09:16", "09:30"]


def test_post_open_adjustment_uses_cutoff_close_as_reference():
    index = pd.to_datetime(
        [
            "2026-04-21 09:15:00",
            "2026-04-21 09:16:00",
            "2026-04-22 09:15:00",
            "2026-04-22 09:16:00",
            "2026-04-22 09:30:00",
        ]
    )
    minute_df = pd.DataFrame(
        {
            "open": [98, 99, 100, 101, 102],
            "high": [99, 100, 101, 103, 104],
            "low": [97, 98, 99, 100, 101],
            "close": [98.5, 99.5, 100.5, 102.0, 103.0],
            "volume": [100, 100, 100, 100, 200],
        },
        index=index,
    )
    session_df = extract_post_open_session(minute_df, pd.Timestamp("2026-04-22"), "09:30")
    feature_row = pd.Series({"market_breadth": 0.2, "risk_on_signal": 0.1, "sector_relative_strength": 0.15})

    adjusted = compute_post_open_adjustment(
        direction="LONG",
        prev_close=99.5,
        base_probability=0.6,
        predicted_magnitude=0.01,
        session_df=session_df,
        minute_df=minute_df,
        feature_row=feature_row,
    )

    assert adjusted["reference_price"] == 103.0
    assert adjusted["cutoff_close"] == 103.0
    assert adjusted["cutoff_timestamp"] == "2026-04-22T09:30:00"
