from __future__ import annotations

from datetime import date

import pandas as pd

from optinet.data_lake import parse_bhavcopy_bytes


def _parse_fo_bhavcopy(csv_text: str, trade_date: date) -> pd.DataFrame | None:
    parsed = parse_bhavcopy_bytes(csv_text, trade_date)
    if parsed.empty:
        return None
    return parsed.rename(
        columns={
            "symbol": "index_name",
            "expiry_date": "expiry_date",
            "strike_price": "strike_price",
        }
    )
