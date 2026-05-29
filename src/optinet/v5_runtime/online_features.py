"""Online feature compute for V5 v1 runtime.

Reuses the same compute_minute_features (V4-A) and compute_fut_features (V5-B)
as the offline training pipeline, fed by a rolling buffer of broker quotes.

The class to use is `OnlineFeatureBuilder`. Each trading day:
1. Construct one builder per (symbol, expiry).
2. At market open, call `warmup_to(market_open_ts)` once.
3. Each minute, call `step(ts)` to fetch new broker quotes and append.
4. Call `compute_at(ts)` to get a feature dict for the latest minute.

`compute_at` returns a dict containing every column in:
- CHAIN_FEATURES, FUT_FEATURES, SIM_CONTEXT  (used by gate model)
- VOL_FEATURES                              (used by V4-B vol kill-switch)

If any required field is NaN/missing, returns None.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from optinet.v4_chain import compute_minute_features
from optinet.v5_futures import compute_fut_features
from optinet.v5_runtime.broker import BrokerClient, MockBroker

# These lists must match the training-time feature set exactly.
CHAIN_FEATURES = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
]
FUT_FEATURES = [
    "fut_basis", "fut_basis_change_30m",
    "fut_oi_change_1m", "fut_oi_change_5m", "fut_oi_change_30m",
    "fut_volume_oi_ratio", "fut_session_position",
    "fut_oi_x_long_buildup", "fut_oi_x_short_buildup",
    "fut_oi_x_short_cover", "fut_oi_x_long_unwind",
]
SIM_CONTEXT = ["minute_of_day", "hour_of_day", "dte"]

# Vol model needs lags over a 15-min window plus minutes_to_close.
LAG_BASE = ["atm_iv", "skew_slope", "pcr_oi", "pcr_vol",
             "iv_rv_spread", "realized_vol_30m", "max_oi_total_dist_pct"]
VOL_FEATURES = (
    CHAIN_FEATURES + ["minute_of_day", "minutes_to_close"]
    + [f"{f}_lag5" for f in LAG_BASE]
    + [f"{f}_lag15" for f in LAG_BASE]
)
GATE_FEATURES = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
MINUTES_PER_SESSION = 375  # 09:15 → 15:30


def _bucket_dte(dte_days: int) -> int:
    """Map raw days-to-expiry into the gate's dte_bucket categorical."""
    if dte_days <= 0:
        return 0
    if dte_days <= 2:
        return 1
    if dte_days <= 4:
        return 2
    return 3


@dataclass
class FeatureSnapshot:
    """Result of a single minute's feature compute."""
    timestamp: datetime
    symbol: str
    expiry: date
    spot: float
    atm_strike: int
    atm_iv: float
    atm_straddle_premium_per_share: float
    dte_days: int
    dte_bucket: int
    gate_features: pd.DataFrame   # 1-row frame, columns = GATE_FEATURES
    vol_features: pd.DataFrame    # 1-row frame, columns = VOL_FEATURES
    raw_chain_minute: pd.Series   # full chain feature row (debug / logging)


class OnlineFeatureBuilder:
    """Rolling feature buffer for one (symbol, expiry).

    Fetches broker quotes minute-by-minute and re-runs the offline feature
    compute on the accumulated frames. Cheap because compute_minute_features
    is vectorized over the day's groupby.
    """

    def __init__(self, broker: BrokerClient, symbol: str, expiry: date):
        self.broker = broker
        self.symbol = symbol
        self.expiry = expiry
        # Rolling buffers in the schema expected by the offline compute fns
        self.spot_df = pd.DataFrame(columns=["datetime", "spot"])
        self.fut_df = pd.DataFrame(columns=["index", "datetime", "fut_close",
                                              "fut_oi", "fut_vol"])
        self.opt_df = pd.DataFrame(columns=["datetime", "index", "strike",
                                              "opt_type", "expiry", "close",
                                              "oi", "volume"])
        self._last_ts: Optional[datetime] = None

    # ------------------------------------------------------------------ buffer
    def _append_spot(self, ts: datetime, spot: float) -> None:
        if (not self.spot_df.empty
                and self.spot_df["datetime"].iloc[-1] >= pd.Timestamp(ts)):
            return  # already have this minute
        self.spot_df.loc[len(self.spot_df)] = {
            "datetime": pd.Timestamp(ts), "spot": spot,
        }

    def _append_fut(self, ts: datetime, fc: float, oi: int, vol: int) -> None:
        if (not self.fut_df.empty
                and self.fut_df["datetime"].iloc[-1] >= pd.Timestamp(ts)):
            return
        self.fut_df.loc[len(self.fut_df)] = {
            "index": self.symbol,
            "datetime": pd.Timestamp(ts),
            "fut_close": fc, "fut_oi": oi, "fut_vol": vol,
        }

    def _append_chain(self, snap) -> None:
        if (not self.opt_df.empty
                and self.opt_df["datetime"].max() >= pd.Timestamp(snap.timestamp)):
            return
        rows = [{
            "datetime": pd.Timestamp(snap.timestamp),
            "index": self.symbol,
            "strike": c.strike,
            "opt_type": c.opt_type,
            "expiry": c.expiry,
            "close": c.close,
            "oi": c.oi,
            "volume": c.volume,
        } for c in snap.contracts]
        if rows:
            self.opt_df = pd.concat(
                [self.opt_df, pd.DataFrame(rows)], ignore_index=True
            )

    # -------------------------------------------------------------- public API
    def step(self, ts: datetime) -> bool:
        """Pull the latest minute of data from the broker and append.

        Returns False if any fetch failed (caller should treat as NO_TRADE_DATA).
        """
        try:
            sq = self.broker.get_spot(self.symbol, ts)
            fq = self.broker.get_futures(self.symbol, self.expiry, ts)
            ch = self.broker.get_option_chain(self.symbol, self.expiry, ts)
        except Exception as exc:  # noqa: BLE001
            # Caller logs; this is a data-feed failure
            self._last_error = repr(exc)
            return False

        # Only accept quotes whose timestamp matches the requested minute
        # (broker may return a stale bar; we tolerate up to 1-min staleness)
        target = pd.Timestamp(ts)
        if abs((pd.Timestamp(sq.timestamp) - target).total_seconds()) > 90:
            self._last_error = f"stale spot: {sq.timestamp} vs target {ts}"
            return False

        self._append_spot(sq.timestamp, sq.spot)
        self._append_fut(fq.timestamp, fq.fut_close, fq.fut_oi, fq.fut_vol)
        self._append_chain(ch)
        self._last_ts = ts
        return True

    def warmup_to(self, ts: datetime, start: Optional[datetime] = None,
                   freq_minutes: int = 1) -> int:
        """Warm up the buffer by stepping from market_open through `ts`.

        For MockBroker this is fast. For UpstoxBroker, prefer to subscribe to
        a live tick stream from market open and call `step` per minute instead;
        warmup_to should only be used for catching up after a startup delay.
        """
        d = ts.date() if start is None else start.date()
        cursor = (datetime.combine(d, MARKET_OPEN) if start is None
                  else start)
        end = ts
        n = 0
        while cursor <= end:
            if self.step(cursor):
                n += 1
            cursor += timedelta(minutes=freq_minutes)
        return n

    def compute_at(self, ts: datetime) -> Optional[FeatureSnapshot]:
        """Compute all features at minute `ts`. Returns None on insufficient data."""
        if self.opt_df.empty or self.spot_df.empty or self.fut_df.empty:
            return None

        # Run the same offline functions
        chain_feats = compute_minute_features(self.opt_df, self.spot_df, self.symbol)
        fut_feats = compute_fut_features(self.fut_df, self.spot_df)
        if chain_feats.empty or fut_feats.empty:
            return None

        # Locate the row for ts (or most-recent row at/before ts)
        target = pd.Timestamp(ts)
        chain_feats = chain_feats[chain_feats["datetime"] <= target]
        fut_feats = fut_feats[fut_feats["datetime"] <= target]
        if chain_feats.empty or fut_feats.empty:
            return None

        # Latest row per source
        cur_chain = chain_feats.iloc[-1]
        cur_fut = fut_feats[fut_feats["datetime"] == cur_chain["datetime"]]
        if cur_fut.empty:
            # fall back to nearest fut row
            cur_fut = fut_feats.iloc[-1:]
        cur_fut = cur_fut.iloc[0]

        # Time / context features
        minute_of_day = (cur_chain["datetime"].time().hour * 60
                         + cur_chain["datetime"].time().minute) - 555  # 09:15 = 555 min
        hour_of_day = cur_chain["datetime"].time().hour
        dte_days = (self.expiry - cur_chain["datetime"].date()).days
        # minutes-to-close (15:30 IST close)
        minutes_to_close = (datetime.combine(cur_chain["datetime"].date(), MARKET_CLOSE)
                             - cur_chain["datetime"].to_pydatetime()).total_seconds() / 60.0

        # Compute lags over the rolling chain history
        chain_h = chain_feats.copy().reset_index(drop=True)
        for f in LAG_BASE:
            chain_h[f"{f}_lag5"] = chain_h[f].shift(5)
            chain_h[f"{f}_lag15"] = chain_h[f].shift(15)
        cur_lags = chain_h.iloc[-1]

        # Build gate-feature row
        gate_row = {f: cur_chain.get(f, np.nan) for f in CHAIN_FEATURES}
        gate_row.update({f: cur_fut.get(f, np.nan) for f in FUT_FEATURES})
        gate_row["minute_of_day"] = int(minute_of_day)
        gate_row["hour_of_day"] = int(hour_of_day)
        gate_row["dte"] = int(dte_days)

        # Build vol-feature row
        vol_row = {f: cur_chain.get(f, np.nan) for f in CHAIN_FEATURES}
        vol_row["minute_of_day"] = int(minute_of_day)
        vol_row["minutes_to_close"] = float(minutes_to_close)
        for f in LAG_BASE:
            vol_row[f"{f}_lag5"] = cur_lags.get(f"{f}_lag5", np.nan)
            vol_row[f"{f}_lag15"] = cur_lags.get(f"{f}_lag15", np.nan)

        return FeatureSnapshot(
            timestamp=cur_chain["datetime"].to_pydatetime(),
            symbol=self.symbol,
            expiry=self.expiry,
            spot=float(cur_chain["spot"]),
            atm_strike=int(cur_chain["atm_strike"]),
            atm_iv=float(cur_chain["atm_iv"]),
            atm_straddle_premium_per_share=float(cur_chain["atm_straddle_premium"]),
            dte_days=int(dte_days),
            dte_bucket=_bucket_dte(int(dte_days)),
            gate_features=pd.DataFrame([gate_row], columns=GATE_FEATURES),
            vol_features=pd.DataFrame([vol_row], columns=VOL_FEATURES),
            raw_chain_minute=cur_chain,
        )
