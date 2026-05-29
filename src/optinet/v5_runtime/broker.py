"""Broker abstraction for V5 v1 runtime.

- BrokerClient    : abstract base for live data sources
- MockBroker      : replays historical option-chain / spot / futures cache files;
                    used for offline tests and dry-run validation
- UpstoxBroker    : stub for live Upstox API (auth via UPSTOX_ACCESS_TOKEN env var);
                    real API integration left as TODO; signatures are final
- yfinance_spot() : non-critical sanity helper; NEVER used for trade decisions
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAIN_CACHE = REPO_ROOT / "cache" / "optinet_v4" / "chain_features"  # not used directly
RAW_OPTIONS = REPO_ROOT / "data" / "option_data"
INDIA_TZ = "Asia/Kolkata"

LOT_SIZE = {"NIFTY": 50, "BANKNIFTY": 15}
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}


@dataclass
class SpotQuote:
    timestamp: datetime
    symbol: str          # 'NIFTY' or 'BANKNIFTY'
    spot: float


@dataclass
class FuturesQuote:
    timestamp: datetime
    symbol: str
    expiry: date
    fut_close: float
    fut_oi: int
    fut_vol: int


@dataclass
class OptionContract:
    strike: int
    opt_type: str        # 'CE' or 'PE'
    expiry: date
    close: float
    oi: int
    volume: int


@dataclass
class OptionChainSnapshot:
    """Full option chain at a single minute. Used by feature compute."""
    timestamp: datetime
    symbol: str
    expiry: date
    spot: float
    contracts: list[OptionContract] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to the v4_chain-compatible per-minute frame."""
        if not self.contracts:
            return pd.DataFrame()
        return pd.DataFrame([{
            "datetime": self.timestamp,
            "index": self.symbol,
            "strike": c.strike,
            "opt_type": c.opt_type,
            "expiry": c.expiry,
            "close": c.close,
            "oi": c.oi,
            "volume": c.volume,
        } for c in self.contracts])


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BrokerClient(ABC):
    """All trade-decision data MUST come through this interface."""

    @abstractmethod
    def get_spot(self, symbol: str, ts: Optional[datetime] = None) -> SpotQuote:
        """Get the latest (or as-of `ts`) spot quote."""

    @abstractmethod
    def get_futures(self, symbol: str, expiry: date,
                    ts: Optional[datetime] = None) -> FuturesQuote:
        """Get the latest (or as-of `ts`) futures quote for the given expiry."""

    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: date,
                         ts: Optional[datetime] = None) -> OptionChainSnapshot:
        """Get the full option chain for one expiry as-of `ts`."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the broker connection is healthy."""

    @abstractmethod
    def name(self) -> str:
        """Return broker identifier for logs."""


# ---------------------------------------------------------------------------
# Mock broker — reads from cached historical CSVs
# ---------------------------------------------------------------------------

class MockBroker(BrokerClient):
    """Replays historical data files as if they were live broker quotes.

    Use this for:
    - Offline end-to-end pipeline tests
    - Dry-run validation of decision logic
    - Replaying specific past days for reconciliation testing

    Data sources (read-only):
    - data/option_data/{nifty,banknifty}_data/{nifty,banknifty}_spot/YYYY/M/*.csv
    - data/option_data/{nifty,banknifty}_data/{nifty,banknifty}_fut/YYYY/M/*.csv
    - data/option_data/{nifty,banknifty}_data/{nifty,banknifty}_options/YYYY/M/*.csv

    The mock caches one trading day's data per (symbol, date) so repeated
    minute-level lookups in a simulated day are cheap.
    """

    def __init__(self, simulate_date: Optional[date] = None,
                 raw_root: Path = RAW_OPTIONS):
        from optinet.v4_chain import load_spot, load_options_day
        self._load_spot = load_spot
        self._load_options_day = load_options_day
        self._raw = raw_root
        self._sim_date = simulate_date  # if set, override "now" to this date
        self._spot_cache: dict[tuple[str, date], pd.DataFrame] = {}
        self._fut_cache: dict[tuple[str, date], pd.DataFrame] = {}
        self._opt_cache: dict[tuple[str, date], pd.DataFrame] = {}

    def name(self) -> str:
        return "MockBroker"

    def health(self) -> bool:
        return self._raw.exists()

    # --- internal data-loading helpers ----------------------------------

    def _resolve_date(self, ts: Optional[datetime]) -> tuple[datetime, date]:
        if ts is None:
            now = datetime.now()
        else:
            now = ts
        d = self._sim_date if self._sim_date is not None else now.date()
        return now, d

    def _spot_path(self, symbol: str, d: date) -> Path:
        sym_lower = symbol.lower()
        # nifty_spotDD_MM_YYYY.csv (note: no underscore between 'spot' and date)
        return (self._raw / f"{sym_lower}_data" / f"{sym_lower}_spot"
                / f"{d.year}" / f"{d.month}"
                / f"{sym_lower}_spot{d.day:02d}_{d.month:02d}_{d.year}.csv")

    def _fut_path(self, symbol: str, d: date) -> Path:
        sym_lower = symbol.lower()
        return (self._raw / f"{sym_lower}_data" / f"{sym_lower}_fut"
                / f"{d.year}" / f"{d.month}" / f"{sym_lower}_fut_{d.day:02d}_{d.month:02d}_{d.year}.csv")

    def _opt_path(self, symbol: str, d: date) -> Path:
        sym_lower = symbol.lower()
        return (self._raw / f"{sym_lower}_data" / f"{sym_lower}_options"
                / f"{d.year}" / f"{d.month}" / f"{sym_lower}_options_{d.day:02d}_{d.month:02d}_{d.year}.csv")

    def _spot_for_day(self, symbol: str, d: date) -> pd.DataFrame:
        key = (symbol, d)
        if key not in self._spot_cache:
            p = self._spot_path(symbol, d)
            if not p.exists():
                raise FileNotFoundError(f"MockBroker: spot file missing: {p}")
            self._spot_cache[key] = self._load_spot(p, symbol)
        return self._spot_cache[key]

    def _fut_for_day(self, symbol: str, d: date) -> pd.DataFrame:
        from optinet.v5_futures import load_fut_day
        key = (symbol, d)
        if key not in self._fut_cache:
            p = self._fut_path(symbol, d)
            if not p.exists():
                raise FileNotFoundError(f"MockBroker: fut file missing: {p}")
            self._fut_cache[key] = load_fut_day(p, symbol)
        return self._fut_cache[key]

    def _opt_for_day(self, symbol: str, d: date) -> pd.DataFrame:
        key = (symbol, d)
        if key not in self._opt_cache:
            p = self._opt_path(symbol, d)
            if not p.exists():
                raise FileNotFoundError(f"MockBroker: options file missing: {p}")
            self._opt_cache[key] = self._load_options_day(p, symbol)
        return self._opt_cache[key]

    # --- public API -----------------------------------------------------

    def get_spot(self, symbol: str, ts: Optional[datetime] = None) -> SpotQuote:
        now, d = self._resolve_date(ts)
        df = self._spot_for_day(symbol, d)
        target = pd.Timestamp(now)
        # Find most recent bar at or before target
        sub = df[df["datetime"] <= target]
        if sub.empty:
            raise RuntimeError(f"No spot bars before {target}")
        row = sub.iloc[-1]
        return SpotQuote(timestamp=row["datetime"].to_pydatetime(),
                         symbol=symbol,
                         spot=float(row["spot"]))

    def get_futures(self, symbol: str, expiry: date,
                    ts: Optional[datetime] = None) -> FuturesQuote:
        # Mock data has only the front-month contract; expiry param is informational
        now, d = self._resolve_date(ts)
        df = self._fut_for_day(symbol, d)
        target = pd.Timestamp(now)
        sub = df[df["datetime"] <= target]
        if sub.empty:
            raise RuntimeError(f"No fut bars before {target}")
        row = sub.iloc[-1]
        return FuturesQuote(
            timestamp=row["datetime"].to_pydatetime(),
            symbol=symbol,
            expiry=expiry,
            fut_close=float(row["fut_close"]),
            fut_oi=int(row["fut_oi"]),
            fut_vol=int(row["fut_vol"]),
        )

    def get_option_chain(self, symbol: str, expiry: date,
                         ts: Optional[datetime] = None) -> OptionChainSnapshot:
        now, d = self._resolve_date(ts)
        df = self._opt_for_day(symbol, d)
        target = pd.Timestamp(now)
        # Filter to the requested expiry and most-recent minute at/before target
        df = df[df["expiry"] == expiry]
        if df.empty:
            raise RuntimeError(f"No chain rows for expiry={expiry} on {d}")
        # Pick the latest minute we have data for at or before target
        avail_mins = df["datetime"][df["datetime"] <= target].unique()
        if len(avail_mins) == 0:
            raise RuntimeError(f"No chain rows before {target}")
        chosen = sorted(avail_mins)[-1]
        snap_df = df[df["datetime"] == chosen]

        spot_quote = self.get_spot(symbol, ts)
        contracts = [
            OptionContract(
                strike=int(r["strike"]),
                opt_type=str(r["opt_type"]),
                expiry=r["expiry"],
                close=float(r["close"]),
                oi=int(r["oi"]) if pd.notna(r["oi"]) else 0,
                volume=int(r["volume"]) if pd.notna(r["volume"]) else 0,
            )
            for _, r in snap_df.iterrows()
        ]
        return OptionChainSnapshot(
            timestamp=chosen.to_pydatetime(),
            symbol=symbol,
            expiry=expiry,
            spot=spot_quote.spot,
            contracts=contracts,
        )

    # --- mock-only helper for end-to-end testing ------------------------

    def get_full_day_chain(self, symbol: str, expiry: date,
                           through: Optional[datetime] = None) -> pd.DataFrame:
        """Return the full intraday options frame up through `through`,
        in the schema expected by `compute_minute_features`."""
        now, d = self._resolve_date(through)
        df = self._opt_for_day(symbol, d)
        df = df[df["expiry"] == expiry]
        if through is not None:
            df = df[df["datetime"] <= pd.Timestamp(through)]
        return df.reset_index(drop=True)

    def get_full_day_spot(self, symbol: str,
                          through: Optional[datetime] = None) -> pd.DataFrame:
        now, d = self._resolve_date(through)
        df = self._spot_for_day(symbol, d)
        if through is not None:
            df = df[df["datetime"] <= pd.Timestamp(through)]
        return df.reset_index(drop=True)

    def get_full_day_fut(self, symbol: str,
                         through: Optional[datetime] = None) -> pd.DataFrame:
        now, d = self._resolve_date(through)
        df = self._fut_for_day(symbol, d)
        if through is not None:
            df = df[df["datetime"] <= pd.Timestamp(through)]
        return df.reset_index(drop=True)

    def get_account_state(self) -> dict:
        """MockBroker is always flat — no real positions exist."""
        return {"is_flat": True, "open_positions": [], "pending_orders": []}

    def resolve_instrument(self, symbol: str, expiry: date) -> str:
        """MockBroker returns a synthetic instrument key."""
        return f"MOCK_FO|{symbol.upper()}{expiry.strftime('%d%b%y').upper()}FUT"


# ---------------------------------------------------------------------------
# Upstox broker stub
# ---------------------------------------------------------------------------

class UpstoxBroker(BrokerClient):
    """Production broker. Real API calls are stubbed with NotImplementedError.

    Setup checklist (operator):
    1. Register an Upstox developer account at https://upstox.com/developer/
    2. Create an app with redirect URI matching local callback
    3. Set env vars before running:
         export UPSTOX_API_KEY=...
         export UPSTOX_API_SECRET=...
         export UPSTOX_ACCESS_TOKEN=...   (refresh daily via OAuth)
    4. `pip install upstox-python-sdk` (add to requirements before live)
    5. Replace each NotImplementedError below with the SDK call

    Endpoints needed:
    - /market-quote/quotes for spot LTP
    - /market-quote/quotes for futures LTP + OI
    - /option/chain or /market-quote/option-chain for full chain

    yfinance is NOT used here. yfinance is only allowed in:
    - scripts/v5_premarket.py (cross-check spot vs broker for sanity)
    - scripts/v5_drift_check.py (offline VIX history)
    """

    def __init__(self):
        self.api_key = os.environ.get("UPSTOX_API_KEY")
        self.access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not self.api_key or not self.access_token:
            # Allow construction but health() will return False
            pass

    def name(self) -> str:
        return "UpstoxBroker"

    def health(self) -> bool:
        if not self.api_key or not self.access_token:
            return False
        # TODO: ping a known-cheap Upstox endpoint (e.g. user/profile)
        # Return True only if HTTP 200
        return False  # remains False until SDK is wired

    def get_spot(self, symbol: str, ts: Optional[datetime] = None) -> SpotQuote:
        raise NotImplementedError(
            "UpstoxBroker.get_spot: wire up upstox.MarketQuoteApi.ltp() with "
            "instrument key 'NSE_INDEX|Nifty 50' / 'NSE_INDEX|Nifty Bank' "
            "and return the last_price."
        )

    def get_futures(self, symbol: str, expiry: date,
                    ts: Optional[datetime] = None) -> FuturesQuote:
        raise NotImplementedError(
            "UpstoxBroker.get_futures: wire up upstox.MarketQuoteApi.full_market_quote() "
            "with the futures instrument key. Return last_price, oi, volume."
        )

    def get_option_chain(self, symbol: str, expiry: date,
                         ts: Optional[datetime] = None) -> OptionChainSnapshot:
        raise NotImplementedError(
            "UpstoxBroker.get_option_chain: wire up upstox.OptionApi.get_put_call_option_chain() "
            "for (instrument_key, expiry_date) and translate each strike row into "
            "an OptionContract."
        )

    def get_account_state(self) -> dict:
        """Return {'is_flat': bool, 'open_positions': [...], 'pending_orders': [...]}.

        Wire up:
          positions = upstox.PortfolioApi.get_positions()
          orders    = upstox.OrderApi.get_order_book()
        is_flat = no open positions AND no pending/open orders.
        """
        raise NotImplementedError(
            "UpstoxBroker.get_account_state: wire up "
            "upstox.PortfolioApi.get_positions() and "
            "upstox.OrderApi.get_order_book(). "
            "Return {'is_flat': bool, 'open_positions': [...], 'pending_orders': [...]}"
        )

    def resolve_instrument(self, symbol: str, expiry: date) -> str:
        """Return the Upstox instrument key for a NIFTY futures contract.

        Example: 'NSE_FO|NIFTY26MAY25FUT'

        Wire up: fetch instrument master from
          https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz
        and look up name=symbol, expiry=expiry, instrument_type='FUT'.
        Cache the result for the trading day.
        """
        raise NotImplementedError(
            "UpstoxBroker.resolve_instrument: download the Upstox instrument master "
            f"and look up the futures key for symbol={symbol!r}, expiry={expiry}."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_broker(kind: Optional[str] = None,
                simulate_date: Optional[date] = None) -> BrokerClient:
    """Construct a broker client.

    Resolution order:
      1. `kind` argument
      2. OPTINET_BROKER env var
      3. default = 'mock'

    For paper trading: leave OPTINET_BROKER unset → defaults to mock.
    For live: set OPTINET_BROKER=upstox (and the UPSTOX_* credentials).
    """
    kind = (kind or os.environ.get("OPTINET_BROKER") or "mock").lower()
    if kind == "mock":
        return MockBroker(simulate_date=simulate_date)
    if kind == "upstox":
        return UpstoxBroker()
    raise ValueError(f"Unknown broker kind: {kind!r}")


# ---------------------------------------------------------------------------
# yfinance fallback (sanity / monitoring only — NOT for trade decisions)
# ---------------------------------------------------------------------------

def yfinance_spot_sanity(symbol: str) -> Optional[float]:
    """Best-effort yfinance spot fetch for cross-check.

    Returns None if yfinance is unavailable or fails. Caller must handle None.
    NEVER call this from trade-decision code paths.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None
    ticker = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}.get(symbol)
    if ticker is None:
        return None
    try:
        h = yf.Ticker(ticker).history(period="1d", interval="1m")
        if h.empty:
            return None
        return float(h["Close"].iloc[-1])
    except Exception:
        return None
