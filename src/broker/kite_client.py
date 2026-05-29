"""Zerodha Kite Connect client.

Handles:
- Headless daily login (TOTP, no browser)
- Instrument token resolution for futures/spot
- Historical minute candle fetch with OI
- Access token persistence to .env

Usage:
    from broker.kite_client import KiteClient
    kite = KiteClient.from_env()
    kite.login()
    df = kite.historical_minute("NIFTY", "FUT", from_dt, to_dt)
"""
from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pyotp
import requests
from kiteconnect import KiteConnect

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

# Kite login endpoints (not in pykiteconnect SDK)
_LOGIN_URL = "https://kite.zerodha.com/api/login"
_TWOFA_URL = "https://kite.zerodha.com/api/twofa"


def _write_env(key: str, value: str) -> None:
    """Upsert a key=value line in .env."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n")


def _load_env() -> None:
    """Load .env into os.environ (simple parser, no dotenv dep)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class KiteClient:
    """Thin wrapper around KiteConnect with headless login."""

    def __init__(self, api_key: str, api_secret: str,
                 user_id: str, password: str, totp_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.user_id = user_id
        self.password = password
        self.totp_secret = totp_secret
        self.kite = KiteConnect(api_key=api_key)
        self._instruments: Optional[pd.DataFrame] = None

    @classmethod
    def from_env(cls) -> "KiteClient":
        _load_env()
        return cls(
            api_key=os.environ["KITE_API_KEY"],
            api_secret=os.environ["KITE_API_SECRET"],
            user_id=os.environ["KITE_USER_ID"],
            password=os.environ["KITE_PASSWORD"],
            totp_secret=os.environ["KITE_TOTP_SECRET"],
        )

    # ── Login ──────────────────────────────────────────────────────────────

    def login(self) -> str:
        """Headless TOTP login. Returns access_token and saves to .env."""
        # Check if existing token is still valid
        existing = os.environ.get("KITE_ACCESS_TOKEN")
        if existing:
            try:
                self.kite.set_access_token(existing)
                self.kite.profile()  # will raise if token expired
                print("  Kite: existing access_token still valid")
                return existing
            except Exception:
                pass

        session = requests.Session()
        session.headers.update({"X-Kite-Version": "3"})

        # Step 1: password login
        r = session.post(_LOGIN_URL, data={
            "user_id": self.user_id,
            "password": self.password,
        })
        r.raise_for_status()
        data = r.json()["data"]
        request_id = data["request_id"]

        # Step 2: TOTP 2FA
        totp = pyotp.TOTP(self.totp_secret).now()
        r = session.post(_TWOFA_URL, data={
            "user_id": self.user_id,
            "request_id": request_id,
            "twofa_value": totp,
            "twofa_type": "totp",
        })
        r.raise_for_status()

        # Step 3: extract request_token from redirect URL
        login_url = self.kite.login_url()
        r = session.get(login_url, allow_redirects=False)
        redirect = r.headers.get("Location", "")
        m = re.search(r"request_token=([^&]+)", redirect)
        if not m:
            # Follow redirect and parse from final URL
            r2 = session.get(login_url, allow_redirects=True)
            m = re.search(r"request_token=([^&]+)", r2.url)
        if not m:
            raise RuntimeError(f"Could not extract request_token from: {redirect}")
        request_token = m.group(1)

        # Step 4: generate session
        sess = self.kite.generate_session(request_token, api_secret=self.api_secret)
        access_token = sess["access_token"]
        self.kite.set_access_token(access_token)

        _write_env("KITE_ACCESS_TOKEN", access_token)
        os.environ["KITE_ACCESS_TOKEN"] = access_token
        print(f"  Kite: login OK, access_token saved to .env")
        return access_token

    def set_access_token(self, token: str) -> None:
        self.kite.set_access_token(token)

    # ── Instruments ────────────────────────────────────────────────────────

    def _load_instruments(self) -> pd.DataFrame:
        if self._instruments is None:
            raw = self.kite.instruments("NFO")
            self._instruments = pd.DataFrame(raw)
        return self._instruments

    def get_fut_token(self, symbol: str, expiry: Optional[date] = None) -> tuple[int, date]:
        """Return (instrument_token, expiry_date) for the nearest FUT contract."""
        df = self._load_instruments()
        futs = df[(df["name"] == symbol) & (df["instrument_type"] == "FUT")].copy()
        futs["expiry"] = pd.to_datetime(futs["expiry"]).dt.date
        futs = futs[futs["expiry"] >= date.today()].sort_values("expiry")
        if futs.empty:
            raise ValueError(f"No live FUT contracts found for {symbol}")
        if expiry:
            row = futs[futs["expiry"] == expiry].iloc[0]
        else:
            row = futs.iloc[0]  # nearest expiry
        return int(row["instrument_token"]), row["expiry"]

    def get_spot_token(self, symbol: str) -> int:
        """Return instrument_token for NSE index spot."""
        sym_map = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK",
                   "INDIA VIX": "NSE:INDIA VIX"}
        key = sym_map.get(symbol, f"NSE:{symbol}")
        q = self.kite.ltp([key])
        return int(q[key]["instrument_token"])

    # ── Historical data ────────────────────────────────────────────────────

    def historical_minute(
        self,
        instrument_token: int,
        from_dt: datetime,
        to_dt: datetime,
        oi: bool = True,
        interval: str = "minute",
    ) -> pd.DataFrame:
        """Fetch minute candles in 60-day chunks. Returns DataFrame with
        columns: datetime, open, high, low, close, volume[, oi]."""
        chunk_days = 59  # stay under 60-day limit
        frames = []
        cursor = from_dt
        while cursor < to_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days), to_dt)
            try:
                candles = self.kite.historical_data(
                    instrument_token,
                    cursor.strftime("%Y-%m-%d %H:%M:%S"),
                    chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    interval,
                    oi=oi,
                )
                if candles:
                    frames.append(pd.DataFrame(candles))
            except Exception as e:
                print(f"  warn: chunk {cursor.date()}→{chunk_end.date()}: {e}")
            cursor = chunk_end + timedelta(seconds=1)
            time.sleep(0.35)  # ~3 req/sec, stay under rate limit

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return df
