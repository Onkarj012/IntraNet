"""Zerodha Kite Connect client.

Handles:
- Daily browser-based login (opens Kite login URL, catches redirect via local server)
- Instrument token resolution for futures/spot
- Historical minute candle fetch with OI (auto-chunked in 59-day windows)
- Access token persistence to .env

Usage:
    from broker.kite_client import KiteClient
    kite = KiteClient.from_env()
    kite.login()                                          # opens browser once
    df = kite.historical_minute(token, from_dt, to_dt)
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
                 user_id: str, password: str, totp_secret: str = ""):
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
        )

    # ── Login ──────────────────────────────────────────────────────────────

    def login(self) -> str:
        """Headless login with one-shot local server to catch redirect.
        Returns access_token and saves to .env."""
        # Check if existing token is still valid
        existing = os.environ.get("KITE_ACCESS_TOKEN")
        if existing:
            try:
                self.kite.set_access_token(existing)
                self.kite.profile()
                print("  Kite: existing access_token still valid")
                return existing
            except Exception:
                pass

        import threading
        import urllib.parse
        from http.server import BaseHTTPRequestHandler, HTTPServer

        request_token_holder: list[str] = []

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                token = params.get("request_token", [None])[0]
                if token:
                    request_token_holder.append(token)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h2>Login successful. You can close this tab.</h2>")

            def log_message(self, *args):
                pass  # suppress server logs

        server = HTTPServer(("127.0.0.1", 5000), _Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.daemon = True
        thread.start()

        # Open the Kite login URL in the browser
        import webbrowser
        login_url = self.kite.login_url()
        print(f"\n  Opening Kite login in browser...")
        print(f"  URL: {login_url}")
        print(f"  Waiting for redirect to http://127.0.0.1:5000/callback ...")
        webbrowser.open(login_url)

        # Wait up to 120 seconds for the user to log in
        thread.join(timeout=120)
        server.server_close()

        if not request_token_holder:
            raise RuntimeError(
                "Login timed out or redirect not received.\n"
                "Make sure your Kite app redirect URL is set to: http://127.0.0.1:5000/callback"
            )

        request_token = request_token_holder[0]
        print(f"  Got request_token: {request_token[:8]}…")

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
            matching = futs[futs["expiry"] == expiry]
            if matching.empty:
                available = futs["expiry"].tolist()
                raise ValueError(
                    f"No FUT contract for {symbol} expiry={expiry}. "
                    f"Available: {available}"
                )
            row = matching.iloc[0]
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
