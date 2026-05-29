"""Shared helpers for V5 v1 runtime: expiry calendar, costs, flag I/O, paths."""
from __future__ import annotations

import os
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
FLAGS_DIR = REPO_ROOT / "flags"
LOGS_DIR = REPO_ROOT / "logs"
MODELS_DIR = REPO_ROOT / "models"

GATE_DIR = MODELS_DIR / "optinet_v5"
VOL_MODEL_PATH = MODELS_DIR / "optinet_v4" / "rv_30m_forward.lgb"

GATE_THRESHOLD = 0.70
PER_TRADE_STOP_LOSS_INR = -3000.0
MAX_TRADES_PER_INDEX_PER_DAY = 2
MAX_TRADES_PER_DAY_TOTAL = 4
DAILY_LOSS_HALT_INR = -15000.0
VOL_KILL_THRESHOLD = 0.2352   # frozen 90th-pct of training pred_rv distribution

LOT_SIZE = {"NIFTY": 50, "BANKNIFTY": 15}
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}

# Hard cutoff (amendment #2): no new trades at or after 14:55 IST
HARD_CUTOFF_TIME = dtime(14, 55)
EOD_FORCE_CLOSE_TIME = dtime(15, 25)
MARKET_OPEN_TIME = dtime(9, 15)
MARKET_CLOSE_TIME = dtime(15, 30)


def _parse_time(s: str, default: dtime) -> dtime:
    try:
        h, m = s.split(":")
        return dtime(int(h), int(m))
    except Exception:
        return default


# Earliest allowed trade entry — opening-session guard added 2026-05-27 after
# paper-trading evidence showed all 32 trades clustering at 09:45/46 due to the
# realized_vol_30m warm-up window, which made every entry a volatile-open entry.
# Default 10:30 IST. Override with V5_EARLIEST_TRADE env var (HH:MM).
EARLIEST_TRADE_TIME = _parse_time(
    os.environ.get("V5_EARLIEST_TRADE", "10:30"), dtime(10, 30)
)


def ensure_dirs() -> None:
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Halt flags
# ---------------------------------------------------------------------------

def halt_today_active() -> bool:
    return (FLAGS_DIR / "halt_today.flag").exists()


def halt_indefinite_active() -> bool:
    return (FLAGS_DIR / "halt_indefinite.flag").exists()


def paused_active() -> Optional[Path]:
    """Return the paused flag path if any exists."""
    matches = list(FLAGS_DIR.glob("paused_*.flag"))
    return matches[0] if matches else None


def any_halt_active() -> Optional[str]:
    """Return a string explaining the halt, or None if clear."""
    if halt_indefinite_active():
        return "halt_indefinite.flag present"
    if halt_today_active():
        return "halt_today.flag present"
    p = paused_active()
    if p is not None:
        return f"paused flag present: {p.name}"
    return None


# ---------------------------------------------------------------------------
# Expiry calendar
# ---------------------------------------------------------------------------
# NIFTY weekly options expire on Tuesday (post-Apr 2024) or Thursday (pre-Apr 2024).
# For paper-trading v1 we accept the actual contract expiry from the broker; the
# helpers below give a best-guess fallback for offline testing.

def next_weekly_expiry(symbol: str, today: date) -> date:
    """Best-effort next weekly expiry. NSE NIFTY weekly is Thursday (legacy)
    and Tuesday (post-2024-04). Adjust by the broker-reported expiry when live.
    """
    # Default: Thursday
    target_weekday = 3  # Mon=0..Sun=6
    delta = (target_weekday - today.weekday()) % 7
    if delta == 0:
        # Today is Thursday — expiry is today; next weekly is +7
        delta = 7 if today.weekday() == target_weekday else delta
    return today + timedelta(days=delta or 7)


def days_to_expiry(today: date, expiry: date) -> int:
    return (expiry - today).days


def dte_bucket_of(dte_days: int) -> int:
    if dte_days <= 0:
        return 0
    if dte_days <= 2:
        return 1
    if dte_days <= 4:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Cost model (mirrors v5_simulator)
# ---------------------------------------------------------------------------

BROKERAGE_PER_LEG_INR = 40.0
SLIPPAGE_ATM = 0.015
SLIPPAGE_OTM = 0.025
STT_RATE_OPT_SELL = 0.000625
EXCHANGE_TXN = 0.000019
SEBI_FEE = 0.000001
STAMP_RATE = 0.00003
GST_RATE = 0.18


def compute_round_trip_cost(
    *, entry_premium_per_share_total: float,
    exit_premium_per_share_total: float,
    lot_size: int, n_legs: int, is_atm: bool,
) -> float:
    """Round-trip cost approximation for a 2-leg straddle at lot_size shares.

    `entry_premium_per_share_total` and `exit_premium_per_share_total` are the
    sum of |leg prices| (CE + PE per share) at entry and exit respectively.
    """
    slip = SLIPPAGE_ATM if is_atm else SLIPPAGE_OTM
    slip_inr = (entry_premium_per_share_total + exit_premium_per_share_total) * slip * lot_size
    brokerage = BROKERAGE_PER_LEG_INR * n_legs * 2  # entry + exit
    sell_premium_inr = entry_premium_per_share_total * lot_size  # we sell at entry
    stt = sell_premium_inr * STT_RATE_OPT_SELL
    exchange = (entry_premium_per_share_total + exit_premium_per_share_total) * lot_size * EXCHANGE_TXN
    sebi = (entry_premium_per_share_total + exit_premium_per_share_total) * lot_size * SEBI_FEE
    stamp = sell_premium_inr * STAMP_RATE
    gst = (brokerage + exchange + sebi) * GST_RATE
    return slip_inr + brokerage + stt + exchange + sebi + stamp + gst
