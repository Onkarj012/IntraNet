"""Paper-trade ledger for V5 v1.

A single parquet file at `ledger/v5_paper_ledger.parquet` records every
paper trade we open, with full context for reconciliation and drift analysis.

Schema (one row per trade):
- trade_id            str   YYYYMMDD-NNN
- entry_ts            ts    minute of entry
- exit_ts             ts    minute of exit (NaT if open)
- symbol              str   NIFTY / BANKNIFTY
- expiry              date  contract expiry
- dte_days            int   days to expiry at entry
- dte_bucket          int   gate bucket (2 or 3 for v1)
- strategy            str   SHORT_STRADDLE_EOD
- atm_strike          int
- lot_size            int   50 for NIFTY, 15 for BANKNIFTY
- n_lots              int   always 1 for v1
- entry_call_px       float per-share
- entry_put_px        float per-share
- entry_premium_inr   float total premium received (CE + PE) × lot
- exit_call_px        float (NaN if open)
- exit_put_px         float
- exit_premium_inr    float total premium paid back × lot
- costs_inr           float total costs (slippage + brokerage + STT + GST + ...)
- live_pnl_inr        float MTM-based PnL at exit (live tracking)
- realized_pnl_inr    float PnL from EOD bhavcopy close prices (set on reconcile)
- reconciliation_gap  float live − realized (set on reconcile)
- was_stopped         bool  True if intra-trade −₹3,000 cap was hit
- exit_reason         str   STOP / EOD / MANUAL / FORCE_CLOSE / NEXT_DAY_CARRY
- status              str   OPEN / CLOSED / RECONCILED
- spot_at_entry       float
- atm_iv_at_entry     float
- gate_score          float
- pred_rv             float
- gate_threshold      float 0.70
- vol_kill_threshold  float 0.235
- feature_snapshot    str   JSON of GATE_FEATURES (for drift analysis)
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_DIR = REPO_ROOT / "ledger"
LEDGER_PATH = LEDGER_DIR / "v5_paper_ledger.parquet"

LEDGER_COLUMNS: list[str] = [
    "trade_id", "entry_ts", "exit_ts",
    "symbol", "expiry", "dte_days", "dte_bucket",
    "strategy", "atm_strike", "lot_size", "n_lots",
    "entry_call_px", "entry_put_px", "entry_premium_inr",
    "exit_call_px", "exit_put_px", "exit_premium_inr",
    "costs_inr", "live_pnl_inr", "realized_pnl_inr", "reconciliation_gap",
    "was_stopped", "exit_reason", "status",
    "spot_at_entry", "atm_iv_at_entry",
    "gate_score", "pred_rv",
    "gate_threshold", "vol_kill_threshold",
    "feature_snapshot",
]

LEDGER_DTYPES = {
    "trade_id": "string",
    "symbol": "string", "strategy": "string",
    "exit_reason": "string", "status": "string",
    "feature_snapshot": "string",
    "atm_strike": "Int64", "lot_size": "Int64", "n_lots": "Int64",
    "dte_days": "Int64", "dte_bucket": "Int64",
    "was_stopped": "boolean",
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    (LEDGER_DIR / "archive").mkdir(parents=True, exist_ok=True)


def load_ledger() -> pd.DataFrame:
    """Load the ledger; return an empty frame if it does not exist yet."""
    _ensure_dir()
    if not LEDGER_PATH.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.read_parquet(LEDGER_PATH)
    # Ensure all columns present (forward-compatible)
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[LEDGER_COLUMNS]


def save_ledger(df: pd.DataFrame) -> None:
    _ensure_dir()
    out = df.reindex(columns=LEDGER_COLUMNS)
    out.to_parquet(LEDGER_PATH, index=False)


def next_trade_id(df: pd.DataFrame, d: date) -> str:
    prefix = d.strftime("%Y%m%d")
    today_n = int((df["trade_id"].astype("string").str.startswith(prefix)).sum())
    return f"{prefix}-{today_n + 1:03d}"


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------

def append_open_trade(
    *, entry_ts: datetime, symbol: str, expiry: date, dte_days: int,
    dte_bucket: int, atm_strike: int, lot_size: int,
    entry_call_px: float, entry_put_px: float,
    spot_at_entry: float, atm_iv_at_entry: float,
    gate_score: float, pred_rv: float,
    gate_threshold: float, vol_kill_threshold: float,
    feature_snapshot: dict, costs_inr_at_entry: float = 0.0,
) -> str:
    """Append one new OPEN paper trade. Returns trade_id."""
    df = load_ledger()
    tid = next_trade_id(df, entry_ts.date())
    entry_premium_inr = (entry_call_px + entry_put_px) * lot_size
    row = {
        "trade_id": tid,
        "entry_ts": pd.Timestamp(entry_ts),
        "exit_ts": pd.NaT,
        "symbol": symbol, "expiry": expiry,
        "dte_days": int(dte_days), "dte_bucket": int(dte_bucket),
        "strategy": "SHORT_STRADDLE_EOD",
        "atm_strike": int(atm_strike), "lot_size": int(lot_size), "n_lots": 1,
        "entry_call_px": float(entry_call_px),
        "entry_put_px": float(entry_put_px),
        "entry_premium_inr": float(entry_premium_inr),
        "exit_call_px": pd.NA, "exit_put_px": pd.NA,
        "exit_premium_inr": pd.NA,
        "costs_inr": float(costs_inr_at_entry),
        "live_pnl_inr": pd.NA,
        "realized_pnl_inr": pd.NA,
        "reconciliation_gap": pd.NA,
        "was_stopped": False, "exit_reason": pd.NA, "status": "OPEN",
        "spot_at_entry": float(spot_at_entry),
        "atm_iv_at_entry": float(atm_iv_at_entry),
        "gate_score": float(gate_score),
        "pred_rv": float(pred_rv),
        "gate_threshold": float(gate_threshold),
        "vol_kill_threshold": float(vol_kill_threshold),
        "feature_snapshot": json.dumps({k: float(v) if v is not None else None
                                          for k, v in feature_snapshot.items()},
                                          default=str),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_ledger(df)
    return tid


def close_trade(
    trade_id: str, *, exit_ts: datetime,
    exit_call_px: float, exit_put_px: float,
    costs_inr: float, live_pnl_inr: float,
    was_stopped: bool, exit_reason: str,
) -> None:
    df = load_ledger()
    mask = df["trade_id"] == trade_id
    if not mask.any():
        raise KeyError(f"trade_id not found: {trade_id}")
    lot = int(df.loc[mask, "lot_size"].iloc[0])
    exit_premium_inr = (exit_call_px + exit_put_px) * lot
    df.loc[mask, "exit_ts"] = pd.Timestamp(exit_ts)
    df.loc[mask, "exit_call_px"] = float(exit_call_px)
    df.loc[mask, "exit_put_px"] = float(exit_put_px)
    df.loc[mask, "exit_premium_inr"] = float(exit_premium_inr)
    df.loc[mask, "costs_inr"] = float(costs_inr)
    df.loc[mask, "live_pnl_inr"] = float(live_pnl_inr)
    df.loc[mask, "was_stopped"] = bool(was_stopped)
    df.loc[mask, "exit_reason"] = str(exit_reason)
    df.loc[mask, "status"] = "CLOSED"
    save_ledger(df)


def attach_realized(trade_id: str, realized_pnl_inr: float) -> None:
    df = load_ledger()
    mask = df["trade_id"] == trade_id
    if not mask.any():
        raise KeyError(f"trade_id not found: {trade_id}")
    live = df.loc[mask, "live_pnl_inr"].iloc[0]
    gap = (None if pd.isna(live) else float(realized_pnl_inr) - float(live))
    df.loc[mask, "realized_pnl_inr"] = float(realized_pnl_inr)
    df.loc[mask, "reconciliation_gap"] = gap
    df.loc[mask, "status"] = "RECONCILED"
    save_ledger(df)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def open_trades() -> pd.DataFrame:
    df = load_ledger()
    return df[df["status"] == "OPEN"].copy()


def trades_on_date(d: date) -> pd.DataFrame:
    df = load_ledger()
    if df.empty:
        return df
    return df[pd.to_datetime(df["entry_ts"]).dt.date == d].copy()


def trades_today_total(d: date) -> int:
    return int(len(trades_on_date(d)))


def trades_today_per_index(d: date) -> dict[str, int]:
    df = trades_on_date(d)
    if df.empty:
        return {}
    return df.groupby("symbol").size().to_dict()


def cumulative_day_pnl_inr(d: date) -> float:
    """Sum of live_pnl_inr for closed trades + 0 for currently-open (no MTM yet)."""
    df = trades_on_date(d)
    if df.empty:
        return 0.0
    closed = df[df["status"].isin(["CLOSED", "RECONCILED"])]
    if closed.empty:
        return 0.0
    return float(closed["live_pnl_inr"].fillna(0.0).sum())


def consecutive_stops(n_recent: int = 3) -> int:
    """Count consecutive recent stops. Used by halt logic."""
    df = load_ledger()
    if df.empty:
        return 0
    closed = df[df["status"].isin(["CLOSED", "RECONCILED"])].copy()
    if closed.empty:
        return 0
    closed = closed.sort_values("exit_ts").tail(n_recent)
    if len(closed) < n_recent:
        return 0
    return int(closed["was_stopped"].astype(bool).sum())
