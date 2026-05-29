#!/usr/bin/env python3
"""Bootstrap the paper-trading ledger from Phase-3 forward-walk trades.

Converts the 1,128-trade forward-walk ledger (Nov 2024 → May 2026) into the
canonical paper_trading_ledger.csv schema, marked source='forward_walk' so
the status dashboard can include or exclude them with --include-bootstrap.

Idempotent: refuses to insert if ledger already has forward_walk rows.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
SOURCE_PARQ   = PROJECT_ROOT / "results/router_v0/phase3_fwd_no_guard.parquet"
LEDGER_PATH   = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"

LOT = 50
COSTS_INR = 105.0
TARGET_PCT = 0.0040
STOP_PCT = 0.0030

LEDGER_COLS = [
    "paper_trade_id", "run_timestamp", "trade_date",
    "datetime_entry", "datetime_exit",
    "side", "entry_px", "exit_px", "target_px", "stop_px",
    "size_mult", "lot", "gross_pnl_inr", "costs_inr", "net_pnl_inr",
    "exit_reason", "regime", "long_score", "reason_codes",
    "model_version", "source",
]


def main() -> int:
    if not SOURCE_PARQ.exists():
        print(f"ERROR: {SOURCE_PARQ} not found")
        return 1

    if LEDGER_PATH.exists():
        existing = pd.read_csv(LEDGER_PATH)
        if (existing["source"] == "forward_walk").any():
            print(f"  ledger already has forward_walk rows; skipping bootstrap")
            print(f"  (delete the rows and re-run if you really need to)")
            return 0

    src = pd.read_parquet(SOURCE_PARQ)
    print(f"  loaded {len(src)} forward-walk trades")

    src["trade_date"] = pd.to_datetime(src["trade_date"]).dt.normalize()
    src["datetime"]   = pd.to_datetime(src["datetime"])
    run_ts = datetime.now().isoformat(timespec="seconds")

    rows = pd.DataFrame({
        "paper_trade_id": [str(uuid.uuid4())[:12] for _ in range(len(src))],
        "run_timestamp": run_ts,
        "trade_date":     src["trade_date"].dt.strftime("%Y-%m-%d"),
        "datetime_entry": src["datetime"].apply(lambda x: x.isoformat()),
        "datetime_exit":  None,                       # not recorded in source
        "side":           "LONG",
        "entry_px":       src["entry_px"],
        "exit_px":        src["exit_px"],
        "target_px":      src["entry_px"] * (1 + TARGET_PCT),
        "stop_px":        src["entry_px"] * (1 - STOP_PCT),
        "size_mult":      src["size_mult"],
        "lot":            LOT,
        "gross_pnl_inr":  src["net_pnl_inr"] + COSTS_INR * src["size_mult"],
        "costs_inr":      COSTS_INR * src["size_mult"],
        "net_pnl_inr":    src["net_pnl_inr"],
        "exit_reason":    src["exit_reason"],
        "regime":         src["regime"],
        "long_score":     src["long_score"],
        "reason_codes":   "BOOTSTRAP_FROM_FORWARD_WALK",
        "model_version":  "futures_long_v1",
        "source":         "forward_walk",
    })[LEDGER_COLS]

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LEDGER_PATH.exists()
    rows.to_csv(LEDGER_PATH, mode="a", header=write_header, index=False)

    total_pnl = rows["net_pnl_inr"].sum()
    n_days    = rows["trade_date"].nunique()
    print(f"  wrote {len(rows)} rows ({n_days} unique days, ₹{total_pnl:+,.0f} total PnL)")
    print(f"  → {LEDGER_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
