#!/usr/bin/env python
"""V5 v1 EOD reconciliation — runs at 15:45 IST.

For every CLOSED paper trade today, recompute realized PnL using actual
close prices (or 15:25-bar close prices) from the broker / data lake, and
record the gap between live MTM-based PnL and the reconciled value.

Reconciliation gap > ₹50 is flagged.

Also computes:
- daily summary
- rolling 5d / 20d metrics
- alerts vs backtest expectations (2.68 trades/day, 70.7% win, +₹378 mean)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from optinet.v5_runtime.broker import make_broker, MockBroker
from optinet.v5_runtime import ledger as ld
from optinet.v5_runtime.runtime_config import (
    compute_round_trip_cost, FLAGS_DIR, ensure_dirs,
)

log = logging.getLogger("v5_reconcile")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")

EXPECTED_TRADES_PER_DAY = 2.68
EXPECTED_WIN_RATE = 0.707
EXPECTED_MEAN_PNL_INR = 378.0
RECONCILIATION_GAP_FLAG_INR = 50.0
ALERT_DEVIATION_STD = 2.0


def reconcile_day(*, today: date, broker, write: bool = True) -> dict:
    df = ld.trades_on_date(today)
    if df.empty:
        log.info(f"no trades on {today}")
        return {"date": today, "n_trades": 0}

    closed = df[df["status"].isin(["CLOSED", "RECONCILED"])].copy()
    if closed.empty:
        log.warning(f"{today}: no CLOSED trades; skipping reconcile")
        return {"date": today, "n_trades": int(len(df)), "n_closed": 0,
                "warning": "no closed trades"}

    flags = []
    for _, row in closed.iterrows():
        if row["status"] == "RECONCILED":
            continue  # already reconciled
        symbol = row["symbol"]
        expiry = pd.to_datetime(row["expiry"]).date()
        strike = int(row["atm_strike"])
        lot = int(row["lot_size"])
        entry_call = float(row["entry_call_px"])
        entry_put = float(row["entry_put_px"])

        # Pull the 15:25 close prices from the broker / cache
        try:
            close_ts = datetime.combine(today, datetime.strptime("15:25", "%H:%M").time())
            chain = broker.get_option_chain(symbol, expiry, close_ts)
            close_call = next((c.close for c in chain.contracts
                                if c.strike == strike and c.opt_type == "CE"), None)
            close_put = next((c.close for c in chain.contracts
                               if c.strike == strike and c.opt_type == "PE"), None)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"{row['trade_id']}: chain fetch failed: {exc}")
            close_call = close_put = None

        if close_call is None or close_put is None:
            log.warning(f"{row['trade_id']}: missing reconciliation price; "
                          f"using exit_call/put as realized")
            close_call = float(row["exit_call_px"])
            close_put = float(row["exit_put_px"])

        # If trade was stopped, realized PnL = stop value (we already exited)
        if row["was_stopped"]:
            realized = float(row["live_pnl_inr"])
        else:
            gross = ((entry_call + entry_put) - (close_call + close_put)) * lot
            costs = compute_round_trip_cost(
                entry_premium_per_share_total=entry_call + entry_put,
                exit_premium_per_share_total=close_call + close_put,
                lot_size=lot, n_legs=2, is_atm=True,
            )
            realized = gross - costs

        if write:
            ld.attach_realized(row["trade_id"], realized)
        gap = abs(realized - float(row["live_pnl_inr"]))
        if gap > RECONCILIATION_GAP_FLAG_INR:
            flags.append({"trade_id": row["trade_id"], "gap_inr": gap,
                            "live": float(row["live_pnl_inr"]),
                            "realized": realized})
            log.warning(f"{row['trade_id']}: reconciliation gap ₹{gap:.0f} "
                          f"(live=₹{row['live_pnl_inr']:.0f}, realized=₹{realized:.0f})")

    # Final summary
    df = ld.load_ledger()
    today_df = df[pd.to_datetime(df["entry_ts"]).dt.date == today]
    realized_arr = today_df["realized_pnl_inr"].astype(float).to_numpy()
    realized_arr = realized_arr[~np.isnan(realized_arr)]
    win = (realized_arr > 0).mean() if len(realized_arr) else float("nan")

    summary = {
        "date": str(today),
        "n_trades": int(len(today_df)),
        "n_reconciled": int((today_df["status"] == "RECONCILED").sum()),
        "n_stopped": int(today_df["was_stopped"].astype(bool).sum()),
        "win_rate": float(win) if not np.isnan(win) else None,
        "mean_pnl_inr": float(np.nanmean(realized_arr)) if len(realized_arr) else None,
        "total_pnl_inr": float(np.nansum(realized_arr)) if len(realized_arr) else None,
        "reconciliation_gaps": flags,
    }
    log.info(f"reconcile {today}: {summary}")
    return summary


def rolling_metrics(*, today: date, ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {}
    ledger = ledger.copy()
    ledger["entry_date"] = pd.to_datetime(ledger["entry_ts"]).dt.date
    daily = (ledger
              .groupby("entry_date")["realized_pnl_inr"]
              .agg(total="sum", n="count", win=lambda s: (s > 0).mean()))
    if daily.empty:
        return {}

    def window(n_days: int) -> dict:
        start = today - timedelta(days=n_days)
        sub = daily[daily.index >= start]
        if sub.empty:
            return {}
        d = sub["total"].dropna()
        sharpe = float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 1 and d.std() > 0 else None
        return {
            "n_days": int(len(sub)),
            "trades": int(sub["n"].sum()),
            "total_pnl": float(sub["total"].sum()),
            "win_rate": float(sub["win"].mean()),
            "sharpe": sharpe,
        }

    return {"5d": window(5), "20d": window(20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=None,
                    help="Override today (YYYY-MM-DD)")
    ap.add_argument("--simulate-broker", action="store_true",
                    help="Use MockBroker (offline reconcile of historical day)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = (date.fromisoformat(args.today) if args.today else date.today())
    broker = (MockBroker(simulate_date=today)
              if args.simulate_broker else make_broker())

    summary = reconcile_day(today=today, broker=broker,
                              write=not args.dry_run)

    df = ld.load_ledger()
    rolls = rolling_metrics(today=today, ledger=df)

    out = {"summary": summary, "rolling": rolls}
    print(json.dumps(out, indent=2, default=str))

    # Drift alert vs backtest expectations
    r20 = rolls.get("20d", {})
    if r20 and r20.get("n_days", 0) >= 15:
        wr = r20.get("win_rate", 0) or 0
        mean_pnl = (r20.get("total_pnl", 0) / max(r20.get("trades", 1), 1))
        if wr < 0.55:
            log.warning(f"20d win rate {wr:.1%} < 55% threshold — manual review required")
            (FLAGS_DIR / f"paused_{today}.flag").touch()
        if mean_pnl < 0:
            log.warning(f"20d mean PnL ₹{mean_pnl:.0f} < 0 — manual review required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
