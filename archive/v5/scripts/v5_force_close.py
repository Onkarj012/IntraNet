#!/usr/bin/env python
"""V5 v1 force-close — flatten any OPEN paper positions at the current price.

Cron call at 15:25 IST. Also used by manual override / rollback.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from optinet.v4_chain import bs_price, _expiry_T
from optinet.v5_runtime.broker import make_broker, MockBroker
from optinet.v5_runtime import ledger as ld
from optinet.v5_runtime.runtime_config import compute_round_trip_cost

log = logging.getLogger("v5_force_close")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")


def force_close_all(*, now: datetime, broker, reason: str = "EOD",
                     write_ledger: bool = True) -> list[dict]:
    open_df = ld.open_trades()
    if open_df.empty:
        log.info("no open trades to flatten")
        return []

    out = []
    for _, row in open_df.iterrows():
        symbol = row["symbol"]
        expiry = pd.to_datetime(row["expiry"]).date()
        strike = int(row["atm_strike"])
        lot = int(row["lot_size"])
        entry_call = float(row["entry_call_px"])
        entry_put = float(row["entry_put_px"])

        try:
            chain = broker.get_option_chain(symbol, expiry, now)
            spot = chain.spot
            cur_call = next((c.close for c in chain.contracts
                              if c.strike == strike and c.opt_type == "CE"), None)
            cur_put = next((c.close for c in chain.contracts
                             if c.strike == strike and c.opt_type == "PE"), None)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"chain fetch failed: {exc}; falling back to BS reprice")
            cur_call = cur_put = None
            spot = float(row["spot_at_entry"])

        if cur_call is None or cur_put is None:
            iv = float(row["atm_iv_at_entry"])
            T = _expiry_T(pd.Timestamp(now), expiry)
            cur_call = bs_price(spot, strike, T, iv, "CE")
            cur_put = bs_price(spot, strike, T, iv, "PE")

        live_pnl = ((entry_call + entry_put) - (cur_call + cur_put)) * lot
        costs = compute_round_trip_cost(
            entry_premium_per_share_total=entry_call + entry_put,
            exit_premium_per_share_total=cur_call + cur_put,
            lot_size=lot, n_legs=2, is_atm=True,
        )
        net = live_pnl - costs
        if write_ledger:
            ld.close_trade(
                row["trade_id"], exit_ts=now,
                exit_call_px=cur_call, exit_put_px=cur_put,
                costs_inr=costs, live_pnl_inr=net,
                was_stopped=False, exit_reason=reason,
            )
        out.append({"trade_id": row["trade_id"], "live_pnl": net,
                    "exit_call": cur_call, "exit_put": cur_put})
        log.info(f"{row['trade_id']} closed: net=₹{net:.0f} reason={reason}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="EOD",
                    help="Exit reason tag (EOD / MANUAL / ROLLBACK)")
    ap.add_argument("--simulate-ts", default=None,
                    help="Pretend now=ts; uses MockBroker")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.simulate_ts:
        now = datetime.fromisoformat(args.simulate_ts)
        broker = MockBroker(simulate_date=now.date())
    else:
        now = datetime.now()
        broker = make_broker()

    out = force_close_all(now=now, broker=broker, reason=args.reason,
                            write_ledger=not args.dry_run)
    print(f"force_close: {len(out)} trades flattened")
    for o in out:
        print(f"  {o['trade_id']}: net=₹{o['live_pnl']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
