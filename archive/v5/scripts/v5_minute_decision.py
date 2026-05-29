#!/usr/bin/env python
"""V5 v1 per-minute decision script.

Implements the decision pipeline (spec §4.1) with:
- Hard 14:55 IST cutoff (amendment #2)
- Halt-flag checks (any halt → NO_TRADE_HALT)
- Broker-API-only data path (yfinance is forbidden here)
- Vol kill-switch → gate → caps → daily-loss halt → place trade
- MTM check on all open positions every minute, with intra-trade −₹3,000 stop

Run modes:
- Cron mode (default): runs once for "now", appends to ledger
- Backtest/sim mode (--simulate-ts YYYY-MM-DDTHH:MM): pretends "now" is the
  given timestamp; uses MockBroker. Required for end-to-end testing.

Returns one of these enum codes (also written to logs):
  PLACED, NO_TRADE_CUTOFF, NO_TRADE_HALT, NO_TRADE_DTE,
  NO_TRADE_DATA, NO_TRADE_VOL, NO_TRADE_GATE,
  NO_TRADE_CAPS, NO_TRADE_LOSS_HALT, NO_TRADE_PREMARKET,
  MTM_STOP, MTM_OK
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from optinet.v4_chain import bs_price
from optinet.v5_runtime.broker import (BrokerClient, MockBroker, make_broker,
                                          OptionChainSnapshot)
from optinet.v5_runtime import ledger as ld
from optinet.v5_runtime.online_features import (
    OnlineFeatureBuilder, GATE_FEATURES, VOL_FEATURES,
)
from optinet.v5_runtime.runtime_config import (
    GATE_DIR, VOL_MODEL_PATH, GATE_THRESHOLD, VOL_KILL_THRESHOLD,
    PER_TRADE_STOP_LOSS_INR, MAX_TRADES_PER_INDEX_PER_DAY,
    MAX_TRADES_PER_DAY_TOTAL, DAILY_LOSS_HALT_INR, LOT_SIZE, STRIKE_STEP,
    HARD_CUTOFF_TIME, EARLIEST_TRADE_TIME, MARKET_OPEN_TIME,
    ensure_dirs, any_halt_active,
    next_weekly_expiry, days_to_expiry, dte_bucket_of,
    compute_round_trip_cost,
)


log = logging.getLogger("v5_decision")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Cached models (so cron-per-minute doesn't reload)
# ---------------------------------------------------------------------------

_MODELS: dict[str, lgb.Booster] = {}


def _load_models() -> dict[str, lgb.Booster]:
    if _MODELS:
        return _MODELS
    _MODELS["gate_dte2"] = lgb.Booster(model_file=str(GATE_DIR / "gate_dte2.lgb"))
    _MODELS["gate_dte3"] = lgb.Booster(model_file=str(GATE_DIR / "gate_dte3.lgb"))
    _MODELS["vol"] = lgb.Booster(model_file=str(VOL_MODEL_PATH))
    return _MODELS


# ---------------------------------------------------------------------------
# Per-minute decision pipeline
# ---------------------------------------------------------------------------

def _tag(action: str, **kw) -> str:
    extras = " ".join(f"{k}={v}" for k, v in kw.items())
    return f"{action} {extras}".rstrip()


def decide_at_minute(
    *, now: datetime, broker: BrokerClient,
    builder_cache: dict[tuple[str, date], OnlineFeatureBuilder],
    enabled_symbols: tuple[str, ...] = ("NIFTY",),
    write_ledger: bool = True,
) -> dict:
    """Run the decision pipeline for a single minute.

    Returns a structured dict (also useful for tests). Side effect:
    appends to the paper ledger if `write_ledger`.
    """
    ensure_dirs()
    today = now.date()
    decisions: list[dict] = []

    # 1. Hard cutoff (amendment #2)
    if now.time() >= HARD_CUTOFF_TIME:
        log.info(_tag("NO_TRADE_CUTOFF", t=now.time()))
        return {"action": "NO_TRADE_CUTOFF", "now": now}

    # 1a. Earliest-trade guard (added 2026-05-27 after paper-trading evidence
    # showed every entry clustered at 09:45/09:46 due to the realized_vol_30m
    # warm-up; opening-session entries are too volatile for SS_EOD).
    if now.time() < EARLIEST_TRADE_TIME:
        return {"action": "NO_TRADE_EARLY", "now": now,
                "earliest": EARLIEST_TRADE_TIME}

    # 2. Halt flags
    halt = any_halt_active()
    if halt:
        log.info(_tag("NO_TRADE_HALT", reason=halt))
        return {"action": "NO_TRADE_HALT", "reason": halt}

    # 3. Daily caps & loss halt (read once)
    n_total_today = ld.trades_today_total(today)
    per_index_today = ld.trades_today_per_index(today)
    cum_pnl_today = ld.cumulative_day_pnl_inr(today)
    if cum_pnl_today <= DAILY_LOSS_HALT_INR:
        log.info(_tag("NO_TRADE_LOSS_HALT", pnl=cum_pnl_today))
        return {"action": "NO_TRADE_LOSS_HALT", "cum_pnl": cum_pnl_today}
    if n_total_today >= MAX_TRADES_PER_DAY_TOTAL:
        log.info(_tag("NO_TRADE_CAPS", reason="daily_total"))
        return {"action": "NO_TRADE_CAPS", "reason": "daily_total"}

    models = _load_models()

    # 4. Per-symbol pipeline
    for symbol in enabled_symbols:
        n_idx = int(per_index_today.get(symbol, 0))
        if n_idx >= MAX_TRADES_PER_INDEX_PER_DAY:
            decisions.append({"symbol": symbol, "action": "NO_TRADE_CAPS",
                              "reason": "per_index"})
            continue

        # Resolve expiry (fallback to next weekly Thursday)
        expiry = next_weekly_expiry(symbol, today)
        dte_days = days_to_expiry(today, expiry)
        bucket = dte_bucket_of(dte_days)
        if bucket not in (2, 3):
            decisions.append({"symbol": symbol, "action": "NO_TRADE_DTE",
                              "dte": dte_days})
            continue

        key = (symbol, expiry)
        if key not in builder_cache:
            builder_cache[key] = OnlineFeatureBuilder(broker, symbol, expiry)

        builder = builder_cache[key]
        # Step (warm-up if first call this session)
        if builder._last_ts is None:
            warm_start = datetime.combine(today, MARKET_OPEN_TIME)
            warmed = builder.warmup_to(now, start=warm_start)
            log.info(f"warmup {symbol}: {warmed} minutes loaded")
            if warmed == 0:
                decisions.append({"symbol": symbol, "action": "NO_TRADE_DATA",
                                  "reason": "warmup_empty"})
                continue
        else:
            ok = builder.step(now)
            if not ok:
                decisions.append({"symbol": symbol, "action": "NO_TRADE_DATA",
                                  "reason": "step_failed"})
                continue

        snap = builder.compute_at(now)
        if snap is None:
            decisions.append({"symbol": symbol, "action": "NO_TRADE_DATA",
                              "reason": "feature_compute_none"})
            continue
        if not (0.08 <= snap.atm_iv <= 1.0):
            decisions.append({"symbol": symbol, "action": "NO_TRADE_DATA",
                              "reason": f"atm_iv_out_of_range:{snap.atm_iv:.4f}"})
            continue
        if snap.atm_straddle_premium_per_share < 50:
            decisions.append({"symbol": symbol, "action": "NO_TRADE_DATA",
                              "reason": f"premium_low:{snap.atm_straddle_premium_per_share:.1f}"})
            continue

        # 5. Vol kill-switch
        vf = snap.vol_features.copy()
        if vf.isna().any(axis=1).iloc[0]:
            decisions.append({"symbol": symbol, "action": "NO_TRADE_DATA",
                              "reason": "vol_features_nan"})
            continue
        pred_rv = float(models["vol"].predict(vf[VOL_FEATURES])[0])
        if pred_rv > VOL_KILL_THRESHOLD:
            decisions.append({"symbol": symbol, "action": "NO_TRADE_VOL",
                              "pred_rv": pred_rv})
            continue

        # 6. Gate
        gf = snap.gate_features.copy()
        if gf.isna().any(axis=1).iloc[0]:
            decisions.append({"symbol": symbol, "action": "NO_TRADE_DATA",
                              "reason": "gate_features_nan"})
            continue
        gate_model = models[f"gate_dte{bucket}"]
        gate_score = float(gate_model.predict(gf[GATE_FEATURES])[0])
        if gate_score < GATE_THRESHOLD:
            decisions.append({"symbol": symbol, "action": "NO_TRADE_GATE",
                              "gate_score": gate_score, "pred_rv": pred_rv})
            continue

        # 7. PLACE PAPER TRADE — short ATM straddle
        atm_call_px = float(snap.raw_chain_minute["atm_call_premium"])
        atm_put_px = float(snap.raw_chain_minute["atm_put_premium"])

        if write_ledger:
            tid = ld.append_open_trade(
                entry_ts=snap.timestamp,
                symbol=symbol, expiry=expiry,
                dte_days=snap.dte_days, dte_bucket=bucket,
                atm_strike=snap.atm_strike, lot_size=LOT_SIZE[symbol],
                entry_call_px=atm_call_px, entry_put_px=atm_put_px,
                spot_at_entry=snap.spot, atm_iv_at_entry=snap.atm_iv,
                gate_score=gate_score, pred_rv=pred_rv,
                gate_threshold=GATE_THRESHOLD,
                vol_kill_threshold=VOL_KILL_THRESHOLD,
                feature_snapshot=snap.gate_features.iloc[0].to_dict(),
            )
        else:
            tid = "DRY-RUN"

        decisions.append({
            "symbol": symbol, "action": "PLACED",
            "trade_id": tid, "atm_strike": snap.atm_strike,
            "atm_call_px": atm_call_px, "atm_put_px": atm_put_px,
            "premium_inr": (atm_call_px + atm_put_px) * LOT_SIZE[symbol],
            "gate_score": gate_score, "pred_rv": pred_rv,
            "spot": snap.spot, "atm_iv": snap.atm_iv,
        })
        log.info(_tag("PLACED", symbol=symbol, tid=tid,
                       strike=snap.atm_strike, gate=round(gate_score, 3),
                       prem_inr=int((atm_call_px + atm_put_px) * LOT_SIZE[symbol])))

        # Update local counters so a multi-symbol pass respects caps
        per_index_today[symbol] = n_idx + 1
        n_total_today += 1
        if n_total_today >= MAX_TRADES_PER_DAY_TOTAL:
            break

    # 8. MTM check on currently-open positions (independent of cap state)
    mtm_actions = mtm_check_and_stop(now=now, broker=broker,
                                       builder_cache=builder_cache,
                                       write_ledger=write_ledger)

    return {"action": "MULTI", "decisions": decisions, "mtm": mtm_actions,
            "now": now, "n_total_today": n_total_today,
            "cum_pnl_today": cum_pnl_today}


def mtm_check_and_stop(*, now: datetime, broker: BrokerClient,
                         builder_cache: dict[tuple[str, date], OnlineFeatureBuilder],
                         write_ledger: bool) -> list[dict]:
    """For each OPEN trade in the ledger, refresh MTM and stop-out if needed.

    Cost approximation: at exit we approximate slippage on the exit-side too.
    """
    open_df = ld.open_trades()
    if open_df.empty:
        return []

    actions = []
    for _, row in open_df.iterrows():
        symbol = row["symbol"]
        expiry = pd.to_datetime(row["expiry"]).date()
        strike = int(row["atm_strike"])
        lot = int(row["lot_size"])
        entry_call = float(row["entry_call_px"])
        entry_put = float(row["entry_put_px"])

        # Get a current chain snapshot
        try:
            chain = broker.get_option_chain(symbol, expiry, now)
        except Exception as exc:  # noqa: BLE001
            actions.append({"trade_id": row["trade_id"],
                            "action": "MTM_DATA_FAIL", "error": repr(exc)})
            continue

        cur_call = _chain_price(chain, strike, "CE")
        cur_put = _chain_price(chain, strike, "PE")
        if cur_call is None or cur_put is None:
            # Fall back to BS reprice using snap atm_iv
            iv = float(row["atm_iv_at_entry"])
            from optinet.v4_chain import _expiry_T
            T = _expiry_T(pd.Timestamp(now), expiry)
            cur_call = bs_price(chain.spot, strike, T, iv, "CE")
            cur_put = bs_price(chain.spot, strike, T, iv, "PE")

        # Live PnL (short straddle): premium received − premium to buy back
        live_pnl_inr = ((entry_call + entry_put) - (cur_call + cur_put)) * lot
        # Cost approximation (round trip)
        costs = compute_round_trip_cost(
            entry_premium_per_share_total=entry_call + entry_put,
            exit_premium_per_share_total=cur_call + cur_put,
            lot_size=lot, n_legs=2, is_atm=True,
        )
        net_live = live_pnl_inr - costs

        if net_live <= PER_TRADE_STOP_LOSS_INR:
            # Stop out
            if write_ledger:
                ld.close_trade(
                    row["trade_id"], exit_ts=now,
                    exit_call_px=cur_call, exit_put_px=cur_put,
                    costs_inr=costs, live_pnl_inr=PER_TRADE_STOP_LOSS_INR,
                    was_stopped=True, exit_reason="STOP",
                )
            actions.append({"trade_id": row["trade_id"], "action": "MTM_STOP",
                            "live_pnl": net_live, "stopped_at": net_live})
            log.info(_tag("MTM_STOP", tid=row["trade_id"],
                           pnl=int(net_live)))
        else:
            actions.append({"trade_id": row["trade_id"], "action": "MTM_OK",
                            "live_pnl": net_live})
    return actions


def _chain_price(chain: OptionChainSnapshot, strike: int, opt_type: str
                  ) -> Optional[float]:
    for c in chain.contracts:
        if c.strike == strike and c.opt_type == opt_type:
            return c.close
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate-ts", default=None,
                    help="Pretend 'now' is this ts (YYYY-MM-DDTHH:MM); uses MockBroker")
    ap.add_argument("--symbols", default="NIFTY",
                    help="Comma-separated enabled symbols")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write to the ledger")
    args = ap.parse_args()

    if args.simulate_ts:
        now = datetime.fromisoformat(args.simulate_ts)
        broker = MockBroker(simulate_date=now.date())
    else:
        now = datetime.now()
        broker = make_broker()

    builder_cache: dict[tuple[str, date], OnlineFeatureBuilder] = {}
    result = decide_at_minute(
        now=now, broker=broker, builder_cache=builder_cache,
        enabled_symbols=tuple(s.strip() for s in args.symbols.split(",")),
        write_ledger=not args.dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
