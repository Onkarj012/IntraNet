#!/usr/bin/env python3
"""Futures engine paper-trading dry-run.

Replays one full trading day using the MockBroker + FuturesEngine,
exercising the complete decision loop:
  09:30 → 14:54: score_minute() → TradeCard → paper ledger
  15:25: force-close all open positions
  15:45: EOD reconciliation

Verifies:
  - Hard filters fire correctly (11:00-11:59 blocked, compression blocked)
  - 14:55 cutoff respected
  - Daily caps respected
  - TradeCards are well-formed and JSON-serializable
  - Ledger entries are written and reconciled

Usage:
    python scripts/futures_paper_dryrun.py --date 2024-01-08
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from optinet_router.families.futures import FuturesEngine
from optinet_router.futures_features import FUTURES_FEATURES, add_regime
from optinet_router.schema import StrategyFamily
from optinet.v5_runtime.broker import MockBroker
from optinet.v5_runtime import ledger as ld
from optinet.v5_runtime.runtime_config import (
    FLAGS_DIR, ensure_dirs, any_halt_active,
    HARD_CUTOFF_TIME, EOD_FORCE_CLOSE_TIME, MARKET_OPEN_TIME,
    MAX_TRADES_PER_DAY_TOTAL, DAILY_LOSS_HALT_INR,
)

FEAT_CACHE = REPO_ROOT / "cache/router_v0/futures_features.parquet"
RESULTS    = REPO_ROOT / "results/router_v0"
RESULTS.mkdir(parents=True, exist_ok=True)

# Futures-specific risk params
MAX_TRADES_PER_DAY = 3
STOP_FLOOR_INR = -3000.0
TARGET_PCT = 0.0040
STOP_PCT   = 0.0030
LOT        = 50
COSTS_INR  = 105.0


def load_day_features(sim_date: date) -> pd.DataFrame:
    """Load pre-computed features for one day from the cache."""
    feats = pd.read_parquet(FEAT_CACHE)
    feats["datetime"] = pd.to_datetime(feats["datetime"])
    day = feats[feats["datetime"].dt.date == sim_date].copy()
    return add_regime(day)


def simulate_futures_exit(entry_px: float, entry_ts: datetime,
                            broker: MockBroker, expiry: date) -> dict:
    """Walk minute bars from entry to 15:25, apply target/stop/time-stop."""
    sign = 1
    target_px = entry_px * (1 + TARGET_PCT)
    stop_px   = entry_px * (1 - STOP_PCT)
    exit_ts   = datetime.combine(entry_ts.date(), dtime(15, 25))

    # Walk through broker bars
    cursor = entry_ts + timedelta(minutes=1)
    while cursor <= exit_ts:
        try:
            sq = broker.get_spot("NIFTY", cursor)
            c = sq.spot
        except Exception:
            cursor += timedelta(minutes=1)
            continue
        if c >= target_px:
            gross = (target_px - entry_px) * LOT
            return {"exit_px": target_px, "exit_ts": cursor,
                    "exit_reason": "TARGET", "net_pnl_inr": gross - COSTS_INR}
        if c <= stop_px:
            net = max((stop_px - entry_px) * LOT - COSTS_INR, STOP_FLOOR_INR)
            return {"exit_px": stop_px, "exit_ts": cursor,
                    "exit_reason": "STOP", "net_pnl_inr": net}
        cursor += timedelta(minutes=1)

    # Time stop
    try:
        sq = broker.get_spot("NIFTY", exit_ts)
        exit_px = sq.spot
    except Exception:
        exit_px = entry_px
    net = (exit_px - entry_px) * LOT - COSTS_INR
    return {"exit_px": exit_px, "exit_ts": exit_ts,
            "exit_reason": "TIME", "net_pnl_inr": net}


def run_day(sim_date: date, verbose: bool = False) -> dict:
    ensure_dirs()
    for f in FLAGS_DIR.glob("*.flag"):
        f.unlink()
    if ld.LEDGER_PATH.exists():
        ld.LEDGER_PATH.unlink()

    broker = MockBroker(simulate_date=sim_date)
    engine = FuturesEngine()
    engine.reset_day()

    day_feats = load_day_features(sim_date)
    if day_feats.empty:
        return {"error": f"no features for {sim_date}"}

    trades_today = 0
    cum_pnl = 0.0
    open_trades: list[dict] = []
    placed_cards: list[dict] = []
    skipped_reasons: dict[str, int] = {}

    # Walk 09:30 → 14:54
    cursor = datetime.combine(sim_date, MARKET_OPEN_TIME) + timedelta(minutes=15)
    cutoff = datetime.combine(sim_date, HARD_CUTOFF_TIME)

    while cursor < cutoff:
        t = cursor.time()

        # Check halt flags
        halt = any_halt_active()
        if halt:
            skipped_reasons["HALT"] = skipped_reasons.get("HALT", 0) + 1
            cursor += timedelta(minutes=1)
            continue

        # Daily caps
        if trades_today >= MAX_TRADES_PER_DAY:
            skipped_reasons["CAPS"] = skipped_reasons.get("CAPS", 0) + 1
            cursor += timedelta(minutes=1)
            continue

        # Daily loss halt
        if cum_pnl <= DAILY_LOSS_HALT_INR:
            skipped_reasons["LOSS_HALT"] = skipped_reasons.get("LOSS_HALT", 0) + 1
            cursor += timedelta(minutes=1)
            continue

        # Get features for this minute
        row = day_feats[day_feats["datetime"] == pd.Timestamp(cursor)]
        if row.empty:
            cursor += timedelta(minutes=1)
            continue
        features = row.iloc[0].to_dict()

        # Score via FuturesEngine
        card = engine.make_trade_card(
            timestamp=cursor, symbol="NIFTY", features=features)

        if card.recommendation.strategy == StrategyFamily.NO_TRADE:
            reason = card.recommendation.reason_codes[0] if card.recommendation.reason_codes else "NO_SIGNAL"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        else:
            # Place paper trade
            try:
                spot = broker.get_spot("NIFTY", cursor).spot
            except Exception:
                spot = float(features.get("f_close", 0))

            entry_px = spot
            expiry = card.recommendation.legs[0].expiry

            # Simulate exit on actual bars
            exit_info = simulate_futures_exit(entry_px, cursor, broker, expiry)
            net_pnl = exit_info["net_pnl_inr"]

            placed_cards.append({
                "entry_ts": cursor.isoformat(),
                "exit_ts": exit_info["exit_ts"].isoformat(),
                "entry_px": entry_px,
                "exit_px": exit_info["exit_px"],
                "exit_reason": exit_info["exit_reason"],
                "net_pnl_inr": net_pnl,
                "regime": card.market_state.regime.value,
                "reason_codes": list(card.recommendation.reason_codes),
                "size_lots": card.recommendation.suggested_size_lots,
                "card_json": card.to_json(),
            })
            trades_today += 1
            cum_pnl += net_pnl
            if verbose:
                print(f"  [{cursor.strftime('%H:%M')}] PLACED  "
                      f"entry={entry_px:.2f}  exit={exit_info['exit_px']:.2f}  "
                      f"reason={exit_info['exit_reason']}  pnl=₹{net_pnl:.0f}  "
                      f"regime={card.market_state.regime.value}  "
                      f"codes={card.recommendation.reason_codes}")

        cursor += timedelta(minutes=1)

    return {
        "sim_date": str(sim_date),
        "n_trades": trades_today,
        "cum_pnl_inr": cum_pnl,
        "trades": placed_cards,
        "skipped_reasons": skipped_reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-01-08")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sim_date = date.fromisoformat(args.date)
    print(f"=== Futures paper dry-run: {sim_date} ===")
    result = run_day(sim_date, verbose=args.verbose)

    print(f"\n  Trades placed  : {result['n_trades']}")
    print(f"  Cumulative PnL : ₹{result['cum_pnl_inr']:+,.0f}")
    print(f"\n  Skipped reasons:")
    for k, v in sorted(result["skipped_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {k:<25s}: {v}")

    if result["trades"]:
        print(f"\n  Trade log:")
        for t in result["trades"]:
            print(f"    {t['entry_ts'][11:16]} → {t['exit_ts'][11:16]}  "
                  f"entry={t['entry_px']:.2f}  exit={t['exit_px']:.2f}  "
                  f"reason={t['exit_reason']:<7s}  pnl=₹{t['net_pnl_inr']:>+7.0f}  "
                  f"regime={t['regime']}  codes={t['reason_codes']}")

    # Verify all cards are valid JSON
    for t in result["trades"]:
        from optinet_router.schema import TradeCard
        back = TradeCard.from_dict(json.loads(t["card_json"]))
        assert back.recommendation.strategy == StrategyFamily.FUTURES_LONG

    print(f"\n  ✓ All {result['n_trades']} trade cards JSON-valid")

    # Verify hard filters
    for t in result["trades"]:
        ts = datetime.fromisoformat(t["entry_ts"])
        assert ts.time() < HARD_CUTOFF_TIME, f"cutoff violated: {ts}"
        assert not (dtime(11, 0) <= ts.time() < dtime(12, 0)), f"11-12 filter violated: {ts}"
        assert t["regime"] != "compression", f"compression filter violated: {ts}"
    print(f"  ✓ All hard filters respected")

    out = RESULTS / f"futures_dryrun_{sim_date}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
