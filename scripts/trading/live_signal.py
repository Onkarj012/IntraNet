#!/usr/bin/env python3
"""Live intraday NIFTY futures signal loop.

Runs during market hours (09:15–15:30 IST). Subscribes to Kite WebSocket
for NIFTY FUT + spot, builds features every minute, scores with the locked
LightGBM model, and emits a signal card when Variant A conditions are met.

Execution is MANUAL — this script only prints signals. No orders are placed.

Usage:
    # Start at 09:14, runs until 15:30
    .venv/bin/python scripts/live_futures.py

    # Replay a past day (for testing)
    .venv/bin/python scripts/live_futures.py --replay 2026-05-27

    # With Telegram alerts
    .venv/bin/python scripts/live_futures.py --telegram
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine.features import FUTURES_FEATURES, add_regime, compute_features
from broker.kite_client import KiteClient, _load_env

MODEL_PATH  = PROJECT_ROOT / "models/router_v0/futures/final_long.lgb"
LOG_DIR     = PROJECT_ROOT / "logs"
FUT_ROOT    = PROJECT_ROOT / "data/option_data/nifty_data/nifty_fut"
SPOT_ROOT   = PROJECT_ROOT / "data/option_data/nifty_data/nifty_spot"

# ── Variant A config (must match paper_trade_daily.py exactly) ────────────────
TARGET_PCT    = 0.0040
STOP_PCT      = 0.0030
HORIZON       = 60
LOT           = 50
COSTS_INR     = 105.0
HARD_CUTOFF   = dtime(14, 55)
SKIP_START    = dtime(11, 0)
SKIP_END      = dtime(12, 0)
ENTRY_MIN     = 30          # minutes from 09:15
SIGNAL_PCT    = 0.85
HIGH_CONF_PCT = 0.95
SKIP_REGIMES  = {"compression"}
MAX_TRADES    = 3
DAILY_HALT    = -15_000.0

OI_FILL = ["oi_chg_1m", "oi_chg_5m", "oi_chg_30m", "vol_oi_ratio", "vol_zscore"]


def _signal_card(ts: datetime, entry_px: float, score: float,
                  p85: float, p95: float, regime: str,
                  features: pd.Series) -> str:
    size_mult = 1.5 if score >= p95 else 1.0
    target_px = entry_px * (1 + TARGET_PCT)
    stop_px   = entry_px * (1 - STOP_PCT)
    codes = []
    if features.get("or_breakout_up", 0): codes.append("OR_BREAKOUT_UP")
    if features.get("ema_slope", 0) > 0.002: codes.append("EMA_TREND_UP")
    if features.get("vwap_dev", 0) > 0.001:  codes.append("ABOVE_VWAP")
    if features.get("oi_long_buildup", 0):    codes.append("OI_LONG_BUILDUP")
    if score >= p95: codes.append("HIGH_CONF")
    reasons = " | ".join(codes) if codes else "MODEL_SIGNAL"
    return (
        f"\n{'━'*62}\n"
        f"  🟢  LONG NIFTY FUT  {ts.strftime('%H:%M')}  {ts.date()}\n"
        f"  Entry  ₹{entry_px:>10,.2f}\n"
        f"  Target ₹{target_px:>10,.2f}  (+{TARGET_PCT*100:.1f}%)\n"
        f"  Stop   ₹{stop_px:>10,.2f}  (-{STOP_PCT*100:.1f}%)\n"
        f"  Size   {size_mult}x lot ({int(size_mult*LOT)} shares)\n"
        f"  Regime {regime}  |  Score {score:.4f} (p85={p85:.4f})\n"
        f"  {reasons}\n"
        f"{'━'*62}\n"
    )


def _send_telegram(msg: str, token: str, chat_id: str) -> None:
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": msg, "parse_mode": ""},
            timeout=5,
        )
    except Exception as e:
        print(f"  Telegram error: {e}")


# ── Replay mode (historical day) ──────────────────────────────────────────────

def _load_replay_day(target_date: date) -> pd.DataFrame:
    """Load one historical day from cached CSVs (same as paper_trade_daily)."""
    from index_options.v5_futures import discover_fut_days
    fut_path = (FUT_ROOT / str(target_date.year) / str(target_date.month) /
                f"nifty_fut_{target_date.day:02d}_{target_date.month:02d}_{target_date.year}.csv")
    spot_path = (SPOT_ROOT / str(target_date.year) / str(target_date.month) /
                 f"nifty_spot{target_date.day:02d}_{target_date.month:02d}_{target_date.year}.csv")

    if fut_path.exists() and spot_path.exists():
        from engine.features import _load_one_day
        return _load_one_day(fut_path, spot_path)

    # Fall back to NIFTY index proxy
    nifty_min = PROJECT_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"
    raw = pd.read_csv(nifty_min)
    raw["datetime"] = pd.to_datetime(raw["date"])
    day_start = pd.Timestamp(target_date)
    raw = raw[(raw["datetime"] >= day_start) &
              (raw["datetime"] < day_start + pd.Timedelta(days=1))].copy()
    raw = raw.rename(columns={"open": "f_open", "high": "f_high",
                               "low": "f_low", "close": "f_close", "volume": "f_vol"})
    raw["f_vol"] = 1.0
    raw["f_oi"] = 0.0
    raw["s_close"] = raw["f_close"]
    return raw.sort_values("datetime").reset_index(drop=True)


def run_replay(target_date: date, model: lgb.Booster,
               use_telegram: bool = False) -> None:
    """Simulate the live loop on a historical day (for testing)."""
    print(f"\n  REPLAY MODE: {target_date}")
    raw = _load_replay_day(target_date)
    if raw.empty:
        print(f"  no data for {target_date}")
        return

    feats = compute_features(raw, target_date)
    for col in OI_FILL:
        if col in feats.columns:
            feats[col] = feats[col].fillna(0.0)
    feats = feats.dropna(subset=FUTURES_FEATURES)
    feats = add_regime(feats)
    feats["datetime"] = pd.to_datetime(feats["datetime"])

    day_scores: list[float] = []
    n_trades = 0
    daily_pnl = 0.0

    for _, row in feats.iterrows():
        ts = row["datetime"]
        t = ts.time()
        mod = row["minute_of_day"] - 9 * 60 - 15

        if mod < ENTRY_MIN or t >= HARD_CUTOFF:
            continue
        if SKIP_START <= t < SKIP_END:
            continue
        if row["regime"] in SKIP_REGIMES:
            continue
        if n_trades >= MAX_TRADES or daily_pnl <= DAILY_HALT:
            continue

        feat_row = pd.DataFrame([{f: row.get(f, 0.0) for f in FUTURES_FEATURES}])
        score = float(model.predict(feat_row)[0])
        day_scores.append(score)

        if len(day_scores) < 10:
            continue

        p85 = float(np.percentile(day_scores, SIGNAL_PCT * 100))
        p95 = float(np.percentile(day_scores, HIGH_CONF_PCT * 100))

        if score >= p85:
            entry_px = float(row["f_close"])
            card = _signal_card(ts.to_pydatetime(), entry_px, score, p85, p95,
                                  str(row["regime"]), row)
            print(card)
            if use_telegram:
                tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                tg_chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
                if tg_token and tg_chat:
                    _send_telegram(card, tg_token, tg_chat)
            n_trades += 1
            # Simulate exit for daily PnL tracking
            target_px = entry_px * (1 + TARGET_PCT)
            stop_px   = entry_px * (1 - STOP_PCT)
            size_mult = 1.5 if score >= p95 else 1.0
            # Look ahead in feats for exit (replay only)
            future = feats[feats["datetime"] > ts].head(HORIZON)
            exit_px = float(future["f_close"].iloc[-1]) if not future.empty else entry_px
            for _, fr in future.iterrows():
                if fr["f_close"] >= target_px:
                    exit_px = target_px; break
                if fr["f_close"] <= stop_px:
                    exit_px = stop_px; break
            net = max((exit_px - entry_px) * LOT * size_mult - COSTS_INR * size_mult,
                      -3000.0)
            daily_pnl += net
            print(f"  → simulated exit: ₹{net:+,.0f}  (day total: ₹{daily_pnl:+,.0f})")

    if n_trades == 0:
        print(f"  no signals fired on {target_date}")


# ── Live mode (WebSocket) ─────────────────────────────────────────────────────

def run_live(kite: KiteClient, model: lgb.Booster,
             use_telegram: bool = False) -> None:
    """Live loop: subscribe WebSocket, build 1-min candles, score each minute."""
    from kiteconnect import KiteTicker

    fut_token, expiry = kite.get_fut_token("NIFTY")
    spot_token = kite.get_spot_token("NIFTY")
    vix_token  = kite.get_spot_token("INDIA VIX")
    today = date.today()

    print(f"\n  Live loop: {today}")
    print(f"  NIFTY FUT token={fut_token} expiry={expiry}")
    print(f"  Subscribing to tokens: {fut_token}, {spot_token}, {vix_token}")

    # Rolling 1-min candle buffers
    tick_buf: dict[int, list] = {fut_token: [], spot_token: []}
    candles: list[dict] = []   # completed 1-min candles (fut OHLCV + spot close)
    current_minute: datetime | None = None

    day_scores: list[float] = []
    n_trades = 0
    daily_pnl = 0.0

    log_path = LOG_DIR / f"live_{today.strftime('%Y%m%d')}.log"
    LOG_DIR.mkdir(exist_ok=True)

    def _log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    def on_ticks(ws, ticks):
        nonlocal current_minute
        now = datetime.now()
        minute = now.replace(second=0, microsecond=0)

        for tick in ticks:
            token = tick["instrument_token"]
            if token in tick_buf:
                tick_buf[token].append({
                    "ts": now,
                    "ltp": tick.get("last_price", 0),
                    "oi": tick.get("oi", 0),
                    "vol": tick.get("volume", 0),
                })

        # On minute boundary, close the candle
        if current_minute is not None and minute > current_minute:
            _close_candle(current_minute)
        current_minute = minute

    def _close_candle(minute: datetime) -> None:
        fut_ticks = tick_buf.get(fut_token, [])
        spot_ticks = tick_buf.get(spot_token, [])
        if not fut_ticks:
            return

        prices = [t["ltp"] for t in fut_ticks]
        candle = {
            "datetime": minute,
            "f_open": prices[0], "f_high": max(prices),
            "f_low": min(prices), "f_close": prices[-1],
            "f_oi": fut_ticks[-1]["oi"],
            "f_vol": fut_ticks[-1]["vol"],
            "s_close": spot_ticks[-1]["ltp"] if spot_ticks else prices[-1],
        }
        candles.append(candle)
        tick_buf[fut_token].clear()
        tick_buf[spot_token].clear()

        _process_candle(minute, candle)

    def _process_candle(ts: datetime, candle: dict) -> None:
        nonlocal n_trades, daily_pnl
        t = ts.time()
        if t < dtime(9, 15) or t > dtime(15, 30):
            return

        # Build features from accumulated candles
        if len(candles) < 5:
            return
        raw = pd.DataFrame(candles)
        raw["trade_date"] = pd.Timestamp(today)
        feats = compute_features(raw, today)
        if feats.empty:
            return
        for col in OI_FILL:
            if col in feats.columns:
                feats[col] = feats[col].fillna(0.0)
        feats = feats.dropna(subset=FUTURES_FEATURES)
        if feats.empty:
            return
        feats = add_regime(feats)
        row = feats.iloc[-1]

        mod = int(row["minute_of_day"]) - 9 * 60 - 15
        if mod < ENTRY_MIN or t >= HARD_CUTOFF:
            return
        if SKIP_START <= t < SKIP_END:
            return
        if str(row["regime"]) in SKIP_REGIMES:
            return
        if n_trades >= MAX_TRADES or daily_pnl <= DAILY_HALT:
            return

        feat_row = pd.DataFrame([{f: row.get(f, 0.0) for f in FUTURES_FEATURES}])
        score = float(model.predict(feat_row)[0])
        day_scores.append(score)

        if len(day_scores) < 10:
            return

        p85 = float(np.percentile(day_scores, SIGNAL_PCT * 100))
        p95 = float(np.percentile(day_scores, HIGH_CONF_PCT * 100))

        if score >= p85:
            entry_px = float(candle["f_close"])
            card = _signal_card(ts, entry_px, score, p85, p95,
                                  str(row["regime"]), row)
            _log(card)
            if use_telegram:
                tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                tg_chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
                if tg_token and tg_chat:
                    _send_telegram(card, tg_token, tg_chat)
            n_trades += 1

    def on_connect(ws, response):
        _log(f"WebSocket connected, subscribing {[fut_token, spot_token, vix_token]}")
        ws.subscribe([fut_token, spot_token, vix_token])
        ws.set_mode(ws.MODE_FULL, [fut_token])
        ws.set_mode(ws.MODE_LTP, [spot_token, vix_token])

    def on_close(ws, code, reason):
        _log(f"WebSocket closed: {code} {reason}")

    def on_error(ws, code, reason):
        _log(f"WebSocket error: {code} {reason}")

    _log(f"Starting live loop for {today}, model={MODEL_PATH.name}")
    kws = KiteTicker(kite.api_key, os.environ["KITE_ACCESS_TOKEN"])
    kws.on_ticks   = on_ticks
    kws.on_connect = on_connect
    kws.on_close   = on_close
    kws.on_error   = on_error
    kws.connect(threaded=True)

    # Keep alive until market close
    close_time = datetime.combine(today, dtime(15, 31))
    _log("Waiting for market open (09:15)...")
    while datetime.now() < close_time:
        time.sleep(1)

    kws.close()
    _log(f"Market closed. Trades today: {n_trades}  Day PnL: ₹{daily_pnl:+,.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=str, default=None,
                   help="Replay a past date YYYY-MM-DD instead of going live")
    p.add_argument("--telegram", action="store_true",
                   help="Send signal cards to Telegram (needs TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)")
    args = p.parse_args()

    _load_env()
    model = lgb.Booster(model_file=str(MODEL_PATH))
    print(f"  Model loaded: {MODEL_PATH.name}")

    if args.replay:
        target_date = datetime.strptime(args.replay, "%Y-%m-%d").date()
        run_replay(target_date, model, use_telegram=args.telegram)
        return 0

    # Live mode
    kite = KiteClient.from_env()
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if not token:
        print("  No KITE_ACCESS_TOKEN — run kite_login.py first")
        return 1
    kite.set_access_token(token)
    run_live(kite, model, use_telegram=args.telegram)
    return 0


if __name__ == "__main__":
    sys.exit(main())
