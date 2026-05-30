#!/usr/bin/env python3
"""Daily paper-trading runner for the OptiNet Router (Variant A).

Replays one trading day end-to-end:
  load NIFTY index minute bars  →  build features (proxy schema)
  →  score final_long.lgb       →  Variant-A filters + 85th pct
  →  simulate fills on same bars →  append to persistent ledger CSV.

Usage:
  scripts/paper_trade_daily.py --date 2026-05-15
  scripts/paper_trade_daily.py --auto              # most recent trading day
  scripts/paper_trade_daily.py --date 2026-05-15 --force  # overwrite

Idempotent: refuses to re-run a date that's already in the ledger
unless --force is passed.

Ledger schema (append-only CSV):
  paper_trade_id, run_timestamp, trade_date, datetime_entry, datetime_exit,
  side, entry_px, exit_px, target_px, stop_px, size_mult, lot,
  gross_pnl_inr, costs_inr, net_pnl_inr, exit_reason, regime, long_score,
  reason_codes, model_version, source
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engine.freshness import (
    check_file_freshness,
    freshness_failed,
    latest_weekday,
    print_freshness,
)
from engine.data_quality import (
    check_minute_session_quality,
    print_session_quality,
)
from engine.features import (
    FUTURES_FEATURES, add_regime, compute_features,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
NIFTY_MIN_PATH  = PROJECT_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"
MODEL_PATH      = PROJECT_ROOT / "models/router_v0/futures/final_long.lgb"
DEFAULT_LEDGER  = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"
KILL_SWITCH     = PROJECT_ROOT / "results/router_v0/PAPER_TRADING_HALTED"

# ── Backtest config (Variant A — must match forward_walk_2024_2026.py) ───────
TARGET_PCT  = 0.0040
STOP_PCT    = 0.0030
HORIZON     = 60
LOT         = 50
COSTS_INR   = 105.0
STOP_FLOOR  = -3000.0
DAILY_HALT  = -15000.0
# Phase-1: tighter intraday cumulative halt
INTRADAY_CUM_HALT = -9000.0
# Phase-1: skip when prior-day VIX >= threshold AND gap <= -0.5%
VIX_SPIKE_THRESHOLD = 20.0
VIX_GAP_THRESHOLD   = -0.005
MAX_TRADES  = 3
HARD_CUTOFF = dtime(14, 55)
SKIP_START  = dtime(11, 0)
SKIP_END    = dtime(12, 0)
ENTRY_MIN   = 30
SIGNAL_PCT  = 0.85
HIGH_CONF_PCT = 0.95
SKIP_REGIMES = {"compression"}

VIX_PATH = PROJECT_ROOT / "data/nifty_intraday/INDIA VIX_day.csv"

MODEL_VERSION = "futures_long_v1"
LEDGER_COLS = [
    "paper_trade_id", "run_timestamp", "trade_date",
    "datetime_entry", "datetime_exit",
    "side", "entry_px", "exit_px", "target_px", "stop_px",
    "size_mult", "lot", "gross_pnl_inr", "costs_inr", "net_pnl_inr",
    "exit_reason", "regime", "long_score", "reason_codes",
    "model_version", "source",
]


# ─────────────────────────────────────────────────────────────────────────────

def load_prior_vix(target_date: pd.Timestamp) -> float:
    """Return prior-day VIX close for target_date. Returns 0.0 if unavailable."""
    if not VIX_PATH.exists():
        return 0.0
    try:
        vix = pd.read_csv(VIX_PATH)
        vix.columns = [c.strip().lower() for c in vix.columns]
        date_col = next((c for c in vix.columns if "date" in c), None)
        close_col = next((c for c in vix.columns if c in ("close", "vix close", "vix_close")), None)
        if date_col is None or close_col is None:
            return 0.0
        vix[date_col] = pd.to_datetime(vix[date_col])
        vix = vix.sort_values(date_col).reset_index(drop=True)
        prior = vix[vix[date_col] < target_date]
        return float(prior[close_col].iloc[-1]) if not prior.empty else 0.0
    except Exception:
        return 0.0


def load_minute_bars(target_date: pd.Timestamp) -> pd.DataFrame:
    """Load and prepare NIFTY index minute bars for one trading day."""
    raw = pd.read_csv(NIFTY_MIN_PATH)
    raw["datetime"] = pd.to_datetime(raw["date"])
    day_start = pd.Timestamp(target_date.date())
    day_end   = day_start + pd.Timedelta(days=1)
    raw = raw[(raw["datetime"] >= day_start) &
                (raw["datetime"] < day_end)].copy()
    if raw.empty:
        return raw
    raw = raw.rename(columns={
        "open": "f_open", "high": "f_high", "low": "f_low",
        "close": "f_close", "volume": "f_vol",
    })
    raw["f_vol"] = 1.0
    raw["f_oi"] = 0.0
    raw["s_close"] = raw["f_close"]
    raw["trade_date"] = day_start
    raw = raw.sort_values("datetime").reset_index(drop=True)
    return raw


def reason_codes(row: pd.Series) -> str:
    codes = []
    if row.get("or_breakout_up", 0): codes.append("OR_BREAKOUT_UP")
    if row.get("ema_slope", 0) > 0.002: codes.append("EMA_TREND_UP")
    if row.get("vwap_dev", 0) > 0.001:  codes.append("ABOVE_VWAP")
    if row.get("regime") == "expansion": codes.append("EXPANSION_REGIME")
    if row.get("regime") == "trend_up":  codes.append("TREND_UP_REGIME")
    if row.get("realized_vol_30m", 0) < 0.10: codes.append("LOW_VOL")
    return "|".join(codes) if codes else "MODEL_SIGNAL"


def simulate_long_trade(entry_idx: int, day_bars: pd.DataFrame,
                         entry_px: float, size_mult: float) -> dict:
    target_px = entry_px * (1 + TARGET_PCT)
    stop_px   = entry_px * (1 - STOP_PCT)
    j_end = min(len(day_bars), entry_idx + 1 + HORIZON)
    walk = day_bars.iloc[entry_idx + 1:j_end]
    exit_px = None
    exit_reason = None
    exit_dt = None
    for _, r in walk.iterrows():
        c = r["fut_close"]
        if c >= target_px:
            exit_px, exit_reason, exit_dt = target_px, "TARGET", r["datetime"]
            break
        if c <= stop_px:
            exit_px, exit_reason, exit_dt = stop_px, "STOP", r["datetime"]
            break
    if exit_px is None:
        if walk.empty:
            exit_px = entry_px
            exit_reason = "NO_BARS"
            exit_dt = day_bars.iloc[entry_idx]["datetime"]
        else:
            exit_px = float(walk["fut_close"].iloc[-1])
            exit_reason = "TIME"
            exit_dt = walk.iloc[-1]["datetime"]
    gross = (exit_px - entry_px) * LOT * size_mult
    costs = COSTS_INR * size_mult
    net = gross - costs
    if net < STOP_FLOOR:
        net = STOP_FLOOR
        exit_reason = "STOP_FLOOR"
    return {
        "entry_px": entry_px, "exit_px": exit_px,
        "target_px": target_px, "stop_px": stop_px,
        "datetime_exit": exit_dt,
        "gross_pnl_inr": gross, "costs_inr": costs, "net_pnl_inr": net,
        "exit_reason": exit_reason, "size_mult": size_mult,
    }


def run_one_day(target_date: pd.Timestamp,
                  model: lgb.Booster) -> pd.DataFrame:
    """Run Variant A on one date, return trade ledger rows."""
    raw = load_minute_bars(target_date)
    if raw.empty:
        print(f"  {target_date.date()}: no bars in source CSV")
        return pd.DataFrame(columns=LEDGER_COLS)
    if len(raw) < 60:
        print(f"  {target_date.date()}: only {len(raw)} bars, need ≥ 60")
        return pd.DataFrame(columns=LEDGER_COLS)

    feats = compute_features(raw, target_date.date())
    if feats.empty:
        return pd.DataFrame(columns=LEDGER_COLS)

    OI_FILL = ["oi_chg_1m", "oi_chg_5m", "oi_chg_30m",
                "vol_oi_ratio", "vol_zscore"]
    for col in OI_FILL:
        if col in feats.columns:
            feats[col] = feats[col].fillna(0.0)
    feats = feats.dropna(subset=FUTURES_FEATURES)
    if feats.empty:
        return pd.DataFrame(columns=LEDGER_COLS)
    feats["datetime"]   = pd.to_datetime(feats["datetime"])
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    feats = add_regime(feats)

    # Variant A hard filters
    t = feats["datetime"].dt.time
    mod = feats["minute_of_day"] - 9 * 60 - 15
    eligible = feats[
        (mod >= ENTRY_MIN) &
        (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~feats["regime"].isin(SKIP_REGIMES)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=LEDGER_COLS)

    eligible["long_score"] = model.predict(eligible[FUTURES_FEATURES])
    p85 = eligible["long_score"].quantile(SIGNAL_PCT)
    p95 = eligible["long_score"].quantile(HIGH_CONF_PCT)
    eligible["take_long"] = eligible["long_score"] >= p85
    eligible["size_mult"] = np.where(eligible["long_score"] >= p95, 1.5, 1.0)

    candidates = eligible[eligible["take_long"]].sort_values("datetime").reset_index(drop=True)

    # Simulation cache
    bars_for_sim = feats[["datetime", "f_close"]].rename(
        columns={"f_close": "fut_close"}).sort_values(
        "datetime").reset_index(drop=True)

    rows = []
    daily_pnl = 0.0
    intraday_cum = 0.0   # Phase-1: cumulative PnL this session
    n_taken = 0
    run_ts = datetime.now().isoformat(timespec="seconds")

    # Phase-1: VIX spike filter — load prior-day VIX
    prior_vix = load_prior_vix(target_date)
    # Compute gap from first eligible bar
    gap_pct = 0.0
    if not candidates.empty:
        gap_pct = float(candidates.iloc[0].get("gap_pct", 0.0))
    vix_spike_day = (prior_vix >= VIX_SPIKE_THRESHOLD and gap_pct <= VIX_GAP_THRESHOLD)
    if vix_spike_day:
        print(f"  Phase-1 VIX spike filter: prior_vix={prior_vix:.1f} gap={gap_pct*100:.2f}% — skipping day")
        return pd.DataFrame(columns=LEDGER_COLS)

    for _, c in candidates.iterrows():
        if n_taken >= MAX_TRADES:
            break
        if daily_pnl <= DAILY_HALT:
            break
        # Phase-1: intraday cumulative halt
        if intraday_cum <= INTRADAY_CUM_HALT:
            print(f"  Phase-1 intraday halt: cum_pnl=₹{intraday_cum:+,.0f}")
            break
        idx_arr = bars_for_sim.index[bars_for_sim["datetime"] == c["datetime"]]
        if len(idx_arr) == 0:
            continue
        i = int(idx_arr[0])
        entry_px = float(bars_for_sim["fut_close"].iat[i])
        sim = simulate_long_trade(i, bars_for_sim, entry_px, float(c["size_mult"]))
        rows.append({
            "paper_trade_id": str(uuid.uuid4())[:12],
            "run_timestamp": run_ts,
            "trade_date": pd.Timestamp(c["trade_date"]).strftime("%Y-%m-%d"),
            "datetime_entry": pd.Timestamp(c["datetime"]).isoformat(),
            "datetime_exit": pd.Timestamp(sim["datetime_exit"]).isoformat()
                if sim["datetime_exit"] is not None else None,
            "side": "LONG",
            "entry_px": sim["entry_px"],
            "exit_px": sim["exit_px"],
            "target_px": sim["target_px"],
            "stop_px": sim["stop_px"],
            "size_mult": sim["size_mult"],
            "lot": LOT,
            "gross_pnl_inr": sim["gross_pnl_inr"],
            "costs_inr": sim["costs_inr"],
            "net_pnl_inr": sim["net_pnl_inr"],
            "exit_reason": sim["exit_reason"],
            "regime": str(c["regime"]),
            "long_score": float(c["long_score"]),
            "reason_codes": reason_codes(c),
            "model_version": MODEL_VERSION,
            "source": "paper",
        })
        daily_pnl += sim["net_pnl_inr"]
        intraday_cum += sim["net_pnl_inr"]
        n_taken += 1

    return pd.DataFrame(rows, columns=LEDGER_COLS)


def append_ledger(rows: pd.DataFrame, ledger_path: Path) -> None:
    if rows.empty:
        return
    write_header = not ledger_path.exists()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ledger_path, mode="a", header=write_header, index=False)


def date_already_logged(target_date: pd.Timestamp,
                          ledger_path: Path) -> bool:
    if not ledger_path.exists():
        return False
    try:
        existing = pd.read_csv(ledger_path, usecols=["trade_date", "source"])
    except Exception:
        return False
    target = target_date.strftime("%Y-%m-%d")
    return ((existing["trade_date"] == target) &
            (existing["source"] == "paper")).any()


def remove_date_from_ledger(target_date: pd.Timestamp,
                              ledger_path: Path) -> int:
    if not ledger_path.exists():
        return 0
    df = pd.read_csv(ledger_path)
    target = target_date.strftime("%Y-%m-%d")
    mask = (df["trade_date"] == target) & (df["source"] == "paper")
    n_removed = int(mask.sum())
    df[~mask].to_csv(ledger_path, index=False)
    return n_removed


def get_most_recent_data_date() -> pd.Timestamp:
    raw = pd.read_csv(NIFTY_MIN_PATH, usecols=["date"])
    raw["datetime"] = pd.to_datetime(raw["date"])
    return raw["datetime"].max().normalize()


def preflight_freshness(target_date: pd.Timestamp, auto_mode: bool) -> list:
    required = latest_weekday() if auto_mode else pd.Timestamp(target_date).normalize()
    return [
        check_file_freshness(
            "NIFTY minute bars",
            NIFTY_MIN_PATH,
            required,
        )
    ]


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                          help="Trade date YYYY-MM-DD")
    parser.add_argument("--auto", action="store_true",
                          help="Use most recent date in source data")
    parser.add_argument("--force", action="store_true",
                          help="Re-run even if date is already in ledger")
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER))
    parser.add_argument("--ignore-kill-switch", action="store_true")
    parser.add_argument("--allow-stale-data", action="store_true",
                          help="Bypass freshness failures for historical/debug replays")
    parser.add_argument("--allow-incomplete-session", action="store_true",
                          help="Bypass minute-session completeness failures")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)

    # Kill switch
    if KILL_SWITCH.exists() and not args.ignore_kill_switch:
        print(f"  ⚠️  PAPER TRADING IS HALTED ({KILL_SWITCH})")
        print(f"     remove that file or pass --ignore-kill-switch to proceed")
        return 2

    # Resolve date
    if args.date:
        target_date = pd.Timestamp(args.date).normalize()
    elif args.auto:
        target_date = get_most_recent_data_date()
        print(f"  --auto resolved to {target_date.date()}")
    else:
        print("  must pass --date YYYY-MM-DD or --auto")
        return 1

    checks = preflight_freshness(target_date, args.auto)
    print_freshness(checks)
    if freshness_failed(checks) and not args.allow_stale_data:
        print("\n  no run: stale data")
        print("  update source data or pass --allow-stale-data for an intentional historical replay")
        return 5

    quality = check_minute_session_quality(NIFTY_MIN_PATH, target_date)
    print_session_quality(quality)
    if not quality.ok and not args.allow_incomplete_session:
        print("\n  no run: incomplete session data")
        print("  repair source bars or pass --allow-incomplete-session for an intentional replay")
        return 6

    # Idempotency
    if date_already_logged(target_date, ledger_path):
        if not args.force:
            print(f"  {target_date.date()}: already in ledger (use --force to re-run)")
            return 0
        n_removed = remove_date_from_ledger(target_date, ledger_path)
        print(f"  --force: removed {n_removed} prior rows for {target_date.date()}")

    # Run
    print(f"\n  paper-trading {target_date.date()} (Variant A)")
    print(f"  ledger: {ledger_path}")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    rows = run_one_day(target_date, model)
    append_ledger(rows, ledger_path)

    # Summary
    if rows.empty:
        print("\n  no trades taken (filters / no signals)")
    else:
        net = rows["net_pnl_inr"].sum()
        wins = (rows["net_pnl_inr"] > 0).sum()
        print(f"\n  trades:  {len(rows)}")
        print(f"  wins:    {wins} / {len(rows)} ({100*wins/len(rows):.0f} %)")
        print(f"  net:     ₹{net:+,.0f}")
        for _, r in rows.iterrows():
            arrow = "▲" if r["net_pnl_inr"] > 0 else "▼"
            print(f"   {arrow}  {r['datetime_entry'][11:16]}  "
                  f"{r['side']}  entry={r['entry_px']:>9,.2f}  "
                  f"exit={r['exit_px']:>9,.2f}  ({r['exit_reason']:>6s})  "
                  f"₹{r['net_pnl_inr']:>+8,.0f}  "
                  f"x{r['size_mult']:.1f}")

    print(f"\n  ledger now has {sum(1 for _ in open(ledger_path)) - 1 if ledger_path.exists() else 0} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
