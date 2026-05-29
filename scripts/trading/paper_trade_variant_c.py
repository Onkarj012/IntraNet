#!/usr/bin/env python3
"""Variant C daily paper-trading runner — runs in parallel with Variant A.

Variant A: scripts/paper_trade_daily.py    (source='paper')
Variant C: scripts/paper_trade_variant_c.py (source='paper_c')

Both write to the same ledger CSV with the `source` column distinguishing
them.  Idempotent and kill-switch aware (same kill-switch file as Variant A).

Variant C config (from results/router_v0/variant_c_config.json):
  ret_5d_cut=-0.015      skip if NIFTY 5-day return < -1.5%
  ret_20d_cut=None       no 20-day filter
  vix_state_block=True   skip if VIX rising AND > 75th-pct of 60d
  intraday_halt=-6000    tighter daily PnL halt
  signal_pct=0.95        top 5% scores only

Usage:
  scripts/paper_trade_variant_c.py --auto
  scripts/paper_trade_variant_c.py --date 2026-05-15
  scripts/paper_trade_variant_c.py --date 2026-05-15 --force
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, time as dtime
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
NIFTY_MIN_PATH   = PROJECT_ROOT / "data/nifty_intraday/NIFTY 50_minute.csv"
NIFTY_DAILY_PATH = PROJECT_ROOT / "data/indices/nifty_daily.csv"
VIX_DAY_PATH     = PROJECT_ROOT / "data/nifty_intraday/INDIA VIX_day.csv"
MODEL_PATH       = PROJECT_ROOT / "models/router_v0/futures/final_long.lgb"
DEFAULT_LEDGER   = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"
KILL_SWITCH      = PROJECT_ROOT / "results/router_v0/PAPER_TRADING_HALTED"
VARIANT_C_CONFIG = PROJECT_ROOT / "results/router_v0/variant_c_config.json"

# ── Common (matches Variant A) ───────────────────────────────────────────────
TARGET_PCT  = 0.0040
STOP_PCT    = 0.0030
HORIZON     = 60
LOT         = 50
COSTS_INR   = 105.0
STOP_FLOOR  = -3000.0
MAX_TRADES  = 3
HARD_CUTOFF = dtime(14, 55)
SKIP_START  = dtime(11, 0)
SKIP_END    = dtime(12, 0)
ENTRY_MIN   = 30
HIGH_CONF_PCT = 0.95
SKIP_REGIMES = {"compression"}

MODEL_VERSION = "futures_long_v1"
VARIANT_TAG   = "C"
LEDGER_SOURCE = "paper_c"

LEDGER_COLS = [
    "paper_trade_id", "run_timestamp", "trade_date",
    "datetime_entry", "datetime_exit",
    "side", "entry_px", "exit_px", "target_px", "stop_px",
    "size_mult", "lot", "gross_pnl_inr", "costs_inr", "net_pnl_inr",
    "exit_reason", "regime", "long_score", "reason_codes",
    "model_version", "source",
]


def load_variant_c_config() -> dict:
    with open(VARIANT_C_CONFIG) as f:
        full = json.load(f)
    return full["recommended"]


def load_macro_signals() -> pd.DataFrame:
    nd = pd.read_csv(NIFTY_DAILY_PATH)
    nd["date"] = pd.to_datetime(nd["date"], utc=True).dt.tz_convert(None).dt.normalize()
    nd = nd.sort_values("date").reset_index(drop=True)
    nd["ret_5d"]  = nd["close"].pct_change(5)
    nd["ret_20d"] = nd["close"].pct_change(20)
    vix = pd.read_csv(VIX_DAY_PATH)
    vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
    vix = vix.sort_values("date").reset_index(drop=True)
    vix["vix_5d_ma"]    = vix["close"].rolling(5,  min_periods=2).mean()
    vix["vix_60d_q75"]  = vix["close"].rolling(60, min_periods=20).quantile(0.75)
    vix["vix_state_block"] = (
        (vix["close"] > vix["vix_5d_ma"]) &
        (vix["close"] > vix["vix_60d_q75"])
    ).astype(int)
    df = pd.merge(
        nd[["date", "ret_5d", "ret_20d"]],
        vix[["date", "close", "vix_state_block"]].rename(columns={"close": "vix_close"}),
        on="date", how="outer",
    ).sort_values("date").ffill()
    for c in ["ret_5d", "ret_20d", "vix_close", "vix_state_block"]:
        df[c] = df[c].shift(1)
    return df.rename(columns={
        "ret_5d": "ret_5d_prev", "ret_20d": "ret_20d_prev",
        "vix_close": "vix_prev_close", "vix_state_block": "vix_state_block_prev",
    })


def load_minute_bars(target_date: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_csv(NIFTY_MIN_PATH)
    raw["datetime"] = pd.to_datetime(raw["date"])
    day_start = pd.Timestamp(target_date.date())
    raw = raw[(raw["datetime"] >= day_start) &
                (raw["datetime"] < day_start + pd.Timedelta(days=1))].copy()
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
    return raw.sort_values("datetime").reset_index(drop=True)


def reason_codes(row, cfg) -> str:
    codes = ["VARIANT_C", f"PCT{int(cfg['signal_pct']*100)}"]
    if cfg.get("vix_state_block"):       codes.append("VIX_STATE_GUARD")
    if cfg.get("ret_5d_cut") is not None: codes.append("RET5D_GUARD")
    if cfg.get("ret_20d_cut") is not None: codes.append("RET20D_GUARD")
    if row.get("regime") == "expansion": codes.append("EXPANSION")
    return "|".join(codes)


def simulate_long_trade(entry_idx, day_bars, entry_px, size_mult):
    target_px = entry_px * (1 + TARGET_PCT)
    stop_px   = entry_px * (1 - STOP_PCT)
    j_end = min(len(day_bars), entry_idx + 1 + HORIZON)
    walk = day_bars.iloc[entry_idx+1:j_end]
    exit_px, exit_reason, exit_dt = None, None, None
    for _, r in walk.iterrows():
        c = r["fut_close"]
        if c >= target_px:
            exit_px, exit_reason, exit_dt = target_px, "TARGET", r["datetime"]; break
        if c <= stop_px:
            exit_px, exit_reason, exit_dt = stop_px, "STOP", r["datetime"]; break
    if exit_px is None:
        if walk.empty:
            exit_px, exit_reason = entry_px, "NO_BARS"
            exit_dt = day_bars.iloc[entry_idx]["datetime"]
        else:
            exit_px, exit_reason = float(walk["fut_close"].iloc[-1]), "TIME"
            exit_dt = walk.iloc[-1]["datetime"]
    gross = (exit_px - entry_px) * LOT * size_mult
    costs = COSTS_INR * size_mult
    net = gross - costs
    if net < STOP_FLOOR:
        net, exit_reason = STOP_FLOOR, "STOP_FLOOR"
    return {
        "entry_px": entry_px, "exit_px": exit_px,
        "target_px": target_px, "stop_px": stop_px,
        "datetime_exit": exit_dt,
        "gross_pnl_inr": gross, "costs_inr": costs, "net_pnl_inr": net,
        "exit_reason": exit_reason, "size_mult": size_mult,
    }


def run_one_day(target_date, model, cfg) -> pd.DataFrame:
    raw = load_minute_bars(target_date)
    if raw.empty or len(raw) < 60:
        return pd.DataFrame(columns=LEDGER_COLS)

    feats = compute_features(raw, target_date.date())
    if feats.empty:
        return pd.DataFrame(columns=LEDGER_COLS)

    OI_FILL = ["oi_chg_1m","oi_chg_5m","oi_chg_30m","vol_oi_ratio","vol_zscore"]
    for c in OI_FILL:
        if c in feats.columns:
            feats[c] = feats[c].fillna(0.0)
    feats = feats.dropna(subset=FUTURES_FEATURES)
    if feats.empty:
        return pd.DataFrame(columns=LEDGER_COLS)

    feats["datetime"]   = pd.to_datetime(feats["datetime"])
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])
    feats = add_regime(feats)

    # Macro signals
    macro = load_macro_signals()
    macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
    target_norm = pd.Timestamp(target_date.date())
    macro_row = macro[macro["date"] == target_norm]
    if macro_row.empty:
        macro_row = macro[macro["date"] <= target_norm].iloc[[-1]]
    ret_5d_prev = float(macro_row["ret_5d_prev"].iloc[0]) if not macro_row.empty else 0.0
    ret_20d_prev = float(macro_row["ret_20d_prev"].iloc[0]) if not macro_row.empty else 0.0
    vix_block = int(macro_row["vix_state_block_prev"].iloc[0]) if not macro_row.empty else 0

    # Variant C macro guard
    if cfg.get("ret_5d_cut") is not None and ret_5d_prev <= cfg["ret_5d_cut"]:
        return pd.DataFrame(columns=LEDGER_COLS)
    if cfg.get("ret_20d_cut") is not None and ret_20d_prev <= cfg["ret_20d_cut"]:
        return pd.DataFrame(columns=LEDGER_COLS)
    if cfg.get("vix_state_block") and vix_block == 1:
        return pd.DataFrame(columns=LEDGER_COLS)

    # Hard filters
    t = feats["datetime"].dt.time
    mod = feats["minute_of_day"] - 9*60 - 15
    eligible = feats[
        (mod >= ENTRY_MIN) & (t < HARD_CUTOFF) &
        ~((t >= SKIP_START) & (t < SKIP_END)) &
        ~feats["regime"].isin(SKIP_REGIMES)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=LEDGER_COLS)

    eligible["long_score"] = model.predict(eligible[FUTURES_FEATURES])
    sp = cfg["signal_pct"]
    p_sig = eligible["long_score"].quantile(sp)
    p_high = eligible["long_score"].quantile(HIGH_CONF_PCT)
    eligible["take_long"] = eligible["long_score"] >= p_sig
    eligible["size_mult"] = np.where(eligible["long_score"] >= p_high, 1.5, 1.0)

    candidates = eligible[eligible["take_long"]].sort_values("datetime").reset_index(drop=True)

    bars_for_sim = feats[["datetime","f_close"]].rename(
        columns={"f_close":"fut_close"}).sort_values("datetime").reset_index(drop=True)

    rows = []
    daily_pnl, n_taken = 0.0, 0
    intraday_halt = cfg.get("intraday_halt", -15000.0)
    run_ts = datetime.now().isoformat(timespec="seconds")

    for _, c in candidates.iterrows():
        if n_taken >= MAX_TRADES: break
        if daily_pnl <= intraday_halt: break
        idx_arr = bars_for_sim.index[bars_for_sim["datetime"] == c["datetime"]]
        if len(idx_arr) == 0: continue
        i = int(idx_arr[0])
        entry_px = float(bars_for_sim["fut_close"].iat[i])
        sim = simulate_long_trade(i, bars_for_sim, entry_px, float(c["size_mult"]))
        rows.append({
            "paper_trade_id": str(uuid.uuid4())[:12],
            "run_timestamp": run_ts,
            "trade_date": pd.Timestamp(c["trade_date"]).strftime("%Y-%m-%d"),
            "datetime_entry": pd.Timestamp(c["datetime"]).isoformat(),
            "datetime_exit": pd.Timestamp(sim["datetime_exit"]).isoformat() if sim["datetime_exit"] is not None else None,
            "side": "LONG",
            "entry_px": sim["entry_px"], "exit_px": sim["exit_px"],
            "target_px": sim["target_px"], "stop_px": sim["stop_px"],
            "size_mult": sim["size_mult"], "lot": LOT,
            "gross_pnl_inr": sim["gross_pnl_inr"],
            "costs_inr": sim["costs_inr"], "net_pnl_inr": sim["net_pnl_inr"],
            "exit_reason": sim["exit_reason"],
            "regime": str(c["regime"]),
            "long_score": float(c["long_score"]),
            "reason_codes": reason_codes(c, cfg),
            "model_version": MODEL_VERSION,
            "source": LEDGER_SOURCE,
        })
        daily_pnl += sim["net_pnl_inr"]
        n_taken += 1
    return pd.DataFrame(rows, columns=LEDGER_COLS)


def date_already_logged(target_date, ledger_path) -> bool:
    if not ledger_path.exists(): return False
    try:
        ex = pd.read_csv(ledger_path, usecols=["trade_date","source"])
    except Exception:
        return False
    target = target_date.strftime("%Y-%m-%d")
    return ((ex["trade_date"] == target) & (ex["source"] == LEDGER_SOURCE)).any()


def remove_date_from_ledger(target_date, ledger_path) -> int:
    if not ledger_path.exists(): return 0
    df = pd.read_csv(ledger_path)
    target = target_date.strftime("%Y-%m-%d")
    mask = (df["trade_date"] == target) & (df["source"] == LEDGER_SOURCE)
    n = int(mask.sum())
    df[~mask].to_csv(ledger_path, index=False)
    return n


def get_most_recent_data_date() -> pd.Timestamp:
    raw = pd.read_csv(NIFTY_MIN_PATH, usecols=["date"])
    return pd.to_datetime(raw["date"]).max().normalize()


def preflight_freshness(target_date: pd.Timestamp, auto_mode: bool) -> list:
    required = latest_weekday() if auto_mode else pd.Timestamp(target_date).normalize()
    return [
        check_file_freshness("NIFTY minute bars", NIFTY_MIN_PATH, required),
        check_file_freshness("NIFTY daily index", NIFTY_DAILY_PATH, required),
        check_file_freshness("INDIA VIX daily", VIX_DAY_PATH, required),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER))
    parser.add_argument("--ignore-kill-switch", action="store_true")
    parser.add_argument("--allow-stale-data", action="store_true",
                          help="Bypass freshness failures for historical/debug replays")
    parser.add_argument("--allow-incomplete-session", action="store_true",
                          help="Bypass minute-session completeness failures")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    if KILL_SWITCH.exists() and not args.ignore_kill_switch:
        print(f"  ⚠️  PAPER TRADING IS HALTED ({KILL_SWITCH})")
        return 2

    if args.date:
        target = pd.Timestamp(args.date).normalize()
    elif args.auto:
        target = get_most_recent_data_date()
        print(f"  --auto resolved to {target.date()}")
    else:
        print("  must pass --date or --auto"); return 1

    checks = preflight_freshness(target, args.auto)
    print_freshness(checks)
    if freshness_failed(checks) and not args.allow_stale_data:
        print("\n  no run: stale data")
        print("  update source data or pass --allow-stale-data for an intentional historical replay")
        return 5

    quality = check_minute_session_quality(NIFTY_MIN_PATH, target)
    print_session_quality(quality)
    if not quality.ok and not args.allow_incomplete_session:
        print("\n  no run: incomplete session data")
        print("  repair source bars or pass --allow-incomplete-session for an intentional replay")
        return 6

    if date_already_logged(target, ledger_path):
        if not args.force:
            print(f"  {target.date()}: Variant C already in ledger")
            return 0
        n = remove_date_from_ledger(target, ledger_path)
        print(f"  --force: removed {n} prior Variant C rows")

    cfg = load_variant_c_config()
    print(f"\n  Variant C config: {cfg}")
    print(f"  paper-trading {target.date()}")
    print(f"  ledger: {ledger_path}")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    rows = run_one_day(target, model, cfg)

    if rows.empty:
        print("\n  no Variant C trades (filters / no signals / regime guarded)")
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not ledger_path.exists()
        rows.to_csv(ledger_path, mode="a", header=write_header, index=False)
        net = rows["net_pnl_inr"].sum()
        wins = (rows["net_pnl_inr"] > 0).sum()
        print(f"\n  trades:  {len(rows)}")
        print(f"  wins:    {wins}/{len(rows)} ({100*wins/len(rows):.0f}%)")
        print(f"  net:     ₹{net:+,.0f}")
        for _, r in rows.iterrows():
            arrow = "▲" if r["net_pnl_inr"] > 0 else "▼"
            print(f"   {arrow}  {r['datetime_entry'][11:16]}  {r['side']}  "
                  f"entry={r['entry_px']:>9,.2f} exit={r['exit_px']:>9,.2f} "
                  f"({r['exit_reason']:>10s})  ₹{r['net_pnl_inr']:>+8,.0f}  x{r['size_mult']:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
