#!/usr/bin/env python3
"""V5 Phase 3 — Composite gated backtest on 2024 blind window.

Pipeline per minute (dte ∈ {2, 3}):
  1. V4-B vol kill-switch: skip if predicted_rv > train_quantile_90
  2. Binary gate: skip if gate_score < 0.70
  3. Ranker: pick top-1 strategy
  4. If top-1 = NO_TRADE → skip
  5. Daily caps: max 2 trades/(index, day), max 4 trades/day total
  6. Daily loss halt: stop trading if cumulative day PnL < -₹15,000
  7. Per-trade stop-loss: simulate path; exit at -₹3,000 if MTM hits it

Reports realistic blind-window metrics: trades/day, win rate, mean PnL,
total PnL, max drawdown, monthly profile.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime as dt_cls, time as dtime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.v4_chain import discover_days, load_options_day, load_spot
from optinet.v5_simulator import (STRATEGIES, STRIKE_STEP, LOT_SIZE,
                                     BROK_PER_LEG, SLIPPAGE_ATM_PCT, SLIPPAGE_OTM_PCT,
                                     STT_RATE, EXCH_RATE, SEBI_RATE, STAMP_RATE, GST_RATE,
                                     OTM_THRESHOLD_PCT,
                                     bs_price, build_chain_lookup, lookup_price,
                                     compute_cost, _round_to_step, _T_years,
                                     RISK_FREE, TRADING_DAYS, TRADING_MINUTES)

LABELS_DIR = PROJECT_ROOT / "cache/optinet_v5/strategy_labels"
CHAIN_DIR  = PROJECT_ROOT / "cache/optinet_v4/chain_features"
FUT_DIR    = PROJECT_ROOT / "cache/optinet_v5/futures_features"
DATA_ROOT  = PROJECT_ROOT / "data/option_data"
MODEL_DIR  = PROJECT_ROOT / "models/optinet_v5"
VOL_MODEL  = PROJECT_ROOT / "models/optinet_v4/rv_30m_forward.lgb"
OUT_DIR    = PROJECT_ROOT / "results/optinet_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Risk parameters ──────────────────────────────────────────────────────────
GATE_THRESHOLD = 0.70
PER_TRADE_STOP_LOSS_INR = -3000.0
MAX_TRADES_PER_INDEX_PER_DAY = 2
MAX_TRADES_PER_DAY_TOTAL = 4
DAILY_LOSS_HALT_INR = -15000.0
VOL_KILL_PCT = 0.90  # skip when predicted_rv > 90th pct of training distribution

# Feature columns (must match training)
CHAIN_FEATURES = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
]
FUT_FEATURES = [
    "fut_basis", "fut_basis_change_30m",
    "fut_oi_change_1m", "fut_oi_change_5m", "fut_oi_change_30m",
    "fut_volume_oi_ratio", "fut_session_position",
    "fut_oi_x_long_buildup", "fut_oi_x_short_buildup",
    "fut_oi_x_short_cover", "fut_oi_x_long_unwind",
]
SIM_CONTEXT = ["minute_of_day", "hour_of_day", "dte"]
DESCRIPTORS = ["sd_horizon_min", "sd_n_legs", "sd_short_vol", "sd_long_vol",
                "sd_directional", "sd_defined_risk", "sd_is_eod", "sd_is_no_trade",
                "sd_atm_dist_pct"]
RANKER_FEATURES = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT + DESCRIPTORS
GATE_FEATURES   = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT
VOL_FEATURES = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
    "minute_of_day", "minutes_to_close",
    "atm_iv_lag5", "atm_iv_lag15",
    "skew_slope_lag5", "skew_slope_lag15",
    "pcr_oi_lag5", "pcr_oi_lag15",
    "pcr_vol_lag5", "pcr_vol_lag15",
    "iv_rv_spread_lag5", "iv_rv_spread_lag15",
    "realized_vol_30m_lag5", "realized_vol_30m_lag15",
    "max_oi_total_dist_pct_lag5", "max_oi_total_dist_pct_lag15",
]

STRATEGY_DESCRIPTORS = {
    0:  {"sd_horizon_min": 0,   "sd_n_legs": 0, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 1, "sd_atm_dist_pct": 0.0},
    1:  {"sd_horizon_min": 30,  "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    2:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    3:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.01},
    4:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.01},
    5:  {"sd_horizon_min": 360, "sd_n_legs": 4, "sd_short_vol": 1, "sd_long_vol": 0, "sd_directional": 0, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.05},
    6:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    7:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    8:  {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    9:  {"sd_horizon_min": 360, "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 0, "sd_directional": 1, "sd_defined_risk": 1, "sd_is_eod": 1, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.02},
    10: {"sd_horizon_min": 30,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    11: {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.0},
    12: {"sd_horizon_min": 60,  "sd_n_legs": 2, "sd_short_vol": 0, "sd_long_vol": 1, "sd_directional": 0, "sd_defined_risk": 0, "sd_is_eod": 0, "sd_is_no_trade": 0, "sd_atm_dist_pct": 0.01},
}


# ─── Path-based stop-loss simulator ───────────────────────────────────────────


def simulate_trade_with_stop_BS_DEBUG_ONLY(spec, entry_t, exit_t, atm_strike, atm_iv,
                                expiry, spot_lookup, chain_lookup, lot, step,
                                stop_loss_inr=PER_TRADE_STOP_LOSS_INR):
    """⚠️ DEBUG-ONLY ⚠️ BS-reprice intra-trade stop simulator.

    DEPRECATED 2026-05-27. The V5 audit found this simulator over-states PnL by
    ~₹870/trade for short-premium strategies because it holds atm_iv constant
    from entry. Use simulate_trade_with_stop_market() for production.

    See archive/v5/POSTMORTEM.md for the failure mode analysis.

    Returns: (net_pnl, was_stopped, exit_minute_offset, entry_premium_inr)
    """
    if spec.id == 0:
        return 0.0, False, 0, 0.0

    S0 = float(spot_lookup.loc[entry_t]) if entry_t in spot_lookup.index else float("nan")
    if not (S0 > 0):
        return float("nan"), False, 0, float("nan")
    T0 = _T_years(entry_t, expiry)

    # Resolve legs and entry prices
    leg_meta = []  # (opt_type, K, direction, off_pct, entry_px_per_share)
    for leg in spec.legs:
        target_K = _round_to_step(atm_strike * (1 + leg.strike_offset_pct), step)
        entry_px, _ = lookup_price(entry_t, target_K, leg.opt_type,
                                    chain_lookup, S0, target_K, T0, atm_iv, step)
        if entry_px <= 0:
            return float("nan"), False, 0, float("nan")
        leg_meta.append((leg.opt_type, target_K, leg.direction, leg.strike_offset_pct, entry_px))

    # Initial "received" or "paid" premium per share (signed)
    entry_total_signed = sum((1 if d == "short" else -1) * px for (_, _, d, _, px) in leg_meta)
    abs_entry_premium_per_share = sum(abs(px) for (*_, px) in leg_meta)
    entry_premium_inr = abs_entry_premium_per_share * lot

    # Cost (computed once at entry — slippage applied on exit too)
    legs_premia_for_cost = [(px, px, d, abs(off)) for (_, _, d, off, px) in leg_meta]
    # Note: exit px will be different but cost approximated at-entry; refine below

    # Walk minute by minute from entry to exit, tracking MTM
    minutes_in_path = pd.date_range(entry_t, exit_t, freq="1min")
    available_minutes = [t for t in minutes_in_path if t in spot_lookup.index]
    if not available_minutes:
        return float("nan"), False, 0, entry_premium_inr

    running_min_pnl = float("inf")
    final_t = available_minutes[-1]
    final_exit_premium_per_share_signed = 0.0
    stopped_at = None

    for i, t in enumerate(available_minutes):
        S = float(spot_lookup.loc[t])
        T_t = max(T0 - i / (TRADING_DAYS * TRADING_MINUTES), 1e-6)

        cur_total_signed = 0.0
        for (opt_type, K, direction, _, _entry_px) in leg_meta:
            cur_px = bs_price(S, K, T_t, atm_iv, opt_type)
            sign = 1 if direction == "short" else -1
            cur_total_signed += sign * cur_px

        # MTM PnL = (entry signed total - current signed total) × lot − approx cost
        mtm_per_share = entry_total_signed - cur_total_signed
        mtm_pnl = mtm_per_share * lot
        # Approximate cost over the trade life:
        legs_for_cost = [(_entry_px, abs(cur_total_signed) / max(len(leg_meta), 1),
                           direction, abs(off))
                         for (_, _, direction, off, _entry_px) in leg_meta]
        approx_cost = compute_cost(legs_for_cost, lot)
        mtm_pnl_after_cost = mtm_pnl - approx_cost

        if mtm_pnl_after_cost < running_min_pnl:
            running_min_pnl = mtm_pnl_after_cost

        if mtm_pnl_after_cost < stop_loss_inr:
            # Stopped out
            stopped_at = i
            return float(stop_loss_inr), True, i, entry_premium_inr

    # No stop-out: compute terminal PnL with proper cost
    final_S = float(spot_lookup.loc[final_t])
    final_T = max(T0 - len(available_minutes) / (TRADING_DAYS * TRADING_MINUTES), 1e-6)
    legs_premia_full_cost = []
    final_total_signed = 0.0
    for (opt_type, K, direction, off, entry_px) in leg_meta:
        exit_px = bs_price(final_S, K, final_T, atm_iv, opt_type)
        if exit_px <= 0:
            # Intrinsic at expiry
            if final_t.date() == expiry.date():
                exit_px = max(final_S - K, 0) if opt_type == "CE" else max(K - final_S, 0)
        sign = 1 if direction == "short" else -1
        final_total_signed += sign * exit_px
        legs_premia_full_cost.append((entry_px, exit_px, direction, abs(off)))

    gross_per_share = entry_total_signed - final_total_signed
    gross_pnl = gross_per_share * lot
    cost = compute_cost(legs_premia_full_cost, lot)
    net_pnl = gross_pnl - cost
    # Apply stop-loss as a floor (in case the running-min check missed it due to cost approx)
    if net_pnl < stop_loss_inr:
        return float(stop_loss_inr), True, len(available_minutes), entry_premium_inr
    return float(net_pnl), False, len(available_minutes), entry_premium_inr


# ─── Market-price stop-loss simulator ─────────────────────────────────────────


def simulate_trade_with_stop_market(spec, entry_t, exit_t, atm_strike, atm_iv,
                                       expiry, spot_lookup, chain_lookup, lot, step,
                                       stop_loss_inr=PER_TRADE_STOP_LOSS_INR):
    """Same interface as simulate_trade_with_stop, but uses ACTUAL minute-bar
    option chain prices for intra-trade MTM and exit. Falls back to BS only if
    a minute's price is missing from the chain (data gap).

    Returns: (net_pnl, was_stopped, exit_minute_offset, entry_premium_inr,
                synthetic_minute_count)
    """
    if spec.id == 0:
        return 0.0, False, 0, 0.0, 0

    S0 = float(spot_lookup.loc[entry_t]) if entry_t in spot_lookup.index else float("nan")
    if not (S0 > 0):
        return float("nan"), False, 0, float("nan"), 0
    T0 = _T_years(entry_t, expiry)

    # Resolve legs and entry prices (these always come from chain when possible)
    leg_meta = []  # (opt_type, K, direction, off_pct, entry_px_per_share)
    for leg in spec.legs:
        target_K = _round_to_step(atm_strike * (1 + leg.strike_offset_pct), step)
        entry_px, _ = lookup_price(entry_t, target_K, leg.opt_type,
                                    chain_lookup, S0, target_K, T0, atm_iv, step)
        if entry_px <= 0:
            return float("nan"), False, 0, float("nan"), 0
        leg_meta.append((leg.opt_type, target_K, leg.direction, leg.strike_offset_pct, entry_px))

    entry_total_signed = sum((1 if d == "short" else -1) * px for (_, _, d, _, px) in leg_meta)
    abs_entry_premium_per_share = sum(abs(px) for (*_, px) in leg_meta)
    entry_premium_inr = abs_entry_premium_per_share * lot

    # Walk minute by minute using ACTUAL chain prices
    minutes_in_path = pd.date_range(entry_t, exit_t, freq="1min")
    available_minutes = [t for t in minutes_in_path if t in spot_lookup.index]
    if not available_minutes:
        return float("nan"), False, 0, entry_premium_inr, 0

    synthetic_minute_count = 0
    final_t = available_minutes[-1]

    for i, t in enumerate(available_minutes):
        S = float(spot_lookup.loc[t])
        T_t = max(T0 - i / (TRADING_DAYS * TRADING_MINUTES), 1e-6)

        cur_total_signed = 0.0
        any_synth = False
        leg_cur_pxs = []
        for (opt_type, K, direction, off_pct, _entry_px) in leg_meta:
            cur_px, is_synth = lookup_price(t, K, opt_type,
                                              chain_lookup, S, K, T_t, atm_iv, step)
            if is_synth:
                any_synth = True
            sign = 1 if direction == "short" else -1
            cur_total_signed += sign * cur_px
            leg_cur_pxs.append(cur_px)
        if any_synth:
            synthetic_minute_count += 1

        mtm_per_share = entry_total_signed - cur_total_signed
        mtm_pnl = mtm_per_share * lot
        # Approximate cost using current legs (entry_px, current_px, direction, off)
        legs_for_cost = [(_entry_px, leg_cur_pxs[j], direction, abs(off))
                         for j, (_, _, direction, off, _entry_px) in enumerate(leg_meta)]
        approx_cost = compute_cost(legs_for_cost, lot)
        mtm_pnl_after_cost = mtm_pnl - approx_cost

        if mtm_pnl_after_cost < stop_loss_inr:
            return float(stop_loss_inr), True, i, entry_premium_inr, synthetic_minute_count

    # No stop: compute terminal PnL with actual exit prices
    final_S = float(spot_lookup.loc[final_t])
    final_T = max(T0 - len(available_minutes) / (TRADING_DAYS * TRADING_MINUTES), 1e-6)
    legs_premia_full_cost = []
    final_total_signed = 0.0
    final_synth = False
    for (opt_type, K, direction, off, entry_px) in leg_meta:
        exit_px, is_synth = lookup_price(final_t, K, opt_type,
                                           chain_lookup, final_S, K, final_T, atm_iv, step)
        if is_synth:
            final_synth = True
            # Last-resort: intrinsic at expiry
            if final_t.date() == expiry.date():
                exit_px = max(final_S - K, 0) if opt_type == "CE" else max(K - final_S, 0)
        sign = 1 if direction == "short" else -1
        final_total_signed += sign * exit_px
        legs_premia_full_cost.append((entry_px, exit_px, direction, abs(off)))

    if final_synth:
        synthetic_minute_count += 1

    gross_per_share = entry_total_signed - final_total_signed
    gross_pnl = gross_per_share * lot
    cost = compute_cost(legs_premia_full_cost, lot)
    net_pnl = gross_pnl - cost
    if net_pnl < stop_loss_inr:
        return float(stop_loss_inr), True, len(available_minutes), entry_premium_inr, synthetic_minute_count
    return float(net_pnl), False, len(available_minutes), entry_premium_inr, synthetic_minute_count


# ─── Feature loading & ranker scoring ─────────────────────────────────────────


def add_lags_and_time(df: pd.DataFrame) -> pd.DataFrame:
    df["trade_date"] = df["datetime"].dt.normalize()
    grp = df.groupby(["index", "trade_date"], sort=False)
    for col in ["atm_iv", "skew_slope", "pcr_oi", "pcr_vol",
                "realized_vol_30m", "iv_rv_spread", "max_oi_total_dist_pct"]:
        df[f"{col}_lag5"]  = grp[col].shift(5)
        df[f"{col}_lag15"] = grp[col].shift(15)
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minutes_to_close"] = ((15 * 60 + 30) - df["minute_of_day"]).clip(lower=0)
    df["hour_of_day"] = df["datetime"].dt.hour
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--dte_buckets", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--variant", choices=["constrained_ranker", "ss_eod_only", "full_ranker"],
                        default="constrained_ranker",
                        help="Which variant to run")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Optional suffix for output files (e.g. '_constrained')")
    parser.add_argument("--chronological", action="store_true",
                        help="Daily caps select FIRST 2 minutes by time (no top-K-by-score look-ahead). "
                              "This is the runtime-equivalent selection rule.")
    parser.add_argument("--apply_runtime_cutoff", action="store_true",
                        help="Reject candidate minutes >= 14:55 (matches runtime cutoff)")
    parser.add_argument("--use_bs_reprice_stop_DEBUG", action="store_true",
                        help="DEBUG ONLY: use BS-reprice intra-trade stop (held atm_iv). "
                              "DEPRECATED 2026-05-27 — over-states PnL by ~₹870/trade "
                              "for short-premium strategies. The default is the realistic "
                              "market-price stop simulator. Only enable for diagnostic comparisons.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Override the gate/ranker model directory. "
                              "Use archive/v5/models/optinet_v5 to read archived V5 gates.")
    parser.add_argument("--force_strategy_id", type=int, default=None,
                        help="Force a specific strategy_id as the only action when "
                              "the gate fires. Overrides --variant. Used by Sprint #2 "
                              "(long-vol mirror): pass 10, 11, or 12.")
    parser.add_argument("--gate_threshold", type=float, default=None,
                        help="Override default gate threshold (0.70).")
    args = parser.parse_args()

    # Allow per-run override of the gate threshold
    global GATE_THRESHOLD
    if args.gate_threshold is not None:
        GATE_THRESHOLD = float(args.gate_threshold)

    print("=" * 80)
    print("V5 Phase 3 — Composite gated backtest")
    print(f"  Year: {args.year}  Dte buckets: {args.dte_buckets}")
    print(f"  VARIANT: {args.variant}")
    print(f"  Gate threshold: {GATE_THRESHOLD}")
    print(f"  Per-trade stop-loss: ₹{PER_TRADE_STOP_LOSS_INR}")
    print(f"  Daily loss halt: ₹{DAILY_LOSS_HALT_INR}")
    print(f"  Vol kill quantile: {VOL_KILL_PCT}")
    print(f"  Caps: {MAX_TRADES_PER_INDEX_PER_DAY}/index/day, "
          f"{MAX_TRADES_PER_DAY_TOTAL} total/day")
    if args.use_bs_reprice_stop_DEBUG:
        print("  ⚠️  STOP SIMULATOR: BS-reprice (DEBUG ONLY — DEPRECATED 2026-05-27)")
        print("  ⚠️  This simulator over-states PnL by ~₹870/trade for short-premium")
        print("  ⚠️  strategies. See archive/v5/POSTMORTEM.md.")
    else:
        print("  STOP SIMULATOR: market-price (realistic, default)")
    print("=" * 80)

    # Load all required features for year + 2020-2022 (for vol threshold calibration)
    print("\nLoading features …")
    chain_files = sorted(CHAIN_DIR.rglob("data.parquet"))
    chain_full = pd.concat([pd.read_parquet(f) for f in chain_files], ignore_index=True)
    chain_full["datetime"] = pd.to_datetime(chain_full["datetime"])
    chain_full = add_lags_and_time(chain_full)

    fut_files = sorted(FUT_DIR.rglob("data.parquet"))
    fut_full = pd.concat([pd.read_parquet(f) for f in fut_files], ignore_index=True)
    fut_full["datetime"] = pd.to_datetime(fut_full["datetime"])

    # Run V4-B vol forecast on 2020-2022 to calibrate kill threshold
    vol_model = lgb.Booster(model_file=str(VOL_MODEL))
    train_mask = (chain_full["datetime"].dt.year < 2023)
    train_chain = chain_full[train_mask].dropna(subset=VOL_FEATURES)
    train_pred_rv = vol_model.predict(train_chain[VOL_FEATURES])
    vol_threshold = float(np.quantile(train_pred_rv, VOL_KILL_PCT))
    print(f"  Vol kill threshold (training {VOL_KILL_PCT*100:.0f}th pct): {vol_threshold:.4f}")

    # Filter to year of interest
    chain_year = chain_full[chain_full["datetime"].dt.year == args.year].copy()
    fut_year = fut_full[fut_full["datetime"].dt.year == args.year].copy()
    feats = chain_year.merge(fut_year, on=["index", "datetime"], how="inner")

    # Predict V4-B on year
    feats_valid_vol = feats.dropna(subset=VOL_FEATURES)
    feats["pred_rv"] = np.nan
    feats.loc[feats_valid_vol.index, "pred_rv"] = vol_model.predict(feats_valid_vol[VOL_FEATURES])
    feats["vol_kill"] = (feats["pred_rv"] > vol_threshold).astype(int)
    print(f"  Year {args.year}: {len(feats):,} minute rows; "
          f"vol_kill activates at {feats['vol_kill'].mean():.2%} of minutes")

    # Load gates and rankers per dte bucket
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    gates = {}
    rankers = {}
    for dte in args.dte_buckets:
        gpath = model_dir / f"gate_dte{dte}.lgb"
        if args.variant == "constrained_ranker":
            rpath = model_dir / f"ranker_dte{dte}_nosid_desc_constrained4.lgb"
        elif args.variant == "full_ranker":
            rpath = model_dir / f"ranker_dte{dte}_nosid_desc.lgb"
        else:  # ss_eod_only — no ranker needed but load for parity
            rpath = model_dir / f"ranker_dte{dte}_nosid_desc.lgb"
        if not gpath.exists():
            print(f"  WARNING missing gate for dte={dte}")
            continue
        gates[dte] = lgb.Booster(model_file=str(gpath))
        if args.variant != "ss_eod_only" and rpath.exists():
            rankers[dte] = lgb.Booster(model_file=str(rpath))
    print(f"  Loaded {len(gates)} gates and {len(rankers)} rankers")

    # Load strategy_labels for the year (for the candidate set)
    label_files = [f for f in sorted(LABELS_DIR.rglob("data.parquet"))
                    if f"year={args.year}" in str(f)]
    labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels = labels[labels["dte_bucket"].isin(args.dte_buckets) & labels["valid"]].copy()

    # Drop overlapping cols, merge with feats
    labels = labels.drop(columns=["atm_iv", "atm_strike", "spot",
                                    "minute_of_day", "hour_of_day", "dte",
                                    "trade_date"],
                          errors="ignore")
    df = labels.merge(feats, on=["index", "datetime"], how="inner")
    # Add descriptors
    desc_df = pd.DataFrame.from_dict(STRATEGY_DESCRIPTORS, orient="index").reset_index()
    desc_df = desc_df.rename(columns={"index": "strategy_id"})
    df = df.merge(desc_df, on="strategy_id", how="left")
    df["dte"] = df["dte_bucket"].map({2: 4, 3: 5})  # median dte within bucket

    # If using constrained ranker, restrict to the same candidate set used in training
    if args.variant == "constrained_ranker" and args.force_strategy_id is None:
        CONSTRAINED_SET = {0, 2, 4, 5}
        before = len(df)
        df = df[df["strategy_id"].isin(CONSTRAINED_SET)].copy()
        print(f"  constrained inference filter: {before:,} → {len(df):,} rows")

    print(f"  Merged: {len(df):,} candidate-rows after vol/feature merge")

    # ─── Per-minute decision pipeline ────────────────────────────────────────
    print("\nRunning pipeline …")
    t0 = time.time()
    df = df.sort_values(["index", "datetime", "strategy_id"]).reset_index(drop=True)

    # Score gate per (index, datetime)
    minute_keys = df[["index", "datetime", "dte_bucket", "vol_kill"] + GATE_FEATURES].drop_duplicates(
        subset=["index", "datetime"]).reset_index(drop=True)
    minute_keys["gate_score"] = np.nan
    for dte in args.dte_buckets:
        m = minute_keys["dte_bucket"] == dte
        if not m.any() or dte not in gates:
            continue
        sub = minute_keys[m].dropna(subset=GATE_FEATURES)
        scores = gates[dte].predict(sub[GATE_FEATURES])
        minute_keys.loc[sub.index, "gate_score"] = scores
    minute_keys["pass_vol"]  = minute_keys["vol_kill"] == 0
    minute_keys["pass_gate"] = minute_keys["gate_score"] >= GATE_THRESHOLD
    minute_keys["pass_pre_ranker"] = minute_keys["pass_vol"] & minute_keys["pass_gate"]
    print(f"  After vol filter:  {minute_keys['pass_vol'].sum():,}/{len(minute_keys):,}")
    print(f"  After vol+gate:    {minute_keys['pass_pre_ranker'].sum():,}/{len(minute_keys):,}")

    # Score ranker per (index, datetime, strategy)
    df["ranker_score"] = np.nan
    if args.force_strategy_id is not None:
        sid = int(args.force_strategy_id)
        df.loc[df["strategy_id"] == sid, "ranker_score"] = 1.0
        df["ranker_score"] = df["ranker_score"].fillna(0.0)
        print(f"  FORCED STRATEGY: strategy_id={sid} ({df['strategy_name'][df['strategy_id']==sid].iloc[0] if (df['strategy_id']==sid).any() else 'unknown'})")
    elif args.variant == "ss_eod_only":
        # Skip ranker; force strategy_id=2 (SHORT_STRADDLE_EOD) on every gate-pass minute
        # Score = 1.0 for SS_EOD, 0.0 for everything else, so top-1 is always SS_EOD
        df.loc[df["strategy_id"] == 2, "ranker_score"] = 1.0
        df["ranker_score"] = df["ranker_score"].fillna(0.0)
    else:
        for dte in args.dte_buckets:
            m = (df["dte_bucket"] == dte)
            if not m.any() or dte not in rankers:
                continue
            sub = df[m].dropna(subset=RANKER_FEATURES)
            scores = rankers[dte].predict(sub[RANKER_FEATURES])
            df.loc[sub.index, "ranker_score"] = scores

    # Top-1 per minute (highest ranker score)
    df_pre = df.merge(minute_keys[["index", "datetime", "gate_score", "pass_pre_ranker"]],
                       on=["index", "datetime"], how="left")
    df_pre = df_pre[df_pre["pass_pre_ranker"]].copy()
    df_pre["rank"] = df_pre.groupby(["index", "datetime"], sort=False)["ranker_score"].rank(
        method="first", ascending=False)
    top1 = df_pre[df_pre["rank"] == 1].copy()
    # Drop NO_TRADE picks
    top1 = top1[top1["strategy_id"] != 0].copy()
    print(f"  After ranker (top-1 != NO_TRADE): {len(top1):,} candidates")

    # Apply daily caps: max 2 per (index, day)
    if args.chronological:
        # Runtime-equivalent: pick FIRST 2 valid minutes by time, no score look-ahead
        if args.apply_runtime_cutoff:
            from datetime import time as dtime_t
            top1 = top1[top1["datetime"].dt.time < dtime_t(14, 55)].copy()
            print(f"  After 14:55 cutoff: {len(top1):,} candidates")
        top1 = top1.sort_values(["trade_date", "datetime"], ascending=[True, True])
        print(f"  SELECTION: chronological-first (no top-K look-ahead)")
    else:
        top1 = top1.sort_values(["trade_date", "gate_score"], ascending=[True, False])
        print(f"  SELECTION: top-K by gate_score (look-ahead — original backtest behavior)")
    top1["rank_within_day_index"] = top1.groupby(["trade_date", "index"]).cumcount()
    top1 = top1[top1["rank_within_day_index"] < MAX_TRADES_PER_INDEX_PER_DAY]
    top1["rank_within_day_total"] = top1.groupby("trade_date").cumcount()
    top1 = top1[top1["rank_within_day_total"] < MAX_TRADES_PER_DAY_TOTAL]
    print(f"  After daily caps:  {len(top1):,} candidates "
          f"({len(top1)/top1['trade_date'].nunique():.2f} trades/day avg)")

    # ─── Path-based stop-loss simulation ──────────────────────────────────────
    print(f"\nSimulating {len(top1):,} trades with intra-trade stop-loss …")
    t1 = time.time()

    # Build per-day spot/chain lookups (cached by date+index)
    cache = {}
    def get_lookups(idx, d):
        key = (idx, d.date())
        if key in cache:
            return cache[key]
        all_days = discover_days(DATA_ROOT, idx)
        for dd, opt_path, spot_path in all_days:
            if dd == d.date():
                options_day = load_options_day(opt_path, idx)
                spot_day = load_spot(spot_path, idx)
                expiry = pd.to_datetime(options_day["expiry"].iloc[0])
                cl = build_chain_lookup(options_day)
                sl = spot_day.set_index("datetime")["spot"]
                cache[key] = (cl, sl, expiry)
                return cache[key]
        return None

    results = []
    daily_pnl_running = {}  # trade_date -> cumulative day PnL

    for _, row in top1.iterrows():
        td = row["trade_date"]
        idx_name = row["index"]
        # Daily loss halt
        running = daily_pnl_running.get(td, 0.0)
        if running < DAILY_LOSS_HALT_INR:
            results.append({**row.to_dict(), "net_pnl_with_stop": 0.0, "was_stopped": False,
                              "halted": True, "exit_offset": 0,
                              "entry_premium_inr": float("nan")})
            continue

        lookups = get_lookups(idx_name, td)
        if lookups is None:
            continue
        chain_lookup, spot_lookup, expiry = lookups

        spec = STRATEGIES[int(row["strategy_id"])]
        atm_strike = int(row["atm_strike"]) if not pd.isna(row.get("atm_strike", np.nan)) else \
                     int(round(row["spot"] / STRIKE_STEP[idx_name]) * STRIKE_STEP[idx_name])
        atm_iv = float(row["atm_iv"])
        entry_t = pd.Timestamp(row["datetime"])
        # Exit time
        if spec.horizon_minutes >= 99999:
            exit_t = pd.Timestamp(dt_cls.combine(entry_t.date(), dtime(15, 25)))
        else:
            exit_t = entry_t + pd.Timedelta(minutes=spec.horizon_minutes)
            eod = pd.Timestamp(dt_cls.combine(entry_t.date(), dtime(15, 25)))
            exit_t = min(exit_t, eod)

        if args.use_bs_reprice_stop_DEBUG:
            net_pnl, stopped, off, entry_prem = simulate_trade_with_stop_BS_DEBUG_ONLY(
                spec, entry_t, exit_t, atm_strike, atm_iv,
                expiry, spot_lookup, chain_lookup,
                LOT_SIZE[idx_name], STRIKE_STEP[idx_name],
                stop_loss_inr=PER_TRADE_STOP_LOSS_INR)
            n_synth = -1  # not tracked for BS variant
        else:
            net_pnl, stopped, off, entry_prem, n_synth = simulate_trade_with_stop_market(
                spec, entry_t, exit_t, atm_strike, atm_iv,
                expiry, spot_lookup, chain_lookup,
                LOT_SIZE[idx_name], STRIKE_STEP[idx_name],
                stop_loss_inr=PER_TRADE_STOP_LOSS_INR)

        if not np.isfinite(net_pnl):
            continue

        daily_pnl_running[td] = running + net_pnl
        results.append({
            "index": idx_name, "datetime": entry_t, "trade_date": td,
            "strategy_id": int(row["strategy_id"]),
            "strategy_name": row["strategy_name"],
            "atm_strike": atm_strike, "atm_iv": atm_iv,
            "gate_score": float(row["gate_score"]),
            "ranker_score": float(row["ranker_score"]),
            "net_pnl_with_stop": net_pnl,
            "was_stopped": stopped,
            "halted": False,
            "exit_offset": off,
            "entry_premium_inr": entry_prem,
            "synthetic_minutes": n_synth,
        })

    trades = pd.DataFrame(results)
    print(f"  Simulated {len(trades):,} trades in {time.time()-t1:.0f}s")

    if trades.empty:
        print("  No trades passed all filters.")
        return

    # ─── Metrics ──────────────────────────────────────────────────────────────
    trades_taken = trades[~trades["halted"]].copy()
    n_days = trades_taken["trade_date"].nunique()
    print("\n=== Honest blind-window metrics ===")
    print(f"  Total trades       : {len(trades_taken):,}")
    print(f"  Trading days       : {n_days}")
    print(f"  Trades per day     : {len(trades_taken) / n_days:.2f}")
    print(f"  Win rate           : {(trades_taken['net_pnl_with_stop'] > 0).mean():.2%}")
    print(f"  Stop-out rate      : {trades_taken['was_stopped'].mean():.2%}")
    print(f"  Mean PnL/trade     : ₹{trades_taken['net_pnl_with_stop'].mean():.0f}")
    print(f"  Median PnL/trade   : ₹{trades_taken['net_pnl_with_stop'].median():.0f}")
    print(f"  Total PnL          : ₹{trades_taken['net_pnl_with_stop'].sum():.0f}")
    print(f"  Best trade         : ₹{trades_taken['net_pnl_with_stop'].max():.0f}")
    print(f"  Worst trade        : ₹{trades_taken['net_pnl_with_stop'].min():.0f}")

    # Daily PnL series
    daily = trades_taken.groupby("trade_date")["net_pnl_with_stop"].agg(
        ["sum", "count", "mean", "min"]).reset_index()
    daily["cum_pnl"] = daily["sum"].cumsum()
    daily["draw_from_peak"] = daily["cum_pnl"] - daily["cum_pnl"].cummax()
    max_dd = daily["draw_from_peak"].min()

    daily_pnl = daily["sum"]
    sharpe_d = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
    profit_factor = (daily_pnl[daily_pnl > 0].sum() /
                       max(-daily_pnl[daily_pnl < 0].sum(), 1))

    print(f"\n  Daily Sharpe       : {sharpe_d:.3f}")
    print(f"  Profit factor      : {profit_factor:.3f}")
    print(f"  Max drawdown       : ₹{max_dd:.0f}")
    print(f"  Best day           : ₹{daily_pnl.max():.0f}")
    print(f"  Worst day          : ₹{daily_pnl.min():.0f}")

    print("\n=== Per dte_bucket ===")
    by_dte = trades_taken.merge(
        df[["index", "datetime", "dte_bucket"]].drop_duplicates(
            subset=["index", "datetime"]),
        on=["index", "datetime"], how="left")
    for dte, sub in by_dte.groupby("dte_bucket"):
        print(f"  dte={int(dte)}: n={len(sub):,}  win={(sub['net_pnl_with_stop']>0).mean():.2%}  "
              f"avg=₹{sub['net_pnl_with_stop'].mean():.0f}  total=₹{sub['net_pnl_with_stop'].sum():.0f}")

    print("\n=== Per index ===")
    for idx, sub in trades_taken.groupby("index"):
        print(f"  {idx}: n={len(sub):,}  win={(sub['net_pnl_with_stop']>0).mean():.2%}  "
              f"avg=₹{sub['net_pnl_with_stop'].mean():.0f}  total=₹{sub['net_pnl_with_stop'].sum():.0f}")

    print("\n=== Per strategy ===")
    for sn, sub in trades_taken.groupby("strategy_name"):
        print(f"  {sn:>22}: n={len(sub):>4}  win={(sub['net_pnl_with_stop']>0).mean():.2%}  "
              f"avg=₹{sub['net_pnl_with_stop'].mean():.0f}  total=₹{sub['net_pnl_with_stop'].sum():.0f}")

    print("\n=== Monthly ===")
    trades_taken["month"] = trades_taken["trade_date"].dt.to_period("M")
    print(trades_taken.groupby("month")["net_pnl_with_stop"].agg(["sum", "count", "mean"]).round(0).to_string())

    # Save
    suffix = args.output_suffix or args.variant
    trades.to_parquet(OUT_DIR / f"phase3_trades_{suffix}.parquet", index=False)
    daily.to_parquet(OUT_DIR / f"phase3_daily_{suffix}.parquet", index=False)
    summary = {
        "year": args.year,
        "dte_buckets": args.dte_buckets,
        "variant": args.variant,
        "vol_kill_threshold": vol_threshold,
        "vol_kill_pct": VOL_KILL_PCT,
        "gate_threshold": GATE_THRESHOLD,
        "per_trade_stop_loss_inr": PER_TRADE_STOP_LOSS_INR,
        "daily_loss_halt_inr": DAILY_LOSS_HALT_INR,
        "n_trades": int(len(trades_taken)),
        "n_days": int(n_days),
        "trades_per_day": round(len(trades_taken) / n_days, 2),
        "win_rate": float((trades_taken["net_pnl_with_stop"] > 0).mean()),
        "stop_rate": float(trades_taken["was_stopped"].mean()),
        "mean_pnl_inr": float(trades_taken["net_pnl_with_stop"].mean()),
        "total_pnl_inr": float(trades_taken["net_pnl_with_stop"].sum()),
        "sharpe_daily": float(sharpe_d),
        "profit_factor": float(profit_factor),
        "max_drawdown_inr": float(max_dd),
    }
    (OUT_DIR / f"phase3_summary_{suffix}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → phase3_*_{suffix}.{{parquet,json}} in {OUT_DIR}")


if __name__ == "__main__":
    main()
