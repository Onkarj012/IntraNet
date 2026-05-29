"""V5 strategy simulator.

For each (index, datetime) decision point, simulates 13 candidate strategies
using real chain prices when available and Black-Scholes re-pricing as fallback.
Marks every strategy with synthetic_price_used flag.

Output schema (long format, one row per minute × strategy):
    index, datetime, trade_date, dte, dte_bucket,
    atm_strike, atm_iv, spot, T_years,
    minute_of_day, hour_of_day,
    strategy_id, strategy_name, horizon_label,
    entry_premium_inr, exit_premium_inr,
    gross_pnl, total_cost, net_pnl, pnl_per_premium,
    synthetic_price_used, valid
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as dt_cls, time as dtime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

# ─── Constants ────────────────────────────────────────────────────────────────

LOT_SIZE = {"NIFTY": 50, "BANKNIFTY": 15}
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}
RISK_FREE = 0.065
TRADING_DAYS = 252
TRADING_MINUTES = 375

# Cost model
BROK_PER_LEG = 40.0           # ₹/leg (₹40 entry + ₹40 exit per leg = ₹80 round-trip per leg)
SLIPPAGE_ATM_PCT = 0.015      # 1.5% per leg of premium for ATM
SLIPPAGE_OTM_PCT = 0.025      # 2.5% per leg of premium for OTM/wings
STT_RATE = 0.000625           # 0.0625% on options sell premium
EXCH_RATE = 0.000019
SEBI_RATE = 0.000001
STAMP_RATE = 0.00003          # entry only
GST_RATE = 0.18

# OTM threshold: anything more than 0.5% from ATM uses OTM slippage
OTM_THRESHOLD_PCT = 0.005


# ─── Strategy definitions ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Leg:
    opt_type: str           # "CE" or "PE"
    strike_offset_pct: float  # 0.0 = ATM, 0.01 = 1% OTM, -0.05 = 5% ITM
    direction: str          # "short" or "long"


@dataclass(frozen=True)
class StrategySpec:
    id: int
    name: str
    legs: tuple[Leg, ...]
    horizon_label: str
    horizon_minutes: int  # 0 = no trade; 99999 = EOD


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(0, "NO_TRADE", (), "NA", 0),

    StrategySpec(1, "SHORT_STRADDLE_30M",
                 (Leg("CE", 0.0, "short"), Leg("PE", 0.0, "short")), "30M", 30),
    StrategySpec(2, "SHORT_STRADDLE_EOD",
                 (Leg("CE", 0.0, "short"), Leg("PE", 0.0, "short")), "EOD", 99999),

    StrategySpec(3, "SHORT_STRANGLE_60M",
                 (Leg("CE", +0.01, "short"), Leg("PE", -0.01, "short")), "60M", 60),
    StrategySpec(4, "SHORT_STRANGLE_EOD",
                 (Leg("CE", +0.01, "short"), Leg("PE", -0.01, "short")), "EOD", 99999),

    StrategySpec(5, "IRON_CONDOR_EOD",
                 (Leg("CE", +0.01, "short"), Leg("PE", -0.01, "short"),
                  Leg("CE", +0.05, "long"),  Leg("PE", -0.05, "long")), "EOD", 99999),

    StrategySpec(6, "DEBIT_CALL_SPREAD_60M",
                 (Leg("CE", 0.0, "long"), Leg("CE", +0.02, "short")), "60M", 60),
    StrategySpec(7, "DEBIT_CALL_SPREAD_EOD",
                 (Leg("CE", 0.0, "long"), Leg("CE", +0.02, "short")), "EOD", 99999),

    StrategySpec(8, "DEBIT_PUT_SPREAD_60M",
                 (Leg("PE", 0.0, "long"), Leg("PE", -0.02, "short")), "60M", 60),
    StrategySpec(9, "DEBIT_PUT_SPREAD_EOD",
                 (Leg("PE", 0.0, "long"), Leg("PE", -0.02, "short")), "EOD", 99999),

    StrategySpec(10, "LONG_STRADDLE_30M",
                 (Leg("CE", 0.0, "long"), Leg("PE", 0.0, "long")), "30M", 30),
    StrategySpec(11, "LONG_STRADDLE_60M",
                 (Leg("CE", 0.0, "long"), Leg("PE", 0.0, "long")), "60M", 60),

    StrategySpec(12, "LONG_STRANGLE_60M",
                 (Leg("CE", +0.01, "long"), Leg("PE", -0.01, "long")), "60M", 60),
)


# ─── BS pricing ───────────────────────────────────────────────────────────────


def bs_price(S: float, K: float, T: float, sigma: float, opt_type: str) -> float:
    """Black-Scholes price (single)."""
    if not (T > 0 and sigma > 0 and S > 0 and K > 0):
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if opt_type == "CE":
        return float(S * norm.cdf(d1) - K * np.exp(-RISK_FREE * T) * norm.cdf(d2))
    return float(K * np.exp(-RISK_FREE * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _round_to_step(strike: float, step: int) -> int:
    return int(round(strike / step) * step)


def _dte_bucket(dte: int) -> int:
    if dte <= 0: return 0
    if dte <= 2: return 1
    if dte <= 4: return 2
    return 3


def _T_years(now: pd.Timestamp, expiry: pd.Timestamp) -> float:
    expiry_dt = pd.Timestamp(dt_cls.combine(expiry.date(), dtime(15, 30)))
    seconds = max((expiry_dt - now).total_seconds(), 0)
    return seconds / (365.25 * 24 * 3600)


def _exit_time(entry_t: pd.Timestamp, horizon_minutes: int) -> pd.Timestamp:
    """Compute exit timestamp; clamp to 15:25 of same day."""
    if horizon_minutes >= 99999:
        # EOD
        return pd.Timestamp(dt_cls.combine(entry_t.date(), dtime(15, 25)))
    proposed = entry_t + pd.Timedelta(minutes=horizon_minutes)
    eod = pd.Timestamp(dt_cls.combine(entry_t.date(), dtime(15, 25)))
    return min(proposed, eod)


# ─── Per-day chain pivot for fast lookups ────────────────────────────────────


def build_chain_lookup(options_day: pd.DataFrame) -> dict:
    """Build dict for O(1) (datetime, strike, opt_type) → close_price lookup.

    Also returns sorted list of (datetime, strikes_at_that_minute) for finding
    nearest strike when needed.
    """
    if options_day.empty:
        return {"prices": {}, "available_strikes_per_minute": {}, "all_strikes": np.array([])}

    prices = {}
    strikes_per_minute: dict[pd.Timestamp, set[int]] = {}
    for row in options_day.itertuples(index=False):
        key = (row.datetime, int(row.strike), row.opt_type)
        prices[key] = float(row.close)
        strikes_per_minute.setdefault(row.datetime, set()).add(int(row.strike))

    all_strikes = np.array(sorted(set(int(s) for s in options_day["strike"].unique())))
    return {"prices": prices,
            "available_strikes_per_minute": strikes_per_minute,
            "all_strikes": all_strikes}


def lookup_price(entry_t: pd.Timestamp, target_strike: int, opt_type: str,
                 chain_lookup: dict, S: float, K_for_bs: int, T: float, atm_iv: float,
                 step: int) -> tuple[float, bool]:
    """Look up real chain price; fall back to BS if missing.

    Returns (price, synthetic_flag).
    """
    # Direct hit on exact strike
    direct = chain_lookup["prices"].get((entry_t, int(target_strike), opt_type))
    if direct is not None and direct > 0:
        return direct, False

    # No real price → BS re-pricing
    bs = bs_price(S, K_for_bs, T, atm_iv, opt_type)
    return bs, True


# ─── Cost model ───────────────────────────────────────────────────────────────


def compute_cost(legs_premia: list[tuple[float, float, str, float]], lot: int) -> float:
    """legs_premia = [(entry_px_per_share, exit_px_per_share, direction, |strike_offset_pct|), ...]

    Returns total round-trip cost in INR.
    """
    n_legs = len(legs_premia)
    brokerage = BROK_PER_LEG * n_legs * 2  # entry + exit per leg

    sell_notional_entry = 0.0  # for STT (only on sells)
    total_notional = 0.0
    slippage_total = 0.0
    stamp_total = 0.0

    for entry_px, exit_px, direction, off_pct in legs_premia:
        slip_pct = SLIPPAGE_ATM_PCT if abs(off_pct) <= OTM_THRESHOLD_PCT else SLIPPAGE_OTM_PCT
        notional_entry = entry_px * lot
        notional_exit = exit_px * lot
        slippage_total += slip_pct * (notional_entry + notional_exit)
        total_notional += notional_entry + notional_exit
        stamp_total += STAMP_RATE * notional_entry  # stamp on entry buy/sell
        if direction == "short":
            sell_notional_entry += notional_entry  # STT on shorts at entry
        else:
            sell_notional_entry += notional_exit   # for longs, exit is the sell

    stt = STT_RATE * sell_notional_entry
    exch = EXCH_RATE * total_notional
    sebi = SEBI_RATE * total_notional
    gst = GST_RATE * (brokerage + exch + sebi)

    return brokerage + slippage_total + stt + exch + sebi + stamp_total + gst


# ─── Per-minute simulation ────────────────────────────────────────────────────


def simulate_minute(entry_t: pd.Timestamp, index_name: str,
                    spot_lookup: pd.Series, atm_strike: int, atm_iv: float,
                    expiry: pd.Timestamp, chain_lookup: dict) -> list[dict]:
    """Simulate all 13 strategies at a single decision minute.

    Returns list of dicts (one per strategy).
    """
    lot = LOT_SIZE.get(index_name, 50)
    step = STRIKE_STEP.get(index_name, 50)

    S0 = float(spot_lookup.loc[entry_t]) if entry_t in spot_lookup.index else float("nan")
    if not (S0 > 0):
        return []
    T0 = _T_years(entry_t, expiry)
    if T0 <= 0:
        return []
    if not (atm_iv > 0):
        return []

    rows = []
    for spec in STRATEGIES:
        # NO_TRADE row
        if spec.id == 0:
            rows.append({
                "strategy_id": 0, "strategy_name": "NO_TRADE", "horizon_label": "NA",
                "entry_premium_inr": 0.0, "exit_premium_inr": 0.0,
                "gross_pnl": 0.0, "total_cost": 0.0, "net_pnl": 0.0,
                "pnl_per_premium": 0.0,
                "synthetic_price_used": False, "valid": True,
            })
            continue

        # Compute exit time
        exit_t = _exit_time(entry_t, spec.horizon_minutes)
        if exit_t <= entry_t:
            rows.append({"strategy_id": spec.id, "strategy_name": spec.name,
                          "horizon_label": spec.horizon_label, "valid": False,
                          "entry_premium_inr": np.nan, "exit_premium_inr": np.nan,
                          "gross_pnl": np.nan, "total_cost": np.nan,
                          "net_pnl": np.nan, "pnl_per_premium": np.nan,
                          "synthetic_price_used": False})
            continue

        S_exit = float(spot_lookup.loc[exit_t]) if exit_t in spot_lookup.index else float("nan")
        if not (S_exit > 0):
            # Try last available bar before exit_t
            valid_idx = spot_lookup.index[spot_lookup.index <= exit_t]
            if len(valid_idx) == 0:
                rows.append({"strategy_id": spec.id, "strategy_name": spec.name,
                              "horizon_label": spec.horizon_label, "valid": False,
                              "entry_premium_inr": np.nan, "exit_premium_inr": np.nan,
                              "gross_pnl": np.nan, "total_cost": np.nan,
                              "net_pnl": np.nan, "pnl_per_premium": np.nan,
                              "synthetic_price_used": False})
                continue
            exit_t = valid_idx[-1]
            S_exit = float(spot_lookup.loc[exit_t])
        T_exit = max(_T_years(exit_t, expiry), 1e-6)

        # Resolve each leg's entry + exit price
        legs_premia = []
        synthetic_any = False
        entry_prem_total = 0.0  # signed: short = positive (received), long = negative (paid)
        exit_prem_total = 0.0   # signed: short = positive (need to pay back), long = positive (sell)
        valid_legs = True

        for leg in spec.legs:
            # Strike: offset from ATM, rounded to step
            target_K = _round_to_step(atm_strike * (1 + leg.strike_offset_pct), step)
            sign = +1 if leg.direction == "short" else -1

            # Entry price
            entry_px, syn_e = lookup_price(entry_t, target_K, leg.opt_type,
                                              chain_lookup, S0, target_K, T0, atm_iv, step)
            if entry_px <= 0:
                valid_legs = False; break

            # Exit price
            exit_px, syn_x = lookup_price(exit_t, target_K, leg.opt_type,
                                            chain_lookup, S_exit, target_K, T_exit, atm_iv, step)
            if exit_px <= 0:
                # for longs at expiry: payoff is intrinsic; allow 0
                if exit_t.date() == expiry.date() and abs(T_exit) < 1e-5:
                    if leg.opt_type == "CE":
                        exit_px = max(S_exit - target_K, 0.0)
                    else:
                        exit_px = max(target_K - S_exit, 0.0)
                    syn_x = True
                if exit_px <= 0:
                    valid_legs = False; break
            synthetic_any = synthetic_any or syn_e or syn_x

            # Accumulate premia
            entry_prem_total += sign * entry_px  # short adds, long subtracts
            exit_prem_total  += sign * exit_px
            legs_premia.append((entry_px, exit_px, leg.direction, leg.strike_offset_pct))

        if not valid_legs:
            rows.append({"strategy_id": spec.id, "strategy_name": spec.name,
                          "horizon_label": spec.horizon_label, "valid": False,
                          "entry_premium_inr": np.nan, "exit_premium_inr": np.nan,
                          "gross_pnl": np.nan, "total_cost": np.nan,
                          "net_pnl": np.nan, "pnl_per_premium": np.nan,
                          "synthetic_price_used": synthetic_any})
            continue

        # Gross PnL: short captures (entry - exit); long captures (exit - entry).
        # Combined sign-aware: pnl = (entry_prem_total - exit_prem_total) × lot
        # because shorts contribute (entry - exit) > 0 when premium decays, and
        # longs contribute (exit - entry) which is (-(entry - exit)).
        gross_per_share = entry_prem_total - exit_prem_total
        gross_pnl = gross_per_share * lot

        cost = compute_cost(legs_premia, lot)
        net_pnl = gross_pnl - cost

        # Entry premium (always positive in INR — total absolute value of premiums traded)
        abs_entry_premium_per_share = sum(abs(lp[0]) for lp in legs_premia)
        entry_premium_inr = abs_entry_premium_per_share * lot
        abs_exit_premium_per_share = sum(abs(lp[1]) for lp in legs_premia)
        exit_premium_inr = abs_exit_premium_per_share * lot

        ppp = net_pnl / entry_premium_inr if entry_premium_inr > 0 else 0.0

        rows.append({
            "strategy_id": spec.id, "strategy_name": spec.name,
            "horizon_label": spec.horizon_label,
            "entry_premium_inr": entry_premium_inr,
            "exit_premium_inr": exit_premium_inr,
            "gross_pnl": gross_pnl, "total_cost": cost,
            "net_pnl": net_pnl, "pnl_per_premium": ppp,
            "synthetic_price_used": synthetic_any, "valid": True,
        })
    return rows


# ─── Day-level orchestration ──────────────────────────────────────────────────


def simulate_day(options_day: pd.DataFrame, spot_day: pd.DataFrame,
                  index_name: str, decision_minutes: list[int] | None = None) -> pd.DataFrame:
    """Simulate all 13 strategies at every decision minute of the day.

    decision_minutes: list of "minute_of_day" (e.g. [555, 615, ...] for 09:15, 10:15 ...)
    If None, uses 09:30..14:55 every minute.
    """
    if options_day.empty or spot_day.empty:
        return pd.DataFrame()

    expiry = pd.to_datetime(options_day["expiry"].iloc[0])
    chain_lookup = build_chain_lookup(options_day)
    spot_lookup = spot_day.set_index("datetime")["spot"]

    if decision_minutes is None:
        # 09:30 to 14:55, every minute
        decision_minutes = list(range(9 * 60 + 30, 14 * 60 + 56))

    # Per-minute ATM_IV (use the average of ATM call+put IV from chain pivot — but
    # we don't have IV in the raw chain. Use the V4-A chain features parquet)
    # NOTE: we'll do this differently — see simulate_day_with_features.
    return pd.DataFrame()  # placeholder; the orchestrator script handles this


def simulate_day_with_features(chain_features_day: pd.DataFrame,
                                  options_day: pd.DataFrame,
                                  spot_day: pd.DataFrame,
                                  index_name: str,
                                  decision_minutes: list[int] | None = None) -> pd.DataFrame:
    """Like simulate_day but uses pre-computed atm_strike/atm_iv from V4-A features.

    chain_features_day: subset of V4-A chain_features for this (index, date)
    """
    if (chain_features_day.empty or options_day.empty or spot_day.empty):
        return pd.DataFrame()

    expiry = pd.to_datetime(options_day["expiry"].iloc[0])
    chain_lookup = build_chain_lookup(options_day)
    spot_lookup = spot_day.set_index("datetime")["spot"]

    cf = chain_features_day.set_index("datetime")[
        ["atm_strike", "atm_iv", "spot", "T_years"]
    ].copy()

    if decision_minutes is None:
        decision_minutes = list(range(9 * 60 + 30, 14 * 60 + 56))

    valid_times = [t for t in cf.index
                    if (t.hour * 60 + t.minute) in set(decision_minutes)]

    rows = []
    trade_date = pd.Timestamp(expiry).normalize() if False else \
                 pd.Timestamp(spot_day["datetime"].iloc[0]).normalize()
    dte = max((expiry.date() - trade_date.date()).days, 0)
    dte_bucket_v = _dte_bucket(dte)

    for entry_t in valid_times:
        try:
            atm_strike = int(cf.loc[entry_t, "atm_strike"])
            atm_iv = float(cf.loc[entry_t, "atm_iv"])
        except (KeyError, ValueError):
            continue
        if atm_iv <= 0:
            continue

        per_min = simulate_minute(entry_t, index_name, spot_lookup,
                                     atm_strike, atm_iv, expiry, chain_lookup)
        if not per_min:
            continue
        for r in per_min:
            r.update({
                "index": index_name,
                "datetime": entry_t,
                "trade_date": trade_date,
                "dte": dte,
                "dte_bucket": dte_bucket_v,
                "atm_strike": atm_strike,
                "atm_iv": atm_iv,
                "spot": float(spot_lookup.loc[entry_t]) if entry_t in spot_lookup.index else float("nan"),
                "minute_of_day": entry_t.hour * 60 + entry_t.minute,
                "hour_of_day": entry_t.hour,
            })
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
