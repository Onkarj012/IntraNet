"""OptiNet V4 chain feature engine.

Reads minute-level intraday option chain CSVs from data/option_data/ and
computes per-minute chain features (ATM IV, skew, PCR, max-OI, IV-RV spread,
synthetic forward, realized vol) for downstream V4 models.

Schema of input options CSV: date,time,symbol,open,high,low,close,oi,volume
Symbol: {NIFTY|BANKNIFTY}{DDMMM YY}{STRIKE}{CE|PE} e.g. NIFTY04JAN2418300CE
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time as dtime, date as date_cls
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

# RBI repo rate proxy averaged over 2020-2024
RISK_FREE_RATE = 0.065
TRADING_MINUTES_PER_DAY = 375
TRADING_DAYS_PER_YEAR = 252
SECONDS_PER_YEAR = 365.25 * 24 * 3600

SYMBOL_RE = re.compile(
    r"^(NIFTY|BANKNIFTY)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$"
)
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


@dataclass(frozen=True)
class ContractKey:
    index: str
    expiry: date_cls
    strike: int
    opt_type: str  # "CE" or "PE"


def parse_symbol(sym: str) -> ContractKey | None:
    m = SYMBOL_RE.match(sym)
    if not m:
        return None
    idx, dd, mmm, yy, strike, opt = m.groups()
    if mmm not in MONTH_MAP:
        return None
    year = 2000 + int(yy)
    try:
        expiry = date_cls(year, MONTH_MAP[mmm], int(dd))
    except ValueError:
        return None
    return ContractKey(index=idx, expiry=expiry, strike=int(strike), opt_type=opt)


# --- Black-Scholes pricing & IV inversion (vectorized) ---


def bs_price(S, K, T, r, sigma, opt_type: str) -> np.ndarray:
    """Black-Scholes price for arrays of S, K, T, sigma. opt_type 'CE' or 'PE'."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    out = np.full_like(S, np.nan, dtype=np.float64)
    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    if not np.any(valid):
        return out

    Sv, Kv, Tv, sv = S[valid], K[valid], T[valid], sigma[valid]
    sqrtT = np.sqrt(Tv)
    d1 = (np.log(Sv / Kv) + (r + 0.5 * sv * sv) * Tv) / (sv * sqrtT)
    d2 = d1 - sv * sqrtT
    if opt_type == "CE":
        price = Sv * norm.cdf(d1) - Kv * np.exp(-r * Tv) * norm.cdf(d2)
    else:
        price = Kv * np.exp(-r * Tv) * norm.cdf(-d2) - Sv * norm.cdf(-d1)
    out[valid] = price
    return out


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                opt_type: str, lo: float = 0.001, hi: float = 5.0,
                tol: float = 1e-4, max_iter: int = 60) -> float:
    """Bisection IV inversion. Returns NaN if not solvable."""
    if not (price > 0 and S > 0 and K > 0 and T > 0):
        return np.nan
    # Sanity bounds: option must be at least worth its intrinsic
    intrinsic = max(S - K, 0) if opt_type == "CE" else max(K - S, 0)
    if price < intrinsic - 1e-6:
        return np.nan
    # Try bounds
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(np.array([S]), np.array([K]),
                         np.array([T]), r, np.array([mid]), opt_type)[0]
        if np.isnan(p_mid):
            return np.nan
        if p_mid > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return mid
    return mid


def _iv_for_chain(prices, strikes, S, T, r, opt_type) -> np.ndarray:
    """Vectorized bisection: invert IV for an array of (price, strike) at fixed S, T."""
    prices = np.asarray(prices, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    n = len(prices)
    out = np.full(n, np.nan)

    # Validity mask
    intrinsic = np.maximum(S - strikes, 0) if opt_type == "CE" else np.maximum(strikes - S, 0)
    valid = (prices > 0) & (T > 0) & (prices >= intrinsic - 1e-6)
    if not np.any(valid):
        return out

    lo = np.full(n, 0.001)
    hi = np.full(n, 5.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        Tarr = np.full(n, T)
        p_mid = bs_price(np.full(n, S), strikes, Tarr, r, mid, opt_type)
        mask_high = (p_mid > prices) & valid
        mask_low = ~mask_high & valid
        hi = np.where(mask_high, mid, hi)
        lo = np.where(mask_low, mid, lo)
    out = np.where(valid, 0.5 * (lo + hi), np.nan)
    return out


# --- Loaders ---


def load_spot(path: Path, index_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.rename(columns={"close": "spot"})
    df["index"] = index_name
    return df[["index", "datetime", "spot"]].sort_values("datetime").reset_index(drop=True)


def load_options_day(path: Path, index_name: str) -> pd.DataFrame:
    """Load one day's intraday option chain. Adds parsed contract fields."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

    # Parse symbol → strike, opt_type, expiry
    sym_unique = df["symbol"].unique()
    parsed = {s: parse_symbol(s) for s in sym_unique}
    keep = {s: k for s, k in parsed.items() if k is not None and k.index == index_name}
    if not keep:
        return df.iloc[0:0].copy()
    df = df[df["symbol"].isin(keep)].copy()
    df["strike"] = df["symbol"].map(lambda s: keep[s].strike)
    df["opt_type"] = df["symbol"].map(lambda s: keep[s].opt_type)
    df["expiry"] = df["symbol"].map(lambda s: keep[s].expiry)
    df["index"] = index_name
    return df.sort_values(["datetime", "strike", "opt_type"]).reset_index(drop=True)


# --- Per-minute chain features ---


def _expiry_T(now: pd.Timestamp, expiry: date_cls) -> float:
    """Time-to-expiry in years assuming 15:30 IST close on expiry day."""
    expiry_dt = pd.Timestamp(datetime.combine(expiry, dtime(15, 30)))
    seconds = (expiry_dt - now).total_seconds()
    return max(seconds, 0) / SECONDS_PER_YEAR


def compute_minute_features(
    options_day: pd.DataFrame,
    spot_day: pd.DataFrame,
    index_name: str,
    skew_distance_pct: float = 0.03,
) -> pd.DataFrame:
    """Compute per-minute chain features for one trading day.

    Returns DataFrame with one row per minute and ~18 chain features.
    """
    if options_day.empty or spot_day.empty:
        return pd.DataFrame()

    # Single expiry per file (verified empirically)
    expiry = options_day["expiry"].iloc[0]
    spot_lookup = spot_day.set_index("datetime")["spot"]

    out_rows: list[dict] = []
    for ts, gminute in options_day.groupby("datetime"):
        if ts not in spot_lookup.index:
            continue
        S = float(spot_lookup.loc[ts])
        if not (S > 0):
            continue
        T = _expiry_T(ts, expiry)
        if T <= 0:
            continue

        ce = gminute[gminute["opt_type"] == "CE"]
        pe = gminute[gminute["opt_type"] == "PE"]
        if ce.empty or pe.empty:
            continue

        # ATM strike = nearest available strike to spot
        all_strikes = np.sort(np.union1d(ce["strike"].values, pe["strike"].values))
        atm_strike = int(all_strikes[np.argmin(np.abs(all_strikes - S))])
        atm_call_row = ce[ce["strike"] == atm_strike]
        atm_put_row = pe[pe["strike"] == atm_strike]
        if atm_call_row.empty or atm_put_row.empty:
            continue
        atm_call_px = float(atm_call_row["close"].iloc[0])
        atm_put_px = float(atm_put_row["close"].iloc[0])

        atm_call_iv = implied_vol(atm_call_px, S, atm_strike, T, RISK_FREE_RATE, "CE")
        atm_put_iv = implied_vol(atm_put_px, S, atm_strike, T, RISK_FREE_RATE, "PE")
        atm_iv = float(np.nanmean([atm_call_iv, atm_put_iv]))

        # Skew: IV at ~3% OTM put vs ~3% OTM call
        otm_put_target = S * (1 - skew_distance_pct)
        otm_call_target = S * (1 + skew_distance_pct)
        otm_put_strike = int(all_strikes[np.argmin(np.abs(all_strikes - otm_put_target))])
        otm_call_strike = int(all_strikes[np.argmin(np.abs(all_strikes - otm_call_target))])
        otm_put_row = pe[pe["strike"] == otm_put_strike]
        otm_call_row = ce[ce["strike"] == otm_call_strike]
        otm_put_iv = (
            implied_vol(float(otm_put_row["close"].iloc[0]), S, otm_put_strike,
                        T, RISK_FREE_RATE, "PE") if not otm_put_row.empty else np.nan
        )
        otm_call_iv = (
            implied_vol(float(otm_call_row["close"].iloc[0]), S, otm_call_strike,
                        T, RISK_FREE_RATE, "CE") if not otm_call_row.empty else np.nan
        )
        skew_slope = (otm_put_iv - otm_call_iv) / atm_iv if atm_iv and atm_iv > 0 else np.nan

        # PCR
        total_call_oi = float(ce["oi"].sum())
        total_put_oi = float(pe["oi"].sum())
        total_call_vol = float(ce["volume"].sum())
        total_put_vol = float(pe["volume"].sum())
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan

        # Max-OI strikes & distances
        ce_oi_g = ce.groupby("strike")["oi"].sum()
        pe_oi_g = pe.groupby("strike")["oi"].sum()
        max_oi_call_strike = int(ce_oi_g.idxmax()) if not ce_oi_g.empty else atm_strike
        max_oi_put_strike = int(pe_oi_g.idxmax()) if not pe_oi_g.empty else atm_strike
        # Max pain proxy: sum of total open interest by strike
        joint_oi = ce_oi_g.add(pe_oi_g, fill_value=0)
        max_oi_total_strike = int(joint_oi.idxmax()) if not joint_oi.empty else atm_strike
        max_call_dist = (max_oi_call_strike - S) / S
        max_put_dist = (max_oi_put_strike - S) / S
        max_total_dist = (max_oi_total_strike - S) / S

        # Synthetic forward via call-put parity
        synthetic_forward = atm_call_px - atm_put_px + atm_strike * np.exp(-RISK_FREE_RATE * T)
        forward_basis = synthetic_forward / S - 1.0

        # Chain breadth
        chain_breadth = int(((ce["volume"] > 0) | (pe["volume"] > 0)).sum())

        out_rows.append({
            "index": index_name,
            "datetime": ts,
            "spot": S,
            "expiry": expiry,
            "T_years": T,
            "atm_strike": atm_strike,
            "atm_call_premium": atm_call_px,
            "atm_put_premium": atm_put_px,
            "atm_straddle_premium": atm_call_px + atm_put_px,
            "atm_call_iv": atm_call_iv,
            "atm_put_iv": atm_put_iv,
            "atm_iv": atm_iv,
            "otm_put_iv": otm_put_iv,
            "otm_call_iv": otm_call_iv,
            "skew_slope": skew_slope,
            "pcr_oi": pcr_oi,
            "pcr_vol": pcr_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "total_oi": total_call_oi + total_put_oi,
            "total_call_vol": total_call_vol,
            "total_put_vol": total_put_vol,
            "total_vol": total_call_vol + total_put_vol,
            "max_oi_call_strike": max_oi_call_strike,
            "max_oi_put_strike": max_oi_put_strike,
            "max_oi_total_strike": max_oi_total_strike,
            "max_oi_call_dist_pct": max_call_dist,
            "max_oi_put_dist_pct": max_put_dist,
            "max_oi_total_dist_pct": max_total_dist,
            "synthetic_forward": synthetic_forward,
            "forward_basis": forward_basis,
            "chain_breadth": chain_breadth,
        })

    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    # Realized vol from spot returns over 30-min trailing window
    df = df.sort_values("datetime").reset_index(drop=True)
    spot_log_ret = np.log(df["spot"] / df["spot"].shift(1))
    rolling_std = spot_log_ret.rolling(30, min_periods=10).std()
    df["realized_vol_30m"] = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY)
    df["iv_rv_spread"] = df["atm_iv"] - df["realized_vol_30m"]
    return df


# --- Day discovery & file paths ---


def discover_days(root: Path, index_name: str) -> list[tuple[date_cls, Path, Path]]:
    """Return list of (date, options_path, spot_path) for one index."""
    if index_name == "NIFTY":
        opt_root = root / "nifty_data/nifty_options"
        spot_root = root / "nifty_data/nifty_spot"
        opt_prefix = "nifty_options_"
        spot_prefix = "nifty_spot"
    elif index_name == "BANKNIFTY":
        opt_root = root / "banknifty_data/banknifty_options"
        spot_root = root / "banknifty_data/banknifty_spot"
        opt_prefix = "banknifty_options_"
        spot_prefix = "banknifty_spot"
    else:
        raise ValueError(f"Unknown index: {index_name}")

    out: list[tuple[date_cls, Path, Path]] = []
    for opt_file in sorted(opt_root.rglob(f"{opt_prefix}*.csv")):
        # nifty_options_DD_MM_YYYY.csv  → date
        stem = opt_file.stem.replace(opt_prefix, "")
        try:
            dd, mm, yyyy = stem.split("_")
            d = date_cls(int(yyyy), int(mm), int(dd))
        except (ValueError, AttributeError):
            continue
        # Locate matching spot file: nifty_spotDD_MM_YYYY.csv (no underscore between prefix and DD!)
        spot_file = spot_root / str(d.year) / str(d.month) / f"{spot_prefix}{dd}_{mm}_{yyyy}.csv"
        if not spot_file.exists():
            continue
        out.append((d, opt_file, spot_file))
    return out


def build_day(opt_path: Path, spot_path: Path, index_name: str) -> pd.DataFrame:
    options = load_options_day(opt_path, index_name)
    spot = load_spot(spot_path, index_name)
    if options.empty or spot.empty:
        return pd.DataFrame()
    return compute_minute_features(options, spot, index_name)
