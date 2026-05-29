"""Futures-native feature builder for the OptiNet Router.

Produces a minute-aligned feature table from raw NIFTY futures + spot CSVs.
All features are causal (no future data leakage).

Feature groups:
  A. Returns & momentum   — 1/5/15/30/60-min returns, log-returns
  B. Volatility           — ATR, realized vol (5/15/30-min rolling)
  C. VWAP                 — VWAP deviation, VWAP slope
  D. Opening range        — distance from OR high/low, OR breakout flag
  E. OI patterns          — OI change 1/5/30m, price×OI interaction
  F. Futures basis        — basis, basis change
  G. Session context      — minute_of_day, hour_of_day, session_progress
  H. Gap                  — overnight gap from prior close
  I. Trend persistence    — consecutive up/down bars, EMA slope
  J. Volume               — volume/OI ratio, volume z-score
"""
from __future__ import annotations

import re
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "option_data"
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
MINUTES_PER_SESSION = 375
EPS = 1e-8


def _load_one_day(fut_path: Path, spot_path: Path) -> pd.DataFrame:
    """Load one day's futures + spot into a merged OHLCV frame."""
    fut = pd.read_csv(fut_path)
    fut.columns = [c.strip().lower() for c in fut.columns]
    fut["datetime"] = pd.to_datetime(fut["date"] + " " + fut["time"])
    fut = fut.rename(columns={"open": "f_open", "high": "f_high",
                                "low": "f_low", "close": "f_close",
                                "oi": "f_oi", "volume": "f_vol"})
    fut = fut[["datetime", "f_open", "f_high", "f_low", "f_close",
                "f_oi", "f_vol"]].sort_values("datetime").reset_index(drop=True)

    spot = pd.read_csv(spot_path)
    spot.columns = [c.strip().lower() for c in spot.columns]
    spot["datetime"] = pd.to_datetime(spot["date"] + " " + spot["time"])
    spot = spot.rename(columns={"close": "s_close"})
    spot = spot[["datetime", "s_close"]].sort_values("datetime").reset_index(drop=True)

    df = fut.merge(spot, on="datetime", how="left")
    df["s_close"] = df["s_close"].ffill()
    return df


def compute_features(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Compute all features for one trading day. Returns one row per minute."""
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    # ── A. Returns & momentum ──────────────────────────────────────────────
    for lag in (1, 5, 15, 30, 60):
        df[f"ret_{lag}m"] = df["f_close"].pct_change(lag)
        df[f"log_ret_{lag}m"] = np.log(df["f_close"] / df["f_close"].shift(lag))

    # ── B. Volatility ──────────────────────────────────────────────────────
    # True range (using futures OHLC)
    df["prev_close"] = df["f_close"].shift(1)
    df["tr"] = np.maximum(
        df["f_high"] - df["f_low"],
        np.maximum(
            (df["f_high"] - df["prev_close"]).abs(),
            (df["f_low"] - df["prev_close"]).abs(),
        )
    )
    for w in (5, 15, 30):
        df[f"atr_{w}m"] = df["tr"].rolling(w, min_periods=3).mean()
        df[f"realized_vol_{w}m"] = (
            df["log_ret_1m"].rolling(w, min_periods=3).std()
            * np.sqrt(MINUTES_PER_SESSION * 252)
        )

    # ── C. VWAP ────────────────────────────────────────────────────────────
    df["cum_vol"] = df["f_vol"].cumsum()
    df["cum_pv"] = (df["f_close"] * df["f_vol"]).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"].replace(0, np.nan)
    df["vwap_dev"] = (df["f_close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    df["vwap_slope_5m"] = df["vwap"].diff(5) / df["vwap"].shift(5).replace(0, np.nan)

    # ── D. Opening range (first 15 minutes: 09:15-09:29) ──────────────────
    or_mask = df["datetime"].dt.time < dtime(9, 30)
    or_high = df.loc[or_mask, "f_high"].max() if or_mask.any() else df["f_high"].iloc[0]
    or_low = df.loc[or_mask, "f_low"].min() if or_mask.any() else df["f_low"].iloc[0]
    or_range = max(or_high - or_low, EPS)
    df["or_high"] = or_high
    df["or_low"] = or_low
    df["or_dist_high"] = (df["f_close"] - or_high) / or_range
    df["or_dist_low"] = (df["f_close"] - or_low) / or_range
    df["or_breakout_up"] = (df["f_close"] > or_high).astype(np.int8)
    df["or_breakout_dn"] = (df["f_close"] < or_low).astype(np.int8)

    # ── E. OI patterns ────────────────────────────────────────────────────
    for lag in (1, 5, 30):
        df[f"oi_chg_{lag}m"] = df["f_oi"].diff(lag) / df["f_oi"].shift(lag).replace(0, np.nan)
    price_up_5 = df["f_close"].diff(5) > 0
    oi_up_5 = df["f_oi"].diff(5) > 0
    df["oi_long_buildup"] = (price_up_5 & oi_up_5).astype(np.int8)
    df["oi_short_buildup"] = (~price_up_5 & oi_up_5).astype(np.int8)
    df["oi_short_cover"] = (price_up_5 & ~oi_up_5).astype(np.int8)
    df["oi_long_unwind"] = (~price_up_5 & ~oi_up_5).astype(np.int8)

    # ── F. Futures basis ──────────────────────────────────────────────────
    df["basis"] = (df["f_close"] - df["s_close"]) / df["s_close"].replace(0, np.nan)
    df["basis_chg_30m"] = df["basis"].diff(30)

    # ── G. Session context ────────────────────────────────────────────────
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["hour_of_day"] = df["datetime"].dt.hour
    session_start = 9 * 60 + 15
    session_end = 15 * 60 + 30
    df["session_progress"] = (
        (df["minute_of_day"] - session_start) / (session_end - session_start)
    ).clip(0, 1)
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Mon

    # ── H. Gap ────────────────────────────────────────────────────────────
    # Gap = (today's open - yesterday's close) / yesterday's close
    # We only know this at the first bar; broadcast to all bars of the day
    first_open = float(df["f_open"].iloc[0])
    prev_close = float(df["prev_close"].iloc[0]) if not np.isnan(df["prev_close"].iloc[0]) else first_open
    df["gap_pct"] = (first_open - prev_close) / max(abs(prev_close), EPS)

    # ── I. Trend persistence ──────────────────────────────────────────────
    df["up_bar"] = (df["f_close"] > df["f_close"].shift(1)).astype(np.int8)
    # Consecutive up/down bars (reset on direction change)
    consec = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if df["up_bar"].iat[i] == 1:
            consec[i] = max(consec[i-1], 0) + 1
        else:
            consec[i] = min(consec[i-1], 0) - 1
    df["consec_bars"] = consec

    # EMA slope (fast 9-bar vs slow 21-bar)
    df["ema9"] = df["f_close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["f_close"].ewm(span=21, adjust=False).mean()
    df["ema_slope"] = (df["ema9"] - df["ema21"]) / df["ema21"].replace(0, np.nan)

    # ── J. Volume ─────────────────────────────────────────────────────────
    df["vol_oi_ratio"] = df["f_vol"] / df["f_oi"].replace(0, np.nan)
    vol_mean = df["f_vol"].rolling(30, min_periods=5).mean()
    vol_std = df["f_vol"].rolling(30, min_periods=5).std()
    df["vol_zscore"] = (df["f_vol"] - vol_mean) / vol_std.replace(0, np.nan)

    df["trade_date"] = pd.Timestamp(trade_date)
    return df


# Feature column list (used by model trainers)
FUTURES_FEATURES = [
    # Returns
    "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m",
    "log_ret_1m", "log_ret_5m", "log_ret_15m", "log_ret_30m",
    # Volatility
    "atr_5m", "atr_15m", "atr_30m",
    "realized_vol_5m", "realized_vol_15m", "realized_vol_30m",
    # VWAP
    "vwap_dev", "vwap_slope_5m",
    # Opening range
    "or_dist_high", "or_dist_low", "or_breakout_up", "or_breakout_dn",
    # OI
    "oi_chg_1m", "oi_chg_5m", "oi_chg_30m",
    "oi_long_buildup", "oi_short_buildup", "oi_short_cover", "oi_long_unwind",
    # Basis
    "basis", "basis_chg_30m",
    # Session
    "minute_of_day", "hour_of_day", "session_progress", "day_of_week",
    # Gap
    "gap_pct",
    # Trend
    "consec_bars", "ema_slope",
    # Volume
    "vol_oi_ratio", "vol_zscore",
]

# Optional chain features to include if available (kept only if they help)
CHAIN_FEATURES_OPTIONAL = [
    "atm_iv", "skew_slope", "pcr_oi", "realized_vol_30m_chain",
    "iv_rv_spread", "atm_straddle_premium",
]


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'regime' column using rule-based labels on futures features."""
    rv = df["realized_vol_30m"]
    rv_p25 = rv.quantile(0.25)
    rv_p75 = rv.quantile(0.75)
    rv_p90 = rv.quantile(0.90)
    trend_up = (df["ema_slope"] > 0.001) & (rv < rv_p75) & (df["ret_30m"] > 0)
    trend_dn = (df["ema_slope"] < -0.001) & (rv < rv_p75) & (df["ret_30m"] < 0)
    expansion = rv > rv_p90
    compression = (rv < rv_p25) & (df["ret_30m"].abs() < 0.001)
    regime = pd.Series("range", index=df.index)
    regime[compression] = "compression"
    regime[expansion] = "expansion"
    regime[trend_dn] = "trend_dn"
    regime[trend_up] = "trend_up"
    df = df.copy()
    df["regime"] = regime
    return df


def discover_days(symbol: str = "NIFTY") -> list[tuple[date, Path, Path]]:
    """Return (date, fut_path, spot_path) for all available days."""
    sym = symbol.lower()
    fut_root = DATA_ROOT / f"{sym}_data" / f"{sym}_fut"
    spot_root = DATA_ROOT / f"{sym}_data" / f"{sym}_spot"
    out = []
    for f in sorted(fut_root.rglob(f"{sym}_fut_*.csv")):
        m = re.match(rf"{sym}_fut_(\d\d)_(\d\d)_(\d\d\d\d)\.csv", f.name)
        if not m:
            continue
        d = date(int(m[3]), int(m[2]), int(m[1]))
        # spot files: nifty_spotDD_MM_YYYY.csv (no underscore between 'spot' and date)
        spot = (spot_root / str(d.year) / str(d.month)
                / f"{sym}_spot{d.day:02d}_{d.month:02d}_{d.year}.csv")
        if spot.exists():
            out.append((d, f, spot))
    return out


def build_all(symbol: str = "NIFTY",
               out_path: Path | None = None) -> pd.DataFrame:
    """Build the full feature table for all available days."""
    days = discover_days(symbol)
    print(f"Building features for {len(days)} {symbol} days …")
    frames = []
    for i, (d, fp, sp) in enumerate(days):
        try:
            raw = _load_one_day(fp, sp)
            feats = compute_features(raw, d)
            if not feats.empty:
                frames.append(feats)
        except Exception as exc:
            print(f"  skip {d}: {exc}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(days)} days done")
    full = pd.concat(frames, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["datetime"])
    full["trade_date"] = pd.to_datetime(full["trade_date"])
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(out_path, index=False)
        print(f"Saved {len(full):,} rows → {out_path}")
    return full
