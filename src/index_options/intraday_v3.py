"""OptiNet v3 Phase 1: build intraday decision-point dataset.

For each trading day x index x decision_time, we compute:
  - Features: per-bar TA on prior 60 min, session context, time-of-day
  - Labels: spot return over +1H and EOD horizons (continuous + binary classifications)

Output: cache/optinet_v3/intraday_dataset.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DECISION_TIMES = ["10:00", "11:15", "12:30", "13:45", "14:30"]
LONG_THRESHOLD_1H = 0.003   # +0.3% over 1 hour
LONG_THRESHOLD_EOD = 0.005  # +0.5% to EOD
SHORT_THRESHOLD_1H = -0.003
SHORT_THRESHOLD_EOD = -0.005

INDEX_FILES = {
    "NIFTY": "data/nifty_intraday/NIFTY 50_minute.csv",
    "BANKNIFTY": "data/nifty_intraday/NIFTY BANK_minute.csv",
    "FINNIFTY": "data/nifty_intraday/NIFTY FIN SERVICE_minute.csv",
    "MIDCPNIFTY": "data/nifty_intraday/NIFTY MIDCAP 50_minute.csv",
}

# Sector indices used for breadth feature (fraction positive at decision time)
SECTOR_INDEX_FILES = {
    "AUTO": "data/nifty_intraday/NIFTY AUTO_minute.csv",
    "BANK": "data/nifty_intraday/NIFTY BANK_minute.csv",
    "ENERGY": "data/nifty_intraday/NIFTY ENERGY_minute.csv",
    "FMCG": "data/nifty_intraday/NIFTY FMCG_minute.csv",
    "IT": "data/nifty_intraday/NIFTY IT_minute.csv",
    "METAL": "data/nifty_intraday/NIFTY METAL_minute.csv",
    "PHARMA": "data/nifty_intraday/NIFTY PHARMA_minute.csv",
    "REALTY": "data/nifty_intraday/NIFTY REALTY_minute.csv",
    "CONSUMPTION": "data/nifty_intraday/NIFTY CONSUMPTION_minute.csv",
}


def _load_minute(path: str | Path) -> pd.DataFrame:
    """Load a minute-bar CSV. Handles both schemas:
       - data/nifty_intraday/*: date,open,high,low,close,volume
       - data/banknifty_intraday/*: Instrument,Date,Time,Open,High,Low,Close
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    if "date" in cols_lower and "time" in cols_lower:
        # Banknifty schema
        date_col = cols_lower["date"]
        time_col = cols_lower["time"]
        df["date"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            format="mixed", dayfirst=True, errors="coerce",
        )
        df = df.rename(columns={
            cols_lower.get("open", "Open"): "open",
            cols_lower.get("high", "High"): "high",
            cols_lower.get("low", "Low"): "low",
            cols_lower.get("close", "Close"): "close",
        })
        if "volume" not in df.columns:
            df["volume"] = 0.0
        keep = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
    else:
        # nifty_intraday schema: already has 'date' as combined datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "volume" not in df.columns:
            df["volume"] = 0.0

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["trade_date"] = df["date"].dt.normalize()
    df["minute"] = df["date"].dt.strftime("%H:%M")
    # Keep only regular session (09:15 – 15:30)
    df = df[(df["minute"] >= "09:15") & (df["minute"] <= "15:30")].reset_index(drop=True)
    return df


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window, min_periods=window // 2).mean()
    loss = (-diff.clip(upper=0)).rolling(window, min_periods=window // 2).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0) / 100.0


def compute_decision_features(day_df: pd.DataFrame, decision_time: str) -> dict | None:
    """Return one feature row for a (day, decision_time) cell, or None if data is incomplete."""
    decision_idx = day_df.index[day_df["minute"] == decision_time].tolist()
    if not decision_idx:
        return None
    cut = decision_idx[0]
    prior = day_df.iloc[: cut + 1]
    if len(prior) < 30:
        return None

    open_price = float(prior["open"].iloc[0])
    decision_close = float(prior["close"].iloc[-1])
    session_high = float(prior["high"].max())
    session_low = float(prior["low"].min())

    last_60 = prior.tail(60)
    last_30 = prior.tail(30)
    last_15 = prior.tail(15)

    rsi_60 = _rsi(prior["close"]).iloc[-1]
    ema20 = prior["close"].ewm(span=20, adjust=False, min_periods=10).mean()
    ema_dist = decision_close / float(ema20.iloc[-1]) - 1.0 if ema20.iloc[-1] > 0 else 0.0

    return {
        "decision_time": decision_time,
        "decision_close": decision_close,
        "session_open": open_price,
        "session_high": session_high,
        "session_low": session_low,
        "ret_from_open": decision_close / open_price - 1.0,
        "session_range_pct": (session_high - session_low) / open_price,
        "dist_to_session_high": (session_high - decision_close) / decision_close,
        "dist_to_session_low": (decision_close - session_low) / decision_close,
        "ret_last_60min": decision_close / float(last_60["close"].iloc[0]) - 1.0,
        "ret_last_30min": decision_close / float(last_30["close"].iloc[0]) - 1.0,
        "ret_last_15min": decision_close / float(last_15["close"].iloc[0]) - 1.0,
        "vol_last_60min": float(last_60["close"].pct_change().std()),
        "vol_last_30min": float(last_30["close"].pct_change().std()),
        "rsi_14": float(rsi_60),
        "ema20_distance": float(ema_dist),
        "vol_ratio_30_60": (
            float(last_30["close"].pct_change().std())
            / max(float(last_60["close"].pct_change().std()), 1e-9)
        ),
        "volume_share_60min": (
            float(last_60["volume"].sum()) / max(float(prior["volume"].sum()), 1.0)
        ),
        "minutes_into_session": int(cut),
        "candle_body_ratio": (
            (decision_close - open_price) / max(session_high - session_low, 1e-9)
        ),
        "close_in_session_range": (
            (decision_close - session_low) / max(session_high - session_low, 1e-9)
        ),
    }


def compute_labels(day_df: pd.DataFrame, decision_time: str) -> dict | None:
    """Return horizon labels for a decision point, or None if horizons aren't available."""
    decision_idx = day_df.index[day_df["minute"] == decision_time].tolist()
    if not decision_idx:
        return None
    cut = decision_idx[0]
    decision_close = float(day_df["close"].iloc[cut])

    horizon_min = 60
    end_1h_idx = cut + horizon_min
    if end_1h_idx >= len(day_df):
        end_1h_idx = len(day_df) - 1
    if end_1h_idx <= cut:
        ret_1h = np.nan
    else:
        end_1h_close = float(day_df["close"].iloc[end_1h_idx])
        ret_1h = end_1h_close / decision_close - 1.0

    eod_close = float(day_df["close"].iloc[-1])
    ret_eod = eod_close / decision_close - 1.0

    return {
        "ret_1h": ret_1h,
        "ret_eod": ret_eod,
        "label_long_1h": int(ret_1h >= LONG_THRESHOLD_1H) if not np.isnan(ret_1h) else 0,
        "label_short_1h": int(ret_1h <= SHORT_THRESHOLD_1H) if not np.isnan(ret_1h) else 0,
        "label_long_eod": int(ret_eod >= LONG_THRESHOLD_EOD),
        "label_short_eod": int(ret_eod <= SHORT_THRESHOLD_EOD),
    }


def _build_sector_breadth(decision_times: Iterable[str]) -> pd.DataFrame:
    """For each (trade_date, decision_time), compute the fraction of sector indices
    with positive ret_from_open at that decision time. Returns long-format rows
    keyed by trade_date + decision_time.
    """
    frames: list[pd.DataFrame] = []
    for sector, path in SECTOR_INDEX_FILES.items():
        if not Path(path).exists():
            continue
        sec = _load_minute(path)
        # First bar of each day = open price
        opens = sec.groupby("trade_date").first()["open"]
        for dt in decision_times:
            cells = sec[sec["minute"] == dt][["trade_date", "close"]].copy()
            cells = cells.merge(opens.rename("session_open"), on="trade_date")
            cells["ret_from_open"] = cells["close"] / cells["session_open"] - 1.0
            cells["sector"] = sector
            cells["decision_time"] = dt
            frames.append(cells[["trade_date", "decision_time", "sector", "ret_from_open"]])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "decision_time",
                                      "sector_breadth", "sector_avg_ret",
                                      "sector_dispersion"])

    long = pd.concat(frames, ignore_index=True)
    agg = (long.groupby(["trade_date", "decision_time"])
                .agg(
                    sector_breadth=("ret_from_open", lambda s: float((s > 0).mean())),
                    sector_avg_ret=("ret_from_open", "mean"),
                    sector_dispersion=("ret_from_open", "std"),
                )
                .reset_index())
    return agg


def _load_prior_day_bhavcopy_features() -> pd.DataFrame:
    """Aggregate bhavcopy parquet partitions into daily PCR, IV rank, OI buildup.
    Returns DataFrame keyed by (index, trade_date) where trade_date is the prior
    bhavcopy day (i.e. yesterday's signal informs today's intraday decision).
    """
    import glob
    rows: list[dict] = []
    for fpath in sorted(glob.glob("data/parquet/symbol=*/year=*/options_*.parquet")):
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue
        if df.empty:
            continue
        sym = df["symbol"].iloc[0]
        d = pd.Timestamp(df["date"].iloc[0]).normalize()
        # Nearest-expiry slice
        nearest_exp = df[df["expiry_date"] >= d]["expiry_date"].min()
        if pd.isna(nearest_exp):
            nearest = df
        else:
            nearest = df[df["expiry_date"] == nearest_exp]

        ce = nearest[nearest["option_type"] == "CE"]
        pe = nearest[nearest["option_type"] == "PE"]
        ce_oi = float(ce["open_interest"].sum())
        pe_oi = float(pe["open_interest"].sum())
        ce_vol = float(ce["volume"].sum())
        pe_vol = float(pe["volume"].sum())
        rows.append({
            "bhav_index": sym,
            "trade_date": d,
            "bhav_pcr_oi": pe_oi / max(ce_oi, 1.0),
            "bhav_pcr_vol": pe_vol / max(ce_vol, 1.0),
            "bhav_total_oi": ce_oi + pe_oi,
            "bhav_total_oi_change": float(nearest["change_in_oi"].sum()),
            "bhav_atm_iv_proxy": float(
                nearest.assign(diff=(nearest["strike_price"] - nearest["underlying_price"]).abs())
                       .nsmallest(4, "diff")["close"].mean()
            ) if "underlying_price" in nearest.columns and nearest["underlying_price"].notna().any() else float("nan"),
        })

    if not rows:
        return pd.DataFrame(columns=["bhav_index", "trade_date",
                                      "bhav_pcr_oi", "bhav_pcr_vol", "bhav_total_oi",
                                      "bhav_total_oi_change", "bhav_atm_iv_proxy"])

    bhav = pd.DataFrame(rows).sort_values(["bhav_index", "trade_date"]).reset_index(drop=True)
    # IV rank: 60-day rolling rank of atm_iv_proxy per index
    bhav["bhav_iv_rank_60d"] = (
        bhav.groupby("bhav_index", group_keys=False)["bhav_atm_iv_proxy"]
            .apply(lambda s: s.rolling(60, min_periods=20).rank(pct=True))
    )
    # PCR change: 1-day diff
    bhav["bhav_pcr_oi_change_1d"] = bhav.groupby("bhav_index")["bhav_pcr_oi"].diff()
    return bhav


def build_intraday_dataset(
    index_files: dict[str, str | Path] = INDEX_FILES,
    decision_times: Iterable[str] = DECISION_TIMES,
    include_sector_breadth: bool = True,
    include_prior_bhavcopy: bool = True,
) -> pd.DataFrame:
    rows = []
    cross_context: dict[tuple, dict] = {}

    # First pass: load both indices; we'll cross-reference later
    minute_data = {}
    for name, path in index_files.items():
        path = Path(path)
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        minute_data[name] = _load_minute(path)

    if not minute_data:
        return pd.DataFrame()

    for index_name, df in minute_data.items():
        for trade_date, day_df in df.groupby("trade_date"):
            day_df = day_df.reset_index(drop=True)
            for dt in decision_times:
                feats = compute_decision_features(day_df, dt)
                if feats is None:
                    continue
                lbls = compute_labels(day_df, dt)
                if lbls is None:
                    continue
                row = {
                    "index": index_name,
                    "trade_date": pd.Timestamp(trade_date).normalize(),
                    **feats,
                    **lbls,
                }
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["trade_date", "decision_time", "index"]).reset_index(drop=True)

    # Cross-index features: for each (decision_time, trade_date), each row gets the
    # mean ret_from_open of *other* indices at that time (a "market breadth" proxy).
    pivot = (out.pivot_table(index=["trade_date", "decision_time"],
                              columns="index",
                              values="ret_from_open",
                              aggfunc="first")
                 .reset_index())
    out["other_idx_ret_from_open"] = 0.0
    out["spread_vs_other"] = 0.0
    for i, row in out.iterrows():
        match = pivot[(pivot["trade_date"] == row["trade_date"])
                      & (pivot["decision_time"] == row["decision_time"])]
        if match.empty:
            continue
        match = match.iloc[0]
        others = [c for c in pivot.columns
                  if c not in ("trade_date", "decision_time") and c != row["index"]]
        vals = [float(match[c]) for c in others if not pd.isna(match[c])]
        if not vals:
            continue
        avg = float(np.mean(vals))
        out.at[i, "other_idx_ret_from_open"] = avg
        out.at[i, "spread_vs_other"] = float(row["ret_from_open"]) - avg

    # Sector breadth at decision time
    if include_sector_breadth:
        breadth = _build_sector_breadth(decision_times)
        if not breadth.empty:
            out = out.merge(breadth, on=["trade_date", "decision_time"], how="left")
            for c in ("sector_breadth", "sector_avg_ret", "sector_dispersion"):
                if c in out.columns:
                    out[c] = out[c].fillna(out[c].median() if out[c].notna().any() else 0.0)

    # Prior-day bhavcopy features (yesterday's PCR/IV/OI informs today's signal)
    if include_prior_bhavcopy:
        bhav = _load_prior_day_bhavcopy_features()
        if not bhav.empty:
            # Map our index codes to bhavcopy symbols (only NIFTY and BANKNIFTY have direct matches)
            sym_map = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY"}
            out["bhav_index"] = out["index"].map(sym_map)
            # Shift by one trading day so bhavcopy from t-1 informs decisions at t
            bhav = bhav.sort_values(["bhav_index", "trade_date"])
            bhav["join_date"] = bhav.groupby("bhav_index")["trade_date"].shift(-1)
            bhav = bhav.dropna(subset=["join_date"])
            bhav = bhav.rename(columns={"trade_date": "_bhav_actual_date", "join_date": "trade_date"})
            keep = ["bhav_index", "trade_date", "bhav_pcr_oi", "bhav_pcr_vol",
                    "bhav_total_oi_change", "bhav_iv_rank_60d", "bhav_pcr_oi_change_1d"]
            out = out.merge(bhav[keep], on=["bhav_index", "trade_date"], how="left")
            out = out.drop(columns=["bhav_index"])
            # Numeric fillna with 0 (means "no prior info available")
            for c in ("bhav_pcr_oi", "bhav_pcr_vol", "bhav_total_oi_change",
                       "bhav_iv_rank_60d", "bhav_pcr_oi_change_1d"):
                if c in out.columns:
                    out[c] = out[c].fillna(0.0)

    return out
