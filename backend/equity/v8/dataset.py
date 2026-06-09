"""Barrier-labeled daily dataset for V8 walk-forward backtesting.

One row per (symbol, trade_date): daily features + barrier outcome labels +
open/close needed to simulate the trade. All features are point-in-time
(daily features use prior data; the barrier label is the *outcome* used only
as the target / simulation result, never as an input feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .barriers import compute_barrier_targets
from .config import V8Config
from .daily_features import DailyFeatureBuilder
from .data_pipeline import extract_sessions, load_minute_data
from .universe_tiers import UniverseTierReport

META_COLS = [
    "symbol", "date", "industry", "tier",
    "open_price", "close_price", "long_label", "short_label", "excess_ret",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def build_barrier_dataset(
    symbols: list[str],
    data_dir: str,
    config: V8Config,
    tier_report: UniverseTierReport,
    *,
    feature_fn=None,
    feature_lag_days: int = 1,
    sentiment_df: pd.DataFrame | None = None,
    min_bars: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a barrier-labeled dataset.

    feature_fn(symbol, industry, sessions, minute_df) -> DataFrame indexed by date.
    Defaults to the V8 DailyFeatureBuilder. feature_lag_days shifts features
    forward so an at-open decision on day D only uses data through D-lag
    (V8 features need lag=1; V7 open-safe features are already lagged → lag=0).
    """
    tgt = config.target
    if feature_fn is None:
        builder = DailyFeatureBuilder(
            market_data_dir="market_data_cache",
            sentiment_df=sentiment_df if sentiment_df is not None and not sentiment_df.empty else None,
        )

        def feature_fn(symbol, industry, sessions, minute_df):
            return builder.build_for_stock(sessions, symbol, industry=industry)

    rows: list[dict] = []
    for i, symbol in enumerate(symbols):
        minute_df = load_minute_data(symbol, data_dir, min_bars=min_bars)
        if minute_df.empty:
            continue
        sessions = extract_sessions(minute_df, min_bars=min_bars)
        if not sessions:
            continue
        assignment = tier_report.assignments.get(symbol)
        industry = assignment.industry if assignment else ""
        tier = assignment.tier.value if assignment else "unknown"
        try:
            feats = feature_fn(symbol, industry, sessions, minute_df)
        except Exception as e:
            if verbose:
                print(f"  feature build failed {symbol}: {e}")
            continue
        if feats is None or feats.empty:
            continue
        feats = feats.sort_index().shift(feature_lag_days)
        for date, sess in sessions.items():
            date_ts = pd.Timestamp(date).normalize()
            if date_ts not in feats.index:
                continue
            feat_row = feats.loc[date_ts]
            if feat_row.isna().all():
                continue
            bt = compute_barrier_targets(
                sess, symbol, target_pct=tgt.target_pct, stop_pct=tgt.stop_pct, min_bars=min_bars,
            )
            if bt is None:
                continue
            row = feat_row.to_dict()
            row.update(
                symbol=symbol, date=date_ts, industry=industry, tier=tier,
                open_price=bt.open_price, close_price=bt.close_price,
                long_label=bt.long_label, short_label=bt.short_label,
            )
            rows.append(row)
        if verbose and (i + 1) % 25 == 0:
            print(f"  dataset: {i + 1}/{len(symbols)} symbols, {len(rows)} rows")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    feat_cols = feature_columns(df)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


def build_barrier_dataset_v7(
    symbols: list[str],
    data_dir: str,
    config: V8Config,
    tier_report: UniverseTierReport,
    *,
    sentiment_csv: str = "data/sentiment/combined_sentiment_2015_2025.csv",
    min_bars: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Same harness, but using V7's open-safe daily feature set (already lagged)."""
    import os

    from equity.features.market_features import MarketFeatureBuilder
    from equity.features.sentiment_features import SentimentFeatureBuilder
    from equity.open_safe_daily_features import build_open_safe_daily_features

    mb = MarketFeatureBuilder()
    sp = sentiment_csv if os.path.exists(sentiment_csv) else "/nonexistent.csv"
    sb = SentimentFeatureBuilder(sp, market_builder=mb)
    try:
        sb._load()
    except Exception:
        pass

    def feat_fn(symbol, industry, sessions, minute_df):
        return build_open_safe_daily_features(minute_df, symbol, mb, sb)

    return build_barrier_dataset(
        symbols, data_dir, config, tier_report,
        feature_fn=feat_fn, feature_lag_days=0, min_bars=min_bars, verbose=verbose,
    )


def _opening_features(w: pd.DataFrame, prev_close: float) -> dict:
    """Self-contained features from the opening window (known at the decision bar)."""
    o = float(w["open"].iloc[0])
    c = float(w["close"].iloc[-1])
    hi = float(w["high"].max())
    lo = float(w["low"].min())
    rng = max(hi - lo, 1e-9)
    vol = float(w["volume"].sum()) if "volume" in w.columns else 0.0
    vwap = (
        float((w["close"] * w["volume"]).sum() / max(w["volume"].sum(), 1e-9))
        if "volume" in w.columns and w["volume"].sum() > 0 else c
    )
    half = max(len(w) // 2, 1)
    first_half_ret = (float(w["close"].iloc[half - 1]) - o) / max(o, 1e-9)
    second_half_ret = (c - float(w["close"].iloc[half - 1])) / max(float(w["close"].iloc[half - 1]), 1e-9)
    return {
        "or_return": (c - o) / max(o, 1e-9),
        "or_range_pct": rng / max(o, 1e-9),
        "or_pos": (c - lo) / rng,
        "vwap_dev": (c - vwap) / max(vwap, 1e-9),
        "gap_pct": (o - prev_close) / prev_close if prev_close > 0 else 0.0,
        "up_bar_frac": float((w["close"].diff() > 0).mean()),
        "accel": second_half_ret - first_half_ret,
        "open_vol": vol,
    }


def build_postopen_dataset(
    symbols: list[str],
    data_dir: str,
    config: V8Config,
    tier_report: UniverseTierReport,
    *,
    cutoff_min: int = 30,
    min_post_bars: int = 60,
    rel_vol_lookback: int = 20,
    min_bars: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Post-open variant: decide after the first `cutoff_min` minutes, enter at that
    bar, and run the barrier over the remainder of the session (exit at close)."""
    tgt = config.target
    rows: list[dict] = []
    for i, symbol in enumerate(symbols):
        minute_df = load_minute_data(symbol, data_dir, min_bars=min_bars)
        if minute_df.empty:
            continue
        sessions = extract_sessions(minute_df, min_bars=min_bars)
        if not sessions:
            continue
        assignment = tier_report.assignments.get(symbol)
        industry = assignment.industry if assignment else ""
        tier = assignment.tier.value if assignment else "unknown"

        prev_close = 0.0
        recent_open_vol: list[float] = []
        for date in sorted(sessions.keys()):
            sess = sessions[date]
            if len(sess) < cutoff_min + min_post_bars:
                prev_close = float(sess["close"].iloc[-1])
                continue
            w = sess.iloc[:cutoff_min]
            post = sess.iloc[cutoff_min:]
            feats = _opening_features(w, prev_close)
            avg_ov = float(np.mean(recent_open_vol)) if recent_open_vol else 0.0
            feats["rel_vol"] = feats["open_vol"] / avg_ov if avg_ov > 0 else 1.0
            del feats["open_vol"]

            bt = compute_barrier_targets(
                post, symbol, target_pct=tgt.target_pct, stop_pct=tgt.stop_pct,
                min_bars=min_post_bars,
            )
            recent_open_vol.append(
                float(w["volume"].sum()) if "volume" in w.columns else 0.0
            )
            recent_open_vol = recent_open_vol[-rel_vol_lookback:]
            prev_close = float(sess["close"].iloc[-1])
            if bt is None:
                continue
            row = dict(feats)
            row.update(
                symbol=symbol, date=pd.Timestamp(date).normalize(), industry=industry, tier=tier,
                open_price=bt.open_price, close_price=bt.close_price,
                long_label=bt.long_label, short_label=bt.short_label,
            )
            rows.append(row)
        if verbose and (i + 1) % 25 == 0:
            print(f"  postopen dataset: {i + 1}/{len(symbols)} symbols, {len(rows)} rows")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    feat_cols = feature_columns(df)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


def build_swing_dataset_v7(
    symbols: list[str],
    data_dir: str,
    config: V8Config,
    tier_report: UniverseTierReport,
    *,
    horizon_days: int = 10,
    sentiment_csv: str = "data/sentiment/combined_sentiment_2015_2025.csv",
    min_bars: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Swing variant: V7 open-safe daily features predict the N-day forward return.

    Entry at close of decision day t (open_price=close_t), exit at close_{t+N}
    (close_price). Cost is paid once per round trip; the long-short sim then
    realises the N-day return. Target = forward return sign.
    """
    import os

    from equity.features.market_features import MarketFeatureBuilder
    from equity.features.sentiment_features import SentimentFeatureBuilder
    from equity.open_safe_daily_features import build_open_safe_daily_features

    mb = MarketFeatureBuilder()
    sp = sentiment_csv if os.path.exists(sentiment_csv) else "/nonexistent.csv"
    sb = SentimentFeatureBuilder(sp, market_builder=mb)
    try:
        sb._load()
    except Exception:
        pass

    rows: list[dict] = []
    for i, symbol in enumerate(symbols):
        minute_df = load_minute_data(symbol, data_dir, min_bars=min_bars)
        if minute_df.empty:
            continue
        try:
            feats = build_open_safe_daily_features(minute_df, symbol, mb, sb)
        except Exception as e:
            if verbose:
                print(f"  feature build failed {symbol}: {e}")
            continue
        if feats is None or feats.empty:
            continue
        close = minute_df["close"].resample("D").last().dropna()
        fwd = close.shift(-horizon_days) / close - 1.0
        exit_px = close.shift(-horizon_days)

        assignment = tier_report.assignments.get(symbol)
        industry = assignment.industry if assignment else ""
        tier = assignment.tier.value if assignment else "unknown"

        for date_ts in feats.index:
            if date_ts not in close.index or date_ts not in fwd.index:
                continue
            f = fwd.loc[date_ts]
            entry = close.loc[date_ts]
            ex = exit_px.loc[date_ts]
            if pd.isna(f) or pd.isna(entry) or pd.isna(ex) or entry <= 0:
                continue
            row = feats.loc[date_ts].to_dict()
            row.update(
                symbol=symbol, date=date_ts, industry=industry, tier=tier,
                open_price=float(entry), close_price=float(ex),
                long_label=int(f > 0), short_label=int(f < 0),
            )
            rows.append(row)
        if verbose and (i + 1) % 25 == 0:
            print(f"  swing dataset: {i + 1}/{len(symbols)} symbols, {len(rows)} rows")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    feat_cols = feature_columns(df)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


def build_swing_factor_dataset(
    symbols: list[str],
    data_dir: str,
    config: V8Config,
    tier_report: UniverseTierReport,
    *,
    horizon_days: int = 10,
    min_bars: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Swing with horizon-appropriate features: classic cross-sectional equity
    factors (multi-month momentum, short-term reversal, volatility, trend,
    52w position) computed as-of close_t to predict the N-day forward return.
    Point-in-time: all features use data through close_t; entry at close_t.
    """
    rows: list[dict] = []
    for i, symbol in enumerate(symbols):
        minute_df = load_minute_data(symbol, data_dir, min_bars=min_bars)
        if minute_df.empty:
            continue
        d = minute_df.resample("D").agg(
            c=("close", "last"), v=("volume", "sum")
        ).dropna()
        if len(d) < 260 + horizon_days:
            continue
        c = d["c"]
        ret1 = c.pct_change()
        f = pd.DataFrame(index=c.index)
        f["mom_21"] = c / c.shift(21) - 1
        f["mom_63"] = c / c.shift(63) - 1
        f["mom_126"] = c / c.shift(126) - 1
        f["mom_252_21"] = c.shift(21) / c.shift(252) - 1   # 12-1 momentum
        f["rev_5"] = -(c / c.shift(5) - 1)                 # short-term reversal
        f["vol_21"] = ret1.rolling(21, min_periods=10).std()
        f["vol_63"] = ret1.rolling(63, min_periods=21).std()
        f["dist_252_high"] = c / c.rolling(252, min_periods=60).max() - 1
        f["dist_252_low"] = c / c.rolling(252, min_periods=60).min() - 1
        f["sma_50_ratio"] = c / c.rolling(50, min_periods=20).mean() - 1
        f["sma_200_ratio"] = c / c.rolling(200, min_periods=60).mean() - 1
        f["vol_ratio"] = d["v"] / d["v"].rolling(21, min_periods=10).mean()

        fwd = c.shift(-horizon_days) / c - 1.0
        exit_px = c.shift(-horizon_days)

        assignment = tier_report.assignments.get(symbol)
        industry = assignment.industry if assignment else ""
        tier = assignment.tier.value if assignment else "unknown"

        for date_ts in f.index:
            fr = fwd.loc[date_ts]
            entry = c.loc[date_ts]
            ex = exit_px.loc[date_ts]
            if pd.isna(fr) or pd.isna(entry) or pd.isna(ex) or entry <= 0:
                continue
            if f.loc[date_ts].isna().all():
                continue
            row = f.loc[date_ts].to_dict()
            row.update(
                symbol=symbol, date=date_ts, industry=industry, tier=tier,
                open_price=float(entry), close_price=float(ex),
                long_label=int(fr > 0), short_label=int(fr < 0),
            )
            rows.append(row)
        if verbose and (i + 1) % 25 == 0:
            print(f"  swing-factor dataset: {i + 1}/{len(symbols)} symbols, {len(rows)} rows")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    feat_cols = feature_columns(df)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


_XSECT_FACTORS = [
    "mom_21", "mom_63", "mom_126", "mom_252_21", "rev_5",
    "vol_21", "vol_63", "dist_252_high", "dist_252_low",
    "sma_50_ratio", "sma_200_ratio", "vol_ratio",
]


def build_xsect_factor_dataset(
    symbols: list[str],
    data_dir: str,
    config: V8Config,
    tier_report: UniverseTierReport,
    *,
    horizon_days: int = 21,
    min_bars: int = 200,
    min_names_per_day: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """Decisive cross-sectional test: factors z-scored within each day, target =
    out/under-performing the day's MEDIAN forward return (market direction removed),
    PnL measured as market-demeaned excess return (beta-neutral alpha)."""
    raw: list[dict] = []
    for i, symbol in enumerate(symbols):
        minute_df = load_minute_data(symbol, data_dir, min_bars=min_bars)
        if minute_df.empty:
            continue
        d = minute_df.resample("D").agg(c=("close", "last"), v=("volume", "sum")).dropna()
        if len(d) < 260 + horizon_days:
            continue
        c = d["c"]
        ret1 = c.pct_change()
        f = pd.DataFrame(index=c.index)
        f["mom_21"] = c / c.shift(21) - 1
        f["mom_63"] = c / c.shift(63) - 1
        f["mom_126"] = c / c.shift(126) - 1
        f["mom_252_21"] = c.shift(21) / c.shift(252) - 1
        f["rev_5"] = -(c / c.shift(5) - 1)
        f["vol_21"] = ret1.rolling(21, min_periods=10).std()
        f["vol_63"] = ret1.rolling(63, min_periods=21).std()
        f["dist_252_high"] = c / c.rolling(252, min_periods=60).max() - 1
        f["dist_252_low"] = c / c.rolling(252, min_periods=60).min() - 1
        f["sma_50_ratio"] = c / c.rolling(50, min_periods=20).mean() - 1
        f["sma_200_ratio"] = c / c.rolling(200, min_periods=60).mean() - 1
        f["vol_ratio"] = d["v"] / d["v"].rolling(21, min_periods=10).mean()
        fwd = c.shift(-horizon_days) / c - 1.0
        exit_px = c.shift(-horizon_days)

        assignment = tier_report.assignments.get(symbol)
        industry = assignment.industry if assignment else ""
        tier = assignment.tier.value if assignment else "unknown"
        for date_ts in f.index:
            fr = fwd.loc[date_ts]
            entry = c.loc[date_ts]
            ex = exit_px.loc[date_ts]
            if pd.isna(fr) or pd.isna(entry) or pd.isna(ex) or entry <= 0:
                continue
            if f.loc[date_ts].isna().all():
                continue
            row = f.loc[date_ts].to_dict()
            row.update(symbol=symbol, date=date_ts, industry=industry, tier=tier,
                       open_price=float(entry), close_price=float(ex), fwd_ret=float(fr))
            raw.append(row)
        if verbose and (i + 1) % 50 == 0:
            print(f"  xsect dataset: {i + 1}/{len(symbols)} symbols, {len(raw)} rows")

    df = pd.DataFrame(raw)
    if df.empty:
        return df
    # keep only days with enough names for a cross-section
    counts = df.groupby("date")["symbol"].transform("size")
    df = df[counts >= min_names_per_day].copy()
    if df.empty:
        return df
    g = df.groupby("date")
    for col in _XSECT_FACTORS:
        mu = g[col].transform("mean")
        sd = g[col].transform("std")
        df[col] = ((df[col] - mu) / sd.replace(0, np.nan)).fillna(0.0)
    mkt = g["fwd_ret"].transform("mean")
    med = g["fwd_ret"].transform("median")
    df["excess_ret"] = df["fwd_ret"] - mkt
    df["long_label"] = (df["fwd_ret"] > med).astype(int)
    df["short_label"] = (df["fwd_ret"] < med).astype(int)
    df = df.drop(columns=["fwd_ret"])
    feat_cols = feature_columns(df)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)
