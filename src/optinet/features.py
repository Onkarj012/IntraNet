from __future__ import annotations

import numpy as np
import pandas as pd

from optinet.math import implied_volatility


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window, min_periods=window // 2).mean()
    loss = (-diff.clip(upper=0)).rolling(window, min_periods=window // 2).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0


def build_index_features(index_bars: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index_name, daily in index_bars.sort_values(["index", "date"]).groupby("index"):
        daily = daily.set_index("date").sort_index()
        close = daily["close"]
        open_ = daily["open"]
        high = daily["high"]
        low = daily["low"]
        ret = close.pct_change()
        ema20 = close.ewm(span=20, adjust=False, min_periods=5).mean()
        mid = close.rolling(20, min_periods=5).mean()
        std = close.rolling(20, min_periods=5).std()
        body = (close - open_).abs()
        range_ = (high - low).replace(0, np.nan)

        feat = pd.DataFrame(index=daily.index)
        feat["return_1d"] = ret
        feat["return_3d"] = close.pct_change(3)
        feat["return_5d"] = close.pct_change(5)
        feat["rsi_14"] = _rsi(close)
        feat["ema20_distance"] = close / ema20.replace(0, np.nan) - 1.0
        feat["bollinger_location"] = ((close - (mid - 2 * std)) / (4 * std.replace(0, np.nan))).clip(0, 1)
        feat["realized_vol_5d"] = ret.rolling(5, min_periods=3).std()
        feat["realized_vol_20d"] = ret.rolling(20, min_periods=5).std()
        feat["volatility_ratio_5_20"] = feat["realized_vol_5d"] / feat["realized_vol_20d"].replace(0, np.nan)
        feat["gap"] = open_ / close.shift(1).replace(0, np.nan) - 1.0
        feat["overnight_return"] = feat["gap"]
        feat["trend_regime"] = np.sign(feat["ema20_distance"]).fillna(0.0)
        feat["day_of_week"] = daily.index.dayofweek / 4.0
        feat["is_expiry_day"] = (daily.index.dayofweek == 3).astype(float)
        feat["range_ratio"] = (range_ / close.replace(0, np.nan)).clip(0, 0.2)
        feat["candle_body_ratio"] = (body / range_).clip(0, 1)
        feat["close_location"] = ((close - low) / range_).clip(0, 1)
        feat["volume_ratio_20d"] = daily["volume"] / daily["volume"].rolling(20, min_periods=5).mean().replace(0, np.nan)
        feat["high_breakout_20d"] = (close / high.rolling(20, min_periods=5).max().replace(0, np.nan) - 1.0)
        feat["low_breakdown_20d"] = (close / low.rolling(20, min_periods=5).min().replace(0, np.nan) - 1.0)
        feat["index"] = index_name
        feat["date"] = feat.index
        frames.append(feat.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _nearest_expiry(chain: pd.DataFrame) -> pd.Timestamp | None:
    valid = chain[chain["days_to_expiry"] >= 0]
    if valid.empty:
        return None
    return pd.Timestamp(valid["expiry"].min())


def _max_pain(chain: pd.DataFrame) -> float:
    strikes = np.sort(chain["strike"].dropna().unique())
    if len(strikes) == 0:
        return float("nan")
    calls = chain[chain["option_type"] == "CE"].groupby("strike")["open_interest"].sum()
    puts = chain[chain["option_type"] == "PE"].groupby("strike")["open_interest"].sum()
    losses = []
    for settlement in strikes:
        call_loss = (np.maximum(settlement - calls.index.to_numpy(dtype=float), 0.0) * calls.to_numpy(dtype=float)).sum()
        put_loss = (np.maximum(puts.index.to_numpy(dtype=float) - settlement, 0.0) * puts.to_numpy(dtype=float)).sum()
        losses.append(call_loss + put_loss)
    return float(strikes[int(np.argmin(losses))])


def _top_oi_strike(chain: pd.DataFrame, option_type: str) -> float:
    typed = chain[chain["option_type"] == option_type]
    if typed.empty:
        return float("nan")
    by_strike = typed.groupby("strike")["open_interest"].sum()
    return float(by_strike.idxmax()) if not by_strike.empty else float("nan")


def _atm_iv(chain: pd.DataFrame, spot: float, option_type: str) -> float:
    typed = chain[chain["option_type"] == option_type].copy()
    if typed.empty or spot <= 0:
        return float("nan")
    row = typed.iloc[(typed["strike"] - spot).abs().argsort().iloc[0]]
    t = max(float(row["days_to_expiry"]), 1.0) / 365.0
    return implied_volatility(float(row["close"]), spot, float(row["strike"]), t, option_type)


def build_fo_features(option_chain_with_spot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    chain_frame = option_chain_with_spot.copy()
    if "days_to_expiry" not in chain_frame.columns:
        chain_frame["days_to_expiry"] = (pd.to_datetime(chain_frame["expiry"]) - pd.to_datetime(chain_frame["date"])).dt.days
    grouped = chain_frame.sort_values(["index", "date"]).groupby(["index", "date"])
    for (index_name, date), chain in grouped:
        expiry = _nearest_expiry(chain)
        if expiry is None:
            continue
        near = chain[chain["expiry"] == expiry].copy()
        spot = float(near["spot"].dropna().iloc[0]) if near["spot"].notna().any() else float("nan")
        calls = near[near["option_type"] == "CE"]
        puts = near[near["option_type"] == "PE"]
        monthly_expiry = chain["expiry"].max()
        monthly = chain[chain["expiry"] == monthly_expiry]
        call_oi = float(calls["open_interest"].sum())
        put_oi = float(puts["open_interest"].sum())
        weekly_oi = call_oi + put_oi
        monthly_oi = float(monthly["open_interest"].sum())
        call_vol = float(calls["volume"].sum())
        put_vol = float(puts["volume"].sum())
        max_pain = _max_pain(near)
        call_top = _top_oi_strike(near, "CE")
        put_top = _top_oi_strike(near, "PE")
        call_iv = _atm_iv(near, spot, "CE")
        put_iv = _atm_iv(near, spot, "PE")
        top3_call = calls.groupby("strike")["open_interest"].sum().nlargest(3).sum()
        top3_put = puts.groupby("strike")["open_interest"].sum().nlargest(3).sum()
        total_oi_change = float(near["change_oi"].sum())

        rows.append(
            {
                "index": index_name,
                "date": pd.Timestamp(date),
                "spot": spot,
                "pcr_oi": put_oi / call_oi if call_oi else np.nan,
                "pcr_volume": put_vol / call_vol if call_vol else np.nan,
                "max_pain": max_pain,
                "max_pain_distance": (spot - max_pain) / spot if spot and not np.isnan(max_pain) else np.nan,
                "atm_iv_proxy_call": call_iv,
                "atm_iv_proxy_put": put_iv,
                "iv_skew": put_iv - call_iv if not np.isnan(put_iv) and not np.isnan(call_iv) else np.nan,
                "total_oi_change": total_oi_change,
                "call_oi_concentration": top3_call / call_oi if call_oi else np.nan,
                "put_oi_concentration": top3_put / put_oi if put_oi else np.nan,
                "call_max_oi_strike": call_top,
                "put_max_oi_strike": put_top,
                "call_wall_distance": (call_top - spot) / spot if spot and not np.isnan(call_top) else np.nan,
                "put_wall_distance": (spot - put_top) / spot if spot and not np.isnan(put_top) else np.nan,
                "expiry_distance": float((expiry - pd.Timestamp(date)).days),
                "is_expiry_week": float((expiry - pd.Timestamp(date)).days <= 5),
                "is_expiry_day": float((expiry - pd.Timestamp(date)).days == 0),
                "weekly_vs_monthly_oi_ratio": weekly_oi / monthly_oi if monthly_oi else np.nan,
                "nearest_expiry": expiry,
            }
        )
    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat
    feat = feat.sort_values(["index", "date"])
    avg_iv = feat[["atm_iv_proxy_call", "atm_iv_proxy_put"]].mean(axis=1)
    feat["iv_change_1d"] = avg_iv.groupby(feat["index"]).diff()
    feat["iv_rank_20d"] = avg_iv.groupby(feat["index"]).transform(
        lambda s: (s - s.rolling(20, min_periods=5).min())
        / (s.rolling(20, min_periods=5).max() - s.rolling(20, min_periods=5).min()).replace(0, np.nan)
    )
    feat["pcr_oi_change_1d"] = feat.groupby("index")["pcr_oi"].diff()
    feat["pcr_oi_5d_avg"] = feat.groupby("index")["pcr_oi"].transform(lambda s: s.rolling(5, min_periods=2).mean())

    # OptiNet v2.2: 60-day rolling-median ratios for level-sensitive features.
    # Absolute levels of PCR/OI/wall-distance drifted heavily after NSE expanded
    # weekly expiries in 2024Q1, so we add ratio-form versions that are regime-robust.
    def _rolling_median_ratio(group: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
        med = group.rolling(window, min_periods=min_periods).median().replace(0, np.nan)
        return group / med

    LEVEL_FEATURES = [
        "pcr_oi", "pcr_volume",
        "call_oi_concentration", "put_oi_concentration",
        "call_wall_distance", "put_wall_distance",
        "total_oi_change",
    ]
    for col in LEVEL_FEATURES:
        if col in feat.columns:
            feat[f"{col}_norm60"] = (
                feat.groupby("index", group_keys=False)[col]
                    .apply(_rolling_median_ratio)
                    .reset_index(level=0, drop=True)
            )

    spot_change = feat.groupby("index")["spot"].pct_change()
    oi_change = feat["total_oi_change"]
    feat["oi_buildup_signal"] = np.select(
        [
            (spot_change > 0) & (oi_change > 0),
            (spot_change < 0) & (oi_change > 0),
            (spot_change > 0) & (oi_change < 0),
            (spot_change < 0) & (oi_change < 0),
        ],
        [1.0, -1.0, 0.5, -0.5],
        default=0.0,
    )
    return feat.replace([np.inf, -np.inf], np.nan)


def build_macro_sentiment_features(
    dates_by_index: pd.DataFrame,
    *,
    market_builder=None,
    sentiment_builder=None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index_name, group in dates_by_index.groupby("index"):
        dates = pd.DatetimeIndex(pd.to_datetime(group["date"]).sort_values().unique())
        frame = pd.DataFrame({"index": index_name, "date": dates})
        if market_builder is not None:
            macro = market_builder.get_features(dates).reset_index(drop=True).add_prefix("macro_")
            frame = pd.concat([frame.reset_index(drop=True), macro], axis=1)
            india = market_builder.get_india_market_features(dates)
            for key, series in india.items():
                frame[f"macro_{key}"] = series.reset_index(drop=True)
        if sentiment_builder is not None:
            sentiment = sentiment_builder.get_features(index_name, dates).reset_index(drop=True).add_prefix("sentiment_")
            frame = pd.concat([frame.reset_index(drop=True), sentiment], axis=1)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_training_frame(
    index_bars: pd.DataFrame,
    option_chain_with_spot: pd.DataFrame,
    *,
    market_builder=None,
    sentiment_builder=None,
    include_sentiment_cache: bool = True,
    include_gdelt: bool = True,
    include_regime: bool = True,
) -> pd.DataFrame:
    index_features = build_index_features(index_bars)
    fo_features = build_fo_features(option_chain_with_spot)
    if index_features.empty:
        return pd.DataFrame()
    frame = index_features.merge(fo_features, on=["index", "date"], how="left")
    macro = build_macro_sentiment_features(frame[["index", "date"]], market_builder=market_builder, sentiment_builder=sentiment_builder)
    if not macro.empty:
        frame = frame.merge(macro, on=["index", "date"], how="left")

    # OptiNet v2.1: index sentiment cache (yfinance + RSS aggregated daily)
    if include_sentiment_cache:
        try:
            from optinet.sentiment import load_sentiment_cache
            sent = load_sentiment_cache()
            if not sent.empty:
                sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
                frame = frame.merge(sent, on=["index", "date"], how="left")
        except Exception:
            pass

    # OptiNet v2.1: GDELT macro/financial event volumes (one column per theme)
    if include_gdelt:
        try:
            from optinet.gdelt import load_gdelt_cache
            gdelt = load_gdelt_cache()
            if not gdelt.empty:
                gdelt["date"] = pd.to_datetime(gdelt["date"]).dt.normalize()
                frame = frame.merge(gdelt, on="date", how="left")
        except Exception:
            pass

    # OptiNet v2.1: regime features + hard-filter flag
    if include_regime:
        try:
            from optinet.regime import compute_regime
            regime = compute_regime(index_bars)
            if not regime.empty:
                regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()
                frame = frame.merge(regime, on=["index", "date"], how="left")
        except Exception:
            pass

    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame.groupby("index", group_keys=False)[numeric].ffill().fillna(0.0)
    return frame.sort_values(["index", "date"]).reset_index(drop=True)
