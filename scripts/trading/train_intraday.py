#!/usr/bin/env python
"""Train the intraday V8 5-specialist LightGBM ensemble.

Uses the daily OHLCV panel + macro cache + sentiment to build features, then
labels each stock-day with a H375 (full-session) barrier target:
  1 = hit +1.5% before -1.0% during the session
  0 = hit -1.0% stop first (or neither)

The barrier labels are computed from daily OHLC (high/low) as an approximation
when per-minute CSVs are unavailable: if high ≥ open*(1+target) → label=1,
elif low ≤ open*(1-stop) → label=0, else label=1 if close > open (held-to-close).

Usage:
    python scripts/trading/train_intraday.py
    python scripts/trading/train_intraday.py --universe nifty500 --start 2021-01-01
    python scripts/trading/train_intraday.py --min-date 2022-01-01 --target-pct 0.015
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

MODEL_DIR = _ROOT / "models/v8_intraday"
PANEL_PATH = _ROOT / "cache/v8/daily_panel_nifty500_adj.parquet"


# ---------------------------------------------------------------------------
# Barrier labeling from daily OHLC (no minute data required)
# ---------------------------------------------------------------------------

def _atr14_pct(o: pd.Series, h: pd.Series, lo: pd.Series, c: pd.Series) -> pd.Series:
    """14-day EWM ATR as % of open. Uses T-1 data (shifted 1 day)."""
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, min_periods=7).mean()
    return (atr / o.replace(0, np.nan)).fillna(0.02)  # fallback 2%


def label_barriers(ohlcv: dict[str, pd.DataFrame],
                   target_pct: float | None = None,
                   stop_pct: float | None = None,
                   k_t: float = 1.5, k_s: float = 0.5) -> pd.DataFrame:
    """
    Compute H375 barrier labels using volatility-scaled levels.

    If target_pct / stop_pct are given (fixed mode), they override k_t/k_s.
    Otherwise:
        target = open * (1 + k_t * ATR14%)   ← typically ~6% at median ATR
        stop   = open * (1 - k_s * ATR14%)   ← typically ~2% at median ATR
        R:R = 3:1, breakeven = 25%

    Both-barriers-hit (whipsaw): resolved by open→close direction as proxy
    (bullish day → target hit first, bearish → stop hit first).

    Also stores 'dir_label': sign(close - open) for dual-head training.
    """
    o = ohlcv["open"];  h = ohlcv["high"]
    lo = ohlcv["low"];  c = ohlcv["close"]
    syms = o.columns.intersection(h.columns).intersection(lo.columns).intersection(c.columns)

    rows = []
    for sym in syms:
        os_ = o[sym].dropna(); hs_ = h[sym].dropna()
        ls_ = lo[sym].dropna(); cs_ = c[sym].dropna()
        common = os_.index.intersection(hs_.index).intersection(ls_.index).intersection(cs_.index)
        os_, hs_, ls_, cs_ = os_[common], hs_[common], ls_[common], cs_[common]

        if fixed := (target_pct is not None and stop_pct is not None):
            t_lvl = os_ * (1 + target_pct)
            s_lvl = os_ * (1 - stop_pct)
        else:
            atr_pct = _atr14_pct(os_, hs_, ls_, cs_).shift(1).fillna(0.02)  # T-1 ATR
            t_lvl = os_ * (1 + k_t * atr_pct)
            s_lvl = os_ * (1 - k_s * atr_pct)

        target_hit = hs_ >= t_lvl
        stop_hit   = ls_ <= s_lvl
        bullish     = cs_ > os_  # used to resolve whipsaws

        # Whipsaw resolution: if both hit, use open→close direction as proxy
        long_label = pd.Series(np.where(
            target_hit & ~stop_hit, 1,
            np.where(stop_hit & ~target_hit, 0,
                     bullish.astype(int))   # both or neither → direction
        ), index=common)

        dir_label = bullish.astype(int)  # open→close direction (always clean)

        for date in common:
            oc = float((cs_[date] - os_[date]) / os_[date]) if os_[date] > 0 else 0.0
            atr_val = float(t_lvl[date] / os_[date] - 1) / 1.5 if os_[date] > 0 else 0.02
            rows.append({
                "symbol": sym, "date": date,
                "long_label": int(long_label[date]),
                "dir_label":  int(dir_label[date]),
                "oc_ret":     round(oc, 5),
                "mag_label":  round(np.clip(oc / max(atr_val, 0.001), -3, 3), 4),
                "open_price": float(os_[date]),
                "long_target": float(t_lvl[date]),
                "long_stop":   float(s_lvl[date]),
                "atr_pct": float(
                    (t_lvl[date] - os_[date]) / (os_[date] * k_t) if not fixed else target_pct
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build training dataset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vectorised feature builder — computes all features at once across all dates
# ---------------------------------------------------------------------------

def _build_all_features_vectorised(ohlcv: dict[str, pd.DataFrame],
                                   symbols: list[str],
                                   market_data_dir: str) -> pd.DataFrame:
    """
    Build the full price-action + macro feature matrix for every (date, symbol)
    in one vectorised pass. Returns a tall DataFrame with MultiIndex (date, symbol).
    Much faster than calling build() per date.
    """
    import warnings
    c_raw = ohlcv["close"].reindex(columns=symbols)
    h_raw = ohlcv.get("high", c_raw).reindex(columns=symbols)
    lo_raw = ohlcv.get("low", c_raw).reindex(columns=symbols)
    o_raw = ohlcv.get("open", c_raw).reindex(columns=symbols)
    v_raw = ohlcv.get("volume", pd.DataFrame(index=c_raw.index, columns=c_raw.columns, data=1e6)).reindex(columns=symbols)

    # ── TEMPORAL BOUNDARY ──────────────────────────────────────────────────
    # A pre-open signal on date T can only use data through close of T-1.
    # Shift c, h, lo, v by 1 so every indicator uses T-1 values.
    # open[T] is kept unshifted — it IS available pre-open (gap calculation).
    c  = c_raw.shift(1)   # close[T-1]
    h  = h_raw.shift(1)   # high[T-1]
    lo = lo_raw.shift(1)  # low[T-1]
    v  = v_raw.shift(1)   # volume[T-1]
    o  = o_raw            # open[T]  ← available pre-open
    # ──────────────────────────────────────────────────────────────────────

    feats: dict[str, pd.DataFrame] = {}

    # Returns
    for lag, name in [(1,"return_1d"),(5,"return_5d"),(10,"return_10d"),(21,"return_21d"),(63,"return_63d")]:
        feats[name] = c.pct_change(lag).clip(-1, 1)

    # Momentum
    for lag, name in [(5,"momentum_5d"),(10,"momentum_10d"),(21,"momentum_21d")]:
        feats[name] = c.pct_change(lag).clip(-1, 1)

    # SMA distances
    for window, name in [(21,"close_vs_sma_21d"),(63,"close_vs_sma_63d")]:
        sma = c.rolling(window).mean()
        feats[name] = (c / sma.replace(0, np.nan) - 1).clip(-1, 1)

    # Overnight / gap
    prev_c = c.shift(1)
    feats["overnight_return"] = ((o - prev_c) / prev_c.replace(0, np.nan)).clip(-0.1, 0.1)
    feats["gap_size"] = feats["overnight_return"].abs()
    feats["gap_direction"] = np.sign(feats["overnight_return"])

    # ATR (14-day)
    prev_c2 = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c2).abs(), (lo - prev_c2).abs()], axis=1).groupby(level=0, axis=0).max() \
        if False else pd.DataFrame(
            np.maximum(np.maximum((h - lo).values, (h - prev_c2).abs().values), (lo - prev_c2).abs().values),
            index=c.index, columns=c.columns)
    atr = tr.ewm(span=14, min_periods=7).mean()
    feats["avg_true_range_14d"] = (atr / c.replace(0, np.nan)).clip(0, 0.5)
    feats["atr_percentile"] = atr.rank(pct=True)

    # Volatility
    hl_log = np.log(h / lo.replace(0, np.nan))
    for window, sfx in [(5,"5d"),(21,"21d")]:
        feats[f"parkinson_vol_{sfx}"] = hl_log.rolling(window).std().clip(0, 0.5)
        co = np.log(c / o.replace(0, np.nan))
        feats[f"gk_vol_{sfx}"] = (
            (0.5 * hl_log**2 - (2*np.log(2)-1) * co**2).clip(0).rolling(window).mean().apply(np.sqrt)
        ).clip(0, 0.5)

    # Price position
    for window, sfx in [(20,"20d"),(63,"63d")]:
        roll_h = h.rolling(window).max()
        roll_l = lo.rolling(window).min()
        rng = (roll_h - roll_l).replace(0, np.nan)
        feats[f"price_position_{sfx}"] = ((c - roll_l) / rng).clip(0, 1)

    # RSI-14
    delta = c.diff()
    ag = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    al = (-delta).where(delta < 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    feats["rsi_14d"] = (100 - 100 / (1 + ag / al.replace(0, 1e-10))) / 100.0

    # Bollinger
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std().replace(0, 1)
    feats["bollinger_position"] = ((c - bb_mid) / (2 * bb_std)).clip(-3, 3)

    # Volume
    vol_mean = v.rolling(21).mean().replace(0, 1)
    feats["rel_volume_20d"] = (v / vol_mean).clip(0, 10)
    feats["volume_trend_5d"] = v.pct_change(5).clip(-1, 1)
    feats["volume_dryup_ratio"] = (v.rolling(5).mean() / v.rolling(21).mean().replace(0, 1)).clip(0, 5)

    # Vol contraction
    feats["vol_contraction_5d"] = (atr.rolling(5).mean() / atr.rolling(21).mean().replace(0, 1)).clip(0, 5)
    feats["vol_contraction_21d"] = (atr.rolling(21).mean() / atr.rolling(63).mean().replace(0, 1)).clip(0, 5)

    # Range
    feats["high_low_range_pct"] = ((h - lo) / c.replace(0, np.nan)).clip(0, 0.5)
    feats["prev_day_range_pct"] = feats["high_low_range_pct"].shift(1)

    # Inside-day count (5d)
    inside = ((h <= h.shift(1)) & (lo >= lo.shift(1))).rolling(5).sum()
    feats["inside_day_count_5d"] = inside.clip(0, 5)
    feats["close_vs_narrow_range"] = ((h - lo) / h.rolling(7).max().replace(0, 1)).clip(0, 1)
    feats["close_vs_vwap"] = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    feats["afternoon_vs_morning"] = pd.DataFrame(0.0, index=c.index, columns=c.columns)

    # --- Macro features (one row per date, broadcast to all symbols) ---
    from equity.features.market_features import MarketFeatureBuilder
    from equity.intraday_features import _is_expiry_week
    mb = MarketFeatureBuilder(market_data_dir)
    mb._load_cached()
    macro_df = mb.get_features(c.index)  # (dates, macro_features)

    # Broadcast macro to symbols by stacking
    macro_broadcast = {}
    for col in macro_df.columns:
        macro_broadcast[col] = pd.DataFrame(
            np.tile(macro_df[col].values.reshape(-1, 1), (1, len(symbols))),
            index=c.index, columns=symbols)
    # Extra macro
    nifty = mb._get_close("nifty50")
    india_vix = mb._get_close("india_vix")
    for col_name, series, window in [
        ("nifty_vs_50dma", nifty, 50), ("nifty_vs_200dma", nifty, 200)]:
        if not series.empty and len(series) >= window:
            val = (series / series.rolling(window).mean() - 1).reindex(c.index, method="ffill").fillna(0)
        else:
            val = pd.Series(0.0, index=c.index)
        macro_broadcast[col_name] = pd.DataFrame(
            np.tile(val.values.reshape(-1,1), (1, len(symbols))), index=c.index, columns=symbols)
    if not india_vix.empty:
        vt = india_vix.pct_change(5).reindex(c.index, method="ffill").fillna(0)
        macro_broadcast["vix_trend_5d"] = pd.DataFrame(
            np.tile(vt.values.reshape(-1,1),(1,len(symbols))), index=c.index, columns=symbols)
    macro_broadcast["breadth_pct_above_20dma"] = pd.DataFrame(
        np.tile(macro_df.get("nifty_5d_return", pd.Series(0, index=c.index)).values.reshape(-1,1),
                (1, len(symbols))), index=c.index, columns=symbols)
    macro_broadcast["nifty_adx"] = pd.DataFrame(0.3, index=c.index, columns=symbols)
    macro_broadcast["sp500_overnight"] = macro_broadcast.get("sp500_overnight_return",
        pd.DataFrame(0.0, index=c.index, columns=symbols))
    macro_broadcast["crude_change"] = macro_broadcast.get("crude_oil_return",
        pd.DataFrame(0.0, index=c.index, columns=symbols))
    macro_broadcast["sector_return_1d"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["sector_return_5d"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["sector_index_prev_return"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["sector_index_5d_return"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["sector_index_volatility"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["industry_relative_strength_rank"] = pd.DataFrame(0.5, index=c.index, columns=symbols)
    macro_broadcast["sector_breadth_proxy"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["secondary_sector_confirmation"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["stock_vs_sector_1d"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["stock_vs_sector_5d"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["nifty_intraday_return"] = pd.DataFrame(0.0, index=c.index, columns=symbols)
    macro_broadcast["vix_change"] = macro_broadcast.get("vix_level",
        pd.DataFrame(0.15, index=c.index, columns=symbols))
    macro_broadcast["market_breadth"] = macro_broadcast.get("nifty_5d_return",
        pd.DataFrame(0.0, index=c.index, columns=symbols))
    macro_broadcast["global_cue"] = macro_broadcast.get("sp500_overnight_return",
        pd.DataFrame(0.0, index=c.index, columns=symbols))
    macro_broadcast["sector_intraday_return"] = pd.DataFrame(0.0, index=c.index, columns=symbols)

    # Calendar (broadcast)
    dow_series = pd.Series(c.index.dayofweek / 4.0, index=c.index)
    mon_series = pd.Series(c.index.month / 12.0, index=c.index)
    exp_series = pd.Series([float(_is_expiry_week(d)) for d in c.index], index=c.index)
    bud_series = pd.Series([(d.month == 2 and d.day == 1) * 1.0 for d in c.index], index=c.index)
    for name, series in [("day_of_week", dow_series), ("month", mon_series),
                         ("expiry_week", exp_series), ("budget_day", bud_series)]:
        macro_broadcast[name] = pd.DataFrame(
            np.tile(series.values.reshape(-1,1),(1,len(symbols))), index=c.index, columns=symbols)

    # Market cap category (broadcast per symbol, constant over time)
    from equity.universe import get_universe
    large = set(get_universe("nifty50")); mid = set(get_universe("nifty100")); mid2 = set(get_universe("nifty200"))
    mktcap = {s: (1.0 if s in large else 0.75 if s in mid else 0.5 if s in mid2 else 0.25) for s in symbols}
    macro_broadcast["market_cap_category"] = pd.DataFrame(
        {s: pd.Series(mktcap[s], index=c.index) for s in symbols})

    # Cross-sectional RS
    for lag, col in [(5,"rs_vs_sector_5d"),(21,"rs_vs_sector_21d")]:
        ret = c.pct_change(lag)
        feats[col] = ret.rank(axis=1, pct=True) - 0.5

    # ── Phase A: Opening-range proxy features (from daily OHLC, T-1) ──────────
    hl = (h - lo).replace(0, np.nan)
    feats["orb_first_ret"]    = ((c - o) / o.replace(0, np.nan)).clip(-0.2, 0.2)
    feats["orb_range_pct"]    = (hl / o.replace(0, np.nan)).clip(0, 0.3)
    feats["orb_close_loc"]    = ((c - lo) / hl).clip(0, 1)
    feats["orb_body_ratio"]   = ((c - o).abs() / hl).clip(0, 1)
    oc_max = o.where(o >= c, c)  # element-wise max(o, c) as DataFrame
    oc_min = o.where(o <= c, c)  # element-wise min(o, c) as DataFrame
    feats["orb_upper_shadow"] = ((h - oc_max) / hl).clip(0, 1)
    feats["orb_lower_shadow"] = ((oc_min - lo) / hl).clip(0, 1)
    # ──────────────────────────────────────────────────────────────────────────

    # Sentiment — load from historical CSV if available
    sentiment_csv = Path(__file__).resolve().parents[2] / "data/sentiment/combined_sentiment_latest.csv"
    sentiment_cols_zero = [
        "premarket_sentiment","premarket_sentiment_count","premarket_sentiment_max",
        "premarket_sentiment_std","sentiment_5d_avg","sentiment_momentum","sentiment_spike",
        "sentiment_price_div","news_volume_shock","sentiment_surprise","sentiment_macro_agreement",
        "sentiment_confidence","industry_premarket_sentiment","industry_premarket_sentiment_count",
        "industry_sentiment_5d_avg","industry_sentiment_momentum","industry_news_volume_shock",
        "industry_sentiment_surprise","industry_sentiment_stock_divergence",
    ]
    if sentiment_csv.exists():
        try:
            from equity.features.sentiment_features import SentimentFeatureBuilder
            from equity.features.market_features import MarketFeatureBuilder as _MB
            _mb = _MB(market_data_dir); _mb._load_cached()
            sfb = SentimentFeatureBuilder(str(sentiment_csv), market_builder=_mb,
                                          mode="premarket", market_open_time="09:15",
                                          universe_metadata_csv=str(
                                              Path(__file__).resolve().parents[2] /
                                              "data/sentiment/ind_nifty500list.csv"))
            print("  Building sentiment features (this may take a moment)...")
            for sym in symbols:
                try:
                    sf = sfb.get_features(sym, c_raw.index)  # use unshifted index
                    for col in [c for c in sentiment_cols_zero if c in sf.columns]:
                        macro_broadcast[col] = pd.DataFrame(
                            {sym: sf[col]}).reindex(index=c_raw.index, columns=symbols, fill_value=0.0)
                except Exception:
                    pass
            print("  Sentiment features loaded.")
        except Exception as e:
            print(f"  Sentiment load failed: {e} — using zeros")
            for col in sentiment_cols_zero:
                macro_broadcast[col] = pd.DataFrame(0.0, index=c_raw.index, columns=symbols)
    else:
        for col in sentiment_cols_zero:
            macro_broadcast[col] = pd.DataFrame(0.0, index=c_raw.index, columns=symbols)

    feats.update(macro_broadcast)

    # Stack to tall DataFrame (date, symbol) → features
    print("  Stacking feature panels...")
    tall_frames = []
    feat_names = list(feats.keys())
    arr = np.stack([feats[k].values for k in feat_names], axis=2)  # (T, S, F)
    T, S, F = arr.shape
    idx = pd.MultiIndex.from_product([c.index, symbols], names=["date", "symbol"])
    tall = pd.DataFrame(
        arr.reshape(T * S, F),
        index=idx,
        columns=feat_names,
    ).reset_index()
    tall = tall.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return tall


def build_dataset(symbols: list[str], ohlcv: dict[str, pd.DataFrame],
                  min_date: str, target_pct: float, stop_pct: float) -> pd.DataFrame:
    print("  Labeling barriers from OHLCV panel...")
    labels = label_barriers(ohlcv)   # vol-scaled: k_t=1.5×ATR, k_s=0.5×ATR
    labels = labels[labels["date"] >= pd.Timestamp(min_date)]
    print(f"  Labels: {len(labels)} stock-days, "
          f"LONG rate {labels['long_label'].mean():.2%}  "
          f"DIR rate {labels['dir_label'].mean():.2%}")

    print("  Building feature matrix (vectorised)...")
    market_data_dir = str(Path(__file__).resolve().parents[2] / "market_data_cache")
    tall = _build_all_features_vectorised(ohlcv, symbols, market_data_dir)

    # Merge with labels
    labels["date"] = pd.to_datetime(labels["date"])
    tall["date"] = pd.to_datetime(tall["date"])

    # oc_ret and mag_label are already columns in labels (from label_barriers)
    dataset = tall.merge(
        labels[["symbol", "date", "long_label", "dir_label",
                "open_price", "long_target", "long_stop", "atr_pct", "oc_ret", "mag_label"]],
        on=["symbol", "date"], how="inner"
    )
    # Drop rows before min_date
    dataset = dataset[dataset["date"] >= pd.Timestamp(min_date)].reset_index(drop=True)
    print(f"  Dataset: {len(dataset)} rows, {dataset['date'].min().date()} → {dataset['date'].max().date()}")
    return dataset


# ---------------------------------------------------------------------------
# Train the 5-specialist ensemble
# ---------------------------------------------------------------------------

def train_ensemble(dataset: pd.DataFrame, model_dir: Path,
                   target_col: str = "long_label") -> None:
    from equity.v8.signal_models import SignalModel, MetaEnsemble, FEATURE_GROUPS
    from equity.v8.config import SignalModelConfig, MetaEnsembleConfig
    from equity.v8.regime_detector import DEFAULT_REGIME_WEIGHTS
    from sklearn.metrics import roc_auc_score, log_loss
    from lightgbm import early_stopping as lgb_early_stopping

    model_dir.mkdir(parents=True, exist_ok=True)

    all_feature_cols = [c for c in dataset.columns
                        if c not in {"symbol", "date", "long_label", "dir_label",
                                     "open_price", "long_target", "long_stop", "atr_pct",
                                     "oc_ret", "mag_label"}]

    dates = dataset["date"].sort_values().unique()
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = dataset["date"] < split_date
    val_mask = ~train_mask

    X_train = dataset.loc[train_mask, all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_val   = dataset.loc[val_mask,   all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    specialist_names = ["momentum", "reversal", "breakout", "sentiment", "macro"]

    def _fit_head(label_col: str, suffix: str, regression: bool = False) -> tuple[dict, float]:
        from lightgbm import LGBMRegressor
        from sklearn.metrics import mean_squared_error

        y_train = dataset.loc[train_mask, label_col].values
        y_val   = dataset.loc[val_mask,   label_col].values
        print(f"\n  [{suffix}] Train: {len(X_train)} rows | "
              f"label mean train={y_train.mean():.4f} val={y_val.mean():.4f}")

        models: dict[str, object] = {}
        for name in specialist_names:
            print(f"  [{suffix}] Training specialist: {name}")
            if regression:
                m = LGBMRegressor(
                    n_estimators=1000, learning_rate=0.05,
                    num_leaves=31, min_child_samples=20,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    metric="rmse", verbosity=-1, random_state=42, n_jobs=-1,
                )
                from equity.v8.signal_models import FEATURE_GROUPS
                feats = [f for f in FEATURE_GROUPS.get("mag", []) if f in X_train.columns]
                if not feats:
                    feats = list(X_train.columns)
                Xtr = X_train[feats]; Xv = X_val[feats]
                m.fit(Xtr, y_train,
                      eval_set=[(Xv, y_val)],
                      callbacks=[lgb_early_stopping(150, verbose=False)])
                import pickle
                with open(model_dir / f"{name}_{suffix}.pkl", "wb") as fh:
                    pickle.dump({"name": name, "model": m, "feature_names": feats}, fh)
                models[name] = m
            else:
                cfg = SignalModelConfig(name=name, lgb_n_estimators=1000,
                                       lgb_early_stopping_rounds=150,
                                       lgb_min_child_samples=20, lgb_num_leaves=31,
                                       lgb_metric="auc")
                model = SignalModel(name=name, config=cfg)
                model.fit(X_train, y_train, X_val, y_val, verbose=True)
                model.calibrate(X_val, y_val, method="isotonic")
                model.save(model_dir / f"{name}_{suffix}.pkl")
                models[name] = model

        if regression:
            val_preds = np.zeros(len(X_val))
            for name in specialist_names:
                m = models[name]
                feats = [f for f in FEATURE_GROUPS.get("mag", []) if f in X_val.columns] or list(X_val.columns)
                val_preds += m.predict(X_val[feats])
            val_preds /= len(specialist_names)
            rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            print(f"\n  [{suffix}] Ensemble val RMSE: {rmse:.4f}")
            np.save(model_dir / f"regime_weights_{suffix}.npy", DEFAULT_REGIME_WEIGHTS)
            return models, rmse
        else:
            np.save(model_dir / f"regime_weights_{suffix}.npy", DEFAULT_REGIME_WEIGHTS)
            ensemble = MetaEnsemble(models=models, regime_weights=DEFAULT_REGIME_WEIGHTS)
            probs, _ = ensemble.predict(X_val)
            auc = roc_auc_score(y_val, probs)
            ll  = log_loss(y_val, np.clip(probs, 1e-7, 1-1e-7))
            print(f"\n  [{suffix}] Ensemble val AUC: {auc:.4f}  log-loss: {ll:.4f}")
            return models, auc

    # Head 1: vol-scaled barrier (target_hit before stop_hit)
    barrier_models, barrier_auc = _fit_head("long_label", "barrier")

    # Head 2: open→close direction (cleaner signal, higher base rate)
    dir_models, dir_auc = _fit_head("dir_label", "dir")

    # Head 3: magnitude regressor (oc_ret / atr_pct)
    mag_models, mag_rmse = _fit_head("mag_label", "mag", regression=True)

    # ── Meta-labeler ──────────────────────────────────────────────────────────
    # Uses the dir-head ensemble predictions on val set.
    # meta_label = 1 if (dir_pred > 0.44) matches (dir_label == 1)
    import pickle as _pickle
    import lightgbm as _lgb
    dir_ensemble = MetaEnsemble(models=dir_models, regime_weights=DEFAULT_REGIME_WEIGHTS)
    dir_val_probs, _ = dir_ensemble.predict(X_val)
    dir_val_labels = dataset.loc[val_mask, "dir_label"].values
    meta_label = ((dir_val_probs > 0.44) == (dir_val_labels == 1)).astype(int)
    print(f"\n  [meta] meta_label positive rate: {meta_label.mean():.2%}")
    meta_clf = _lgb.LGBMClassifier(
        n_estimators=500, num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        metric="auc", verbosity=-1, random_state=42, n_jobs=-1,
    )
    meta_clf.fit(
        X_val, meta_label,
        eval_set=[(X_val, meta_label)],
        callbacks=[lgb_early_stopping(50, verbose=False)],
    )
    with open(model_dir / "meta_model.pkl", "wb") as fh:
        _pickle.dump({"model": meta_clf, "feature_names": list(X_val.columns)}, fh)
    meta_val_probs = meta_clf.predict_proba(X_val)[:, 1]
    meta_auc = roc_auc_score(meta_label, meta_val_probs)
    print(f"  [meta] Meta-labeler val AUC: {meta_auc:.4f}")
    # ─────────────────────────────────────────────────────────────────────────

    # Save metadata
    meta = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "train_rows": int(len(X_train)),
        "val_rows":   int(len(X_val)),
        "train_cutoff": str(split_date.date()),
        "barrier_auc": round(barrier_auc, 4),
        "dir_auc":     round(dir_auc, 4),
        "mag_rmse":    round(float(mag_rmse), 4),
        "meta_auc":    round(meta_auc, 4),
        "val_auc":     round(barrier_auc, 4),   # compat
        "val_logloss": 0.0,
        "feature_count": len(all_feature_cols),
        "feature_names": all_feature_cols,
        "long_rate_train": round(float(dataset.loc[train_mask, "long_label"].mean()), 4),
        "long_rate_val":   round(float(dataset.loc[val_mask,   "long_label"].mean()), 4),
        "dir_rate_train":  round(float(dataset.loc[train_mask, "dir_label"].mean()), 4),
        "dir_rate_val":    round(float(dataset.loc[val_mask,   "dir_label"].mean()), 4),
        "specialists": specialist_names,
        "heads": ["barrier", "dir", "mag", "meta"],
        "k_t": 1.5, "k_s": 0.5,
    }
    (model_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  Models saved → {model_dir}")
    print(f"  barrier AUC={barrier_auc:.4f}  dir AUC={dir_auc:.4f}  "
          f"mag RMSE={mag_rmse:.4f}  meta AUC={meta_auc:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="nifty500")
    p.add_argument("--panel", default=str(PANEL_PATH))
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--min-date", default="2021-01-01",
                   help="Earliest date for training labels")
    p.add_argument("--target-pct", type=float, default=0.020,
                   help="Intraday target barrier (default 2.0%%)")
    p.add_argument("--stop-pct", type=float, default=0.010,
                   help="Intraday stop barrier (default 1.0%%)")
    args = p.parse_args()

    from equity.universe import get_universe
    from equity.momentum import load_ohlcv

    panel = Path(args.panel)
    if not panel.exists():
        print(f"Panel not found: {panel}")
        print("Run first: python scripts/data/equity_eod_panel.py --update")
        return 1

    print(f"\n  train_intraday  [{pd.Timestamp.now().isoformat(timespec='seconds')}]")
    print(f"  Universe: {args.universe}  |  min-date: {args.min_date}")
    print(f"  Barrier: +{args.target_pct*100:.1f}% / -{args.stop_pct*100:.1f}%")

    symbols = get_universe(args.universe)
    print(f"  Loading OHLCV panel ({len(symbols)} symbols)...")
    ohlcv = load_ohlcv(panel)

    # Check OHLCV completeness
    missing = [c for c in ["open", "high", "low"] if c not in ohlcv]
    if missing:
        print(f"  Panel missing columns: {missing}")
        print("  Re-run equity_eod_panel.py to rebuild with OHLCV.")
        return 1

    t0 = time.time()
    print("\nStep 1: Building dataset...")
    dataset = build_dataset(symbols, ohlcv, args.min_date, args.target_pct, args.stop_pct)
    if dataset.empty:
        print("Dataset is empty — cannot train. Check panel and feature builder.")
        return 1
    print(f"  Dataset: {len(dataset)} rows, elapsed {time.time()-t0:.0f}s")

    print("\nStep 2: Training ensemble...")
    train_ensemble(dataset, Path(args.model_dir))
    print(f"\nDone. Total elapsed: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
