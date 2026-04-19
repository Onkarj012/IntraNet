# IntradayNet — Complete LightGBM Overhaul Plan

---

## Phase 0: Audit & Fix Foundation (Week 1)

### 0.1 Lookahead Bias Audit

Go through every single feature and verify it uses **only past data**.

```python
# FILE: scripts/audit_features.py

"""
Run this script to flag every feature that MIGHT have lookahead bias.
Manual review required for each flagged feature.
"""

FEATURES_TO_AUDIT = {
    # PER-BAR FEATURES (25)
    "log_return": "✅ SAFE — uses current and previous bar only",
    "volume_ratio": "⚠️ CHECK — ratio to what? Session avg = LEAK, rolling avg = SAFE",
    "vwap_distance": "⚠️ CHECK — VWAP computed from session start to current bar? Or full day?",
    "ema_9_distance": "✅ SAFE — EMA is causal by definition",
    "ema_20_distance": "✅ SAFE — EMA is causal",
    "rsi_14": "✅ SAFE — uses past 14 bars",
    "bb_zscore": "✅ SAFE — rolling 20-bar window",
    "bb_width": "✅ SAFE — rolling 20-bar window",
    "body_ratio": "✅ SAFE — single bar OHLC",
    "upper_shadow_ratio": "✅ SAFE — single bar OHLC",
    "lower_shadow_ratio": "✅ SAFE — single bar OHLC",
    "spread_pct": "✅ SAFE — single bar high-low",
    "cum_volume_pct": "🚩 LIKELY LEAK — percentage of WHAT total? If full-day volume, this is future data",
    "time_normalized": "✅ SAFE — just clock time / total session",
    "orb_high_dist": "⚠️ CHECK — ORB defined from first 15 min only? Must not update after",
    "orb_low_dist": "⚠️ CHECK — same as above",
    "day_return": "🚩 CHECK — return from open to CURRENT bar, or open to CLOSE?",
    "momentum_5": "✅ SAFE — past 5 bars",
    "momentum_20": "✅ SAFE — past 20 bars",
    "vol_momentum": "✅ SAFE — if rolling",
    "atr_14": "✅ SAFE — past 14 bars",
    "close_vs_day_range": "🚩 LIKELY LEAK — day range uses full day high/low = future data",
    "session_volatility": "🚩 LIKELY LEAK — if computed from full session, this is future",
    "obv_slope": "✅ SAFE — if rolling OBV",
    "trade_intensity": "⚠️ CHECK — intensity relative to what baseline?",

    # SESSION FEATURES (20)
    "prev_day_rsi": "✅ SAFE — previous day",
    "prev_day_macd": "✅ SAFE — previous day",
    "prev_day_bollinger": "✅ SAFE — previous day",
    "prev_day_trend_strength": "✅ SAFE — previous day",
    "prev_day_regime": "✅ SAFE — previous day",
    "prev_day_volatility": "✅ SAFE — previous day",
    "prev_day_adx": "✅ SAFE — previous day",
    "overnight_return": "✅ SAFE — prev close to current open",
    "gap_size": "✅ SAFE — open vs prev close",
    "gap_direction": "✅ SAFE — derived from gap_size",
    "close_location": "⚠️ CHECK — close location of PREVIOUS day? SAFE. Current day? LEAK",
    "volume_zscore": "⚠️ CHECK — z-score relative to what window?",
    "day_of_week": "✅ SAFE",
    "is_expiry": "✅ SAFE — known in advance",
    "is_monthly_expiry": "✅ SAFE",
    "is_result_season": "✅ SAFE — approximate",
    "near_52w_high": "✅ SAFE — uses past data",
    "near_52w_low": "✅ SAFE — uses past data",
    "avg_intraday_range": "✅ SAFE — if computed from past N days",

    # SENTIMENT FEATURES (24)
    # All sentiment features should use T-1 or earlier data
    "premarket_sentiment_mean": "⚠️ CHECK — sentiment from BEFORE market open? Timestamp matters",
    "premarket_sentiment_max": "⚠️ CHECK — same",
    "vix_level": "✅ SAFE — previous close VIX",
    "vix_change": "✅ SAFE — previous day change",
    "nifty_intraday_return": "🚩 CHECK — if this is TODAY's NIFTY return, it's future data at 9:15 AM",
}

print("=" * 70)
print("LOOKAHEAD BIAS AUDIT")
print("=" * 70)

leaks = []
checks = []
for feature, status in FEATURES_TO_AUDIT.items():
    if "🚩" in status:
        leaks.append((feature, status))
    elif "⚠️" in status:
        checks.append((feature, status))

print(f"\n🚩 PROBABLE LEAKS ({len(leaks)}):")
for f, s in leaks:
    print(f"  {f}: {s}")

print(f"\n⚠️  NEEDS MANUAL CHECK ({len(checks)}):")
for f, s in checks:
    print(f"  {f}: {s}")

print(f"\nAction: Fix every 🚩 before ANY further training.")
```

### 0.2 Fix Leaked Features

```python
# FILE: src/intradaynet/features/per_bar_features.py (FIXES)

def compute_per_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features must be CAUSAL — only use data up to current bar."""

    # ❌ WRONG — uses full-day volume
    # df["cum_volume_pct"] = df["Volume"].cumsum() / df["Volume"].sum()

    # ✅ FIXED — cumulative volume as ratio to rolling average
    avg_session_volume = df.groupby("date")["Volume"].transform(
        lambda x: x.expanding().mean()
    )
    df["cum_volume_ratio"] = df["Volume"] / avg_session_volume.clip(lower=1)

    # ❌ WRONG — uses full-day high/low
    # df["close_vs_day_range"] = (
    #     (df["Close"] - df.groupby("date")["Low"].transform("min"))
    #     / (df.groupby("date")["High"].transform("max")
    #        - df.groupby("date")["Low"].transform("min"))
    # )

    # ✅ FIXED — uses EXPANDING high/low (only past + current)
    df["close_vs_running_range"] = df.groupby("date").apply(
        lambda g: (g["Close"] - g["Low"].expanding().min())
        / (g["High"].expanding().max() - g["Low"].expanding().min() + 1e-8)
    ).reset_index(level=0, drop=True)

    # ❌ WRONG — session volatility from full session
    # df["session_volatility"] = df.groupby("date")["log_return"].transform("std")

    # ✅ FIXED — expanding volatility (grows through the day)
    df["session_volatility"] = df.groupby("date")["log_return"].transform(
        lambda x: x.expanding(min_periods=5).std()
    )

    # ORB must be frozen after first 15 minutes
    def compute_orb(group):
        first_15 = group.head(15)  # first 15 bars = 15 minutes
        orb_high = first_15["High"].max()
        orb_low = first_15["Low"].min()
        group["orb_high"] = orb_high
        group["orb_low"] = orb_low
        # Before ORB is established, use NaN
        group.loc[group.index[:14], "orb_high"] = np.nan
        group.loc[group.index[:14], "orb_low"] = np.nan
        return group

    df = df.groupby("date", group_keys=False).apply(compute_orb)
    df["orb_high_dist"] = (df["Close"] - df["orb_high"]) / df["orb_high"]
    df["orb_low_dist"] = (df["Close"] - df["orb_low"]) / df["orb_low"]

    return df
```

### 0.3 Convert CSVs to Parquet

```python
# FILE: scripts/convert_to_parquet.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm

csv_dir = Path("nifty500/")
parquet_dir = Path("nifty500_parquet/")
parquet_dir.mkdir(exist_ok=True)

csv_files = sorted(csv_dir.glob("*.csv"))
print(f"Converting {len(csv_files)} CSV files to Parquet...")

for csv_file in tqdm(csv_files):
    df = pd.read_csv(csv_file, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Optimize dtypes before saving
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    parquet_path = parquet_dir / csv_file.with_suffix(".parquet").name
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

total_csv = sum(f.stat().st_size for f in csv_files) / 1e9
total_parquet = sum(
    f.stat().st_size for f in parquet_dir.glob("*.parquet")
) / 1e9
print(f"CSV total: {total_csv:.2f} GB → Parquet total: {total_parquet:.2f} GB")
print(f"Compression ratio: {total_csv / total_parquet:.1f}x")
```

---

## Phase 1: Data Pipeline Rebuild (Week 1-2)

### 1.1 Thresholded Target Construction

```python
# FILE: src/intradaynet/targets.py

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class TargetConfig:
    horizons: dict  # horizon_name -> bars
    move_threshold: float = 0.003  # 0.3% minimum meaningful move
    magnitude_clip: float = 0.05   # clip extreme moves at ±5%
    cost_adjustment: float = 0.001 # 0.1% round-trip cost to subtract


HORIZONS = {
    "H15": 15,
    "H30": 30,
    "H60": 60,
    "H375": 375,
}


def compute_targets(
    df: pd.DataFrame,
    config: TargetConfig = TargetConfig(horizons=HORIZONS),
) -> pd.DataFrame:
    """
    Compute direction and magnitude targets for each horizon.

    Returns DataFrame with columns:
        dir_H15, dir_H30, dir_H60, dir_H375  (0/1 or NaN)
        mag_H15, mag_H30, mag_H60, mag_H375  (float)
        valid_H15, valid_H30, valid_H60, valid_H375  (bool)
    """
    targets = pd.DataFrame(index=df.index)

    for name, bars in config.horizons.items():
        # Raw future return
        future_close = df["Close"].shift(-bars)
        raw_return = (future_close / df["Close"]) - 1

        # Subtract transaction costs
        net_return = raw_return.abs() - config.cost_adjustment
        net_return = net_return.clip(lower=0) * np.sign(raw_return)

        # Direction: only train on CLEAR signals
        # Bars with |return| < threshold are excluded (set to NaN)
        direction = pd.Series(np.nan, index=df.index)
        direction[net_return > config.move_threshold] = 1.0
        direction[net_return < -config.move_threshold] = 0.0

        # Magnitude: clipped absolute return
        magnitude = raw_return.clip(
            lower=-config.magnitude_clip,
            upper=config.magnitude_clip,
        )

        # Valid mask: has future data AND clear signal
        valid = direction.notna() & future_close.notna()

        # Don't use last N bars of each session (no future data)
        session_ends = df.groupby("date").tail(bars).index
        valid.loc[session_ends] = False

        targets[f"dir_{name}"] = direction
        targets[f"mag_{name}"] = magnitude
        targets[f"valid_{name}"] = valid

    return targets


def get_target_stats(targets: pd.DataFrame) -> dict:
    """Print target distribution stats."""
    stats = {}
    for h in HORIZONS:
        valid = targets[f"valid_{h}"]
        dirs = targets.loc[valid, f"dir_{h}"]
        mags = targets.loc[valid, f"mag_{h}"]
        stats[h] = {
            "total_samples": valid.sum(),
            "pct_up": (dirs == 1).mean() * 100,
            "pct_down": (dirs == 0).mean() * 100,
            "mean_magnitude": mags.abs().mean() * 100,
            "median_magnitude": mags.abs().median() * 100,
        }
        print(f"\n{h}:")
        print(f"  Samples: {stats[h]['total_samples']:,}")
        print(f"  Up/Down: {stats[h]['pct_up']:.1f}% / {stats[h]['pct_down']:.1f}%")
        print(f"  Mean |move|: {stats[h]['mean_magnitude']:.2f}%")
        print(f"  Median |move|: {stats[h]['median_magnitude']:.2f}%")
    return stats
```

### 1.2 Smart Subsampling

```python
# FILE: src/intradaynet/sampling.py

import numpy as np
from typing import Tuple


def smart_subsample(
    X: np.ndarray,
    y_dir: np.ndarray,
    y_mag: np.ndarray,
    valid_mask: np.ndarray,
    max_samples: int = 2_000_000,
    extreme_percentile: float = 80,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep all extreme-move samples, subsample the rest.

    Strategy:
    1. Keep ALL samples in top/bottom 20% by magnitude (the signal)
    2. Randomly sample from the middle 60% (the noise)
    3. Balance up/down classes

    This preserves the rare big-move events that matter most
    while reducing training set size 3-5x.
    """
    rng = np.random.RandomState(seed)

    # Filter to valid only
    valid_idx = np.where(valid_mask)[0]
    X_valid = X[valid_idx]
    y_dir_valid = y_dir[valid_idx]
    y_mag_valid = y_mag[valid_idx]

    abs_mag = np.abs(y_mag_valid)
    threshold = np.percentile(abs_mag, extreme_percentile)

    # Always keep extreme moves
    extreme_mask = abs_mag >= threshold
    extreme_idx = np.where(extreme_mask)[0]

    # Sample from non-extreme
    normal_idx = np.where(~extreme_mask)[0]
    n_extreme = len(extreme_idx)
    n_normal_budget = max_samples - n_extreme

    if n_normal_budget > 0 and len(normal_idx) > n_normal_budget:
        # Stratified: keep class balance
        up_normal = normal_idx[y_dir_valid[normal_idx] == 1]
        down_normal = normal_idx[y_dir_valid[normal_idx] == 0]

        n_up = min(len(up_normal), n_normal_budget // 2)
        n_down = min(len(down_normal), n_normal_budget // 2)

        sampled_up = rng.choice(up_normal, size=n_up, replace=False)
        sampled_down = rng.choice(down_normal, size=n_down, replace=False)
        sampled_normal = np.concatenate([sampled_up, sampled_down])
    else:
        sampled_normal = normal_idx

    # Combine
    final_idx = np.concatenate([extreme_idx, sampled_normal])
    rng.shuffle(final_idx)

    print(f"Subsampling: {len(valid_idx):,} → {len(final_idx):,}")
    print(f"  Extreme (top {100 - extreme_percentile:.0f}%): {n_extreme:,}")
    print(f"  Normal (sampled): {len(sampled_normal):,}")
    print(
        f"  Class balance: {(y_dir_valid[final_idx] == 1).mean():.1%} up"
        f" / {(y_dir_valid[final_idx] == 0).mean():.1%} down"
    )

    return (
        X_valid[final_idx],
        y_dir_valid[final_idx],
        y_mag_valid[final_idx],
    )
```

### 1.3 Enhanced Feature Flattening

```python
# FILE: src/intradaynet/features/flatten.py

import numpy as np
import pandas as pd
from typing import List


def flatten_window_for_lgbm(
    window: np.ndarray,
    session: np.ndarray,
    sentiment: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """
    Flatten a (120, 25) window into a 1D feature vector for LightGBM.

    Categories:
    1. Rolling statistics at multiple windows (existing)
    2. Distribution shape features (NEW)
    3. Time-weighted features (NEW)
    4. Regime shift features (NEW)
    5. Sequence summary features (NEW)
    6. Session features (pass-through)
    7. Sentiment features (pass-through)
    """
    features = {}

    # ─── 1. Rolling statistics (existing, keep these) ───
    for i, name in enumerate(feature_names):
        col = window[:, i]
        for w in [5, 15, 30, 60, 120]:
            if w > len(col):
                continue
            segment = col[-w:]
            features[f"{name}_mean_{w}"] = np.nanmean(segment)
            features[f"{name}_std_{w}"] = np.nanstd(segment)

        # Diff: recent vs older
        if len(col) >= 30:
            features[f"{name}_diff_5v30"] = (
                np.nanmean(col[-5:]) - np.nanmean(col[-30:])
            )
        if len(col) >= 120:
            features[f"{name}_diff_30v120"] = (
                np.nanmean(col[-30:]) - np.nanmean(col[:90])
            )

    # ─── 2. Distribution shape (NEW) ───
    for i, name in enumerate(feature_names):
        col = window[:, i]
        features[f"{name}_p10"] = np.nanpercentile(col, 10)
        features[f"{name}_p25"] = np.nanpercentile(col, 25)
        features[f"{name}_p75"] = np.nanpercentile(col, 75)
        features[f"{name}_p90"] = np.nanpercentile(col, 90)
        features[f"{name}_iqr"] = (
            features[f"{name}_p75"] - features[f"{name}_p25"]
        )

        # Skew and kurtosis (robust to NaN)
        valid = col[~np.isnan(col)]
        if len(valid) > 3:
            mean = np.mean(valid)
            std = np.std(valid)
            if std > 1e-8:
                features[f"{name}_skew"] = float(
                    np.mean(((valid - mean) / std) ** 3)
                )
                features[f"{name}_kurt"] = float(
                    np.mean(((valid - mean) / std) ** 4) - 3
                )
            else:
                features[f"{name}_skew"] = 0.0
                features[f"{name}_kurt"] = 0.0
        else:
            features[f"{name}_skew"] = 0.0
            features[f"{name}_kurt"] = 0.0

    # ─── 3. Time-weighted (exponential, recent bars matter more) ───
    weights = np.exp(np.linspace(-3, 0, len(window)))
    weights /= weights.sum()
    for i, name in enumerate(feature_names):
        col = window[:, i]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() > 0:
            w = weights[valid_mask]
            w /= w.sum()
            features[f"{name}_ewm"] = float(np.average(col[valid_mask], weights=w))
        else:
            features[f"{name}_ewm"] = 0.0

    # ─── 4. Regime shift detection ───
    key_features = [
        "rsi_14", "bb_zscore", "volume_ratio", "momentum_5", "atr_14"
    ]
    for name in key_features:
        if name in feature_names:
            i = feature_names.index(name)
            col = window[:, i]
            q1 = col[:40]   # first third
            q2 = col[40:80] # middle third
            q3 = col[80:]   # last third

            features[f"{name}_regime_q1vq3"] = (
                np.nanmean(q3) - np.nanmean(q1)
            )
            features[f"{name}_regime_q2vq3"] = (
                np.nanmean(q3) - np.nanmean(q2)
            )
            features[f"{name}_regime_trend"] = float(
                np.polyfit(
                    np.arange(len(col)),
                    np.nan_to_num(col, 0),
                    1,
                )[0]
            )

    # ─── 5. Sequence summary ───
    ret_idx = feature_names.index("log_return") if "log_return" in feature_names else 0
    returns = window[:, ret_idx]
    valid_ret = returns[~np.isnan(returns)]

    if len(valid_ret) > 1:
        # Longest positive/negative streak
        signs = (valid_ret > 0).astype(int)
        streaks = []
        current = 1
        for j in range(1, len(signs)):
            if signs[j] == signs[j - 1]:
                current += 1
            else:
                streaks.append(current)
                current = 1
        streaks.append(current)

        features["max_streak"] = max(streaks) if streaks else 1
        features["n_reversals"] = len(streaks)

        # Max drawdown within window
        cumret = np.cumsum(valid_ret)
        running_max = np.maximum.accumulate(cumret)
        drawdowns = cumret - running_max
        features["max_intra_drawdown"] = float(np.min(drawdowns))

        # Max runup within window
        running_min = np.minimum.accumulate(cumret)
        runups = cumret - running_min
        features["max_intra_runup"] = float(np.max(runups))

        # Volatility clustering: std of first half vs second half
        half = len(valid_ret) // 2
        features["vol_clustering"] = (
            np.std(valid_ret[half:]) / (np.std(valid_ret[:half]) + 1e-8)
        )
    else:
        features["max_streak"] = 0
        features["n_reversals"] = 0
        features["max_intra_drawdown"] = 0
        features["max_intra_runup"] = 0
        features["vol_clustering"] = 1.0

    # ─── 6. Last bar snapshot (most recent state) ───
    for i, name in enumerate(feature_names):
        features[f"{name}_last"] = float(window[-1, i])

    # ─── 7. Session + Sentiment (pass-through) ───
    for i, val in enumerate(session):
        features[f"session_{i}"] = float(val)
    for i, val in enumerate(sentiment):
        features[f"sentiment_{i}"] = float(val)

    return np.array(list(features.values()), dtype=np.float32), list(
        features.keys()
    )
```

### 1.4 Complete Prebatch Pipeline

```python
# FILE: scripts/prebatch_lgbm_v2.py

"""
Master prebatching script. Run ONCE, then all training is instant.

Usage:
    python scripts/prebatch_lgbm_v2.py --output prebatched_v2/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

from intradaynet.features.per_bar_features import compute_per_bar_features
from intradaynet.features.session_features import compute_session_features
from intradaynet.features.sentiment_features import load_sentiment_data
from intradaynet.features.flatten import flatten_window_for_lgbm
from intradaynet.targets import compute_targets, HORIZONS, TargetConfig

SEQ_LENGTH = 120
DATA_DIR = Path("nifty500_parquet/")  # use parquet!
SENTIMENT_PATH = Path("sentiment/combined_sentiment_2015_2025.csv")


def process_single_stock(
    stock_path: Path,
    sentiment_df: pd.DataFrame,
    target_config: TargetConfig,
) -> dict:
    """Process one stock file → flat features + targets."""

    symbol = stock_path.stem.replace(".NS", "")

    try:
        df = pd.read_parquet(stock_path)

        # Filter market hours
        df["time"] = df["Datetime"].dt.time
        market_open = pd.Timestamp("09:15").time()
        market_close = pd.Timestamp("15:30").time()
        df = df[(df["time"] >= market_open) & (df["time"] <= market_close)]
        df["date"] = df["Datetime"].dt.date

        if len(df) < SEQ_LENGTH + 375:
            return None

        # Compute features
        df = compute_per_bar_features(df)
        session_features = compute_session_features(df)

        # Compute targets
        targets = compute_targets(df, target_config)

        # Merge sentiment
        stock_sentiment = sentiment_df.get(symbol, None)

        # Extract windows
        all_X = []
        all_targets = {h: {"dir": [], "mag": [], "valid": []} for h in HORIZONS}
        feature_names = None

        dates = df["date"].unique()
        for date in dates:
            day_df = df[df["date"] == date].reset_index(drop=True)
            day_targets = targets.loc[day_df.index]

            if len(day_df) < SEQ_LENGTH + 60:
                continue

            # Get sentiment for this date
            if stock_sentiment is not None and date in stock_sentiment.index:
                sent = stock_sentiment.loc[date].values.astype(np.float32)
            else:
                sent = np.zeros(24, dtype=np.float32)

            # Get session features for this date
            sess = session_features.get(date, np.zeros(20, dtype=np.float32))

            # Slide window
            bar_feature_cols = [
                c
                for c in day_df.columns
                if c
                not in [
                    "Datetime", "date", "time", "Open", "High",
                    "Low", "Close", "Volume",
                ]
            ]

            for start in range(0, len(day_df) - SEQ_LENGTH, 15):
                # Step by 15 bars (every 15 min) to reduce redundancy
                end = start + SEQ_LENGTH
                window = day_df.iloc[start:end][bar_feature_cols].values

                if np.isnan(window).mean() > 0.3:
                    continue

                flat, names = flatten_window_for_lgbm(
                    window.astype(np.float32),
                    sess,
                    sent,
                    bar_feature_cols,
                )
                feature_names = names
                all_X.append(flat)

                for h in HORIZONS:
                    idx = end - 1  # prediction from last bar of window
                    if idx < len(day_targets):
                        all_targets[h]["dir"].append(
                            day_targets.iloc[idx][f"dir_{h}"]
                        )
                        all_targets[h]["mag"].append(
                            day_targets.iloc[idx][f"mag_{h}"]
                        )
                        all_targets[h]["valid"].append(
                            day_targets.iloc[idx][f"valid_{h}"]
                        )
                    else:
                        all_targets[h]["dir"].append(np.nan)
                        all_targets[h]["mag"].append(0.0)
                        all_targets[h]["valid"].append(False)

        if not all_X:
            return None

        result = {
            "symbol": symbol,
            "X": np.stack(all_X),
            "feature_names": feature_names,
        }
        for h in HORIZONS:
            result[f"dir_{h}"] = np.array(all_targets[h]["dir"], dtype=np.float32)
            result[f"mag_{h}"] = np.array(all_targets[h]["mag"], dtype=np.float32)
            result[f"valid_{h}"] = np.array(all_targets[h]["valid"], dtype=bool)

        return result

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="prebatched_v2/")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load sentiment
    print("Loading sentiment data...")
    sentiment_df = load_sentiment_data(SENTIMENT_PATH)

    # Get all stock files
    stock_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Processing {len(stock_files)} stocks...")

    target_config = TargetConfig(horizons=HORIZONS)

    # Process all stocks (parallel)
    all_X = []
    all_targets = {h: {"dir": [], "mag": [], "valid": []} for h in HORIZONS}
    feature_names = None

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_stock, f, sentiment_df, target_config
            ): f
            for f in stock_files
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Stocks"
        ):
            result = future.result()
            if result is None:
                continue

            all_X.append(result["X"])
            feature_names = result["feature_names"]

            for h in HORIZONS:
                all_targets[h]["dir"].append(result[f"dir_{h}"])
                all_targets[h]["mag"].append(result[f"mag_{h}"])
                all_targets[h]["valid"].append(result[f"valid_{h}"])

    # Concatenate
    X = np.concatenate(all_X, axis=0)
    print(f"\nTotal samples: {X.shape[0]:,}")
    print(f"Total features: {X.shape[1]:,}")

    # Save
    save_dict = {"X": X, "feature_names": np.array(feature_names)}
    for h in HORIZONS:
        save_dict[f"dir_{h}"] = np.concatenate(all_targets[h]["dir"])
        save_dict[f"mag_{h}"] = np.concatenate(all_targets[h]["mag"])
        save_dict[f"valid_{h}"] = np.concatenate(all_targets[h]["valid"])

    np.savez_compressed(output_dir / "lgbm_dataset.npz", **save_dict)
    print(f"Saved to {output_dir / 'lgbm_dataset.npz'}")


if __name__ == "__main__":
    main()
```

---

## Phase 2: Training Pipeline (Week 2-3)

### 2.1 Walk-Forward Training

```python
# FILE: scripts/train_lgbm_v2.py

"""
LightGBM training with walk-forward validation.

Usage:
    python scripts/train_lgbm_v2.py \
        --data prebatched_v2/lgbm_dataset.npz \
        --output runs/lgbm_v2/ \
        --horizon H60
"""

import argparse
import json
import lightgbm as lgb
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
)

from intradaynet.sampling import smart_subsample


def get_time_splits(
    n_samples: int,
    n_folds: int = 5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> list:
    """
    Walk-forward expanding window splits.

    Fold 1: Train [0, 50%], Val [50%, 60%], Test [60%, 70%]
    Fold 2: Train [0, 60%], Val [60%, 70%], Test [70%, 80%]
    Fold 3: Train [0, 70%], Val [70%, 80%], Test [80%, 90%]
    Fold 4: Train [0, 80%], Val [80%, 90%], Test [90%, 100%]
    """
    splits = []
    base = 0.5  # start with 50% as initial training set
    step = (1.0 - base - val_ratio - test_ratio) / max(n_folds - 1, 1)

    for fold in range(n_folds):
        train_end = base + fold * step
        val_end = train_end + val_ratio
        test_end = val_end + test_ratio

        if test_end > 1.0:
            break

        train_idx = np.arange(0, int(train_end * n_samples))
        val_idx = np.arange(
            int(train_end * n_samples), int(val_end * n_samples)
        )
        test_idx = np.arange(
            int(val_end * n_samples), int(test_end * n_samples)
        )

        splits.append(
            {
                "fold": fold,
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
        )

    return splits


def get_direction_params() -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 100,
        "max_bin": 127,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.01,
        "num_threads": -1,
        "verbose": -1,
        "seed": 42,
    }


def get_magnitude_params() -> dict:
    return {
        "objective": "regression_l1",  # MAE, robust to outliers
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 100,
        "max_bin": 127,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.01,
        "num_threads": -1,
        "verbose": -1,
        "seed": 42,
    }


def train_direction_model(
    X_train, y_train, X_val, y_val, params, n_rounds=2000
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_rounds,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )
    return model


def train_magnitude_model(
    X_train, y_train, X_val, y_val, params, n_rounds=2000
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_rounds,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )
    return model


def evaluate_direction(model, X, y_true) -> dict:
    y_prob = model.predict(X)
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),

        # HIGH CONFIDENCE metrics (what actually matters for trading)
        "precision_at_60": float(
            precision_score(
                y_true[y_prob > 0.6],
                (y_prob[y_prob > 0.6] > 0.5).astype(int),
                zero_division=0,
            )
        )
        if (y_prob > 0.6).sum() > 0
        else 0.0,
        "n_above_60": int((y_prob > 0.6).sum()),
        "precision_at_65": float(
            precision_score(
                y_true[y_prob > 0.65],
                (y_prob[y_prob > 0.65] > 0.5).astype(int),
                zero_division=0,
            )
        )
        if (y_prob > 0.65).sum() > 0
        else 0.0,
        "n_above_65": int((y_prob > 0.65).sum()),
    }


def evaluate_magnitude(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "n_samples": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to lgbm_dataset.npz")
    parser.add_argument("--output", default="runs/lgbm_v2/")
    parser.add_argument("--horizon", default="H60", choices=["H15", "H30", "H60", "H375"])
    parser.add_argument("--max-samples", type=int, default=2_000_000)
    parser.add_argument("--n-folds", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    y_dir = data[f"dir_{args.horizon}"]
    y_mag = data[f"mag_{args.horizon}"]
    valid = data[f"valid_{args.horizon}"]
    feature_names = data["feature_names"].tolist()

    print(f"Total: {len(X):,} samples, {X.shape[1]} features")
    print(f"Valid for {args.horizon}: {valid.sum():,}")

    # Subsample
    X_sub, y_dir_sub, y_mag_sub = smart_subsample(
        X, y_dir, y_mag, valid, max_samples=args.max_samples
    )

    # Walk-forward splits
    splits = get_time_splits(len(X_sub), n_folds=args.n_folds)
    print(f"\n{len(splits)} walk-forward folds")

    dir_metrics_all = []
    mag_metrics_all = []

    for split in splits:
        fold = split["fold"]
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"  Train: {len(split['train']):,}")
        print(f"  Val:   {len(split['val']):,}")
        print(f"  Test:  {len(split['test']):,}")
        print(f"{'='*60}")

        X_train = X_sub[split["train"]]
        X_val = X_sub[split["val"]]
        X_test = X_sub[split["test"]]

        y_dir_train = y_dir_sub[split["train"]]
        y_dir_val = y_dir_sub[split["val"]]
        y_dir_test = y_dir_sub[split["test"]]

        y_mag_train = y_mag_sub[split["train"]]
        y_mag_val = y_mag_sub[split["val"]]
        y_mag_test = y_mag_sub[split["test"]]

        # Train direction model
        print("\n--- Direction Model ---")
        dir_model = train_direction_model(
            X_train, y_dir_train, X_val, y_dir_val, get_direction_params()
        )
        dir_test_metrics = evaluate_direction(dir_model, X_test, y_dir_test)
        dir_metrics_all.append(dir_test_metrics)

        print(f"  Test AUC:      {dir_test_metrics['auc']:.4f}")
        print(f"  Test Accuracy: {dir_test_metrics['accuracy']:.4f}")
        print(f"  Prec@0.60:     {dir_test_metrics['precision_at_60']:.4f} (n={dir_test_metrics['n_above_60']})")
        print(f"  Prec@0.65:     {dir_test_metrics['precision_at_65']:.4f} (n={dir_test_metrics['n_above_65']})")

        # Train magnitude model
        print("\n--- Magnitude Model ---")
        mag_model = train_magnitude_model(
            X_train, y_mag_train, X_val, y_mag_val, get_magnitude_params()
        )
        mag_test_metrics = evaluate_magnitude(mag_model, X_test, y_mag_test)
        mag_metrics_all.append(mag_test_metrics)

        print(f"  Test MAE:         {mag_test_metrics['mae']:.5f}")
        print(f"  Test Correlation: {mag_test_metrics['correlation']:.4f}")

    # ─── Summary across folds ───
    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*60}")

    avg_auc = np.mean([m["auc"] for m in dir_metrics_all])
    avg_acc = np.mean([m["accuracy"] for m in dir_metrics_all])
    avg_prec60 = np.mean([m["precision_at_60"] for m in dir_metrics_all])
    avg_mae = np.mean([m["mae"] for m in mag_metrics_all])
    avg_corr = np.mean([m["correlation"] for m in mag_metrics_all])

    print(f"  Avg Test AUC:       {avg_auc:.4f} ± {np.std([m['auc'] for m in dir_metrics_all]):.4f}")
    print(f"  Avg Test Accuracy:  {avg_acc:.4f}")
    print(f"  Avg Prec@0.60:      {avg_prec60:.4f}")
    print(f"  Avg Test MAE:       {avg_mae:.5f}")
    print(f"  Avg Test Corr:      {avg_corr:.4f}")

    # ─── Train final model on ALL data ───
    print(f"\n{'='*60}")
    print("FINAL MODEL (trained on all data)")
    print(f"{'='*60}")

    # 90/10 split for final early stopping
    n = len(X_sub)
    split_idx = int(n * 0.9)
    X_final_train, X_final_val = X_sub[:split_idx], X_sub[split_idx:]
    y_dir_final_train, y_dir_final_val = (
        y_dir_sub[:split_idx],
        y_dir_sub[split_idx:],
    )
    y_mag_final_train, y_mag_final_val = (
        y_mag_sub[:split_idx],
        y_mag_sub[split_idx:],
    )

    final_dir_model = train_direction_model(
        X_final_train,
        y_dir_final_train,
        X_final_val,
        y_dir_final_val,
        get_direction_params(),
    )

    final_mag_model = train_magnitude_model(
        X_final_train,
        y_mag_final_train,
        X_final_val,
        y_mag_final_val,
        get_magnitude_params(),
    )

    # Save
    final_dir_model.save_model(str(output_dir / f"dir_{args.horizon}.lgb"))
    final_mag_model.save_model(str(output_dir / f"mag_{args.horizon}.lgb"))

    # Feature importance
    importance = final_dir_model.feature_importance(importance_type="gain")
    feat_imp = sorted(
        zip(feature_names, importance), key=lambda x: x[1], reverse=True
    )

    print("\nTop 20 features by gain:")
    for name, imp in feat_imp[:20]:
        print(f"  {imp:12.1f}  {name}")

    # Zero-importance features
    zero_feats = [name for name, imp in feat_imp if imp == 0]
    print(f"\nZero-importance features ({len(zero_feats)}):")
    for name in zero_feats[:10]:
        print(f"  {name}")
    if len(zero_feats) > 10:
        print(f"  ... and {len(zero_feats) - 10} more")

    # Save metrics
    metrics = {
        "horizon": args.horizon,
        "n_samples": len(X_sub),
        "n_features": X_sub.shape[1],
        "n_folds": len(splits),
        "walk_forward": {
            "direction": dir_metrics_all,
            "magnitude": mag_metrics_all,
        },
        "summary": {
            "avg_auc": avg_auc,
            "avg_accuracy": avg_acc,
            "avg_precision_at_60": avg_prec60,
            "avg_mae": avg_mae,
            "avg_correlation": avg_corr,
        },
    }

    with open(output_dir / f"metrics_{args.horizon}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
```

### 2.2 Feature Pruning Script

```python
# FILE: scripts/prune_features.py

"""
After training, identify and remove dead-weight features.
Retrain with pruned feature set for speed + regularization.

Usage:
    python scripts/prune_features.py \
        --model runs/lgbm_v2/dir_H60.lgb \
        --data prebatched_v2/lgbm_dataset.npz \
        --threshold 0.001
"""

import argparse
import json
import lightgbm as lgb
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Drop features below this fraction of total gain",
    )
    args = parser.parse_args()

    model = lgb.Booster(model_file=args.model)
    data = np.load(args.data, allow_pickle=True)
    feature_names = data["feature_names"].tolist()

    importance = model.feature_importance(importance_type="gain")
    total_gain = importance.sum()
    threshold = total_gain * args.threshold

    keep_mask = importance > threshold
    kept = [
        (name, imp)
        for name, imp, k in zip(feature_names, importance, keep_mask)
        if k
    ]
    dropped = [
        (name, imp)
        for name, imp, k in zip(feature_names, importance, keep_mask)
        if not k
    ]

    print(f"Total features: {len(feature_names)}")
    print(f"Keeping: {len(kept)} (>{args.threshold * 100:.1f}% of total gain)")
    print(f"Dropping: {len(dropped)}")

    print(f"\nDropped features:")
    for name, imp in sorted(dropped, key=lambda x: x[1]):
        print(f"  {name}: gain={imp:.1f}")

    # Save keep list
    keep_names = [name for name, _ in kept]
    keep_indices = [i for i, k in enumerate(keep_mask) if k]

    output = {
        "keep_names": keep_names,
        "keep_indices": keep_indices,
        "n_original": len(feature_names),
        "n_kept": len(keep_names),
        "n_dropped": len(dropped),
    }

    out_path = Path(args.model).parent / "feature_selection.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved feature selection to {out_path}")
    print(f"Use keep_indices to slice X before training/inference.")


if __name__ == "__main__":
    main()
```

---

## Phase 3: Calibration & Confidence (Week 3)

### 3.1 Probability Calibration

```python
# FILE: src/intradaynet/calibration.py

import numpy as np
import pickle
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb


class LGBMCalibrationWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make LightGBM compatible with sklearn calibration."""

    def __init__(self, booster: lgb.Booster):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self  # already trained

    def predict_proba(self, X):
        pos_prob = self.booster.predict(X)
        return np.column_stack([1 - pos_prob, pos_prob])

    def predict(self, X):
        return (self.booster.predict(X) > 0.5).astype(int)


def calibrate_model(
    booster: lgb.Booster,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",  # or "sigmoid"
) -> CalibratedClassifierCV:
    """
    Calibrate LightGBM probabilities using isotonic regression.

    After calibration:
    - When model says P=0.70, stocks actually go up ~70% of the time
    - This is CRITICAL for setting confidence thresholds
    """
    wrapper = LGBMCalibrationWrapper(booster)

    calibrator = CalibratedClassifierCV(
        wrapper,
        method=method,
        cv="prefit",  # model already trained
    )
    calibrator.fit(X_cal, y_cal)

    # Verify calibration
    raw_probs = booster.predict(X_cal)
    cal_probs = calibrator.predict_proba(X_cal)[:, 1]

    # Bin and check
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (cal_probs >= bins[i]) & (cal_probs < bins[i + 1])
        if mask.sum() > 0:
            actual = y_cal[mask].mean()
            predicted = cal_probs[mask].mean()
            n = mask.sum()
            gap = abs(actual - predicted)
            status = "✅" if gap < 0.05 else "⚠️"
            print(
                f"  {status} Bin [{bins[i]:.1f}, {bins[i+1]:.1f}): "
                f"predicted={predicted:.3f}, actual={actual:.3f}, "
                f"n={n}, gap={gap:.3f}"
            )

    return calibrator


def save_calibrator(calibrator, path: str):
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
```

---

## Phase 4 (continued): Realistic Backtester

### 4.1 Transaction Costs

```python
# FILE: src/intradaynet/costs.py

from dataclasses import dataclass


@dataclass
class IndianMarketCosts:
    """Accurate NSE transaction costs as of 2026."""

    brokerage_per_order: float = 20.0
    stt_rate: float = 0.00025       # sell side intraday
    exchange_txn: float = 0.0000345
    sebi_turnover: float = 0.000001
    gst_rate: float = 0.18
    stamp_duty: float = 0.00003     # buy side
    slippage: float = 0.0005        # 0.05% per side

    def total_cost(self, entry_price: float, qty: int) -> float:
        buy_value = entry_price * qty
        sell_value = entry_price * qty

        brokerage = self.brokerage_per_order * 2
        stt = sell_value * self.stt_rate
        exchange = (buy_value + sell_value) * self.exchange_txn
        sebi = (buy_value + sell_value) * self.sebi_turnover
        gst = (brokerage + exchange + sebi) * self.gst_rate
        stamp = buy_value * self.stamp_duty
        slippage_cost = (buy_value + sell_value) * self.slippage

        return brokerage + stt + exchange + sebi + gst + stamp + slippage_cost

    def cost_as_pct(self, entry_price: float, qty: int) -> float:
        position_value = entry_price * qty
        return self.total_cost(entry_price, qty) / position_value * 100

    def breakeven_move_pct(self, entry_price: float, qty: int) -> float:
        """Minimum % move needed just to break even."""
        return self.cost_as_pct(entry_price, qty)
```

### 4.2 Backtester

```python
# FILE: scripts/backtest_lgbm_v2.py

import argparse
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from intradaynet.costs import IndianMarketCosts


@dataclass
class Trade:
    date: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    target_price: float
    stop_loss: float
    predicted_prob: float
    predicted_mag: float
    confidence_score: float
    # Filled after resolution
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    outcome: Optional[str] = None  # "TARGET", "STOPLOSS", "EOD"
    costs: Optional[float] = None


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    trades: List[Trade]
    daily_equity: List[dict]

    @property
    def total_return_pct(self):
        return (self.final_equity / self.initial_capital - 1) * 100

    @property
    def total_trades(self):
        return len(self.trades)

    @property
    def win_rate(self):
        if not self.trades:
            return 0
        wins = sum(1 for t in self.trades if t.pnl and t.pnl > 0)
        return wins / len(self.trades) * 100

    @property
    def profit_factor(self):
        gross_profit = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
        gross_loss = abs(
            sum(t.pnl for t in self.trades if t.pnl and t.pnl < 0)
        )
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown_pct(self):
        if not self.daily_equity:
            return 0
        equities = [d["equity"] for d in self.daily_equity]
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self):
        if len(self.daily_equity) < 2:
            return 0
        returns = []
        for i in range(1, len(self.daily_equity)):
            r = (
                self.daily_equity[i]["equity"]
                / self.daily_equity[i - 1]["equity"]
                - 1
            )
            returns.append(r)
        returns = np.array(returns)
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)


def resolve_trade_with_minute_data(
    trade: Trade,
    minute_df: pd.DataFrame,
    costs: IndianMarketCosts,
    position_size: float,
) -> Trade:
    """
    Walk through minute bars to see if target or stop-loss hit first.
    If neither hit by EOD, exit at last bar close.
    """
    qty = int(position_size / trade.entry_price)
    if qty == 0:
        trade.outcome = "SKIP"
        trade.pnl = 0
        trade.pnl_pct = 0
        trade.costs = 0
        return trade

    cost = costs.total_cost(trade.entry_price, qty)

    for _, bar in minute_df.iterrows():
        if trade.direction == "LONG":
            # Check stop-loss first (conservative)
            if bar["Low"] <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.outcome = "STOPLOSS"
                break
            if bar["High"] >= trade.target_price:
                trade.exit_price = trade.target_price
                trade.outcome = "TARGET"
                break
        else:  # SHORT
            if bar["High"] >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.outcome = "STOPLOSS"
                break
            if bar["Low"] <= trade.target_price:
                trade.exit_price = trade.target_price
                trade.outcome = "TARGET"
                break
    else:
        # Neither hit — exit at EOD
        trade.exit_price = minute_df.iloc[-1]["Close"]
        trade.outcome = "EOD"

    # Calculate PnL
    if trade.direction == "LONG":
        gross_pnl = (trade.exit_price - trade.entry_price) * qty
    else:
        gross_pnl = (trade.entry_price - trade.exit_price) * qty

    trade.costs = cost
    trade.pnl = gross_pnl - cost
    trade.pnl_pct = trade.pnl / (trade.entry_price * qty) * 100

    return trade


def run_backtest(
    model_dir: Path,
    horizon: str,
    data_dir: Path,
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000,
    position_size: float = 100_000,
    max_positions: int = 5,
    dir_threshold: float = 0.60,
    min_confidence: float = 0.55,
    stop_loss_pct: float = 0.01,
) -> BacktestResult:
    """Full walk-forward backtest with realistic costs."""

    # Load models
    dir_model = lgb.Booster(model_file=str(model_dir / f"dir_{horizon}.lgb"))
    mag_model = lgb.Booster(model_file=str(model_dir / f"mag_{horizon}.lgb"))

    costs = IndianMarketCosts()

    # Print breakeven analysis
    sample_price = 1000
    sample_qty = int(position_size / sample_price)
    print(f"Breakeven move needed: {costs.breakeven_move_pct(sample_price, sample_qty):.3f}%")
    print(f"Total cost per trade (₹{sample_price} × {sample_qty}): ₹{costs.total_cost(sample_price, sample_qty):.2f}")

    equity = initial_capital
    all_trades = []
    daily_equity = []

    # Get trading dates
    dates = pd.bdate_range(start_date, end_date)

    for date in tqdm(dates, desc="Backtesting"):
        date_str = date.strftime("%Y-%m-%d")

        # ... (feature computation for this date, same as morning_picks)
        # This is where you'd compute features for all stocks on this date
        # and run inference to get picks

        # Placeholder: you'd integrate your existing feature pipeline here
        # picks = generate_picks_for_date(dir_model, mag_model, date_str, ...)

        # For each pick, resolve with minute data
        # for pick in picks[:max_positions]:
        #     trade = Trade(...)
        #     minute_df = load_minute_data(pick.symbol, date_str)
        #     trade = resolve_trade_with_minute_data(
        #         trade, minute_df, costs, position_size
        #     )
        #     equity += trade.pnl
        #     all_trades.append(trade)

        daily_equity.append({"date": date_str, "equity": equity})

    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=equity,
        trades=all_trades,
        daily_equity=daily_equity,
    )


def print_backtest_report(result: BacktestResult):
    print(f"\n{'='*60}")
    print("BACKTEST REPORT")
    print(f"{'='*60}")
    print(f"Initial Capital:  ₹{result.initial_capital:,.2f}")
    print(f"Final Equity:     ₹{result.final_equity:,.2f}")
    print(f"Total Return:     {result.total_return_pct:.2f}%")
    print(f"Total Trades:     {result.total_trades}")
    print(f"Win Rate:         {result.win_rate:.2f}%")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")

    if result.trades:
        outcomes = {}
        for t in result.trades:
            outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
        print(f"\nOutcome Breakdown:")
        for outcome, count in sorted(outcomes.items()):
            print(f"  {outcome}: {count} ({count/len(result.trades)*100:.1f}%)")

        total_costs = sum(t.costs for t in result.trades if t.costs)
        print(f"\nTotal Transaction Costs: ₹{total_costs:,.2f}")
        print(
            f"Costs as % of PnL: "
            f"{total_costs / abs(result.final_equity - result.initial_capital) * 100:.1f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--horizon", default="H60")
    parser.add_argument("--data", default="nifty500_parquet/")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--position-size", type=float, default=100_000)
    parser.add_argument("--max-positions", type=int, default=5)
    args = parser.parse_args()

    result = run_backtest(
        model_dir=Path(args.model),
        horizon=args.horizon,
        data_dir=Path(args.data),
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        position_size=args.position_size,
        max_positions=args.max_positions,
    )

    print_backtest_report(result)
```

---

## Phase 5: Market Regime Filter (Week 4)

### 5.1 Regime Detection

```python
# FILE: src/intradaynet/regime.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class MarketRegime(Enum):
    CALM_BULL = "calm_bull"         # Low VIX, uptrend
    CALM_BEAR = "calm_bear"         # Low VIX, downtrend
    VOLATILE_BULL = "volatile_bull" # High VIX, uptrend
    VOLATILE_BEAR = "volatile_bear" # High VIX, downtrend
    EXTREME = "extreme"             # VIX spike, avoid trading


@dataclass
class RegimeConfig:
    vix_low: float = 15.0
    vix_high: float = 22.0
    vix_extreme: float = 28.0
    trend_window: int = 10          # days
    max_gap_pct: float = 1.5       # skip if gap > 1.5%
    skip_expiry: bool = False       # can set True if model bad on expiry


def detect_regime(
    vix: float,
    vix_change: float,
    nifty_returns_10d: np.ndarray,
    gap_pct: float,
    is_expiry: bool,
    config: RegimeConfig = RegimeConfig(),
) -> Tuple[MarketRegime, bool, str]:
    """
    Returns (regime, should_trade, reason).
    """

    # Extreme VIX — don't trade
    if vix > config.vix_extreme:
        return (
            MarketRegime.EXTREME,
            False,
            f"VIX={vix:.1f} > {config.vix_extreme} (extreme volatility)",
        )

    # VIX spike (>20% single day increase) — don't trade
    if vix_change > 0.20:
        return (
            MarketRegime.EXTREME,
            False,
            f"VIX spiked {vix_change*100:.1f}% (fear event)",
        )

    # Large gap — skip
    if abs(gap_pct) > config.max_gap_pct:
        return (
            MarketRegime.EXTREME,
            False,
            f"Gap={gap_pct:.2f}% > ±{config.max_gap_pct}% (extreme gap)",
        )

    # Expiry skip
    if is_expiry and config.skip_expiry:
        return (
            MarketRegime.EXTREME,
            False,
            "Expiry day — skipping",
        )

    # Determine trend
    trend = np.mean(nifty_returns_10d) > 0  # simple
    is_volatile = vix > config.vix_high

    if is_volatile:
        regime = (
            MarketRegime.VOLATILE_BULL if trend
            else MarketRegime.VOLATILE_BEAR
        )
    else:
        regime = (
            MarketRegime.CALM_BULL if trend
            else MarketRegime.CALM_BEAR
        )

    return regime, True, f"Regime={regime.value}, VIX={vix:.1f}"


def get_regime_adjustments(regime: MarketRegime) -> dict:
    """
    Adjust trading parameters based on regime.
    """
    adjustments = {
        MarketRegime.CALM_BULL: {
            "dir_threshold": 0.58,    # slightly relaxed
            "min_confidence": 0.55,
            "stop_loss_pct": 0.008,   # tighter SL in calm markets
            "max_positions": 5,
            "prefer_direction": "LONG",
        },
        MarketRegime.CALM_BEAR: {
            "dir_threshold": 0.60,
            "min_confidence": 0.58,
            "stop_loss_pct": 0.010,
            "max_positions": 4,
            "prefer_direction": "SHORT",
        },
        MarketRegime.VOLATILE_BULL: {
            "dir_threshold": 0.63,    # stricter in volatility
            "min_confidence": 0.60,
            "stop_loss_pct": 0.012,   # wider SL
            "max_positions": 3,       # fewer positions
            "prefer_direction": "LONG",
        },
        MarketRegime.VOLATILE_BEAR: {
            "dir_threshold": 0.65,    # very strict
            "min_confidence": 0.62,
            "stop_loss_pct": 0.015,
            "max_positions": 2,
            "prefer_direction": "SHORT",
        },
        MarketRegime.EXTREME: {
            "dir_threshold": 1.0,     # effectively no trades
            "min_confidence": 1.0,
            "stop_loss_pct": 0.02,
            "max_positions": 0,
            "prefer_direction": None,
        },
    }
    return adjustments[regime]
```

---

## Phase 6: Adversarial Validation & Monitoring (Week 4-5)

### 6.1 Distribution Shift Detection

```python
# FILE: scripts/adversarial_validation.py

"""
Check if your training data distribution matches recent market data.
If AUC > 0.70, the market regime has shifted and model may underperform.

Usage:
    python scripts/adversarial_validation.py \
        --train-data prebatched_v2/lgbm_dataset.npz \
        --recent-days 30
"""

import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


def adversarial_validation(
    X_train: np.ndarray,
    X_recent: np.ndarray,
    n_folds: int = 3,
) -> float:
    """
    Train a classifier to distinguish train vs recent data.
    High AUC = distributions differ = model may not generalize.
    """
    y = np.concatenate([
        np.zeros(len(X_train)),
        np.ones(len(X_recent)),
    ])
    X = np.concatenate([X_train, X_recent])

    # Subsample train if much larger
    if len(X_train) > len(X_recent) * 10:
        idx = np.random.choice(
            len(X_train), size=len(X_recent) * 5, replace=False
        )
        X_sub = np.concatenate([X_train[idx], X_recent])
        y_sub = np.concatenate([
            np.zeros(len(idx)),
            np.ones(len(X_recent)),
        ])
    else:
        X_sub, y_sub = X, y

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        max_depth=5,
        verbose=-1,
        n_jobs=-1,
    )

    scores = cross_val_score(
        clf, X_sub, y_sub, cv=n_folds, scoring="roc_auc"
    )
    auc = scores.mean()

    print(f"\nAdversarial Validation AUC: {auc:.4f} ± {scores.std():.4f}")
    if auc > 0.80:
        print("🚩 SEVERE distribution shift — retrain immediately")
    elif auc > 0.70:
        print("⚠️  Moderate shift — monitor closely, consider retraining")
    elif auc > 0.60:
        print("📊 Mild shift — normal market evolution")
    else:
        print("✅ Distributions are similar — model should generalize well")

    # Feature importance of the adversarial model
    clf.fit(X_sub, y_sub)
    importance = clf.feature_importances_
    top_idx = np.argsort(importance)[-10:]

    print("\nTop features distinguishing train vs recent:")
    for i in reversed(top_idx):
        print(f"  Feature {i}: importance={importance[i]}")

    return auc
```

### 6.2 Daily Health Check

```python
# FILE: scripts/daily_health_check.py

"""
Run before morning_picks.py to verify everything is working.

Usage:
    python scripts/daily_health_check.py --model runs/lgbm_v2/
"""

import argparse
import json
import lightgbm as lgb
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def check_model_files(model_dir: Path, horizon: str) -> bool:
    dir_path = model_dir / f"dir_{horizon}.lgb"
    mag_path = model_dir / f"mag_{horizon}.lgb"

    if not dir_path.exists():
        print(f"🚩 Missing {dir_path}")
        return False
    if not mag_path.exists():
        print(f"🚩 Missing {mag_path}")
        return False

    dir_model = lgb.Booster(model_file=str(dir_path))
    mag_model = lgb.Booster(model_file=str(mag_path))

    print(f"✅ Direction model: {dir_model.num_trees()} trees, {dir_model.num_feature()} features")
    print(f"✅ Magnitude model: {mag_model.num_trees()} trees, {mag_model.num_feature()} features")
    return True


def check_data_freshness(data_dir: Path) -> bool:
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        parquet_files = list(data_dir.glob("*.csv"))

    if not parquet_files:
        print("🚩 No data files found")
        return False

    latest = max(f.stat().st_mtime for f in parquet_files)
    latest_dt = datetime.fromtimestamp(latest)
    age_hours = (datetime.now() - latest_dt).total_seconds() / 3600

    if age_hours > 24:
        print(f"⚠️  Data is {age_hours:.0f} hours old — may need refresh")
    else:
        print(f"✅ Data freshness: {age_hours:.1f} hours old")

    print(f"✅ {len(parquet_files)} stock files found")
    return True


def check_metrics(model_dir: Path, horizon: str) -> bool:
    metrics_path = model_dir / f"metrics_{horizon}.json"
    if not metrics_path.exists():
        print(f"⚠️  No metrics file found at {metrics_path}")
        return True  # not critical

    with open(metrics_path) as f:
        metrics = json.load(f)

    summary = metrics.get("summary", {})
    auc = summary.get("avg_auc", 0)
    prec60 = summary.get("avg_precision_at_60", 0)

    print(f"📊 Walk-forward AUC: {auc:.4f}")
    print(f"📊 Precision@0.60:   {prec60:.4f}")

    if auc < 0.52:
        print("🚩 AUC barely above random — model may not be useful")
        return False
    if prec60 < 0.55:
        print("⚠️  Low precision at trading threshold — be cautious")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="nifty500_parquet/")
    parser.add_argument("--horizon", default="H60")
    args = parser.parse_args()

    model_dir = Path(args.model)
    data_dir = Path(args.data)

    print("=" * 50)
    print("DAILY HEALTH CHECK")
    print("=" * 50)

    all_ok = True
    all_ok &= check_model_files(model_dir, args.horizon)
    all_ok &= check_data_freshness(data_dir)
    all_ok &= check_metrics(model_dir, args.horizon)

    print("=" * 50)
    if all_ok:
        print("✅ ALL CHECKS PASSED — ready for morning picks")
    else:
        print("🚩 ISSUES FOUND — review before trading")


if __name__ == "__main__":
    main()
```

---

## Phase 7: Automated Retraining (Week 5-6)

### 7.1 Weekly Retrain Script

```python
# FILE: scripts/weekly_retrain.py

"""
Run every weekend to retrain with latest data.

Usage:
    crontab: 0 6 * * 0 cd /path/to/project && python scripts/weekly_retrain.py

    python scripts/weekly_retrain.py --horizon H60
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon", default="H60", choices=["H15", "H30", "H60", "H375"]
    )
    parser.add_argument("--keep-versions", type=int, default=4)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d")

    print(f"{'='*60}")
    print(f"WEEKLY RETRAIN — {timestamp}")
    print(f"{'='*60}")

    # Step 1: Update data
    print("\n[1/5] Syncing latest market data...")
    subprocess.run(
        [sys.executable, "scripts/sync_data.py"],
        check=True,
    )

    # Step 2: Rebatch features
    print("\n[2/5] Rebatching features...")
    subprocess.run(
        [
            sys.executable,
            "scripts/prebatch_lgbm_v2.py",
            "--output",
            "prebatched_v2/",
        ],
        check=True,
    )

    # Step 3: Train
    output_dir = f"runs/lgbm_v2_{timestamp}/"
    print(f"\n[3/5] Training {args.horizon}...")
    subprocess.run(
        [
            sys.executable,
            "scripts/train_lgbm_v2.py",
            "--data",
            "prebatched_v2/lgbm_dataset.npz",
            "--output",
            output_dir,
            "--horizon",
            args.horizon,
        ],
        check=True,
    )

    # Step 4: Adversarial validation
    print("\n[4/5] Running adversarial validation...")
    subprocess.run(
        [
            sys.executable,
            "scripts/adversarial_validation.py",
            "--train-data",
            "prebatched_v2/lgbm_dataset.npz",
            "--recent-days",
            "30",
        ],
        check=True,
    )

    # Step 5: Update symlink to latest model
    print("\n[5/5] Updating model symlink...")
    latest_link = Path("runs/lgbm_latest")
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(Path(output_dir).resolve())
    print(f"  runs/lgbm_latest → {output_dir}")

    # Cleanup old versions
    versions = sorted(Path("runs/").glob("lgbm_v2_*"))
    if len(versions) > args.keep_versions:
        for old in versions[: -args.keep_versions]:
            print(f"  Removing old version: {old}")
            shutil.rmtree(old)

    print(f"\n{'='*60}")
    print("RETRAIN COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

---

## Complete Execution Timeline

```
WEEK 1 ─── Phase 0: Audit & Fix
  │  Day 1-2: Run audit_features.py, fix all 🚩 leaks
  │  Day 3:   Convert CSV → Parquet
  │  Day 4-5: Verify fixed features produce sane outputs
  │
WEEK 2 ─── Phase 1: Data Pipeline Rebuild
  │  Day 1-2: Implement thresholded targets (targets.py)
  │  Day 3:   Implement smart subsampling (sampling.py)
  │  Day 4-5: Enhanced feature flattening + prebatch v2
  │
WEEK 3 ─── Phase 2-3: Training + Calibration
  │  Day 1-3: Walk-forward training (train_lgbm_v2.py)
  │  Day 4:   Feature pruning
  │  Day 5:   Probability calibration
  │
WEEK 4 ─── Phase 4-5: Backtest + Regime
  │  Day 1-3: Realistic backtester with NSE costs
  │  Day 4-5: Market regime filter
  │
WEEK 5 ─── Phase 6: Validation + Monitoring
  │  Day 1-2: Adversarial validation
  │  Day 3-5: Daily health check, paper trading starts
  │
WEEK 6 ─── Phase 7: Automation + Paper Trading
  │  Day 1-2: Weekly retrain automation
  │  Day 3-5: Paper trade, compare picks vs actual outcomes
  │
WEEK 7-10 ── Paper Trading (minimum 30 sessions)
  │  Track every pick, measure REAL win rate
  │  Compare to backtest expectations
  │  If Sharpe drops from 7.29 to 1.5-2.5, that's NORMAL
  │  If Sharpe < 0.5, go back to Phase 0
  │
WEEK 11+ ── Live Trading (small size)
     Start with 20% of intended capital
     Scale up only after 30+ profitable live sessions
```

---

## Key Files Summary

```
NEW/MODIFIED FILES:
├── scripts/
│   ├── audit_features.py          # Phase 0: leak detection
│   ├── convert_to_parquet.py      # Phase 0: CSV → Parquet
│   ├── prebatch_lgbm_v2.py        # Phase 1: complete rebatch
│   ├── train_lgbm_v2.py           # Phase 2: walk-forward training
│   ├── prune_features.py          # Phase 2: feature selection
│   ├── backtest_lgbm_v2.py        # Phase 4: realistic backtest
│   ├── adversarial_validation.py  # Phase 6: distribution shift
│   ├── daily_health_check.py      # Phase 6: pre-market check
│   └── weekly_retrain.py          # Phase 7: automated retrain
├── src/intradaynet/
│   ├── targets.py                 # Phase 1: thresholded targets
│   ├── sampling.py                # Phase 1: smart subsampling
│   ├── costs.py                   # Phase 4: NSE transaction costs
│   ├── regime.py                  # Phase 5: market regime filter
│   ├── calibration.py             # Phase 3: probability calibration
│   └── features/
│       └── flatten.py             # Phase 1: enhanced flattening
```

**Rule of thumb:** Don't skip Phase 0. If your Sharpe drops from 7.29 to 2.0 after fixing leaks, that 2.0 is the **real** number, and it's still excellent.