#!/usr/bin/env python
"""Fit and save the market RegimeDetector from cached yfinance data.

Usage:
    python scripts/trading/fit_regime_detector.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "models/v8_intraday/regime_detector.pkl"
EXPOSURE_BY_REGIME = {
    "strong_trend_up":    0.60,  # momentum names get over-extended in rip days — partial
    "low_vol_compression": 0.80,
    "choppy_reverting":   1.00,  # BEST: choppy regime = cleaner intraday directional calls
    "strong_trend_down":  0.50,
    "high_vol_crisis":    0.20,  # reduce but don't sit out — some big days happen in crisis
}


def build_regime_features(market_cache: str) -> pd.DataFrame:
    from equity.features.market_features import MarketFeatureBuilder
    mb = MarketFeatureBuilder(market_cache)
    mb._load_cached()

    nifty = mb._get_close("nifty50").dropna()
    vix   = mb._get_close("india_vix").dropna()

    ret = nifty.pct_change()
    vix5d = vix.pct_change(5).fillna(0)
    vix_pct = vix.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)

    # Trend strength: 5d Sharpe of NIFTY
    nifty_adx = (ret.rolling(5).mean() / ret.rolling(5).std().replace(0, 1)).fillna(0)

    # Breadth: NIFTY above 20-DMA
    breadth = (nifty / nifty.rolling(20).mean().replace(0, np.nan) - 1).fillna(0)

    # Autocorrelation: trending (positive) vs choppy (negative)
    autocorr = ret.rolling(20).apply(
        lambda x: x.autocorr(1) if len(x) > 5 else 0, raw=False
    ).fillna(0)

    # Sector dispersion
    sector_keys = ["auto", "bank", "fmcg", "it", "metal", "pharma", "oil_gas", "realty"]
    sector_rets = {}
    for k in sector_keys:
        s = mb._get_close(k)
        if not s.empty:
            sector_rets[k] = s.pct_change()
    dispersion = pd.DataFrame(sector_rets).std(axis=1).fillna(0.01)

    df = pd.DataFrame({
        "vix_level":        vix_pct,
        "vix_5d_change":    vix5d,
        "nifty_adx":        nifty_adx,
        "breadth_20d":      breadth,
        "nifty_autocorr":   autocorr,
        "sector_dispersion": dispersion,
    }).dropna()

    return df


def main():
    from equity.v8.regime_detector import RegimeDetector
    import pickle

    print("Building regime features from market cache...")
    df = build_regime_features(str(ROOT / "market_data_cache"))
    print(f"Feature matrix: {len(df)} rows  {df.index.min().date()} → {df.index.max().date()}")

    print("Fitting K-means regime detector (5 regimes)...")
    detector = RegimeDetector(n_regimes=5, seed=42)
    detector.fit(df)

    # Override regime labels with rule-based logic on cluster centroids
    # so all 5 regimes are represented even in mostly-bullish data
    def _classify(row) -> str:
        if row["vix_level"] > 0.85:              return "high_vol_crisis"
        if row["vix_level"] > 0.70 and row["breadth_20d"] < -0.02: return "strong_trend_down"
        if row["nifty_adx"] > 0.3 and row["breadth_20d"] > 0.01: return "strong_trend_up"
        if row["vix_level"] < 0.40 and row["sector_dispersion"] < 0.012: return "low_vol_compression"
        return "choppy_reverting"

    labels = df.apply(_classify, axis=1)
    # Inject labels back into detector's regime_map via cluster→label vote
    assignments = detector.predict(df)
    cluster_ids = [a.regime_id for a in assignments]
    from collections import Counter
    for cluster_id in range(detector.n_regimes):
        mask = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
        if mask:
            votes = Counter(labels.iloc[mask])
            detector._regime_map[cluster_id] = votes.most_common(1)[0][0]

    # Label the full history
    assignments = detector.predict(df)
    df["regime"] = [a.regime_label for a in assignments]
    df["regime"] = df.apply(_classify, axis=1)  # use rule-based directly — more interpretable

    counts = df["regime"].value_counts()
    print("\nRegime distribution:")
    for regime, count in counts.items():
        exp = EXPOSURE_BY_REGIME.get(regime, 0.5)
        print(f"  {regime:<25} {count:>5} days ({count/len(df):.1%})  exposure={exp:.0%}")

    # Save detector + exposure map
    OUT.parent.mkdir(parents=True, exist_ok=True)
    detector.save(OUT)

    # Save rule-based regime history (used in backtest)
    import json
    (ROOT / "models/v8_intraday/exposure_by_regime.json").write_text(
        json.dumps(EXPOSURE_BY_REGIME, indent=2)
    )
    df["regime"].to_frame().to_parquet(ROOT / "models/v8_intraday/regime_history.parquet")

    print(f"\nSaved: {OUT}")

    # Quick validation: show recent regimes
    print("\nRecent regimes:")
    print(df["regime"].tail(10).to_string())


if __name__ == "__main__":
    main()
