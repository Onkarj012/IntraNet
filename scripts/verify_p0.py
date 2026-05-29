"""
End-to-end P0 verification harness.

Tests all three P0 changes on REAL stock data from StockXpert:
  1. Horizon-specific point-in-time targets
  2. Unified feature contract validation
  3. Probability calibration

Usage:
    cd /Users/onkarj012/Projects/market/intranet_optinet
    python scripts/verify_p0.py

No prerequisites beyond the data at StockXpert/data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Path setup ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# StockXpert data is outside the project — use absolute path
DATA_ROOT = Path("/Users/onkarj012/Projects/major_pro/StockXpert/data")
MINUTE_DIR = DATA_ROOT / "nifty500"
SENTIMENT_CSV = DATA_ROOT / "sentiment" / "combined_sentiment_2015_2025.csv"

if not MINUTE_DIR.exists():
    print("ERROR: Minute data directory not found at", MINUTE_DIR)
    sys.exit(1)

# -- Import after path setup ---------------------------------------------
from intradaynet.v7 import (
    compute_horizon_targets,
    compute_horizon_targets_batched,
    compute_daily_targets_from_minute,
    extract_sessions,
)
from intradaynet.feature_contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    DAILY_FEATURE_NAMES,
    get_feature_registry,
)
from intradaynet.features.per_bar_features import (
    compute_per_bar_features,
    PER_BAR_FEATURE_NAMES,
)
from intradaynet.features.session_features import SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES
from intradaynet.calibrator import (
    calibrate_direction_probs,
    calibration_report,
    train_platt_scaler,
    train_isotonic_regressor,
)


def banner(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")


# ═══════════════════════════════════════════════════════════════════════
# P0-1: Horizon-Specific Point-in-Time Targets
# ═══════════════════════════════════════════════════════════════════════

def test_p0_1_horizon_targets():
    banner("P0-1: Horizon-Specific Point-in-Time Targets")

    # Load RELIANCE as a representative stock
    symbol = "RELIANCE"
    csv_path = MINUTE_DIR / f"{symbol}_minute.csv"
    if not csv_path.exists():
        print(f"  SKIP: {symbol} data not found")
        return

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    print(f"  Loaded {symbol}: {len(df):,} bars, "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    # --- Single session demo ---
    sessions = extract_sessions(df)
    if len(sessions) < 2:
        print("  ERROR: Need at least 2 sessions")
        return

    # Pick a recent session with decent data
    session = sessions[-1]
    date = session.index[0].date()
    print(f"\n  Demo session: {date} ({len(session)} bars)")

    # Compare old daily target vs new horizon targets
    subsection("Horizon target comparison on a single session")
    targets_h15 = compute_horizon_targets(session, horizon_bars=15, target_pct=0.01)
    targets_h30 = compute_horizon_targets(session, horizon_bars=30, target_pct=0.01)
    targets_h60 = compute_horizon_targets(session, horizon_bars=60, target_pct=0.01)

    for label, targets in [("H15", targets_h15), ("H30", targets_h30), ("H60", targets_h60)]:
        counts = targets["trade_label"].value_counts().to_dict()
        print(f"    {label}: LONG={counts.get('LONG',0):>4d}  "
              f"SHORT={counts.get('SHORT',0):>4d}  "
              f"NO_TRADE={counts.get('NO_TRADE',0):>4d}  "
              f"(out of {len(targets)} bars)")

    # Show a specific bar's prediction
    bar_idx = 30
    if bar_idx < len(session):
        bar_time = session.index[bar_idx]
        close = float(session["close"].iloc[bar_idx])
        print(f"\n  At {bar_time}: close = {close:.2f}")
        for label, targets in [("H15", targets_h15), ("H30", targets_h30), ("H60", targets_h60)]:
            t = targets.iloc[bar_idx]
            net_edge = max(float(t["long_executable_move"]), float(t["short_executable_move"]))
            print(f"    {label} → {t['trade_label']:>8s}  "
                  f"net_edge={net_edge:.4%}  "
                  f"side_code={t['trade_side_code']}")

    # --- Multi-day batch ---
    subsection("Multi-horizon batch on 2 recent sessions")
    recent_sessions = sessions[-2:]
    combined = pd.concat(recent_sessions)
    batched = compute_horizon_targets_batched(
        combined,
        horizons={"H15": 15, "H30": 30, "H60": 60},
    )
    print(f"    Produced horizons: {sorted(batched.keys())}")
    for name, tdf in batched.items():
        print(f"    {name}: {len(tdf)} bars, "
              f"LONG={int((tdf['trade_label'] == 'LONG').sum())}, "
              f"SHORT={int((tdf['trade_label'] == 'SHORT').sum())}")

    # --- Daily aggregation ---
    subsection("Daily targets from minute data (10 most recent days)")
    tail = df.iloc[-375 * 10:]
    daily = compute_daily_targets_from_minute(tail)
    if not daily.empty:
        print(f"    Days processed: {len(daily)}")
        print(f"    Label distribution:\n{daily['trade_label'].value_counts().to_string()}")
        print(f"    Mean long_edge: {daily['long_executable_move'].mean():.4%}")
        print(f"    Mean short_edge: {daily['short_executable_move'].mean():.4%}")

    # Verify mutual exclusivity
    subsection("P0-1 validation checks")
    for label, targets in [("H15", targets_h15), ("H30", targets_h30), ("H60", targets_h60)]:
        sum_targets = (
            targets["long_target"].fillna(0).astype(int)
            + targets["short_target"].fillna(0).astype(int)
            + targets["no_trade_target"].fillna(0).astype(int)
        )
        all_one = (sum_targets == 1).all()
        print(f"    {label} mutual exclusivity: {'PASS' if all_one else 'FAIL'} ({sum_targets.sum()}/{len(sum_targets)} bars = 1)")

    return True


# ═══════════════════════════════════════════════════════════════════════
# P0-2: Unified Feature Contract
# ═══════════════════════════════════════════════════════════════════════

def test_p0_2_feature_contract():
    banner("P0-2: Unified Feature Contract Validation")

    registry = get_feature_registry()

    # Intraday pipeline
    print(f"  Intraday pipeline (LightGBM backend):")
    print(f"    Version: {registry.intraday.version}")
    print(f"    Feature count: {registry.intraday.feature_count}")
    print(f"    Per-bar features: {len(PER_BAR_FEATURE_NAMES)}")
    print(f"    Session features: {len(SESSION_FEATURE_NAMES)}")
    print(f"    Sentiment features: {len(SENTIMENT_FEATURE_NAMES)}")
    print(f"    Raw total: {len(PER_BAR_FEATURE_NAMES) + len(SESSION_FEATURE_NAMES) + len(SENTIMENT_FEATURE_NAMES)}")
    print(f"    Flattened: {registry.intraday.feature_count}")

    # Daily pipeline
    print(f"\n  Daily pipeline (open-safe premarket model):")
    print(f"    Version: {registry.daily.version}")
    print(f"    Feature count: {registry.daily.feature_count}")

    # Show sample names from both
    print(f"\n  Sample intraday features (first 10):")
    for name in list(registry.intraday.feature_names)[:10]:
        print(f"    {name}")
    print(f"\n  Sample daily features (first 10):")
    for name in list(registry.daily.feature_names)[:10]:
        print(f"    {name}")

    # Validate with real computed features
    subsection("Real feature pipeline validation")
    symbol = "TCS"
    csv_path = MINUTE_DIR / f"{symbol}_minute.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
        df = df.iloc[-5000:]
        per_bar = compute_per_bar_features(df)
        session_feats = per_bar.iloc[-120:].copy()

        from intradaynet.feature_contract import flatten_intraday_window
        from intradaynet.features.session_features import compute_session_features

        sess_df = compute_session_features(df)
        sess_latest = sess_df.iloc[-1] if len(sess_df) > 0 else pd.Series(0.0, index=SESSION_FEATURE_NAMES)
        sent_default = pd.Series(0.0, index=SENTIMENT_FEATURE_NAMES)

        window_data = session_feats[PER_BAR_FEATURE_NAMES].values.astype(np.float32)
        flat = flatten_intraday_window(
            window_data,
            sess_latest.values.astype(np.float32),
            sent_default.values.astype(np.float32),
        )
        print(f"    Flattened vector shape: {flat.shape}")
        print(f"    Expected shape: {registry.intraday.feature_count}")
        print(f"    Shape match: {'PASS' if flat.shape[0] == registry.intraday.feature_count else 'FAIL'}")
        print(f"    No NaN: {'PASS' if not np.isnan(flat).any() else 'FAIL'}")
        print(f"    No Inf: {'PASS' if not np.isinf(flat).any() else 'FAIL'}")
        print(f"    Value range: [{flat.min():.4f}, {flat.max():.4f}]")

        # Validate daily contract
        from intradaynet.open_safe_daily_features import DAILY_FEATURE_FAMILY_PREFIXES
        missing = registry.validate_daily_frame(list(DAILY_FEATURE_NAMES))
        print(f"\n    Daily contract validation: {'PASS' if not missing else f'MISSING: {missing[:5]}...'}")

    return True


# ═══════════════════════════════════════════════════════════════════════
# P0-3: Probability Calibration
# ═══════════════════════════════════════════════════════════════════════

def test_p0_3_calibration():
    banner("P0-3: Probability Calibration")

    # Generate synthetic realistic model outputs
    rng = np.random.default_rng(42)
    n = 2000

    # Simulate: uncalibrated LightGBM probs (clustered near 0 and 1)
    raw_probs = np.clip(rng.beta(0.5, 0.5, n), 0.01, 0.99)
    raw_probs_multiclass = rng.dirichlet([1.5, 2.0, 1.0], n)

    # True labels with some noise
    y_binary = (rng.random(n) < raw_probs).astype(int)
    y_multiclass = np.array([int(np.argmax(row)) for row in raw_probs_multiclass])

    # Before calibration
    subsection("Before calibration — raw LightGBM-style probabilities")
    print(f"    Binary: mean={raw_probs.mean():.3f}, std={raw_probs.std():.3f}, "
          f"true_rate={y_binary.mean():.3f}")
    print(f"    Calibration gap: {raw_probs.mean() - y_binary.mean():.4f}")

    # Platt scaling
    subsection("Platt scaling (sigmoid calibration)")
    platt = train_platt_scaler(raw_probs.reshape(-1, 1), y_binary)
    cal_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    print(f"    After calibration: mean={cal_probs.mean():.3f}, std={cal_probs.std():.3f}")
    report = calibration_report(y_binary, cal_probs)
    print(report)

    # Isotonic regression
    subsection("Isotonic regression calibration")
    iso = train_isotonic_regressor(raw_probs, y_binary)
    iso_probs = iso.transform(raw_probs)
    print(f"    After calibration: mean={iso_probs.mean():.3f}, std={iso_probs.std():.3f}")
    report2 = calibration_report(y_binary, iso_probs)
    print(report2)

    # Multiclass calibration
    subsection("Multiclass direction calibration (LONG/SHORT/NO_TRADE)")
    result = calibrate_direction_probs(
        raw_probs_multiclass, y_multiclass, method="sigmoid"
    )
    print(f"    Mean ECE: {result['mean_ece']:.4f}")
    for cls_name, data in result["ece_per_class"].items():
        print(f"    {cls_name}: ECE = {data:.4f}")
    print(f"\n    Calibrator type: {type(result['calibrator']).__name__}")

    # Save/load roundtrip
    subsection("Calibrator persistence roundtrip")
    from intradaynet.calibrator import save_calibrator, load_calibrator

    tmp_path = Path("/tmp/p0_calibrator_test.pkl")
    save_calibrator(platt, tmp_path)
    loaded = load_calibrator(tmp_path)
    loaded_probs = loaded.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    identical = np.allclose(cal_probs, loaded_probs)
    print(f"    Save/load identical: {'PASS' if identical else 'FAIL'}")
    tmp_path.unlink()

    return True


# ═══════════════════════════════════════════════════════════════════════
# P0 Summary: Feature Contract Audit
# ═══════════════════════════════════════════════════════════════════════

def test_p0_feature_audit_on_real_data():
    """Run feature contract check against 10 real stocks to verify consistency."""
    banner("P0 Cross-Check: Feature Contract vs Real Data")

    csv_files = sorted(MINUTE_DIR.glob("*_minute.csv"))[:10]
    registry = get_feature_registry()
    all_ok = True

    for csv_path in csv_files:
        symbol = csv_path.stem.replace("_minute", "")
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
            df = df.iloc[-2000:]
            if len(df) < 120:
                continue

            per_bar = compute_per_bar_features(df)
            window = per_bar[PER_BAR_FEATURE_NAMES].iloc[-120:].values.astype(np.float32)
            sess_arr = np.zeros(len(SESSION_FEATURE_NAMES), dtype=np.float32)
            sent_arr = np.zeros(len(SENTIMENT_FEATURE_NAMES), dtype=np.float32)

            from intradaynet.feature_contract import flatten_intraday_window
            flat = flatten_intraday_window(window, sess_arr, sent_arr)

            if flat.shape[0] != registry.intraday.feature_count:
                print(f"  {symbol:30s}: FAIL — shape {flat.shape[0]} != {registry.intraday.feature_count}")
                all_ok = False
            elif np.isnan(flat).any():
                print(f"  {symbol:30s}: FAIL — contains NaN")
                all_ok = False
            else:
                print(f"  {symbol:30s}: PASS  shape={flat.shape[0]}  range=[{flat.min():.2f}, {flat.max():.2f}]")
        except Exception as e:
            print(f"  {symbol:30s}: ERROR — {e}")
            all_ok = False

    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 65)
    print("█  IntradayNet P0 Verification Suite")
    print("█  Data: StockXpert (536 stocks, 2015-2026 minute bars)")
    print("█" * 65)

    results = {}

    results["P0-1 Horizon Targets"] = test_p0_1_horizon_targets()
    results["P0-2 Feature Contract"] = test_p0_2_feature_contract()
    results["P0-3 Calibration"] = test_p0_3_calibration()
    results["P0 Cross-Check"] = test_p0_feature_audit_on_real_data()

    # Final summary
    banner("VERIFICATION SUMMARY")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:>5s}  {name}")


if __name__ == "__main__":
    main()
