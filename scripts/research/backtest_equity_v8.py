#!/usr/bin/env python
"""
V8 IntradayNet — End-to-end walk-forward backtest with V7 comparison.

Compares the V8 redesign against:
1. The current V7 system (loaded from existing backtest results)
2. Buy-and-hold baseline (Nifty 50)
3. Random baseline (statistical null hypothesis)

Usage:
    # Run full backtest
    python scripts/backtest_v8.py --universe nifty100 --model-dir models/v8

    # Quick test with synthetic data
    python scripts/backtest_v8.py --universe nifty50 --quick

    # Compare with V7 results
    python scripts/backtest_v8.py --universe nifty100 --compare-v7 path/to/v7/backtest_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root / "src"))

from equity.v8 import (
    V8Config,
    TargetConfig,
    PortfolioConfig,
    BacktestConfig,
    WalkForwardBacktest,
    BacktestMetrics,
    TradeRecord,
    RegimeDetector,
    MetaEnsemble,
    SignalModel,
    classify_tiers,
    compute_buy_hold_metrics,
    compute_random_baseline,
)
from equity.v8.dataset import build_barrier_dataset
from equity.universe import get_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V8 walk-forward backtest with V7 comparison",
    )
    parser.add_argument("--universe", type=str, default="nifty100",
                        choices=["nifty50", "nifty100", "nifty200", "nifty500"])
    parser.add_argument("--model-dir", type=str, default="models/v8")
    parser.add_argument("--output-dir", type=str, default="outputs/v8/backtest")
    parser.add_argument("--data-dir", type=str, default="data/nifty500")
    parser.add_argument("--compare-v7", type=str, default=None,
                        help="Path to V7 backtest results for comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with synthetic data")
    parser.add_argument("--target-pct", type=float, default=0.015)
    parser.add_argument("--stop-pct", type=float, default=0.010)
    parser.add_argument("--max-picks", type=int, default=5)
    parser.add_argument("--features", type=str, default="v8",
                        choices=["v8", "v7", "postopen", "swing", "swingf", "xsect"],
                        help="v8, v7, postopen, swing, swingf, xsect (cross-sectional factors)")
    parser.add_argument("--cutoff-min", type=int, default=30,
                        help="Post-open decision cutoff in minutes (postopen mode)")
    parser.add_argument("--horizon-days", type=int, default=10,
                        help="Swing holding horizon in trading days (swing mode)")
    parser.add_argument("--sim", type=str, default="barrier", choices=["barrier", "longshort", "xsect"],
                        help="Simulation: barrier (long-only), longshort, or xsect (beta-neutral)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expanding-window walk-forward splits (adaptive: folds with no data are skipped).
    train_splits = ((2021,), (2021, 2022), (2021, 2022, 2023), (2021, 2022, 2023, 2024))
    test_splits = ((2022,), (2023,), (2024,), (2025,))
    config = V8Config(
        target=TargetConfig(target_pct=args.target_pct, stop_pct=args.stop_pct),
        portfolio=PortfolioConfig(max_picks=args.max_picks),
        backtest=BacktestConfig(train_years=train_splits, test_years=test_splits),
    )

    # -----------------------------------------------------------------------
    # Quick test mode — generate synthetic trades for demonstration
    # -----------------------------------------------------------------------
    if args.quick:
        print("\nRunning quick test with synthetic data...")
        trades, metrics = _run_quick_backtest(config)
        _print_comparison(metrics, trades, None, args)
        return

    # -----------------------------------------------------------------------
    # Classify tiers + build barrier-labeled dataset (features + outcomes)
    # -----------------------------------------------------------------------
    print(f"\nClassifying tiers ({args.universe})...")
    tier_report = classify_tiers(args.data_dir, universe=args.universe, verbose=True)
    symbols = get_universe(args.universe)

    print(f"\nBuilding barrier-labeled dataset ({args.features} features)...")
    if args.features == "v7":
        from equity.v8.dataset import build_barrier_dataset_v7
        dataset = build_barrier_dataset_v7(symbols, args.data_dir, config, tier_report, verbose=True)
    elif args.features == "postopen":
        from equity.v8.dataset import build_postopen_dataset
        dataset = build_postopen_dataset(symbols, args.data_dir, config, tier_report,
                                         cutoff_min=args.cutoff_min, verbose=True)
    elif args.features == "swing":
        from equity.v8.dataset import build_swing_dataset_v7
        dataset = build_swing_dataset_v7(symbols, args.data_dir, config, tier_report,
                                         horizon_days=args.horizon_days, verbose=True)
    elif args.features == "swingf":
        from equity.v8.dataset import build_swing_factor_dataset
        dataset = build_swing_factor_dataset(symbols, args.data_dir, config, tier_report,
                                             horizon_days=args.horizon_days, verbose=True)
    elif args.features == "xsect":
        from equity.v8.dataset import build_xsect_factor_dataset
        dataset = build_xsect_factor_dataset(symbols, args.data_dir, config, tier_report,
                                             horizon_days=args.horizon_days, verbose=True)
    else:
        dataset = build_barrier_dataset(symbols, args.data_dir, config, tier_report, verbose=True)
    if dataset.empty:
        print(f"No dataset rows built from {args.data_dir}. Check minute CSV availability.")
        return
    print(f"Dataset: {len(dataset)} rows, "
          f"{dataset['date'].min().date()} → {dataset['date'].max().date()}, "
          f"LONG-target rate {(dataset['long_label'] == 1).mean():.2%}")

    # -----------------------------------------------------------------------
    # Run walk-forward backtest (retrains specialists each fold)
    # -----------------------------------------------------------------------
    print(f"\nRunning walk-forward backtest ({args.universe})...")
    v7_metrics = _load_v7_results(args.compare_v7) if args.compare_v7 else None

    wf = WalkForwardBacktest(config, tier_report, output_dir, dataset=dataset, sim=args.sim)
    trades, metrics = wf.run(verbose=True)

    _print_comparison(metrics, trades, v7_metrics, args)
    _save_comparison_report(metrics, trades, v7_metrics, output_dir)


def _run_quick_backtest(config: V8Config) -> tuple[list[TradeRecord], BacktestMetrics]:
    """Generate synthetic trades for quick testing."""
    np.random.seed(42)

    n_days = 252 * 3  # 3 years
    n_trades = 500
    win_rate = 0.48
    avg_win_pct = 1.8
    avg_loss_pct = -1.1

    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "BHARTIARTL",
               "ICICIBANK", "SBIN", "HINDUNILVR", "ITC", "KOTAKBANK"]

    regimes = ["strong_trend_up", "choppy_reverting", "low_vol_compression",
               "strong_trend_down", "high_vol_crisis"]
    regime_probs = [0.30, 0.25, 0.20, 0.15, 0.10]

    trades = []
    for i in range(n_trades):
        date = pd.Timestamp("2023-01-02") + pd.Timedelta(
            days=int(np.random.randint(0, n_days))
        )

        is_winner = np.random.random() < win_rate
        pnl_pct = avg_win_pct if is_winner else avg_loss_pct
        entry_price = np.random.uniform(100, 5000)
        regime = np.random.choice(regimes, p=regime_probs)

        trade = TradeRecord(
            symbol=np.random.choice(symbols),
            date=date,
            side=np.random.choice(["LONG", "SHORT"]),
            entry_price=entry_price,
            target_price=entry_price * (1.015 if is_winner else 1.0),
            stop_price=entry_price * 0.99,
            exit_price=entry_price * (1.0 + pnl_pct / 100),
            exit_reason="TARGET_HIT" if is_winner else "STOP_HIT",
            pnl_pct=pnl_pct / 100,
            pnl_absolute=100000 * pnl_pct / 100,
            regime=regime,
            confidence=np.random.uniform(0.55, 0.85),
            signal_model=np.random.choice(["momentum", "reversal", "breakout",
                                          "sentiment", "macro"]),
        )
        trades.append(trade)

    from equity.v8.walk_forward import _compute_metrics
    metrics = _compute_metrics(trades)

    return trades, metrics


def _load_v7_results(v7_path: str) -> dict | None:
    """Load V7 backtest results for comparison."""
    v7_path = Path(v7_path)
    if not v7_path.exists():
        print(f"V7 results not found: {v7_path}")
        return None

    try:
        if v7_path.suffix == ".json":
            return json.loads(v7_path.read_text())
        elif v7_path.suffix == ".parquet":
            df = pd.read_parquet(v7_path)
            return {"trades": len(df), "source": "parquet"}
    except Exception as e:
        print(f"Failed to load V7 results: {e}")
    return None


def _print_comparison(
    metrics: BacktestMetrics,
    trades: list[TradeRecord],
    v7_metrics: dict | None,
    args: argparse.Namespace,
) -> None:
    """Print V8 vs V7 comparison table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Compute baselines
    n_days = len(set(t.date.date() for t in trades)) if trades else 252
    bh_metrics = _compute_bh_baseline(n_days, args)
    rand_metrics = compute_random_baseline(
        n_trades=len(trades),
        n_days=max(n_days, 1),
        win_rate=0.48,
        avg_win_pct=0.018,
        avg_loss_pct=-0.011,
    )

    # Header
    headers = ["Metric", "V8", "V7", "Buy-Hold", "Random"]
    print(f"{headers[0]:<25} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>10}")
    print("-" * 65)

    rows = [
        ("Total Trades", metrics.total_trades, v7_metrics.get("trades", "N/A") if v7_metrics else "N/A", "N/A", "N/A"),
        ("Win Rate", f"{metrics.win_rate:.1%}", "N/A", "N/A", "N/A"),
        ("Profit Factor", f"{metrics.profit_factor:.2f}", "N/A", "N/A", "N/A"),
        ("Total PnL %", f"{metrics.total_pnl_pct:.2%}", "N/A", f"{bh_metrics.get('total_return', 0):.2%}", f"{rand_metrics['median_return']:.2%}"),
        ("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}", "N/A", f"{bh_metrics.get('sharpe', 0):.2f}", "N/A"),
        ("Max Drawdown", f"{metrics.max_drawdown:.2%}", "N/A", f"{bh_metrics.get('max_drawdown', 0):.2%}", "N/A"),
        ("Calmar Ratio", f"{metrics.calmar_ratio:.2f}", "N/A", f"{bh_metrics.get('calmar', 0):.2f}", "N/A"),
        ("Expectancy", f"{metrics.expectancy:.4f}", "N/A", "N/A", "N/A"),
    ]

    for row in rows:
        print(f"{row[0]:<25} {str(row[1]):>10} {str(row[2]):>10} {str(row[3]):>10} {str(row[4]):>10}")

    print("=" * 80)

    # Regime-stratified breakdown
    if metrics.regime_metrics:
        print("\nRegime-Stratified Performance:")
        print(f"{'Regime':<25} {'Trades':>8} {'Win Rate':>10} {'Avg PnL %':>10}")
        print("-" * 55)
        for regime, rmetrics in sorted(metrics.regime_metrics.items()):
            print(
                f"{regime:<25} {rmetrics['n_trades']:>8} "
                f"{rmetrics['win_rate']:>9.1%} {rmetrics['avg_pnl_pct']:>9.2%}"
            )
        print("=" * 80)


def _compute_bh_baseline(n_days: int, args: argparse.Namespace) -> dict:
    """Compute buy-and-hold baseline."""
    nifty_path = Path("market_data_cache/nifty50.csv")

    if nifty_path.exists():
        try:
            nifty = pd.read_csv(nifty_path, index_col=0, parse_dates=True)
            if "close" in nifty.columns:
                daily_rets = nifty["close"].pct_change().dropna()
                if len(daily_rets) > 20:
                    return compute_buy_hold_metrics(daily_rets)
        except Exception:
            pass

    # Fallback: synthetic
    np.random.seed(42)
    daily_rets = pd.Series(
        np.random.normal(0.0004, 0.012, n_days),
        index=pd.date_range("2023-01-01", periods=n_days, freq="B"),
    )
    return compute_buy_hold_metrics(daily_rets)


def _save_comparison_report(
    metrics: BacktestMetrics,
    trades: list[TradeRecord],
    v7_metrics: dict | None,
    output_dir: Path,
) -> None:
    """Save comprehensive comparison report."""
    report = {
        "timestamp": str(pd.Timestamp.now()),
        "v8": {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "total_pnl_pct": metrics.total_pnl_pct,
            "max_drawdown": metrics.max_drawdown,
            "sharpe_ratio": metrics.sharpe_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "expectancy": metrics.expectancy,
            "regime_metrics": metrics.regime_metrics,
            "tier_metrics": metrics.tier_metrics,
            "signal_metrics": metrics.signal_metrics,
        },
        "v7": v7_metrics,
    }

    report_path = output_dir / "comparison_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nComparison report saved to {report_path}")


if __name__ == "__main__":
    main()
