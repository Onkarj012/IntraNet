"""
Walk-forward backtest scaffold for V8.

Implements the walk-forward methodology:
- Train on years T₁, test on year T₂
- Roll forward: train on T₁+T₂, test on T₃
- Continue through all test years

Design supports:
1. Barrier target computation
2. Curve embedding generation (with optional yearly retraining)
3. Signal model training/inference
4. Portfolio construction and trade simulation
5. Realistic cost modeling (slippage by tier, NSE fees)
6. Regime-stratified performance metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import V8Config
from .universe_tiers import DataTier, UniverseTierReport


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Complete trade lifecycle record."""
    symbol: str
    date: pd.Timestamp
    side: str  # LONG or SHORT
    entry_price: float
    target_price: float
    stop_price: float
    exit_price: float
    exit_reason: str  # TARGET_HIT | STOP_HIT | CLOSE_EOD | TRAILING_STOP
    exit_time: Optional[pd.Timestamp] = None
    pnl_pct: float = 0.0
    pnl_absolute: float = 0.0
    position_size: float = 100000.0
    tier: str = "tier_1"
    slippage_pct: float = 0.0
    costs: float = 0.0
    regime: str = "unknown"
    confidence: float = 0.0
    signal_model: str = ""

    @property
    def is_winner(self) -> bool:
        return self.pnl_pct > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": str(self.date.date()),
            "side": self.side,
            "entry": self.entry_price,
            "target": self.target_price,
            "stop": self.stop_price,
            "exit": self.exit_price,
            "exit_reason": self.exit_reason,
            "exit_time": str(self.exit_time) if self.exit_time else None,
            "pnl_pct": round(self.pnl_pct, 6),
            "pnl_absolute": round(self.pnl_absolute, 2),
            "position_size": self.position_size,
            "tier": self.tier,
            "slippage_pct": self.slippage_pct,
            "costs": round(self.costs, 2),
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "signal_model": self.signal_model,
        }


# ---------------------------------------------------------------------------
# Backtest metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0
    trades_per_day: float = 0.0

    # Stratified by regime
    regime_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Stratified by tier
    tier_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Stratified by signal model
    signal_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "BACKTEST SUMMARY",
            "=" * 50,
            f"Total Trades:     {self.total_trades}",
            f"Win Rate:         {self.win_rate:.1%}",
            f"Profit Factor:    {self.profit_factor:.2f}",
            f"Total PnL %:      {self.total_pnl_pct:.2%}",
            f"Avg Winner:       {self.avg_winner_pct:.2%}",
            f"Avg Loser:        {self.avg_loser_pct:.2%}",
            f"Max Drawdown:     {self.max_drawdown:.2%}",
            f"Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"Calmar Ratio:     {self.calmar_ratio:.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-forward orchestrator
# ---------------------------------------------------------------------------

class WalkForwardBacktest:
    """
    Walk-forward backtest orchestrator for V8.

    Runs the full pipeline for each train/test split:
    1. Build feature + target datasets for train/test periods
    2. Optionally retrain curve embeddings
    3. Train 5 specialist signal models
    4. Calibrate probabilities
    5. Run meta-ensemble inference
    6. Construct portfolio with diversification
    7. Simulate trades with realistic costs
    8. Compute stratified performance metrics

    Usage:
        wf = WalkForwardBacktest(config, tier_report, output_dir)
        trades, metrics = wf.run()
    """

    def __init__(
        self,
        config: V8Config,
        tier_report: UniverseTierReport,
        output_dir: str | Path,
        *,
        baseline_trades: pd.DataFrame | None = None,
    ):
        self.config = config
        self.tier_report = tier_report
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_trades = baseline_trades

        # Internal state
        self._trades: list[TradeRecord] = []
        self._fold_metrics: list[dict] = []
        self._log: list[str] = []

    @property
    def train_test_splits(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """All train/test year splits."""
        bc = self.config.backtest
        return list(zip(bc.train_years, bc.test_years))

    def run(self, verbose: bool = True) -> tuple[list[TradeRecord], BacktestMetrics]:
        """
        Execute full walk-forward backtest.

        Returns all trades and aggregated metrics.
        """
        self._trades = []
        self._fold_metrics = []
        self._log = []

        for fold_idx, (train_years, test_years) in enumerate(self.train_test_splits):
            if verbose:
                self._log_event(f"Fold {fold_idx + 1}/{len(self.train_test_splits)}: "
                                f"train={train_years[0]}-{train_years[-1]}, "
                                f"test={test_years[0]}-{test_years[-1]}")

            fold_trades = self._run_fold(fold_idx, train_years, test_years, verbose)
            self._trades.extend(fold_trades)

            fold_metrics = _compute_metrics(fold_trades)
            self._fold_metrics.append({
                "fold": fold_idx,
                "train_years": list(train_years),
                "test_years": list(test_years),
                "n_trades": len(fold_trades),
                "win_rate": fold_metrics.win_rate,
                "profit_factor": fold_metrics.profit_factor,
                "total_pnl_pct": fold_metrics.total_pnl_pct,
                "sharpe": fold_metrics.sharpe_ratio,
            })

        aggregate_metrics = _compute_metrics(self._trades)

        if verbose:
            self._log_event(aggregate_metrics.summary())

        self._save_results(aggregate_metrics, self._trades)
        return self._trades, aggregate_metrics

    def _run_fold(
        self,
        fold_idx: int,
        train_years: tuple[int, ...],
        test_years: tuple[int, ...],
        verbose: bool,
    ) -> list[TradeRecord]:
        """Run a single walk-forward fold."""
        # Placeholder — full implementation in Phase 3-4
        # For now, returns empty list
        self._log_event(f"  Fold {fold_idx} placeholder — signal models not yet implemented")
        return []

    def _log_event(self, msg: str) -> None:
        self._log.append(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

    def _save_results(
        self,
        metrics: BacktestMetrics,
        trades: list[TradeRecord],
    ) -> None:
        """Save backtest results to disk."""
        # Save trades
        trades_df = pd.DataFrame([t.to_dict() for t in trades])
        trades_path = self.output_dir / "trades.parquet"
        trades_df.to_parquet(trades_path, index=False)

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        metrics_dict = {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "total_pnl_pct": metrics.total_pnl_pct,
            "avg_winner_pct": metrics.avg_winner_pct,
            "avg_loser_pct": metrics.avg_loser_pct,
            "max_drawdown": metrics.max_drawdown,
            "sharpe_ratio": metrics.sharpe_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "expectancy": metrics.expectancy,
            "regime_metrics": metrics.regime_metrics,
            "tier_metrics": metrics.tier_metrics,
            "signal_metrics": metrics.signal_metrics,
            "fold_metrics": self._fold_metrics,
        }
        metrics_path.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

        # Save log
        log_path = self.output_dir / "backtest.log"
        log_path.write_text("\n".join(self._log), encoding="utf-8")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(trades: list[TradeRecord]) -> BacktestMetrics:
    """Compute comprehensive backtest metrics from trade records."""
    if not trades:
        return BacktestMetrics()

    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]
    win_rate = len(winners) / len(trades) if trades else 0

    gross_profits = sum(t.pnl_absolute for t in winners) if winners else 0
    gross_losses = abs(sum(t.pnl_absolute for t in losers)) if losers else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

    total_pnl = sum(t.pnl_absolute for t in trades)
    total_pnl_pct = sum(t.pnl_pct for t in trades)

    avg_winner_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loser_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0

    daily_pnl = _compute_daily_pnl(trades)
    max_drawdown = _compute_max_drawdown(daily_pnl)
    sharpe = _compute_sharpe(daily_pnl)
    calmar = abs(total_pnl / max_drawdown) if max_drawdown > 0 else 0

    expectancy = (win_rate * avg_winner_pct) - ((1 - win_rate) * abs(avg_loser_pct))

    dates = sorted({t.date.date() for t in trades})
    trading_days = len(dates)
    trades_per_day = len(trades) / max(trading_days, 1)

    # Stratified metrics
    regime_metrics = _compute_stratified_metrics(trades, "regime")
    tier_metrics = _compute_stratified_metrics(trades, "tier")
    signal_metrics = _compute_stratified_metrics(trades, "signal_model")

    return BacktestMetrics(
        total_trades=len(trades),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        avg_winner_pct=avg_winner_pct,
        avg_loser_pct=avg_loser_pct,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        expectancy=expectancy,
        trades_per_day=trades_per_day,
        regime_metrics=regime_metrics,
        tier_metrics=tier_metrics,
        signal_metrics=signal_metrics,
    )


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compute_buy_hold_metrics(
    daily_returns: pd.Series,
    initial_capital: float = 1000000.0,
) -> dict[str, float]:
    """
    Buy-and-hold baseline metrics.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns (pct) for the benchmark (e.g., Nifty 50).
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    dict
        Buy-and-hold performance metrics.
    """
    if daily_returns.empty:
        return {}

    equity = (1.0 + daily_returns).cumprod() * initial_capital
    total_return = (equity.iloc[-1] / initial_capital) - 1.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    annualized_return = (1.0 + total_return) ** (252.0 / len(daily_returns)) - 1.0
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0
    calmar = annualized_return / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def compute_random_baseline(
    n_trades: int,
    n_days: int,
    win_rate: float = 0.50,
    avg_win_pct: float = 0.015,
    avg_loss_pct: float = 0.010,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Random baseline: what if you randomly picked LONG/SHORT?

    Simulates n_simulations portfolios of random trades and returns
    summary statistics (median, p5, p95).
    """
    rng = np.random.RandomState(seed)
    pnl_curves = []

    for _ in range(n_simulations):
        daily_pnl = rng.choice(
            [avg_win_pct, -avg_loss_pct],
            size=n_trades,
            p=[win_rate, 1 - win_rate],
        )
        dates = rng.choice(n_days, size=n_trades, replace=True)
        pnl_by_date = pd.Series(daily_pnl, index=dates).groupby(level=0).sum()

        equity = (1.0 + pnl_by_date.reindex(range(n_days), fill_value=0.0)).cumprod()
        pnl_curves.append(equity)

    final_returns = [c.iloc[-1] - 1.0 for c in pnl_curves]

    return {
        "median_return": float(np.median(final_returns)),
        "p5_return": float(np.percentile(final_returns, 5)),
        "p95_return": float(np.percentile(final_returns, 95)),
        "positive_probability": float(np.mean([r > 0 for r in final_returns])),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_daily_pnl(trades: list[TradeRecord]) -> pd.Series:
    """Aggregate trades into daily PnL series."""
    if not trades:
        return pd.Series(dtype=float)

    pnl_records = [
        {"date": pd.Timestamp(t.date.date()), "pnl": t.pnl_pct}
        for t in trades
    ]
    df = pd.DataFrame(pnl_records)
    return df.groupby("date")["pnl"].sum().sort_index()


def _compute_max_drawdown(daily_pnl: pd.Series) -> float:
    """Compute maximum drawdown from daily PnL series."""
    if daily_pnl.empty:
        return 0.0

    equity = (1.0 + daily_pnl).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(abs(drawdown.min()))


def _compute_sharpe(daily_pnl: pd.Series, risk_free: float = 0.06) -> float:
    """Annualized Sharpe ratio from daily PnL."""
    if daily_pnl.empty:
        return 0.0

    excess = daily_pnl - risk_free / 252.0
    mean_excess = excess.mean()
    std_excess = excess.std()

    if std_excess == 0:
        return 0.0

    return float(mean_excess / std_excess * np.sqrt(252))


def _compute_stratified_metrics(
    trades: list[TradeRecord],
    attribute: str,
) -> dict[str, dict[str, float]]:
    """Compute win rate and average PnL by some attribute (regime, tier, signal_model)."""
    if not trades:
        return {}

    unique_values = {getattr(t, attribute, "unknown") for t in trades}
    stratified = {}

    for value in unique_values:
        group = [t for t in trades if getattr(t, attribute, "unknown") == value]
        if not group:
            continue
        wins = sum(1 for t in group if t.is_winner)
        stratified[str(value)] = {
            "n_trades": len(group),
            "win_rate": wins / len(group),
            "avg_pnl_pct": float(np.mean([t.pnl_pct for t in group])),
            "total_pnl_pct": float(sum(t.pnl_pct for t in group)),
        }

    return stratified
