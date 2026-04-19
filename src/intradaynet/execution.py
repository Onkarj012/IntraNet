"""
Execution Infrastructure for IntradayNet v3.0 - Phase 5

5.1: Paper Trade Logger with 30-day validation gate
5.2: Automated monthly retraining pipeline

Usage:
    from intradaynet.execution.paper_trading import PaperTradeLogger
    
    # Initialize logger
    logger = PaperTradeLogger(
        output_dir="paper_trades",
        validation_days=20,  # 20 trading days
    )
    
    # Log predictions each morning
    logger.log_predictions(
        date="2025-01-15",
        picks=predicted_picks,
        regime=current_regime,
    )
    
    # Log actual results after market close
    logger.log_results(
        date="2025-01-15",
        actual_openings=market_data,
        actual_highs=...,
        actual_lows=...,
    )
    
    # Check validation status
    status = logger.get_validation_status()
    # Returns whether paper trade metrics are within 5% of backtest
    
    # Gate live trading
    if status['can_go_live']:
        print("Ready for live trading!")
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger("intradaynet.execution")


@dataclass
class TradeRecord:
    """Record of a single paper trade."""
    trade_date: str
    symbol: str
    side: str
    predicted_entry: float
    predicted_target: float
    predicted_stop: float
    predicted_horizon: str
    
    # Actual results (filled after market close)
    actual_open: Optional[float] = None
    actual_high: Optional[float] = None
    actual_low: Optional[float] = None
    actual_close: Optional[float] = None
    
    # Execution quality metrics
    entry_slippage_pct: Optional[float] = None  # (actual_open - predicted) / predicted
    target_fill_rate: Optional[float] = None  # % of target achieved
    stop_hit_rate: Optional[float] = None  # % of stop hit
    
    # Outcome
    outcome: Optional[str] = None  # target_hit, stop_hit, time_exit, pending
    gross_return_pct: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationMetrics:
    """Metrics for validation gate."""
    n_trading_days: int
    n_predictions: int
    n_trades: int
    
    # Predicted vs actual metrics
    predicted_win_rate: float
    actual_win_rate: float
    win_rate_diff: float
    
    predicted_avg_return: float
    actual_avg_return: float
    return_diff: float
    
    fill_rate: float  # % of predicted entries achieved within 0.1%
    avg_slippage_pct: float
    
    # Validation gate
    within_tolerance: bool
    tolerance_pct: float = 5.0
    
    def can_go_live(self) -> bool:
        """Check if metrics are within tolerance for live trading."""
        return self.within_tolerance and self.n_trading_days >= 20


class PaperTradeLogger:
    """
    5.1: Paper trade logger with 30-day validation gate.
    
    Tracks every prediction and compares to actual market results.
    Computes execution quality and validates backtest assumptions.
    """
    
    def __init__(
        self,
        output_dir: str = "paper_trades",
        validation_days: int = 20,
        entry_tolerance_pct: float = 0.1,  # 0.1% entry tolerance
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.validation_days = validation_days
        self.entry_tolerance_pct = entry_tolerance_pct
        
        self.trade_records: List[TradeRecord] = []
        self.daily_summaries: Dict[str, Dict] = {}
        
        # Load existing records if any
        self._load_existing()
    
    def _load_existing(self):
        """Load existing paper trade records."""
        records_file = self.output_dir / "trade_records.json"
        if records_file.exists():
            with open(records_file) as f:
                data = json.load(f)
                for record_data in data:
                    self.trade_records.append(TradeRecord(**record_data))
            logger.info(f"Loaded {len(self.trade_records)} existing trade records")
    
    def log_predictions(
        self,
        date: str,
        picks: List[Dict],
        regime: str,
        model_version: str = "v3.0",
    ):
        """
        Log predictions at market open (9:00-9:15 AM).
        
        Args:
            date: Trading date
            picks: List of predicted picks with entry/target/stop
            regime: Current market regime
            model_version: Model version identifier
        """
        timestamp = datetime.now().isoformat()
        
        daily_log = {
            'date': date,
            'timestamp': timestamp,
            'regime': regime,
            'model_version': model_version,
            'n_picks': len(picks),
            'picks': [],
        }
        
        for pick in picks:
            record = TradeRecord(
                trade_date=date,
                symbol=pick['symbol'],
                side=pick['side'],
                predicted_entry=pick['entry_reference'],
                predicted_target=pick['target'],
                predicted_stop=pick['stop_loss'],
                predicted_horizon=pick.get('horizon', 'H60'),
            )
            
            self.trade_records.append(record)
            daily_log['picks'].append({
                'symbol': pick['symbol'],
                'side': pick['side'],
                'predicted_entry': pick['entry_reference'],
                'predicted_target': pick['target'],
                'predicted_stop': pick['stop_loss'],
                'confidence': pick.get('confidence', 0),
            })
        
        # Save daily log
        daily_file = self.output_dir / f"predictions_{date}.json"
        with open(daily_file, 'w') as f:
            json.dump(daily_log, f, indent=2)
        
        logger.info(f"Logged {len(picks)} predictions for {date}")
    
    def log_results(
        self,
        date: str,
        actual_openings: Dict[str, float],
        actual_highs: Dict[str, float],
        actual_lows: Dict[str, float],
        actual_closes: Dict[str, float],
    ):
        """
        Log actual market results after market close (3:30+ PM).
        
        Computes execution quality metrics.
        """
        updated_count = 0
        
        for record in self.trade_records:
            if record.trade_date != date:
                continue
            
            symbol = record.symbol
            
            if symbol not in actual_openings:
                continue
            
            # Fill actual prices
            record.actual_open = actual_openings[symbol]
            record.actual_high = actual_highs.get(symbol)
            record.actual_low = actual_lows.get(symbol)
            record.actual_close = actual_closes.get(symbol)
            
            # Compute entry slippage
            record.entry_slippage_pct = (
                (record.actual_open - record.predicted_entry) / record.predicted_entry * 100
            )
            
            # Determine outcome
            if record.side == "LONG":
                if record.actual_high and record.actual_high >= record.predicted_target:
                    record.outcome = "target_hit"
                    record.gross_return_pct = (
                        (record.predicted_target - record.actual_open) / record.actual_open * 100
                    )
                elif record.actual_low and record.actual_low <= record.predicted_stop:
                    record.outcome = "stop_hit"
                    record.gross_return_pct = (
                        (record.predicted_stop - record.actual_open) / record.actual_open * 100
                    )
                else:
                    record.outcome = "time_exit"
                    if record.actual_close:
                        record.gross_return_pct = (
                            (record.actual_close - record.actual_open) / record.actual_open * 100
                        )
            else:  # SHORT
                if record.actual_low and record.actual_low <= record.predicted_target:
                    record.outcome = "target_hit"
                    record.gross_return_pct = (
                        (record.actual_open - record.predicted_target) / record.actual_open * 100
                    )
                elif record.actual_high and record.actual_high >= record.predicted_stop:
                    record.outcome = "stop_hit"
                    record.gross_return_pct = (
                        (record.actual_open - record.predicted_stop) / record.actual_open * 100
                    )
                else:
                    record.outcome = "time_exit"
                    if record.actual_close:
                        record.gross_return_pct = (
                            (record.actual_open - record.actual_close) / record.actual_open * 100
                        )
            
            updated_count += 1
        
        # Save updated records
        self._save_records()
        
        logger.info(f"Updated {updated_count} records with actual results for {date}")
    
    def _save_records(self):
        """Save all trade records to disk."""
        records_file = self.output_dir / "trade_records.json"
        with open(records_file, 'w') as f:
            json.dump([r.to_dict() for r in self.trade_records], f, indent=2, default=str)
    
    def compute_execution_metrics(self) -> Dict[str, Any]:
        """Compute execution quality metrics."""
        completed_trades = [r for r in self.trade_records if r.outcome is not None]
        
        if not completed_trades:
            return {'error': 'no_completed_trades'}
        
        # Fill rate: % of entries achieved within tolerance
        fills = [
            r for r in completed_trades
            if abs(r.entry_slippage_pct or 100) <= self.entry_tolerance_pct
        ]
        fill_rate = len(fills) / len(completed_trades) * 100
        
        # Average slippage
        slippages = [r.entry_slippage_pct for r in completed_trades if r.entry_slippage_pct is not None]
        avg_slippage = np.mean(slippages) if slippages else 0
        
        # Outcome distribution
        outcomes = {}
        for r in completed_trades:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        
        # Win rate
        wins = outcomes.get('target_hit', 0)
        total_completed = len(completed_trades)
        win_rate = wins / total_completed * 100 if total_completed > 0 else 0
        
        # Average return
        returns = [r.gross_return_pct for r in completed_trades if r.gross_return_pct is not None]
        avg_return = np.mean(returns) if returns else 0
        
        return {
            'n_completed_trades': len(completed_trades),
            'fill_rate': fill_rate,
            'avg_slippage_pct': avg_slippage,
            'win_rate': win_rate,
            'avg_gross_return': avg_return,
            'outcome_distribution': outcomes,
        }
    
    def get_validation_status(self, backtest_win_rate: float = 61.0) -> Dict[str, Any]:
        """
        Check if paper trade results validate against backtest.
        
        Gate: Paper trade win rate must be within 5% of backtest win rate.
        """
        metrics = self.compute_execution_metrics()
        
        if 'error' in metrics:
            return {
                'can_go_live': False,
                'reason': metrics['error'],
                'n_trading_days': 0,
            }
        
        # Get unique trading days
        trading_days = set(r.trade_date for r in self.trade_records)
        n_days = len(trading_days)
        
        # Compare to backtest
        actual_win_rate = metrics['win_rate']
        win_rate_diff = abs(actual_win_rate - backtest_win_rate)
        within_tolerance = win_rate_diff <= 5.0  # 5% tolerance
        
        # Additional checks
        min_days_met = n_days >= self.validation_days
        fill_rate_ok = metrics['fill_rate'] >= 80  # At least 80% fill rate
        
        can_go_live = within_tolerance and min_days_met and fill_rate_ok
        
        return {
            'can_go_live': can_go_live,
            'n_trading_days': n_days,
            'n_trades': metrics['n_completed_trades'],
            'backtest_win_rate': backtest_win_rate,
            'actual_win_rate': actual_win_rate,
            'win_rate_diff': win_rate_diff,
            'within_tolerance': within_tolerance,
            'fill_rate': metrics['fill_rate'],
            'avg_slippage': metrics['avg_slippage_pct'],
            'reason': self._get_gate_reason(within_tolerance, min_days_met, fill_rate_ok),
        }
    
    def _get_gate_reason(self, tolerance: bool, min_days: bool, fill_rate: bool) -> str:
        """Get reason for gate status."""
        if not tolerance:
            return "Win rate deviation exceeds 5% tolerance"
        if not min_days:
            return f"Need {self.validation_days} trading days (have fewer)"
        if not fill_rate:
            return "Fill rate below 80%"
        return "All gates passed - ready for live"
    
    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        status = self.get_validation_status()
        metrics = self.compute_execution_metrics()
        
        report = []
        report.append("="*70)
        report.append("PAPER TRADE VALIDATION REPORT")
        report.append("="*70)
        report.append(f"\nTrading Days: {status['n_trading_days']}")
        report.append(f"Total Trades: {status['n_trades']}")
        report.append(f"\nExecution Quality:")
        report.append(f"  Fill Rate: {metrics['fill_rate']:.1f}%")
        report.append(f"  Avg Slippage: {metrics['avg_slippage_pct']:.3f}%")
        report.append(f"\nPerformance:")
        report.append(f"  Actual Win Rate: {status['actual_win_rate']:.1f}%")
        report.append(f"  Avg Gross Return: {metrics['avg_gross_return']:.3f}%")
        report.append(f"\nValidation Gate:")
        report.append(f"  Within Tolerance: {status['within_tolerance']}")
        report.append(f"  Can Go Live: {status['can_go_live']}")
        if not status['can_go_live']:
            report.append(f"  Reason: {status['reason']}")
        report.append("="*70)
        
        return "\n".join(report)


class AutomatedRetrainingPipeline:
    """
    5.2: Automated monthly retraining pipeline.
    
    Runs on first Saturday of every month:
    1. Pull latest month's data
    2. Append to training set
    3. Retrain LightGBM + best DL model
    4. Run walk-forward validation
    5. Compare metrics to previous model
    6. Promote if better by >0.5% Sharpe
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        data_dir: str = "nifty500",
        runs_dir: str = "runs",
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.runs_dir = Path(runs_dir)
        
        self.model_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        
        self.training_log: List[Dict] = []
    
    def should_retrain(self, date: Optional[datetime] = None) -> bool:
        """
        Check if retraining should run.
        
        Trigger: First Saturday of every month.
        """
        if date is None:
            date = datetime.now()
        
        # Check if first Saturday
        if date.weekday() != 5:  # Saturday
            return False
        
        if date.day > 7:  # Not first week
            return False
        
        return True
    
    def run_retraining(
        self,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        min_sharpe_improvement: float = 0.005,  # 0.5%
    ) -> Dict[str, Any]:
        """
        Execute full retraining pipeline.
        
        Returns result dict with status and metrics.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.runs_dir / f"retrain_{run_timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        logger.info("="*70)
        logger.info(f"AUTOMATED RETRAINING - {run_timestamp}")
        logger.info("="*70)
        
        result = {
            'timestamp': run_timestamp,
            'start_date': start_date,
            'end_date': end_date,
            'status': 'started',
            'metrics': {},
        }
        
        try:
            # Step 1: Load and prepare data
            logger.info("Step 1: Loading data...")
            # This would call your data loading pipeline
            
            # Step 2: Retrain models
            logger.info("Step 2: Retraining models...")
            # This would call your training pipeline
            
            # Step 3: Run walk-forward validation
            logger.info("Step 3: Running walk-forward validation...")
            from intradaynet.walkforward_v3 import WalkForwardEngine, WalkForwardConfig
            
            config = WalkForwardConfig()
            engine = WalkForwardEngine(config)
            
            # Run validation
            results = engine.run_full_walkforward(start_date, end_date)
            
            # Get aggregate metrics
            summary = engine.aggregate_results()
            
            result['metrics'] = summary
            
            # Step 4: Compare to previous model
            logger.info("Step 4: Comparing to previous model...")
            
            prev_sharpe = self._get_previous_sharpe()
            new_sharpe = summary.get('sharpe_ratio', {}).get('mean', 0)
            
            sharpe_diff = new_sharpe - prev_sharpe
            
            logger.info(f"  Previous Sharpe: {prev_sharpe:.3f}")
            logger.info(f"  New Sharpe: {new_sharpe:.3f}")
            logger.info(f"  Difference: {sharpe_diff:.3f}")
            
            # Step 5: Promote if better
            if sharpe_diff > min_sharpe_improvement:
                logger.info(f"✓ New model better by {sharpe_diff:.3f} Sharpe - PROMOTING")
                self._promote_model(run_dir)
                result['status'] = 'promoted'
                result['action'] = 'New model promoted to production'
            else:
                logger.info(f"✗ New model not better (diff {sharpe_diff:.3f} < {min_sharpe_improvement})")
                result['status'] = 'rejected'
                result['action'] = 'Keeping existing model'
            
            # Save artifacts
            self._save_run_artifacts(run_dir, result)
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        # Log run
        self.training_log.append(result)
        self._save_training_log()
        
        logger.info("="*70)
        logger.info(f"RETRAINING COMPLETE - Status: {result['status']}")
        logger.info("="*70)
        
        return result
    
    def _get_previous_sharpe(self) -> float:
        """Get Sharpe from previous production model."""
        production_metrics = self.model_dir / "production_metrics.json"
        if production_metrics.exists():
            with open(production_metrics) as f:
                data = json.load(f)
                return data.get('sharpe_ratio', 0)
        return 0.0
    
    def _promote_model(self, run_dir: Path):
        """Promote new model to production."""
        # Copy model files to production
        production_dir = self.model_dir / "production"
        production_dir.mkdir(exist_ok=True)
        
        # This would copy actual model files
        logger.info(f"Model promoted to {production_dir}")
    
    def _save_run_artifacts(self, run_dir: Path, result: Dict):
        """Save all artifacts from retraining run."""
        with open(run_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _save_training_log(self):
        """Save training log."""
        with open(self.runs_dir / "training_log.json", 'w') as f:
            json.dump(self.training_log, f, indent=2, default=str)
    
    def get_training_history(self) -> pd.DataFrame:
        """Get history of all retraining runs."""
        return pd.DataFrame(self.training_log)


def main():
    """Demo execution infrastructure."""
    print("\n" + "="*70)
    print("Phase 5: Execution Infrastructure Demo")
    print("="*70)
    
    # 1. Paper Trade Logger
    print("\n1. PAPER TRADE LOGGER")
    print("-"*70)
    
    paper_logger = PaperTradeLogger(
        output_dir="demo_paper_trades",
        validation_days=5,  # Shorter for demo
    )
    
    # Simulate 5 days of predictions
    for day in range(1, 6):
        date = f"2025-01-{day+10:02d}"
        
        # Log predictions
        picks = [
            {
                'symbol': 'RELIANCE',
                'side': 'LONG',
                'entry_reference': 2500.0,
                'target': 2530.0,
                'stop_loss': 2480.0,
                'horizon': 'H60',
                'confidence': 0.65,
            },
            {
                'symbol': 'TCS',
                'side': 'SHORT',
                'entry_reference': 3500.0,
                'target': 3470.0,
                'stop_loss': 3530.0,
                'horizon': 'H60',
                'confidence': 0.62,
            },
        ]
        
        paper_logger.log_predictions(date, picks, regime='trending_calm')
        
        # Simulate actual results
        actuals = {
            'RELIANCE': {'open': 2502.0, 'high': 2535.0, 'low': 2498.0, 'close': 2525.0},
            'TCS': {'open': 3498.0, 'high': 3520.0, 'low': 3465.0, 'close': 3475.0},
        }
        
        paper_logger.log_results(
            date,
            {s: v['open'] for s, v in actuals.items()},
            {s: v['high'] for s, v in actuals.items()},
            {s: v['low'] for s, v in actuals.items()},
            {s: v['close'] for s, v in actuals.items()},
        )
    
    # Check validation status
    status = paper_logger.get_validation_status(backtest_win_rate=60.0)
    
    print(f"\nValidation Status:")
    print(f"  Trading Days: {status['n_trading_days']}")
    print(f"  Actual Win Rate: {status['actual_win_rate']:.1f}%")
    print(f"  Backtest Win Rate: 60.0%")
    print(f"  Can Go Live: {status['can_go_live']}")
    
    # 2. Automated Retraining
    print("\n\n2. AUTOMATED RETRAINING PIPELINE")
    print("-"*70)
    
    retrain = AutomatedRetrainingPipeline()
    
    # Check if retraining should run
    today = datetime(2025, 2, 1)  # First Saturday of month
    should_run = retrain.should_retrain(today)
    
    print(f"Date: {today.strftime('%Y-%m-%d (%A)')}")
    print(f"Should retrain: {should_run}")
    
    if should_run:
        print("\nRetraining would execute:")
        print("  1. Pull latest month's data")
        print("  2. Append to training set")
        print("  3. Retrain LightGBM + DL models")
        print("  4. Run walk-forward validation")
        print("  5. Compare metrics (need >0.5% Sharpe improvement)")
        print("  6. Promote if better")
    
    print("\n" + "="*70)
    print("Demo Complete")
    print("="*70)


if __name__ == "__main__":
    main()
