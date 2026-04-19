"""
V5 Backtest on 2024 Test Data (Simulated Paper Trading)
Tests V5 model on 2024 data as proxy for paper trading
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

warnings.filterwarnings('ignore')

console = Console()

# Top 25 feature indices
TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]


def load_v5_model():
    """Load the trained V5 model."""
    model_dir = Path("results/models/v5_selected_top25")
    
    if not model_dir.exists():
        model_dir = Path("results/models/v5_comprehensive")
    
    direction_model = lgb.Booster(model_file=str(model_dir / "direction_model.lgb"))
    magnitude_model = lgb.Booster(model_file=str(model_dir / "magnitude_model.lgb"))
    confidence_model = lgb.Booster(model_file=str(model_dir / "confidence_model.lgb"))
    
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return direction_model, magnitude_model, confidence_model, metadata


def run_backtest_simulation():
    """Run backtest on 2024 test data (proxy for paper trading)."""
    console.print(Panel.fit(
        "[bold blue]📊 V5 PAPER TRADING SIMULATION (2024 Test Data)[/bold blue]\n"
        "[cyan]Testing V5 model on 2024 data as proxy for live trading[/cyan]\n"
        "[white]Capital: ₹25,000 | Risk per trade: 5% | Threshold: 0.55[/white]",
        title="Experiment B",
        border_style="blue"
    ))
    
    # Load model
    try:
        direction_model, magnitude_model, confidence_model, metadata = load_v5_model()
        console.print(f"[green]✅ Model loaded (AUC: {metadata['test_metrics']['direction_auc']:.4f})[/green]\n")
    except Exception as e:
        console.print(f"[red]❌ Failed to load model: {e}[/red]")
        return
    
    # Load 2024 test data
    PREBATCH_DIR = Path("cache/prebatched_features_v5")
    console.print(f"[yellow]Loading 2024 test data from {PREBATCH_DIR}...[/yellow]")
    
    all_test_samples = []
    batch_files = list(PREBATCH_DIR.glob("*_features_v5.pkl"))
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'rb') as f:
                samples = pickle.load(f)
                for s in samples:
                    # Check if 2024 data
                    date = pd.to_datetime(s['date'])
                    if date.year == 2024:
                        # Select only top 25 features
                        s['features'] = s['features'][TOP_25_INDICES]
                        all_test_samples.append(s)
        except:
            pass
    
    if len(all_test_samples) == 0:
        console.print("[red]❌ No 2024 test data found![/red]")
        return
    
    console.print(f"[green]✅ Loaded {len(all_test_samples):,} samples from 2024[/green]\n")
    
    # Prepare data
    X_test = np.array([s['features'] for s in all_test_samples])
    y_dir_test = np.array([s['y_dir'] for s in all_test_samples])
    y_mag_test = np.array([s['y_mag'] for s in all_test_samples])
    dates_test = pd.to_datetime([s['date'] for s in all_test_samples])
    
    # Get predictions
    console.print("[yellow]Generating predictions...[/yellow]")
    dir_proba = direction_model.predict(X_test)
    dir_preds = (dir_proba > 0.5).astype(int)
    mag_preds = magnitude_model.predict(X_test)
    conf_preds = confidence_model.predict(X_test)
    
    # Trading simulation
    # Only trade when confidence > 0.55 and model confidence > 0.5
    threshold = 0.55
    trade_mask = (dir_proba > threshold) & (conf_preds > 0.5)
    
    trades = []
    for i in range(len(X_test)):
        if trade_mask[i]:
            # Simulate trade
            actual_return = y_mag_test[i] if y_dir_test[i] == 1 else -y_mag_test[i]
            predicted_proba = dir_proba[i]
            predicted_dir = 1 if predicted_proba > 0.5 else 0
            
            # Trade outcome
            won = (predicted_dir == y_dir_test[i])
            pnl = actual_return if won else -actual_return
            
            trades.append({
                'date': dates_test[i],
                'predicted_proba': predicted_proba,
                'predicted_dir': predicted_dir,
                'actual_dir': y_dir_test[i],
                'actual_return': actual_return,
                'won': won,
                'pnl': pnl,
                'confidence': conf_preds[i]
            })
    
    # Results
    if len(trades) == 0:
        console.print("[yellow]⚠️ No trades met the threshold criteria[/yellow]")
        return
    
    console.print(f"\n[bold cyan]📈 PAPER TRADING RESULTS[/bold cyan]\n")
    
    # Statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['won'])
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades
    
    avg_pnl = np.mean([t['pnl'] for t in trades])
    total_pnl = sum([t['pnl'] for t in trades])
    
    # Capital simulation
    capital = 25000
    risk_per_trade = 0.05
    position_size = capital * risk_per_trade
    
    profits = [t['pnl'] * position_size for t in trades]
    total_profit = sum(profits)
    
    # Daily aggregation
    trades_df = pd.DataFrame(trades)
    trades_df['date_only'] = trades_df['date'].dt.date
    daily_stats = trades_df.groupby('date_only').agg({
        'pnl': ['count', 'sum', 'mean'],
        'won': 'sum'
    }).reset_index()
    daily_stats.columns = ['date', 'trades', 'daily_pnl', 'avg_pnl', 'wins']
    daily_stats['win_rate'] = daily_stats['wins'] / daily_stats['trades']
    
    profitable_days = sum(1 for p in daily_stats['daily_pnl'] if p > 0)
    total_days = len(daily_stats)
    
    # Display results
    results_table = Table(title=f"📊 Paper Trading Results ({total_trades} Trades, {total_days} Days)", box=box.ROUNDED)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_column("Target", style="yellow")
    
    results_table.add_row("Total Trades", str(total_trades), "-")
    results_table.add_row("Trading Days", str(total_days), "-")
    results_table.add_row("Win Rate", f"{win_rate:.1%}", "✅ > 54%" if win_rate > 0.54 else "❌ < 54%")
    results_table.add_row("Profitable Days", f"{profitable_days}/{total_days} ({profitable_days/total_days:.1%})", "✅ > 50%" if profitable_days/total_days > 0.5 else "❌")
    results_table.add_row("Avg Return/Trade", f"{avg_pnl:.3f}%", "✅ > 0%" if avg_pnl > 0 else "❌")
    results_table.add_row("Total Return", f"{total_pnl:.2f}%", "-")
    results_table.add_row("Simulated Profit (₹25k)", f"₹{total_profit:,.2f}", "✅ PROFIT" if total_profit > 0 else "❌")
    
    # Sharpe-like metric
    daily_returns = daily_stats['daily_pnl'].values
    sharpe_like = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    results_table.add_row("Sharpe-like Ratio", f"{sharpe_like:.2f}", "✅ > 0.8" if sharpe_like > 0.8 else "❌")
    
    console.print(results_table)
    console.print()
    
    # Save results
    results_dir = Path("results/paper_trading")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'model_version': 'v5_selected_top25',
        'simulation': '2024_test_data_proxy',
        'capital': 25000,
        'risk_per_trade': 0.05,
        'threshold': threshold,
        'metrics': {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': float(win_rate),
            'profitable_days': int(profitable_days),
            'total_days': int(total_days),
            'day_profitability_rate': float(profitable_days/total_days),
            'avg_pnl': float(avg_pnl),
            'total_pnl': float(total_pnl),
            'sharpe_like': float(sharpe_like),
            'simulated_profit_inr': float(total_profit)
        },
        'daily_stats': daily_stats.to_dict(orient='records'),
        'trade_count': len(trades)
    }
    
    with open(results_dir / "v5_backtest_2024_results.json", 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Summary
    is_profitable = win_rate > 0.54 and total_profit > 0 and sharpe_like > 0.8
    
    if is_profitable:
        console.print(Panel.fit(
            "[bold green]✅🎯 PROFITABLE TRADING SYSTEM![/bold green]\n\n"
            f"[white]Win Rate: {win_rate:.1%} (target: >54%)[/white]\n"
            f"[white]Daily Profit Rate: {profitable_days/total_days:.1%} (target: >50%)[/white]\n"
            f"[white]Total Return: {total_pnl:.2f}%[/white]\n"
            f"[white]Sharpe-like: {sharpe_like:.2f} (target: >0.8)[/white]\n"
            f"[white]Simulated Profit: ₹{total_profit:,.2f}[/white]\n\n"
            "[bold green]🚀 READY FOR LIVE PAPER TRADING![/bold green]",
            border_style="green"
        ))
    elif win_rate > 0.50 and total_profit > 0:
        console.print(Panel.fit(
            "[bold yellow]⚠️ SLIGHTLY PROFITABLE[/bold yellow]\n\n"
            f"[white]Win Rate: {win_rate:.1%} (below 54% target)[/white]\n"
            f"[white]Total Return: {total_pnl:.2f}%[/white]\n"
            f"[white]Sharpe-like: {sharpe_like:.2f}[/white]\n\n"
            "[dim]Try:\n"
            "• Lowering threshold to get more trades\n"
            "• Adding more features (Experiment C)[/dim]",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]❌ NOT PROFITABLE[/bold red]\n\n"
            f"[white]Win Rate: {win_rate:.1%}[/white]\n"
            f"[white]Total Return: {total_pnl:.2f}%[/white]\n"
            f"[white]Sharpe-like: {sharpe_like:.2f}[/white]\n\n"
            "[dim]Need to:\n"
            "• Add advanced features (sentiment, options)\n"
            "• Try different model architecture\n"
            "• Increase training data[/dim]",
            border_style="red"
        ))
    
    console.print(f"\n[dim]Results saved to: {results_dir}/v5_backtest_2024_results.json[/dim]")


if __name__ == "__main__":
    try:
        run_backtest_simulation()
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Interrupted[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
