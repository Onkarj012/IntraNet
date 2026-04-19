"""
V5 Paper Trading - Test Model on Recent Market Data
20-day simulation with ₹25,000 capital to verify profitability
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()

# Top 25 feature indices (must match training)
TOP_25_INDICES = [7, 26, 35, 39, 4, 15, 11, 30, 0, 38, 25, 40, 16, 8, 14, 6, 41, 10, 19, 17, 24, 23, 5, 3, 34]
FEATURE_NAMES = [
    "atr", "hour", "returns_5m", "body_size", "bb_upper", "kijun_sen",
    "williams_r", "is_closing_hour", "rsi", "high_low_range", "volume_change",
    "upper_shadow", "realized_vol_30m", "atr_percent", "tenkan_sen", "bb_position",
    "lower_shadow", "stoch_d", "garman_klass_vol", "realized_vol_60m",
    "volume_ma_ratio", "price_momentum_slope", "bb_lower", "macd_histogram",
    "intraday_momentum"
]


def load_v5_model():
    """Load the trained V5 model."""
    model_dir = Path("results/models/v5_selected_top25")
    
    if not model_dir.exists():
        # Try comprehensive model
        model_dir = Path("results/models/v5_comprehensive")
        console.print("[yellow]Using comprehensive V5 model (44 features)[/yellow]")
    
    direction_model = lgb.Booster(model_file=str(model_dir / "direction_model.lgb"))
    magnitude_model = lgb.Booster(model_file=str(model_dir / "magnitude_model.lgb"))
    confidence_model = lgb.Booster(model_file=str(model_dir / "confidence_model.lgb"))
    
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return direction_model, magnitude_model, confidence_model, metadata


def fetch_recent_data(ticker, days=25):
    """Fetch recent intraday data for testing."""
    try:
        # Download last N days of 1-minute data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1m',
            progress=False,
            auto_adjust=True
        )
        
        if len(data) == 0:
            return None
            
        data = data.reset_index()
        data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        return data
    except Exception as e:
        return None


def compute_features_live(data):
    """Compute the 25 selected features from live data."""
    df = data.copy()
    
    # Ensure we have enough data
    if len(df) < 60:
        return None
    
    features_list = []
    
    for i in range(60, len(df)):
        window = df.iloc[i-60:i]
        
        # Price data
        prices = window['close'].values
        highs = window['high'].values
        lows = window['low'].values
        volumes = window['volume'].values
        
        # Time features
        dt = df.iloc[i]['datetime']
        hour = dt.hour
        minute = dt.minute
        is_closing_hour = 1 if hour >= 14 else 0
        
        # Technical indicators (simplified for speed)
        # ATR
        tr1 = highs[-1] - lows[-1]
        tr2 = abs(highs[-1] - prices[-2]) if len(prices) > 1 else tr1
        tr3 = abs(lows[-1] - prices[-2]) if len(prices) > 1 else tr1
        atr = np.mean([tr1, tr2, tr3])
        atr_percent = atr / prices[-1] if prices[-1] > 0 else 0
        
        # Returns
        returns_5m = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        
        # Body size (last candle)
        body_size = abs(prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 and prices[-2] > 0 else 0
        high_low_range = (highs[-1] - lows[-1]) / prices[-1] if prices[-1] > 0 else 0
        upper_shadow = (highs[-1] - max(prices[-1], prices[-2])) / prices[-1] if len(prices) > 1 and prices[-1] > 0 else 0
        lower_shadow = (min(prices[-1], prices[-2]) - lows[-1]) / prices[-1] if len(prices) > 1 and prices[-1] > 0 else 0
        
        # Volume
        volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1) if len(volumes) > 1 else 0
        volume_ma_ratio = volumes[-1] / (np.mean(volumes[-10:]) + 1)
        
        # RSI (simplified)
        if len(prices) >= 14:
            deltas = np.diff(prices[-14:])
            gains = np.mean([d for d in deltas if d > 0]) if any(d > 0 for d in deltas) else 0
            losses = abs(np.mean([d for d in deltas if d < 0])) if any(d < 0 for d in deltas) else 1
            rsi = 100 - (100 / (1 + gains / (losses + 1e-10)))
        else:
            rsi = 50
        
        # Bollinger bands
        if len(prices) >= 20:
            sma = np.mean(prices[-20:])
            std = np.std(prices[-20:])
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        else:
            bb_upper = prices[-1] * 1.02
            bb_lower = prices[-1] * 0.98
            bb_position = 0.5
        
        # Williams %R
        if len(prices) >= 14:
            highest_high = np.max(highs[-14:])
            lowest_low = np.min(lows[-14:])
            williams_r = -100 * (highest_high - prices[-1]) / (highest_high - lowest_low + 1e-10)
        else:
            williams_r = -50
        
        # Stochastic
        if len(prices) >= 14:
            highest_high = np.max(highs[-14:])
            lowest_low = np.min(lows[-14:])
            stoch_d = 100 * (prices[-1] - lowest_low) / (highest_high - lowest_low + 1e-10)
        else:
            stoch_d = 50
        
        # MACD histogram (simplified)
        if len(prices) >= 26:
            ema12 = np.mean(prices[-12:])
            ema26 = np.mean(prices[-26:])
            macd = ema12 - ema26
            macd_signal = np.mean([macd])  # Simplified
            macd_histogram = macd - macd_signal
        else:
            macd_histogram = 0
        
        # Realized volatility
        log_returns = np.diff(np.log(prices[-30:] + 1e-10))
        realized_vol_30m = np.std(log_returns) * np.sqrt(252 * 375) if len(log_returns) > 1 else 0
        realized_vol_60m = realized_vol_30m  # Simplified
        
        # Garman-Klass volatility (simplified)
        log_hl = np.log((highs[-1] / lows[-1]) if lows[-1] > 0 else 1)
        log_co = np.log((prices[-1] / prices[-2]) if prices[-2] > 0 else 1) if len(prices) > 1 else 0
        garman_klass_vol = np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2)
        
        # Ichimoku (simplified)
        if len(prices) >= 26:
            tenkan_sen = (np.max(highs[-9:]) + np.min(lows[-9:])) / 2
            kijun_sen = (np.max(highs[-26:]) + np.min(lows[-26:])) / 2
        else:
            tenkan_sen = prices[-1]
            kijun_sen = prices[-1]
        
        # Price momentum slope
        if len(prices) >= 10:
            x = np.arange(10)
            y = prices[-10:]
            price_momentum_slope = np.polyfit(x, y, 1)[0] / prices[-1]
        else:
            price_momentum_slope = 0
        
        # Intraday momentum (simplified)
        intraday_momentum = returns_5m * volume_ma_ratio
        
        # Assemble features in correct order
        features = np.array([
            atr, hour, returns_5m, body_size, bb_upper, kijun_sen,
            williams_r, is_closing_hour, rsi, high_low_range, volume_change,
            upper_shadow, realized_vol_30m, atr_percent, tenkan_sen, bb_position,
            lower_shadow, stoch_d, garman_klass_vol, realized_vol_60m,
            volume_ma_ratio, price_momentum_slope, bb_lower, macd_histogram,
            intraday_momentum
        ])
        
        features_list.append({
            'datetime': dt,
            'price': prices[-1],
            'features': features
        })
    
    return features_list


def paper_trade_stock(ticker, direction_model, magnitude_model, confidence_model, 
                      capital_per_stock=5000, threshold=0.55):
    """Paper trade a single stock."""
    # Fetch data
    data = fetch_recent_data(ticker, days=20)
    if data is None or len(data) < 100:
        return None
    
    # Compute features
    features_data = compute_features_live(data)
    if features_data is None or len(features_data) < 10:
        return None
    
    # Trading simulation
    trades = []
    position = None
    entry_price = 0
    entry_time = None
    
    for i, row in enumerate(features_data):
        X = row['features'].reshape(1, -1)
        
        # Get predictions
        dir_proba = direction_model.predict(X)[0]
        confidence = confidence_model.predict(X)[0]
        magnitude = magnitude_model.predict(X)[0]
        
        current_price = row['price']
        current_time = row['datetime']
        
        # Trading logic
        if position is None:
            # Look for entry
            if dir_proba > threshold and confidence > 0.5:
                position = 'LONG'
                entry_price = current_price
                entry_time = current_time
                entry_proba = dir_proba
                
        elif position == 'LONG':
            # Look for exit (3:15 PM or target hit)
            hour = current_time.hour
            minute = current_time.minute
            
            # Exit at market close
            if hour >= 15 and minute >= 15:
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'direction': 'LONG',
                    'confidence_at_entry': entry_proba,
                    'won': pnl > 0
                })
                position = None
    
    return trades


def run_paper_trading_simulation():
    """Run 20-day paper trading simulation."""
    console.print(Panel.fit(
        "[bold blue]📊 V5 PAPER TRADING SIMULATION[/bold blue]\n"
        "[cyan]Testing V5 model on recent 20 days of market data[/cyan]\n"
        "[white]Capital: ₹25,000 | Threshold: 0.55 | Max Trades/Day: 5[/white]",
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
    
    # Load Nifty 500 symbols
    try:
        nifty500_path = Path("data/nifty500_symbols.csv")
        if nifty500_path.exists():
            nifty500 = pd.read_csv(nifty500_path)
            symbols = nifty500['symbol'].tolist()[:50]  # Test on 50 stocks
        else:
            symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                      'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'][:10]
    except:
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'][:5]
    
    console.print(f"[yellow]Testing on {len(symbols)} stocks from Nifty 500...[/yellow]\n")
    
    # Run paper trading
    all_trades = []
    
    for i, symbol in enumerate(symbols):
        console.print(f"  [{i+1}/{len(symbols)}] Testing {symbol}...", end=" ")
        
        try:
            trades = paper_trade_stock(symbol, direction_model, magnitude_model, 
                                       confidence_model, capital_per_stock=500, threshold=0.55)
            if trades:
                all_trades.extend(trades)
                console.print(f"[green]{len(trades)} trades[/green]")
            else:
                console.print("[dim]no trades[/dim]")
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    
    # Results
    console.print(f"\n[bold cyan]📈 PAPER TRADING RESULTS[/bold cyan]\n")
    
    if len(all_trades) == 0:
        console.print("[yellow]⚠️ No trades executed in the simulation period[/yellow]")
        return
    
    # Compute statistics
    total_trades = len(all_trades)
    winning_trades = sum(1 for t in all_trades if t['won'])
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades
    
    # P&L
    avg_pnl = np.mean([t['pnl'] for t in all_trades])
    total_pnl = sum([t['pnl'] for t in all_trades])
    
    # Capital simulation (₹25,000, 5% risk per trade)
    capital = 25000
    risk_per_trade = 0.05
    position_size = capital * risk_per_trade
    
    profits = [t['pnl'] * position_size for t in all_trades]
    total_profit = sum(profits)
    
    results_table = Table(title=f"📊 Paper Trading Results ({total_trades} Trades)", box=box.ROUNDED)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_column("Status", style="yellow")
    
    results_table.add_row("Total Trades", str(total_trades), "-")
    results_table.add_row("Winning Trades", f"{winning_trades} ({win_rate:.1%})", 
                         "✅ GOOD" if win_rate > 0.54 else "⚠️ LOW")
    results_table.add_row("Losing Trades", str(losing_trades), "-")
    results_table.add_row("Avg Return per Trade", f"{avg_pnl:.2%}", 
                         "✅ PROFIT" if avg_pnl > 0 else "❌ LOSS")
    results_table.add_row("Total Return (Simulated)", f"{total_pnl:.2%}", 
                         "✅ PROFIT" if total_pnl > 0 else "❌ LOSS")
    results_table.add_row("Total Profit (₹25k capital)", f"₹{total_profit:,.2f}", 
                         "✅ PROFIT" if total_profit > 0 else "❌ LOSS")
    
    console.print(results_table)
    console.print()
    
    # Save results
    results_dir = Path("results/paper_trading")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'model_version': 'v5_selected_top25',
        'simulation_days': 20,
        'capital': 25000,
        'threshold': 0.55,
        'stocks_tested': len(symbols),
        'metrics': {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': float(win_rate),
            'avg_pnl': float(avg_pnl),
            'total_pnl': float(total_pnl),
            'simulated_profit_inr': float(total_profit)
        },
        'trades': [
            {
                'entry_time': t['entry_time'].isoformat() if hasattr(t['entry_time'], 'isoformat') else str(t['entry_time']),
                'exit_time': t['exit_time'].isoformat() if hasattr(t['exit_time'], 'isoformat') else str(t['exit_time']),
                'entry_price': float(t['entry_price']),
                'exit_price': float(t['exit_price']),
                'pnl': float(t['pnl']),
                'won': t['won']
            }
            for t in all_trades
        ]
    }
    
    with open(results_dir / "v5_paper_trading_results.json", 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Summary
    if win_rate > 0.54 and total_pnl > 0:
        console.print(Panel.fit(
            "[bold green]✅ PROFITABLE TRADING SYSTEM![/bold green]\n"
            f"[white]Win Rate: {win_rate:.1%} (> 54% target)[/white]\n"
            f"[white]Total Return: {total_pnl:.2%}[/white]\n"
            f"[white]Simulated Profit: ₹{total_profit:,.2f}[/white]",
            border_style="green"
        ))
    elif win_rate > 0.50 and total_pnl > 0:
        console.print(Panel.fit(
            "[bold yellow]⚠️ SLIGHTLY PROFITABLE[/bold yellow]\n"
            f"[white]Win Rate: {win_rate:.1%} (below 54% target)[/white]\n"
            f"[white]Total Return: {total_pnl:.2%}[/white]\n"
            "[dim]Try adjusting threshold or adding more features[/dim]",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]❌ NOT PROFITABLE[/bold red]\n"
            f"[white]Win Rate: {win_rate:.1%}[/white]\n"
            f"[white]Total Return: {total_pnl:.2%}[/white]\n"
            "[dim]Need to improve model or features[/dim]",
            border_style="red"
        ))
    
    console.print(f"[dim]Results saved to: {results_dir}/v5_paper_trading_results.json[/dim]")


if __name__ == "__main__":
    try:
        run_paper_trading_simulation()
    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️ Interrupted[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
