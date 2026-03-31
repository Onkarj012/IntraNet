#!/usr/bin/env python3
"""
IntradayNet Pro Backtester — Day-by-day intraday trading simulation.

Simulates realistic intraday trading with capital management, position limits,
long/short positions, stop-loss, take-profit, and transaction costs.

Usage:
    python scripts/backtest_pro.py --model runs/intraday/resnls/best_model.pt
    python scripts/backtest_pro.py --model runs/intraday/resnls/best_model.pt --capital 500000 --max-positions 10
"""

import argparse
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from intradaynet.config import load_config
from intradaynet.features.per_bar_features import PER_BAR_FEATURE_NAMES
from intradaynet.features.session_features import SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES

console = Console()


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class Trade:
    symbol: str
    date: str
    direction: str  # "LONG" or "SHORT"
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    position_size: float  # ₹ allocated
    shares: int
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str  # "take_profit", "stop_loss", "eod"
    confidence: float
    pred_prob: float
    pred_magnitude: float
    actual_magnitude: float


@dataclass
class DailyResult:
    date: str
    trades_taken: int
    winning_trades: int
    daily_pnl: float
    daily_return_pct: float
    cumulative_pnl: float
    equity: float
    max_drawdown_pct: float


# ── CLI Arguments ────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="IntradayNet Pro Backtester")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--features-cache", type=str, default="features_cache")
    parser.add_argument("--output-dir", type=str, default="backtest_results")

    # Capital & Position Management
    parser.add_argument("--capital", type=float, default=1000000,
                        help="Initial capital in ₹ (default: 10,00,000)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Max concurrent open positions")
    parser.add_argument("--position-pct", type=float, default=20,
                        help="Per-trade allocation as %% of capital (default: 20%%)")

    # Strategy Parameters
    parser.add_argument("--horizon", type=str, default="H60",
                        help="Horizon: H15, H30, H60, H375")
    parser.add_argument("--min-confidence", type=float, default=0.55,
                        help="Min model confidence to trade (default: 0.55)")
    parser.add_argument("--dir-threshold", type=float, default=0.60,
                        help="Direction probability threshold for LONG/SHORT (default: 0.60)")
    parser.add_argument("--stop-loss", type=float, default=0.005,
                        help="Stop loss as fraction (default: 0.5%%)")
    parser.add_argument("--take-profit", type=float, default=0.01,
                        help="Take profit as fraction (default: 1.0%%)")

    # Transaction Costs
    parser.add_argument("--brokerage", type=float, default=20,
                        help="Brokerage per order in ₹")
    parser.add_argument("--stt", type=float, default=0.00025,
                        help="STT rate (default: 0.025%%)")
    parser.add_argument("--slippage", type=float, default=0.0005,
                        help="Slippage rate (default: 0.05%%)")

    # Risk Management
    parser.add_argument("--daily-loss-limit", type=float, default=0.02,
                        help="Max daily loss as fraction of capital (default: 2%%)")
    parser.add_argument("--min-trade-value", type=float, default=10000,
                        help="Min trade value in ₹ (avoids brokerage eating profits)")
    parser.add_argument("--min-pred-magnitude", type=float, default=0.002,
                        help="Min predicted return magnitude to trade (default: 0.2%%)")
    parser.add_argument("--capital-protection", type=float, default=0.80,
                        help="Stop trading if capital drops below this fraction (default: 80%%)")

    # Stock selection
    parser.add_argument("--stocks", type=str, default="",
                        help="Comma-separated symbols (default: all in cache)")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--max-price", type=float, default=0,
                        help="Filter out stocks with median close > this price (0 = no filter)")
    parser.add_argument("--min-avg-volume", type=float, default=0,
                        help="Min average daily volume to include stock (0 = no filter)")

    # Trading mode
    parser.add_argument("--long-only", action="store_true",
                        help="Only take LONG positions (disable SHORT signals)")
    parser.add_argument("--zero-cost", action="store_true",
                        help="Set brokerage, STT, and slippage to zero (pure signal analysis)")

    args = parser.parse_args()

    # Apply --zero-cost override
    if args.zero_cost:
        args.brokerage = 0.0
        args.stt = 0.0
        args.slippage = 0.0

    # Sanity cap: position_pct cannot exceed 100/max_positions
    max_safe_pct = 100.0 / max(args.max_positions, 1)
    if args.position_pct > max_safe_pct:
        args.position_pct = max_safe_pct

    return args


# ── Model Loading ────────────────────────────────────────────────────────────


def load_model(checkpoint_path, cfg):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "tcn_attention")

    if model_type == "resnls":
        from intradaynet.models.resnls_intraday import IntradayResNLS
        model = IntradayResNLS(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            hidden_dim=64, lstm_layers=2,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "compact_cnn":
        from intradaynet.models.compact_cnn import CompactCNN
        model = CompactCNN(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "lightweight_gru":
        from intradaynet.models.lightweight_gru import LightweightGRU
        model = LightweightGRU(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            hidden_dim=48,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "mlp_mixer":
        from intradaynet.models.mlp_mixer import IntradayMLPMixer
        model = IntradayMLPMixer(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            patch_size=15, hidden_dim=64, num_mixer_blocks=3,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    else:
        from intradaynet.models.tcn_attention import IntradayTCNAttention
        model = IntradayTCNAttention(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=cfg.model.num_sentiment_features,
            hidden_dim=cfg.model.hidden_dim,
            tcn_channels=cfg.model.tcn.channels,
            kernel_size=cfg.model.tcn.kernel_size,
            dilation_base=cfg.model.tcn.dilation_base,
            attn_heads=cfg.model.attn_heads,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_type, checkpoint.get("epoch", "?")


# ── Core Backtesting Logic ───────────────────────────────────────────────────


@torch.no_grad()
def get_signals_for_session(model, per_bar_feats, close_prices, session_feats,
                            sentiment_feats, seq_length, horizon_idx, sample_interval):
    """
    Run inference on all valid bars in a session. Returns list of signal dicts.
    """
    n_bars = len(per_bar_feats)
    signals = []

    if n_bars < seq_length + 1:
        return signals

    # Batch all valid bars
    batch_per_bar = []
    batch_bar_offsets = []

    for bar_offset in range(seq_length, n_bars, sample_interval):
        window = per_bar_feats[bar_offset - seq_length: bar_offset]
        batch_per_bar.append(window)
        batch_bar_offsets.append(bar_offset)

    if not batch_per_bar:
        return signals

    per_bar_t = torch.from_numpy(np.array(batch_per_bar, dtype=np.float32))
    context_t = torch.from_numpy(
        np.tile(session_feats, (len(batch_per_bar), 1)).astype(np.float32)
    )
    sentiment_t = torch.from_numpy(
        np.tile(sentiment_feats, (len(batch_per_bar), 1)).astype(np.float32)
    )

    preds = model(per_bar_t, context_t, sentiment_t)
    probs = torch.sigmoid(preds["direction_logits"])[:, horizon_idx].numpy()
    mags = preds["magnitudes"][:, horizon_idx].numpy()
    confs = preds["confidences"][:, horizon_idx].numpy()

    for i, bar_offset in enumerate(batch_bar_offsets):
        signals.append({
            "bar_offset": bar_offset,
            "entry_price": float(close_prices[bar_offset]),
            "prob": float(probs[i]),
            "magnitude": float(mags[i]),
            "confidence": float(confs[i]),
        })

    return signals


def simulate_day(signals, close_prices, n_bars, horizon_bars, args, current_capital):
    """
    Simulate trading for one day. Returns list of Trade objects.

    Logic:
    1. Rank all signals by |confidence × magnitude| (strongest first)
    2. Take up to max_positions trades
    3. For each trade: enter at signal bar, exit at min(bar + horizon, EOD)
    4. Apply stop-loss and take-profit intraday
    """
    position_size = current_capital * (args.position_pct / 100)
    trades = []

    # Filter and rank signals
    tradeable = []
    for sig in signals:
        if sig["confidence"] < args.min_confidence:
            continue
        if sig["prob"] >= args.dir_threshold:
            sig["direction"] = "LONG"
        elif sig["prob"] <= (1 - args.dir_threshold):
            sig["direction"] = "SHORT"
        else:
            continue

        sig["score"] = sig["confidence"] * abs(sig["magnitude"])
        tradeable.append(sig)

    # Sort by signal strength
    tradeable.sort(key=lambda s: s["score"], reverse=True)

    # Take positions (no overlapping — simple sequential)
    used_bars = set()
    for sig in tradeable:
        if len(trades) >= args.max_positions:
            break

        entry_bar = sig["bar_offset"]
        if entry_bar in used_bars:
            continue

        entry_price = sig["entry_price"]
        if entry_price <= 0:
            continue

        shares = int(position_size / entry_price)
        if shares == 0:
            continue

        # Determine exit
        exit_bar = min(entry_bar + horizon_bars, n_bars - 1)
        exit_price = entry_price
        exit_reason = "eod"
        is_long = sig["direction"] == "LONG"

        # Simulate bar-by-bar for stop-loss / take-profit
        for b in range(entry_bar + 1, exit_bar + 1):
            current_price = float(close_prices[b])
            ret = (current_price - entry_price) / entry_price
            if not is_long:
                ret = -ret

            if ret <= -args.stop_loss:
                exit_price = current_price
                exit_reason = "stop_loss"
                exit_bar = b
                break
            elif ret >= args.take_profit:
                exit_price = current_price
                exit_reason = "take_profit"
                exit_bar = b
                break
            else:
                exit_price = current_price

        # Calculate PnL
        if is_long:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        # Transaction costs
        trade_value = shares * entry_price
        brokerage_cost = 2 * args.brokerage  # entry + exit
        stt_cost = trade_value * args.stt
        slippage_cost = trade_value * args.slippage * 2  # both sides
        total_costs = brokerage_cost + stt_cost + slippage_cost

        gross_pnl = gross_return * trade_value
        net_pnl = gross_pnl - total_costs

        # Actual magnitude (for validation)
        actual_exit = float(close_prices[min(entry_bar + horizon_bars, n_bars - 1)])
        actual_mag = (actual_exit - entry_price) / entry_price

        trades.append(Trade(
            symbol="",  # filled by caller
            date="",    # filled by caller
            direction=sig["direction"],
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=trade_value,
            shares=shares,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=gross_return * 100,
            exit_reason=exit_reason,
            confidence=sig["confidence"],
            pred_prob=sig["prob"],
            pred_magnitude=sig["magnitude"],
            actual_magnitude=actual_mag,
        ))

        # Mark bars as used
        for b in range(entry_bar, exit_bar + 1):
            used_bars.add(b)

    return trades


# ── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    cfg = load_config(args.config)
    features_cache = Path(args.features_cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse horizon
    horizon_map = {f"H{h}": (i, h) for i, h in enumerate(cfg.horizons)}
    if args.horizon not in horizon_map:
        console.print(f"[red]Unknown horizon: {args.horizon}. Choose from {list(horizon_map.keys())}[/red]")
        return
    horizon_idx, horizon_bars = horizon_map[args.horizon]

    # Header
    mode_tags = []
    if args.zero_cost:
        mode_tags.append("ZERO-COST")
    if args.long_only:
        mode_tags.append("LONG-ONLY")
    mode_str = f" | Mode: {', '.join(mode_tags)}" if mode_tags else ""

    console.print(Panel.fit(
        f"[bold cyan]IntradayNet Pro Backtester[/bold cyan]\n"
        f"[dim]Capital: ₹{args.capital:,.0f} | Max Positions: {args.max_positions} | "
        f"Horizon: {args.horizon} | SL: {args.stop_loss:.1%} | TP: {args.take_profit:.1%}{mode_str}\n"
        f"Daily Loss Limit: {args.daily_loss_limit:.1%} | Min Trade: ₹{args.min_trade_value:,.0f} | "
        f"Min Pred Mag: {args.min_pred_magnitude:.2%} | Pos: {args.position_pct:.0f}%[/dim]",
        border_style="cyan",
    ))

    # Load model
    console.print("[dim]Loading model...[/dim]")
    model, model_type, epoch = load_model(args.model, cfg)
    console.print(f"  Model: [green]{model_type}[/green] (epoch {epoch})")

    # Select stocks
    if args.stocks:
        symbols = [s.strip() for s in args.stocks.split(",")]
    elif args.max_stocks > 0:
        all_npz = sorted(features_cache.glob("*.npz"))
        symbols = [f.stem for f in all_npz[:args.max_stocks]]
    else:
        symbols = sorted([f.stem for f in features_cache.glob("*.npz")])

    initial_count = len(symbols)

    # ── Max price filter ──
    if args.max_price > 0:
        filtered_symbols = []
        for sym in symbols:
            npz_path = features_cache / f"{sym}.npz"
            if not npz_path.exists():
                continue
            try:
                npz = np.load(npz_path, allow_pickle=True)
                close_arr = npz["close"]
                median_price = float(np.median(close_arr[close_arr > 0])) if len(close_arr[close_arr > 0]) > 0 else 0
                npz.close()
                if median_price <= args.max_price:
                    filtered_symbols.append(sym)
            except Exception:
                pass
        symbols = filtered_symbols
        console.print(f"  Max price filter (≤ ₹{args.max_price:,.0f}): [green]{len(symbols)}[/green] / {initial_count} stocks")

    # ── Min avg volume filter ──
    if args.min_avg_volume > 0:
        vol_filtered = []
        for sym in symbols:
            npz_path = features_cache / f"{sym}.npz"
            if not npz_path.exists():
                continue
            try:
                npz = np.load(npz_path, allow_pickle=True)
                per_bar = npz["per_bar_features"]
                dates = npz["per_bar_dates"]
                # Volume ratio is feature index 1 — use close × count as proxy
                close_arr = npz["close"]
                npz.close()
                # Rough daily volume: count bars per day as proxy
                unique_dates = np.unique(dates)
                avg_bars = len(dates) / max(len(unique_dates), 1)
                if avg_bars >= 50:  # at least 50 bars average means reasonable liquidity
                    vol_filtered.append(sym)
            except Exception:
                vol_filtered.append(sym)  # keep on error
        before_vol = len(symbols)
        symbols = vol_filtered
        if before_vol != len(symbols):
            console.print(f"  Volume filter: [green]{len(symbols)}[/green] / {before_vol} stocks")

    console.print(f"  Stocks: [green]{len(symbols)}[/green]")
    console.print(f"  Test period: {cfg.splits.test_start} → {cfg.splits.test_end}\n")

    seq_length = cfg.model.sequence_length
    date_start = str(cfg.splits.test_start)[:10]
    date_end = str(cfg.splits.test_end)[:10]

    # ── Day-by-day simulation ──
    all_trades: List[Trade] = []
    daily_results: List[DailyResult] = []
    capital = args.capital
    peak_equity = capital
    max_drawdown = 0.0

    # Collect all trading days across all stocks
    console.print("[dim]Scanning trading days...[/dim]")
    stock_data = {}
    all_dates = set()

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Loading stocks...", total=len(symbols))
        for symbol in symbols:
            npz_path = features_cache / f"{symbol}.npz"
            if not npz_path.exists():
                progress.update(task, advance=1)
                continue

            # Load into memory and close file handle immediately
            # (np.load returns lazy NpzFile that keeps fd open)
            npz = np.load(npz_path, allow_pickle=True)
            dates = npz["per_bar_dates"]
            mask = (dates >= date_start) & (dates <= date_end)

            if mask.sum() == 0:
                npz.close()
                progress.update(task, advance=1)
                continue

            unique_dates = np.unique(dates[mask])
            for d in unique_dates:
                all_dates.add(d)

            # Only keep test-period data in memory (saves ~90% RAM)
            all_dates_arr = npz["per_bar_dates"]
            bar_mask = (all_dates_arr >= date_start) & (all_dates_arr <= date_end)

            per_bar_feats = np.array(npz["per_bar_features"][bar_mask])
            close_arr = np.array(npz["close"][bar_mask])
            dates_arr = np.array(all_dates_arr[bar_mask])

            sess_dates_all = np.array(npz["session_dates"])
            sess_feats_all = np.array(npz["session_features"])
            sent_feats_all = np.array(npz["sentiment_features"])

            # Filter session-level features to test period
            sess_str = np.array([str(d)[:10] for d in sess_dates_all])
            sess_mask = (sess_str >= date_start) & (sess_str <= date_end)

            stock_data[symbol] = {
                "per_bar_features": per_bar_feats,
                "per_bar_dates": dates_arr,
                "close": close_arr,
                "session_features": sess_feats_all[sess_mask],
                "session_dates": sess_str[sess_mask],
                "sentiment_features": sent_feats_all[sess_mask],
            }
            npz.close()
            progress.update(task, advance=1)

    trading_days = sorted(all_dates)
    console.print(f"  Trading days: [green]{len(trading_days)}[/green]")
    console.print(f"  Active stocks: [green]{len(stock_data)}[/green]\n")

    # ── Run simulation ──
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total} days"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Backtesting...", total=len(trading_days))

        for day in trading_days:
            day_trades = []

            # Gather signals across all stocks for this day
            all_day_signals = []

            for symbol, data in stock_data.items():
                dates = data["per_bar_dates"]
                session_mask = dates == day
                if session_mask.sum() < seq_length + 1:
                    continue

                session_indices = np.where(session_mask)[0]
                sess_start = session_indices[0]
                n_bars = len(session_indices)

                per_bar_feats = data["per_bar_features"][sess_start:sess_start + n_bars]
                close = data["close"][sess_start:sess_start + n_bars]

                # Get session & sentiment features
                sess_dates = data["session_dates"]
                sess_feats = data["session_features"]
                sent_feats = data["sentiment_features"]

                # Find matching session date
                sess_idx = np.where(sess_dates == day)[0]
                if len(sess_idx) == 0:
                    s_feat = np.zeros(len(SESSION_FEATURE_NAMES), dtype=np.float32)
                    se_feat = np.zeros(len(SENTIMENT_FEATURE_NAMES), dtype=np.float32)
                else:
                    s_feat = sess_feats[sess_idx[0]]
                    se_feat = sent_feats[sess_idx[0]]

                signals = get_signals_for_session(
                    model, per_bar_feats, close, s_feat, se_feat,
                    seq_length, horizon_idx, cfg.train.sample_interval,
                )

                for sig in signals:
                    sig["symbol"] = symbol
                    sig["close_prices"] = close
                    sig["n_bars"] = n_bars
                    all_day_signals.append(sig)

            # ── RISK CHECK 1: Capital protection ──
            if capital <= args.capital * args.capital_protection:
                progress.update(task, advance=1)
                daily_results.append(DailyResult(
                    date=day, trades_taken=0, winning_trades=0,
                    daily_pnl=0, daily_return_pct=0,
                    cumulative_pnl=capital - args.capital, equity=capital,
                    max_drawdown_pct=((peak_equity - capital) / peak_equity) * 100,
                ))
                continue

            # ── Filter & rank signals ──
            tradeable = []
            for sig in all_day_signals:
                # Filter: confidence threshold
                if sig["confidence"] < args.min_confidence:
                    continue
                # Filter: minimum predicted magnitude (signal must be worth trading)
                if abs(sig["magnitude"]) < args.min_pred_magnitude:
                    continue

                if sig["prob"] >= args.dir_threshold:
                    sig["direction"] = "LONG"
                elif not args.long_only and sig["prob"] <= (1 - args.dir_threshold):
                    sig["direction"] = "SHORT"
                else:
                    continue

                sig["score"] = sig["confidence"] * abs(sig["magnitude"])
                tradeable.append(sig)

            tradeable.sort(key=lambda s: s["score"], reverse=True)

            # Position sizing
            position_size = max(capital * (args.position_pct / 100), args.min_trade_value)
            daily_loss_budget = capital * args.daily_loss_limit
            daily_pnl_so_far = 0.0
            taken = 0

            for sig in tradeable:
                if taken >= args.max_positions:
                    break

                # ── RISK CHECK 2: Daily loss limit ──
                if daily_pnl_so_far <= -daily_loss_budget:
                    break

                entry_price = sig["entry_price"]
                if entry_price <= 0:
                    continue

                # Calculate trade value and shares
                trade_value = min(position_size, capital - abs(daily_pnl_so_far))
                if trade_value < args.min_trade_value:
                    continue  # Skip if remaining budget too small

                shares = int(trade_value / entry_price)
                if shares == 0:
                    continue

                actual_trade_value = shares * entry_price

                # ── RISK CHECK 3: Cost vs expected return ──
                # Don't trade if transaction costs exceed expected profit
                round_trip_cost = (2 * args.brokerage +
                                   actual_trade_value * args.stt +
                                   actual_trade_value * args.slippage * 2)
                cost_as_pct = round_trip_cost / actual_trade_value
                expected_return = abs(sig["magnitude"])

                if cost_as_pct >= expected_return * 0.5:
                    continue  # Costs eat >50% of expected return — skip

                close = sig["close_prices"]
                entry_bar = sig["bar_offset"]
                n_bars = sig["n_bars"]
                is_long = sig["direction"] == "LONG"

                # Simulate bar-by-bar exit
                exit_bar = min(entry_bar + horizon_bars, n_bars - 1)
                exit_price = entry_price
                exit_reason = "eod"

                for b in range(entry_bar + 1, exit_bar + 1):
                    if b >= n_bars:
                        break
                    current_price = float(close[b])
                    ret = (current_price - entry_price) / entry_price
                    if not is_long:
                        ret = -ret

                    if ret <= -args.stop_loss:
                        exit_price = current_price
                        exit_reason = "stop_loss"
                        exit_bar = b
                        break
                    elif ret >= args.take_profit:
                        exit_price = current_price
                        exit_reason = "take_profit"
                        exit_bar = b
                        break
                    else:
                        exit_price = current_price

                # PnL calculation
                if is_long:
                    gross_return = (exit_price - entry_price) / entry_price
                else:
                    gross_return = (entry_price - exit_price) / entry_price

                gross_pnl = gross_return * actual_trade_value
                net_pnl = gross_pnl - round_trip_cost

                # Actual magnitude
                actual_exit_bar = min(entry_bar + horizon_bars, n_bars - 1)
                actual_mag = (float(close[actual_exit_bar]) - entry_price) / entry_price

                trade = Trade(
                    symbol=sig["symbol"], date=day, direction=sig["direction"],
                    entry_bar=entry_bar, exit_bar=exit_bar,
                    entry_price=entry_price, exit_price=exit_price,
                    position_size=actual_trade_value, shares=shares,
                    gross_pnl=gross_pnl, net_pnl=net_pnl,
                    return_pct=gross_return * 100, exit_reason=exit_reason,
                    confidence=sig["confidence"], pred_prob=sig["prob"],
                    pred_magnitude=sig["magnitude"], actual_magnitude=actual_mag,
                )
                day_trades.append(trade)
                daily_pnl_so_far += net_pnl
                taken += 1

            # Daily stats
            daily_pnl = sum(t.net_pnl for t in day_trades)
            capital += daily_pnl
            peak_equity = max(peak_equity, capital)
            drawdown = (peak_equity - capital) / peak_equity if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            daily_results.append(DailyResult(
                date=day,
                trades_taken=len(day_trades),
                winning_trades=sum(1 for t in day_trades if t.net_pnl > 0),
                daily_pnl=daily_pnl,
                daily_return_pct=(daily_pnl / args.capital) * 100,
                cumulative_pnl=capital - args.capital,
                equity=capital,
                max_drawdown_pct=drawdown * 100,
            ))

            all_trades.extend(day_trades)
            progress.update(task, advance=1)

    # ── Results ──────────────────────────────────────────────────────────────

    console.print()

    if not all_trades:
        console.print("[yellow]⚠ No trades executed! Try lowering thresholds.[/yellow]")
        return

    # Summary stats
    total_pnl = sum(t.net_pnl for t in all_trades)
    win_trades = [t for t in all_trades if t.net_pnl > 0]
    lose_trades = [t for t in all_trades if t.net_pnl <= 0]
    long_trades = [t for t in all_trades if t.direction == "LONG"]
    short_trades = [t for t in all_trades if t.direction == "SHORT"]
    returns = [t.return_pct for t in all_trades]
    sharpe = (np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0

    # Summary table
    summary = Table(title="Backtest Summary", show_header=False, border_style="cyan")
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", style="bold")

    summary.add_row("Initial Capital", f"₹{args.capital:,.0f}")
    summary.add_row("Final Equity", f"₹{capital:,.0f}")
    pnl_style = "green" if total_pnl > 0 else "red"
    summary.add_row("Total P&L", f"[{pnl_style}]₹{total_pnl:,.0f}[/{pnl_style}]")
    summary.add_row("Total Return", f"[{pnl_style}]{(total_pnl/args.capital)*100:.2f}%[/{pnl_style}]")
    summary.add_row("", "")
    summary.add_row("Total Trades", f"{len(all_trades)}")
    summary.add_row("Winning Trades", f"[green]{len(win_trades)}[/green]")
    summary.add_row("Losing Trades", f"[red]{len(lose_trades)}[/red]")
    summary.add_row("Win Rate", f"{len(win_trades)/max(len(all_trades),1):.1%}")
    summary.add_row("", "")
    summary.add_row("Long Trades", f"{len(long_trades)}")
    summary.add_row("Short Trades", f"{len(short_trades)}")
    long_win = sum(1 for t in long_trades if t.net_pnl > 0)
    short_win = sum(1 for t in short_trades if t.net_pnl > 0)
    summary.add_row("Long Win Rate", f"{long_win/max(len(long_trades),1):.1%}")
    summary.add_row("Short Win Rate", f"{short_win/max(len(short_trades),1):.1%}")
    summary.add_row("", "")
    summary.add_row("Avg P&L/Trade", f"₹{np.mean([t.net_pnl for t in all_trades]):,.0f}")
    summary.add_row("Max Win", f"[green]₹{max(t.net_pnl for t in all_trades):,.0f}[/green]")
    summary.add_row("Max Loss", f"[red]₹{min(t.net_pnl for t in all_trades):,.0f}[/red]")
    summary.add_row("Sharpe Ratio", f"{sharpe:.2f}")
    summary.add_row("Max Drawdown", f"[red]{max_drawdown*100:.2f}%[/red]")
    summary.add_row("", "")
    gross_profit = sum(t.net_pnl for t in win_trades) if win_trades else 0
    gross_loss = abs(sum(t.net_pnl for t in lose_trades)) if lose_trades else 1
    summary.add_row("Profit Factor", f"{gross_profit/max(gross_loss,1):.2f}")

    # Exit reason breakdown
    sl_count = sum(1 for t in all_trades if t.exit_reason == "stop_loss")
    tp_count = sum(1 for t in all_trades if t.exit_reason == "take_profit")
    eod_count = sum(1 for t in all_trades if t.exit_reason == "eod")
    summary.add_row("", "")
    summary.add_row("Take Profit Exits", f"[green]{tp_count}[/green] ({tp_count/len(all_trades):.0%})")
    summary.add_row("Stop Loss Exits", f"[red]{sl_count}[/red] ({sl_count/len(all_trades):.0%})")
    summary.add_row("EOD Exits", f"{eod_count} ({eod_count/len(all_trades):.0%})")

    console.print(summary)

    # ── Monthly PnL Heatmap ──
    monthly_pnl = {}
    for dr in daily_results:
        month = dr.date[:7]
        monthly_pnl[month] = monthly_pnl.get(month, 0) + dr.daily_pnl

    month_table = Table(title="Monthly P&L")
    month_table.add_column("Month", style="cyan")
    month_table.add_column("P&L", justify="right")
    month_table.add_column("Return", justify="right")

    for month, pnl in sorted(monthly_pnl.items()):
        style = "green" if pnl > 0 else "red"
        ret = (pnl / args.capital) * 100
        month_table.add_row(month, f"[{style}]₹{pnl:,.0f}[/{style}]", f"[{style}]{ret:.2f}%[/{style}]")

    console.print()
    console.print(month_table)

    # ── Top Performing Stocks ──
    stock_pnl = {}
    stock_trades_count = {}
    for t in all_trades:
        stock_pnl[t.symbol] = stock_pnl.get(t.symbol, 0) + t.net_pnl
        stock_trades_count[t.symbol] = stock_trades_count.get(t.symbol, 0) + 1

    stock_table = Table(title="Top 10 Stocks by P&L")
    stock_table.add_column("Stock", style="cyan")
    stock_table.add_column("Trades", justify="right")
    stock_table.add_column("P&L", justify="right")

    sorted_stocks = sorted(stock_pnl.items(), key=lambda x: x[1], reverse=True)
    for sym, pnl in sorted_stocks[:10]:
        style = "green" if pnl > 0 else "red"
        stock_table.add_row(sym, str(stock_trades_count[sym]),
                            f"[{style}]₹{pnl:,.0f}[/{style}]")

    console.print()
    console.print(stock_table)

    # Bottom 5
    if len(sorted_stocks) > 10:
        bottom_table = Table(title="Bottom 5 Stocks by P&L")
        bottom_table.add_column("Stock", style="cyan")
        bottom_table.add_column("Trades", justify="right")
        bottom_table.add_column("P&L", justify="right")

        for sym, pnl in sorted_stocks[-5:]:
            style = "green" if pnl > 0 else "red"
            bottom_table.add_row(sym, str(stock_trades_count[sym]),
                                 f"[{style}]₹{pnl:,.0f}[/{style}]")

        console.print()
        console.print(bottom_table)

    # ── Save Results ──
    # Daily equity curve
    daily_df = pd.DataFrame([asdict(d) for d in daily_results])
    daily_df.to_csv(output_dir / "daily_equity.csv", index=False)

    # All trades
    trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    trades_df.to_csv(output_dir / "all_trades.csv", index=False)

    # Summary JSON
    summary_data = {
        "initial_capital": args.capital,
        "final_equity": capital,
        "total_pnl": total_pnl,
        "total_return_pct": (total_pnl / args.capital) * 100,
        "total_trades": len(all_trades),
        "win_rate": len(win_trades) / max(len(all_trades), 1),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "profit_factor": gross_profit / max(gross_loss, 1),
        "model": args.model,
        "model_type": model_type,
        "horizon": args.horizon,
        "params": vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    console.print(f"\n[bold green]✓ Results saved to {output_dir}/[/bold green]")
    console.print(f"  [dim]daily_equity.csv  — {len(daily_results)} days[/dim]")
    console.print(f"  [dim]all_trades.csv    — {len(all_trades)} trades[/dim]")
    console.print(f"  [dim]summary.json      — key metrics[/dim]\n")


if __name__ == "__main__":
    main()
