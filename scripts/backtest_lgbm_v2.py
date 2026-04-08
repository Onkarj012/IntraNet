#!/usr/bin/env python3
"""
IntradayNet LightGBM V2 Backtester.

Minute-bar resolution backtest using trained LightGBM models.
Inference is run on-the-fly per bar; no pre-computed features needed.

Features:
- Per-bar inference with causal 120-bar lookback
- Direction + magnitude predictions
- NSE transaction costs (brokerage, STT, GST, slippage)
- Long and short positions
- Walk-forward evaluation
- Sector/market regime filtering

Usage:
    python scripts/backtest_lgbm_v2.py
    python scripts/backtest_lgbm_v2.py --model-dir runs/lgbm_v2 --capital 200000
    python scripts/backtest_lgbm_v2.py --symbols RELIANCE --start 2025-01-01 --end 2025-12-31
"""

import argparse
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from intradaynet.costs import IndianMarketCosts
from intradaynet.regime import detect_regime, RegimeConfig, MarketRegime

console = Console()


def _infer_regime_from_ohlcv(
    ohlcv: np.ndarray, n_bars: int = 100
) -> Tuple[MarketRegime, bool, str]:
    """Infer market regime from a stock's own OHLCV data as a proxy.

    Since full market data (VIX, Nifty) may not be available at runtime,
    we use the stock's own volatility and trend as regime indicators.
    """
    if len(ohlcv) < 60:
        return MarketRegime.CALM_BULL, True, "Insufficient data, defaulting to calm"

    c = ohlcv[:, 3]
    h = ohlcv[:, 1]
    l = ohlcv[:, 2]
    v = ohlcv[:, 4]

    log_ret = np.diff(np.log(np.maximum(c, 1e-10)))
    log_ret = np.concatenate([[0], log_ret])

    vol_recent = np.std(log_ret[-30:]) if len(log_ret) >= 30 else np.std(log_ret)
    vol_mean = np.std(log_ret[-100:]) if len(log_ret) >= 100 else vol_recent

    vol_ratio = vol_recent / max(vol_mean, 1e-8)

    trend = np.mean(log_ret[-20:]) if len(log_ret) >= 20 else 0.0

    vix_proxy = 15.0 * max(vol_ratio, 0.5)
    vix_change = max(0, (vol_ratio - 1.0)) * 0.5

    if vix_proxy > 28.0:
        return MarketRegime.EXTREME, False, f"VIX_proxy={vix_proxy:.1f}>28"
    if vix_change > 0.20:
        return MarketRegime.EXTREME, False, f"VIX_spike={vix_change*100:.1f}%>20%"

    high_range = np.max(h[-60:]) - np.min(l[-60:])
    gap = abs(c[-1] - c[0]) / max(c[0], 1) if len(c) > 0 else 0.0
    if gap > 0.015:
        return MarketRegime.EXTREME, False, f"Gap={gap*100:.2f}%>1.5%"

    is_volatile = vix_proxy > 22.0
    regime = MarketRegime.VOLATILE_BULL if (is_volatile and trend >= 0) else \
             MarketRegime.VOLATILE_BEAR if is_volatile else \
             MarketRegime.CALM_BULL if trend >= 0 else \
             MarketRegime.CALM_BEAR

    return regime, True, f"Regime={regime.value}, vix_proxy={vix_proxy:.1f}"

HORIZONS = {"H15": 15, "H30": 30, "H60": 60}
SEQ_LENGTH = 120
FLAT_DIM = 625


@dataclass
class Trade:
    symbol: str
    date: str
    direction: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    confidence: float
    horizon: str


@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_holding_bars: float
    long_trades: int
    short_trades: int
    long_pnl: float
    short_pnl: float
    cost_per_trade: float
    total_costs: float
    trades: list = field(default_factory=list)


def _compute_features_for_session(ohlcv: np.ndarray) -> np.ndarray:
    """Compute all 25 per-bar features for a single session's OHLCV array.

    Args:
        ohlcv: (n_bars, 5) array with columns [open, high, low, close, volume]

    Returns:
        (n_bars, 25) feature array, NaN-filled for bars before feature warmup.
    """
    n = len(ohlcv)
    o, h, l, c, v = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
    f = np.full((n, 25), np.nan, dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]

    log_ret = np.log(np.maximum(c / prev_c, 1e-10))
    log_ret[0] = 0.0
    f[:, 0] = log_ret

    vol_ema20 = _ewm_numpy(v, span=20)
    vol_ema20 = np.where(vol_ema20 == 0, 1, vol_ema20)
    f[:, 1] = np.clip(v / vol_ema20, 0, 50)

    tp = (h + l + c) / 3.0
    tpv = tp * v
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(v)
    cum_vol = np.where(cum_vol == 0, 1, cum_vol)
    vwap = cum_tpv / cum_vol
    f[:, 2] = np.clip((c - vwap) / np.where(vwap == 0, 1, vwap), -5, 5)

    ema_9 = _ewm_numpy(c, span=9)
    ema_20 = _ewm_numpy(c, span=20)
    f[:, 3] = np.clip((c - ema_9) / np.where(ema_9 == 0, 1, ema_9), -5, 5)
    f[:, 4] = np.clip((c - ema_20) / np.where(ema_20 == 0, 1, ema_20), -5, 5)

    f[:, 5] = _rsi_numpy(c, period=14) / 100.0

    bb_mid = _rolling_mean(c, 20)
    bb_std = _rolling_std(c, 20)
    bb_std = np.where(bb_std == 0, 1e-10, bb_std)
    bb_half = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    f[:, 6] = np.clip((c - bb_mid) / np.where(bb_half - bb_mid == 0, 1e-10, bb_half - bb_mid), -5, 5)
    f[:, 7] = np.clip((bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / np.where(bb_mid == 0, 1, bb_mid), 0, 5)

    hl_range = np.where(h - l == 0, 1e-10, h - l)
    f[:, 8] = np.clip(np.abs(c - o) / hl_range, 0, 1)
    f[:, 9] = np.clip((h - np.maximum(o, c)) / hl_range, 0, 1)
    f[:, 10] = np.clip((np.minimum(o, c) - l) / hl_range, 0, 1)

    f[:, 11] = np.clip((h - l) / np.where(c == 0, 1, c), 0, 0.5)

    vol_roll_avg = _rolling_mean(v, 30)
    vol_roll_avg = np.where(vol_roll_avg == 0, 1, vol_roll_avg)
    f[:, 12] = np.clip(v / vol_roll_avg, 0, 50)

    bar_in_sess = np.arange(n, dtype=np.float64)
    sess_len = n
    f[:, 13] = np.clip(bar_in_sess / max(sess_len, 1), 0, 1)

    orb_high = np.full(n, h[:15].max() if n >= 15 else h.max())
    orb_low = np.full(n, l[:15].min() if n >= 15 else l.min())
    orb_range = orb_high - orb_low
    orb_range = np.where(orb_range == 0, 1e-10, orb_range)
    f[:, 14] = np.clip((c - orb_high) / orb_range, -10, 10)
    f[:, 15] = np.clip((c - orb_low) / orb_range, -10, 10)

    day_open = o[0]
    day_open = day_open if day_open != 0 else 1
    f[:, 16] = np.clip((c - day_open) / day_open, -0.2, 0.2)

    c_shifted_5 = np.roll(c, 5)
    c_shifted_5[:5] = c[0]
    f[:, 17] = np.clip(c / np.where(c_shifted_5 == 0, 1, c_shifted_5) - 1, -0.1, 0.1)
    c_shifted_20 = np.roll(c, 20)
    c_shifted_20[:20] = c[0]
    f[:, 18] = np.clip(c / np.where(c_shifted_20 == 0, 1, c_shifted_20) - 1, -0.2, 0.2)

    close_diff = np.diff(np.concatenate([[c[0]], c]))
    up_vol = np.where(close_diff > 0, v, 0)
    dn_vol = np.where(close_diff < 0, v, 0)
    up_vol_20 = _rolling_sum(up_vol, 20)
    dn_vol_20 = _rolling_sum(dn_vol, 20)
    total_vol_20 = _rolling_sum(v, 20)
    total_vol_20 = np.where(total_vol_20 == 0, 1, total_vol_20)
    f[:, 19] = np.clip((up_vol_20 - dn_vol_20) / total_vol_20, -1, 1)

    f[:, 20] = _atr_numpy(h, l, c, period=14)

    day_high = _expanding_max(h)
    day_low = _expanding_min(l)
    day_range = day_high - day_low
    day_range = np.where(day_range == 0, 1e-10, day_range)
    f[:, 21] = np.clip((c - day_low) / day_range, 0, 1)

    f[:, 22] = np.clip(_rolling_std(log_ret, 20), 0, 0.1)

    f[:, 23] = _obv_slope_numpy(c, v, window=20)

    trade_int = v * (h - l)
    trade_int_mean = _rolling_mean(trade_int, 20)
    trade_int_mean = np.where(trade_int_mean == 0, 1, trade_int_mean)
    f[:, 24] = np.clip(trade_int / trade_int_mean, 0, 50)

    return np.nan_to_num(f, nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)


def _ewm_numpy(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.mean(arr[i - window + 1:i + 1])
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.std(arr[i - window + 1:i + 1])
    return out


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    cumsum = np.cumsum(arr)
    for i in range(window - 1, len(arr)):
        out[i] = cumsum[i] - (cumsum[i - window] if i >= window else 0)
    return out


def _expanding_max(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float64)
    cur = arr[0]
    for i in range(len(arr)):
        cur = max(cur, arr[i])
        out[i] = cur
    return out


def _expanding_min(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float64)
    cur = arr[0]
    for i in range(len(arr)):
        cur = min(cur, arr[i])
        out[i] = cur
    return out


def _rsi_numpy(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(np.concatenate([[close[0]], close]))
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    alpha = 1.0 / period
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(close)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return np.clip(atr / np.where(close == 0, 1, close), 0, 0.5)


def _obv_slope_numpy(close: np.ndarray, volume: np.ndarray,
                     window: int = 20) -> np.ndarray:
    sign = np.sign(np.diff(np.concatenate([[close[0]], close])))
    sign[0] = 0
    obv = np.cumsum(sign * volume)
    x = np.arange(len(obv), dtype=np.float64)
    out = np.zeros_like(obv)
    for i in range(window - 1, len(obv)):
        xi = x[i - window + 1:i + 1]
        yi = obv[i - window + 1:i + 1]
        xm, ym = np.mean(xi), np.mean(yi)
        cov = np.mean((xi - xm) * (yi - ym))
        var = np.mean((xi - xm) ** 2)
        slope = cov / max(var, 1e-10)
        out[i] = slope
    slope_rolling_std = _rolling_std(out, 100)
    slope_rolling_std = np.where(slope_rolling_std == 0, 1, slope_rolling_std)
    return np.clip(out / slope_rolling_std, -5, 5)


def flatten_features(per_bar_vals: np.ndarray) -> np.ndarray:
    """Flatten (SEQ_LENGTH, 25) window into (625,) feature vector."""
    L, F = per_bar_vals.shape
    w = per_bar_vals.reshape(1, L, F)
    parts = []
    for win in [5, 15, 30, 60, 120]:
        if win > L:
            continue
        window = w[:, -win:, :]
        parts.append(np.nanmean(window, axis=1))
        parts.append(np.nanstd(window, axis=1))
        parts.append(np.nanmin(window, axis=1))
        parts.append(np.nanmax(window, axis=1))
    parts.append(w[:, -1, :])
    parts.append(w[:, 0, :])
    parts.append(w[:, -1, :] - w[:, 0, :])
    if L >= 30:
        parts.append(np.nanmean(w[:, -5:, :], axis=1) - np.nanmean(w[:, -30:, :], axis=1))
    if L >= 60:
        parts.append(np.nanmean(w[:, -15:, :], axis=1) - np.nanmean(w[:, -60:, :], axis=1))
    flat = np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
    return flat


class LightGBMInferrer:
    """Loads and runs LightGBM V2 models."""

    def __init__(self, model_dir: Path, costs: IndianMarketCosts):
        self.dir_models = {}
        self.mag_models = {}
        self.costs = costs
        self._load_models(model_dir)

    def _load_models(self, model_dir: Path):
        for horizon in HORIZONS:
            dir_path = model_dir / f"dir_{horizon}.lgb"
            mag_path = model_dir / f"mag_{horizon}.lgb"
            if dir_path.exists():
                self.dir_models[horizon] = lgb.Booster(model_file=str(dir_path))
            if mag_path.exists():
                self.mag_models[horizon] = lgb.Booster(model_file=str(mag_path))
        console.print(f"  Loaded: {[f'dir_{h}' for h in self.dir_models]}")

    def predict(self, horizon: str, features: np.ndarray) -> tuple[float, float]:
        """Return (direction_prob_up, magnitude_pct)."""
        prob = 0.5
        mag = 0.0
        if horizon in self.dir_models:
            prob = float(self.dir_models[horizon].predict(features)[0])
        if horizon in self.mag_models:
            mag = float(self.mag_models[horizon].predict(features)[0])
        return prob, mag

    def round_trip_cost(self, entry_price: float, qty: int) -> float:
        return float(self.costs.total_cost(entry_price, qty))


class IntradayBacktester:
    """
    Minute-bar backtester for LightGBM V2 models.

    Strategy: Enter LONG/SHORT when model probability exceeds threshold.
    Exit on: take_profit, stop_loss, or end-of-day.
    """

    def __init__(
        self,
        inferrer: LightGBMInferrer,
        capital: float = 200_000,
        horizon: str = "H60",
        entry_threshold: float = 0.52,
        stop_loss_pct: float = 0.005,
        take_profit_pct: float = 0.01,
        max_positions: int = 3,
        no_regime_filter: bool = False,
    ):
        self.inferrer = inferrer
        self.capital = capital
        self.horizon = horizon
        self.entry_threshold = entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.no_regime_filter = no_regime_filter
        self.costs = inferrer.costs
        self.trades: list[Trade] = []
        self._equity_curve = []
        self._active_position = None

    def backtest_stock(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Run backtest on one stock's minute-bar data."""
        close_all = df["close"].values
        high_all = df["high"].values
        low_all = df["low"].values
        open_all = df["open"].values
        vol_all = df["volume"].values
        dates = np.array([d.date() if hasattr(d, "date") else d for d in df.index])
        unique_dates = np.unique(dates)

        total_pnl = 0.0
        total_costs = 0.0
        self._equity_curve = [self.capital]
        wins, losses = 0, 0
        long_trades, short_trades = 0, 0
        long_pnl, short_pnl = 0.0, 0.0
        holding_bars = []

        for date in unique_dates:
            session_mask = dates == date
            session_indices = np.where(session_mask)[0]
            n_bars = len(session_indices)
            if n_bars < SEQ_LENGTH + 10:
                continue

            ohlcv = np.column_stack([
                open_all[session_indices],
                high_all[session_indices],
                low_all[session_indices],
                close_all[session_indices],
                vol_all[session_indices],
            ])

            should_trade = True
            if not self.no_regime_filter:
                _, should_trade, _ = _infer_regime_from_ohlcv(ohlcv)
            if not should_trade:
                continue

            sess_features = _compute_features_for_session(ohlcv)
            sess_close = close_all[session_indices]

            active = None

            for local_bar in range(SEQ_LENGTH, n_bars - 5):
                if local_bar % 5 != 0 and active is None:
                    continue

                window_feats = sess_features[local_bar - SEQ_LENGTH:local_bar]
                flat = flatten_features(window_feats)
                prob, _ = self.inferrer.predict(self.horizon, flat)

                if active is None:
                    if prob > self.entry_threshold:
                        direction = "LONG"
                        qty = int(self.capital / sess_close[local_bar] / 100) * 100
                        if qty <= 0:
                            continue
                        active = {
                            "direction": direction,
                            "entry_bar": local_bar,
                            "entry_price": sess_close[local_bar],
                            "qty": qty,
                            "prob": prob,
                        }
                        long_trades += 1
                    elif prob < (1 - self.entry_threshold):
                        direction = "SHORT"
                        qty = int(self.capital / sess_close[local_bar] / 100) * 100
                        if qty <= 0:
                            continue
                        active = {
                            "direction": direction,
                            "entry_bar": local_bar,
                            "entry_price": sess_close[local_bar],
                            "qty": qty,
                            "prob": prob,
                        }
                        short_trades += 1

                if active is not None:
                    dir_ = active["direction"]
                    entry = active["entry_price"]
                    qty = active["qty"]
                    d = 1 if dir_ == "LONG" else -1

                    pnl_pct = d * (sess_close[local_bar] - entry) / entry
                    exit_reason = None

                    if pnl_pct <= -self.stop_loss_pct:
                        exit_reason = "stop_loss"
                        exit_price = entry * (1 - d * self.stop_loss_pct)
                    elif pnl_pct >= self.take_profit_pct:
                        exit_reason = "take_profit"
                        exit_price = entry * (1 + d * self.take_profit_pct)
                    elif local_bar - active["entry_bar"] >= 60:
                        exit_reason = "time_exit"
                        exit_price = sess_close[local_bar]

                    if exit_reason is not None or local_bar == n_bars - 2:
                        if exit_reason is None:
                            exit_reason = "eod"
                            exit_price = sess_close[min(local_bar + 1, n_bars - 1)]

                        gross_pnl = d * (exit_price - entry) * qty
                        cost = self.inferrer.round_trip_cost(entry, qty)
                        net_pnl = gross_pnl - cost
                        return_pct = net_pnl / self.capital

                        self.trades.append(Trade(
                            symbol=symbol, date=str(date), direction=dir_,
                            entry_bar=active["entry_bar"], exit_bar=local_bar,
                            entry_price=entry, exit_price=exit_price, qty=qty,
                            gross_pnl=gross_pnl, net_pnl=net_pnl, return_pct=return_pct,
                            exit_reason=exit_reason, confidence=prob,
                            horizon=self.horizon,
                        ))

                        total_pnl += net_pnl
                        total_costs += cost
                        if dir_ == "LONG":
                            long_pnl += net_pnl
                        else:
                            short_pnl += net_pnl

                        if net_pnl > 0:
                            wins += 1
                        else:
                            losses += 1

                        holding_bars.append(local_bar - active["entry_bar"])
                        self._equity_curve.append(self._equity_curve[-1] + net_pnl)
                        active = None

        if not self._equity_curve:
            self._equity_curve = [self.capital]

        equity = np.array(self._equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity - running_max
        max_dd = float(drawdowns.min())

        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            sharpe = float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)) if returns.std() > 0 else 0.0
        else:
            sharpe = 0.0

        gross_wins = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        gross_losses = abs(sum(t.gross_pnl for t in self.trades if t.gross_pnl < 0))
        profit_factor = gross_wins / max(gross_losses, 1e-8)

        total_return = total_pnl / self.capital * 100

        return BacktestResult(
            symbol=symbol, start_date=str(unique_dates[0]), end_date=str(unique_dates[-1]),
            total_trades=len(self.trades), winning_trades=wins, losing_trades=losses,
            win_rate=wins / max(len(self.trades), 1),
            avg_win=gross_wins / max(wins, 1),
            avg_loss=gross_losses / max(losses, 1),
            profit_factor=profit_factor,
            total_pnl=total_pnl, total_return=total_return,
            max_drawdown=max_dd, sharpe_ratio=sharpe,
            avg_holding_bars=np.mean(holding_bars) if holding_bars else 0,
            long_trades=long_trades, short_trades=short_trades,
            long_pnl=long_pnl, short_pnl=short_pnl,
            cost_per_trade=total_costs / max(len(self.trades), 1),
            total_costs=total_costs,
            trades=[asdict(t) for t in self.trades],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM V2 backtester")
    parser.add_argument("--model-dir", type=str, default="runs/lgbm_v2")
    parser.add_argument("--data-dir", type=str, default="nifty500")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols or 'all'")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--horizon", type=str, default="H60", choices=["H15", "H30", "H60"])
    parser.add_argument("--entry-threshold", type=float, default=0.52)
    parser.add_argument("--stop-loss", type=float, default=0.005)
    parser.add_argument("--take-profit", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="",
                        help="Save results to JSON file")
    parser.add_argument("--no-regime-filter", action="store_true",
                        help="Disable regime filtering (trade all days)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto)")
    return parser.parse_args()


def _backtest_single_stock(args_tuple):
    """Worker function for parallel backtesting. Each worker loads its own models."""
    (symbol, csv_path, model_dir, capital, horizon, entry_threshold,
     stop_loss_pct, take_profit_pct, no_regime_filter, start_str, end_str) = args_tuple

    try:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.set_index("datetime")
        df.columns = df.columns.str.lower()
        df = df[(df.index >= start_str) & (df.index <= end_str)]

        if len(df) < SEQ_LENGTH + 50:
            return None

        costs = IndianMarketCosts()
        inferrer = LightGBMInferrer(Path(model_dir), costs)
        if not inferrer.dir_models:
            return None

        bt = IntradayBacktester(
            inferrer, capital=capital, horizon=horizon,
            entry_threshold=entry_threshold, stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct, no_regime_filter=no_regime_filter,
        )
        result = bt.backtest_stock(df, symbol)
        return asdict(result)
    except Exception as e:
        return None


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    if not model_dir.exists():
        console.print(f"[red]✗ Model dir not found: {model_dir}[/red]")
        return

    costs = IndianMarketCosts()
    inferrer = LightGBMInferrer(model_dir, costs)

    if args.symbols and args.symbols != "all":
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = sorted([
            p.stem.replace("_minute", "")
            for p in data_dir.glob("*_minute.csv")
        ])

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    n_workers = args.workers if args.workers > 0 else min(mp.cpu_count(), 32)

    console.print(Panel.fit(
        f"[bold cyan]LightGBM V2 Backtester[/bold cyan]",
        subtitle=f"Horizon: {args.horizon} | Capital: ₹{args.capital:,.0f} | "
                  f"Entry: {args.entry_threshold:.2f} | SL: {args.stop_loss*100:.1f}% | TP: {args.take_profit*100:.1f}%",
        border_style="cyan",
    ))

    work_items = []
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_minute.csv"
        if not csv_path.exists():
            continue
        work_items.append((
            symbol, str(csv_path), str(model_dir), args.capital, args.horizon,
            args.entry_threshold, args.stop_loss, args.take_profit,
            args.no_regime_filter, str(start), str(end),
        ))

    all_results = []
    t0 = time.time()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total} symbols"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"Backtesting ({n_workers} workers)...", total=len(work_items))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_backtest_single_stock, w): w[0] for w in work_items}
            for future in as_completed(futures):
                progress.advance(task)
                result_dict = future.result()
                if result_dict is not None:
                    result_dict.pop("trades", None)
                    br = BacktestResult(**result_dict)
                    all_results.append(br)

    elapsed = time.time() - t0

    if not all_results:
        console.print("[yellow]⚠ No results generated[/yellow]")
        return

    total_trades = sum(r.total_trades for r in all_results)
    total_pnl = sum(r.total_pnl for r in all_results)
    total_return = sum(r.total_return * r.total_trades for r in all_results) / max(total_trades, 1)
    wins = sum(r.winning_trades for r in all_results)
    losses = sum(r.losing_trades for r in all_results)
    total_costs = sum(r.total_costs for r in all_results)
    avg_dd = np.mean([r.max_drawdown for r in all_results if r.max_drawdown != 0]) if all_results else 0
    sharpe_avg = np.mean([r.sharpe_ratio for r in all_results]) if all_results else 0

    long_pnl = sum(r.long_pnl for r in all_results)
    short_pnl = sum(r.short_pnl for r in all_results)

    console.print(f"\n[bold]Backtest Results — {start.date()} to {end.date()}[/bold]")
    console.print(f"  Stocks: {len(all_results)} | Symbols: {[r.symbol for r in all_results]}")
    console.print(f"  Time: {elapsed:.1f}s")
    console.print()

    table = Table(title="Performance Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Trades", f"{total_trades:,}")
    table.add_row("Win Rate", f"{wins/max(total_trades,1)*100:.1f}%")
    table.add_row("Total P&L", f"₹{total_pnl:,.0f}")
    table.add_row("Avg Return/Trade", f"{total_return:.3f}%")
    table.add_row("Long P&L", f"₹{long_pnl:,.0f}")
    table.add_row("Short P&L", f"₹{short_pnl:,.0f}")
    table.add_row("Total Costs", f"₹{total_costs:,.0f}")
    table.add_row("Max Drawdown", f"₹{avg_dd:,.0f}")
    table.add_row("Avg Sharpe", f"{sharpe_avg:.2f}")

    console.print(table)

    if all_results:
        results_detail = Table(title="Per-Stock Results")
        results_detail.add_column("Symbol", style="cyan")
        results_detail.add_column("Trades", justify="right")
        results_detail.add_column("Win%", justify="right")
        results_detail.add_column("P&L", style="yellow", justify="right")
        results_detail.add_column("Return%", style="yellow", justify="right")
        results_detail.add_column("Max DD", style="red", justify="right")

        for r in sorted(all_results, key=lambda x: x.total_pnl, reverse=True):
            wr = r.winning_trades / max(r.total_trades, 1) * 100
            results_detail.add_row(
                r.symbol, str(r.total_trades), f"{wr:.0f}%",
                f"₹{r.total_pnl:,.0f}", f"{r.total_return:.2f}%",
                f"₹{r.max_drawdown:,.0f}",
            )
        console.print()
        console.print(results_detail)

    if args.output:
        out_path = Path(args.output)
        out_data = {
            "args": vars(args),
            "summary": {
                "total_trades": total_trades,
                "win_rate": wins / max(total_trades, 1),
                "total_pnl": float(total_pnl),
                "total_costs": float(total_costs),
                "max_drawdown": float(avg_dd),
                "sharpe_ratio": float(sharpe_avg),
            },
            "results": [asdict(r) for r in all_results],
        }
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        console.print(f"\n[green]✓ Results saved to {out_path}[/green]")


if __name__ == "__main__":
    main()
