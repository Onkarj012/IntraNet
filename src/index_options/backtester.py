from __future__ import annotations

import numpy as np
import pandas as pd

from index_options.config import INDEX_SPECS
from index_options.models import OptiNetModelBundle, score_frame
from index_options.translator import OptionTrade, translate_ranked_signals


def _next_trade_date(dates: pd.Series, date: pd.Timestamp) -> pd.Timestamp | None:
    future = pd.to_datetime(dates[pd.to_datetime(dates) > date]).sort_values().unique()
    if len(future) == 0:
        return None
    return pd.Timestamp(future[0])


def _exit_trade(trade: OptionTrade, option_chain: pd.DataFrame, entry_date: pd.Timestamp) -> dict[str, object] | None:
    row = option_chain[
        (option_chain["index"] == trade.index)
        & (pd.to_datetime(option_chain["date"]).dt.normalize() == entry_date)
        & (option_chain["expiry"] == trade.expiry)
        & (option_chain["strike"] == trade.strike)
        & (option_chain["option_type"] == trade.option_type)
    ]
    if row.empty:
        return None
    bar = row.iloc[0]
    entry = float(bar["open"]) if float(bar.get("open", 0.0)) > 0 else trade.entry
    target = entry * (trade.target / trade.entry)
    stop = entry * (trade.stop / trade.entry)
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])
    if low <= stop:
        exit_price = stop
        reason = "stop"
    elif high >= target:
        exit_price = target
        reason = "target"
    else:
        exit_price = close
        reason = "close"
    lot_size = INDEX_SPECS.get(trade.index, INDEX_SPECS["NIFTY"]).lot_size
    gross = (exit_price - entry) * lot_size
    costs = max(entry * lot_size * 0.001, 1.0)
    return {
        **trade.as_dict(),
        "signal_date": pd.Timestamp(entry_date) - pd.offsets.BDay(1),
        "entry_date": entry_date,
        "actual_entry": round(entry, 2),
        "exit_price": round(exit_price, 2),
        "exit_reason": reason,
        "gross_pnl": round(gross, 2),
        "costs": round(costs, 2),
        "net_pnl": round(gross - costs, 2),
        "return_pct": round((exit_price - entry) / entry if entry else 0.0, 6),
    }


def backtest_daily(
    bundle: OptiNetModelBundle,
    features: pd.DataFrame,
    option_chain: pd.DataFrame,
    *,
    profile: str = "balanced",
    min_confidence: float = 0.40,
    top_k_per_day: int = 2,
    apply_regime_filter: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    scores = score_frame(bundle, features, apply_regime_filter=apply_regime_filter)
    rows: list[dict[str, object]] = []
    all_dates = pd.to_datetime(option_chain["date"]).drop_duplicates().sort_values()
    for date, day_scores in scores.groupby("date"):
        day_scores = day_scores[day_scores["confidence"] >= min_confidence]
        if day_scores.empty:
            continue
        day_features = features[pd.to_datetime(features["date"]) == pd.Timestamp(date)]
        trades = translate_ranked_signals(
            day_scores,
            option_chain,
            day_features,
            profile=profile,
            top_k=top_k_per_day,
        )
        entry_date = _next_trade_date(all_dates, pd.Timestamp(date))
        if entry_date is None:
            continue
        for trade in trades:
            result = _exit_trade(trade, option_chain, entry_date)
            if result is not None:
                result["signal_date"] = pd.Timestamp(date)
                rows.append(result)
    trades_df = pd.DataFrame(rows)
    return trades_df, summarize_trades(trades_df)


def summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    pnl = trades["net_pnl"].astype(float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    returns = trades["return_pct"].astype(float)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    return {
        "trades": float(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "net_pnl": float(pnl.sum()),
        "avg_return_pct": float(returns.mean()),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "target_exit_rate": float((trades["exit_reason"] == "target").mean()),
        "stop_exit_rate": float((trades["exit_reason"] == "stop").mean()),
        "close_exit_rate": float((trades["exit_reason"] == "close").mean()),
    }
