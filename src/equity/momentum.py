"""Single source of truth for the long-only momentum factor.

Shared by the backtest (scripts/research/factor_portfolio.py) and the live
picks generator / paper ledger so live decisions are identical to what was
validated. All inputs are point-in-time (use data through the decision date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Defaults are the validated durable config (see factor_portfolio OOS results).
TOP_N = 20
REBALANCE_DAYS = 10
MIN_PRICE = 20.0
MIN_ADV = 5e7          # ₹5 cr 21d avg daily traded value
WEIGHT = "invvol"
TREND_MA = 200


def load_panel(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = pd.read_parquet(path)
    return store.xs("close", axis=1, level=0), store.xs("volume", axis=1, level=0)


def _z(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def momentum_score(close: pd.DataFrame) -> pd.DataFrame:
    """Average of 4 cross-sectionally z-scored momentum horizons (12-1 included)."""
    rets = {h: close.shift(s) / close.shift(h) - 1 for h, s in [(21, 0), (63, 0), (126, 0), (252, 21)]}
    return sum(_z(r) for r in rets.values()) / len(rets)


def vol63(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change().rolling(63, min_periods=21).std()


def adv(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    return (close * vol).rolling(21, min_periods=10).mean()


def market_index(close: pd.DataFrame) -> pd.Series:
    """Equal-weight universe index (returns clipped for data-error robustness)."""
    return (1 + close.pct_change().clip(-0.5, 0.5).mean(axis=1).fillna(0.0)).cumprod()


def trend_on(close: pd.DataFrame, date, ma: int = TREND_MA) -> bool:
    ew = market_index(close)
    ma_s = ew.rolling(ma, min_periods=ma // 2).mean()
    return bool(ew.loc[date] >= ma_s.loc[date])


def select(
    date,
    close: pd.DataFrame,
    score: pd.DataFrame,
    v63: pd.DataFrame,
    adv_panel: pd.DataFrame,
    *,
    top_n: int = TOP_N,
    min_price: float = MIN_PRICE,
    min_adv: float = MIN_ADV,
    weight: str = WEIGHT,
    use_trend: bool = True,
) -> tuple[dict[str, float], str]:
    """Return ({symbol: weight}, state) for `date`. Empty dict if not invested."""
    if use_trend and not trend_on(close, date):
        return {}, "risk_off"
    s = score.loc[date].dropna()
    px = close.loc[date].reindex(s.index)
    a = adv_panel.loc[date].reindex(s.index)
    elig = s[(px >= min_price) & (a >= min_adv)].dropna()
    if len(elig) < top_n:
        return {}, "insufficient_universe"
    picks = list(elig.nlargest(top_n).index)
    if weight == "invvol":
        iv = (1.0 / v63.loc[date, picks].replace(0, np.nan)).fillna(0.0)
        w = (iv / iv.sum()) if iv.sum() > 0 else pd.Series(1.0 / top_n, index=picks)
    elif weight == "equal":
        w = pd.Series(1.0 / top_n, index=picks)
    else:
        raise ValueError(f"Unsupported weight mode: {weight!r}. Use 'invvol' or 'equal'.")
    return {sym: float(w[sym]) for sym in picks}, "invested"
