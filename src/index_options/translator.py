from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from index_options.config import INDEX_SPECS, PROFILE_SPECS
from index_options.math import black_scholes_delta


@dataclass(frozen=True)
class OptionTrade:
    index: str
    direction: str
    profile: str
    structure: str
    strike: float
    expiry: pd.Timestamp
    option_type: str
    entry: float
    target: float
    stop: float
    confidence: float
    score: float
    delta: float
    iv_rank: float
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "direction": self.direction,
            "profile": self.profile,
            "structure": self.structure,
            "strike": self.strike,
            "expiry": self.expiry.date().isoformat(),
            "option_type": self.option_type,
            "premium_entry": round(self.entry, 2),
            "premium_target": round(self.target, 2),
            "premium_stop": round(self.stop, 2),
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 6),
            "delta": round(self.delta, 4),
            "iv_rank": round(self.iv_rank, 4),
            "warnings": list(self.warnings),
        }


def _expiry_candidates(chain: pd.DataFrame, profile: str) -> pd.Series:
    expiries = chain[chain["days_to_expiry"] >= 0]["expiry"].drop_duplicates().sort_values()
    spec = PROFILE_SPECS[profile]
    if expiries.empty:
        return expiries
    if spec.prefer_monthly and len(expiries) > 1:
        monthly = expiries[expiries.dt.month == expiries.max().month]
        return monthly if not monthly.empty else expiries
    return expiries


def translate_signal(
    signal: pd.Series,
    option_chain: pd.DataFrame,
    feature_row: pd.Series,
    *,
    profile: str = "balanced",
    rate: float = 0.06,
) -> OptionTrade | None:
    spec = PROFILE_SPECS[profile]
    index_name = str(signal["index"])
    date = pd.Timestamp(signal["date"]).normalize()
    direction = str(signal["direction"]).upper()
    option_type = "CE" if direction == "LONG" else "PE"
    day_chain = option_chain[
        (option_chain["index"] == index_name)
        & (pd.to_datetime(option_chain["date"]).dt.normalize() == date)
        & (option_chain["option_type"] == option_type)
    ].copy()
    if day_chain.empty:
        return None
    if "days_to_expiry" not in day_chain.columns:
        day_chain["days_to_expiry"] = (pd.to_datetime(day_chain["expiry"]) - pd.to_datetime(day_chain["date"])).dt.days
    spot = float(day_chain["spot"].dropna().iloc[0]) if "spot" in day_chain and day_chain["spot"].notna().any() else np.nan
    if not np.isfinite(spot) or spot <= 0:
        return None
    iv_rank = float(feature_row.get("iv_rank_20d", 0.5))
    iv = float(np.nanmean([feature_row.get("atm_iv_proxy_call", np.nan), feature_row.get("atm_iv_proxy_put", np.nan)]))
    if not np.isfinite(iv) or iv <= 0:
        iv = 0.20
    warnings: list[str] = []
    if iv_rank > 0.70:
        structure = "debit_spread"
        warnings.append("High IV: spread preferred over naked buy.")
    else:
        structure = "naked_buy"
    if iv_rank < 0.30:
        structure = "naked_buy"
    if bool(feature_row.get("is_expiry_day", 0.0)) and not spec.allow_expiry_day:
        warnings.append("Conservative profile skips expiry-day theta risk.")
        return None

    candidates = _expiry_candidates(day_chain, profile)
    if candidates.empty:
        return None
    expiry = pd.Timestamp(candidates.iloc[0])
    expiry_chain = day_chain[day_chain["expiry"] == expiry].copy()
    expiry_chain = expiry_chain[expiry_chain["close"] > 0]
    if expiry_chain.empty:
        return None
    t = max(float((expiry - date).days), 1.0) / 365.0
    expiry_chain["delta"] = expiry_chain["strike"].map(
        lambda strike: black_scholes_delta(spot, float(strike), t, rate, iv, option_type)
    )
    target_delta = (spec.min_delta + spec.max_delta) / 2.0
    in_band = expiry_chain[(expiry_chain["delta"] >= spec.min_delta) & (expiry_chain["delta"] <= spec.max_delta)]
    choose_from = in_band if not in_band.empty else expiry_chain
    row = choose_from.iloc[(choose_from["delta"] - target_delta).abs().argsort().iloc[0]]
    entry = float(row["close"])
    delta = float(row["delta"])
    premium_target_pct = spec.target_pct * delta * spot / entry
    premium_stop_pct = spec.stop_pct * delta * spot / entry
    target = entry * (1.0 + max(premium_target_pct, 0.05))
    stop = entry * (1.0 - min(max(premium_stop_pct, 0.03), 0.80))
    return OptionTrade(
        index=index_name,
        direction=direction,
        profile=profile,
        structure=structure,
        strike=float(row["strike"]),
        expiry=expiry,
        option_type=option_type,
        entry=entry,
        target=target,
        stop=stop,
        confidence=float(signal["confidence"]),
        score=float(signal["score"]),
        delta=delta,
        iv_rank=iv_rank,
        warnings=tuple(warnings),
    )


def translate_ranked_signals(
    signals: pd.DataFrame,
    option_chain: pd.DataFrame,
    features: pd.DataFrame,
    *,
    profile: str = "balanced",
    top_k: int = 4,
) -> list[OptionTrade]:
    trades: list[OptionTrade] = []
    feature_lookup = features.set_index(["index", "date"])
    ranked = signals.sort_values("score", ascending=False)
    for _, signal in ranked.iterrows():
        key = (signal["index"], pd.Timestamp(signal["date"]))
        if key not in feature_lookup.index:
            continue
        trade = translate_signal(signal, option_chain, feature_lookup.loc[key], profile=profile)
        if trade is not None:
            trades.append(trade)
        if len(trades) >= top_k:
            break
    return trades
