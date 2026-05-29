from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    checks: dict[str, bool]
    warnings: list[str]
    coverage: dict[str, object]


def evaluate_readiness(features: pd.DataFrame, option_chain: pd.DataFrame) -> ReadinessReport:
    warnings: list[str] = []
    checks = {
        "features_available": bool(not features.empty),
        "option_chain_available": bool(not option_chain.empty),
        "has_iv_context": bool("iv_rank_20d" in features.columns and features["iv_rank_20d"].notna().any()),
        "has_recent_expiry": False,
        "fresh_same_day": False,
    }
    coverage: dict[str, object] = {"indices": [], "latest_feature_date": None, "latest_chain_date": None}
    if not features.empty:
        coverage["indices"] = sorted(features["index"].dropna().unique().tolist())
        coverage["latest_feature_date"] = str(pd.to_datetime(features["date"]).max().date())
    if not option_chain.empty:
        latest_chain_date = pd.to_datetime(option_chain["date"]).max().normalize()
        coverage["latest_chain_date"] = str(latest_chain_date.date())
        latest_chain = option_chain[pd.to_datetime(option_chain["date"]).dt.normalize() == latest_chain_date]
        checks["has_recent_expiry"] = bool((pd.to_datetime(latest_chain["expiry"]) >= latest_chain_date).any())
        if not features.empty:
            checks["fresh_same_day"] = bool(pd.to_datetime(features["date"]).max().normalize() == latest_chain_date)
    for key, ok in checks.items():
        if not ok:
            warnings.append(key.replace("_", " ") + " check failed")
    status = "READY" if all(checks.values()) else "DEGRADED" if checks["features_available"] and checks["option_chain_available"] else "NOT_READY"
    return ReadinessReport(status=status, checks=checks, warnings=warnings, coverage=coverage)
