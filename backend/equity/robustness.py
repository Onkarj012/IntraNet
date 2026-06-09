from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONFIDENCE_BUCKETS = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.55),
    (0.55, 0.65),
    (0.65, 0.75),
    (0.75, 1.0000001),
)


@dataclass(frozen=True)
class PromotionGateConfig:
    min_blind_trades: int
    min_blind_pnl: float = 0.0
    min_blind_sharpe: float = 0.0
    max_stop_rate: float = 0.60
    allow_confidence_inversion: bool = False


@dataclass(frozen=True)
class RiskPolicy:
    profile: str = "moderate"
    max_risk_per_trade_pct: float = 0.005
    max_daily_loss_pct: float = 0.015
    max_weekly_loss_pct: float = 0.03
    max_stops_per_day: int = 2


@dataclass(frozen=True)
class ReadinessVerdict:
    status: str
    reasons: list[str]
    metrics: dict[str, float]
    gate_config: dict[str, Any]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def summarize_trade_frame(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "target_exit_rate": 0.0,
            "stop_exit_rate": 0.0,
            "close_exit_rate": 0.0,
        }
    pnl = trades["net_pnl"].astype(float)
    returns = trades["return_pct"].astype(float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
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


def confidence_bucket_diagnostics(trades: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, float | str]] = []
    if trades.empty or "confidence" not in trades.columns:
        return {"buckets": rows, "has_inversion": False}

    frame = trades.copy()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["net_pnl"] = pd.to_numeric(frame.get("net_pnl", 0.0), errors="coerce").fillna(0.0)
    frame["return_pct"] = pd.to_numeric(frame.get("return_pct", 0.0), errors="coerce").fillna(0.0)
    previous_win_rate: float | None = None
    has_inversion = False

    for low, high in CONFIDENCE_BUCKETS:
        label = f"{low:.2f}-{min(high, 1.0):.2f}"
        bucket = frame[(frame["confidence"] >= low) & (frame["confidence"] < high)]
        if bucket.empty:
            rows.append({
                "bucket": label,
                "trades": 0.0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_return_pct": 0.0,
                "stop_rate": 0.0,
                "target_rate": 0.0,
                "avg_confidence": 0.0,
                "calibration_error": 0.0,
            })
            continue
        win_rate = float((bucket["net_pnl"] > 0).mean())
        avg_confidence = float(bucket["confidence"].mean())
        if previous_win_rate is not None and win_rate + 0.05 < previous_win_rate:
            has_inversion = True
        previous_win_rate = win_rate
        rows.append({
            "bucket": label,
            "trades": float(len(bucket)),
            "win_rate": win_rate,
            "net_pnl": float(bucket["net_pnl"].sum()),
            "avg_return_pct": float(bucket["return_pct"].mean()),
            "stop_rate": float((bucket.get("exit_reason") == "stop").mean()) if "exit_reason" in bucket else 0.0,
            "target_rate": float((bucket.get("exit_reason") == "target").mean()) if "exit_reason" in bucket else 0.0,
            "avg_confidence": avg_confidence,
            "calibration_error": abs(avg_confidence - win_rate),
        })

    return {"buckets": rows, "has_inversion": has_inversion}


def evaluate_promotion_gates(
    blind_summary: dict[str, float],
    confidence_diagnostics: dict[str, Any],
    config: PromotionGateConfig,
) -> ReadinessVerdict:
    reasons: list[str] = []
    trades = float(blind_summary.get("trades", 0.0))
    net_pnl = float(blind_summary.get("net_pnl", 0.0))
    sharpe = float(blind_summary.get("sharpe", 0.0))
    stop_rate = float(blind_summary.get("stop_exit_rate", 0.0))

    if trades < config.min_blind_trades:
        reasons.append(f"Blind trade count {trades:.0f} is below required {config.min_blind_trades}.")
    if net_pnl <= config.min_blind_pnl:
        reasons.append(f"Blind net P&L {net_pnl:.2f} is not above {config.min_blind_pnl:.2f}.")
    if sharpe <= config.min_blind_sharpe:
        reasons.append(f"Blind Sharpe {sharpe:.2f} is not above {config.min_blind_sharpe:.2f}.")
    if stop_rate > config.max_stop_rate:
        reasons.append(f"Blind stop rate {stop_rate:.2%} is above allowed {config.max_stop_rate:.2%}.")
    if confidence_diagnostics.get("has_inversion") and not config.allow_confidence_inversion:
        reasons.append("Confidence buckets are inverted; higher confidence underperforms lower confidence.")

    status = "BLOCKED" if reasons else "PAPER_ONLY"
    return ReadinessVerdict(
        status=status,
        reasons=reasons,
        metrics={
            "blind_trades": trades,
            "blind_net_pnl": net_pnl,
            "blind_sharpe": sharpe,
            "blind_stop_rate": stop_rate,
        },
        gate_config=asdict(config),
    )
