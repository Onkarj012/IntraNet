"""JSON run reports for daily futures paper-trading operations."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pandas as pd


@dataclass
class OpsStep:
    label: str
    command: list[str]
    return_code: int


@dataclass
class OpsRunReport:
    run_timestamp: str
    steps: list[OpsStep] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(step.return_code == 0 for step in self.steps)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ok"] = self.ok
        return out

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


def summarize_paper_ledger(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path)
    if df.empty:
        return {"exists": True, "rows": 0}
    summary = {
        "exists": True,
        "rows": int(len(df)),
        "sources": df["source"].value_counts().to_dict() if "source" in df else {},
    }
    if "trade_date" in df:
        dates = pd.to_datetime(df["trade_date"], errors="coerce")
        summary["date_min"] = str(dates.min().date()) if dates.notna().any() else None
        summary["date_max"] = str(dates.max().date()) if dates.notna().any() else None
    return summary


def next_report_path(log_dir: Path, prefix: str = "paper_trade_ops") -> Path:
    ts = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{prefix}_{ts}.json"
