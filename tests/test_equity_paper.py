from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity.equity_paper import create_equity_paper_ledger, reconcile_equity_paper_ledger


def _recommendations_payload(status: str = "PAPER_ONLY") -> dict:
    return {
        "picks_for_date": "2026-04-30",
        "mode": "premarket",
        "risk_profile": "balanced",
        "readiness": {"status": status, "reasons": ["not ready"] if status == "NOT_READY" else []},
        "long_picks": [
            {
                "symbol": "ABC",
                "direction": "LONG",
                "entry_basis": "Previous Close",
                "entry_price": 100.0,
                "target_price": 102.0,
                "stop_loss_price": 99.0,
                "confidence": 0.62,
                "score": 0.01,
            }
        ],
        "short_picks": [],
    }


def test_equity_paper_ledger_blocks_not_ready_recommendations(tmp_path):
    recommendations = tmp_path / "recommendations.json"
    recommendations.write_text(json.dumps(_recommendations_payload("NOT_READY")), encoding="utf-8")
    output = tmp_path / "ledger.csv"

    frame = create_equity_paper_ledger(recommendations_path=recommendations, output_path=output)

    assert output.exists()
    assert frame.iloc[0]["status"] == "blocked"
    assert "not ready" in frame.iloc[0]["reason"]


def test_equity_paper_ledger_sizes_open_pick_and_reconciles_target(tmp_path):
    recommendations = tmp_path / "recommendations.json"
    recommendations.write_text(json.dumps(_recommendations_payload()), encoding="utf-8")
    ledger = tmp_path / "ledger.csv"
    frame = create_equity_paper_ledger(recommendations_path=recommendations, output_path=ledger, capital=100_000)

    assert frame.iloc[0]["status"] == "open"
    assert frame.iloc[0]["quantity"] > 0

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "date": ["2026-04-30 09:15:00", "2026-04-30 09:16:00"],
            "open": [100.0, 100.5],
            "high": [101.0, 102.2],
            "low": [99.5, 100.0],
            "close": [100.5, 102.0],
            "volume": [1000, 1000],
        }
    ).to_csv(data_dir / "ABC_minute.csv", index=False)
    output = tmp_path / "reconciled.csv"

    reconciled = reconcile_equity_paper_ledger(ledger_path=ledger, data_dir=data_dir, output_path=output)

    assert output.exists()
    assert reconciled.iloc[0]["status"] == "closed"
    assert reconciled.iloc[0]["exit_reason"] == "target"
    assert reconciled.iloc[0]["net_pnl"] > 0
