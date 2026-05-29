from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from optinet.v5_runtime.data_quality import check_minute_session_quality
from optinet.v5_runtime.orders import DryRunOrderClient, OrderTicket, UpstoxOrderClient
from optinet.v5_runtime.reconcile import reconcile_order_tickets
from optinet.v5_runtime.risk import BrokerState, RiskLimits, evaluate_ticket_risk, order_count_for_date


def _ticket(ticket_id: str = "A-abc", variant: str = "A") -> OrderTicket:
    return OrderTicket(
        ticket_id=ticket_id,
        paper_trade_id="abc",
        timestamp="2026-05-15T10:00:00",
        symbol="NIFTY",
        expiry=None,
        side="BUY",
        qty_lots=1,
        order_type="MARKET",
        limit_price=None,
        target_price=101.0,
        stop_price=99.0,
        horizon_minutes=60,
        size_mult=1.0,
        variant=variant,
        intended_for_live=False,
    )


def test_session_quality_rejects_incomplete_day(tmp_path: Path):
    p = tmp_path / "minute.csv"
    rows = [
        {"date": "2026-05-15 09:15:00", "close": 100.0},
        {"date": "2026-05-15 09:16:00", "close": 100.5},
        {"date": "2026-05-15 13:20:00", "close": 101.0},
    ]
    pd.DataFrame(rows).to_csv(p, index=False)

    report = check_minute_session_quality(p, pd.Timestamp("2026-05-15"))

    assert not report.ok
    assert report.missing_close
    assert any("before 15:30:00" in r for r in report.reasons)


def test_dry_run_order_client_dedupes_ticket_id(tmp_path: Path):
    sink = tmp_path / "orders.jsonl"
    client = DryRunOrderClient(sink)

    first = client.place_order(_ticket())
    second = client.place_order(_ticket())

    assert first["status"] == "DRYRUN_LOGGED"
    assert second["status"] == "DRYRUN_DUPLICATE_SKIPPED"
    assert len(sink.read_text().splitlines()) == 1


def test_risk_governor_blocks_variant_c_for_live(tmp_path: Path):
    decision = evaluate_ticket_risk(
        _ticket("C-abc", "C"),
        trade_date="2026-05-15",
        order_ledger=tmp_path / "orders.jsonl",
        limits=RiskLimits(),
        broker_state=BrokerState(),
    )

    assert not decision.ok
    assert any("variant C" in reason for reason in decision.reasons)


def test_reconciliation_detects_missing_and_duplicates(tmp_path: Path):
    ledger = tmp_path / "paper.csv"
    orders = tmp_path / "orders.jsonl"
    pd.DataFrame([
        {
            "paper_trade_id": "abc",
            "trade_date": "2026-05-15",
            "source": "paper",
        }
    ]).to_csv(ledger, index=False)
    record = {
        "simulated_order_id": "DRY-1",
        "ticket": {"ticket_id": "A-abc", "variant": "A"},
    }
    orders.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")

    report = reconcile_order_tickets(ledger, orders, trade_date="2026-05-15")

    assert not report.ok
    assert report.duplicate_tickets == ["A-abc"]
    assert report.missing_tickets == []


def test_risk_daily_order_count_uses_ticket_trade_date(tmp_path: Path):
    orders = tmp_path / "orders.jsonl"
    client = DryRunOrderClient(orders)
    ticket = _ticket()
    ticket.trade_date = "2026-05-15"
    client.place_order(ticket)

    assert order_count_for_date(orders, "2026-05-15") == 1
    assert order_count_for_date(orders, "2026-05-16") == 0


def test_upstox_order_client_remains_stubbed():
    try:
        UpstoxOrderClient().place_order(_ticket())
    except NotImplementedError:
        pass
    else:
        raise AssertionError("UpstoxOrderClient must remain stubbed")
