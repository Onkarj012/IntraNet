"""Order-ticket reconciliation helpers for the futures runtime."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReconciliationReport:
    ledger_rows: int
    order_records: int
    missing_tickets: list[str]
    duplicate_tickets: list[str]
    unknown_tickets: list[str]
    variant_mismatches: list[str]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_tickets or self.duplicate_tickets or
            self.unknown_tickets or self.variant_mismatches
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ok"] = self.ok
        return out


def ticket_id_for_row(row: pd.Series) -> str:
    variant = "C" if str(row["source"]) == "paper_c" else "A"
    return f"{variant}-{row['paper_trade_id']}"


def load_order_records(order_ledger: Path) -> list[dict]:
    if not order_ledger.exists():
        return []
    records = []
    with order_ledger.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def reconcile_order_tickets(
    paper_ledger: Path,
    order_ledger: Path,
    *,
    trade_date: str | None = None,
    sources: set[str] | None = None,
) -> ReconciliationReport:
    if sources is None:
        sources = {"paper", "paper_c"}
    df = pd.read_csv(paper_ledger) if paper_ledger.exists() else pd.DataFrame()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df[df["source"].isin(sources)].copy()
        if trade_date is not None:
            df = df[df["trade_date"] == trade_date].copy()

    expected = {ticket_id_for_row(row): str(row["source"]) for _, row in df.iterrows()}
    records = load_order_records(order_ledger)
    seen: dict[str, list[dict]] = {}
    for record in records:
        ticket = record.get("ticket", {})
        ticket_id = ticket.get("ticket_id")
        if ticket_id:
            seen.setdefault(str(ticket_id), []).append(record)

    missing = sorted([tid for tid in expected if tid not in seen])
    duplicate = sorted([tid for tid, recs in seen.items() if len(recs) > 1 and tid in expected])
    unknown = sorted([tid for tid in seen if tid not in expected])
    mismatches = []
    for tid, src in expected.items():
        recs = seen.get(tid, [])
        if not recs:
            continue
        ticket_variant = recs[0].get("ticket", {}).get("variant")
        expected_variant = "C" if src == "paper_c" else "A"
        if ticket_variant != expected_variant:
            mismatches.append(tid)

    return ReconciliationReport(
        ledger_rows=int(len(df)),
        order_records=sum(len(v) for v in seen.values()),
        missing_tickets=missing,
        duplicate_tickets=duplicate,
        unknown_tickets=unknown,
        variant_mismatches=sorted(mismatches),
    )
