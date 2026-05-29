from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LEDGER_COLUMNS = [
    "system",
    "status",
    "reason",
    "signal_date",
    "symbol",
    "direction",
    "mode",
    "risk_profile",
    "entry_basis",
    "entry_price",
    "target_price",
    "stop_loss_price",
    "confidence",
    "score",
    "quantity",
    "risk_amount",
    "position_value",
    "actual_entry",
    "exit_price",
    "exit_reason",
    "gross_pnl",
    "costs",
    "net_pnl",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_minute_data(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    frame.columns = frame.columns.str.lower()
    if "date" not in frame.columns:
        return None
    return frame.sort_values("date")


def create_equity_paper_ledger(
    *,
    recommendations_path: str | Path,
    output_path: str | Path,
    capital: float = 100_000.0,
    risk_per_trade_pct: float = 0.005,
    max_position_pct: float = 0.20,
    allowed_statuses: set[str] | None = None,
) -> pd.DataFrame:
    payload = _load_json(recommendations_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    allowed = allowed_statuses or {"PAPER_ONLY", "SMALL_LIVE", "READY"}

    readiness = payload.get("readiness", {})
    readiness_status = str(readiness.get("status", "NOT_READY"))
    if readiness_status not in allowed:
        reasons = "; ".join(readiness.get("reasons", [])) or f"Readiness is {readiness_status}."
        frame = pd.DataFrame([{
            "system": "intradaynet",
            "status": "blocked",
            "reason": reasons,
            "signal_date": payload.get("picks_for_date"),
        }], columns=LEDGER_COLUMNS)
        frame.to_csv(output, index=False)
        return frame

    picks = list(payload.get("long_picks", [])) + list(payload.get("short_picks", []))
    risk_amount = capital * risk_per_trade_pct
    max_position_value = capital * max_position_pct
    rows: list[dict[str, Any]] = []
    for pick in picks:
        entry = float(pick["entry_price"])
        stop = float(pick["stop_loss_price"])
        risk_per_share = abs(entry - stop)
        if entry <= 0 or risk_per_share <= 0:
            continue
        risk_quantity = int(risk_amount // risk_per_share)
        cap_quantity = int(max_position_value // entry)
        quantity = max(0, min(risk_quantity, cap_quantity))
        if quantity <= 0:
            rows.append({
                "system": "intradaynet",
                "status": "skipped",
                "reason": "Risk budget too small for one share.",
                "signal_date": payload.get("picks_for_date"),
                "symbol": pick.get("symbol"),
            })
            continue
        rows.append({
            "system": "intradaynet",
            "status": "open",
            "reason": "",
            "signal_date": payload.get("picks_for_date"),
            "symbol": pick["symbol"],
            "direction": pick["direction"],
            "mode": payload.get("mode", pick.get("mode")),
            "risk_profile": payload.get("risk_profile"),
            "entry_basis": pick.get("entry_basis"),
            "entry_price": entry,
            "target_price": float(pick["target_price"]),
            "stop_loss_price": stop,
            "confidence": float(pick.get("confidence", 0.0)),
            "score": float(pick.get("score", 0.0)),
            "quantity": quantity,
            "risk_amount": round(risk_per_share * quantity, 2),
            "position_value": round(entry * quantity, 2),
        })

    frame = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    frame.to_csv(output, index=False)
    return frame


def reconcile_equity_paper_ledger(
    *,
    ledger_path: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    brokerage_per_trade: float = 40.0,
) -> pd.DataFrame:
    ledger = pd.read_csv(ledger_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if ledger.empty:
        ledger.to_csv(output, index=False)
        return ledger

    ledger = ledger.astype({"status": "object", "reason": "object", "exit_reason": "object"})
    data_root = Path(data_dir)
    for idx, row in ledger.iterrows():
        if row.get("status") != "open":
            continue
        symbol = str(row["symbol"])
        day = pd.Timestamp(row["signal_date"]).normalize()
        minute_data = _read_minute_data(data_root / f"{symbol}_minute.csv")
        if minute_data is None:
            ledger.at[idx, "status"] = "unreconciled"
            ledger.at[idx, "reason"] = "Missing minute data file."
            continue
        session = minute_data[pd.to_datetime(minute_data["date"]).dt.normalize() == day]
        if session.empty:
            ledger.at[idx, "status"] = "unreconciled"
            ledger.at[idx, "reason"] = "Missing session bars for signal date."
            continue

        entry = float(row["entry_price"])
        target = float(row["target_price"])
        stop = float(row["stop_loss_price"])
        direction = str(row["direction"]).upper()
        exit_price = float(session["close"].iloc[-1])
        exit_reason = "close"

        for _, bar in session.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            if direction == "LONG":
                if low <= stop:
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if high >= target:
                    exit_price = target
                    exit_reason = "target"
                    break
            else:
                if high >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if low <= target:
                    exit_price = target
                    exit_reason = "target"
                    break

        quantity = int(float(row["quantity"]))
        gross = (exit_price - entry) * quantity if direction == "LONG" else (entry - exit_price) * quantity
        costs = brokerage_per_trade
        ledger.at[idx, "status"] = "closed"
        ledger.at[idx, "actual_entry"] = round(entry, 2)
        ledger.at[idx, "exit_price"] = round(exit_price, 2)
        ledger.at[idx, "exit_reason"] = exit_reason
        ledger.at[idx, "gross_pnl"] = round(gross, 2)
        ledger.at[idx, "costs"] = round(costs, 2)
        ledger.at[idx, "net_pnl"] = round(gross - costs, 2)

    ledger.to_csv(output, index=False)
    return ledger
