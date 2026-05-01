from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from optinet.data import align_spot_to_chain, load_index_bars, load_option_chain
from optinet.models import OptiNetModelBundle
from optinet.recommender import build_dataset, recommend_latest


LEDGER_COLUMNS = [
    "system",
    "status",
    "reason",
    "signal_date",
    "index",
    "direction",
    "profile",
    "structure",
    "strike",
    "expiry",
    "option_type",
    "premium_entry",
    "premium_target",
    "premium_stop",
    "confidence",
    "score",
    "actual_entry",
    "exit_price",
    "exit_reason",
    "net_pnl",
]


def _read_readiness(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    import json

    readiness_path = Path(path)
    if not readiness_path.exists():
        return None
    return json.loads(readiness_path.read_text(encoding="utf-8"))


def create_optinet_paper_ledger(
    *,
    model_path: str | Path,
    index_paths,
    option_paths,
    output_path: str | Path,
    profile: str = "balanced",
    top_k: int = 4,
    min_confidence: float = 0.55,
    readiness_path: str | Path | None = None,
) -> pd.DataFrame:
    readiness = _read_readiness(readiness_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if readiness and readiness.get("status") == "BLOCKED":
        reasons = "; ".join(readiness.get("reasons", []))
        frame = pd.DataFrame([{
            "system": "optinet",
            "status": "blocked",
            "reason": reasons or "Model readiness is BLOCKED.",
        }], columns=LEDGER_COLUMNS)
        frame.to_csv(output, index=False)
        return frame

    features, option_chain = build_dataset(index_paths, option_paths)
    bundle = OptiNetModelBundle.load(model_path)
    payload = recommend_latest(
        bundle,
        features,
        option_chain,
        profile=profile,
        top_k=top_k,
        min_confidence=min_confidence,
    )
    signal_date = payload.get("as_of")
    rows = []
    for pick in payload.get("picks", []):
        rows.append({
            "system": "optinet",
            "status": "open",
            "reason": "",
            "signal_date": signal_date,
            **pick,
        })
    frame = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    frame.to_csv(output, index=False)
    return frame


def reconcile_optinet_paper_ledger(
    *,
    ledger_path: str | Path,
    index_paths,
    option_paths,
    output_path: str | Path,
) -> pd.DataFrame:
    ledger = pd.read_csv(ledger_path)
    if ledger.empty or "signal_date" not in ledger.columns:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(output_path, index=False)
        return ledger

    index_bars = load_index_bars(index_paths)
    option_chain = align_spot_to_chain(load_option_chain(option_paths), index_bars)
    option_chain["date"] = pd.to_datetime(option_chain["date"], format="mixed", errors="coerce")
    ledger = ledger.astype({"reason": "object", "status": "object", "exit_reason": "object"})
    ledger = ledger.copy()

    for idx, row in ledger.iterrows():
        if row.get("status") != "open":
            continue
        signal_date = pd.Timestamp(row["signal_date"])
        future_dates = option_chain[option_chain["date"] > signal_date]["date"].drop_duplicates().sort_values()
        if future_dates.empty:
            ledger.at[idx, "status"] = "unreconciled"
            ledger.at[idx, "reason"] = "No next option-chain date available."
            continue
        entry_date = pd.Timestamp(future_dates.iloc[0])
        match = option_chain[
            (option_chain["date"] == entry_date)
            & (option_chain["index"] == row["index"])
            & (option_chain["expiry"] == pd.Timestamp(row["expiry"]))
            & (option_chain["strike"] == float(row["strike"]))
            & (option_chain["option_type"] == row["option_type"])
        ]
        if match.empty:
            ledger.at[idx, "status"] = "unreconciled"
            ledger.at[idx, "reason"] = "Selected contract missing on next trading day."
            continue
        bar = match.iloc[0]
        entry = float(bar["open"]) if float(bar.get("open", 0.0)) > 0 else float(row["premium_entry"])
        target = entry * (float(row["premium_target"]) / float(row["premium_entry"]))
        stop = entry * (float(row["premium_stop"]) / float(row["premium_entry"]))
        if float(bar["low"]) <= stop:
            exit_price = stop
            exit_reason = "stop"
        elif float(bar["high"]) >= target:
            exit_price = target
            exit_reason = "target"
        else:
            exit_price = float(bar["close"])
            exit_reason = "close"
        ledger.at[idx, "status"] = "closed"
        ledger.at[idx, "actual_entry"] = round(entry, 2)
        ledger.at[idx, "exit_price"] = round(exit_price, 2)
        ledger.at[idx, "exit_reason"] = exit_reason
        ledger.at[idx, "net_pnl"] = round(exit_price - entry, 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_path, index=False)
    return ledger
