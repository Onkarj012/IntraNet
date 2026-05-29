from __future__ import annotations

from pathlib import Path
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equity.robustness import (
    PromotionGateConfig,
    confidence_bucket_diagnostics,
    evaluate_promotion_gates,
    write_json,
)
from index_options.data import align_spot_to_chain, load_index_bars, load_option_chain
from index_options.features import build_fo_features, build_index_features, build_training_frame
from index_options.labels import build_labels, merge_features_labels
from index_options.models import score_frame, train_model_stack
from index_options.paper import create_optinet_paper_ledger, reconcile_optinet_paper_ledger
from index_options.translator import translate_signal


def _index_bars(days: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=days)
    base = 24000 + np.arange(days) * 18 + np.sin(np.arange(days) / 3) * 80
    return pd.DataFrame(
        {
            "date": dates,
            "index": "NIFTY",
            "open": base,
            "high": base * 1.006,
            "low": base * 0.994,
            "close": base * (1 + np.sin(np.arange(days)) * 0.001),
            "volume": 1_000_000,
        }
    )


def _option_chain(index_bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, bar in index_bars.iterrows():
        date = pd.Timestamp(bar["date"])
        expiry = date + pd.Timedelta(days=(3 - date.dayofweek) % 5 + 1)
        spot = float(bar["close"])
        atm = round(spot / 50) * 50
        for strike in range(int(atm - 300), int(atm + 351), 50):
            for typ in ["CE", "PE"]:
                intrinsic = max(0.0, spot - strike) if typ == "CE" else max(0.0, strike - spot)
                distance = abs(strike - spot)
                premium = intrinsic + max(18.0, 140.0 - distance * 0.22)
                rows.append(
                    {
                        "date": date,
                        "index": "NIFTY",
                        "expiry": expiry,
                        "strike": strike,
                        "option_type": typ,
                        "open": premium * 0.98,
                        "high": premium * 1.20,
                        "low": premium * 0.82,
                        "close": premium,
                        "volume": max(1, 5000 - distance * 3),
                        "open_interest": max(1, 10000 - distance * 8),
                        "change_oi": 100 if strike >= atm else -50,
                    }
                )
    return pd.DataFrame(rows)


def test_feature_and_label_builders_create_training_frame():
    bars = _index_bars()
    chain = align_spot_to_chain(_option_chain(bars), bars)
    features = build_training_frame(bars, chain)
    labels = build_labels(bars)
    dataset = merge_features_labels(features, labels)

    assert {"pcr_oi", "iv_rank_20d", "return_5d", "balanced_long_label"}.issubset(dataset.columns)
    assert len(dataset) > 50
    assert dataset["balanced_long_label"].isin([0, 1]).all()


def test_loaders_accept_common_bhavcopy_names(tmp_path):
    bars = _index_bars(5).rename(columns={"date": "Date", "index": "SYMBOL", "open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"})
    index_csv = tmp_path / "index.csv"
    bars.to_csv(index_csv, index=False)
    options = _option_chain(_index_bars(5)).rename(
        columns={
            "date": "TIMESTAMP",
            "index": "SYMBOL",
            "expiry": "EXPIRY_DT",
            "strike": "STRIKE_PR",
            "option_type": "OPTION_TYP",
            "open_interest": "OPEN_INT",
            "change_oi": "CHG_IN_OI",
        }
    )
    options_csv = tmp_path / "options.csv"
    options.to_csv(options_csv, index=False)

    loaded_bars = load_index_bars(index_csv)
    loaded_options = load_option_chain(options_csv)

    assert loaded_bars["index"].unique().tolist() == ["NIFTY"]
    assert set(loaded_options["option_type"].unique()) == {"CE", "PE"}


def test_translator_selects_contract_and_premium_levels():
    bars = _index_bars()
    chain = align_spot_to_chain(_option_chain(bars), bars)
    features = build_training_frame(bars, chain)
    last = features.iloc[-1]
    signal = pd.Series(
        {
            "index": "NIFTY",
            "date": last["date"],
            "direction": "LONG",
            "confidence": 0.72,
            "score": 0.01,
        }
    )

    trade = translate_signal(signal, chain, last, profile="balanced")

    assert trade is not None
    assert trade.option_type == "CE"
    assert trade.target > trade.entry > trade.stop


def test_model_stack_scores_rows():
    bars = _index_bars(120)
    chain = align_spot_to_chain(_option_chain(bars), bars)
    features = build_training_frame(bars, chain)
    dataset = merge_features_labels(features, build_labels(bars))

    bundle = train_model_stack(dataset, profile="balanced", validation_fraction=0.25)
    scores = score_frame(bundle, dataset.tail(5))

    assert len(bundle.feature_columns) >= 35
    assert len(scores) == 5
    assert scores["confidence"].between(0, 1).all()


import pytest
@pytest.mark.skip(reason="optinet_data_builder removed")
def test_builder_parser_accepts_legacy_nse_fo_format():
    csv_text = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,OPEN_INT,CHG_IN_OI,TIMESTAMP
OPTIDX,NIFTY,25-JUL-2024,23350,CE,100,120,90,110,111,500,1000,50,05-JUL-2024
OPTIDX,BANKNIFTY,10-JUL-2024,42000,PE,200,250,180,230,231,700,2000,-20,05-JUL-2024
FUTIDX,NIFTY,25-JUL-2024,0,XX,0,0,0,0,0,0,0,0,05-JUL-2024
"""
    parsed = builder._parse_fo_bhavcopy(csv_text, date(2024, 7, 5))

    assert parsed is not None
    assert parsed.shape[0] == 2
    assert set(parsed["index_name"]) == {"NIFTY", "BANKNIFTY"}
    assert set(parsed["option_type"]) == {"CE", "PE"}


@pytest.mark.skip(reason="optinet_data_builder removed")
def test_builder_parser_accepts_udiff_fo_format():
    csv_text = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2025-01-02,2025-01-02,FO,NSE,IDO,1,,NIFTY,,2025-01-30,2025-01-30,23350,PE,NIFTY25JAN23350PE,140,150,120,147.3,147.3,130,23500,181.4,1200,20,250,0,0,F1,25,,,,,
2025-01-02,2025-01-02,FO,NSE,IDO,2,,BANKNIFTY,,2025-02-27,2025-02-27,55300,CE,BANKNIFTY25FEB55300CE,500,550,450,520,520,480,52500,510,2200,-10,320,0,0,F1,15,,,,,
2025-01-02,2025-01-02,FO,NSE,STO,3,,RELIANCE,,2025-01-30,2025-01-30,1200,CE,REL25JAN1200CE,10,15,8,12,12,11,1210,12,100,0,50,0,0,F1,250,,,,,
"""
    parsed = builder._parse_fo_bhavcopy(csv_text, date(2025, 1, 2))

    assert parsed is not None
    assert parsed.shape[0] == 2
    assert set(parsed["index_name"]) == {"NIFTY", "BANKNIFTY"}
    assert set(parsed["option_type"]) == {"CE", "PE"}


def test_loaders_accept_builder_output_schema_with_mixed_dates(tmp_path):
    index_frame = pd.DataFrame(
        {
            "index_name": ["NIFTY", "BANKNIFTY"],
            "date": ["2025-01-02", "2025-01-02 00:00:00"],
            "open": [24000, 52000],
            "high": [24100, 52100],
            "low": [23900, 51900],
            "close": [24050, 52050],
        }
    )
    option_frame = pd.DataFrame(
        {
            "index_name": ["NIFTY", "BANKNIFTY"],
            "date": ["2025-01-02", "2025-01-02 00:00:00"],
            "expiry_date": ["2025-01-30", "2025-02-27 00:00:00"],
            "strike_price": [23350, 55300],
            "option_type": ["PE", "CE"],
            "open": [140.0, 500.0],
            "high": [150.0, 550.0],
            "low": [120.0, 450.0],
            "close": [147.3, 520.0],
            "settlement_price": [181.4, 510.0],
            "volume": [250, 320],
            "open_interest": [1200, 2200],
            "change_in_oi": [20, -10],
            "days_to_expiry": [28, 56],
        }
    )
    index_csv = tmp_path / "builder_index.csv"
    options_csv = tmp_path / "builder_options.csv"
    index_frame.to_csv(index_csv, index=False)
    option_frame.to_csv(options_csv, index=False)

    loaded_bars = load_index_bars(index_csv)
    loaded_options = load_option_chain(options_csv)

    assert set(loaded_bars["index"]) == {"NIFTY", "BANKNIFTY"}
    assert set(loaded_options["index"]) == {"NIFTY", "BANKNIFTY"}
    assert loaded_options["date"].notna().all()
    assert loaded_options["expiry"].notna().all()


def test_confidence_buckets_and_readiness_gate_block_bad_blind_model():
    trades = pd.DataFrame(
        {
            "confidence": [0.25, 0.60, 0.72],
            "net_pnl": [100.0, -50.0, -75.0],
            "return_pct": [0.1, -0.05, -0.07],
            "exit_reason": ["target", "stop", "stop"],
        }
    )
    diagnostics = confidence_bucket_diagnostics(trades)
    verdict = evaluate_promotion_gates(
        {
            "trades": 3.0,
            "net_pnl": -25.0,
            "sharpe": -1.0,
            "stop_exit_rate": 2 / 3,
        },
        diagnostics,
        PromotionGateConfig(min_blind_trades=50),
    )

    assert diagnostics["has_inversion"] is True
    assert verdict.status == "BLOCKED"
    assert len(verdict.reasons) >= 4


def test_paper_ledger_writes_blocked_row_when_readiness_blocks(tmp_path):
    readiness = tmp_path / "readiness.json"
    readiness.write_text('{"status":"BLOCKED","reasons":["blind failed"]}', encoding="utf-8")
    output = tmp_path / "ledger.csv"

    frame = create_optinet_paper_ledger(
        model_path=tmp_path / "missing.pkl",
        index_paths=[],
        option_paths=[],
        output_path=output,
        readiness_path=readiness,
    )

    assert output.exists()
    assert frame.iloc[0]["status"] == "blocked"
    assert "blind failed" in frame.iloc[0]["reason"]


def test_model_registry_json_serializes_common_metadata_types(tmp_path):
    registry = tmp_path / "model_registry.json"

    write_json(
        registry,
        {
            "model_path": tmp_path / "model.pkl",
            "feature_schema_version": "optinet_v1",
            "created_at": pd.Timestamp("2026-04-30"),
            "metrics": {"rows": np.float64(10.0)},
        },
    )

    payload = registry.read_text(encoding="utf-8")
    assert "optinet_v1" in payload
    assert "2026-04-30T00:00:00" in payload


def test_reconcile_paper_handles_missing_contract(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "system,status,reason,signal_date,index,direction,profile,structure,strike,expiry,option_type,premium_entry,premium_target,premium_stop,confidence,score,actual_entry,exit_price,exit_reason,net_pnl\n"
        "optinet,open,,2025-01-01,NIFTY,LONG,balanced,naked_buy,24000,2025-01-30,CE,100,120,80,0.7,0.01,,,,\n",
        encoding="utf-8",
    )
    index_csv = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "index": ["NIFTY", "NIFTY"],
            "open": [24000, 24050],
            "high": [24100, 24150],
            "low": [23900, 23950],
            "close": [24050, 24100],
        }
    ).to_csv(index_csv, index=False)
    options_csv = tmp_path / "options.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "index": ["NIFTY"],
            "expiry": ["2025-01-30"],
            "strike": [24500],
            "option_type": ["CE"],
            "open": [10],
            "high": [12],
            "low": [8],
            "close": [9],
        }
    ).to_csv(options_csv, index=False)
    output = tmp_path / "reconciled.csv"

    frame = reconcile_optinet_paper_ledger(
        ledger_path=ledger,
        index_paths=[index_csv],
        option_paths=[options_csv],
        output_path=output,
    )

    assert output.exists()
    assert frame.iloc[0]["status"] == "unreconciled"
    assert "missing" in frame.iloc[0]["reason"].lower()


def test_reconcile_paper_closes_target_stop_and_close(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "system,status,reason,signal_date,index,direction,profile,structure,strike,expiry,option_type,premium_entry,premium_target,premium_stop,confidence,score,actual_entry,exit_price,exit_reason,net_pnl\n"
        "optinet,open,,2025-01-01,NIFTY,LONG,balanced,naked_buy,24000,2025-01-30,CE,100,120,80,0.7,0.01,,,,\n"
        "optinet,open,,2025-01-01,NIFTY,LONG,balanced,naked_buy,24100,2025-01-30,CE,100,120,80,0.7,0.01,,,,\n"
        "optinet,open,,2025-01-01,NIFTY,LONG,balanced,naked_buy,24200,2025-01-30,CE,100,120,80,0.7,0.01,,,,\n",
        encoding="utf-8",
    )
    index_csv = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "index": ["NIFTY", "NIFTY"],
            "open": [24000, 24050],
            "high": [24100, 24150],
            "low": [23900, 23950],
            "close": [24050, 24100],
        }
    ).to_csv(index_csv, index=False)
    options_csv = tmp_path / "options.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-02"],
            "index": ["NIFTY", "NIFTY", "NIFTY"],
            "expiry": ["2025-01-30", "2025-01-30", "2025-01-30"],
            "strike": [24000, 24100, 24200],
            "option_type": ["CE", "CE", "CE"],
            "open": [100, 100, 100],
            "high": [125, 110, 110],
            "low": [95, 75, 90],
            "close": [118, 82, 105],
        }
    ).to_csv(options_csv, index=False)
    output = tmp_path / "reconciled.csv"

    frame = reconcile_optinet_paper_ledger(
        ledger_path=ledger,
        index_paths=[index_csv],
        option_paths=[options_csv],
        output_path=output,
    )

    assert frame["status"].tolist() == ["closed", "closed", "closed"]
    assert frame["exit_reason"].tolist() == ["target", "stop", "close"]
    assert frame["net_pnl"].tolist() == [20.0, -20.0, 5.0]
