"""
Tests for the live LightGBM backend contract and profile selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intradaynet.feature_contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    flatten_intraday_batch,
    flatten_intraday_window,
)
from intradaynet.features.per_bar_features import PER_BAR_FEATURE_NAMES
from intradaynet.features.sentiment_features import SENTIMENT_FEATURE_NAMES
from intradaynet.features.session_features import SESSION_FEATURE_NAMES
from intradaynet.model_bundle import (
    HorizonBundleMetadata,
    ModelBundleManifest,
    load_manifest,
    save_manifest,
    validate_feature_contract,
)
from intradaynet.recommendation import build_recommendation_payload


def test_shared_feature_contract_shape():
    window = np.zeros((120, len(PER_BAR_FEATURE_NAMES)), dtype=np.float32)
    session = np.zeros(len(SESSION_FEATURE_NAMES), dtype=np.float32)
    sentiment = np.zeros(len(SENTIMENT_FEATURE_NAMES), dtype=np.float32)

    flat = flatten_intraday_window(window, session, sentiment)
    assert flat.shape[0] == len(FEATURE_NAMES)
    assert flat.shape[0] == FEATURE_SCHEMA.feature_count


def test_batch_flatten_matches_single():
    windows = np.zeros((2, 120, len(PER_BAR_FEATURE_NAMES)), dtype=np.float32)
    sessions = np.zeros((2, len(SESSION_FEATURE_NAMES)), dtype=np.float32)
    sentiments = np.zeros((2, len(SENTIMENT_FEATURE_NAMES)), dtype=np.float32)

    batch = flatten_intraday_batch(windows, sessions, sentiments)
    assert batch.shape == (2, len(FEATURE_NAMES))
    assert np.allclose(batch[0], flatten_intraday_window(windows[0], sessions[0], sentiments[0]))


def test_manifest_validation_rejects_bad_schema():
    bad_names = list(FEATURE_NAMES)
    bad_names[0] = "broken_name"
    try:
        validate_feature_contract(bad_names)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected feature contract validation to fail")


def test_manifest_roundtrip(tmp_path: Path):
    manifest = ModelBundleManifest(
        bundle_name="test_bundle",
        horizons=["H15"],
        horizon_files={
            "H15": HorizonBundleMetadata(
                direction_model="dir_H15.lgb",
                gross_return_model="ret_H15.lgb",
                net_edge_model="edge_H15.lgb",
                calibrator="calibrator_H15.pkl",
            )
        },
    )
    path = save_manifest(tmp_path, manifest)
    assert path.exists()
    loaded = load_manifest(tmp_path)
    assert loaded.bundle_name == "test_bundle"
    assert loaded.feature_count == len(FEATURE_NAMES)


def test_profile_payload_caps_lists():
    candidates = []
    for idx in range(12):
        candidates.append(
            {
                "symbol": f"LONG{idx}",
                "side": "LONG",
                "horizon": "H60",
                "entry_reference": 100.0,
                "expected_gross_return": 0.02,
                "expected_net_edge": 0.004,
                "confidence": 0.8,
                "probability": 0.8,
                "prob_strength": 0.6,
                "liquidity_score": 0.9,
                "avg_daily_traded_value": 50_000_000.0,
                "median_minute_turnover": 500_000.0,
                "regime": "calm_bull",
                "regime_alignment": 1.0,
                "reward_cost_ratio": 5.0,
                "sector": "UNKNOWN",
                "target": 102.0,
                "stop_loss": 99.0,
                "driver_flags": [],
                "score": 5.0 - idx * 0.1,
            }
        )
        candidates.append(
            {
                "symbol": f"SHORT{idx}",
                "side": "SHORT",
                "horizon": "H60",
                "entry_reference": 100.0,
                "expected_gross_return": 0.02,
                "expected_net_edge": 0.004,
                "confidence": 0.8,
                "probability": 0.2,
                "prob_strength": 0.6,
                "liquidity_score": 0.9,
                "avg_daily_traded_value": 50_000_000.0,
                "median_minute_turnover": 500_000.0,
                "regime": "calm_bear",
                "regime_alignment": 1.0,
                "reward_cost_ratio": 5.0,
                "sector": "UNKNOWN",
                "target": 98.0,
                "stop_loss": 101.0,
                "driver_flags": [],
                "score": 5.0 - idx * 0.1,
            }
        )

    payload = build_recommendation_payload(
        trade_date="2025-01-01",
        market_regime="calm_bull",
        market_summary={"should_trade": True},
        candidates=candidates,
    )

    assert len(payload["profiles"]["conservative"]["long"]) == 3
    assert len(payload["profiles"]["conservative"]["short"]) == 3
    assert len(payload["profiles"]["balanced"]["long"]) == 5
    assert len(payload["profiles"]["balanced"]["short"]) == 5
    assert len(payload["profiles"]["aggressive"]["long"]) == 8
    assert len(payload["profiles"]["aggressive"]["short"]) == 8
