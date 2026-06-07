"""
Intraday model inference module.

Loads the trained V8 5-specialist ensemble, detects today's market regime,
scores each symbol with a calibrated intraday probability (p_up), and
attaches barrier-aligned trade levels + expected value.

Output per stock:
  symbol, direction, p_up, confidence, entry, target, stop, roi, rr,
  expected_value, regime, drivers (per-specialist breakdown), horizon

Usage:
    from equity.intraday_model import IntradayModel
    model = IntradayModel()                    # loads from models/v8_intraday/
    picks = model.predict(X, date)            # X from IntradayFeatureAssembler.build()
    recs  = model.top_picks(picks, n=10)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("equity.intraday_model")

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _ROOT / "models/v8_intraday"

# ── Best real-life config (from exit_strategy_sweep.py) ──────────────────────
# Strategy: pre-open → buy top-N by dir_p at open → protective stop → square off
# at close (intraday). Validated Jan-May 2026: +12.6%, Sharpe ~2.3, max DD ~-4%.
STOP_ATR_MULT  = 2.0    # protective stop = entry × (1 - 2.0×ATR)  (catastrophe guard)
SOFT_TARGET_ATR_MULT = 1.0  # optional take-profit reference (strategy holds to close)
COST_PCT       = 0.0018 # round-trip transaction cost
MIN_CONFIDENCE = 0.44   # dir_p threshold — only trade names the direction head favours
EXIT_RULE      = "EOD square-off (15:15) or stop"
RANK_BY        = "dir_p"  # rank by direction-head probability

# Legacy fixed barrier (kept for fallback compatibility)
TARGET_PCT = 0.020
STOP_PCT = 0.010


class IntradayModel:
    """
    Wraps the MetaEnsemble + RegimeDetector for daily inference.

    Parameters
    ----------
    model_dir : Path, optional
        Directory containing {specialist}_signal.pkl + regime_weights.npy.
        Defaults to models/v8_intraday/.
    target_pct : float
        Intraday target barrier (must match training).
    stop_pct : float
        Intraday stop barrier (must match training).
    min_confidence : float
        Minimum p_up for a pick to be recommended.
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        target_pct: float = TARGET_PCT,
        stop_pct: float = STOP_PCT,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.model_dir = Path(model_dir or _DEFAULT_MODEL_DIR)
        self.target_pct = target_pct
        self.stop_pct = stop_pct
        self.min_confidence = min_confidence

        self._ensemble = None
        self._regime_detector = None
        self._train_meta: dict = {}
        self._dir_ensemble = None
        self._mag_models: dict[str, object] = {}
        self._mag_feature_names: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> IntradayModel:
        """Load ensemble and regime detector from disk."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                "Run: python scripts/trading/train_intraday.py"
            )
        from equity.v8.signal_models import MetaEnsemble, SignalModel
        import numpy as _np
        from equity.v8.regime_detector import DEFAULT_REGIME_WEIGHTS

        # Load barrier head (primary)
        self._ensemble = MetaEnsemble.load(self.model_dir)

        # Load direction head (secondary blend)
        dir_models = {}
        for name in ["momentum","reversal","breakout","sentiment","macro"]:
            p = self.model_dir / f"{name}_dir.pkl"
            if p.exists():
                dir_models[name] = SignalModel.load(p)
        w = self.model_dir / "regime_weights_dir.npy"
        self._dir_ensemble = MetaEnsemble(
            models=dir_models,
            regime_weights=_np.load(w) if w.exists() else DEFAULT_REGIME_WEIGHTS
        ) if dir_models else None

        # Load magnitude regressors (one pkl per specialist, plain dict with 'model')
        import pickle as _pickle
        self._mag_models: dict[str, object] = {}
        self._mag_feature_names: dict[str, list[str]] = {}
        for name in ["momentum","reversal","breakout","sentiment","macro"]:
            p = self.model_dir / f"{name}_mag.pkl"
            if p.exists():
                with p.open("rb") as fh:
                    state = _pickle.load(fh)
                self._mag_models[name] = state["model"]
                self._mag_feature_names[name] = state.get("feature_names", [])

        meta_path = self.model_dir / "train_meta.json"
        if meta_path.exists():
            self._train_meta = json.loads(meta_path.read_text())
            logger.info("Model trained at %s, barrier AUC %.4f  dir AUC %.4f",
                        self._train_meta.get("trained_at","?"),
                        self._train_meta.get("barrier_auc", self._train_meta.get("val_auc", 0)),
                        self._train_meta.get("dir_auc", 0))

        try:
            from equity.v8.regime_detector import RegimeDetector
            self._regime_detector = RegimeDetector()
            self._regime_detector.load(self.model_dir)
        except Exception:
            self._regime_detector = None

        return self

    def predict(self, X: pd.DataFrame, date: pd.Timestamp | str) -> pd.DataFrame:
        """
        Score all symbols in X and return a predictions DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix from IntradayFeatureAssembler.build().
            Index = symbol.
        date : str | Timestamp
            Today's date (for display / metadata).

        Returns
        -------
        pd.DataFrame
            Columns: symbol, date, p_up, regime, regime_id,
                     momentum_prob, reversal_prob, breakout_prob,
                     sentiment_prob, macro_prob,
                     entry, target, stop, roi, rr, expected_value, confidence.
        """
        if self._ensemble is None:
            self.load()

        date = pd.Timestamp(date).normalize()

        # Detect regime from macro features
        regime_id, regime_label = self._detect_regime(X)

        # Score all symbols — barrier head + direction head
        X_clean = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        barrier_probs, model_probs = self._ensemble.predict(X_clean, regime_id=regime_id)
        if self._dir_ensemble is not None and len(self._dir_ensemble.models) > 0:
            dir_probs, _ = self._dir_ensemble.predict(X_clean, regime_id=regime_id)
        else:
            dir_probs = barrier_probs

        # Magnitude predictions: average across loaded specialist regressors
        if self._mag_models:
            mag_raw = np.zeros(len(X_clean))
            for name, m in self._mag_models.items():
                feats = self._mag_feature_names.get(name, [])
                avail = [f for f in feats if f in X_clean.columns]
                if avail:
                    mag_raw += m.predict(X_clean[avail])
            mag_preds = mag_raw / len(self._mag_models)
        else:
            mag_preds = np.zeros(len(X_clean))

        # Meta-labeler predictions
        _meta_model = getattr(self._ensemble, "meta_model", None)
        if _meta_model is not None:
            _meta_state = _meta_model if not isinstance(_meta_model, dict) else _meta_model
            if isinstance(_meta_model, dict):
                _meta_clf = _meta_model["model"]
                _meta_feats = _meta_model.get("feature_names", list(X_clean.columns))
            else:
                _meta_clf = _meta_model
                _meta_feats = list(X_clean.columns)
            _meta_avail = [f for f in _meta_feats if f in X_clean.columns]
            meta_probs = _meta_clf.predict_proba(X_clean[_meta_avail])[:, 1] if _meta_avail else np.full(len(X_clean), 0.5)
        else:
            meta_probs = np.full(len(X_clean), 0.5)

        def _sigmoid(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

        # Ranking signal = blended final score
        ensemble_probs = 0.5 * dir_probs + 0.3 * _sigmoid(mag_preds) + 0.2 * meta_probs

        rows = []
        for i, sym in enumerate(X.index):
            dir_p = float(np.clip(dir_probs[i], 0.0, 1.0))
            barrier_p = float(np.clip(barrier_probs[i], 0.0, 1.0))
            mag_pred = float(mag_preds[i])
            meta_p = float(np.clip(meta_probs[i], 0.0, 1.0))
            # final_score = 0.5*dir_p + 0.3*sigmoid(mag_pred) + 0.2*meta_p
            p_up = float(np.clip(ensemble_probs[i], 0.0, 1.0))

            price = float(X.at[sym, "__price"]) if "__price" in X.columns else np.nan

            row = {
                "symbol": sym,
                "date": date,
                "p_up": round(p_up, 4),
                "dir_p": round(dir_p, 4),
                "barrier_p": round(barrier_p, 4),
                "mag_pred": round(mag_pred, 4),
                "meta_p": round(meta_p, 4),
                "regime": regime_label,
                "regime_id": regime_id,
            }
            for spec in ["momentum", "reversal", "breakout", "sentiment", "macro"]:
                if spec in model_probs and model_probs[spec] is not None:
                    row[f"{spec}_prob"] = round(float(np.clip(model_probs[spec][i], 0, 1)), 4)
                else:
                    row[f"{spec}_prob"] = np.nan

            # ── Hold-to-close strategy with protective stop ──
            atr_pct_val = float(X.at[sym, "avg_true_range_14d"]) if "avg_true_range_14d" in X.columns else 0.0
            atr_pct_val = max(min(atr_pct_val, 0.10), 0.005) if atr_pct_val > 0 else 0.02
            row["atr_pct"] = round(atr_pct_val, 4)
            row["exit_rule"] = EXIT_RULE
            row["entry"] = round(price, 2) if not np.isnan(price) else np.nan
            if not np.isnan(price) and price > 0:
                stop = price * (1 - STOP_ATR_MULT * atr_pct_val)
                soft_target = price * (1 + SOFT_TARGET_ATR_MULT * atr_pct_val)
                row["stop"] = round(stop, 2)
                row["target"] = round(soft_target, 2)   # soft reference; strategy holds to close
                row["roi"] = round(SOFT_TARGET_ATR_MULT * atr_pct_val, 4)
                row["rr"] = round(SOFT_TARGET_ATR_MULT / STOP_ATR_MULT, 2)
                # Expected value: edge from direction drift, risk capped by stop
                # Use historical mean drift conditional on dir_p as proxy
                expected_drift = (dir_p - 0.5) * 2 * atr_pct_val  # scaled by conviction
                ev = expected_drift - COST_PCT
                row["expected_value"] = round(ev, 4)
            else:
                row.update({"target": np.nan, "stop": np.nan, "roi": np.nan,
                            "rr": np.nan, "expected_value": np.nan})

            row["confidence"] = int(round(p_up * 100))
            row["horizon"] = "intraday (H375)"
            rows.append(row)

        return pd.DataFrame(rows).set_index("symbol")

    def top_picks(self, predictions: pd.DataFrame, n: int = 10, *,
                  sector_map: dict[str, str] | None = None) -> list[dict]:
        """
        Filter, rank, and diversify picks (best real-life config).

        Strategy: rank by dir_p (direction head), keep dir_p ≥ 0.44,
        equal-weight top-N, hold to close with 2.0×ATR protective stop.
        """
        df = predictions.copy()
        df = df[df["p_up"] >= self.min_confidence]   # dir_p ≥ 0.44
        if df.empty:
            return []

        df = df.sort_values("p_up", ascending=False)  # rank by conviction

        # Sector diversification: max 4 per sector (top-12 across ~13 sectors)
        if sector_map:
            sector_counts: dict[str, int] = {}
            selected = []
            for sym, row in df.iterrows():
                sec = sector_map.get(str(sym), "other")
                if sector_counts.get(sec, 0) >= 4:
                    continue
                selected.append(sym)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(selected) >= n:
                    break
        else:
            selected = list(df.index[:n])

        n_sel = len(selected)
        picks = []
        for sym in selected:
            row = df.loc[sym]
            picks.append({
                "symbol": str(sym),
                "direction": "LONG",
                "p_up": float(row["p_up"]),
                "dir_p": float(row.get("dir_p", row["p_up"])),
                "barrier_p": float(row.get("barrier_p", np.nan)),
                "confidence": int(row["confidence"]),
                "weight": round(1.0 / n_sel, 4),   # equal weight (sweep: beats conf-weight)
                "entry": float(row["entry"]) if not np.isnan(row["entry"]) else None,
                "target": float(row["target"]) if not np.isnan(row.get("target", np.nan)) else None,
                "stop": float(row["stop"]) if not np.isnan(row.get("stop", np.nan)) else None,
                "atr_pct": float(row.get("atr_pct", np.nan)),
                "roi": float(row["roi"]) if not np.isnan(row.get("roi", np.nan)) else None,
                "rr": float(row.get("rr", np.nan)) if not np.isnan(row.get("rr", np.nan)) else None,
                "expected_value": float(row["expected_value"]),
                "exit_rule": str(row.get("exit_rule", EXIT_RULE)),
                "regime": str(row["regime"]),
                "drivers": {
                    spec: float(row[f"{spec}_prob"])
                    for spec in ["momentum", "reversal", "breakout", "sentiment", "macro"]
                    if f"{spec}_prob" in row and not np.isnan(row[f"{spec}_prob"])
                },
                "horizon": "intraday (hold-to-close)",
                "horizon_days": 1,
            })
        return picks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_regime(self, X: pd.DataFrame) -> tuple[int, str]:
        """
        Detect market regime from macro features in X.

        Falls back to a simple rule-based classifier when the trained
        RegimeDetector is unavailable (e.g., first run before regime fitting).
        """
        from equity.v8.regime_detector import REGIME_LABELS

        if self._regime_detector is not None:
            try:
                regime_features = [
                    "vix_level", "vix_trend_5d", "nifty_adx",
                    "breadth_pct_above_20dma", "nifty_5d_return",
                ]
                avail = [f for f in regime_features if f in X.columns]
                if avail:
                    regime_row = X[avail].mean()  # market-level average
                    result = self._regime_detector.predict(regime_row.to_frame().T)
                    return int(result.iloc[0]["regime_id"]), str(result.iloc[0]["regime_label"])
            except Exception as e:
                logger.debug("Regime detector failed: %s", e)

        # Rule-based fallback
        vix = float(X.get("vix_level", pd.Series([0.15])).mean())
        ret5 = float(X.get("nifty_5d_return", pd.Series([0.0])).mean())
        if vix > 0.30:
            return 3, "high_vol_crisis"
        if ret5 > 0.02:
            return 0, "strong_trend_up"
        if ret5 < -0.02:
            return 1, "strong_trend_down"
        if vix < 0.12:
            return 4, "low_vol_compression"
        return 2, "choppy_reverting"


def score_universe(date: str | pd.Timestamp, symbols: Sequence[str], *,
                   model_dir: Path | str | None = None,
                   live_news: bool = False,
                   news_df: pd.DataFrame | None = None,
                   n_picks: int = 10,
                   post_open: bool = False) -> list[dict]:
    """
    Convenience function: build features + score + return top picks.

    Parameters
    ----------
    news_df : pd.DataFrame, optional
        Pre-fetched news articles (from google_news.fetch_overnight_news).
        If provided, used instead of fetching live yfinance news.
    post_open : bool
        If True, enrich the feature matrix with real 5-min ORB data via
        yfinance (call at 09:20 IST after the first bar closes).
    """
    from equity.intraday_features import IntradayFeatureAssembler
    from equity.universe import get_symbol_metadata

    date = pd.Timestamp(date).normalize()

    # Build features
    asm = IntradayFeatureAssembler()
    asm.refresh_macro()

    # Pass news_df to sentiment builder if provided
    if news_df is not None and not news_df.empty:
        from equity.features.sentiment_features import SentimentFeatureBuilder
        from equity.features.market_features import MarketFeatureBuilder
        _mb = MarketFeatureBuilder(str(_ROOT / "market_data_cache"))
        _mb._load_cached()
        asm._sentiment_builder = SentimentFeatureBuilder(
            csv_path=str(_ROOT / "data/sentiment/combined_sentiment_latest.csv"),
            market_builder=_mb,
            mode="premarket",
            market_open_time="09:15",
            universe_metadata_csv=str(_ROOT / "data/sentiment/ind_nifty500list.csv"),
            articles_df=news_df,
        )

    X = asm.build(date, symbols, live_news=live_news if news_df is None else False)
    if X.empty:
        logger.warning("Empty feature matrix for %s", date.date())
        return []

    # Enrich with real first-bar ORB features when running post-09:20
    if post_open:
        from equity.intraday_features import enrich_with_opening_bars
        logger.info("Post-open: fetching 5-min ORB bars for %d symbols", len(symbols))
        X = enrich_with_opening_bars(X, list(X.index), date)

    # Attach prev-close price for trade levels
    try:
        from equity.momentum import load_ohlcv
        ohlcv = load_ohlcv(_ROOT / "cache/v8/daily_panel_nifty500_adj.parquet")
        close = ohlcv["close"]
        prev_date = close.index[close.index < date]
        if len(prev_date):
            prices = close.loc[prev_date[-1]]
            X["__price"] = prices.reindex(X.index)
    except Exception as e:
        logger.warning("Could not attach prices: %s", e)

    # Build sector map for diversification
    sector_map = {sym: get_symbol_metadata(sym).get("industry", "other") for sym in X.index}

    # Score
    model = IntradayModel(model_dir=model_dir)
    predictions = model.predict(X, date)
    return model.top_picks(predictions, n=n_picks, sector_map=sector_map)
