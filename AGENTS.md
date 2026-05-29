# IntradayNet — Agent Context

## What This Project Does

IntradayNet is a machine learning system for intraday stock prediction on Indian equities (NSE Nifty 500). It generates daily LONG/SHORT trade recommendations with entry, target, and stop-loss levels — designed to be run before market open.

## Architecture

Three pipelines:

| Pipeline | Status | Purpose |
|----------|--------|---------|
| **V8 Redesign** | 🔨 Building | Ensemble of 5 specialist models + curve embeddings + barrier targets + diversified portfolio construction |
| **Open-safe premarket (V7)** | ✅ Active | Daily LONG/SHORT picks from ~85 daily features + 4 LightGBM models |
| **Live LightGBM backend** | ⏸️ Deferred | Per-minute intraday scalping with ~669 flattened features (not needed for daily picks) |

### V8 Redesign (New — May 2026)

Complete architectural redesign based on 5 principles:
1. **Learn from raw curves** — Masked autoencoder (Transformer) on 375-min OHLC curves produces 128-dim embeddings
2. **Predict path, not point** — Barrier targets: "hit +X% before -Y%?" instead of point-to-point return
3. **Ensemble of specialists** — 5 LightGBM models (Momentum, Reversal, Breakout, Sentiment, Macro) combined via regime-weighted meta-ensemble
4. **Calibration > Accuracy** — Isotonic calibration on every model
5. **Portfolio diversification** — Greedy sector/correlation-diversified selection (not just top-K)

**Data Flow:**
Minute bars → Barrier targets + Curve embeddings → Daily features (3 streams) → 5 specialist LBGM models → Isotonic calibration → Regime-weighted ensemble → Diversified portfolio selection → Trade recommendations

**Key V8 Files:**
- `src/intradaynet/v8/__init__.py` — Package, all exports
- `src/intradaynet/v8/config.py` — Single-source config (Target, Embedding, Signal, MetaEnsemble, Portfolio, Backtest, Sentiment, DailyFeatures)
- `src/intradaynet/v8/barriers.py` — Path-dependent barrier target computation (LONG/SHORT/NEUTRAL + multi-horizon support)
- `src/intradaynet/v8/curve_embedding.py` — Masked autoencoder (CurveMaskedEncoder), dataset, trainer, embedding generation
- `src/intradaynet/v8/data_pipeline.py` — Normalized minute data loading, session extraction, training data assembly
- `src/intradaynet/v8/universe_tiers.py` — Tier classification (T1: 2015-2026, T2: 2-3yr, T3: <2yr)
- `src/intradaynet/v8/per_stock_sentiment.py` — Per-stock sentiment features via yfinance news (8 features per stock per day)
- `src/intradaynet/v8/signal_models.py` — 5 specialist LightGBM classifiers + MetaEnsemble with regime weights
- `src/intradaynet/v8/regime_detector.py` — K-means regime clustering (5 regimes) + signal weights per regime
- `src/intradaynet/v8/portfolio.py` — Greedy diversified portfolio construction + position sizing
- `src/intradaynet/v8/walk_forward.py` — Walk-forward backtest scaffold, TradeRecord, BacktestMetrics, baseline comparison

The premarket pipeline:
1. Load minute-bar data from `data/nifty500/*_minute.csv`
2. Build daily features: price action (21), volume (4), Fibonacci (14), market macro (16), India-specific (12), sentiment (50), meta (1) → 91 features after dedup
3. 4 LightGBM models: LONG classifier, SHORT classifier, UP magnitude regressor, DOWN magnitude regressor
4. Confidence = `0.75 * primary_prob + 0.25 * (primary_prob - secondary_prob)`
5. Rank by score = `confidence * executable_edge`
6. Filter by risk profile (balanced: 5 picks, 0.65 min confidence, 1.5% target, 1% stop)
7. Post-open mode adjusts premarket predictions with live market data (gap, VWAP, volume, breadth)

## Key Files

### Source (`src/intradaynet/`)
- `v7.py` — Target labeling, confidence, scoring, readiness evaluation
- `v7_modes.py` — Post-open gap-aware adjustment, regime classification
- `open_safe_daily_features.py` — Daily feature builder from minute data
- `features/per_bar_features.py` — 25 bar-level technical features (RSI, BB, ATR, etc.)
- `features/session_features.py` — 41 session-level context features
- `features/sentiment_features.py` — 50 sentiment/market features from news data
- `features/market_features.py` — Macro data downloader (VIX, crude, gold, indices)
- `feature_contract.py` — Single source of truth for feature names (both pipelines)
- `config.py` — YAML config loader (dataclasses)
- `model_bundle.py` — LightGBM model manifest+bundle management
- `calibrator.py` — Platt/isotonic probability calibration
- `recommendation.py` — Profile-based recommendation engine
- `universe.py` — Stock universe definitions (Nifty 50/100/200/500)
- `costs.py` — NSE transaction cost calculator
- `live_news.py` — yfinance news ingestion + sentiment inference
- `regime.py` — Market regime detection (VIX, gap, trend)
- `robustness.py` — Backtest promotion gates, confidence diagnostics
- `equity_paper.py` — Paper trading ledger creation/reconciliation
- `run_logging.py` — Tee-duplicated stdout/stderr to log files

### Scripts (`scripts/`)
- `train_intraday_model.py` — Train 4 LightGBM models, saves `.pkl` + `.training_config.json`
- `recommend_intraday.py` — Full morning picks pipeline (premarket + post-open modes)
- `post_open_picks.py` — Fast 2-phase: premarket cache → post-open adjustment (<30s)
- `backtest_intraday_2025.py` — Risk-based backtest with stop/target/trailing
- `sync_data.py` — Download minute bars via yfinance to `data/nifty500/`
- `evaluate_equity.py` — Promotion gate evaluation from backtest summary
- `feature_analysis.py` — Per-feature quality diagnostics
- `daily_health_check.py` — Pre-flight checks (data freshness, model integrity)
- `readiness_report.py` — V7 deployability report
- `trading_system_status.py` — Combined equity+options readiness snapshot
- `verify_p0.py` — P0 correctness tests (targets, contracts, calibration)

### Config (`configs/`)
- `intraday_config.yaml` — Horizons, model architecture, data paths, training params, backtest costs

### Data (`data/`)
- `nifty500/*_minute.csv` — 531 stocks, 1-minute OHLCV bars, ~20 GB (gitignored)
- `sentiment/` — 8 news sentiment CSVs (gitignored)
- `prices/` — 42 daily EOD reference CSVs (tracked)
- `us/` — 25 US stock EOD CSVs
- `yfinance_daily/` — 83 daily CSVs from yfinance (misleading `_minute` suffix)

## Usage

### V8 Training & Backtest
```bash
# Full training with curve embeddings on MPS (Apple Silicon)
python scripts/train_v8.py --universe nifty100 --device mps

# Train without embeddings (faster, engineered features only)
python scripts/train_v8.py --universe nifty500 --no-embeddings

# Backtest with V7 comparison
python scripts/backtest_v8.py --universe nifty100 --model-dir models/v8 --compare-v7 <v7_results>

# Quick synthetic test
python scripts/backtest_v8.py --universe nifty50 --quick
```

### V7 Training (current active pipeline)
```bash
# Train on Nifty 100
python scripts/train_intraday_model.py --universe nifty100 --target-pct 0.015
# → models/intraday_model_nifty100.pkl + .training_config.json

# Train on Nifty 500  
python scripts/train_intraday_model.py --universe nifty500 --target-pct 0.015
# → models/intraday_model_nifty500.pkl + .training_config.json
```

### Generate Recommendations
```bash
# Premarket (full pipeline, 2-5 min)
python scripts/recommend_intraday.py --model models/intraday_model_nifty500.pkl --universe nifty500

# Fast post-open (cache-based, <30s)
python scripts/post_open_picks.py --universe nifty500
```

### Backtest
```bash
python scripts/backtest_intraday_2025.py --model models/intraday_model_nifty500.pkl --universe nifty100
```

## Path Conventions
- Minute data: `data/nifty500/{SYMBOL}_minute.csv`
- Sentiment: `data/sentiment/combined_sentiment_2015_2025.csv`
- Universe metadata: `data/sentiment/ind_nifty500list.csv`
- Market cache: `market_data_cache/*.csv`
- Models: `models/intraday_model_{UNIVERSE}.pkl`
- Premarket cache: `cache/premarket_cache_{UNIVERSE}_{DATE}.json`
- Outputs: `recommendations/`, `backtest_results/`, `logs/`

## Current State (May 2026)
- ✅ Data: 531 Nifty 500 stocks with minute data through Apr 8, 2026
- ✅ Sentiment: Combined 2015-2025 + Mar-Apr 2026 data
- ✅ Market cache: 25 macro/sector CSV files
- ✅ Models trained: Nifty 100 (AUC 0.610) + Nifty 500 (AUC 0.625)
- ✅ Post-open fast pipeline working
- ✅ V8 Redesign: Core library implemented (barriers, embeddings, signal models, portfolio, backtest framework)
- 🔨 V8 Training pending on real data
- ⚠️ Backtest: Very few trades (6 in 3 months), model is conservative
- ⚠️ AUC ~0.62 — modest predictive power, needs threshold tuning
- ⚠️ 57 of 531 stocks skipped in training (insufficient data)
- ❌ Live LightGBM backend not trained (deferred — not needed)

## Next Steps
1. Run `scripts/train_v8.py --universe nifty100 --device mps` to train curve embeddings + signal models
2. Run `scripts/backtest_v8.py --universe nifty100` to validate V8 vs V7 performance
3. Tune barrier target thresholds based on backtest results
4. Implement real market data integration for regime detector (replace synthetic data)
5. Build daily premarket pipeline script (`scripts/v8_pipeline.py`) for production picks

## OptiNet v2.1 — Sentiment + Regime + Calibration (May 2026, PROMOTED)

Separate system in the same repo. Predicts next-day NIFTY/BANKNIFTY direction and translates signals into option contracts.

### Status (v2.1)
- ✅ Data lake: 3.25M rows, 1,082 trading days (2022-01-03 → 2026-05-25), NIFTY + BANKNIFTY
- ✅ Parquet lake: `data/parquet/symbol={NIFTY,BANKNIFTY}/year=YYYY/options_YYYYMMDD.parquet`
- ✅ Spot OHLC: `data/indices/{nifty,banknifty}_daily.csv`
- ✅ GDELT 2.0 events: 1,589 days × 5 themes in `data/sentiment/gdelt_india_2022_2026.csv`
- ✅ Index sentiment cache: yfinance + RSS daily aggregates (accumulates over time)
- ✅ Regime detector: VIX + gap + ATR + trend, 23.8% of days flagged as block
- ✅ 4-LGBM stack with `class_weight=balanced` + `CalibratedClassifierCV(sigmoid, TimeSeriesSplit)`
- ✅ Pipeline: `scripts/optinet_pipeline.py` (train / recommend / daily-update)
- ✅ A/B harness: `scripts/optinet_ab_eval.py` with 6 configs

### A/B results (2026 YTD blind, balanced profile, min_conf=0.20)
| Config | Trades | Win % | Stop % | Sharpe | Net PnL |
|---|---|---|---|---|---|
| baseline (v2.1) | 22 | 23% | 73% | −6.8 | −23,588 |
| +regime_filter | 6 | 83% | 17% | +6.6 | +7,882 |
| **full (canonical)** | **4** | **100%** | **0%** | **+22.7** | **+10,750** |

The regime hard filter is the dominant lift. GDELT contributes +0.06 AUC. Sentiment
will accumulate signal over time as the daily cache fills.

### Key Files (v2.1)
- `src/optinet/sentiment.py` — yfinance + RSS daily index sentiment (NEW)
- `src/optinet/gdelt.py` — GDELT 2.0 event-volume features (NEW)
- `src/optinet/regime.py` — VIX/gap/ATR regime classifier + hard filter (NEW)
- `src/optinet/parquet_loader.py` — bridge parquet lake to OptiNet data loaders
- `src/optinet/models.py` — 4-LGBM stack + CalibratedClassifierCV + executable_edge
- `src/optinet/features.py` — index TA + F&O microstructure + sentiment + GDELT + regime (~80 features)
- `src/optinet/translator.py` — Black-Scholes delta-band contract picker
- `src/optinet/backtester.py` — daily-resolution backtest with regime-filter toggle
- `src/optinet/evaluation.py` — walk-forward + blind + per-fold calibration + readiness gates
- `src/optinet/data_lake.py` — bhavcopy download/parse/validate pipeline
- `scripts/optinet_pipeline.py` — train / recommend / daily-update CLI
- `scripts/optinet_ab_eval.py` — A/B harness across configs (NEW)
- `scripts/optinet_data_lake.py` — bhavcopy download / parse / validate CLI

### Next Steps for OptiNet
1. Promote v2.1 to paper trading (canonical model: `models/optinet/optinet_balanced.pkl`)
2. Set up daily cron: `update_sentiment_cache()` + `update_gdelt_cache()` + recommend
3. Monitor: stop rate must stay < 30%, win rate > 50% over rolling 30-day window
4. If month-over-month performance degrades → escalate to scope (b): multi-class model, cost-aware backtest, trailing stops
