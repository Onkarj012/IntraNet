# IntradayNet — Complete System Architecture

## Overview

IntradayNet is a machine learning system for intraday stock prediction on Indian equities (NSE Nifty 500). It generates daily LONG/SHORT trade recommendations with entry, target, and stop-loss levels, designed to run before market open. The system comprises three pipelines at different stages of maturity: the active V7 production pipeline, the in-progress V8 redesign, and a deferred live per-minute backend.

---

## 1. V7 Production Pipeline (Open-Safe Premarket)

The V7 pipeline is the active daily-recommendation system. It consumes minute-bar OHLCV data, builds ~91 daily features per stock, evaluates them through 4 LightGBM models, and produces ranked LONG/SHORT picks with calibrated confidence scores.

### 1.1 Data Input

**Minute-bar data:** 531 stocks × 1-minute OHLCV bars sourced from `data/nifty500/{SYMBOL}_minute.csv`. Coverage spans 2015 through April 2026, totalling approximately 20 GB. Each file contains timestamp-indexed rows with open, high, low, close, and volume columns.

**Sentiment data:** A unified CSV at `data/sentiment/combined_sentiment_2015_2025.csv`, aggregated from multiple historical news sources plus live yfinance scraping. Each row carries a symbol, timestamp, sentiment score, industry tag, and headline text. The `SentimentFeatureBuilder` in `src/intradaynet/features/sentiment_features.py` loads this file and produces 74 daily sentiment features per stock per day, covering stock-specific aggregates (mean, max, std, count, momentum, divergence from industry), industry-level aggregates, and macro-sentiment agreement signals.

**Market macro data:** The `MarketFeatureBuilder` in `src/intradaynet/features/market_features.py` downloads and caches daily close prices for 14 macro tickers from yfinance: India VIX (^INDIAVIX), Nifty 50 (^NSEI), S&P 500, crude Brent, gold, USD/INR, US 10Y, DXY, Dow, Nasdaq, Nikkei, Hang Seng, Shanghai Composite, and CBOE VIX. It also tracks 13 sector indices (auto, bank, financial services, FMCG, healthcare, IT, media, metal, pharma, private bank, PSU bank, realty, oil & gas). From these, it computes 16 market features (crude and gold returns, dollar/yield pressure, Asia sentiment, global volatility regime, VIX percentile, Nifty 5-day return, risk-on signal) and 12 India-specific features (sector index returns, market breadth, industry relative strength rank, secondary sector confirmation).

**Universe metadata:** The `ind_nifty500list.csv` file provides symbol-to-industry and symbol-to-company-name mappings for all 531 stocks, consumed by `src/intradaynet/universe.py`.

### 1.2 Feature Engineering

Feature construction is orchestrated by `src/intradaynet/open_safe_daily_features.py` through the `build_open_safe_daily_features` function. It takes raw minute data for one symbol, resamples to daily OHLCV, and computes features in 8 families:

1. **Price Action (21 features):** `prev_day_return`, `prev_day_volatility` (Parkinson), `prev_day_range`, `prev_day_atr`, `overnight_gap`, `prev_gap_size`, `prev_gap_direction`, `price_momentum_5d`/`10d`/`20d`, `close_vs_day_high`, `close_vs_day_low`, `range_expansion_5d`, `close_vs_vwap`, `vwap`, `sector_relative_strength`, `stock_vs_sector_1d`/`5d`, `breadth_momentum_confirmation`, `volatility_normalized_gap`, `volatility_normalized_momentum_5d`.

2. **Volume (4 features):** `prev_volume`, `volume`, `vol_momentum`, `volume_zscore`.

3. **Fibonacci (24 features):** Computed by `_append_fibonacci_features` using 5-day and 20-day lookbacks. For each horizon: `swing_range`, `prior_swing_position` (where close sits in the swing range), distances from prior close to the 23.6%, 38.2%, 50%, 61.8%, and 78.6% retracement levels, `fib_confluence` (fraction of levels within 0.5%), and `distance_to_nearest_fib`.

4. **Market Macro (16 features):** Crude oil return and 5-day change, gold return, USD/INR change, US 10Y yield change, DXY change, Asia sentiment (mean of Nikkei + Hang Seng + Shanghai), Dow and Nasdaq overnight returns, global volatility regime (CBOE VIX/100), India VIX 252-day percentile, Nifty 5-day return, S&P 500 overnight, commodity pressure (crude minus gold), dollar-yield pressure, composite risk-on signal.

5. **India-Specific (12 features):** Nifty intraday return, sector intraday return, VIX level, VIX change, market breadth, global cue, sector index previous return, sector 5-day return, sector volatility, industry relative strength rank (percentile across 13 sectors), sector breadth proxy, secondary sector confirmation.

6. **Sentiment (74 features):** Stock-level: `premarket_sentiment`, `premarket_sentiment_count`, `_max`, `_std`, `sentiment_5d_avg`, `sentiment_momentum`, `sentiment_spike`, `news_volume_shock`, `sentiment_surprise`, `sentiment_confidence`. Industry-level: `industry_premarket_sentiment`, `_count`, `_5d_avg`, `_momentum`, `_news_volume_shock`, `_surprise`. Cross-level: `industry_sentiment_stock_divergence`. Plus India market and global market features merged directly. Also `sentiment_macro_agreement` = sign(stock sentiment) × sign(global cue).

7. **Meta (1 feature):** `feature_version_code = 7.0`.

8. **Cross-Sectional (8 derived features):** Computed after all merges: sector relative strength, stock-vs-sector 1-day and 5-day, breadth-momentum confirmation, volatility-normalized gap and momentum.

After deduplication, the final feature set contains approximately 91 unique columns. The `FeatureContract` in `src/intradaynet/feature_contract.py` serves as the canonical registry — it validates that output DataFrames contain all required columns and logs any extras.

### 1.3 Target Labeling

Targets are computed by `src/intradaynet/v7.py` via `compute_directional_targets` and `compute_horizon_targets`. These functions operate on daily OHLC data and answer: "Did this stock achieve a +X% move before hitting a -X% stop today?" The algorithm:

1. Sets a LONG target = open_price × (1 + target_pct) and a LONG stop = open_price × (1 - stop_pct).
2. Sets a SHORT target = open_price × (1 - target_pct) and a SHORT stop = open_price × (1 + stop_pct).
3. Checks whether the day's high reached the long target, low hit the long stop, low reached the short target, high hit the short stop.
4. Labels as LONG (target hit), SHORT (target hit), or NEUTRAL (neither or both stopped out).
5. A cost buffer of 0.18% and an ambiguity band of 0.25% filter out marginal moves.

The daily target is a binary classification problem: `trade_label` ∈ {LONG, SHORT, NO_TRADE} with target version `v7_directional_executable_v1`.

### 1.4 Model Architecture

The V7 system uses 4 independent LightGBM models, all trained with identical hyperparameters:
- `n_estimators`: 1000 (with early stopping at 50 rounds)
- `max_depth`: 5
- `num_leaves`: 31
- `learning_rate`: 0.05
- `min_child_samples`: 50
- `subsample`: 0.8
- `colsample_bytree`: 0.8

| Model | Type | Objective | Output |
|-------|------|-----------|--------|
| `models["long"]` | LGBMClassifier | Binary classification (does stock go LONG?) | Probability [0,1] |
| `models["short"]` | LGBMClassifier | Binary classification (does stock go SHORT?) | Probability [0,1] |
| `models["up_mag"]` | LGBMRegressor | Regression (expected upward magnitude %) | Float % |
| `models["down_mag"]` | LGBMRegressor | Regression (expected downward magnitude %) | Float % |

Training is performed by `scripts/train_intraday_model.py`, which iterates all symbols in the chosen universe, builds daily features and targets for each, concatenates into a unified DataFrame (dates as rows, stocks as columns/stacked), performs an 80/20 temporal split, and trains all 4 models. The checkpoint is saved as `models/intraday_model_{UNIVERSE}.pkl` with a companion `.training_config.json` containing all hyperparameters, feature family counts, label distribution, and per-fold AUC/MAE metrics.

### 1.5 Confidence Scoring

The confidence scoring pipeline in `v7.py` operates as follows:

1. **Margin-adjusted confidence:** `confidence = 0.75 × primary_prob + 0.25 × (primary_prob - secondary_prob)`. The 0.25 margin term rewards models that are confident the stock will move in one direction but NOT the other — penalizing ambiguous predictions where both LONG and SHORT probabilities are high.

2. **Executable edge:** `edge = clip(predicted_magnitude, 0, target_pct) - cost_buffer_pct`. The predicted magnitude is capped at the target percentage (since beyond-target moves are unpredictable) and reduced by the transaction cost buffer (0.18%).

3. **Composite score:** `score = confidence × max(edge, 1e-6)`. This is the ranking metric: it rewards both high conviction and large expected moves net of costs.

4. **Probability calibration:** The `calibrator.py` module provides Platt scaling (logistic regression) and isotonic regression calibration. These are trained on validation-set raw probabilities and apply monotonic transformations to produce well-calibrated probabilities. The calibration report includes Expected Calibration Error (ECE) binned across 10 confidence buckets.

### 1.6 Output Processing and Recommendation

`scripts/recommend_intraday.py` and `scripts/post_open_picks.py` are the daily production scripts.

**Premarket mode** (run before 9:15 AM IST):
1. Load the trained model checkpoint (4 LGBM models + feature column names).
2. Download fresh market macro data and live news (optional).
3. For each symbol: load minute data, build features, take the last row (most recent date before the picks date), run all 4 model predictions.
4. Compute confidence and score for LONG and SHORT sides separately.
5. Apply the preferred filter gate: confidence ≥ min_confidence, predicted_magnitude ≥ min_predicted_magnitude, and (in post-open mode) alignment_ok and regime_ok.
6. Rank candidates by score, select top-K via `select_candidates()`, ensuring no more than `max_per_side` picks per direction.
7. Compute trade levels: `target_price = reference_price × (1 ± target_pct)`, `stop_loss_price = reference_price × (1 ∓ stop_pct)`.

**Post-open mode** (run after 9:15 AM):
1. If a premarket cache exists (`cache/premarket_cache_{universe}_{date}.json`), load it to skip expensive feature building.
2. For each cached symbol, extract the post-open minute data (09:15 to cutoff, default 09:30).
3. Apply post-open adjustments via `v7_modes.compute_post_open_adjustment()`:
   - Compute 6 alignment components: gap (weight 0.22), move-from-open (0.28), location-in-range (0.16), relative volume (0.14), VWAP displacement (0.10), market confirmation (0.10).
   - Each component is clamped to [-1, 1] and scaled by predicted magnitude.
   - `adjusted_probability = base_probability + 0.10 × alignment_score`.
   - `adjusted_magnitude = predicted_magnitude × (1 + 0.20 × alignment_score)`.
4. Also factor in market regime: trending vs choppy based on 5-day momentum, higher vs lower volatility based on previous day's volatility.

**Risk profiles** (defined as `StrategyConfig` dataclasses):

| Profile | Stop Loss | Target | Min Confidence | Min Magnitude |
|---------|-----------|--------|----------------|---------------|
| conservative | 0.5% | 1.0% | 0.65 | 0.013 |
| balanced | 1.0% | 1.5% | 0.65 | 0.012 |
| aggressive | 2.0% | 2.5% | 0.60 | 0.010 |

### 1.7 Readiness Assessment

Before finalizing recommendations, the system runs `evaluate_readiness()` from `v7.py`, which checks 9 boolean gates:
- **target_alignment:** Feature date matches expected previous business day.
- **freshness_ok:** Feature staleness ≤ max allowed business days.
- **mode_backtested:** The chosen mode (premarket/post-open) has been backtested.
- **locked_positive:** Locked backtest has positive net PnL.
- **forward_positive:** Forward blind period has positive net PnL.
- **hit_rate_ok:** Target hit rate ≥ 30%.
- **target_before_stop_ok:** Target-before-stop rate ≥ 18%.
- **live_data_ok:** At least max(5, 20% of universe) symbols loaded successfully.
- **logic_match_ok:** Logical consistency in outputs.

Results in one of: `READY` (all 9 pass), `SMALL_LIVE` (core 5 pass), `PAPER_ONLY` (3 pass), or `NOT_READY`.

### 1.8 Transaction Costs

The `IndianMarketCosts` model in `src/intradaynet/costs.py` computes round-trip costs with current NSE rates:
- Brokerage: ₹20 per order × 2 (buy + sell)
- STT: 0.025% on sell side only
- Exchange transaction: 0.00345% each side
- SEBI turnover fee: 0.0001% each side
- GST: 18% on (brokerage + exchange + SEBI)
- Stamp duty: 0.003% on buy side only
- Slippage: 0.05% each side

A ₹100,000 position incurs approximately ₹182 in total round-trip costs (0.182%), which defines the cost buffer used in executable edge calculations.

---

## 2. V8 Redesign (Under Construction — May 2026)

The V8 redesign is a complete architectural overhaul based on five design principles:
1. Learn from raw intraday price curves (not just engineered features).
2. Predict path-dependent outcomes ("hit +X% before -Y%?") rather than point-to-point returns.
3. Ensemble of 5 specialist models, each with its own feature set and predictive philosophy.
4. Full probability calibration (isotonic regression) on every model.
5. Portfolio construction as a first-class optimization problem (diversification, not just top-K).

### 2.1 Data Input

V8 shares the same minute-bar data source as V7 (`data/nifty500/*_minute.csv`) but processes it differently:
- **Data pipeline** (`v8/data_pipeline.py`): Loads and normalizes CSVs (8 candidate datetime column names, lowercase column mapping, numeric coercion), extracts per-session DataFrames (grouped by date, filtered to minimum 200 bars), caches sessions to disk.
- **Universe tiers** (`v8/universe_tiers.py`): Classifies stocks into Tier 1 (≥7 years of data, 2015–2026, ~334 stocks), Tier 2 (2–3 years, ~100 stocks), Tier 3 (<2 years, ~97 stocks). Tier assignment affects which features are used, confidence multiplier (T1: 1.0×, T2: 1.1×, T3: 1.25×), and slippage allowance (T1: 0.05%, T3: 0.20%).
- **Per-stock sentiment** (`v8/per_stock_sentiment.py`): Computes 8 daily sentiment features per stock (score_1d, score_3d, momentum_5d, count_1d, count_3d, volatility_5d, headline_bias, news_to_price_ratio) from yfinance news data, distinct from the market-level V7 sentiment.

### 2.2 Curve Embeddings

The signature innovation of V8. A Transformer-based masked autoencoder learns 128-dimensional representations of intraday OHLC curves, capturing shape patterns (V-recoveries, slow grinds, gap-and-traps, compression breakouts) without requiring labels.

**Model architecture** (`v8/curve_embedding.py`):
```
Input: (batch, 75 timesteps, 4 channels)  # OHLC at 5-min resolution
  → Linear projection to 128-dim
  → Sinusoidal positional encoding
  → Random masking (60% of positions)
  → 4-layer Transformer encoder (4 heads, 512-dim feedforward, GELU, pre-norm)
  → Lightweight linear decoder → reconstructed (batch, 75, 4)
  → MSE loss on masked positions only
```

At inference, the encoder runs unmasked: all 75 timesteps pass through the encoder, the output is mean-pooled across time, and projected through a linear → GELU → LayerNorm head to produce a 128-dim embedding vector.

**Training:** The `CurveTrainer` uses AdamW with OneCycleLR scheduling, gradient clipping at 1.0, and early stopping. Data is prepared by `prepare_curve_data()` which downsamples 1-min bars to 5-min, pads/truncates to 75 timesteps, and normalizes via z-score per day. Training is limited to Tier 1 stocks with a subsample cap of 15,000 sessions.

### 2.3 Barrier Targets

Unlike V7's point-to-point labeling, V8 uses path-dependent barrier targets (`v8/barriers.py`). For each session:

1. LONG barrier: target = open × (1 + target_pct), stop = open × (1 - stop_pct).
2. SHORT barrier: target = open × (1 - target_pct), stop = open × (1 + stop_pct).
3. The algorithm iterates through every minute bar, checking if high/low crosses the levels.
4. If both target and stop are hit, the one that occurred first (earlier bar index) wins.
5. Final label: LONG (long target hit first), SHORT (short target hit first), or NEUTRAL.

The `BarrierTarget` dataclass captures 25 fields per session: open/high/low/close, target and stop levels (both sides), hit booleans, hit minute indices, resolved labels, and the total bars processed. Multi-horizon support (H15, H30, H60, H375) truncates sessions to progressively shorter windows.

### 2.4 Daily Features

The V8 `DailyFeatureBuilder` (`v8/daily_features.py`) computes approximately 60 engineered daily features organized by specialist model:

| Specialist | Key Features (10-13 each) |
|------------|--------------------------|
| **Momentum** | return_1d through return_63d, momentum signals, RS vs sector, SMA distances |
| **Reversal** | price_position_20d/63d (0=bottom, 1=top of range), gap, overnight_return, rsi_14d, bollinger_position, volatility |
| **Breakout** | vol_contraction_5d/21d, volume_dryup_ratio, high_low_range_pct, atr_14d/percentile, inside_day_count_5d, narrow range |
| **Sentiment** | sentiment_score_1d/3d, sentiment_momentum_5d, sentiment_volatility_5d, article_count_1d/3d, headline_bias |
| **Macro** | vix_level, vix_trend_5d, nifty_vs_50dma/200dma, breadth, sp500_overnight, usdinr_change, crude_change, gold_return, dxy_change, asia_sentiment, risk_on_signal |
| **Market structure** | day_of_week, month, expiry_week, budget_day |

### 2.5 Signal Models (5 Specialist LightGBMs)

Each specialist is a `SignalModel` (`v8/signal_models.py`) — a wrapper around `lgb.Booster` with feature subset selection, auto-scale_pos_weight, isotonic calibration, and serialization. All use binary classification predicting the barrier target label.

| Specialist | Philosophy | Feature Count |
|------------|-----------|---------------|
| **Momentum** | Trend continuation — stocks with strong recent returns continue | 10 |
| **Reversal** | Mean reversion — overextended positions snap back | 10 |
| **Breakout** | Range expansion — contracting volatility precedes explosive moves | 7 |
| **Sentiment** | News-driven — sentiment shifts precede price moves | 8 |
| **Macro** | Market context — macro regime drives broad participation | 13 |

Each model is trained independently on its feature subset, then calibrated via isotonic regression on a held-out calibration period.

### 2.6 Regime Detector

The `RegimeDetector` (`v8/regime_detector.py`) clusters each trading day into one of 5 regimes using K-means (n_clusters=5) on 6 market-level features: VIX level, VIX 5-day change, Nifty ADX proxy, 20-day breadth, Nifty autocorrelation, and sector dispersion. The resulting cluster centers are heuristically mapped to interpretable labels:

| Regime ID | Label | Description |
|-----------|-------|-------------|
| 0 | strong_trend_up | High ADX, positive breadth |
| 1 | strong_trend_down | High ADX, negative breadth |
| 2 | choppy_reverting | Low ADX, mean-reverting patterns |
| 3 | high_vol_crisis | Elevated VIX, extreme moves |
| 4 | low_vol_compression | Depressed VIX, range-bound |

Each regime carries predefined specialist weights (e.g., in trending regimes, Momentum gets 0.40 weight; in compression, Breakout gets 0.45) and confidence scales (crisis regime requires 1.20× confidence multiplier to reduce trade count).

### 2.7 Meta-Ensemble

The `MetaEnsemble` combines the 5 specialist predictions using regime-weighted averaging:
```
ensemble_prob = Σ(w_i × prob_i) / Σ(w_i)
```
where `w_i` comes from the per-regime weight matrix. It supports both single-regime prediction (when all stocks share the same regime) and per-sample multi-regime prediction (when individual stocks may be in different regimes).

### 2.8 Portfolio Construction

V8's `PortfolioConstructor` (`v8/portfolio.py`) replaces simple top-K selection with a greedy diversified selector:

1. **Filter:** Apply tier-specific confidence multipliers (Tier 3 stocks need 25% higher confidence), check `min_confidence` and `min_expected_value` thresholds.
2. **Score:** `raw_score = probability × expected_edge`.
3. **Penalize:** For each candidate, compute penalized score:
   - `sector_penalty`: If same-industry picks already exceed `max_picks × max_sector_exposure` (default 40%), apply 2× penalty factor.
   - `correlation_penalty`: Compute average absolute Pearson correlation with already-selected stocks over a 21-day lookback window.
   - `penalized_score = raw_score × (1 - sector_penalty) × (1 - correlation_penalty × avg_corr)`.
4. **Greedy select:** Pick highest penalized-score candidates in order, respecting `max_long` and `max_short` direction caps.
5. **Position sizing:** `position_size = risk_per_trade_pct × capital / stop_loss_pct`, capped at `max_position_pct × capital` (default 20%), floored at 1% of capital. If total positions exceed capital, all are scaled down proportionally.

### 2.9 Walk-Forward Backtest Framework

The `WalkForwardBacktest` class (`v8/walk_forward.py`) implements an expanding-window validation protocol:

- **Train/test splits:** 4 folds with expanding windows: (2015–2020 → 2021), (2015–2021 → 2022), (2015–2022 → 2023), (2015–2023 → 2024).
- **Per-fold:** Train all 5 signal models on in-sample data, calibrate, ensemble, predict out-of-sample, simulate trades.

**Trade simulation** (`TradeRecord` dataclass): Each trade records entry/exit prices, exit reason (TARGET_HIT, STOP_HIT, CLOSE_EOD, TRAILING_STOP), PnL percentage and absolute value, position size, tier, regime, and contributing signal model.

**Backtest metrics** (`BacktestMetrics` dataclass): Total trades, winning/losing trades, win rate, profit factor (gross profits / gross losses), total PnL %, average winner/loser %, max drawdown (equity curve based), annualized Sharpe ratio (risk-free rate 6%), Calmar ratio (annualized return / max drawdown), expectancy (win_rate × avg_win - (1-win_rate) × abs(avg_loss)), and trades per trading day. All metrics are further stratified by regime, tier, and signal model.

**Baselines:** The framework computes a buy-and-hold benchmark from Nifty 50 data and a Monte Carlo random picker baseline (1000 simulations at a given win rate) to provide context for V8 performance. Comparison reports are saved as JSON with per-column breakdowns.

---

## 3. Live LightGBM Backend (Deferred)

The third pipeline, designed for per-minute intraday scalping, uses a flattened feature scheme. It was deferred because the daily premarket pipeline proved sufficient.

**Features:** `feature_contract.py` defines `flatten_intraday_window`, which takes a sliding window of 25 per-bar technical features (RSI, Bollinger, ATR, OBV slope, VWAP distance, candlestick ratios, etc.) plus 26 session-level context features and 74 sentiment features, and flattens them into a single ~669-dimensional vector per minute bar. The flattening computes mean, std, min, max over 5/15/30/60/120-bar windows for each of the 25 per-bar features, plus the last value, first value, and cross-window differences.

**Session features** (`session_features.py`): 26 daily context features including previous day's RSI/MACD/BB/ADX, overnight gap, volume z-score, day-of-week, expiry flags, 52-week high/low distance, average intraday range, and 20-day Fibonacci features.

**Models:** The backend uses 3 LightGBM models per horizon (15min, 30min, 60min, 375min): a direction classifier, gross return regressor, and net edge regressor, plus optional calibrators. Model management is handled by `model_bundle.py`, which loads manifest files tracking feature schemas, horizon configurations, and calibration metadata.

**Recommendation engine** (`recommendation.py`): Builds candidates by combining model outputs, scoring via `score = expected_net_edge × 1000 + probability_strength × 0.9 + liquidity_score × 0.8 + regime_alignment × 0.6`, then filters through risk profiles (conservative/balanced/aggressive) with configurable max total picks, max per side, confidence floors, and liquidity gates.

**Regime detection** (`regime.py`): Rule-based 5-regime classifier using VIX level, VIX change, Nifty 10-day trend, gap size, and expiry status. Each regime provides parameter overrides (e.g., volatile bear: direction threshold 0.65, min confidence 0.62, max positions 2; extreme regime: no trading at all).

---

## 4. Promotion Gates and Risk Controls

The `robustness.py` module provides the gating logic that controls whether a model advances from backtest to paper trading:

- **Minimum blind trades:** The forward-blind period must contain at least the configured minimum number of trades.
- **Profitability:** Net PnL must be positive.
- **Sharpe ratio:** Annualized Sharpe must be positive.
- **Stop rate cap:** The fraction of trades hitting stop-loss must not exceed 60%.
- **Confidence inversion check:** Trades are binned into 6 confidence buckets (0.0–0.2 through 0.75–1.0). If any higher-confidence bucket has a worse win rate than a lower-confidence bucket (by 5% or more), the model fails — indicating miscalibrated probabilities.

The `ReadinessVerdict` outputs either `PAPER_ONLY` (all gates pass) or `BLOCKED` (any gate fails), with reasons tracked per gate.

---

## 5. Configuration and Versioning

- **V7 config:** Baked into code — target_pct is the primary tunable (default 0.015), with strategy profiles defined as dataclasses in `v7.py`.
- **V8 config:** Centralized in `src/intradaynet/v8/config.py` via frozen `@dataclass` hierarchy: `TargetConfig`, `EmbeddingConfig`, `SignalModelConfig`, `MetaEnsembleConfig`, `PortfolioConfig`, `BacktestConfig`, `SentimentConfig`, `DailyFeaturesConfig`, all composed into `V8Config`. Supports JSON serialization.
- **YAML config** (`configs/intraday_config.yaml`): Legacy configuration for the original Phase 1 TCN+Attention deep learning approach (not actively used).
- **Feature contracts:** `feature_contract.py` serves as the single source of truth for all feature names across both V7 daily and backend flattened pipelines.

---

## 6. Current Results and State (May 2026)

| Metric | Value | Notes |
|--------|-------|-------|
| V7 Nifty 100 AUC | 0.610 | Modest predictive power |
| V7 Nifty 500 AUC | 0.625 | Better with broad universe |
| V7 backtest trades | 6 in 3 months (Nifty 100) | Model is conservative, few candidates pass threshold |
| V8 implementation | Core library complete | Barrier targets, curve embeddings, signal models, portfolio, walk-forward scaffold all coded |
| V8 training | Pending on real data | Scripts exist, `WalkForwardBacktest._run_fold()` is currently a placeholder |
| Stocks with data | 531 of 531 | Full Nifty 500 coverage through Apr 8, 2026 |
| Sentiment coverage | 2015–2025 + Mar–Apr 2026 | Combined historical + live news |
| Market cache | 25 macro/sector CSV files | Updated via yfinance downloads |

---

## 7. Script Inventory

| Script | Purpose | Pipeline |
|--------|---------|----------|
| `scripts/train_intraday_model.py` | Train 4 LGBM models on daily features | V7 |
| `scripts/recommend_intraday.py` | Generate daily LONG/SHORT picks | V7 |
| `scripts/post_open_picks.py` | Fast 2-phase post-open adjustment (cache + live) | V7 |
| `scripts/backtest_intraday_2025.py` | Risk-based walk-forward backtest | V7 |
| `scripts/train_v8.py` | Train full V8 pipeline (embeddings + 5 specialists + ensemble) | V8 |
| `scripts/backtest_v8.py` | Walk-forward backtest with V7/buy-hold/random comparison | V8 |
| `scripts/sync_data.py` | Download minute bars via yfinance | Data |
| `scripts/evaluate_equity.py` | Promotion gate evaluation from backtest summary | QA |
| `scripts/feature_analysis.py` | Per-feature quality diagnostics | QA |
| `scripts/daily_health_check.py` | Pre-flight checks (data freshness, model integrity) | Ops |
| `scripts/readiness_report.py` | V7 deployability report | Ops |
| `scripts/verify_p0.py` | P0 correctness tests (targets, contracts, calibration) | QA |
