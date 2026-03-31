# IntradayNet — Detailed Project Report

---

## 1. Project Overview

**IntradayNet** is a deep learning-based intraday stock prediction and recommendation system built specifically for the **Indian equities market (NSE Nifty 500)**. It analyzes minute-level price data, technical indicators, news sentiment, and global macroeconomic signals to generate **daily LONG and SHORT trade recommendations** with precise Entry, Target, and Stop-Loss levels.

The system is designed to be run **before market open (~8:30–9:15 AM IST)**, producing a concise actionable list that a trader can execute at market open and walk away.

---

## 2. The Idea

The core idea is to apply modern deep learning and gradient-boosted tree models to **intraday (1-minute bar) stock data** — a domain traditionally dominated by rule-based technical analysis or simple quantitative models.

Unlike daily-scale prediction systems (which forecast 1–10 day horizons), IntradayNet operates at **sub-hour granularity**, predicting price movements at 15-minute, 30-minute, 60-minute, and end-of-day (EOD) horizons. It fuses three signal streams:

1. **Price action** — 25 per-bar technical features (RSI, VWAP distance, Bollinger Bands, candlestick ratios, momentum, volume profile, etc.)
2. **Session context** — 20 session-level features (overnight gap, previous-day indicators, expiry flags, day-of-week, 52-week high/low proximity)
3. **Sentiment & macro** — 24 sentiment/market features (news sentiment via FinBERT, VIX, NIFTY intraday return, crude oil, gold, USD/INR, US yields, DXY, Asian markets, Dow/NASDAQ overnight)

The system runs a **multi-architecture tournament** — training and evaluating 5+ deep learning architectures (TCN+Attention, ResNLS, Compact CNN, Lightweight GRU, MLP-Mixer) plus LightGBM — and selects the best performer via walk-forward backtesting.

---

## 3. Motivation

| Problem | IntradayNet's Solution |
|---------|----------------------|
| Retail traders lack institutional-grade signal generation | Provides ML-driven daily picks with quantified confidence |
| Manual technical analysis is slow and subjective | Automates feature extraction across 499 stocks in seconds |
| Most ML stock systems predict daily — too slow for intraday | Purpose-built for 1-minute bar data with 4 prediction horizons |
| Sentiment integration in trading is ad-hoc | Systematic 3-layer sentiment pipeline (stock news + market regime + global macro) |
| No risk management baked into predictions | Every pick includes Entry, Target, Stop-Loss, and Confidence Score |
| Existing tools require constant monitoring | Morning Picks generates a set-and-forget daily trading plan |

The project also serves as a **research platform** to empirically compare deep learning architectures on financial time-series data — something rarely done rigorously in the retail trading community.

---

## 4. Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+ (MPS/CPU) |
| **Gradient Boosting** | LightGBM 4.0+ |
| **Data** | pandas 2.0+, NumPy 1.24+ |
| **ML Utilities** | scikit-learn 1.3+ |
| **Data Source** | yfinance (Yahoo Finance API) |
| **Sentiment** | FinBERT (HuggingFace), VADER, GDELT, Google News |
| **CLI Dashboard** | Rich (tables, progress bars, panels) |
| **Config** | YAML-based with dataclass parsing |
| **Build** | Hatchling (pyproject.toml) |

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MORNING PICKS PIPELINE                        │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────┐     │
│  │ yfinance │──▶│  499 Stocks  │──▶│  Feature Extraction  │     │
│  │ Live Data│   │ Minute CSVs  │   │  (25 per-bar +       │     │
│  └──────────┘   └──────────────┘   │   20 session +       │     │
│                                    │   24 sentiment)      │     │
│  ┌──────────┐                      └─────────┬───────────┘     │
│  │  Macro   │────────────────────────────────┘                 │
│  │ (VIX,    │   ┌─────────────────────┐                        │
│  │  Gold,   │──▶│  Model Inference    │                        │
│  │  Crude)  │   │  (PyTorch / LGBM)   │                        │
│  └──────────┘   └─────────┬───────────┘                        │
│                           │                                     │
│                    ┌──────▼──────┐                               │
│                    │  Filter &   │                               │
│                    │  Rank by    │                               │
│                    │  Score      │                               │
│                    └──────┬──────┘                               │
│                           │                                     │
│              ┌────────────▼────────────┐                        │
│              │  TOP 5 LONG  +  TOP 5   │                        │
│              │  SHORT PICKS            │                        │
│              │  (Entry/Target/SL)      │                        │
│              └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Model Architectures

### 6.1 Deep Learning Models (PyTorch)

| Model | Architecture | Parameters | Key Strength |
|-------|-------------|------------|-------------|
| **TCN + Attention** | 5 dilated causal conv blocks + 1 MHA layer | ~500K | Fast training, stable gradients, receptive field ~124 bars |
| **ResNLS** | ResNet blocks + BiLSTM | ~80K | Proven lightweight architecture, fast CPU inference |
| **Compact CNN** | 1D CNN with multi-scale kernels | ~150K | Minimal parameters, good for quick screening |
| **Lightweight GRU** | 2-layer GRU with attention pooling | ~120K | Efficient sequential modeling |
| **MLP-Mixer** | Patch-based MLP mixing | ~200K | Novel architecture, no convolutions or attention |

All PyTorch models share a common output format per prediction horizon:
- **Direction logit** → P(up) via sigmoid
- **Magnitude** → predicted % move
- **Confidence** → model's self-assessed certainty

### 6.2 LightGBM (Gradient Boosted Trees)

Separate `.lgb` booster files per horizon for direction and magnitude:
- `dir_H15.lgb`, `dir_H30.lgb`, `dir_H60.lgb`, `dir_H375.lgb` (direction)
- `mag_H15.lgb`, `mag_H30.lgb`, `mag_H60.lgb`, `mag_H375.lgb` (magnitude)

Features are flattened from the 120-bar window using rolling means, stds, and diffs at windows of 5, 30, and 120 bars, plus session and sentiment features.

---

## 7. Data Pipeline

### 7.1 Input Data

| Data Source | Format | Volume | Description |
|-------------|--------|--------|-------------|
| **Stock minute data** | CSV (499 files) | ~400M+ bars total | 1-minute OHLCV bars, 2015–2026, Nifty 500 stocks |
| **Sentiment data** | CSV | ~10 years | Daily sentiment scores (FinBERT/VADER), news headlines |
| **Macro data** | yfinance live | Real-time | VIX, NIFTY 50, crude oil, gold, USD/INR, US yields, DXY, Dow, NASDAQ, Nikkei, Hang Seng |

### 7.2 Feature Engineering

**Per-Bar Features (25 features per minute bar):**
- log_return, volume_ratio, vwap_distance, ema_9_distance, ema_20_distance
- rsi_14, bb_zscore, bb_width, body_ratio, upper_shadow_ratio, lower_shadow_ratio
- spread_pct, cum_volume_pct, time_normalized, orb_high_dist, orb_low_dist
- day_return, momentum_5, momentum_20, vol_momentum, atr_14
- close_vs_day_range, session_volatility, obv_slope, trade_intensity

**Session Features (20 features, static per session):**
- Previous-day RSI, MACD, Bollinger, trend strength, regime, volatility, ADX
- Overnight return, gap size/direction, close location, volume z-score
- Day-of-week, expiry flags, result season, 52-week high/low proximity, avg intraday range

**Sentiment & Macro Features (24 features):**
- Stock-level: premarket_sentiment (mean/max/std/count), 5d_avg, momentum, spike, price_divergence
- India market: NIFTY intraday return, sector return, VIX level/change, market breadth, global cue
- Global macro: crude oil return/change, gold return, USD/INR change, US 10Y yield, DXY, Asia sentiment, Dow/NASDAQ overnight, global volatility regime

### 7.3 Data Flow

```
Minute CSVs ──▶ Filter market hours (9:15–15:30)
            ──▶ Compute 25 per-bar features
            ──▶ Compute 20 session features
            ──▶ Merge 24 sentiment/macro features
            ──▶ Sliding window (seq_length=120 bars ≈ 2 hours)
            ──▶ Model input: (120, 25) + (20,) + (24,)
```

---

## 8. Usage

### 8.1 Morning Picks (Primary Use Case)

```bash
# PyTorch model
python scripts/morning_picks.py --model runs/intraday/resnls/best_model.pt

# LightGBM model
python scripts/morning_picks.py --model runs/lgbm/

# With filters
python scripts/morning_picks.py --model runs/lgbm/ --max-price 1000 --top-n 10 --horizon H375

# Backtest specific date
python scripts/morning_picks.py --model runs/lgbm/ --date 2026-03-30

# Save to CSV
python scripts/morning_picks.py --model runs/intraday/resnls/best_model.pt --save-csv picks.csv
```

### 8.2 Training

```bash
# Precompute features
python scripts/precompute_features.py

# Pre-batch training data
python scripts/prebatch_training_data.py

# Train PyTorch model
python scripts/train_intraday.py --model-type resnls --prebatched prebatched/

# Train LightGBM
python scripts/train_lgbm.py --prebatched prebatched/
```

### 8.3 Backtesting

```bash
python scripts/backtest_intraday.py --model runs/intraday/resnls/best_model.pt --strategy momentum
```

### 8.4 Key CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | Path to `.pt` file or `.lgb` directory |
| `--horizon` | `H60` | Prediction horizon (H15, H30, H60, H375) |
| `--top-n` | `5` | Max picks per direction |
| `--dir-threshold` | `0.60` | Min P(up) for LONG, max P(up) for SHORT |
| `--min-confidence` | `0.55` | Minimum model confidence |
| `--max-price` | `0` (none) | Max stock price filter |
| `--stop-loss` | `0.01` (1%) | Stop-loss percentage |
| `--save-csv` | auto | Save picks to CSV |
| `--no-download` | false | Skip live yfinance download |
| `--date` | auto | Specific date for picks |

---

## 9. Input & Output

### 9.1 Input

The system ingests three types of input at inference time:

1. **Historical minute-level OHLCV data** — Last 120 bars (~2 hours) per stock from `nifty500/` CSV files
2. **Sentiment CSV** — Daily sentiment scores from `sentiment/combined_sentiment_2015_2025.csv`
3. **Live macro data** — Downloaded via yfinance at runtime (VIX, crude, gold, USD/INR, NIFTY 50, global indices)

### 9.2 Output

The system outputs a formatted CLI dashboard with two tables:

**LONG Picks (example):**
```
#  Stock       Last Close   Entry      Target     Stop Loss   Target%   SL%     Confidence  Score
1  BAJFINANCE  8,450.00     8,450.00   8,619.00   8,365.50    +2.00%    -1.0%   0.78        0.0156
2  RELIANCE    2,890.50     2,890.50   2,948.31   2,861.60    +2.00%    -1.0%   0.72        0.0144
```

**SHORT Picks (example):**
```
#  Stock       Last Close   Entry      Target     Stop Loss   Target%   SL%     Confidence  Score
1  ASTRAL      1,920.00     1,920.00   1,881.60   1,939.20    +2.00%    -1.0%   0.81        0.0162
2  SCHNEIDER   3,450.00     3,450.00   3,381.00   3,484.50    +2.00%    -1.0%   0.75        0.0150
```

**Output columns:**
- **Stock** — NSE ticker symbol
- **Last Close** — Official NSE closing price (entry price)
- **Target** — `Entry × (1 + |magnitude|)` for LONG, `Entry × (1 - |magnitude|)` for SHORT
- **Stop Loss** — `Entry × (1 - stop_loss%)` for LONG, `Entry × (1 + stop_loss%)` for SHORT
- **Confidence** — Model's self-assessed probability (0–1)
- **Score** — `Confidence × |Magnitude|` (primary ranking metric)

**CSV output** (via `--save-csv`) includes: `picks_for_date, stock, direction, last_close, entry_price, target_price, stop_loss, predicted_move_pct, confidence, score, ref_date, horizon, model_type`

---

## 10. Working — Step-by-Step Pipeline

```
Step 1: DATA SYNC
  ├─ Read existing minute CSVs from nifty500/
  ├─ Determine last recorded timestamp per stock
  ├─ Download new 1-minute bars via yfinance (last 7 days)
  ├─ Append to CSVs (deduplication by timestamp)
  └─ Download official daily close prices (NSE closing auction)

Step 2: MACRO DOWNLOAD
  ├─ Download VIX, NIFTY 50, crude oil, gold, USD/INR
  ├─ Download US 10Y yield, DXY, Dow, NASDAQ, Nikkei, Hang Seng
  ├─ Cache to market_data_cache/
  └─ Compute 10 global macro features per date

Step 3: MODEL LOADING
  ├─ Load config from intraday_config.yaml
  ├─ Detect model type (PyTorch .pt or LightGBM .lgb directory)
  ├─ Load weights, detect sentiment feature count (backward compat)
  └─ Set model to eval mode

Step 4: FEATURE COMPUTATION (per stock)
  ├─ Load minute CSV → filter to market hours (9:15–15:30)
  ├─ Compute 25 per-bar features (RSI, VWAP, BB, momentum, etc.)
  ├─ Compute 20 session features (overnight gap, prev-day indicators)
  ├─ Merge 24 sentiment/macro features
  ├─ Extract last 120-bar window
  ├─ Get official daily close price
  └─ Return (window, session, sentiment, last_close, ref_date)

Step 5: INFERENCE
  ├─ PyTorch: torch.no_grad() forward pass
  │   → direction_logits (B, 4), magnitudes (B, 4), confidences (B, 4)
  └─ LightGBM: flatten features → predict direction prob + magnitude

Step 6: FILTERING & RANKING
  ├─ Remove stocks below min_confidence threshold
  ├─ Classify: LONG if P(up) ≥ dir_threshold, SHORT if P(up) ≤ 1-dir_threshold
  ├─ Compute Score = Confidence × |Magnitude|
  └─ Sort by Score descending, take top-N per direction

Step 7: OUTPUT
  ├─ Display Rich CLI dashboard (LONG table + SHORT table)
  ├─ Print summary (avg move, avg confidence, risk:reward ratio)
  └─ Optionally save to CSV
```

---

## 11. Backtest Results (ResNLS Model, H60 Horizon)

| Metric | Value |
|--------|-------|
| **Initial Capital** | ₹1,00,000 |
| **Final Equity** | ₹3,36,173 |
| **Total P&L** | ₹2,36,173 |
| **Total Return** | 236.17% |
| **Total Trades** | 1,205 |
| **Win Rate** | 56.68% |
| **Long Trades** | 1,103 |
| **Short Trades** | 102 |
| **Sharpe Ratio** | 7.29 |
| **Max Drawdown** | 3.68% |
| **Profit Factor** | 1.79 |
| **Position Size** | ₹1,00,000 per trade |
| **Max Concurrent** | 5 positions |
| **Brokerage** | ₹20/order |
| **Slippage** | 0.05% |

---

## 12. Trained Models

### PyTorch Checkpoints (`runs/intraday/`)
- `resnls/best_model.pt` — ResNLS (ResNet + BiLSTM), best performer
- `compact_cnn/` — Compact CNN
- `lightweight_gru/` — Lightweight GRU
- `best_model.pt` — TCN + Attention

### LightGBM Boosters (`runs/lgbm/`)
- `dir_H15.lgb`, `dir_H30.lgb`, `dir_H60.lgb`, `dir_H375.lgb` — Direction models
- `mag_H15.lgb`, `mag_H30.lgb`, `mag_H60.lgb`, `mag_H375.lgb` — Magnitude models

---

## 13. Project File Structure

```
intraday_antigravity/
├── configs/
│   └── intraday_config.yaml          # Model, training, and backtest config
├── src/intradaynet/
│   ├── config.py                     # Config loading (dataclasses + YAML)
│   ├── dataset/
│   │   └── intraday_dataset.py       # PyTorch Dataset for minute data
│   ├── features/
│   │   ├── per_bar_features.py       # 25 per-bar technical features
│   │   ├── session_features.py       # 20 session-level features
│   │   ├── sentiment_features.py     # 24 sentiment/macro features
│   │   └── market_features.py        # Global macro data downloader
│   └── models/
│       ├── tcn_attention.py          # TCN + Attention architecture
│       ├── resnls_intraday.py        # ResNLS (ResNet + BiLSTM)
│       ├── compact_cnn.py            # Compact CNN
│       ├── lightweight_gru.py        # Lightweight GRU
│       ├── mlp_mixer.py              # MLP-Mixer
│       └── intraday_loss.py          # Multi-task loss function
├── scripts/
│   ├── morning_picks.py              # Pre-market recommendation engine
│   ├── train_intraday.py             # PyTorch model training
│   ├── train_lgbm.py                 # LightGBM training
│   ├── backtest_intraday.py          # Intraday backtester
│   ├── backtest_pro.py               # Advanced backtester
│   ├── precompute_features.py        # Feature precomputation
│   ├── prebatch_training_data.py     # Pre-batch data for training
│   ├── threshold_sweep.py            # Hyperparameter sweep
│   ├── hit_rate.py                   # Hit rate analysis
│   └── verify_picks.py               # Pick verification
├── nifty500/                         # 499 stock minute CSVs
├── sentiment/                        # Sentiment data CSVs
├── prebatched/                       # Pre-batched training data (.npz)
├── runs/
│   ├── intraday/                     # PyTorch model checkpoints
│   └── lgbm/                         # LightGBM booster files
├── recommendations/                  # Historical morning pick CSVs
├── backtest_results/                 # Backtest output (trades, equity, summary)
├── market_data_cache/                # Cached macro data
├── features_cache/                   # Cached computed features
├── IntradayNet_Architecture_Plan.md.resolved  # Architecture research doc
├── README_morning_picks.md           # Morning Picks usage guide
├── PROJECT_REPORT.md                 # This file
├── recommendations_accuracy.csv      # Pick outcome tracking
├── pyproject.toml                    # Build config (Hatchling)
└── uv.lock                          # Dependency lock file
```

---

## 14. Current State & Next Steps

The project is **fully functional** with:
- 5 trained PyTorch models + 8 LightGBM boosters
- 19 historical morning pick sessions saved
- Backtested results showing 236% return with 56.7% win rate
- Live daily recommendation pipeline via `morning_picks.py`

The user is now planning to build a **backend + frontend** to:
1. Serve predictions via a web API (backend)
2. Display picks in a dashboard UI (frontend)
3. Track historical pick accuracy
4. Enable real-time monitoring and alerts
