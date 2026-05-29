Good, I have the full picture. Here's a completely fresh design.

---

# IntradayNet — Complete Redesign Plan

## Design Principles

1. **The minute data is the crown jewel.** 11 years of 1-min OHLCV for 531 stocks is rare. The system should be built around learning from raw intraday curves, not just 85 hand-crafted aggregations of them.

2. **Predict path, not point.** "Will this stock hit +X% before -Y% today?" is a more actionable question than "What will the close price be?"

3. **One model doesn't fit all markets.** Different edges work in different regimes. An ensemble of specialized models, each exploiting a different market inefficiency, will be more robust than a single black box.

4. **Calibration > Accuracy.** A probability of 0.65 that actually means 65% is worth more than an AUC of 0.70 with miscalibrated scores.

5. **Portfolio construction is a first-class problem.** Picking 5 stocks that all move together on a bank-nifty theme isn't diversification — it's a single bet with extra steps.

---

## System Architecture

```
┌────────────────┐
│                       DATA LAYER                               │
│ Minute Bars (531 stocks, 2015-2026) │ Daily EOD (84 stocks) │
│ Sentiment (83 stocks)              │ Macro (25 indices)     │
│ Sector Indices                      │ US Global Context      │
└────────┬────────────────┘
              ↓
┌────────────────┐
│                   FEATURE STORE (daily increment)              │
│ ┌────────┐ ┌────────────────┐ ┌────────────────┐  │
│ │ Curve      │ │ Daily Technical │ │ Context / Regime │  │
│ │ Embedings │ │ Features (30-40) │ │ Features (15-20) │  │
│ │ (learned)  │ │ (enginered)    │ │ (macro+sentiment) │  │
│ └────────┘ └────────────────┘ └────────────────┘  │
└────────┬────────────────┘
              ↓
┌────────────────┐
│                   SIGNAL LAYER (ensemble)                      │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────┐ │
│ │ Momentum │ Reversal │ Breakout │ Sentiment│ │ Macro │
│ │ Signal  │ Signal  │ Signal  │ Signal  │ Signal│ │
│ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──┬────┘ │
│      └────────┴────────┴────────┴────────┘      │
│                             ↓                                  │
│                   ┌────────────────┐                         │
│                   │ Meta-Ensemble  │                         │
│                   │ (regime-weighted│                         │
│                   │  combination)  │                         │
│                   └────────────────┘                         │
└────────┬────────────────┘
              ↓
┌────────────────┐
│                   RECOMMENDATION LAYER                         │
│ ┌────────┐ ┌────────┐ ┌────────────────┐  │
│ │ Risk Scoring │ │ Portfolio   │ │ Post-Open Adjustment │  │
│ │ (calibrated │ │ Construction │ │ (gap, VWAP, open    │  │
│ │ probability)│ │ (diversify) │ │ drift confirmation) │  │
│ └────────┘ └────────┘ └────────────────┘  │
└────────┬────────────────┘
              ↓
        Final Picks: LONG/SHORT + Entry + Target + Stop
```

---

## Component Designs

### 1. Stock Universe Tiers

Not all 531 stocks are equal. Categorize by data quality:

| Tier | Count | Data Range | Treatment |
|------|-------|-----------|-----------|
| **Tier 1** | ~334 | Full 2015-2026 | All features + embeddings |
| **Tier 2** | ~100 | 2023-2026 (2-3 yrs) | Reduced feature set, no embedding |
| **Tier 3** | ~97 | < 2 years (IPOs) | Momentum/volume features only, higher confidence threshold |

### 2. Feature Engineering (3 streams)

**Stream A: Learned Curve Embedings (deep learning)**

Train a masked autoencoder on minute-level curves:
- Input: One stock-day as a 375×4 tensor (375 minutes × OHLC)
- Architecture: Transformer encoder with ~60% random masking of minute bars
- Pre-training task: Reconstruct masked bars
- Output: A 128-dim embedding vector per stock-day

This is trained once on ALL Tier 1 stocks' entire history. The embedding captures intraday "shape" patterns — V-recoveries, slow grinds, gap-and-traps, compression breakouts — without needing labels. It's a reusable foundation model.

**Why this instead of hand-crafted features?** Because patterns like "the stock traded in a 0.3% range for 4 hours then broke out with 3x normal volume in the last hour" are hard to capture with 14 Fibonacci retracement features but easy for a transformer to learn.

**Stream B: Enginered Daily Features (~30-40)**

Keep a compact set of proven features:
- **Returns**: 1d, 5d, 10d, 21d, 63d (log returns)
- **Volatility**: Parkinson, Garman-Klass, realized vol (5d, 21d)
- **Volume**: Relative volume vs 20d MA, volume trend
- **Price position**: Where is close relative to N-day range (like a stochastic)
- **Gap**: Today's gap from yesterday's close
- **Overnight**: Overnight return (for use in next-day prediction)
- **Relative strength**: Stock return minus sector index return (5d, 21d)
- **Intraday metrics**: High-low range, close vs VWAP, afternoon vs morning return

**Stream C: Context Features (~15-20)**

- **Market regime**: VIX level (low/medium/high), VIX trend, Nifty vs 50/200 DMA
- **Breadth**: % of Nifty 500 stocks above 20-day MA
- **Sector rotation**: Sector index returns (1d, 5d)
- **Global**: S&P 500 overnight return, USD/INR change, crude change
- **Calendar**: Day of week, month, expiry week dummy, budget day dummy
- **Sentiment**: Market-level news sentiment aggregate (not stock-level since coverage is thin)

### 3. Target Definition (Redesigned)

Current target: "Did the stock move +X% from open to close?" — This is a noisy point-to-point measurement.

**New target — Path-dependent barrier:**
- LONG target = 1 if price hits `open * (1 + target_pct)` before hitting `open * (1 - stop_pct)` during the day
- SHORT target = 1 if price hits `open * (1 - target_pct)` before hitting `open * (1 + stop_pct)` during the day
- Null/neutral if neither barrier is hit

Why this is better:
- It directly answers the trader's question: "If I enter at open with a stop-loss, will I hit my target?"
- It's less noisy than point-to-point returns (a stock that wanders +1.5% then closes flat is still a winner)
- It maps 1:1 to how the recommendations are actually used

Parameters to tune: `target_pct` (1.5%? 2%? variable by volatility?) and `stop_pct` (1%? ATR-based?)

### 4. Model Architecture — Ensemble of Specialists

Instead of 4 monolithic models (LONG/SHORT × classify/regress), train 5 specialized signal models:

| Signal | What it predicts | Features emphasized | When it works |
|--------|----------------|----------------|---------------|
| **Momentum** | Continuation of recent trend | Returns, curve embeddings showing trend days, relative strength | Trending markets |
| **Reversal** | Mean reversion after extreme moves | Overbought/oversold from range, gap size, volatility spike | Choppy/sideways markets |
| **Breakout** | Range expansion after compression | Volatility contraction, volume drying up then spiking, curve embedding showing compression patterns | Low-vol regimes |
| **Sentiment** | News-driven moves | Sentiment scores, headline count, sentiment change | High-news days, event-driven |
| **Macro** | Market-direction bets | VIX, breadth, global indices | Regime-transition days |

Each is a calibrated LightGBM binary classifier predicting the barrier target. Each model outputs a probability for each stock each day.

**Meta-Ensemble:**
The final score for each stock is a regime-weighted combination:

```
final_score = w₁ × P(momentum) + w₂ × P(reversal) + w₃ × P(breakout) + w₄ × P(sentiment) + w₅ × P(macro)
```

Where weights depend on current market regime (detected by clustering VIX, breadth, and Nifty trend). In a trending market, momentum gets higher weight. In a choppy market, reversal gets higher weight.

This naturally diversifies predictions — different signals fire on different stocks on different days.

### 5. Market Regime Detection

Cluster each trading day into one of 5 regimes using these features:
- VIX level and VIX 5-day change
- Nifty 50 trend strength (ADX or return/vol ratio)
- % stocks above 20-day MA (breadth)
- Nifty correlation with previous day (1 = trending, 0 = choppy)
- Sector dispersion (high = stock-picker's market, low = macro-driven)

Regimes:
1. **Strong Trend Up** — weight momentum high, macro long
2. **Strong Trend Down** — weight momentum high, macro short
3. **Choppy/Mean-Reverting** — weight reversal high, breakout
4. **High Vol / Crisis** — reduce all positions, wider stops
5. **Low Vol / Compression** — weight breakout high

### 6. Calibration Layer

Every model probability goes through isotonic regression calibration on a held-out validation set. This ensures that when the system says "65% confidence," it truly means 65 out of 100 such predictions were correct historically.

Additionally, add a **confidence diagnostic**:
- For each prediction, compute: how many similar historical predictions (similar confidence, similar regime, similar stock volatility) were correct?
- Reject predictions below a configurable "reliability threshold"

### 7. Portfolio Construction

The current system picks the top-K stocks by score. This leads to correlated picks.

**New approach:**

1. **Score stocks**: `score = calibrated_probability × expected_edge` (target_pct for longs, adjusted for transaction costs)

2. **Filter by minimum thresholds**: `min_confidence > 0.58`, `min_expected_value > 0`

3. **Diversify**: Greedy selection with sector penalty
  - Start with highest-score stock
  - For next pick, penalize stocks in the same sector (infer sectors from sector index correlation, since we lack explicit sector labels)
  - Penalize stocks highly correlated to already-selected stocks (from historical returns)
  - Continue until we have N picks or no stocks pass thresholds

4. **Position sizing**: Equal risk allocation — each pick gets `risk_budget / stop_loss_pct` of capital

5. **Direction balance**: In a strong bull market, allow up to 4 LONG + 1 SHORT. In neutral, 3+2.

### 8. Post-Open Adjustment

When the market opens, reality arrives:

**Pre-market → Post-open adjustments:**
- **Gap fade**: If a LONG pick gaps up >1.5%, reduce confidence (gap likely priced in)
- **Gap continuation**: If a LONG pick gaps up 0.3-1% with strong opening volume, increase confidence
- **VWAP confirmation**: If price is above VWAP 15 minutes in for LONG picks (below for SHORT), confirm
- **Open drift**: If a stock opens against the prediction by more than 0.5× stop, cancel the pick
- **Breadth check**: If market breadth at open contradicts the pick direction, reduce position size

### 9. Backtesting Framework

Walk-forward by calendar year (train on 2015-2020, test on 2021; train on 2015-2021, test on 202; etc.)

**Realistic assumptions:**
- Entry at open price + realistic slippage (0.05% for Tier 1, 0.1% for Tier 2, 0.2% for Tier 3)
- Exit at target or stop using intraday high/low (can the trade actually get filled?)
- Transaction costs: brokerage + ST + stamp duty + GST (~0.05% round trip for delivery, higher for intraday)
- Position limits: No position > 10% of portfolio
- Trailing stop option (configurable)

**Metrics per regime:**
- Win rate, profit factor, Sharpe, max drawdown, Calmar
- Broken down by market regime (is the system profitable in all regimes or only trending?)
- Broken down by tier (do Tier 3 stocks add value or just costs?)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- Data pipeline: normalize all minute data, handle `_new` files, validate quality
- Universe tier classification
- Barrier target computation and validation
- Walk-forward backtesting scaffold

### Phase 2: Curve Embeding Model (Week 2-4)
- Build masked autoencoder for minute curves
- Train on Tier 1 stocks (GPU recommended)
- Validate: do embeddings cluster sensibly? (trend days near trend days, reversals near reversals)
- Generate embeddings for all historical days

### Phase 3: Signal Models (Week 4-6)
- Train 5 specialist LightGBM models
- Calibrate each with isotonic regression
- Combine with regime-weighted meta-ensemble
- Backtest baseline performance

### Phase 4: Portfolio & Post-Open (Week 6-8)
- Implement diversification algorithm
- Implement post-open adjustment logic
- End-to-end backtest with realistic costs
- Compare against buy-and-hold and current system baseline

### Phase 5: Production (Week 8-10)
- Daily automation: data sync → feature generation → prediction → recommendations
- Health checks, monitoring, drift detection
- A/B testing framework for model improvements

---

## Key Design Decisions (Need Your Input)

Before I start implementing, a few decisions to lock in:

1. **Target horizon**: Single-day barrier target, or multi-horizon (30min, 60min, EOD)? Single-day is simpler and maps to pre-market picks. Multi-horizon would also enable intraday trading.

2. **Deep learning vs pure classical**: The curve embedding model adds complexity but could capture patterns hand-crafted features miss. Worth the investment? Or start with engineered features only and add embeddings later?

3. **Number of picks**: 5 balanced? 3 aggressive? Configurable by risk profile? The diversification algorithm naturally handles this.

4. **Sentiment scope**: Use sentiment as market-level only (since coverage is thin), or build the sentiment specialist only for stocks with news data?

5. **Backtest baseline**: Should the first backtest compare against the current system's results, or just establish standalone metrics?

What do you think about this direction?