# IntradayNet / Intraday_Antigravity — Critical Review

## Overall Impression

This is a well-engineered system with thoughtful architecture choices. The multi-modal feature set, multiple model paradigms, regime detection, and realistic cost modeling put it ahead of most retail quant projects. That said, there are several significant concerns that range from potential data leakage to overfitting risks to structural gaps that would undermine live profitability.

---

## Critical Issues

### 1. The Backtest Results Are Almost Certainly Overstated

The ResNLS backtest showing **236% return, 7.29 Sharpe, 3.68% max drawdown** is an extraordinary claim. For context:

- Renaissance Medallion averages ~66% annually before fees with Sharpes around 3-4
- A 7.29 Sharpe on intraday Indian equities with a 56.7% win rate is implausible unless the backtest period is very short or there's leakage

**Red flags:**

- **3.68% max drawdown** with 1,205 trades is suspiciously low — this suggests the backtest may not account for correlated losses (multiple positions hitting stops on the same gap-down day)
- The **LightGBM V2 metrics tell a very different story**: 52.7% direction accuracy and 0.516 AUC at H60 is barely above random. This massive disconnect between the LGBM live metrics and the ResNLS backtest suggests the ResNLS backtest has issues
- 1,103 longs vs 102 shorts — this is a **10:1 long bias** during what was largely a bull market (2015-2026 Nifty). The system may simply be capturing beta, not alpha

**Verdict:** The backtest likely suffers from survivorship bias (Nifty 500 constituents change), look-ahead bias in feature construction, or insufficient out-of-sample discipline.

### 2. Potential Look-Ahead Bias in Feature Engineering

Several features are suspicious:

- **Session features using "previous day indicators"** — how exactly is the previous day's close defined? If you're using the yfinance daily close that includes the closing auction (3:30 PM), but your minute bars stop at 3:29 PM, you may be leaking information
- **Sentiment features** — "pre-market sentiment" is fine, but if sentiment scores are computed daily without strict timestamp filtering, you could be using intraday news that arrives after your entry signal
- **ORB (Opening Range Breakout) distances as per-bar features** — if ORB is computed from the first 15-30 minutes and you're making predictions at market open, this is future data. If it's previous day's ORB, this should be made explicit

### 3. The 120-Bar Window Problem

Extracting the "last 120-bar window (~2 hours)" for morning picks raises a critical question: **which 2 hours?**

- If it's the last 2 hours of the previous trading day (1:30-3:30 PM), you're betting that end-of-day microstructure predicts next-morning direction. This is a weak signal at best
- If it's supposed to be the first 2 hours of the current day, you can't use it for pre-market picks
- The system says it runs at 8:30-9:15 AM IST, before market open — so it cannot use any current-day minute bars

### 4. Target Construction Concerns

- **0.3% threshold for direction** — after accounting for 0.15-0.20% round-trip costs, your net edge on a "correct" prediction near the threshold is only 0.10-0.15%. This is razor-thin
- **Magnitude clipped at ±5%** — reasonable, but the distribution of intraday moves is heavily concentrated in ±0.5%. The model will overwhelmingly see small moves during training, making it poorly calibrated for the larger moves that actually matter for profitability
- **Multiple horizons (H15, H30, H60, H375)** — which one actually drives the morning picks? If you're training on all four but deploying on one, the others add noise to the training process

### 5. Regime Detection Without Regime-Conditioned Training

You detect 5 market regimes and adjust thresholds at inference time, but the models themselves are trained on all regimes pooled together. This means:

- The model learns an **average relationship** across all regimes
- You then apply post-hoc filters that the model wasn't optimized for
- A VOLATILE_BEAR regime might have fundamentally different feature-return relationships than CALM_BULL, but the model treats them identically

---

## Moderate Concerns

### 6. Feature Count vs Signal-to-Noise

69 features × 120 timesteps = 8,280 input dimensions per sample (for deep learning) or 625 flattened features (for LGBM). Given that intraday return prediction has notoriously low signal-to-noise ratio, this is a **curse of dimensionality** problem. The LightGBM's near-random AUC (0.516) confirms this — most features are noise.

### 7. No Ensemble Strategy Documented

You have 5 deep learning architectures + LightGBM, but the system description doesn't explain how they're combined. Are you:

- Picking the best one? (overfitting to validation set)
- Averaging predictions? (diluting any edge)
- Stacking? (added complexity, more overfitting risk)

### 8. Survivorship Bias in Universe

Using "Nifty 500 stocks" with data from 2015-2026 means the current constituent list. Stocks that were in the Nifty 500 in 2015 but got delisted/demoted aren't in your training data. This creates a systematic bias toward stocks that performed well enough to stay in the index.

### 9. Sentiment Data Quality

FinBERT/VADER on financial headlines is known to be noisy. Without knowing:

- The source of headlines (which provider, coverage per stock)
- How missing sentiment is handled (many small-caps will have zero coverage)
- Whether sentiment is actually predictive in your specific context (have you tested ablation?)

...it could be adding noise rather than signal.

### 10. Fixed Position Sizing Is Suboptimal

₹1,00,000 per trade regardless of confidence, volatility, or regime is leaving money on the table. A CALM_BULL pick with 0.95 confidence and a VOLATILE_BEAR pick with 0.56 confidence get identical capital allocation.

---

## Structural Gaps

### 11. No Walk-Forward Validation Framework

The system has `train_intraday.py` and `backtest_lgbm_v2.py` as separate scripts, but there's no documented **walk-forward** or **expanding window** validation. Without this, every reported metric is suspect.

### 12. No Execution Simulation

The system assumes execution at market open. In reality:

- Market open in NSE is chaotic (pre-open auction, first-minute volatility)
- Slippage of 0.05% is optimistic for small-caps at open
- No modeling of order book depth, impact cost, or partial fills

### 13. No Correlation/Diversification Logic

The system picks top-N per direction, but doesn't check whether picks are correlated. Picking SAIL, TATA STEEL, and JINDAL STEEL as your 3 longs means a single sector event wipes all positions.

### 14. No Model Staleness Detection

Models degrade. There's no documented mechanism to detect when a trained model's live predictions have diverged from its training performance (concept drift).

---

## Detailed Suggestions for Robustness and Profitability

### A. Fix the Foundation

**A1. Implement Strict Walk-Forward Validation**

```text
Train:   2015-01 ─── 2020-12
Val:     2021-01 ─── 2021-06
Test:    2021-07 ─── 2021-12
         ──── slide forward 6 months ────
Train:   2015-01 ─── 2021-06
Val:     2021-07 ─── 2021-12
Test:    2022-01 ─── 2022-06
         ... repeat ...
```

Report the **concatenated out-of-sample** performance across all test folds. This is your true expected performance.

**A2. Audit for Look-Ahead Bias**

Write a unit test that, for any given prediction timestamp T:

- Asserts no feature uses data from time > T
- Asserts the target is computed only from data at time > T
- Asserts the stock's Nifty 500 membership status at time T (not current)

**A3. Build a Point-in-Time Universe**

```python
# Maintain a mapping of Nifty 500 constituents over time
universe = {
    "2015-01": ["RELIANCE", "TCS", ...],  # 500 stocks as of Jan 2015
    "2015-07": ["RELIANCE", "TCS", ...],  # rebalanced
    ...
}
# During training, only use stocks that were IN the index at that time
```

### B. Improve Signal Quality

**B1. Aggressive Feature Selection**

Run feature importance with proper out-of-fold methodology:

```python
# Permutation importance on TRUE out-of-sample data
from sklearn.inspection import permutation_importance

# Only keep features that are consistently important
# across multiple time periods
important_features = []
for fold in walk_forward_folds:
    result = permutation_importance(
        model, X_test_fold, y_test_fold,
        n_repeats=10, scoring='roc_auc'
    )
    top_k = np.where(result.importances_mean > 0.001)[0]
    important_features.append(set(top_k))

# Intersect across folds — features that are ALWAYS important
stable_features = set.intersection(*important_features)
```

Target: reduce from 69 features to 15-25 that are stable across time.

**B2. Regime-Conditioned Models**

Instead of one model + post-hoc regime filters, train separate model heads or entirely separate models per regime:

```python
class RegimeConditionedModel(nn.Module):
    def __init__(self, base_dim, n_regimes=5):
        self.shared_encoder = SharedEncoder(base_dim)
        self.regime_heads = nn.ModuleList([
            PredictionHead(base_dim) for _ in range(n_regimes)
        ])

    def forward(self, x, regime_id):
        shared = self.shared_encoder(x)
        return self.regime_heads[regime_id](shared)
```

**B3. Calibrated Probabilities**

Your confidence scores need to be calibrated. A model saying 0.70 confidence should win ~70% of the time. Add post-hoc calibration:

```python
from sklearn.calibration import CalibratedClassifierCV

# Or for neural nets, use temperature scaling
class TemperatureScaling(nn.Module):
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature
```

Validate calibration with reliability diagrams on out-of-sample data.

**B4. Add Microstructure Features**

You have 1-minute bars but are underutilizing the microstructure information:

| Feature                              | Rationale                                  |
| ------------------------------------ | ------------------------------------------ |
| Kyle's lambda (price impact)         | Estimates informed trading activity        |
| Order flow imbalance (OFI)           | Net buying/selling pressure from tick data |
| Amihud illiquidity                   | Captures liquidity risk                    |
| Intraday volatility signature        | How volatility decays through the day      |
| Autocorrelation of returns (lag 1-5) | Mean-reversion vs momentum at micro level  |

### C. Improve Execution Realism

**C1. Realistic Entry Simulation**

Don't assume entry at the open price. Model entry as VWAP of the first 5 minutes:

```python
def realistic_entry_price(minute_bars, entry_time, n_minutes=5):
    """Simulate entry over first 5 minutes using VWAP"""
    window = minute_bars[entry_time : entry_time + n_minutes]
    vwap = (window['close'] * window['volume']).sum() / window['volume'].sum()
    # Add random slippage component
    slippage = np.random.uniform(0.01, 0.08) / 100  # 1-8 bps
    return vwap * (1 + slippage)  # for longs
```

**C2. Impact-Adjusted Position Sizing**

```python
def max_position_size(stock, adv_20d):
    """Don't take more than 1% of average daily volume"""
    max_shares = int(0.01 * adv_20d)
    max_value = max_shares * current_price
    return min(max_value, 100000)  # cap at ₹1L
```

**C3. Correlation-Aware Portfolio Construction**

```python
import numpy as np

def diversified_selection(candidates, returns_history, max_picks=5):
    """Select picks that are maximally uncorrelated"""
    selected = [candidates[0]]  # start with best score

    for candidate in candidates[1:]:
        if len(selected) >= max_picks:
            break
        # Check correlation with all selected
        corrs = [
            np.corrcoef(
                returns_history[candidate['symbol']],
                returns_history[s['symbol']]
            )[0, 1]
            for s in selected
        ]
        if max(abs(c) for c in corrs) < 0.6:  # correlation threshold
            selected.append(candidate)

    return selected
```

### D. Risk Management Improvements

**D1. Dynamic Position Sizing (Kelly-Inspired)**

```python
def kelly_position_size(
    confidence, win_rate, avg_win, avg_loss, base_capital=100000
):
    """Half-Kelly position sizing"""
    b = avg_win / abs(avg_loss)  # win/loss ratio
    p = win_rate
    kelly_fraction = (b * p - (1 - p)) / b
    half_kelly = max(0, kelly_fraction / 2)  # conservative half-Kelly

    # Scale by confidence
    adjusted = half_kelly * confidence
    return base_capital * min(adjusted, 0.25)  # max 25% of capital
```

**D2. Daily Loss Limits**

```python
class RiskManager:
    def __init__(self, daily_loss_limit_pct=2.0, weekly_limit_pct=5.0):
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.daily_limit = daily_loss_limit_pct
        self.weekly_limit = weekly_limit_pct

    def can_trade(self):
        if self.daily_pnl <= -self.daily_limit:
            return False, "Daily loss limit hit"
        if self.weekly_pnl <= -self.weekly_limit:
            return False, "Weekly loss limit hit"
        return True, "OK"
```

**D3. Adaptive Stop-Losses**

A fixed 1% stop-loss is wrong for stocks with wildly different volatilities. RELIANCE (low vol) and ADANI ENTERPRISES (high vol) need different stops:

```python
def adaptive_stop_loss(entry_price, atr_14, direction, multiplier=1.5):
    """ATR-based stop loss"""
    stop_distance = atr_14 * multiplier
    if direction == "LONG":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance
```

### E. Model Improvements

**E1. Focus on One Model, Do It Well**

Five architectures is research-mode thinking. For production, pick the one that performs best on walk-forward OOS and invest all effort in making it robust. My recommendation: **LightGBM** because:

- Easier to debug and interpret
- Feature importance is transparent
- Faster iteration cycles
- More robust to noise than deep learning on tabular data
- Deep learning shines with raw sequences, but your 120-bar window is small

**E2. Target Engineering Overhaul**

Instead of predicting raw direction at a fixed horizon, predict **risk-adjusted opportunity**:

```python
def compute_target(future_bars, cost=0.002):
    """
    Target: Was there a tradeable opportunity after costs?
    """
    max_up = future_bars['high'].max() / future_bars['open'].iloc[0] - 1
    max_down = 1 - future_bars['low'].min() / future_bars['open'].iloc[0]

    # Net of costs
    long_opportunity = max_up - cost > 0
    short_opportunity = max_down - cost > 0

    return {
        'long_viable': long_opportunity,
        'short_viable': short_opportunity,
        'long_magnitude': max(0, max_up - cost),
        'short_magnitude': max(0, max_down - cost),
    }
```

**E3. Add a Rejection Class**

Most timesteps have no tradeable signal. Add an explicit "NO TRADE" class instead of forcing every prediction into LONG/SHORT:

```python
# 3-class target: LONG, SHORT, NO_TRADE
# Where NO_TRADE = abs(return) < threshold + costs
```

This lets the model learn when NOT to trade, which is often more valuable than learning when to trade.

**E4. Temporal Attention on What Matters**

Instead of feeding raw 120-bar windows, extract event-driven features:

```python
# Key intraday events that actually matter
features = {
    'last_30min_trend': slope of VWAP in final 30 min,
    'closing_auction_imbalance': buy vs sell volume in last 5 min,
    'overnight_gap': next day open vs previous close,
    'vix_term_structure': near vs far VIX,
    'sector_relative_strength': stock return vs sector ETF,
    'institutional_flow_proxy': large trade detection,
}
```

### F. Monitoring and Decay Detection

**F1. Live Performance Tracking Dashboard**

```python
class ModelMonitor:
    def __init__(self, lookback=50):
        self.predictions = []
        self.actuals = []

    def update(self, pred, actual):
        self.predictions.append(pred)
        self.actuals.append(actual)

    def rolling_accuracy(self):
        recent = list(zip(self.predictions, self.actuals))[-50:]
        correct = sum(1 for p, a in recent if p == a)
        return correct / len(recent)

    def is_degraded(self, threshold=0.52):
        """Alert if rolling accuracy drops below threshold"""
        return self.rolling_accuracy() < threshold
```

**F2. Automatic Retraining Trigger**

```python
if monitor.is_degraded():
    # Option 1: Fall back to simpler rules
    use_fallback_model()

    # Option 2: Retrain with recent data
    trigger_retraining(include_last_n_days=60)

    # Option 3: Reduce position sizes
    risk_manager.scale_down(factor=0.5)
```

---

## Priority-Ordered Action Items

| Priority  | Action                                                      | Expected Impact                      |
| --------- | ----------------------------------------------------------- | ------------------------------------ |
| 🔴 **P0** | Implement walk-forward validation and re-report all metrics | Gives you true expected performance  |
| 🔴 **P0** | Audit for look-ahead bias (features + universe)             | Could invalidate current results     |
| 🟠 **P1** | Feature selection — reduce from 69 to ~20 stable features   | Reduce overfitting, improve LGBM AUC |
| 🟠 **P1** | Add correlation-aware portfolio construction                | Avoid concentrated sector bets       |
| 🟠 **P1** | ATR-based adaptive stop-losses                              | Better risk per trade                |
| 🟡 **P2** | Calibrate probability outputs                               | Reliable confidence scores           |
| 🟡 **P2** | Dynamic position sizing (half-Kelly)                        | Better capital allocation            |
| 🟡 **P2** | Add NO_TRADE class to target                                | Model learns when to sit out         |
| 🟢 **P3** | Regime-conditioned model heads                              | Better regime-specific accuracy      |
| 🟢 **P3** | Live monitoring + staleness detection                       | Catch model decay early              |
| 🟢 **P3** | Point-in-time universe construction                         | Remove survivorship bias             |

---

## Questions I'd Want Answered

1. **What is the exact backtest period for the 236% return?** If it's 2015-2026, that's ~21% annualized which is more plausible but the Sharpe still doesn't add up. If it's 1-2 years, it's likely overfit.
2. **Why is there a 10:1 long-to-short ratio?** Is the model genuinely finding few short opportunities, or is there a bug in the short signal logic?
3. **What does the LGBM model's performance look like on the exact same backtest period as ResNLS?** The gap between 0.516 AUC and 56.7% win rate needs explaining.
4. **Are the 24 live recommendation sessions being tracked with P&L?** If so, what's the live Sharpe vs backtest Sharpe? This is the single most important number.
5. **How is the 120-bar window aligned for morning picks?** This is the most critical implementation detail that isn't clear from the documentation.

---

## Bottom Line

The system architecture is solid and the engineering quality is high. But the reported performance metrics are not credible without walk-forward validation on a point-in-time universe. The LightGBM metrics (0.516 AUC) are much more likely to reflect true out-of-sample performance, and at that level, **the system is not yet profitable after costs**.

The path to profitability runs through: ruthless feature selection, proper validation, execution realism, and learning when NOT to trade. The infrastructure is all there — the signal extraction needs hardening.
