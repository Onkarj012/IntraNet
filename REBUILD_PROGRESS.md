# Rebuild Progress Report - Lean Signal-First Architecture

## Current Status

Branch: `rebuild/lean-signal-first`
Commits: 2

## What Has Been Built

### 1. Stock Universe Definitions (`src/intradaynet/universe.py`)
- NIFTY50, NIFTY100, NIFTY200 stock lists
- Helper functions: `get_universe()`, `is_in_universe()`

### 2. Lean Features (`src/intradaynet/lean_features.py`)
Reduced from 625 to **36 essential features** in 5 categories:

| Category | Features | Purpose |
|----------|----------|---------|
| Price Action | 8 | End-of-day microstructure for gap prediction |
| Market Context | 10 | VIX, Nifty, crude, USD/INR, global cues |
| Sentiment | 6 | News sentiment and momentum |
| Gap-Specific | 4 | Historical gap patterns |
| Microstructure | 8 | Volume, OBV, support levels |

### 3. Gap Targets (`src/intradaynet/gap_targets.py`)
7 target types for morning picks:
- `gap_direction`: -1, 0, 1 classification
- `gap_magnitude`: Signed gap size (regression)
- `gaps_up`/`gaps_down`: Binary direction
- `gap_fills`: Will gap fill within 60 min?
- `profitable_long`/`profitable_short`: Cost-adjusted profitability

### 4. Signal Audit (`scripts/signal_audit.py`)
Full walk-forward ICIR analysis:
- Tests each feature against each target
- Walk-forward folds (default: 8)
- Computes ICIR = mean(IC) / std(IC)
- Identifies features with ICIR >= 0.2 threshold

### 5. Quick Signal Test (`scripts/quick_signal_test.py`)
Fast preliminary test with 5 simple features:
- `volatility`: 0.065 correlation with gaps ← **shows signal!**
- Other features: near-zero correlation

### 6. Data Loader (`src/intradaynet/data_loader.py`)
Memory-efficient loading for 16GB Mac:
- Stock-by-stock streaming (not all at once)
- Date filtering (2021+ default)
- Universe filtering (NIFTY50/100/200)
- Resampling from minute to daily

## Quick Test Results

Ran on 10 NIFTY50 stocks:

| Feature | Correlation with Next-Day Gap |
|---------|------------------------------|
| volatility | **0.0652** ← Weak but real signal |
| volume_trend | 0.0147 |
| prev_day_return | -0.0080 |
| momentum_5d | -0.0101 |
| rsi_proxy | -0.0158 |

**Interpretation:** Volatility predicts gap size, which makes sense. Need to run full ICIR analysis to find more signals.

## Next Steps

### Phase 5: Full Signal Audit
Run the full signal audit on NIFTY100:
```bash
python scripts/signal_audit.py --universe nifty100 --start-date 2021-01-01
```

This will take 30-60 minutes but will tell us:
1. Which of the 36 features have predictive power (ICIR >= 0.2)
2. Which target types are most predictable
3. Whether gap prediction is viable at all

### Phase 6: Walk-Forward Framework
If signal exists, build strict temporal validation.

### Phase 7: Minimal Model
Train aggressively regularized LightGBM with selected features.

## Key Files Created

```
src/intradaynet/
├── universe.py          # Stock universe definitions
├── lean_features.py     # 36 essential features
├── gap_targets.py       # Gap prediction targets
└── data_loader.py       # Memory-efficient data loading

scripts/
├── signal_audit.py      # Full ICIR analysis
└── quick_signal_test.py # Fast preliminary test
```

## How to Run

```bash
# Quick test (1-2 minutes)
python scripts/quick_signal_test.py --universe nifty50 --max-stocks 10

# Full signal audit (30-60 minutes)
python scripts/signal_audit.py --universe nifty100 --start-date 2021-01-01
```

## Current Blockers

None - system is ready for signal audit.

## Decision Needed

Should I:
1. Run the full signal audit now (will take 30-60 min)
2. Build the walk-forward framework first
3. Create a minimal model with just volatility feature as proof of concept

Recommendation: Run signal audit first to know if there's any signal worth modeling.
