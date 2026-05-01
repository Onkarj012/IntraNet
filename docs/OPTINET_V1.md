# OptiNet v1

OptiNet is a parallel index-options recommendation system for NIFTY and BANKNIFTY. It reuses the project's macro and sentiment concepts, but keeps options data, labels, strategy translation, and backtesting separate from the equity intraday stack.

## Modules

| Area | File |
|---|---|
| Config and profiles | `src/optinet/config.py` |
| Data loaders | `src/optinet/data.py` |
| Index/F&O/macro feature builders | `src/optinet/features.py` |
| Profile labels | `src/optinet/labels.py` |
| LightGBM model stack | `src/optinet/models.py` |
| Direction-to-contract translator | `src/optinet/translator.py` |
| Daily options backtester | `src/optinet/backtester.py` |
| Readiness checks | `src/optinet/readiness.py` |
| End-to-end orchestration | `src/optinet/recommender.py` |

## Risk Profiles

| Profile | Index Target | Index Stop | Contract Bias |
|---|---:|---:|---|
| Conservative | 0.5% | 0.3% | ATM/ITM, monthly preference, no expiry day |
| Balanced | 1.0% | 0.6% | ATM/slight OTM, nearest weekly |
| Aggressive | 1.5% | 1.0% | OTM, nearest expiry |

## Data Contract

Index files need date, index/symbol, open, high, low, close, and optionally volume. Option files need date, index/symbol/underlying, expiry, strike, option type, open, high, low, close, and optionally volume, open interest, and change in OI. NSE-style names such as `TIMESTAMP`, `SYMBOL`, `EXPIRY_DT`, `STRIKE_PR`, `OPTION_TYP`, `OPEN_INT`, and `CHG_IN_OI` are normalized automatically.

## Commands

Build a dataset:

```bash
uv run intradaynet optinet-dataset --index data/index.csv --options data/options.csv --output cache/optinet/training_dataset.parquet
```

Train:

```bash
uv run intradaynet optinet-train --dataset cache/optinet/training_dataset.parquet --profile balanced --output results/models/optinet/optinet_balanced.pkl
```

Recommend:

```bash
uv run intradaynet optinet-picks --model results/models/optinet/optinet_balanced.pkl --index data/index.csv --options data/options.csv --profile balanced
```

Backtest:

```bash
uv run intradaynet optinet-backtest --model results/models/optinet/optinet_balanced.pkl --index data/index.csv --options data/options.csv --profile balanced
```

## Backtest Semantics

Signals are generated from day T features. The simulator enters the translated option at day T+1 open, checks the T+1 option high/low against premium target/stop, and exits at close if neither level is hit. P&L is premium difference times lot size minus a small cost estimate.

## Current Limits

OptiNet v1 uses EOD option bars and Black-Scholes approximations for IV/delta. It is suitable for daily-resolution research and recommendations, not true intraday execution simulation. Minute-level option data and live Greeks should be added before any broker automation.
