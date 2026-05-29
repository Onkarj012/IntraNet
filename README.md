# intranet_optinet

Two trading systems share this repo:

- **IntradayNet** — daily LONG/SHORT equity picks on NSE Nifty 500 (see `AGENTS.md`, `README_morning_picks.md`)
- **OptiNet** — index-options recommender on NIFTY / BANKNIFTY

This README covers the OptiNet **data lake** — how to fetch and prepare F&O bhavcopy data for training.

## OptiNet F&O data lake

The data lake downloads NSE F&O bhavcopy archives, validates them, and writes per-symbol parquet partitions ready for training.

- Source: `https://archives.nseindia.com/`
- Auto-routes by date:
  - `< 2024-07-08` → legacy bhavcopy (`fo<DD><MMM><YYYY>bhav.csv.zip`)
  - `≥ 2024-07-08` → UDiFF format (`BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip`)
- Idempotent: existing files are skipped, so re-running resumes from where it left off.

Code: [`src/optinet/data_lake.py`](src/optinet/data_lake.py) · CLI: [`scripts/optinet_data_lake.py`](scripts/optinet_data_lake.py) · Tests: [`tests/test_optinet_data_lake.py`](tests/test_optinet_data_lake.py)

### One-time setup

```bash
uv pip install --python .venv/bin/python pytest pyarrow
```

### Backfill 2022 → today (training data)

```bash
# 1) Download raw zips (~1,100 trading days, ~1.5 GB, ~30–60 min wall clock)
mkdir -p logs
.venv/bin/python scripts/optinet_data_lake.py \
    --data-root data \
    download --start 2022-01-01 --end 2026-05-26 \
    | tee logs/optinet_data_lake_download_$(date +%Y%m%d_%H%M%S).log

# 2) Parse + validate + write parquet (fast — a few minutes)
.venv/bin/python scripts/optinet_data_lake.py \
    --data-root data \
    parse --start 2022-01-01 --end 2026-05-26 --overwrite \
    | tee logs/optinet_data_lake_parse_$(date +%Y%m%d_%H%M%S).log
```

### Daily incremental update

```bash
# Append today's bhavcopy after NSE publishes it (~6 PM IST)
TODAY=$(date +%Y-%m-%d)
.venv/bin/python scripts/optinet_data_lake.py --data-root data download --start "$TODAY" --end "$TODAY"
.venv/bin/python scripts/optinet_data_lake.py --data-root data parse    --start "$TODAY" --end "$TODAY" --overwrite
```

### Output layout

```
data/
├── raw/
│   ├── legacy/<year>/fo<DD><MMM><YYYY>bhav.csv.zip          # immutable
│   └── udiff/<year>/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip
├── parquet/
│   └── symbol={NIFTY,BANKNIFTY}/year=<YYYY>/options_<YYYYMMDD>.parquet
├── normalized/validation/                                     # per-day audit
│   ├── <basename>_report.json
│   └── <basename>_bad_rows.csv
└── metadata/                                                  # expiry calendar, contracts
```

### Other CLI subcommands

```bash
# Show whether a date routes to legacy or UDiFF and where it lands
python scripts/optinet_data_lake.py route 2024-07-08

# Build expiry calendar (weekly vs monthly) from raw files
python scripts/optinet_data_lake.py expiry-calendar data/raw/udiff/2026/*.zip --output expiry_calendar_2026.csv

# Validate raw files and dump report + bad-rows CSV
python scripts/optinet_data_lake.py validate data/raw/udiff/2026/*.zip --name may2026

# Normalize index spot OHLC CSVs (NIFTY / BANKNIFTY) into the lake
python scripts/optinet_data_lake.py normalize-index data/indices/nifty_spot.csv --default-symbol NIFTY
```

### Tests

```bash
.venv/bin/python -m pytest tests/test_optinet_data_lake.py -v
```

## OptiNet v2.1 — Sentiment + Regime + Calibration (May 2026)

v2.1 adds three components on top of v2 to fix the BLOCKED readiness state:
news sentiment (yfinance + RSS, GDELT events), regime filter (VIX/gap/ATR),
and Platt-calibrated classifiers (replaces over-conservative isotonic).

### A/B results on 2026 YTD blind hold-out

| Config | Trades | Win % | Stop % | Sharpe | Net PnL |
|---|---|---|---|---|---|
| baseline (v2.1) | 22 | 23% | 73% | −6.8 | −23,588 |
| +sentiment | 22 | 23% | 73% | −6.8 | −23,588 |
| +gdelt | 23 | 26% | 70% | −5.3 | −21,717 |
| +regime_feat | 28 | 29% | 68% | −4.5 | −26,533 |
| +regime_filter | 6 | 83% | 17% | +6.6 | +7,882 |
| **full** (canonical) | **4** | **100%** | **0%** | **+22.7** | **+10,750** |

The regime hard filter is the dominant lift — it skips March-2026 crash days that
caused all of v2's stop-outs. GDELT contributes +0.06 AUC. Sentiment contribution
is currently flat because RSS feeds only return current articles (cache will
accumulate signal as the daily cron runs).

### A/B harness

```bash
.venv/bin/python scripts/optinet_ab_eval.py \
    --profile balanced \
    --train-end-year 2025 \
    --blind-year 2026 \
    --min-confidence 0.20
# → results/optinet/ab_history.csv (appended)
```

### Daily sentiment snapshot

Run before market open each day to accumulate the index sentiment cache:

```bash
.venv/bin/python -c "from optinet.sentiment import update_sentiment_cache; update_sentiment_cache()"
```

### GDELT incremental update

```bash
.venv/bin/python -c "
from datetime import date, timedelta
from optinet.gdelt import update_gdelt_cache
update_gdelt_cache(start=date.today() - timedelta(days=7))
"
```

## OptiNet v2 — Training & Recommendations

### Architecture

4-LGBM stack (long_clf, short_clf, up_magnitude_reg, down_magnitude_reg) with isotonic calibration, trained on daily-resolution F&O features (PCR, max-pain, IV-skew, OI walls, index TA) from the parquet lake. Signals are translated to option contracts via Black-Scholes delta-band selection.

### Train

```bash
# Train on full lake through end of last year
.venv/bin/python scripts/optinet_pipeline.py train \
    --profile balanced \
    --cutoff 2025-12-31 \
    --output models/optinet/optinet_balanced.pkl
```

### Evaluate (walk-forward + blind hold-out + readiness gate)

```bash
.venv/bin/python scripts/evaluate_optinet.py \
    --index data/indices/nifty_daily.csv data/indices/banknifty_daily.csv \
    --options data/parquet/symbol=NIFTY/year=2025/options_20250103.parquet \
    --profile balanced \
    --train-start 2022-01-01 --train-end 2025-12-31 \
    --blind-start 2026-01-01 --blind-end 2026-05-25 \
    --output-dir results/optinet/evaluation
```

### Generate daily recommendations

```bash
.venv/bin/python scripts/optinet_pipeline.py recommend \
    --model models/optinet/optinet_balanced.pkl \
    --profile balanced \
    --top-k 4 \
    --min-confidence 0.55
# → recommendations/optinet_picks_YYYYMMDD.json
```

### Daily update (download + parse + recommend in one shot)

```bash
.venv/bin/python scripts/optinet_pipeline.py daily-update \
    --model models/optinet/optinet_balanced.pkl \
    --profile balanced
```

### Key source files

| File | Purpose |
|---|---|
| `src/optinet/parquet_loader.py` | Bridge parquet lake → OptiNet data loaders |
| `src/optinet/models.py` | 4-LGBM stack + isotonic calibration + executable_edge |
| `src/optinet/features.py` | Index TA + F&O microstructure features |
| `src/optinet/translator.py` | BS-delta strike picker → `OptionTrade` |
| `src/optinet/backtester.py` | Daily-resolution backtest with target/stop exits |
| `src/optinet/evaluation.py` | Walk-forward + blind + readiness gates |
| `scripts/optinet_pipeline.py` | train / recommend / daily-update CLI |
| `scripts/optinet_data_lake.py` | Bhavcopy download / parse / validate CLI |
