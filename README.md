# intranet_optinet

Three trading systems share this repo:

- **NIFTY Futures Router** — long-only NIFTY futures engine, paper trading live (see `AGENTS.md`, `docs/paper_trading_runbook.md`)
- **IntradayNet** — daily LONG/SHORT equity picks on NSE Nifty 500 (V7 active, V8 trained)
- **Index Options** — NIFTY/BANKNIFTY options recommender (research/archived)

---

## NIFTY Futures Router — Paper Trading

Forward-walk validated (Nov 2024 → May 2026): +₹429k, Sharpe 2.17, PF 1.39, win 49.9%, max DD −₹74k.
Currently in 30-day paper trading. See `docs/paper_trading_runbook.md` for the full runbook.

### Daily cron (18:00 IST Mon-Fri)

```bash
0 18 * * 1-5  cd /path/to/repo && .venv/bin/python scripts/trading/daily_run.py >> logs/daily_run.log 2>&1
```

Refresh the Kite token each morning first:

```bash
.venv/bin/python scripts/data/kite_login.py
```

### Check status

```bash
.venv/bin/python scripts/trading/paper_status.py --include-bootstrap
```

---

## OptiNet F&O data lake

The data lake downloads NSE F&O bhavcopy archives, validates them, and writes per-symbol parquet partitions ready for training.

- Source: `https://archives.nseindia.com/`
- Auto-routes by date:
  - `< 2024-07-08` → legacy bhavcopy (`fo<DD><MMM><YYYY>bhav.csv.zip`)
  - `≥ 2024-07-08` → UDiFF format (`BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip`)
- Idempotent: existing files are skipped, so re-running resumes from where it left off.

Code: [`src/index_options/data_lake.py`](src/index_options/data_lake.py) · CLI: [`scripts/data/optinet_data_lake.py`](scripts/data/optinet_data_lake.py) · Tests: [`tests/test_optinet_data_lake.py`](tests/test_optinet_data_lake.py)

### One-time setup

```bash
uv pip install --python .venv/bin/python pytest pyarrow
```

### Backfill 2022 → today (training data)

```bash
# 1) Download raw zips (~1,100 trading days, ~1.5 GB, ~30–60 min wall clock)
mkdir -p logs
.venv/bin/python scripts/data/optinet_data_lake.py \
    --data-root data \
    download --start 2022-01-01 --end 2026-05-26 \
    | tee logs/optinet_data_lake_download_$(date +%Y%m%d_%H%M%S).log

# 2) Parse + validate + write parquet (fast — a few minutes)
.venv/bin/python scripts/data/optinet_data_lake.py \
    --data-root data \
    parse --start 2022-01-01 --end 2026-05-26 --overwrite \
    | tee logs/optinet_data_lake_parse_$(date +%Y%m%d_%H%M%S).log
```

### Daily incremental update

```bash
TODAY=$(date +%Y-%m-%d)
.venv/bin/python scripts/data/optinet_data_lake.py --data-root data download --start "$TODAY" --end "$TODAY"
.venv/bin/python scripts/data/optinet_data_lake.py --data-root data parse    --start "$TODAY" --end "$TODAY" --overwrite
```

### Output layout

```
data/
├── raw/
│   ├── legacy/<year>/fo<DD><MMM><YYYY>bhav.csv.zip
│   └── udiff/<year>/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip
├── parquet/
│   └── symbol={NIFTY,BANKNIFTY}/year=<YYYY>/options_<YYYYMMDD>.parquet
├── normalized/validation/
│   ├── <basename>_report.json
│   └── <basename>_bad_rows.csv
└── metadata/
```

### Tests

```bash
.venv/bin/python -m pytest tests/test_optinet_data_lake.py -v
```

---

## IntradayNet Equity

### V7 (active)

```bash
# Train
.venv/bin/python scripts/research/train_equity.py --universe nifty100 --target-pct 0.015

# Backtest
.venv/bin/python scripts/research/backtest_equity_v8.py --universe nifty100 --compare-v7
```

Models: `models/intraday_model_nifty100.pkl`, `models/intraday_model_nifty500.pkl`

### V8 (trained, not yet in production)

All 5 specialist models + curve embedding are trained in `models/v8/`. Next step: run the backtest to validate vs V7, then build the daily pipeline.

```bash
.venv/bin/python scripts/research/backtest_equity_v8.py --universe nifty100
```
