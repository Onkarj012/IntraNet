# Two-Day Trading System MVP

This system is paper-first. It can produce governed equity paper trades and keeps OptiNet options blocked until blind tests pass.

## Daily Equity Loop

1. Generate equity recommendations.
2. Convert recommendations into a paper ledger.
3. Reconcile the ledger after market data is available.
4. Review `system-status` before trusting any signal.

```bash
uv run intradaynet picks \
  --model results/models/models/intraday_model_nifty500.pkl \
  --universe nifty500 \
  --data-dir nifty500 \
  --risk-profile balanced \
  --per-side 2 \
  --save-json outputs/paper/equity_recommendations_latest.json \
  --save-csv outputs/paper/equity_recommendations_latest.csv
```

```bash
uv run intradaynet equity-paper-ledger \
  --recommendations outputs/paper/equity_recommendations_latest.json \
  --capital 100000 \
  --risk-per-trade-pct 0.005 \
  --max-position-pct 0.20 \
  --output outputs/paper/equity_paper_ledger.csv
```

```bash
uv run intradaynet reconcile-equity-paper \
  --ledger outputs/paper/equity_paper_ledger.csv \
  --data-dir nifty500 \
  --output outputs/paper/equity_paper_ledger_reconciled.csv
```

```bash
uv run intradaynet system-status \
  --output outputs/system/trading_system_status.json
```

## Options Loop

OptiNet is currently blocked by blind 2026 results. Keep it in research/paper diagnostics only.

```bash
uv run intradaynet optinet-evaluate \
  --index optinet_data/index/index_spot_daily.csv \
  --options optinet_data/options/options_eod_2021.csv optinet_data/options/options_eod_2022.csv optinet_data/options/options_eod_2023.csv optinet_data/options/options_eod_2024.csv optinet_data/options/options_eod_2025.csv optinet_data/options/options_eod_2026.csv \
  --profile balanced \
  --train-start 2021-01-01 \
  --train-end 2025-12-31 \
  --blind-start 2026-01-01 \
  --blind-end 2026-04-30 \
  --output-dir results/optinet/robust_eval_2021_2026
```

## Promotion Rules

- Equity can run paper ledgers when recommendation readiness is `PAPER_ONLY`, `SMALL_LIVE`, or `READY`.
- Options stay blocked while `results/optinet/robust_eval_2021_2026/readiness.json` is `BLOCKED`.
- Live broker execution is intentionally unsupported.
- Default moderate risk: `0.5%` risk per trade, `1.5%` max daily loss, `3.0%` max weekly loss, stop after two stopped trades in one day.
