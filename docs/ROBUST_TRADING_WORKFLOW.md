# Robust Trading Workflow

This project is paper-first. A model is not tradeable just because it trains or has a profitable historical backtest.

## Required Workflow

1. Download or update market data.
2. Build a point-in-time dataset.
3. Train the model on a declared training window.
4. Evaluate walk-forward and blind periods.
5. Review readiness artifacts.
6. Generate paper-trading ledger rows only if readiness is not `BLOCKED`.
7. Reconcile paper trades against the next available market data.

## OptiNet Commands

```bash
uv run intradaynet optinet-evaluate \
  --index optinet_data/index/index_spot_daily.csv \
  --options optinet_data/options/options_eod_2021.csv optinet_data/options/options_eod_2022.csv optinet_data/options/options_eod_2023.csv optinet_data/options/options_eod_2024.csv optinet_data/options/options_eod_2025.csv optinet_data/options/options_eod_2026.csv \
  --profile balanced \
  --train-start 2021-01-01 \
  --train-end 2025-12-31 \
  --blind-start 2026-01-01 \
  --blind-end 2026-04-30 \
  --output-dir results/optinet/evaluation_2021_2026
```

```bash
uv run intradaynet paper-ledger \
  --system optinet \
  --model results/optinet/evaluation_2021_2026/optinet_balanced.pkl \
  --index optinet_data/index/index_spot_daily.csv \
  --options optinet_data/options/options_eod_2026.csv \
  --readiness results/optinet/evaluation_2021_2026/readiness.json \
  --output outputs/paper/optinet_paper_ledger.csv
```

```bash
uv run intradaynet reconcile-paper \
  --ledger outputs/paper/optinet_paper_ledger.csv \
  --index optinet_data/index/index_spot_daily.csv \
  --options optinet_data/options/options_eod_2026.csv \
  --output outputs/paper/optinet_paper_ledger_reconciled.csv
```

## Readiness Rules

Default gates block models when:

- blind trade count is below `50` for options
- blind net P&L is not positive
- blind Sharpe is not positive
- blind stop rate exceeds `60%`
- confidence buckets are inverted

Live broker execution is intentionally unsupported. The next milestone is one full paper-trading month with passing readiness and reconciliation metrics.
