# Agent Context

## Repo layout

Three systems share this repo:

| System | Package | Status |
|--------|---------|--------|
| **IntradayNet equity** | `src/equity/` | V7 active, V8 trained |
| **NIFTY futures router** | `src/engine/` + `src/broker/` | ✅ Paper trading live |
| **Index options** | `src/index_options/` | Research / archived |

---

## 1. NIFTY Futures Router (router_v0) — ACTIVE

Long-only NIFTY futures engine. Forward-walk validated Nov 2024 → May 2026:
+₹429k, Sharpe 2.17, PF 1.39, win 49.9%, max DD −₹74k. Currently in
**30-day paper trading** (started 2026-05-15).

### Key paths

| Path | Purpose |
|------|---------|
| `models/router_v0/futures/final_long.lgb` | Production model (locked — do not replace) |
| `results/router_v0/paper_trading_ledger.csv` | Append-only paper trade ledger |
| `results/router_v0/tier1_validation.json` | Tier-1 validation record (all pass) |
| `data/nifty_intraday/NIFTY 50_minute.csv` | NIFTY index minute bars (updated daily via Kite) |
| `data/nifty_intraday/INDIA VIX_day.csv` | Daily VIX (updated daily via Kite) |

### Source

- `src/engine/features.py` — feature builder (`FUTURES_FEATURES`, `compute_features`, `add_regime`)
- `src/engine/risk.py` — `RiskLimits`, `BrokerState`, `evaluate_ticket_risk`
- `src/engine/freshness.py` — data freshness checks
- `src/engine/data_quality.py` — session quality checks
- `src/engine/orders.py` — order scaffold (dry-run by default, `NotImplementedError` on live)
- `src/broker/kite_client.py` — Zerodha Kite Connect client (historical data, WebSocket)

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/trading/daily_run.py` | **Single cron entrypoint** — EOD cache → paper ops |
| `scripts/trading/paper_trade.py` | One-day Variant A replay + ledger append |
| `scripts/trading/paper_trade_variant_c.py` | One-day Variant C replay |
| `scripts/trading/paper_ops.py` | Orchestrates A + C + status |
| `scripts/trading/paper_status.py` | Dashboard + halt checks (hard halt only writes kill-switch) |
| `scripts/trading/live_signal.py` | Real-time Kite WebSocket signal cards (manual, prints only) |
| `scripts/trading/execute_order.py` | Live execution scaffold (triple-key gate, dry-run default) |
| `scripts/data/kite_login.py` | Kite browser login — run each morning to refresh token |
| `scripts/data/kite_eod_cache.py` | Fetch + append today's NIFTY bars + VIX from Kite |
| `scripts/data/kite_backfill.py` | Backfill historical minute bars |
| `scripts/research/forward_walk.py` | Full Phase 0-3 validation pipeline |
| `scripts/research/tier1_validate.py` | Tier-1 pre-deployment validation |
| `scripts/research/backtest_futures.py` | Backtest with Phase-1 risk filters |
| `scripts/research/variant_grid.py` | Variant C grid search + Phase-3 threshold sweep |
| `scripts/research/train_futures_v2.py` | Phase-2 model retrain research (v2 worse than v1 — do not promote) |

### Cron (18:00 IST Mon-Fri)

```
0 18 * * 1-5  cd /path/to/repo && .venv/bin/python scripts/trading/daily_run.py >> logs/daily_run.log 2>&1
```

Requires `KITE_ACCESS_TOKEN` in `.env` (refresh manually each morning via `kite_login.py`).

### Halt logic

- **HARD halt** (≤ −₹150k cumulative DD) → writes `results/router_v0/PAPER_TRADING_HALTED`, blocks further runs
- **SOFT halts** (30d Sharpe < 0.5, 5d PnL ≤ −₹50k, 7+ consecutive losing days) → WARNING only, run continues

### Current paper state (as of 2026-05-29)

- 30 live trades (Variant A), net −₹19,771, win 33%, 30d Sharpe −3.29
- Soft halt active (Sharpe alert) — measurement continues, no kill-switch
- Variant C: 25 trades, net +₹4,530

---

## 2. IntradayNet Equity — NEXT PHASE

Daily LONG/SHORT equity picks on NSE Nifty 500.

### V7 (active, not being improved)

4 LightGBM models (LONG clf, SHORT clf, UP/DOWN magnitude regressors) trained on ~91 daily features from minute bars. Models exist and work; AUC ~0.62.

| Path | Purpose |
|------|---------|
| `models/intraday_model_nifty500.pkl` | Trained V7 model (Nifty 500, AUC 0.625) |
| `models/intraday_model_nifty100.pkl` | Trained V7 model (Nifty 100, AUC 0.610) |
| `src/equity/v7.py` | Target labeling, confidence, scoring |
| `src/equity/v7_modes.py` | Post-open gap-aware adjustment, regime classification |
| `src/equity/open_safe_daily_features.py` | Daily feature builder from minute data |
| `src/equity/features/` | per_bar_features.py, session_features.py, sentiment_features.py, market_features.py |
| `src/equity/recommendation.py` | Profile-based recommendation engine |
| `src/equity/calibrator.py` | Platt/isotonic calibration |
| `src/equity/model_bundle.py` | LightGBM model manifest + bundle management |
| `src/equity/equity_paper.py` | Paper trading ledger creation/reconciliation |
| `src/equity/universe.py` | Stock universe definitions (Nifty 50/100/200/500) |
| `src/equity/costs.py` | NSE transaction cost calculator |
| `src/equity/run_logging.py` | Tee-duplicated stdout/stderr to log files |

### V8 (library built + trained, not yet in production)

Complete redesign: curve embeddings (masked autoencoder) + barrier targets + 5 specialist LightGBM models + regime-weighted meta-ensemble + diversified portfolio construction.

**All 5 specialist models are trained** (`models/v8/*.pkl`). Curve embedding trained (`models/v8/best_model.pt` + `embeddings.npz`). Not yet wired into a daily production pipeline.

| Path | Purpose |
|------|---------|
| `models/v8/` | Trained models: momentum, reversal, breakout, sentiment, macro signals + curve embedding |
| `src/equity/v8/config.py` | Single-source config dataclasses |
| `src/equity/v8/barriers.py` | Path-dependent barrier target computation |
| `src/equity/v8/curve_embedding.py` | Masked autoencoder (CurveMaskedEncoder) |
| `src/equity/v8/data_pipeline.py` | Minute data loading, session extraction |
| `src/equity/v8/signal_models.py` | 5 specialist LightGBM classifiers + MetaEnsemble |
| `src/equity/v8/regime_detector.py` | K-means regime clustering (5 regimes) |
| `src/equity/v8/portfolio.py` | Greedy diversified portfolio construction |
| `src/equity/v8/walk_forward.py` | Walk-forward backtest scaffold |
| `src/equity/v8/universe_tiers.py` | Tier classification (T1/T2/T3 by data history) |
| `src/equity/v8/per_stock_sentiment.py` | Per-stock sentiment features via yfinance |
| `scripts/research/train_equity_v8.py` | V8 training script |
| `scripts/research/backtest_equity_v8.py` | V8 backtest + V7 comparison |

### Next steps for equity

1. Run `scripts/research/backtest_equity_v8.py --universe nifty100` to validate V8 vs V7
2. Build `scripts/trading/equity_daily.py` — daily premarket picks pipeline
3. Wire into paper trading via `src/equity/equity_paper.py`

### Data

- `data/nifty500/{SYMBOL}_minute.csv` — 531 stocks, 1-min OHLCV bars (gitignored)
- `data/sentiment/` — news sentiment CSVs
- `data/prices/` — 42 daily EOD reference CSVs
- `market_data_cache/*.csv` — 25 macro/sector CSVs (VIX, crude, gold, indices, sectors)
- `configs/intraday_config.yaml` — horizons, model architecture, data paths, training params

---

## 3. Index Options (src/index_options/) — Research / Archived

Lineage: v2.1 (LGBM stack + regime filter) → v3 (resnls) → v4 (chain) → v5 (simulator + futures). Most work archived under `archive/`. The `src/index_options/` package contains the full lineage but is not actively run.

Key research scripts: `scripts/research/index_options_pipeline.py`, `scripts/research/index_options_ab_eval.py`.

---

## Environment

- Python 3.12, venv at `.venv/`
- Key deps: `lightgbm>=4.0`, `kiteconnect==5.0.1`, `pyotp`, `torch>=2.0`, `pandas>=2.0`, `scikit-learn>=1.3`
- Credentials: `.env` (KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_PASSWORD, KITE_ACCESS_TOKEN)
- Entry point: `nifty-trader` CLI → `src/equity/cli.py`
