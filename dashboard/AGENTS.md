# OptiNet Dashboard

Next.js control room for intraday equity picks, paper ledger, walk-forward validation, and ops monitoring.

## Run locally

```bash
cd dashboard
npm install
npm run dev
```

The dashboard reads artifacts from the parent repo (`results/`, `models/v8_intraday/`) when run locally. Set `REPO_ROOT` if the repo is not one level up from `dashboard/`.

## Hosted mode

When `CONVEX_HTTP_URL` and `DASHBOARD_PUSH_SECRET` are set, artifacts pushed by `scripts/trading/push_dashboard.py` are read over HTTP instead of local files.

## Pages

| Route | Purpose |
|-------|---------|
| `/` | Morning brief — equity picks + NIFTY futures signal |
| `/ledger` | Intraday paper trade ledger + equity curve |
| `/performance` | Walk-forward validation + regime breakdown |
| `/ops` | Pipeline status, model health, drift monitor, kill switches |
