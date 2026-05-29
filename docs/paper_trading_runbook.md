# Paper Trading Runbook — NIFTY long-only futures engine (Variant A)

**Status**: live as of 2026-05-28. Cleared by Tier-1 validation
(`results/router_v0/tier1_validation.json`).

## What this is

A daily paper-trading harness for the OptiNet Router's NIFTY long-only
futures engine. The engine itself was forward-walk-validated on
Nov 2024 → May 2026 (19 months, true out-of-sample): +₹429k, Sharpe 2.17,
PF 1.39, win 49.9 %, max DD −₹74k.

Paper trading is the next gate before any real capital. We replay each
day's NIFTY index minute bars through the locked model + Variant-A
filters, append simulated fills to a persistent ledger, and watch
performance drift over time.

## Architecture

```
data/nifty_intraday/NIFTY 50_minute.csv      (updated daily, EOD)
        ↓
scripts/paper_trade_daily.py                 ← run after market close
        ↓
results/router_v0/paper_trading_ledger.csv   (append-only)
        ↓
scripts/paper_trade_status.py                ← run anytime
```

Locked components (do NOT modify without restarting validation):
- Model: `models/router_v0/futures/final_long.lgb`
- Feature builder: `src/optinet_router/futures_features.py`
- Hard filters: `TARGET_PCT=0.0040, STOP_PCT=0.0030, HORIZON=60,
  COSTS_INR=105, SIGNAL_PCT=0.85, HIGH_CONF_PCT=0.95,
  SKIP_REGIMES={"compression"}, SKIP 11:00-11:59, ENTRY_MIN=30,
  HARD_CUTOFF=14:55, MAX_TRADES=3/day, DAILY_HALT=-₹15,000`
- Variant: A (no regime guards). Variant B is a runtime flag for later.

## Daily workflow

### 1) Update source data

Make sure `data/nifty_intraday/NIFTY 50_minute.csv` has today's bars.
That update is upstream of this engine; ensure your data pipeline writes
to that file on EOD.

```bash
# Sanity check: is today's data there?
tail -1 "data/nifty_intraday/NIFTY 50_minute.csv"
```

### 2) Run paper trader

```bash
# Most common: replay today / most recent day in the data file
.venv/bin/python scripts/paper_trade_daily.py --auto

# Or explicit
.venv/bin/python scripts/paper_trade_daily.py --date 2026-05-28

# Or re-run a date (overwrites prior rows for that date)
.venv/bin/python scripts/paper_trade_daily.py --date 2026-05-28 --force
```

The runner is **idempotent**: re-running a date that's already in the
ledger is a no-op unless `--force` is passed. This makes cron-driven
operation safe — duplicate cron runs don't double-count.

### 3) Check status

```bash
# Just live paper trades (start showing real signal after 30+ days)
.venv/bin/python scripts/paper_trade_status.py

# Include the bootstrap forward-walk rows (full historical context)
.venv/bin/python scripts/paper_trade_status.py --include-bootstrap

# With per-month breakdown
.venv/bin/python scripts/paper_trade_status.py --include-bootstrap --by-month

# As part of a cron job — write the kill-switch file when halts trigger
.venv/bin/python scripts/paper_trade_status.py --write-halt
```

The kill-switch file is `results/router_v0/PAPER_TRADING_HALTED`. While it
exists, `paper_trade_daily.py` refuses to run. Delete the file (or pass
`--ignore-kill-switch`) to resume after investigation.

## Cron setup

Run on every NSE trading day at 18:00 IST (after the bhavcopy is published
and source data is updated):

```cron
# m  h    dom  mon  dow   command
  0  18    *    *   1-5   cd /Users/onkarj012/Projects/market/intranet_optinet && \
                          .venv/bin/python scripts/paper_trade_daily.py --auto \
                          >> logs/paper_trade_daily.log 2>&1 && \
                          .venv/bin/python scripts/paper_trade_status.py --write-halt \
                          >> logs/paper_trade_status.log 2>&1
```

(NSE holidays will produce no bars for that date, so paper_trade_daily
will simply log "no bars" and exit clean.)

## Halt conditions

Halts are checked by `paper_trade_status.py`. With `--write-halt`, a
kill-switch file is written; the daily runner will refuse to run while it
exists.

| Condition | Severity | Threshold |
|---|---|---|
| Cumulative drawdown | **HARD HALT** | ≤ −₹150,000 |
| Trailing 30-day Sharpe | SOFT HALT | < 0 (with ≥ 30 trades) |
| Trailing 5-day net PnL | SOFT HALT | ≤ −₹50,000 |
| Consecutive losing days | SOFT HALT | ≥ 7 |

**HARD HALT** = stop trading immediately, do a full review before
resuming. **SOFT HALT** = stop trading, investigate, resume only if
the cause is understood (e.g. known macro event vs model degradation).

Tune thresholds in `scripts/paper_trade_status.py`. Defaults are sized
against the forward-walk drawdown of −₹74k (so −₹150k is roughly 2× that).

## What "good" looks like

After 30 days of live paper trading you should see roughly:
- 50-65 trades, 13-20 trading days
- Win rate ≈ 50 % ± 5pp
- Trailing 30-day Sharpe ≥ 1.0
- Cumulative PnL trending up, drawdowns < −₹50k

Compare to the forward-walk monthly numbers at
`docs/forward_walk_verdict.md` — the per-month range was −₹66k to +₹116k,
with 13 of 19 months positive.

## What "bad" looks like and what to do

| Symptom | Likely cause | Action |
|---|---|---|
| 30-day Sharpe trending toward 0 | Regime change vs training | Investigate, don't tweak yet. If 60-day Sharpe also < 0.5, escalate. |
| Many `STOP_FLOOR` exits (−₹3k) | Increased intraday volatility | Compare INDIA VIX vs training distribution. Don't widen the stop. |
| Win rate < 40 % over 30 days | Model is missing | Check Tier-1C: regenerate score-quintile distribution on recent trades. |
| All trades cluster in 09:45-11:00 | 85th-pct threshold rarely fires elsewhere | Expected. Variant A's threshold is per-day, ties cluster early when realized vol is highest. |
| Costs eating most of the edge | Slippage worse than ₹105/lot | Re-run Tier 1B at higher cost levels and decide. |

When in doubt, **stop and investigate** rather than tweak parameters.
Live paper trading is a measurement, not an optimization loop.

## Forward roadmap (deferred — see `docs/forward_walk_verdict.md`)

After 1 calendar month of clean paper trading data, evaluate Tier 2 and
Tier 3 follow-ups:

- **Tier 2** — selectivity tuning (90th, 95th percentile thresholds),
  Variant A vs Variant B comparison on live data.
- **Tier 3** — investigate Feb 2025 / Feb 2026 failure modes; retrain
  with proper boosting (>1 tree).

Do NOT retrain the locked model on 2025 or 2026 data. That destroys the
forward-walk validity that earned us the right to paper trade.

## Variant C — parallel macro-regime variant

In addition to Variant A (current live), `scripts/paper_trade_variant_c.py`
runs Variant C on the same days. Variant C adds three extra filters on top
of Variant A's hard filters:

| Filter | Value | Source |
|---|---|---|
| 5-day NIFTY return floor | > −1.5 % | grid search |
| VIX state guard | skip when VIX rising AND > 75th-pct of 60d | grid search |
| Intraday daily PnL halt | −₹6,000 (vs A's −₹15,000) | grid search |
| Score percentile | 0.95 (top 5 %, vs A's 0.85) | grid search |

In-sample on the 19-month forward walk:
- 688 trades vs A's 1,128 (39 % fewer)
- Win rate 53.5 % vs A's 49.9 %
- Sharpe +3.92 vs A's +2.17
- Max DD −₹41k vs A's −₹74k
- 2 negative months vs A's 4

Daily run:
```bash
.venv/bin/python scripts/paper_trade_variant_c.py --auto
```

The runner writes to the same `paper_trading_ledger.csv` but with
`source='paper_c'`. Both variants are tracked side-by-side in the status
dashboard. **Do not promote Variant C to live based on backtest alone** —
let it run for 1 calendar month, then compare to Variant A on live data.

Variant C config is auto-loaded from
`results/router_v0/variant_c_config.json` so it can be regenerated by
re-running `scripts/variant_c_grid.py`.

## Forward-walk tracking

The status dashboard now reports current 30-day metrics against the
forward-walk reference distribution (computed from the 1,128 bootstrap
rows). Each metric is classified into one of:

- 🔻 **below p10**  — worse than 90 % of historical 30-day windows. Investigate.
- 🟡 **below p25**  — worse than 75 %. Watch.
- 🟢 **in IQR**     — within central 50 %. Tracking.
- 🟢 **above p75**  — better than 75 %. Tracking favorably.
- ✨ **above p90**  — better than 90 %. Possibly transient outlier.

Reference distribution at deployment (will not change):
- 30-day Sharpe: median +2.06, IQR [+0.32, +3.85]
- 30-day PnL:    median ₹+18,255, IQR [₹+3,199, ₹+38,317]
- 30-day Win %:  median 50.0 %, IQR [43.3 %, 56.7 %]

If 30-day Sharpe falls below p10 for 3 consecutive days, escalate.

## Live execution scaffolding (DRY-RUN BY DEFAULT)

`scripts/live_execute.py` reads ledger rows and emits order tickets to
`results/router_v0/order_tickets.jsonl`. By default it logs only — no
real orders are placed.

```bash
# Default: dry-run, most recent paper-trading day, Variant A only
.venv/bin/python scripts/live_execute.py --auto

# Specific date, Variant C only
.venv/bin/python scripts/live_execute.py --date 2026-04-15 --variants C

# Both variants
.venv/bin/python scripts/live_execute.py --date 2026-04-15 --variants A,C
```

### Triple-key live-execution gate

Real broker placement (when wired) requires ALL of:

1. CLI flag: `--live`
2. Env var: `OPTINET_LIVE=1`
3. Confirm token: `--confirm-token /path/to/token.txt` whose content
   matches `results/router_v0/LIVE_TOKEN`
4. Kill-switch file `results/router_v0/PAPER_TRADING_HALTED` must NOT exist

If ANY gate fails, the script silently falls back to dry-run.

### Final safety layer

Even when the gate clears, `UpstoxOrderClient.place_order()` raises
`NotImplementedError`. Real execution requires an operator to manually
edit `src/optinet/v5_runtime/orders.py` to wire the Upstox SDK. This is
intentional — the Variant A model has not yet earned the right to fire
real orders, and the codebase enforces that on every run.

### When you're ready to wire live (after 1+ month of clean paper data)

1. `pip install upstox-python-sdk` (or your broker's SDK)
2. Set Upstox env vars: `UPSTOX_API_KEY`, `UPSTOX_API_SECRET`,
   `UPSTOX_ACCESS_TOKEN`
3. Implement `place_order` / `get_status` / `cancel` in
   `src/optinet/v5_runtime/orders.py:UpstoxOrderClient` against the SDK
4. Generate a one-time token:
   ```bash
   openssl rand -hex 16 > results/router_v0/LIVE_TOKEN
   chmod 400 results/router_v0/LIVE_TOKEN
   ```
5. Copy the token to a separate file the operator passes via
   `--confirm-token`. Treat the token like an SSH key — don't commit it.
6. Test on TINY size only (1 lot, ideally NIFTY mini if available)
7. Reconcile every fill against `paper_trading_ledger.csv` for the
   first week.
8. If any reconciliation gap > ±10 % of expected fill: stop, investigate.

## File reference

| File | Purpose |
|---|---|
| `scripts/paper_trade_daily.py` | One-day replay + ledger append |
| `scripts/paper_trade_status.py` | Read-only metrics + halt checks |
| `scripts/paper_trade_bootstrap.py` | One-shot ledger seeding |
| `scripts/forward_walk_2024_2026.py` | The full Phase 0-3 validation pipeline |
| `scripts/tier1_validation.py` | Pre-deployment validation (already passed) |
| `results/router_v0/paper_trading_ledger.csv` | Append-only ledger |
| `results/router_v0/PAPER_TRADING_HALTED` | Kill-switch file (when present) |
| `results/router_v0/tier1_validation.json` | Tier-1 validation record |
| `docs/forward_walk_verdict.md` | Why we're paper trading |

## Ledger schema

```
paper_trade_id    short uuid
run_timestamp     when the row was written
trade_date        YYYY-MM-DD
datetime_entry    ISO8601 entry minute
datetime_exit     ISO8601 exit minute
side              "LONG"
entry_px          fill price at entry
exit_px           fill price at exit
target_px         entry_px * 1.0040
stop_px           entry_px * 0.9970
size_mult         1.0 or 1.5
lot               50
gross_pnl_inr     before costs
costs_inr         105 * size_mult
net_pnl_inr       gross - costs (clipped at -3000 floor)
exit_reason       TARGET / STOP / TIME / STOP_FLOOR / NO_BARS
regime            range / expansion / trend_up / trend_dn (compression skipped)
long_score        model output
reason_codes      pipe-separated descriptive tags
model_version     "futures_long_v1"
source            "paper" | "forward_walk"
```

## Recovery procedures

### Ledger corrupted or accidentally edited

The CSV is append-only. If a manual edit broke schema, restore from git
(it's version controlled) or recover from per-day re-runs:

```bash
# 1) move corrupt ledger aside
mv results/router_v0/paper_trading_ledger.csv results/router_v0/paper_trading_ledger.corrupt.csv

# 2) re-bootstrap historical
.venv/bin/python scripts/paper_trade_bootstrap.py

# 3) re-replay every paper-trading date
for d in 2026-05-15 2026-05-16 ...; do
  .venv/bin/python scripts/paper_trade_daily.py --date "$d" --force
done
```

### Model file missing

`models/router_v0/futures/final_long.lgb` is the locked production model.
If lost, regenerate from sources by re-running `scripts/train_futures_engine.py`
(this will produce a new model — not byte-identical, so re-run Tier 1
and re-bootstrap before resuming).

### Source data file missing or stale

`data/nifty_intraday/NIFTY 50_minute.csv` is the upstream input. If the
file is missing or its last date is older than the date you're running,
fix the upstream pipeline before running the paper trader.

## Contact / escalation

- Halt triggered → stop, read the halt reason in
  `results/router_v0/PAPER_TRADING_HALTED`, investigate, decide.
- Unclear behavior → re-run `scripts/tier1_validation.py` against current
  data; if any Tier-1 check now fails, the engine has degraded and
  needs investigation.
