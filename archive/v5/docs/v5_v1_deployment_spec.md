# OptiNet V5 v1 — Deployment Spec (Paper Trading)

**Status:** **FROZEN — 2026-05-27**  
**Scope:** paper trading only. No real-money execution.

**Locked configuration:** gate + always `SHORT_STRADDLE_EOD`, dte=2 or dte=3 only,
V4-B vol kill-switch + current hard risk caps. Ranker not shipped.

---

## Amendments since draft (2026-05-27)

1. **Broker API is the primary live data source.** Upstox or Zerodha for spot, futures,
   and option chain. yfinance is *only* a fallback sanity check / non-critical
   monitoring source. The pipeline **must not** make trade decisions when the
   broker feed is unavailable, regardless of yfinance availability.
2. **Hard no-new-trades cutoff at 14:55 IST inside `v5_minute_decision.py`.**
   Any decision minute at or after 14:55 immediately returns `NO_TRADE_CUTOFF`
   without scoring the gate. Force-close at 15:25 is unchanged.
3. **Wednesday launch is preferred, not required.** Any non-holiday weekday
   where dte_today ∈ {2, 3} for NIFTY weekly expiry is acceptable for v1 launch.
   Wednesday remains the recommended first day.

---

## 1. System overview

### 1.1 What v1 does

For each decision minute (every 1 minute, 09:30–14:54 IST) on weekly-expiry NIFTY
where the trade date is 2 or 3 days from expiry:

```
1. Reject if minute >= 14:55  (NO_TRADE_CUTOFF)
2. Fetch live spot, futures (1-min OHLC + OI), option chain via broker API
3. Compute V4-A chain features (ATM IV, skew, PCR, max-OI, ...)
4. Compute V5-B futures features (basis, OI deltas, session position, ...)
5. Compute V4-B predicted realized vol → vol_kill if pred_rv > 0.235
6. Compute binary gate score → skip if gate_score < 0.70
7. Apply daily caps (2/index/day, 4 total/day)
8. Apply daily loss halt (stop new trades if cumulative day PnL < −₹15,000)
9. If all checks pass → log paper trade entry: SHORT ATM straddle, hold to 15:25
10. Track MTM intra-trade; force-exit if PnL hits −₹3,000
11. EOD reconciliation against close prices
```

### 1.2 What v1 does NOT do

- ❌ No live broker execution — paper only
- ❌ No real-money trades
- ❌ No notifications/SMS/Slack alerts (deferred to v1.1)
- ❌ No web UI (operate via CLI + log files + parquet ledger)
- ❌ No multi-strategy ranking (decided not to ship)
- ❌ No BANKNIFTY in v1 default config (see §1.3)

### 1.3 NIFTY primary, BANKNIFTY deferred

On the 2024 blind window, gate + SS_EOD delivered:

| Index     | Total PnL  | Win rate | Per-trade mean |
|-----------|------------|----------|----------------|
| NIFTY     | +₹111,990  | 75.3%    | +₹691          |
| BANKNIFTY | +₹1,455    | 65.2%    | +₹11           |

NIFTY drove ~99% of profit. **v1 default: NIFTY only.** BANKNIFTY available
behind a config flag for v1.1 evaluation but should not contribute meaningful capital.

---

## 2. Data refresh and validation

### 2.1 Data sources (amendment #1)

| Source                       | Role                       | Frequency               |
|------------------------------|----------------------------|-------------------------|
| Broker API (Upstox/Zerodha)  | **PRIMARY** — all live data | Every 1 min, 09:15–15:30 |
| NSE bhavcopy                 | EOD reconciliation         | Once daily, ~6 PM IST   |
| yfinance ^NSEI / ^INDIAVIX   | **FALLBACK ONLY** — sanity  | Pre-market only        |

**Critical rule:** trade decisions consume only broker-API data. yfinance is
used only for: (a) pre-market spot sanity vs broker, (b) operator dashboards,
(c) EOD VIX history. If broker feed fails mid-day, the system halts new
entries; it does **not** fall back to yfinance for trade decisions.

### 2.2 Pre-market refresh (08:00 IST)

```bash
.venv/bin/python scripts/v5_premarket.py
# - Updates NSE bhavcopy lake for prior trading day
# - Refreshes V4-A chain features and V5-B futures features (lag context)
# - Tests broker API auth
# - Cross-checks broker spot vs yfinance spot (warn if > 0.5% diff)
# - Writes today's go/no-go decision to flags/premarket_<date>.json
```

### 2.3 Health check (`scripts/v5_health_check.py`)

Validates before allowing any trades:

- [ ] Latest chain feature partition is from previous trading day
- [ ] `models/optinet_v5/gate_dte{2,3}.lgb` exist and load
- [ ] `models/optinet_v4/rv_30m_forward.lgb` exists and loads
- [ ] Broker API auth succeeds and returns a NIFTY spot quote
- [ ] Disk space > 5 GB free
- [ ] Today's dte_today ∈ {2, 3} for NIFTY weekly
- [ ] India VIX ∈ [8, 50] (pre-market sanity)
- [ ] Yesterday's paper ledger reconciled (or N/A if first day)
- [ ] No `flags/halt_*.flag` present unless explicitly cleared

If any required check fails → exit non-zero, decisions cron sees missing
go/no-go file and refuses to trade.

---

## 3. Daily cron schedule

```cron
# 08:00 IST — pre-market
0  8  * * 1-5  cd $OPTINET && .venv/bin/python scripts/v5_premarket.py >> logs/premarket_$(date +\%Y\%m\%d).log 2>&1

# 09:25 IST — final pre-flight
25 9  * * 1-5  cd $OPTINET && .venv/bin/python scripts/v5_health_check.py --strict || \
               touch flags/halt_today.flag

# 09:30 – 14:54 IST — every minute
*/1 9-14 * * 1-5 cd $OPTINET && .venv/bin/python scripts/v5_minute_decision.py >> logs/decisions_$(date +\%Y\%m\%d).log 2>&1

# 15:25 IST — force-exit any open positions
25 15 * * 1-5  cd $OPTINET && .venv/bin/python scripts/v5_force_close.py >> logs/eod_$(date +\%Y\%m\%d).log 2>&1

# 15:45 IST — EOD reconciliation
45 15 * * 1-5  cd $OPTINET && .venv/bin/python scripts/v5_eod_reconcile.py >> logs/reconcile_$(date +\%Y\%m\%d).log 2>&1

# 18:30 IST — drift check + daily report
30 18 * * 1-5  cd $OPTINET && .venv/bin/python scripts/v5_drift_check.py
```

Note: `v5_minute_decision.py` enforces the 14:55 cutoff internally regardless
of cron schedule. The cron `9-14` hour range and the script's own cutoff are
defense-in-depth.

macOS: prefer `launchd` over cron, or run with `caffeinate -i` to keep the
machine awake during market hours.

---

## 4. Live paper-trading checklist

### 4.1 Per decision minute (`v5_minute_decision.py`) — amendment #2

```
INPUT: current minute t
STEPS:
  0. If now() >= 14:55 IST → return NO_TRADE_CUTOFF       (HARD CUTOFF)
  1. If flags/halt_*.flag exists → return NO_TRADE_HALT
  2. If today's premarket file says no-go → return NO_TRADE_PREFLIGHT
  3. dte_today = (next_weekly_expiry - today).days
     If dte_today not in {2, 3} → return NO_TRADE_DTE
  4. Fetch via broker API: spot, fut, chain.
     If fetch fails → return NO_TRADE_DATA
  5. Compute V4-A chain features + V5-B futures features
  6. Score V4-B vol model → if pred_rv > 0.235 → NO_TRADE_VOL
  7. Score gate_dte{dte_today}.lgb → if score < 0.70 → NO_TRADE_GATE
  8. Read ledger; if trades_today_total >= 4 or trades_today_index >= 2 → NO_TRADE_CAPS
  9. If cumulative_day_pnl <= -15000 → NO_TRADE_HALT
  10. PLACE PAPER TRADE: SHORT ATM straddle, log full snapshot
RETURN: action enum + telemetry
```

### 4.2 Per minute MTM check on open positions

Every minute:
- Fetch current call + put price at entry strike from broker
- Compute MTM = (entry_premium − current_premium) × lot_size − costs_so_far
- If MTM ≤ −₹3,000 → exit at current price, log as STOPPED

### 4.3 EOD reconciliation (15:45 IST)

For each paper trade entered today:
- If exit recorded (stop or 15:25 force-close): keep
- If not exited: critical alert
- Recompute realized PnL using actual close prices from NSE bhavcopy
- Compare to live MTM-based PnL → flag if gap > ₹50
- Append to `ledger/v5_paper_ledger.parquet`

Compute and persist:
- Daily PnL summary
- Cumulative PnL since v1 start
- Win rate, profit factor, max DD, Sharpe (rolling 5d, 20d)

Alert if rolling-20d metrics deviate from backtest expectations
(2.68 trades/day, 70.7% win, +₹378 mean) by > 2 std.

---

## 5. Risk checks and halt conditions

### 5.1 Pre-trade risk checks (every decision minute)

| Check                | Threshold                       | Action if failed             |
|----------------------|----------------------------------|------------------------------|
| Decision time        | minute < 14:55                  | NO_TRADE_CUTOFF (hard)       |
| Data freshness       | last bar < 2 min old            | Skip minute                  |
| Spot sanity          | within 5% of yesterday's close  | Skip + alert                 |
| ATM IV sanity        | 0.08 ≤ atm_iv ≤ 1.0             | Skip + alert                 |
| Straddle premium     | ≥ ₹50 per share                 | Skip (insufficient juice)    |
| Days to expiry       | 2 ≤ dte ≤ 3                     | Skip (wrong bucket)          |
| Daily trade cap      | < 4 total today                 | Skip                         |
| Per-index cap        | < 2 per index today             | Skip                         |

### 5.2 Real-time halt conditions (system-wide)

Halt all new trades for the rest of the day:

| Condition                       | Threshold                     | Recovery                            |
|---------------------------------|-------------------------------|-------------------------------------|
| Cumulative day PnL              | < −₹15,000                    | Resume next trading day             |
| Consecutive stops               | 3 in a row                    | Resume next day after manual review |
| Broker feed failure             | > 5 minutes of missing data   | Resume when feed restored           |
| Spot move > 2% within 30 min    | flash crash                   | Resume next day after manual review |

Open positions are not auto-flattened; per-trade −₹3,000 stop continues to enforce.

### 5.3 Weekly halt conditions (system pause)

Pause v1 entirely until manual review:

| Condition                       | Threshold              |
|---------------------------------|------------------------|
| Rolling 5-day PnL               | < −₹40,000             |
| Rolling 20-day Sharpe           | < 0.5                  |
| Rolling 20-day win rate         | < 55%                  |
| Drift in feature distributions  | > 2 std vs training    |

If any triggers → write `flags/paused_<date>.flag`. Manual `flag_clear.sh`
required to resume.

### 5.4 Manual override

```bash
# Halt new trades for today
touch flags/halt_today.flag

# Halt indefinitely
touch flags/halt_indefinite.flag

# Force close all open positions immediately
.venv/bin/python scripts/v5_force_close.py --reason "manual halt"
```

Both `v5_minute_decision.py` and `v5_force_close.py` honor these flags every minute.

---

## 6. Rollback / disable procedure

### 6.1 Mid-day disable

1. `touch flags/halt_today.flag`
2. `.venv/bin/python scripts/v5_force_close.py --reason "rollback"`
3. Verify no open positions in `ledger/v5_paper_ledger.parquet`
4. Comment out or `crontab -r` the v5 cron entries
5. Notify anyone monitoring

### 6.2 Permanent disable

```bash
# Stop cron
crontab -l | grep -v "v5_" | crontab -

# Archive models
mv models/optinet_v5 models/optinet_v5.deprecated_$(date +%Y%m%d)
mv models/optinet_v4/rv_30m_forward.lgb models/optinet_v4/rv_30m_forward.lgb.deprecated_$(date +%Y%m%d)

# Final ledger snapshot
cp ledger/v5_paper_ledger.parquet ledger/archive/v5_final_$(date +%Y%m%d).parquet

# Final report
.venv/bin/python scripts/v5_final_report.py > docs/v5_v1_final_report_$(date +%Y%m%d).md
```

### 6.3 Rollback to a prior model version

```bash
mv models/optinet_v5/gate_dte3.lgb models/optinet_v5/gate_dte3_v2_demoted.lgb
cp models/optinet_v5/archive/gate_dte3_v1_2026MMDD.lgb models/optinet_v5/gate_dte3.lgb
.venv/bin/python scripts/v5_quickback.py --date $(date -v-1d +%Y-%m-%d)
```

Always retain ≥ 3 versioned snapshots in `models/optinet_v5/archive/`.

---

## 7. Operational dashboard (manual, daily)

| Metric                       | Today | 5d | 20d | Expected         |
|------------------------------|-------|----|-----|------------------|
| Trades taken                 |       |    |     | 2.7/day          |
| Trades attempted but skipped |       |    |     | (telemetry)      |
| Win rate                     |       |    |     | 70.7%            |
| Mean PnL/trade               |       |    |     | +₹378            |
| Cumulative PnL               |       |    |     | (cumulative)     |
| Stop-out rate                |       |    |     | 7.7%             |
| Worst trade                  |       |    |     | ≥ −₹3,000 (stop) |
| Worst day                    |       |    |     | ≥ −₹15,000 (halt)|

If actuals deviate > 2σ from expected over 20 days → invoke §5.3.

---

## 8. Roadmap to v1.1 (NOT in v1 scope)

After 3 months of paper-trading evidence:
- Re-evaluate BANKNIFTY with separate 2021–2024 trained gate
- Add live broker execution (Upstox/Zerodha) — convert paper to real
- Add Slack/SMS alerts for halts and reconciliation gaps
- Add per-index trade halt conditions
- Position sizing beyond 1-lot
- Re-evaluate constrained ranker with 2024+2025 data

---

## 9. Build order (concrete TODO)

1. `src/optinet/v5_runtime/broker.py` — abstraction + MockBroker + UpstoxBroker stub
2. `src/optinet/v5_runtime/ledger.py` — paper ledger schema & IO
3. `src/optinet/v5_runtime/online_features.py` — chain/futures feature compute at minute t
4. `scripts/v5_health_check.py`
5. `scripts/v5_minute_decision.py`
6. `scripts/v5_force_close.py`
7. `scripts/v5_eod_reconcile.py`
8. `scripts/v5_premarket.py`
9. `scripts/v5_drift_check.py`
10. End-to-end mock-broker test simulating one full day

Estimated effort: 5–7 days after broker API access.

---

## 10. Launch acceptance criteria (amendment #3)

Before promoting from "built" to "running paper":
- [ ] Health check passes 5 consecutive days in dry-run
- [ ] One full simulated trading day with mock broker shows correct decision flow
- [ ] EOD reconciliation correctly compares paper trades to actual close prices
- [ ] Stop-loss verified by injecting synthetic adverse spot move
- [ ] Daily-loss halt verified by injecting synthetic losing trades
- [ ] Halt flags (`halt_today.flag`, `halt_indefinite.flag`) tested
- [ ] Operator runbook printed and physically accessible
- [ ] **First paper-trading day is preferably a Wednesday (dte=2 NIFTY).**
      Other dte ∈ {2, 3} weekdays acceptable.

After 20 paper-trading days:
- [ ] Realized win rate within 60–80% (model expected 70.7%)
- [ ] No data integrity issues
- [ ] No reconciliation gaps > ₹50
- [ ] No unexpected halt triggers
- [ ] If all above: ready for v1.1 (real money) consideration

---

**This spec is frozen as of 2026-05-27.** Implementation focuses entirely on
operational robustness, data integrity, reconciliation accuracy, and validating
that live paper-trading behavior matches blind-window backtest expectations.
No additional strategy research, ranker work, or feature expansion in v1.
