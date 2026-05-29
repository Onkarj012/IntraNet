# V5 Postmortem — `gate + SHORT_STRADDLE_EOD` strategy archived

**Date archived:** 2026-05-27
**Final blind-window result (realistic execution, full 2024):** −₹153,685 over 300 trades, 50.3% win rate, profit factor 0.43, Sharpe −5.19.
**Decision:** archive the strategy path; do not deploy.

---

## What was built

A gated short-volatility strategy on NIFTY/BANKNIFTY index options:

```
candidate minute
  → V4-B vol kill-switch (skip if pred_rv > 0.235)
  → binary gate (skip if score < 0.70, dte bucket ∈ {2, 3})
  → SHORT_STRADDLE_EOD entry, hold to 15:25
  → per-trade −₹3,000 stop-loss
  → daily caps (2/index/day, 4 total/day)
  → daily loss halt at −₹15,000
```

Backtested across 2024 with the original simulator: **+₹113,445** total, 70.7% win rate, Sharpe +4.45. The proposed v1 ship config.

---

## Why it was archived

The strategy was tested progressively under more realistic execution rules. At each step its apparent edge eroded, until under fully realistic execution the result flipped from "+₹113k profit" to "−₹154k loss" — a swing of ~₹267k on the same 300 trades.

### Stage 1: Look-ahead-bias test (chronological-first selection)

The original backtest selected the top-2 minutes per day by gate score. Live execution can't do that — it has to commit to the first qualifying minute chronologically. Re-running the same backtest with chronological-first selection:

| | Top-K by score | Chronological first | Delta |
|---|---|---|---|
| Total PnL | +₹113,445 | +₹107,945 | −₹5,500 |
| Win rate | 70.7% | 70.3% | −0.4 pp |
| Stop-out rate | 7.7% | 11.3% | +3.6 pp |

**Look-ahead bias was small (~5%).** Not the culprit.

### Stage 2: Stop-loss simulator audit (BS-reprice vs market-price)

The original simulator priced the trade's intra-minute MTM via Black-Scholes with `atm_iv` held constant from entry. A new simulator was built that uses the actual minute-bar option chain prices. On the same 300 trades:

| | BS-reprice (held IV) | Market-price (actual chain) | Delta |
|---|---|---|---|
| Total PnL | +₹107,945 | **−₹153,685** | **−₹261,630** |
| Win rate | 70.3% | 50.3% | −20.0 pp |
| Stop-out rate | 11.3% | **23.0%** | +11.7 pp |
| Mean PnL/trade | +₹360 | −₹512 | −₹872 |
| Profit factor | 1.83 | 0.43 | −1.40 |
| Sharpe (daily, ann.) | +3.77 | −5.19 | −8.96 |
| Max drawdown | −₹27k | −₹150k | −₹123k |

**This is where the entire apparent edge lived.** ~₹870 of phantom edge per trade.

### Stage 3: Narrow-slice last-resort test

The audit suggested the realistic edge, if it existed at all, would be concentrated in NIFTY × Monday × dte_bucket=2 × early-session entries. Re-running the realistic simulator on just that slice:

```
NIFTY × Monday × dte_bucket=2 × 09:15-10:15 entries:
  76 trades, 38 days, 55.3% win rate
  Total: -₹26,530    PF 0.56    Sharpe -3.62
  Max DD: -₹23,530   Stop rate: 17.1%
```

**The narrowest slice with the strongest theoretical edge is still negative.** Only 4 of 10 months were positive even within this slice.

---

## Failure mode — why BS-reprice over-states short-premium PnL

The held-IV assumption fails systematically when:

1. **Vol expands intraday** — the held atm_iv from entry doesn't update, so BS predicts a smaller option-price move than what really happens. Real stops fire; BS stops don't.
2. **Skew shifts** — atm_iv is one number, but skew can rotate without atm_iv changing. Real wing prices move; BS assumes they stay fixed relative to atm.
3. **Spot gaps mid-trade** — BS assumes smooth diffusion; real markets have discontinuous jumps that immediately move option prices outside the BS-implied range.

All three of these scenarios punish short-premium strategies disproportionately. The V5 gate had no signal that distinguished "vol about to stay calm" from "vol about to expand" — it produced ~70% directional-edge appearance under BS assumptions, but ~50% under real prices.

The held-IV BS simulator was therefore the **principal source of optimism** in the V5 backtest, not look-ahead bias.

---

## Slice-level results (full 2024, realistic simulator)

**Every slice is negative:**

```
By weekday:        Monday   −₹26k   Thursday −₹15k   Friday −₹115k
By dte_bucket:     bucket 2 −₹25k   bucket 3 −₹128k
By entry-time:     09:15-09:45 −₹52k   09:45-10:15 −₹66k   10:45-12:15 −₹26k
By index:          NIFTY −₹98k       BANKNIFTY −₹56k
By month:          Every single month negative; March +₹0.3k flat at best.
```

Pushing entries later (the originally-recommended 10:30 fix) made things WORSE in the realistic simulator — same as it did in early paper-trading e2e tests. The early-session theoretical edge from the BS audit was a vol-curve artifact too.

---

## What was kept and why

**Kept (reusable infrastructure):**

| Path | Reason |
|---|---|
| `scripts/v5_gated_backtest.py` | Now defaults to the realistic market-price simulator. Useful for any future short-vol research. BS function renamed to `simulate_trade_with_stop_BS_DEBUG_ONLY` and gated behind `--use_bs_reprice_stop_DEBUG`. |
| `src/optinet/v5_runtime/broker.py` | Generic `BrokerClient` ABC + `MockBroker` + `UpstoxBroker` stub. Strategy-agnostic. |
| `src/optinet/v5_runtime/online_features.py` | Generic online feature compute (chain + futures). Strategy-agnostic. |
| `src/optinet/v5_runtime/ledger.py` | Paper-trade ledger schema + IO. Strategy-agnostic. |
| `src/optinet/v5_runtime/runtime_config.py` | Risk constants, expiry calendar, cost model. Strategy-agnostic. |
| `src/optinet/v5_simulator.py` | Strategy label generator (13 strategies × every minute). Reusable for any future research that needs PnL ground-truth per (minute, strategy). |
| `src/optinet/v5_futures.py` | Per-minute futures feature builder. Reusable. |
| `cache/optinet_v5/` | 10M-row labelled dataset + futures features. Reusable. |
| `cache/optinet_v4/` | 900K-row chain features (V4-A). Reusable. |
| `models/optinet_v4/rv_30m_forward.lgb` | Vol forecaster (V4-B). Useful as a generic feature. |
| `scripts/build_v5_*.py` | Feature/label orchestrators. Reusable. |
| `scripts/smoke_v5_ranker.py`, `scripts/gate_v5.py` | Model trainers. Useful templates for any future LambdaRank/binary-gate work. |

**Archived (V5-strategy-specific):**

| Path | Reason |
|---|---|
| `archive/v5/scripts/v5_premarket.py` | V5 paper-trading orchestrator |
| `archive/v5/scripts/v5_health_check.py` | V5 pre-flight |
| `archive/v5/scripts/v5_minute_decision.py` | V5 per-minute decision pipeline |
| `archive/v5/scripts/v5_force_close.py` | V5 EOD close |
| `archive/v5/scripts/v5_eod_reconcile.py` | V5 EOD reconciliation |
| `archive/v5/scripts/v5_drift_check.py` | V5 drift-check cron |
| `archive/v5/scripts/v5_e2e_test.py` | V5 e2e test |
| `archive/v5/scripts/v5_thorough_test.py` | V5 multi-day test runner |
| `archive/v5/docs/v5_v1_deployment_spec.md` | Frozen V5 v1 deployment spec |
| `archive/v5/models/optinet_v5/` | All V5 gate + ranker model files |
| `archive/v5/ledger/v5_paper_ledger*.parquet` | All paper-trading test ledgers |
| `archive/v5/results/phase3_*` | All V5 backtest result snapshots |
| `archive/v5/logs/v5_*` | All V5 build/audit log files |

---

## Lessons recorded for future research

1. **The held-IV BS-reprice stop-loss simulator is structurally wrong for short-premium strategies.** Always use minute-bar market prices when validating any strategy that depends on intra-trade MTM. The repo's default is now the market-price simulator; the BS variant requires opt-in via `--use_bs_reprice_stop_DEBUG`.

2. **Backtest selection bias (look-ahead) is much smaller than simulator bias.** Selecting top-K by score vs first-K chronologically only changed PnL by ~5%. The much bigger trap is what the simulator assumes about prices the model hasn't seen.

3. **A 70% backtest win rate on a short-premium strategy is a red flag.** Realistic short-vol strategies in retail option chains rarely sustain win rates above 55-60% net of stops. If a backtest shows 70%+, suspect the stop-loss simulator first, not the model.

4. **Paper-trading e2e tests would have surfaced this earlier.** The first 16-day paper-trading run already showed −₹13k vs the backtest's expected +₹6k for the same period. The audit just confirmed and quantified it. Any future strategy should be paper-tested with realistic prices before any production rollout.

5. **The V4-B vol kill-switch and the gate are individually weak signals.** They produced apparent edge only when combined with an over-optimistic stop simulator. Their edge does not survive realistic execution.

---

## What NOT to do next

- **Do not retune V5 parameters.** The gate threshold, vol kill threshold, time-of-day floor, dte bucket selection, and per-trade stop level were all explored. None made the realistic-simulator result positive on any meaningful slice. The strategy concept is the failure point, not the parameter values.
- **Do not extend to BANKNIFTY or any other underlying.** BANKNIFTY tested worse than NIFTY (−₹56k vs NIFTY −₹98k in the realistic simulator).
- **Do not re-run the BS-reprice simulator and treat its output as a target.** Any future optimism rooted in BS-reprice numbers is not real.

---

## Possible next research directions (not committed)

These are suggestions, not commitments. The user has not authorized any new research path.

1. **Long-vol strategy on the same gate.** If the gate has any directional edge, it might be on the LONG side rather than SHORT — long premium pays when realized > implied, and held-IV BS would understate (not overstate) those wins. Worth a one-day audit if the user wants. Use the same 13-strategy label set but evaluate LONG_STRADDLE_30M and LONG_STRANGLE_60M only.
2. **Volatility forecasting as a standalone product.** V4-B has measurable test R² 0.61. A pure vol forecast (no trade selection) might be useful as a feature for other systems.
3. **Different timeframes.** All V5 work was on dte≤7. Monthly options (dte 14-28) have different microstructure and might survive realistic stops better — but that requires fresh feature engineering.
4. **Different structures on the same gate.** Iron condors, butterflies, calendar spreads. These have defined-risk profiles that may respond differently to vol expansion than naked short premium.

None of these are validated yet. Each would need its own honest realistic-simulator audit before any deployment work.

---

## Closing

V5 was a 12-day research effort that produced reusable infrastructure (broker abstraction, online feature compute, ledger, market-price simulator, label dataset) but no deployable strategy. The infrastructure is preserved; the strategy is archived.

The most important deliverable from this work is the **realistic stop-loss simulator** at `simulate_trade_with_stop_market()` in `scripts/v5_gated_backtest.py`. This is now the default for any future short-premium backtest in this codebase, preventing the same trap from being re-encountered.
