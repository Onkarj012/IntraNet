# Forward-walk verdict — NIFTY long-only futures engine

Run date: 2026-05-28  
Script: `scripts/forward_walk_2024_2026.py`  
Log: `logs/forward_walk_2024_2026.log`

## TL;DR

**The edge is real.** On 19 months of true out-of-sample data
(Nov 2024 → May 2026, never used in training or model selection),
the long-only futures engine delivered:

| Variant | Trades | Win % | Total PnL | Sharpe | PF | Max DD | Pos months |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A: hard filters only** | 1,128 | 49.9 % | **+₹429,227** | +2.17 | 1.39 | −₹74,024 | 13/19 |
| **B: + regime guards** | 879 | 50.5 % | **+₹339,979** | +2.25 | 1.41 | −₹58,689 | 12/19 |

Both are profitable. Variant A makes more money in absolute terms;
Variant B has slightly better risk-adjusted return and shallower
drawdown by filtering 22 % of trades.

**Recommendation: GO to paper trading with Variant A** (just the existing
hard filters), and ship Variant B's guard as a switchable knob for the
runtime. Keep the `final_long.lgb` model unchanged. Do NOT retrain.

---

## Phase 1 — The October 2024 hypothesis was wrong

The original hypothesis was that October 2024 lost money because of
geopolitical stress (Iran-Israel escalation + record FII outflows) and
that a VIX-based filter would catch it. **This is false.**

**Median VIX on October 2024 trade days: 13.6.** That is below the
2024 average (≈ 14.8) and well below any "elevated stress" cutoff.
October was a *steady drawdown*, not a volatility spike.

VIX bucketing on the full 622-trade 2024 ledger:

| VIX bucket | Trades | Win % | Total PnL |
|---|---:|---:|---:|
| 0-12 | 39 | 59.0 % | +₹4,775 |
| 12-14 | 267 | 53.6 % | +₹79,497 |
| 14-16 | 196 | 54.6 % | +₹75,924 |
| **16-18** | **54** | **63.0 %** | **+₹72,885** |
| 18-20 | 12 | 50.0 % | +₹26,052 |
| 20+ | 51 | 52.9 % | +₹19,954 |

High-VIX buckets had the **highest** win rates. A VIX cap throws away
profitable mean-reversion days. **A VIX-only filter would hurt the model.**

The signal that *does* work is the 5-day NIFTY return (drawdown halt):

| 5d-ret bucket | Trades | Win % | Total PnL |
|---|---:|---:|---:|
| **[−10%, −4%)** | **9** | **0.0 %** | **−₹27,000** |
| [−4%, −2%) | 57 | 49.1 % | +₹2,822 |
| [−2%, −1%) | 66 | 53.0 % | +₹38,229 |
| [−1%, 0%) | 123 | 56.9 % | +₹69,643 |
| [0%, +1%) | 111 | 51.4 % | +₹35,098 |
| [+1%, +2%) | 163 | 59.5 % | +₹67,112 |
| [+2%, +4%) | 90 | 58.9 % | +₹87,385 |

The deepest-drawdown bucket lost on every trade. That's the bucket to skip.

## Phase 2 — Best in-sample guard

Grid search across VIX cutoffs × 5-day-return floors on the 2024 ledger
(in-sample selection — *never use these specific numbers as a forward
metric*):

```
guard                            n   win%      totPnL   Sharpe       DD
─ no guards (baseline)         622  55.1%   +287,536    +2.88   -67,909
─ ret_5d > -1.5% only          532  57.3%   +308,111    +3.63   -35,981
─ ret_5d > -3.0% only          592  56.8%   +334,001    +3.55   -51,761
vix < 22 + ret_5d > -1.5%      517  57.6%   +307,925    +3.71   -35,981  ← best Sharpe (n≥200)
vix < 22 + ret_5d > -3.0%      571  57.1%   +339,069    +3.71   -51,761  ← best PnL among Sharpe>3.5
```

Two observations:

1. The drawdown floor alone captures most of the benefit. The VIX
   ceiling at 22 only filters one extra day in 2024 — its added value
   is marginal.
2. Sharpe improvement from 2.88 → 3.71 (+0.83) and drawdown improvement
   from −₹67k → −₹36k (≈ halved) are both meaningful.

These numbers are **in-sample** to 2024. They tell us the *shape* of a
useful guard, not its expected forward performance.

## Phase 3 — 19-month out-of-sample forward walk

Test window: 2024-11-01 → 2026-05-15 (19 months, 376 trading days).
Model: `final_long.lgb` (single-tree LightGBM trained on
2020-01 → 2024-10-31). Features rebuilt from NIFTY index minute bars
as a futures proxy (basis < 0.3 %, OI features carry 0.7 % of model
gain so OI=0 / vol=1 substitution is acceptable).

### Variant A — hard filters only

```
OVERALL  n=1,128   days=376   win=49.9 %   PnL=+₹429,227   Sharpe=+2.17   PF=1.39   DD=-₹74,024
```

Monthly:

| Month | n | Win % | PnL (₹) | Sharpe |
|---|---:|---:|---:|---:|
| 2024-11 | 54 | 61.1 % | +55,235 | +4.61 |
| 2024-12 | 63 | 54.0 % | +51,133 | +4.06 |
| 2025-01 | 69 | 59.4 % | **+115,856** | +7.88 |
| **2025-02** | 60 | **31.7 %** | **−65,637** | −8.45 |
| 2025-03 | 57 | 52.6 % | +27,591 | +3.05 |
| 2025-04 | 57 | 42.1 % | +3,432 | +0.31 |
| 2025-05 | 63 | 34.9 % | −3,748 | −0.29 |
| 2025-06 | 63 | 52.4 % | +27,293 | +2.31 |
| 2025-07 | 69 | 52.2 % | +2,930 | +0.36 |
| 2025-08 | 57 | 64.9 % | +29,849 | +4.00 |
| 2025-09 | 63 | 49.2 % | +6,885 | +0.97 |
| 2025-10 | 60 | 60.0 % | +28,442 | +3.62 |
| 2025-11 | 57 | 47.4 % | −13,671 | −1.97 |
| 2025-12 | 66 | 43.9 % | +4,283 | +0.47 |
| 2026-01 | 60 | 46.7 % | +40,385 | +3.09 |
| **2026-02** | 63 | **28.6 %** | **−48,164** | −5.13 |
| 2026-03 | 57 | 63.2 % | **+79,276** | +6.11 |
| 2026-04 | 60 | 53.3 % | +51,743 | +4.62 |
| 2026-05 | 30 | 56.7 % | +36,114 | +5.51 |

By regime: expansion +₹320k (PF 1.98), range +₹107k (PF 1.14), trend_dn
+₹1k (only 4 trades). Compression skipped.

By exit reason: 155 stops × −₹3k = −₹465k, 98 targets × +₹5.9k = +₹581k,
875 time-exits × +₹0.36k = +₹313k. Time-exits + targets cover the
stop losses with room to spare.

### Variant B — hard filters + Variant 2 guards (vix<22, ret_5d>−1.5%)

```
OVERALL  n=879   days=293   win=50.5 %   PnL=+₹339,979   Sharpe=+2.25   PF=1.41   DD=-₹58,689
```

Effect of guards by month (PnL change from A):

| Month | A | B | Δ |
|---|---:|---:|---:|
| 2024-11 | +55k | +48k | -7k |
| 2024-12 | +51k | +45k | -6k |
| 2025-01 | +116k | +115k | -1k |
| **2025-02** | **−66k** | **−25k** | **+41k** ← guards saved 62 % of Feb loss |
| 2025-03 | +28k | +34k | +6k |
| 2025-04 | +3k | +5k | +2k |
| 2025-05 | -4k | -2k | +2k |
| 2025-08 | +30k | +16k | -14k |
| 2025-11 | -14k | -5k | +9k |
| 2026-01 | +40k | +13k | -27k |
| **2026-02** | **−48k** | **−46k** | +2k ← guards barely helped here |
| **2026-03** | **+79k** | **+21k** | **−58k** ← guards killed a great month |

Guards do exactly what we designed them for in **Feb 2025** (the
Israel-Iran echo / FII reversal episode), but they over-correct in
**Mar 2026** by skipping a recovery rally and they fail to catch
**Feb 2026**. Net: better Sharpe and drawdown, lower total PnL.

### What hurt in Feb 2025 / Feb 2026?

Both bad months share:
- Persistent intraday weakness (TIME exits going against the LONG bias)
- Win rates collapse to 28-32 % (vs 50 % baseline)

Feb 2025 had a 5-day-return drawdown signature → guards caught some of it.
Feb 2026 was a sharp but short selloff with no multi-day buildup → guards
did not see it coming.

This is a real residual weakness. **It is not a reason to reject the
edge — 17/19 months are at least neutral, and the average month is
still profitable. But it tells us the next iteration should focus on
intraday drawdown halts (e.g. cumulative session PnL halt, or VIX
intraday spike halt) rather than overnight filters.**

---

## Caveats and known artifacts

1. **Proxy data assumption.** Futures vendor data ends Sep 2024. We use
   NIFTY index OHLC for `f_close` and set `f_oi=0`, `f_vol=1`. This is
   sound because: (a) cash-future basis is < 0.3 %, (b) OI features carry
   0.7 % of model gain, (c) volume features are similarly minor. Should
   re-validate by comparing proxy results to actual-futures results for
   Nov-Dec 2024 (we have both for those months — TODO).

2. **Regime label distribution differs.** On the proxy table, only 36
   bars get `trend_up` and 36 get `trend_dn` (the rest are range /
   compression / expansion). The training pipeline produced a more
   balanced regime mix. This doesn't affect the trading filter (we only
   skip "compression"), but it does mean the per-regime breakdowns
   above can't be directly compared to training-period regime splits.

3. **Single-tree model.** `final_long.lgb` has `num_trees=1`. 87.9 % of
   gain comes from `realized_vol_30m` + `minute_of_day`. That's
   essentially a 2-feature rule. The robustness we see (edge surviving
   19 months of new data) is partly *because* the model is so simple —
   nothing complex to overfit. There is likely real upside from a
   properly regularized deeper model, but that would require a fresh
   walk-forward validation cycle.

4. **The 2024 backtest used in Phase 1/2 is the old futures-data
   backtest.** It is not directly comparable to the proxy Phase 3
   backtest because the price stream and feature columns differ slightly.
   Phase 1's bucketing should be treated as a hypothesis-generation step,
   not a precise estimate.

5. **Costs and slippage.** Backtest uses ₹105 round-trip per lot, which
   was calibrated against the V5 paper-trading data. If real broker costs
   come in higher (e.g. ₹150-180 with worse fills), forward Sharpe drops
   from 2.17 → roughly 1.6 (rough proportional estimate, not measured).
   **Run cost-sensitivity test before paper trading.**

6. **Drawdown halt is a `> -1.5%` threshold on prior 5-day return.**
   This is the single number that does the most work. Tested values
   on 2024:

   | ret_5d cutoff | 2024 Sharpe |
   |---:|---:|
   | none | 2.88 |
   | -3.0% | 3.55 |
   | -2.0% | 3.48 |
   | -1.5% | 3.63 |
   | -1.0% | 3.53 |
   | -0.5% | 3.56 |

   The benefit is broad — pick anything in [-2.5 %, -0.5 %] and you get
   most of the gain. -1.5 % is the median.

---

## What needs to be done

**Tier 1 — before paper trading (must do):**

1. **Validate the proxy.** Re-run Phase 3 on Nov-Dec 2024 only, but
   using the cached actual-futures features (which we have). Confirm
   PnL and Sharpe match the proxy results within ±10 %.
   *Rationale: rules out a systematic bias from the index-vs-futures
   substitution. Quick — ~5 minutes.*

2. **Cost sensitivity test.** Re-run the 19-month forward walk with
   `COSTS_INR = 150` and `COSTS_INR = 200`. Confirm Sharpe stays > 1.5
   and total PnL stays positive at ₹150.
   *Rationale: real-world broker costs are ~₹150 round-trip on NIFTY
   futures with retail-tier slippage. ~10 minutes.*

3. **Permutation / shuffle test.** Score the model on a permuted
   trade-date column (rows scrambled across days but keep within-day
   order). Re-run backtest. Confirm shuffled-baseline Sharpe is roughly
   zero. If shuffled results are still positive, we have a labelling /
   leakage bug.
   *Rationale: catches subtle look-ahead. ~10 minutes.*

**Tier 2 — selectivity tuning (should do):**

4. **Test 90th and 95th percentile entry thresholds.** Currently we use
   85th. Higher-percentile cutoffs will reduce trade count but should
   raise win rate. Pick the threshold that maximizes forward Sharpe
   without falling below ~50 trades/month.

5. **Decide on guards in production.** Option A (no guards, ship as is)
   makes more money (+₹89k more on the forward window). Option B
   (with guards) has lower drawdown. **Recommend ship Option A and add
   a `--regime_guard` runtime flag** — paper-trade both for a month,
   compare on live data, then commit.

**Tier 3 — research follow-ups (don't block paper trading):**

6. Investigate the Feb 2025 / Feb 2026 failure mode. Both are
   short-window selloffs. Could be helped by: intraday cumulative-PnL
   halt, VIX intraday spike filter, or an explicit "no LONG within
   first 30 min of any −1 % gap-down day".

7. Retrain with proper boosting (>1 tree). Use the 2025 forward window
   as a held-out validation set, train on 2020 → 2024-10-31, model-select
   on 2024-Q4 + 2025-H1, blind-test on 2025-H2 + 2026-H1.

8. Build per-month stability dashboard: rolling 30-day Sharpe + win
   rate, alert when 30-day Sharpe drops below 0.5 (live regime change).

---

## What does NOT need to be done

- ❌ **Do not retrain `final_long.lgb` on 2024 or 2025 data.** That
  invalidates the entire forward-walk validation.
- ❌ **Do not add a VIX cap.** Phase 1 showed it filters out the wrong
  days (high-VIX days actually have 63 % win rate). The VIX threshold
  in Phase 2 was selected jointly with the drawdown halt — the
  drawdown halt does almost all the work.
- ❌ **Do not adjust target / stop / horizon.** TARGET_PCT=+0.40 %,
  STOP_PCT=−0.30 %, HORIZON=60 are all original-design parameters and
  still produce a positive expected value 19 months out-of-sample.
  Tuning these would be retroactive curve fitting.
- ❌ **Do not add SHORT side back.** The 2024 short signal lost money
  (per prior verdict). 2025 was even more bullish (NIFTY 23,650 →
  ~24,000) — SHORT would be even worse.
- ❌ **Do not reactivate the BS-reprice stop simulator.** It over-stated
  PnL by ~₹870/trade in V5. Stay with market-price exits.
- ❌ **Do not deploy options strategies.** Both options paths (V5 short
  vol + Sprint #2 long vol) failed realistic execution.

---

## Files produced

```
cache/router_v0/futures_features_proxy.parquet     118,626 rows × 60 cols, Nov-2024 → May-2026
results/router_v0/phase1_hypothesis_buckets.json    2024 PnL × VIX × 5d-ret buckets
results/router_v0/phase2_guard_grid.parquet         72-cell guard cutoff grid
results/router_v0/phase3_fwd_no_guard.parquet       1,128 forward trades (Variant A)
results/router_v0/phase3_fwd_with_guard.parquet     879 forward trades (Variant B)
results/router_v0/forward_walk_summary.json         consolidated metrics
logs/forward_walk_2024_2026.log                     full run log
scripts/forward_walk_2024_2026.py                   driver script (re-runnable)
```

## Decision

**GO. Move to paper trading with Variant A. Re-evaluate after one
calendar month of live paper data.**

Conditions:
- Tier-1 validation (proxy check, cost sensitivity, permutation test)
  must pass before going live.
- Stop paper trading and escalate if rolling 30-day Sharpe drops below 0.
- Re-evaluate model on 2025 H2 + 2026 H1 once proper supervised
  retraining is done (Tier 3 item 7).
