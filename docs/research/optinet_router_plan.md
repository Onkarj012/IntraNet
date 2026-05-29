# OptiNet Router — Sprint Plan

**Frozen 2026-05-27**, after V5 was archived.

This document is the source of truth for what the router will and will not
attempt. The architecture is fixed; only the family engines and the router
glue change between sprints.

---

## Architecture

```
features ─→ regime detector ─→ family engines (futures / long_vol / debit_spread)
                                       │
                                       ▼
                              router selects best (or NO_TRADE)
                                       │
                                       ▼
                                   TradeCard
```

**Core principles (from the postmortem and the user's pivot directive):**

1. The system is a **router**, not a single-strategy bot. It chooses among
   strategy families based on market state, or refuses.
2. **NO_TRADE is a first-class output.** Most market states should produce it.
3. **Realistic execution is non-negotiable.** Every family validates against
   the market-price stop simulator. The BS-reprice simulator is debug-only.
4. **NIFTY first.** BANKNIFTY only after a NIFTY family is proven.
5. **One family at a time.** Each sprint is a yes/no test of one family.
6. **Trade cards always have all sections** — market state, recommendation,
   risk, feature snapshot — for full audit trail.

---

## Foundation (DONE, 2026-05-27)

| Deliverable | Status | Path |
|---|---|---|
| Trade-card schema | ✅ Done | `src/optinet_router/schema.py` |
| Family-engine ABC | ✅ Done | `src/optinet_router/families/base.py` |
| Family stubs (3) | ✅ Done | `src/optinet_router/families/{futures,long_vol,debit_spread}.py` |
| JSON round-trip + 6 invariants | ✅ Verified | (5 smoke tests passing) |
| Realistic market-price simulator | ✅ Reused from V5 | `scripts/v5_gated_backtest.py` |
| Online feature compute | ✅ Reused from V5 | `src/optinet/v5_runtime/online_features.py` |
| Broker abstraction | ✅ Reused from V5 | `src/optinet/v5_runtime/broker.py` |
| Paper-trading ledger | ✅ Reused from V5 | `src/optinet/v5_runtime/ledger.py` |

Total foundation effort: ~half day. The infrastructure inherited from V5 is
substantial and reusable — that was the silver lining of the V5 archive.

---

## Sprint #1 — Futures-first directional engine

**Status:** not started.
**Effort estimate:** 3-5 days.
**Why this might work:** simpler instrument, fewer moving parts, easier to
risk-manage. No vol-curve assumptions to get wrong. Cleanest first edge to
search for.

### Tasks
1. Build futures-direction labels (per-minute target: signed return over
   N-minute horizon, after costs). Likely horizons: 30/60/120/240 min.
2. Engineer features specific to futures direction (momentum, breakout,
   regime, time-of-day, OI-price interactions — the latter we already have).
3. Train classifier(s): LightGBM or similar, with calibration.
4. Validate under the realistic simulator (cost model, slippage, intra-trade
   stop simulation against actual minute bars).
5. Walk-forward eval on 2024 blind window.
6. Implement `FuturesEngine.score_minute()` to produce `FUTURES_LONG` or
   `FUTURES_SHORT` recommendations or return None.

### Success gates (before considering deployment)
- Net positive PnL on 2024 blind window after realistic costs (≥ ₹50k/year)
- Win rate ≥ 50% (futures need not have high win rate — large winners can carry)
- Profit factor ≥ 1.4
- Max drawdown ≤ ₹50k
- Sharpe (daily, ann.) ≥ 1.0
- 1-3 trades per day average (selectivity)
- Survives walk-forward (train < test cutoff for every fold)

If any fail → no deployment, document findings, move on.

---

## Sprint #2 — Long-vol options mirror

**Status:** not started.
**Effort estimate:** 1-2 days for directional answer.
**Why this might work:** the V5 postmortem flagged this as the most obvious
follow-up. Held-IV BS reprice would have *under*stated long-premium wins
(opposite of what it did to short premium). The same gate concept that
failed on short side may show edge on long side.

### Tasks (lightweight first pass)
1. Reuse `cache/optinet_v5/strategy_labels/` — it already has per-minute
   realized PnL for `LONG_STRADDLE_30M`, `LONG_STRADDLE_60M`,
   `LONG_STRANGLE_60M`. **No new label work required.**
2. Run `v5_gated_backtest.py` with a long-vol-only candidate set
   (strategy_id ∈ {10, 11, 12}) under chronological-first + market-price
   simulator. **This is essentially one CLI invocation** once the candidate
   filter supports a "long_vol_only" mode.
3. Slice the result the same way as the V5 audit (entry-time, dte_bucket,
   weekday, month, index).
4. If positive on any meaningful slice → write `LongVolEngine` and integrate.
   If negative everywhere → file the result and move on.

### Success gates (lower bar — this is a probe, not a product)
- At least one slice positive after realistic costs (≥ ₹20k/year on that slice)
- Win rate ≥ 30% (long premium has lower win rate but larger winners)
- Profit factor ≥ 1.3
- If gate exists in archive: works on read-only archived V5 gate models;
  otherwise ship as a no-go decision.

### Failure mode
This sprint primarily answers: **"does the V5 gate concept have any
salvageable signal?"** If long-vol also fails realistic execution, the
microstructure features in the gate are not predictive of realized vs
implied vol movement — and the entire feature set should be deprecated for
options strategies (futures-direction may still work).

---

## Sprint #3 — Defined-risk directional spreads

**Status:** not started.
**Effort estimate:** 3-4 days.
**Why this might work:** middle ground between futures (no vol risk) and
naked options (full vol risk). Capped downside removes the fat-tail problem
that hurt V5 short-vol. Aligns with TREND regime when paired with the right
family-router.

### Tasks
1. Generate defined-risk labels (CALL_DEBIT_SPREAD, PUT_DEBIT_SPREAD) per minute,
   under realistic execution. The existing v5_simulator already has these
   structures (strategy_ids 6-9) — verify the labels and reuse if sound.
2. Train a directional gate (call-side vs put-side vs no-trade) that
   incorporates regime + futures momentum + chain features.
3. Validate under realistic simulator with proper spread costs (4 legs,
   4× brokerage, slippage on 4 legs).
4. Walk-forward eval, slice diagnostics.
5. Implement `DebitSpreadEngine.score_minute()`.

### Success gates
- Net positive PnL ≥ ₹40k/year (lower than futures because spreads have
  smaller per-trade size)
- Win rate ≥ 45%
- Profit factor ≥ 1.4
- Max single-trade loss is bounded (defined risk = max premium paid)
- Survives walk-forward

---

## Sprint priority — recommended order

| Sprint | Effort | Risk | Information value | Recommended order |
|---|---|---|---|---|
| #2 Long-vol mirror | **1-2 days** | low (reuses everything) | high — clean yes/no on V5 gate | **First** |
| #1 Futures-first | 3-5 days | medium (new labels + features) | high — most likely real edge | Second |
| #3 Defined-risk spreads | 3-4 days | medium-high (4-leg cost model) | medium | Third |

### Rationale for going long-vol first

1. **Cheapest test** — labels and infrastructure already exist
2. **Tightest informational signal** — directly answers whether the gate
   concept has any salvageable use, regardless of which family wins overall
3. **Bounded downside** — long premium has capped loss per trade by design
4. **If positive**, even modestly, validates that the V5 microstructure
   features have direction-of-vol signal, just polarity-inverted
5. **If negative**, we save 3-5 days on a futures sprint by knowing the
   gate isn't relevant for futures either (futures direction is a different
   feature problem, but knowing the gate has no signal is informative)

### After all three sprints

The router will combine winners. If only one survives, that becomes the
single-family v1 product. If two or three survive, the router becomes a
true regime-aware multi-family system per the user's vision.

If **none** survive realistic execution: the conclusion is that this
codebase's chain-microstructure feature set does not produce intraday
options edge on weekly NIFTY contracts. At that point, the next research
direction would be **longer-dated options** (monthly expiry, dte 14-28)
or **completely different feature sources** (order flow, microstructure
imbalance, options flow).

---

## What stays archived

- The V5 short-vol path is closed. Nothing in `archive/v5/` is being revived.
- The BS-reprice stop simulator is permanently gated behind the
  `--use_bs_reprice_stop_DEBUG` flag for one-off comparisons only.
- The V5 v1 deployment spec is closed. The next deployable product is a
  router v1 product, with its own spec, after at least one sprint passes
  its success gates.

---

## Definition of done for the router project

The router has a v1 product when **at least one** of the three families:

1. Passes all success gates listed above
2. Has been validated under walk-forward (not just blind 2024)
3. Has been paper-traded for ≥ 20 sessions with realized PnL within 2σ of
   the realistic-simulator expectation
4. Has a frozen deployment spec analogous to (but more honest than) the
   V5 v1 spec

Until then, no live or paper deployment.
