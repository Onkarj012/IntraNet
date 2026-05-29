# Sprint #2 Verdict — Long-vol mirror

**Date:** 2026-05-27
**Decision:** **FAIL.** No salvageable signal in V5 features for long-vol entries.
**Effort spent:** ~30 min (cheapest sprint, as predicted).
**Move to next sprint:** Yes → Sprint #1 (futures-first directional).

---

## What was tested

Three configurations, all under realistic execution (chronological-first +
14:55 cutoff + market-price stop simulator + same risk caps as V5):

### Test A — V5 short-vol gate, force long-vol structure (naive)

Uses the archived V5 gate models (which fire when conditions favor short-vol)
and forces a long-vol entry at those minutes. Expected to be anti-edge.

| Strategy | Trades | Win % | Mean | Total | PF | Sharpe |
|---|---|---|---|---|---|---|
| LONG_STRADDLE_30M  (id 10) | 300 | 7.0% | −₹671 | −₹201,414 | 0.04 | −19.81 |
| LONG_STRADDLE_60M  (id 11) | 300 | 15.3% | −₹678 | −₹203,295 | 0.10 | −15.22 |
| LONG_STRANGLE_60M  (id 12) | 300 | 14.3% | −₹564 | −₹169,219 | 0.10 | −14.38 |

All confirm: V5 gate is anti-edge for long-vol (entering long when the model
expects realized vol to come in below implied).

### Test B — Fresh long-vol-specific gate

Trained a new binary gate using the SAME V4-A chain features + V5-B futures
features, but with the label: "did `LONG_STRADDLE_60M` produce positive PnL
at this minute?" Train 2020-2022, validate 2023, test 2024.

| dte_bucket | Train rows | Pos rate | best_iter | VAL AUC | TEST AUC |
|---|---|---|---|---|---|
| 2 | 88,697 | 8.4% | **1** | 0.513 | **0.505** |
| 3 | 90,579 | 8.2% | **1** | 0.575 | **0.456** |

`best_iter=1` is the killer fact — the LightGBM trainer found nothing
meaningful to learn beyond the marginal positive rate. Test AUC of **0.456**
on dte=3 means the model is **anti-predictive** out-of-sample.

The gate's positive predictions never exceed 0.70 (because the marginal rate
is only ~8%). At threshold = 0.10 (lowest plausible cut):

| | Trades | Win % | Total PnL | Sharpe | PF |
|---|---|---|---|---|---|
| Long-vol gate @ 0.10, force=LS_60M | 292 | **8.56%** | **−₹220,050** | −16.46 | 0.08 |

Win rate of 8.56% is **essentially the base rate** (8.4%). The gate is
selecting random minutes within the eligible set — no skill at all.

---

## Interpretation

This was the cheapest probe in the router plan, designed to answer one
question:

> Does the V5 microstructure feature set (V4-A chain + V5-B futures + lags +
> time-of-day) have any salvageable predictive power for option strategies,
> regardless of polarity (short-vol or long-vol)?

The answer is **no**. Specifically:

1. **The V5 feature set does not predict realized-vs-implied vol for either
   direction.** Short-vol failed under realistic execution (already archived).
   Long-vol fails to learn at all (best_iter=1, test AUC ≈ 0.5).
2. **No simple polarity flip rescues the V5 gate.** Inverting the gate or
   training a fresh one with the same features produces noise.
3. **The V5 gate's apparent edge under BS-reprice was therefore not a
   directional signal at all** — it was a simulator artifact.

This kills any future research that wants to reuse the V5 chain-microstructure
feature set as a primary signal for options strategies on weekly NIFTY.

It does **not** kill futures-first (Sprint #1), because futures direction
is a different feature problem — futures momentum, breakout, opening-range
behavior, and trend continuation use different microstructure signals than
the options chain features.

---

## Files written

- `scripts/train_long_vol_gate.py` — fresh long-vol gate trainer
- `models/router_v0/long_vol/gate_dte{2,3}.lgb` — the (failed) long-vol gates
- `models/router_v0/long_vol/long_vol_gate_summary.json` — training metrics
- `results/optinet_v5/phase3_summary_sprint2_*.json` — backtest summaries
- `logs/sprint2_*.log` — full logs
- This document

The trained gates have no production value; kept only for reproducibility.
The trainer script is reusable if we want to test other long-premium
structures later.

---

## What this changes for sprints #1 and #3

### Sprint #1 (futures-first) — proceeds
The V5 feature set has no options-direction signal, but futures direction
uses a different model (forward returns, momentum, opening range), and
futures execution is much simpler (one instrument, one entry, one exit).
Sprint #1 is now the most likely path to a real edge.

### Sprint #3 (defined-risk spreads) — deprioritized or canceled
The hypothesis behind sprint #3 was that defined-risk option structures
might survive realistic execution where naked short premium failed. But
sprint #2 shows the V5 feature set has no options-direction signal at all,
so a directional spread engine using the same features would inherit the
same problem. **Sprint #3 should not run unless sprint #1 proves there's
a real directional signal in the feature set first**, in which case spreads
can be considered as a defined-risk way to express the same signal.

---

## Decision

Move to Sprint #1: Futures-first directional engine on NIFTY.
