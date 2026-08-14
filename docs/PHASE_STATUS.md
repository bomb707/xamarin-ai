# Xamarinbot V2 - Phase Status

Tracks implementation status against the two source specs:
- `Xamarinbot_V2_Detailed_Development_Roadmap.docx` ("Roadmap")
- `Xamarinbot_V2_Detailed_Strategy_and_Mathematical_Model.docx` ("Strategy")

This build covers **Roadmap Phases 0-4** (the foundation layers the roadmap
itself says must exist before any predictive complexity, per its "Roadmap
principle" and SS22.1 "Recommended immediate next sprint"). Phases 5-14 are
**not implemented** - they are listed below as explicitly future work, not
silently dropped.

## Implemented

| Phase | Status | Notes |
|---|---|---|
| Phase 0 - Baseline Freeze and Audit | Done (reconstructed) | No V1 code existed in this repo; baseline reconstructed from Strategy doc SS1/SS3 narrative description. See "Known reconstruction gaps" below. `strategy_version`/`config_hash` stamped on every `AuditRecord` ([config.py](../src/xamarinbot/config.py), [baseline/](../src/xamarinbot/baseline/)). |
| Phase 1 - Market Data and Clock Foundation | Done (interfaces + mock; partial real) | [clock.py](../src/xamarinbot/clock.py), [feeds/base.py](../src/xamarinbot/feeds/base.py), [feeds/mock.py](../src/xamarinbot/feeds/mock.py), [feeds/freshness.py](../src/xamarinbot/feeds/freshness.py). Real adapters: see "Live adapter confidence" below. |
| Phase 2 - Causal Event Store and Replay Engine | Done | [events/](../src/xamarinbot/events/). Causality, ordering-tie-break, and disconnect/resnapshot behavior covered by `tests/test_event_causality.py`. |
| Phase 3 - Exact Portfolio Mathematics Kernel | Done | [portfolio/](../src/xamarinbot/portfolio/). 100% of Strategy-doc identities covered by property tests (`tests/test_portfolio_math.py`, Hypothesis). No dependency on predictor or exchange client (verified by import graph). |
| Phase 4 - Feature Engineering (TWAP/spot/CLOB/time) | Done | [features/](../src/xamarinbot/features/). `compute()` is a pure, stateless function of a causal event list - the same code path for live and replay by construction, and future events in the input list are filtered out internally rather than trusted from the caller. All SS5-SS7 formulas (`G_T`, `G_S`, `L`, `V_T,model`, `Z_gap`, `L_clob`, `Z_clob`, OFI) implemented; two formulas the source docs don't pin down exactly (`Z_clob`'s robust_scale estimator, realized-volatility scaling convention) are documented inline and in "Known reconstruction gaps" below. Missing/stale inputs return an explicit `InvalidFeatureState`, never a silent default. |

Supporting pieces built to make Phases 0-4 demonstrable end-to-end:
- `journal/` - SS20 schema (all entities declared; market_config, feed_event,
  portfolio_state, order_event, fill, settlement, audit, and now
  feature_state are populated).
- `synthetic/rounds.py` - a **synthetic, clearly-labeled** regression
  dataset generator standing in for Phase 0's "200+ representative historical
  rounds" (none were supplied with the source spec). Now emits a full book
  (bids and asks, not asks-only) so a true CLOB midpoint is computable, and
  periodic full resnapshots so a quiet book doesn't read as stale.
- `reports/baseline_report.py` - the Phase 0 performance report + failure
  taxonomy.
- `reports/leadlag_report.py` - the Phase 4 / SS22.1 empirical table: P(UP)
  and realized edge by gap bucket x spot direction x CLOB direction x time
  bucket.
- `scripts/run_baseline_replay.py`, `scripts/run_feature_engine_demo.py` -
  wire the above into two end-to-end causal replays you can run today.

Run them:
```
PYTHONPATH=src python scripts/run_baseline_replay.py [n_rounds]
PYTHONPATH=src python scripts/run_feature_engine_demo.py [n_rounds]
```
Tests: `PYTHONPATH=src pytest -q` (or `pip install -e .[dev]` first)

## Not yet implemented (future work)

| Phase | What it needs |
|---|---|
| Phase 5 - Probability Model and Calibration | Logistic `q` model, walk-forward splits, isotonic/Platt calibration, `ModelRegistry`. |
| Phase 6 - Seed Regime / Middle-Ground State Machine | `RegimeClassifier`, `ActionPermissionMatrix` from Strategy doc SS8's table. |
| Phase 7 - Order-Book and Execution Simulator | Depth walking, 250ms taker delay/revalidation, FAK partial fills, maker queue/fill model. **This is what today's Phase-0 demo simplifies away** - every fill in `run_baseline_replay.py` is assumed to execute completely at the decision-time limit price. |
| Phase 8 - One-Step Semi-Controlled Optimizer | Candidate generation + EV/G scoring using the Phase 3 kernel + Phase 5 `q`. |
| Phase 9 - Proactive Maker/Taker Place-Cancel-Replace | `OrderSupervisor`, cancel/replace predicates, churn rate-limiting. |
| Phase 10 - Short-Horizon MPC | Scenario tree, receding-horizon optimization, latency-bounded fallback. |
| Phase 11 - Walk-Forward Calibration and Ablations | Requires Phases 4-10 plus **real historical data** (see below). |
| Phase 12 - Shadow / Paper Trading | Requires live feeds (Phase 1's real adapters, verified) running continuously. |
| Phase 13 - Limited Live Rollout | Requires Phase 12 passing its exit gate, plus a funded, authenticated Polymarket account. **Do not attempt without explicit, deliberate operator sign-off** - this phase risks real capital. |
| Phase 14 - Adaptive Optimization | Requires an accumulated live/shadow event log to calibrate against. |

## Known reconstruction gaps (Phase 0 baseline, Phase 4 features)

The source docs describe some behaviors *narratively*, not with exact
formulas. This build's choices are documented at the point of
implementation and repeated here for visibility:

1. **Directional-lead sizing** ([baseline/strategy.py](../src/xamarinbot/baseline/strategy.py)) -
   the docs say V1 uses this but give no formula. This build adds a fixed
   `lead_size_bonus` on top of `clip` when the spot-vs-TWAP lead exceeds
   `lead_bonus_threshold_bp` and agrees with the trade direction.
2. **Failure taxonomy classification** ([reports/baseline_report.py](../src/xamarinbot/reports/baseline_report.py)) -
   the docs name all seven categories (high-price loss, late entry, false
   TWAP confirmation, reversal, stale data, missed/partial fill, oversizing)
   but not their classification rules. Heuristics are documented inline;
   `missed_partial_fill` is always 0 because partial fills aren't modeled
   until Phase 7.
3. **`ask_range_bp`'s reference point** ([baseline/strategy.py](../src/xamarinbot/baseline/strategy.py)) -
   compares the executable ask to the *current* CLOB mid (its DOWN-side
   complement for DOWN orders), not a prior-time value, since `BaselineInputs`
   has no "previous ask" field to compare against directly. Functions as a
   wide-quote/illiquidity cap rather than a temporal-range check.
4. **Z_clob's robust scale** ([features/engine.py](../src/xamarinbot/features/engine.py)) -
   Strategy doc SS7 names `robust_scale` but not its estimator; this build
   uses MAD * 1.4826 (the normal-consistent MAD-to-stdev factor), floored at
   `clob_min_robust_scale`.
5. **Realized volatility scaling convention** ([features/engine.py](../src/xamarinbot/features/engine.py)) -
   SS6 requires `sigma_t` "consistently scaled to the time unit used in
   tau" but doesn't specify how. This build uses the stdev of per-tick log
   returns each divided by `sqrt(dt)` between samples, so `sigma_t * sqrt(tau)`
   is dimensionally a standard Brownian-motion scaling in seconds.

None of these gaps affect the Phase 3 math kernel, which implements the
source docs' formulas exactly.

## Notable bugs caught during Phase 0/4 build-out (fixed, worth knowing about)

Building Phase 4 exercised the synthetic data pipeline far more heavily than
Phase 0 alone did, and surfaced three real bugs versus just calibration
issues - noted here since they affected numbers already reported to the
user in earlier sessions:

- The order book generator recomputed "the previous tick's ask price" from
  a formula that mixed the current tick's TWAP with the previous tick's
  spot, instead of tracking what was actually emitted last tick. This left
  stale price levels in the book forever, so `best_bid` could end up above
  `best_ask`. Fixed by carrying the actual previous per-level prices as
  loop state instead of recomputing them.
- The synthetic book only ever emitted ask levels, never bids, so
  `clob_mid` had to be approximated from the raw ask price - which bakes
  the spread into the "midpoint" and breaks the DOWN-side reference
  (`1 - clob_mid`) by roughly double the spread. Fixed by emitting real bid
  levels so a true `(best_bid + best_ask) / 2` midpoint is computable.
- Skipping unchanged book levels (an efficiency change) meant a
  since-static price never re-announced itself, so the freshness monitor
  correctly-but-uselessly flagged it as stale. Fixed by adding a periodic
  full resnapshot even when nothing changed - which is also literally what
  Roadmap Phase 1 asks for ("periodic/safety resnapshot").
- The logistic slope mapping TWAP-vs-spot gap to CLOB mid (0.15) saturated
  `clob_mid` to ~1.0 within the first few seconds of any trending round,
  after which `clob_direction` was stuck at 0 (uninformative) for the rest
  of the round - flattening one whole dimension of the Phase 4 lead-lag
  table. Lowered to 0.001 so mid stays graduated across the observed
  gap range.

## Live adapter confidence (Phase 1)

Endpoints/message shapes were pulled from docs.polymarket.com directly
(2026-08-13) rather than guessed, but confidence varies by adapter -
**verify against a live response before trusting any of these in
production**:

| Adapter | Confidence | Caveat |
|---|---|---|
| `feeds/polymarket_clob.py` - book snapshot (REST) + market channel (WSS) | High | Endpoint, subscribe message, and event shapes (`book`/`price_change`/`tick_size_change`) confirmed from current docs. |
| `feeds/polymarket_clob.py` - market metadata (`get_market_config`) | Medium | Gamma API endpoint/fields confirmed, but `UP`/`DOWN` token-to-outcome mapping is **not implemented** (raises `NotImplementedError`) - the fetched docs excerpt didn't confirm the outcomes field name/order, and guessing token order wrong in a trading system is worse than failing loudly. |
| `feeds/polymarket_user.py` - order/fill WSS stream | High (WSS) / Unconfirmed (REST) | Auth/subscribe/message shapes confirmed. `open_orders()`/`reconcile()`'s REST fallback path is unverified - only the WSS path was checked against current docs. |
| `feeds/chainlink_twap.py` | Low | REST/WSS base URLs and report field names came from docs.polymarket.com, but Chainlink Data Streams' actual auth/subscription handshake was not in that excerpt (it normally requires HMAC-signed requests). `auth_headers` is a pluggable, caller-supplied dict, not a working implementation. |
| `feeds/spot_composite.py` | High | Coinbase/Binance public spot-price REST endpoints are standard, stable, well-known APIs - lower integration risk than the Polymarket/Chainlink-specific pieces above. |

## Before real historical data or live trading

- Replace `synthetic/rounds.py`'s dataset with real recorded rounds before
  trusting Phase 0's baseline performance report or any later phase's
  backtest - the synthetic generator only proves the pipeline runs and
  reconciles, not that any strategy has edge.
- Resolve the `_map_tokens_to_sides` gap in `polymarket_clob.py`.
- Confirm the Chainlink Data Streams auth handshake against Chainlink's own
  docs (not Polymarket's).
- Phases 7, 12, and 13 all have hard exit gates in the Roadmap doc
  (execution-model error tolerance, live/replay parity, reconciliation with
  no unexplained ledger discrepancies) that must pass before any real
  capital is at risk - none of those gates have been exercised yet.
