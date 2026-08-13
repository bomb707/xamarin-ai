# Xamarinbot V2 - Phase Status

Tracks implementation status against the two source specs:
- `Xamarinbot_V2_Detailed_Development_Roadmap.docx` ("Roadmap")
- `Xamarinbot_V2_Detailed_Strategy_and_Mathematical_Model.docx` ("Strategy")

This build covers **Roadmap Phases 0-3** (the foundation layers the roadmap
itself says must exist before any predictive complexity, per its "Roadmap
principle" and SS22.1 "Recommended immediate next sprint"). Phases 4-14 are
**not implemented** - they are listed below as explicitly future work, not
silently dropped.

## Implemented

| Phase | Status | Notes |
|---|---|---|
| Phase 0 - Baseline Freeze and Audit | Done (reconstructed) | No V1 code existed in this repo; baseline reconstructed from Strategy doc SS1/SS3 narrative description. See "Known reconstruction gaps" below. `strategy_version`/`config_hash` stamped on every `AuditRecord` ([config.py](../src/xamarinbot/config.py), [baseline/](../src/xamarinbot/baseline/)). |
| Phase 1 - Market Data and Clock Foundation | Done (interfaces + mock; partial real) | [clock.py](../src/xamarinbot/clock.py), [feeds/base.py](../src/xamarinbot/feeds/base.py), [feeds/mock.py](../src/xamarinbot/feeds/mock.py), [feeds/freshness.py](../src/xamarinbot/feeds/freshness.py). Real adapters: see "Live adapter confidence" below. |
| Phase 2 - Causal Event Store and Replay Engine | Done | [events/](../src/xamarinbot/events/). Causality, ordering-tie-break, and disconnect/resnapshot behavior covered by `tests/test_event_causality.py`. |
| Phase 3 - Exact Portfolio Mathematics Kernel | Done | [portfolio/](../src/xamarinbot/portfolio/). 100% of Strategy-doc identities covered by property tests (`tests/test_portfolio_math.py`, Hypothesis). No dependency on predictor or exchange client (verified by import graph). |

Supporting pieces built to make Phases 0-3 demonstrable end-to-end:
- `journal/` - SS20 schema (all entities declared; market_config, feed_event,
  portfolio_state, order_event, fill, settlement, audit are populated now).
- `synthetic/rounds.py` - a **synthetic, clearly-labeled** regression
  dataset generator standing in for Phase 0's "200+ representative historical
  rounds" (none were supplied with the source spec).
- `reports/baseline_report.py` - the Phase 0 performance report + failure
  taxonomy.
- `scripts/run_baseline_replay.py` - wires all of the above into one
  end-to-end causal replay you can run today.

Run it: `PYTHONPATH=src python scripts/run_baseline_replay.py [n_rounds]`
Tests: `PYTHONPATH=src pytest -q` (or `pip install -e .[dev]` first)

## Not yet implemented (future work)

| Phase | What it needs |
|---|---|
| Phase 4 - Feature Engineering (TWAP/spot/CLOB/time) | `FeatureEngine`, `G_T`/`G_S`/`L`/`Z_gap`/OFI computation, feature_version-tagged storage. |
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

## Known reconstruction gaps (Phase 0 baseline)

The source docs describe the V1 baseline *narratively*, not with exact
formulas, for two behaviors. This build's choices are documented at the
point of implementation and repeated here for visibility:

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

Neither gap affects the Phase 3 math kernel, which implements the source
docs' formulas exactly.

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
