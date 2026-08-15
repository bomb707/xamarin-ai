# Xamarinbot V2 - Phase Status

Tracks implementation status against the two source specs:
- `Xamarinbot_V2_Detailed_Development_Roadmap.docx` ("Roadmap")
- `Xamarinbot_V2_Detailed_Strategy_and_Mathematical_Model.docx` ("Strategy")

This build covers **Roadmap Phases 0-12** (the foundation layers the
roadmap itself says must exist before any predictive complexity, per its
"Roadmap principle" and SS22.1 "Recommended immediate next sprint").
Phases 13-14 are **not implemented** - they are listed below as explicitly
future work, not silently dropped.

**Phase 12B correctness pass (see `docs/PHASE_12B_AUDIT.md`)**: an audit
of Phases 8-12 found and fixed several real bugs invalidating specific
numbers reported below - a baseline-harness bug that made the baseline
strategy trade in 0 rounds regardless of the market (two independent
copies of it, both fixed), taker candidate sizing that offered only raw
depth-level sums instead of risk/marginal-edge-aware quantities, train/
evaluation round overlap in every demo script and test fixture that
trained a model (every "held-out" evaluation was frequently evaluating on
its own training data), an uncalibrated probability model used by every
Phase 8-12 controller, a cumulative-round-spend check that only compared
one order at a time, and a hardcoded worst-price assumption
(`limit_price=1.0`) with no real protection. **Every specific number,
demo-output claim, or "finding" below dated before this pass should be
treated as historical record of what a bug produced, not current
behavior** - fixed instances are marked SUPERSEDED inline with what
changed; PHASE_12B_AUDIT.md has the complete list, root causes, and
regression tests.

## Implemented

| Phase | Status | Notes |
|---|---|---|
| Phase 0 - Baseline Freeze and Audit | Done (reconstructed) | No V1 code existed in this repo; baseline reconstructed from Strategy doc SS1/SS3 narrative description. See "Known reconstruction gaps" below. `strategy_version`/`config_hash` stamped on every `AuditRecord` ([config.py](../src/xamarinbot/config.py), [baseline/](../src/xamarinbot/baseline/)). |
| Phase 1 - Market Data and Clock Foundation | Done (interfaces + mock; partial real) | [clock.py](../src/xamarinbot/clock.py), [feeds/base.py](../src/xamarinbot/feeds/base.py), [feeds/mock.py](../src/xamarinbot/feeds/mock.py), [feeds/freshness.py](../src/xamarinbot/feeds/freshness.py). Real adapters: see "Live adapter confidence" below. |
| Phase 2 - Causal Event Store and Replay Engine | Done | [events/](../src/xamarinbot/events/). Causality, ordering-tie-break, and disconnect/resnapshot behavior covered by `tests/test_event_causality.py`. |
| Phase 3 - Exact Portfolio Mathematics Kernel | Done | [portfolio/](../src/xamarinbot/portfolio/). 100% of Strategy-doc identities covered by property tests (`tests/test_portfolio_math.py`, Hypothesis). No dependency on predictor or exchange client (verified by import graph). |
| Phase 4 - Feature Engineering (TWAP/spot/CLOB/time) | Done | [features/](../src/xamarinbot/features/). `compute()` is a pure, stateless function of a causal event list - the same code path for live and replay by construction, and future events in the input list are filtered out internally rather than trusted from the caller. All SS5-SS7 formulas (`G_T`, `G_S`, `L`, `V_T,model`, `Z_gap`, `L_clob`, `Z_clob`, OFI) implemented; two formulas the source docs don't pin down exactly (`Z_clob`'s robust_scale estimator, realized-volatility scaling convention) are documented inline and in "Known reconstruction gaps" below. Missing/stale inputs return an explicit `InvalidFeatureState`, never a silent default. |
| Phase 5 - Probability Model and Calibration | Done | [model/](../src/xamarinbot/model/). Pure-Python L2-regularized logistic regression (no numpy/sklearn dependency - see "Known reconstruction gaps"), fit on a chronological (not random) train split, calibrated on validation via Platt or isotonic, evaluated on a held-out test split with Brier/log-loss/accuracy. `ModelRegistry` freezes model + feature_version + training_window + metrics and gates promotion on a Brier threshold - matches the exit gate "No production use until calibration is acceptable." TWAP-only / spot-only / combined lead-lag models are all built from a single shared `FeatureVector` computation per decision point (not recomputed per model). |
| Phase 6 - Seed Regime / Middle-Ground State Machine | Done | [regime/](../src/xamarinbot/regime/). `classify_seed_action` is a complete, exhaustively-tested 54-cell mapping (6 `GapRegime` buckets x 3 CLOB directions x 3 spot directions) extending the Strategy doc SS8 table's 5 example rows + catch-all - see "Known reconstruction gaps" for exactly how the un-named cells were filled in. `ActionPermissionMatrix` always includes WAIT alongside any directional candidate; the module has no import dependency on the portfolio math kernel or anything order-placing (Phase 6 exit gate: "no matrix entry directly bypasses EV/risk gates" - every action here is a candidate family only). `RegimeClassifier` is stateful per round, logs every transition with dwell time, and substitutes CANCEL for WAIT specifically when a prior directional thesis just lapsed. Gated by the new `FeatureFlags.use_regime_seed_policy` flag (off by default) alongside the existing baseline policy, per "keep the original baseline rule available as a separate policy." |
| Phase 7 - Order-Book and Execution Simulator | Done | [execution/](../src/xamarinbot/execution/). `walk_depth` builds the taker cost curve over real multi-level `BookSnapshot` depth (reused from Phase 1, not reinvented) with FAK partial-fill semantics falling out naturally - whatever isn't filled by the walk simply isn't, no resting remainder. The 250ms delay model requires the *caller* to supply the book at both submission and revalidation time (from causal replay), correctly modeling "revalidation/repricing risk" without the *strategy* seeing future data - only the simulated exchange does, exactly as a real matching engine would. `OrderState` is a real lifecycle state machine (`PENDING_DELAY -> OPEN/PARTIALLY_FILLED -> FILLED/CANCELED`); a real bug here (fills were applied eagerly at submission regardless of delay, silently defeating the pending-delay-no-cancel behavior) is documented below. The maker fill-probability/adverse-selection model (`rho(p,h)`, `q_fill`) is explicitly an uncalibrated placeholder - no real fill data exists to estimate it from (Roadmap's own step: "Estimate q_fill / maker adverse selection from historical fills" - not possible yet). Stochastic maker fills reuse Phase 2's `seeded_random` for reproducibility, exactly the use case it was built for. |
| Phase 8 - One-Step Semi-Controlled Optimizer | Done | [optimizer/](../src/xamarinbot/optimizer/). `CandidateAction` matches Strategy doc SS23.3's interface almost verbatim. One EV formula (`EV_after = q*delta_U + (1-q)*delta_D - delta_C`) is used for every candidate, alpha or hedge, taker or maker - it reduces exactly to SS13's `DeltaEV_U`/`DeltaEV_D` for a certain taker fill and to SS14's maker EV core term once weighted by fill probability and evaluated with `q_fill` (Phase 7's adverse-selection adjustment) instead of raw `q`, rather than several ad hoc formulas per candidate type. Hard-constraint rejection reuses Phase 3's `evaluate_constraints` directly (not reimplemented) via a real `FillSimulationResult`; maker candidates are checked against their *if-filled* portfolio, conservatively, per SS16's "projected fill would push G below G_min -> Cancel/shrink." WAIT is always present and always valid, the guaranteed fallback. **Both roadmap-named verification items are automated**: `tests/test_optimizer.py`'s Hypothesis stress test (200 random portfolio/book/q combinations) asserts the chosen candidate never has `g_after` below `g_min` (accounting for the edge case where the *starting* portfolio is already below an unreachable `g_min` - no action, including WAIT, can fix that; see the test), and `scripts/run_one_step_controller_demo.py` runs the controller against the Phase 0 baseline over an identical causal replay. |
| Phase 9 - Proactive Maker/Taker Place-Cancel-Replace | Done | [supervisor/](../src/xamarinbot/supervisor/). `OrderSupervisor` is a thin policy layer over Phase 7's `OrderState` mechanics: it never computes EV/G itself (the caller re-evaluates each tracked order via Phase 8's `evaluate_maker_candidate` and passes the numbers in), keeping it a single, testable place implementing Strategy doc SS16's trigger table (edge failure, regime flip, risk breach, time compression, feed staleness -> cancel; book displacement -> replace, gated by a churn threshold) rather than duplicating EV math a second time. **All four roadmap-named verification scenarios are direct tests**: rapid regime flip (plus its rate-limiting counterpart - a *second* flip inside the minimum action interval must not thrash), partial fill then cancel (the partial fill survives, only the remainder is canceled), tick-size change with an open order (a `REPLACE` after a tick-size change prices on the new grid, not the stale one), and feed-stale-while-resting (cancels regardless of how good the order otherwise looks - checked first, before every other trigger). Cancel-regret analytics (`reports/supervisor_report.py`) are explicitly post-hoc-only, using a documented approximation (did the touch price cross the canceled order's price afterward) rather than a full counterfactual fill simulation. |
| Phase 10 - Short-Horizon MPC | Done | [mpc/](../src/xamarinbot/mpc/). `MPCController` is a generalization of Phase 8's `OneStepController`, not a parallel implementation - at `horizon_steps<=1` it must (and does, exactly, bit-for-bit) reduce to a plain one-step decision. Continuation value at each hypothetical future state is estimated by recursively calling the greedy one-step policy itself (a standard rollout-policy approximation), over a small discrete scenario tree that evolves only `GapRegime` (not the full 54-state `RegimeState`) using a transition model estimated from Phase 6's own `RegimeTransition` records over a causal replay - "historical state transitions," synthetic for now. **All three roadmap-named verification items are direct tests**: degenerate-horizon equivalence (checked across several states/portfolios, not just one), timeout fallback (a zero time budget forces the plain one-step decision), and scenario-probability sanity checks (every state's next-state distribution sums to 1.0 and is bounded in [0,1], including the top-k-truncation-then-renormalize path). A real bug turned up mid-build: continuation projections initially reused the *undepleted* original book at every step, so a large taker fill's continuation value double-counted the exact liquidity it had just consumed - fixed by depleting the relevant side's book by the chosen candidate's fill before recursing, with a dedicated regression test. |
| Phase 11 - Walk-Forward Calibration and Ablations | Done (synthetic data) | [walkforward/](../src/xamarinbot/walkforward/). `windows.py` builds rolling train/validate/test splits that are chronologically disjoint *within* each window (a smaller-than-default `step` can legitimately reuse a round as one window's test and a later window's train/validate - documented in the module and covered by a test that checks this doesn't cross a single window's own ordering); `bootstrap.py` gives resampled confidence intervals on PnL reusing Phase 2's `seeded_random` for reproducibility; `ablations.py` runs the **exact 8 mandatory ablations from Strategy doc SS20.1** over the identical causal replay harness, requiring two real additions to the optimizer that Phases 8-10 never needed: SS17 hedge-candidate generation (`generate_hedge_candidate`, gated by `enable_portfolio_repair`) and SS18's actual risk-adjusted selection objective (`lambda_g`, since pure `ev_after` ranking can *never* select a hedge - SS17 hedges have negative standalone EV by construction and always lose to WAIT at `lambda_g=0`). `sensitivity.py` sweeps one `OneStepConfig` field at a time and reports the *full* curve, not just the argmax, plus a `parameter_stability_across_windows` check (does the best value drift between walk-forward windows - instability there is itself the finding). **Both roadmap-named verification items are automated**: "no test-period tuning" is a direct test that spies on every round_id `sweep_parameter` is ever handed and asserts none belong to any window's `test_round_ids`; bootstrap CI correctness is covered by dedicated tests (point estimate = plain mean, lower<=upper, deterministic per seed key, degenerate at n<=1, widens with variance). Ablations 6-8 report **zero non-wait actions** on this synthetic dataset - two distinct, diagnosed-not-guessed findings, not bugs, pinned down by a regression test and detailed in "Notable bugs" below. |
| Phase 12 - Shadow / Paper Trading | Done (synthetic data) | [shadow/](../src/xamarinbot/shadow/). `ShadowRunner` reuses Phase 1's exact pluggable feed interfaces (mock/replay today; real adapters once credentials exist), so this is the same code that would run against a live stream, not a separate implementation to swap in later. The one genuinely new mechanism: `MockFeedCursor` gained a `time_attr` field (`feeds/mock.py`) so a cursor can gate causal visibility on `recv_ts` (true wire-arrival time) instead of Phase 2's `event_time` (source_ts-preferring, and therefore mildly optimistic relative to when data actually arrived) - `ShadowRunner` is the one place in the codebase that must use the stricter gate, since "no timestamp leakage" for a genuinely live system means never acting on data before it's actually arrived. SS21's Latency gate ("Controller misses deadline -> Fallback to simpler one-step/WAIT policy") is a direct per-decision check (`decide_elapsed_ms > decision_deadline_ms` forces WAIT); 24/7 reconnect stability is exercised via `FaultInjector`, a demo/test-only simulated outage that the runner must survive without crashing or wedging. `shadow/parity.py`'s `compare_live_vs_replay` isolates exactly the causal-gate difference by evaluating both decision streams against an identical frozen `PortfolioState()` (matching Phase 8-10's own demo convention), so a fill-timing divergence downstream can never masquerade as a causal-gating mismatch upstream. **All roadmap-named verification items are automated**: no-timestamp-leakage (a hand-crafted event with `source_ts < recv_ts` proven invisible before its true arrival), 24/7 reconnect stability (outage injection mid-round, runner completes and resumes), controller-deadline-met (a 0ms deadline forces every decision to WAIT; a 10s deadline never misses). The live-vs-replay parity rate on this synthetic dataset is ~96% - not a bug, but a real, fully-diagnosed finding about feature-parameter fragility under a genuinely live gate; see the dedicated finding below. |

Supporting pieces built to make Phases 0-10 demonstrable end-to-end:
- `journal/` - SS20 schema (all entities declared; market_config, feed_event,
  portfolio_state, order_event, fill, settlement, audit, and now
  feature_state are populated).
- `synthetic/rounds.py` - a **synthetic, clearly-labeled** regression
  dataset generator standing in for Phase 0's "200+ representative historical
  rounds" (none were supplied with the source spec). Emits a full book
  (bids and asks, not asks-only), periodic full resnapshots so a quiet book
  doesn't read as stale, and now 3 depth levels per side (the original
  single touch level plus two more, moving in lockstep so best_bid/best_ask
  and everything computed from them are unaffected) so depth-walking has
  real depth to walk.
- `reports/baseline_report.py` - the Phase 0 performance report + failure
  taxonomy.
- `reports/leadlag_report.py` - the Phase 4 / SS22.1 empirical table: P(UP)
  and realized edge by gap bucket x spot direction x CLOB direction x time
  bucket.
- `reports/regime_report.py` - the Phase 6 transition-statistics report:
  seed-action counts, top transition pairs, average dwell time per state.
- `reports/execution_report.py` - the Phase 7 taker slippage/delay report.
- `reports/supervisor_report.py` - the Phase 9 cancel/replace analytics
  (action/reason counts, cancel regret rate).
- `reports/mpc_report.py` - the Phase 10 controller latency benchmark
  (p50/p95/p99/max, fallback rate).
- `reports/walkforward_report.py` - the Phase 11 ablation matrix (mean PnL +
  bootstrap CI + fill rate per ablation) and sensitivity/stability table
  formatting.
- `reports/shadow_report.py` - the Phase 12 daily shadow report (decision
  latency percentiles, reconnects, missed deadlines, mean hypothetical G)
  and live-vs-replay parity report formatting.
- `scripts/dev_synthetic/run_synthetic_baseline_replay.py`,
  `scripts/run_model_training_demo.py`, `scripts/run_regime_classifier_demo.py`,
  `scripts/run_execution_simulator_demo.py`, `scripts/run_one_step_controller_demo.py`,
  `scripts/run_order_supervisor_demo.py`, `scripts/run_mpc_controller_demo.py`,
  `scripts/run_walk_forward_ablation_demo.py`, `scripts/run_shadow_demo.py` -
  wire the above into ten end-to-end causal pipelines you can run today.
  `run_baseline_replay.py` still simplifies fills to same-tick-at-decision-
  price - it predates Phase 7 and was never retrofitted with
  `ExecutionSimulator`, unlike `run_one_step_controller_demo.py` and
  `run_order_supervisor_demo.py`, which use it properly for both taker
  depth-walking and stochastic maker fills.

Run them:
```
PYTHONPATH=src python scripts/run_baseline_replay.py [n_rounds]
PYTHONPATH=src python scripts/run_feature_engine_demo.py [n_rounds]
PYTHONPATH=src python scripts/run_model_training_demo.py [n_rounds]
PYTHONPATH=src python scripts/run_regime_classifier_demo.py [n_rounds]
PYTHONPATH=src python scripts/run_execution_simulator_demo.py [n_rounds]
PYTHONPATH=src python scripts/run_one_step_controller_demo.py [n_eval_rounds]
PYTHONPATH=src python scripts/run_order_supervisor_demo.py [n_eval_rounds]
PYTHONPATH=src python scripts/run_mpc_controller_demo.py [n_eval_rounds]
PYTHONPATH=src python scripts/run_walk_forward_ablation_demo.py [n_rounds]
PYTHONPATH=src python scripts/dev_synthetic/run_synthetic_shadow_demo.py [n_rounds]
```
Tests: `PYTHONPATH=src pytest -q` (or `pip install -e .[dev]` first)

## Phase 12C - Real Market Recorder and Event-Level Shadow Foundation

**This is where the project changes character**: it stops designing around
synthetic behavior and starts measuring the actual market. Tranche 2.2 was
the final synthetic correctness closure; no further synthetic strategy
optimization is performed, and no real orders are placed.

| Area | Status | Notes |
|---|---|---|
| Market discovery + metadata (item 2) | Done, **live-verified** | [realtime/discovery.py](../src/xamarinbot/realtime/discovery.py). Dynamic per-round discovery of the `btc-up-or-down-5m` series; nothing hardcoded but the series slug pattern. Persists event/market id, condition_id, slug, question, rules, resolution source, start/end, UP/DOWN token ids, tick size, min order size, fee configuration, taker delay, and both raw payloads. |
| CLOB market WebSocket adapter (item 3) | Done, **live-verified** | [realtime/clob_ws.py](../src/xamarinbot/realtime/clob_ws.py) + [feeds/polymarket_clob.py](../src/xamarinbot/feeds/polymarket_clob.py). Real in-memory book from the WS; REST only for bootstrap/resync/integrity. Fixed the `price_change` routing bug that silently discarded every book delta. |
| RTDS reference feeds (item 4) | Done, **live-verified** | [realtime/rtds.py](../src/xamarinbot/realtime/rtds.py). One shared connection for Chainlink BTC/USD, Binance BTCUSDT, and Chainlink 30s/60s TWAP. No credentials. Chainlink observation time, RTDS publisher time, local wall and local monotonic receive times all preserved separately. |
| Raw recorder layer (item 6) | Done | [realtime/raw_events.py](../src/xamarinbot/realtime/raw_events.py), [raw_store.py](../src/xamarinbot/realtime/raw_store.py), [recorder.py](../src/xamarinbot/realtime/recorder.py). Immutable append-only log, integer-nanosecond timestamps, verbatim wire payloads, bounded async queue + batched WAL writer. A dropped event disqualifies the interval for training. |
| Pre-round history (item 7) | Done | `PRE_ROUND` lifecycle state, default 420s lead. Pre-round momentum/TWAP history comes from real observations, never approximated from `p0`. |
| Round lifecycle (item 8) | Done | [realtime/lifecycle.py](../src/xamarinbot/realtime/lifecycle.py). DISCOVERED -> PRE_ROUND -> ACTIVE -> ENDED -> RESOLVED -> FINALIZED. |
| Independent label reconstruction (items 5, 8) | Done | [realtime/label.py](../src/xamarinbot/realtime/label.py). Reconstructs under BOTH the market's declared settlement basis and the plain-Chainlink-reference basis, and compares each to Polymarket's own resolution. **See "Settlement rule discrepancy" below.** |
| ShadowRunner tau/sigma (item 9) | Done | `tau=fv.tau, sigma=fv.realized_vol` passed explicitly; dynamic maker mode can no longer silently run with `tau=None, sigma=0` while real values exist. |
| Real freshness (item 10) | Done | [realtime/freshness.py](../src/xamarinbot/realtime/freshness.py). Replaces the hardcoded `is_fresh=True`. Computed from source timestamps. Stale/missing input -> no new ALPHA, resting orders reviewed against an explicitly stale view, explicit reason recorded. |
| Maker-fill counterfactuals (item 11) | Done | [realtime/counterfactual.py](../src/xamarinbot/realtime/counterfactual.py). Records the evidence needed to LATER estimate `rho` and `q_fill` from real data. **No fill probability is fitted in this phase.** |
| Replacement provenance (item 12) | Done | `ReplacementPlan` carries its own `g_after_if_fill`/`fair_value`/`q`; the replacement `TrackedOrder` no longer inherits the canceled order's stale thesis. |
| Own-order event ordering (item 13) | Done | [events/types.py](../src/xamarinbot/events/types.py)'s `causal_sort`. A FILL/CANCEL/ORDER_STATUS can never precede its own ORDER_SUBMIT, and no causal fact is fabricated between two external events sharing a timestamp. |
| Authenticated trading (item 14) | **Deliberately not implemented** | No private key, no order, cancel or replacement is ever sent. Recorder + paper shadow only. |
| Real integration capture (item 15) | Done | See [REAL_RECORDER_STATUS.md](REAL_RECORDER_STATUS.md) for the captured sample and its integrity verdict. |

Docs: [REAL_RECORDER_ARCHITECTURE.md](REAL_RECORDER_ARCHITECTURE.md) (design
+ the full live-verification log), [REAL_DATA_SCHEMA.md](REAL_DATA_SCHEMA.md)
(recorder schema), [REAL_RECORDER_STATUS.md](REAL_RECORDER_STATUS.md)
(capture results).

```
# inspect the live market metadata for the next rounds (read-only, no capture)
PYTHONPATH=src python scripts/run_market_discovery.py --rounds 2

# capture N consecutive real rounds (NO ORDERS ARE PLACED)
PYTHONPATH=src python scripts/run_real_recorder.py --rounds 3 --db captures/run1.db

# fill in venue resolutions + label agreement afterwards (settlement lands
# 3-8 min after a round closes, long after the recorder has finalized it)
PYTHONPATH=src python scripts/resolve_capture_labels.py captures/run1.db
```

`run_real_recorder.py` exits non-zero when the capture is not training-grade
(any dropped event, parse failure, or book-integrity mismatch), so a cron or
CI invocation surfaces a bad capture instead of silently banking it.

### Settlement rule discrepancy (unresolved by design, pending the approval gate)

The Phase 12C brief states the BTC five-minute rule as
`UP <=> ChainlinkBTC_end >= ChainlinkBTC_start` against the **plain
Chainlink reference price**, with TWAP as a predictive feature only.

Every live BTC 5-minute market inspected on 2026-08-15 declares otherwise:

```
"cryptoMarketConfigId": "btc-5m-twap-60",
"cryptoMarketConfig": {"twapEnabled": true, "twapLookbackSeconds": 60},
"resolutionSource": "https://data.chain.link/streams/btc-usd-twap-60s-streams"
```

with a rules description that resolves on the Chainlink **TWAP** over the
window. The recorder does not choose between these readings. It
reconstructs the label under both bases every round and reports both
agreement rates, so which basis actually reproduces Polymarket's
resolutions is an empirical result of the capture rather than an assumption.
Binance is never used as a settlement basis under either reading.

## Phase 12C.1 - Production Data Provenance & Runtime Market Constraints

Separates REAL / REAL_REPLAY / SYNTHETIC data paths and removes ambiguous
mock/default market assumptions from the production path. Full detail in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).

Two invariants are now structurally enforced rather than conventionally
observed:

- **Production/live code can never accidentally consume synthetic data** -
  `tests/test_import_boundaries.py` AST-parses every shipped module and
  fails on the import statement.
- **All executable market constraints come from the current Polymarket
  market** - `MarketConstraints` is built per round from that market's own
  metadata; `OneStepConfig.taker_min_size = 1.0` is deleted.

| Change | Detail |
|---|---|
| `DataProvenance` | `REAL_LIVE` / `REAL_REPLAY` / `SYNTHETIC_TEST`, carried on `EventStore`, `RoundLabel`, `MarketConstraints`, `ShadowRoundResult`; **defaults to SYNTHETIC_TEST so the system fails closed** |
| `feeds/mock.py` -> `replay/feeds.py` | `Mock*` -> `Replay*`. The module fabricates nothing - it replays a recorded `EventStore`, including real captures |
| `synthetic/` -> `devtools/synthetic/` | the data fabricator leaves the shipped package entirely |
| 10 demo scripts -> `scripts/dev_synthetic/run_synthetic_*` | `scripts/` now holds only real-market entry points |
| `feeds/polymarket_clob.py` -> `realtime/feed_adapter.py` | `realtime/*` is the canonical real-market SSOT |
| `feeds/chainlink_twap.py`, `feeds/spot_composite.py` | **deleted** - superseded by `realtime/rtds.py`; zero callers |
| `feeds/polymarket_user.py` | **deprecated, retained** - no `realtime/` equivalent; Phase 13 needs it |
| `replay/projection.py` | **REAL raw -> normalized event projection** (the highest-priority gap): a captured round is now consumable by the feature engine |
| `market/constraints.py` | runtime `MarketConstraints`; `min_order_shares` enforced for every order purpose at one point (`_finalize`) |
| `market/order_request.py` | BUY dollar-encoding translation boundary; **submits nothing** |
| `q = 0.5` | removed from the real path - `MODEL_UNAVAILABLE` -> NO_ALPHA |
| `draw_maker_fill` | raises on `REAL_*`; real expiries are recorded UNRESOLVED |
| `REPLAY_FRESHNESS_POLICY` | renamed `SYNTHETIC_REPLAY_FRESHNESS_POLICY`; new `REAL_REPLAY_FRESHNESS_POLICY` from measured real cadences |
| `LabelStatus` | `LABEL_AMBIGUOUS` rounds are excluded from training |

### The static minimum was wrong, not merely un-wired

`OneStepConfig.taker_min_size = 1.0` was the production minimum-order
fallback. Every BTC five-minute market sampled in the Phase 12C captures
reports **`min_order_size = 5.0` SHARES** and `tick_size = 0.01`, so the
static value was wrong by 5x in the direction that generates orders the
venue rejects. `MarketConfig.min_order_size` already carried the real value
from discovery and dead-ended - nothing in the optimizer or execution layer
read it.

`generate_hedge_candidate` and the fixed-quantity maker path checked **no**
minimum at all; enforcement now happens once, in `_finalize`, which every
candidate of every purpose flows through.

### REAL_REPLAY acceptance evidence

`scripts/run_real_replay_smoke.py captures/phase12c_verify.db` projects the
verified capture (round `btc-updown-5m-1786777800`):

```
MARKET_CONFIG 1  TWAP 793  SPOT 833  BOOK_SNAPSHOT 3144  BOOK_DELTA 177570
settlement basis   rtds_twap_60 (from the market's own metadata)
p0                 63010.53498847699 (real observation at/before the open)
min_order_shares   5.0     tick_size 0.01     fee 0.07
provenance         REAL_REPLAY, 182342/182342 events, 0 synthetic
FeatureVectors     23 VALID, 8 STALE_BOOK
```

**Finding**: the 8 `STALE_BOOK` points are the last ~76 seconds of the
round. The UP token's ASK side empties entirely at t~223s once the mid
reaches 0.990 and the outcome is effectively decided, so `clob_mid` is
genuinely undefined there. Missing real data correctly stays missing rather
than being interpolated.

### Phase 12C.2 - final real-replay correctness closure

Four fabrications removed. Each produced a plausible replay that was quietly
wrong in a way no aggregate metric would have exposed.

| # | Was | Now |
|---|---|---|
| 1 | `ShadowRunner`/`OneStepController`/`TradingSession` took `FeeConfig`/`ExecutionConfig` independently of the market, so `Fee_sim != Fee_market` and `Delay_sim != Delay_market` were both possible | `reconcile_execution_state()` DERIVES both from the round's `MarketConstraints` on REAL data; a contradicting caller value raises `ExecutionStateConflict` |
| 2 | one immutable tick for the whole replay; `tick_size_change` was dropped as "no normalized event" | projected as a later `MARKET_CONFIG`; `tick(t) = latestRecordedTickVisibleByDecisionTime`, re-read per decision. **The canonical capture really does change 0.01 -> 0.001 at t=+280.2s** |
| 3 | `config_ts = min(earliest, start_ts) - 1e-6`; `SETTLEMENT(source_ts=end_ts, recv_ts=end_ts)` | MARKET_CONFIG uses the real `market_metadata_discovered` recv time with `source_ts=None`; SETTLEMENT is off by default and the label rides `RoundLabel`. **The outcome was learned +91.6s after the close** - that stamp was 92s of foreknowledge |
| 4 | `row.get("settlement_kind") or "chainlink_reference"`, and an implicit `settlement_kind="chainlink_twap"` default | required `MarketConfig` field, fail-closed end to end |
| 5 | hand-maintained `REAL_SCRIPTS` tuple, already two scripts stale | auto-discovers every top-level `scripts/*.py` |

## Not yet implemented (future work)

| Phase | What it needs |
|---|---|
| Phase 13 - Limited Live Rollout | Requires Phase 12 passing its exit gate, plus a funded, authenticated Polymarket account. **Do not attempt without explicit, deliberate operator sign-off** - this phase risks real capital. |
| Phase 14 - Adaptive Optimization | Requires an accumulated live/shadow event log to calibrate against. |

## Known reconstruction gaps (Phase 0 baseline, Phase 4 features, Phase 5 model, Phase 6 regime, Phase 7 execution, Phase 8 optimizer, Phase 9 supervisor, Phase 10 MPC, Phase 11 walk-forward/ablations, Phase 12 shadow)

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
6. **Z_spot** ([features/engine.py](../src/xamarinbot/features/engine.py)) -
   Strategy doc SS9's logit formula names `Z_spot` but SS9 never defines
   it (only `Z_gap` and `Z_clob` get explicit formulas elsewhere). This
   build uses the same vol*sqrt(time) normalization convention as `Z_gap`,
   applied to the spot log-return at `FeatureConfig.canonical_horizon_s`.
7. **No ML library dependency** ([model/logistic.py](../src/xamarinbot/model/logistic.py)) -
   the Roadmap doesn't mandate pure-Python, but this project has stayed
   dependency-free outside the optional `live` extra; logistic regression
   at this data scale (thousands of rows, under a dozen features) doesn't
   need numpy/scikit-learn. Uses proximal-gradient (not plain gradient
   descent) for the L2 term specifically because the naive explicit-step
   version diverges for `lr * l2 >= 2` - see "Notable bugs" below.
8. **Platt over isotonic in the demo script** ([scripts/run_model_training_demo.py](../scripts/run_model_training_demo.py)) -
   both are implemented and tested, but the synthetic data is close to
   deterministically separable, so isotonic's PAVA pools almost nothing on
   an 888-row validation split (882 near-unique blocks observed) and
   memorizes it instead of learning a smooth curve. Platt's 2-parameter
   fit doesn't have that failure mode at this data scale. Real historical
   data with more genuine label noise may make isotonic viable.
9. **The 54-cell action matrix** ([regime/matrix.py](../src/xamarinbot/regime/matrix.py)) -
   Strategy doc SS8's table gives 5 example rows (synchronized bullish/
   bearish, CLOB-pullback maker cases, and one near-center reversal case)
   plus a catch-all "any region, conflict/stale -> WAIT/CANCEL", not all 54
   combinations of 6 `GapRegime` buckets x 3 CLOB directions x 3 spot
   directions. The un-named cells are filled in by a documented, symmetric
   extension of the table's own stated logic (mirroring the near-center
   reversal case to the negative side; WAIT for any FLAT leg; WAIT for
   synchronized fast signals that strongly oppose the gap regime; WAIT for
   three-way conflicts) - see the module docstring for the full reasoning,
   since this is this build's biggest interpretive gap-fill so far.
10. **GapRegime bucket boundaries** ([regime/matrix.py](../src/xamarinbot/regime/matrix.py)) -
    SS8's table names only 3 regions (positive/upper-middle, near-center,
    negative/lower-middle) but Roadmap Phase 6's verification step asks to
    "replay transitions around ±1/±0.5/0 seeds," implying finer breakpoints
    than the table's 3 named regions use. This build splits each side at
    ±1 into a "middle" and "strong" sub-bucket (6 buckets total) so every
    named breakpoint is an actual state boundary, with the strong buckets
    behaving identically to their middle counterpart wherever SS8 doesn't
    distinguish further.
11. **Maker fill-probability model** ([execution/maker.py](../src/xamarinbot/execution/maker.py)) -
    Strategy doc SS14 defines `rho(p,h)` conceptually ("distance to touch,
    queue depth, trade flow, state and time") but gives no formula, and the
    Roadmap step to estimate it from historical fills can't be done without
    real fill data. This build uses a constant-hazard-rate model
    (`rho = 1 - exp(-lambda*h)`) with `lambda` decaying exponentially in
    distance-to-touch and damped by queue-ahead - a standard, simple shape,
    but its parameters (`base_fill_rate_per_s`, `distance_decay_per_tick`,
    `queue_normalization_shares`) are uncalibrated placeholders. Same for
    `adverse_selection_bp` in the `q_fill` adjustment.
12. **Taker-order design: caller supplies both book snapshots** ([execution/taker.py](../src/xamarinbot/execution/taker.py)) -
    `simulate_taker_order` takes `asks_at_submission` and, for delayed
    orders, `asks_at_revalidation` as separate caller-supplied arguments
    rather than looking either up itself. This keeps the module decoupled
    from event-store/cursor plumbing (reusing the existing `MockBookFeed` +
    `MockFeedCursor` pattern already established in earlier scripts), but
    it does mean the caller - not this module - is responsible for
    correctly querying the book at `submit_ts + delay`, which is a
    real "the exchange sees data the strategy didn't decide with" case,
    not a causality violation - see the module docstring.
13. **One EV formula for every candidate** ([optimizer/candidates.py](../src/xamarinbot/optimizer/candidates.py)) -
    SS13/SS14 give separate-looking formulas for taker alpha EV and maker
    EV; this build uses one general delta-EV formula
    (`q*delta_U + (1-q)*delta_D - delta_C`) that both are special cases of
    (see the module docstring for the derivation), rather than implementing
    them as visibly-different code paths. This is a design choice, not a
    literal reading of the docs, made because it's provably equivalent and
    keeps alpha/hedge/taker/maker candidates on one consistent footing.
14. **`edge_min` and maker candidate placement** ([optimizer/config.py](../src/xamarinbot/optimizer/config.py), [optimizer/candidates.py](../src/xamarinbot/optimizer/candidates.py)) -
    SS21 names `edge_min` without a formula; this build applies it as a
    flat floor on `ev_after`. Maker candidates' `queue_ahead_shares` (an
    input to Phase 7's already-uncalibrated fill-probability model) uses
    the resting size at that price level when placing exactly at the
    current touch, and 0 for any price the book doesn't already show a
    level at - a simplification, not a queue model.
15. **Trigger priority order and `churn_threshold`/rate-limit values** ([supervisor/supervisor.py](../src/xamarinbot/supervisor/supervisor.py), [supervisor/config.py](../src/xamarinbot/supervisor/config.py)) -
    SS16's table doesn't specify what happens when multiple triggers fire
    at once; this build checks feed-staleness and risk-breach first (hard
    safety gates), then regime flip, edge failure, time compression, and
    finally book displacement (replace) - a defensible but not
    doc-mandated ordering. `churn_threshold` and `min_action_interval_s`
    are named by the roadmap ("Replace only when the new expected value
    exceeds the old order by a churn threshold," "Rate-limit cancel/
    replace") without values or formulas; both are flat, uncalibrated
    placeholders here.
16. **Cancel regret as a crossing-price proxy** ([reports/supervisor_report.py](../src/xamarinbot/reports/supervisor_report.py)) -
    "canceled orders that would have filled profitably" would need a full
    counterfactual fill simulation (queue position, competing orders) to
    answer precisely; this build approximates it as "did the touch price
    cross the canceled order's price within a lookback window afterward,"
    documented in the module docstring as a proxy, not a fill simulation.
17. **Scenario tree evolves GapRegime only, not full state** ([mpc/scenario.py](../src/xamarinbot/mpc/scenario.py)) -
    SS15's `X_(t+dt) = F(X_t, A_t, W_t)` is a general stochastic state
    transition; this build's "small discrete scenario tree" (the Roadmap's
    own phrasing) only evolves the 6-state `GapRegime`, holding CLOB/spot
    direction, the order book's price levels, and `q` fixed across
    continuation steps (the book's *depth* does deplete with consumption -
    see the bug below - but its prices don't move). A full 54-state
    `RegimeState` chain, or a book-price evolution model, would need far
    more transition data than a dataset this size can estimate reliably.
18. **Continuation policy is the greedy one-step policy, not a second
    search** ([mpc/controller.py](../src/xamarinbot/mpc/controller.py)) -
    a standard rollout-policy approximation for the "optimize a short
    action sequence" step, not literal backward-induction dynamic
    programming over all action sequences; explicitly what "Recommended
    first implementation: one-step / short-horizon MPC with discrete
    candidate actions" describes rather than a fuller search.
19. **Maker continuation uses the if-filled portfolio** ([mpc/controller.py](../src/xamarinbot/mpc/controller.py)) -
    same simplification as Phase 8's maker candidates (#14 above),
    inherited here since continuation reuses Phase 8's evaluation; the
    immediate EV used for candidate *selection* still correctly
    probability-weights the fill.
20. **`lambda_g` is a single scalar weight on `G_after`, not SS18's full
    `J = E[PnL_T] + lambda_G*G_T - lambda_slip*Slippage - ...` objective**
    ([optimizer/config.py](../src/xamarinbot/optimizer/config.py)) - the
    source doc names several other penalty terms (`Slippage`, and others
    left unformalized) that this build doesn't have data to calibrate
    (no real slippage history exists yet); `churn_penalty` and
    `opportunity_cost` (Phase 8) already cover the two SS18 terms that
    *were* formularizable. Extending `J` with the remaining terms is
    future work once real fill data exists to fit them against, not a
    silent omission.
21. **Ablation feature-set assignment for #2/#3/#4** ([walkforward/ablations.py](../src/xamarinbot/walkforward/ablations.py)) -
    SS20.1 names the three model-only ablations by description
    ("TWAP-only," "current-BTC-only," "TWAP + current-BTC lead-lag") but
    not by exact `FeatureSet` composition; this build maps them onto
    Phase 5's existing `TWAP_ONLY`/`SPOT_ONLY`/`COMBINED_LEAD_LAG` sets
    (the same three the Phase 5 model-comparison report already used) plus
    a new `LEAD_LAG_ONLY` set (`z_gap`, `lead_gap_bp`, `tau` and their
    `tau` interactions - deliberately excluding CLOB/OFI features so #4 is
    a genuine "TWAP + current-BTC" ablation distinct from #5's full
    `COMBINED_LEAD_LAG`, which also carries `z_clob`/`ofi`).
22. **Paper executor reuses Phase 7's `ExecutionSimulator` unmodified**
    ([shadow/runner.py](../src/xamarinbot/shadow/runner.py)) - the Roadmap
    asks for "a parallel paper executor with realistic delay/queue
    assumptions"; rather than build a second execution model, `ShadowRunner`
    calls the exact same `submit_maker_order`/`draw_maker_fill`/taker-fee
    machinery Phases 8-11 already use, so "realistic" here means "as
    realistic as Phase 7's existing simulator," not a new, separately-
    calibrated live-specific model.
23. **What counts as a live "reconnect"** ([shadow/runner.py](../src/xamarinbot/shadow/runner.py)) -
    the mock feeds have no real socket to lose, so `FaultInjector` stands
    in for a genuine outage by making the runner skip a decision point
    entirely and call `reconnect()` (a no-op on every mock adapter, a real
    reconnect handshake on the real ones) - this exercises the runner's
    survive-and-resume control flow, not real network-failure behavior,
    which can't be tested without live credentials.

None of these gaps affect the Phase 3 math kernel, which implements the
source docs' formulas exactly.

**Phase 10 demo finding worth flagging**: in `run_mpc_controller_demo.py`,
MPC chose differently than plain one-step in 0/888 sampled decisions. This
isn't the mechanism being inert - unit tests directly construct scenarios
where it changes the ranking (`test_nonzero_churn_penalty_favors_consolidating_into_fewer_actions`)
- it's that real decision points in this dataset almost always offer
either zero or exactly two non-WAIT alternatives (a histogram check found
835/888 decisions had only WAIT valid, 53/888 had exactly WAIT plus two
MAKER price offsets), and those two candidates have a monotonic EV
relationship that a mostly-self-persisting regime (~90-95% per the
transition model) rarely reorders. A richer candidate set - more price/
quantity options, or real data where regimes don't correlate so strongly
with a single dominant signal - would be needed to see organic divergence
in the demo itself, not just in a targeted unit test.

**Phase 9 demo finding worth flagging**: in `run_order_supervisor_demo.py`,
100% of placed maker orders get canceled via `REGIME_FLIP` before ever
reaching their TTL, across every heartbeat tested (10s and 5s both gave
the same result) - no order ever survives to fill or expire naturally.
This isn't a supervisor bug; it's the direct, compounding consequence of
two things already documented above: regime dwell times are typically
1-3s (Phase 6), so almost any sampling interval will catch at least one
flip before a maker order's ~10s horizon elapses, and the supervisor
correctly cancels immediately per SS16 ("Regime flip ... -> Cancel
immediately") with no hesitation or grace period. It's a legitimate,
sharp demonstration that the trigger fires exactly as specified - but it
also means this demo can't yet show a REPLACE or a natural fill/expiry in
its own output (both are exercised directly in `tests/test_supervisor.py`
instead, with hand-constructed scenarios). Real historical data - where
regimes presumably don't flip on every single tick - would be needed to
see a more varied trigger mix in the demo itself.

**SUPERSEDED by Phase 12B Tranche 1 - see `docs/PHASE_12B_AUDIT.md`.**
~~Phase 8 demo finding worth flagging: in `run_one_step_controller_demo.py`,
the Phase 0 baseline strategy took 0 positions across the sampled rounds
while the one-step controller took 19 actions~~ - **this "0 positions" was
never a cadence effect; it was `run_one_step_controller_demo.py`'s own
local baseline runner reimplementing the same `spot_prev`-always-equal-to-
`spot` bug found and fixed in `walkforward/ablations.py` (Phase 12B audit
items 3/D), which makes unanimity impossible regardless of heartbeat.**
Fixed identically (shared `elapsed_t()` helper,
`ExecutionSimulator.execute_taker`); after the fix, the same demo config
shows the baseline taking a position in 4/4 sampled rounds. The heartbeat-
starves-unanimity mechanism described in the original paragraph below is
real and worth knowing, but it is not what produced the "0 positions"
number this finding was built around - that number was a bug artifact.
Kept for the record, not as a current claim: ~~the demo evaluates both at
a 10s heartbeat ... CLOB direction is already noisy at 1s resolution
(Phase 6's finding), and sampling even more sparsely makes three-way
alignment rarer still.~~ Real historical data and a matched cadence would
still be needed before any baseline-vs-one-step comparison is evidence of
relative strategy quality, not just this specific number.

**Phase 5 demo finding worth flagging**: at `n_rounds=80`, the combined
lead-lag model does *not* beat the TWAP-only baseline out-of-sample (Brier
0.128 vs 0.116). This isn't a bug - `gap_twap_bp` (`Z_gap`) is what the
synthetic generator's settlement outcome is actually keyed off of (see
`synthetic/rounds.py`), so a simpler model built on exactly that signal has
an inherent advantage on this data, and the combined model's extra features
mostly add estimation variance rather than information. This is precisely
the comparison the Phase 5 exit gate ("must beat or justify complexity
relative to simpler baselines") exists to catch - real historical data,
where TWAP isn't definitionally the whole story, is needed before this
comparison means anything about real edge.

**SUPERSEDED by Phase 12B Tranche 1 - see `docs/PHASE_12B_AUDIT.md` item 7/8.**
~~Phase 11 demo finding: ablations 6, 7, and 8 report zero non-wait actions
across the evaluation set~~ - the root cause identified below (raw
depth-level-only taker sizing, no marginal/risk-budget-derived
intermediate quantities) was correctly diagnosed **as a real defect**, not
correctly classified as a "not a bug" finding - Phase 12B audit item 7/8
fixed it directly (`taker_sizing_boundaries`, wiring in the previously-dead
`max_directional_spend`). After the fix, on the same class of dataset, all
three ablations trade (confirmed directly: 11, 11, and 114 actions
respectively across one 6-round eval set at fix time - see
`tests/test_walkforward.py::test_ablations_6_7_8_now_trade_after_taker_sizing_fix`).
Kept below for the diagnostic record, since the mechanism described was
real and is exactly what got fixed:
- **Ablation 6 (taker-only)**: this dataset's taker candidates were sized
  from raw cumulative order-book depth only (`taker_quantities`, no
  intermediate/partial quantities), which ran large enough that every one
  of them breached the `g_min=-100` risk floor, while maker's small fixed
  `maker_quantity=20` clip fit comfortably. With `taker_only=True` removing
  maker generation entirely, there was nothing feasible left to trade.
  `taker_sizing_boundaries` now generates the exact partial quantity that
  *does* fit the risk budget (e.g. "189.6 shares" out of a 500-share first
  level, not just 0 or 500), so taker-only execution can trade again.
- **Ablations 7/8 (cancel/replace, MPC - both wrap `OrderSupervisor`)**:
  since taker fills are immediate/synchronous (no resting period), they
  were never vulnerable to the REGIME_FLIP-before-TTL cancellation
  dynamic that killed these ablations' *maker* orders (Phase 9's own
  documented finding, still real and still reproducible for maker orders
  specifically - see the Phase 9 section above). Once sizing made a
  risk-feasible taker candidate available, it could win selection and
  fill before any regime flip had a chance to cancel it, so #7/#8 trade
  again too. Real historical data is still needed before any ablation
  comparison here is evidence of relative strategy value rather than an
  artifact of this synthetic generator's dynamics - that caveat, at
  least, was correct in the original finding.

**Phase 12 demo finding worth flagging (not a bug) - the dominant driver
of the ~96% live-vs-replay parity rate**: `run_shadow_demo.py` shows the
shadow (recv_ts-gated) decision stream choosing WAIT at effectively every
decision point, even on rounds where offline replay finds 20+ non-WAIT
actions - initially looked exactly like a bug in the new `time_attr`
plumbing, so it was traced rather than assumed. Root cause, confirmed by
directly comparing the two "latest spot observation" lookups both z_spot
and the regime classifier depend on: `FeatureConfig.canonical_horizon_s`
(1.0s) exactly equals this synthetic generator's spot-tick spacing (also
1.0s). Under `event_time` gating (offline), the observation "now" and the
observation "1 canonical horizon ago" are two different ticks, so Z_spot's
momentum is a real, nonzero number. Under `recv_ts` gating (shadow), the
very latest tick hasn't "arrived" yet at any whole-second decision
timestamp (recv_ts = source_ts + a small fixed offset, always > 0), so
"now" resolves to the *same* prior tick as "1 horizon ago" - confirmed
directly: at `decision_ts=1000.0` on `synthetic-round-0002`, both lookups
returned the identical `(source_ts=999.0, value=80790.51...)` observation.
`log(x/x) = 0.0` exactly, every single time a decision point lands on a
whole second (effectively always, since this generator ticks on whole
seconds) - so Z_spot is deterministically zero, `spot_direction` is always
`FLAT`, and regime rule #1 ("Either leg FLAT -> WAIT") locks the controller
into WAIT for the entire round, regardless of what the CLOB/gap features
are doing. This is *not* a bug in `time_attr`/`ShadowRunner` - both gates
are behaving exactly as specified, and this is precisely the class of
issue Phase 12 exists to surface before it reaches live capital: a feature
parameter with *zero* safety margin against any nonzero feed-arrival
latency, however small. It only fully saturates to 100% because this
synthetic generator's tick spacing (1.0s) exactly coincides with
`canonical_horizon_s` (1.0s); real market data ticks far more irregularly,
so this exact total lockout is a synthetic-data artifact, but the
*underlying fragility* (a momentum lookback with no margin above expected
feed latency) is real and would need `canonical_horizon_s` tuned with a
safety margin above measured live latency before Phase 13. Deliberately
left unfixed here rather than widening `canonical_horizon_s` - that's a
Phase 4/5 model parameter with cross-cutting effects on Phase 5's already-
calibrated model and Phase 11's ablation numbers, out of scope for a
Phase 12 change. The overall ~96% parity rate is still meaningfully above
this "always WAIT" floor because a large majority of offline decisions are
already WAIT too (the two streams agree by coincidence, not because the
underlying cause is rare); the ~4% of genuine mismatches are attributable
to a second, smaller, and separate effect - a `BOOK_DELTA` right at a
decision boundary not yet "arrived" live can flip `z_clob` enough to
change which maker candidate wins (confirmed directly: `z_clob=0.0` offline
vs `0.80` live at one such boundary) - which is the kind of small,
occasional, and expected live/replay divergence Phase 12's parity report
is meant to catch and quantify.

## Notable bugs caught during Phase 0/4/5/7/8/9/10/11/12B build-out (fixed, worth knowing about)

Building each phase exercised the previous phases' pipeline far more
heavily than their own build-out did, and surfaced real bugs versus just
calibration issues - noted here since they affected numbers already
reported to the user in earlier sessions:

- **(Phase 12B, item 3/D)** `walkforward/ablations.py::_run_baseline_round`
  (and an independent second copy in
  `scripts/run_one_step_controller_demo.py`) passed the same value for
  `spot` and `spot_prev` (making `spot_direction` always 0, breaking the
  baseline's unanimity check) and passed absolute replay time where
  elapsed round time was required (making the decision-window gate fail
  for the entire duration of every round except the one starting at
  `t=0`). Both bugs made the baseline strategy trade in 0 rounds
  regardless of the underlying market, invalidating every baseline
  comparison reported before this fix. Fixed via a shared `elapsed_t()`
  helper (`baseline/inputs.py`) and a genuine spot lookback cursor,
  mirroring the pattern `scripts/run_baseline_replay.py` already had
  right, one file over. Full detail: `docs/PHASE_12B_AUDIT.md`.
- **(Phase 12B, item 7/8)** Taker candidate quantities were raw cumulative
  book-depth sums only (`taker_quantities`), with no intermediate,
  marginal-edge, or risk-budget-derived sizes - `max_directional_spend()`
  had existed since Phase 3, unit-tested in isolation, but was never
  called from anywhere in candidate generation. Whenever even the
  smallest depth level breached `g_min`, taker execution had nothing
  feasible to offer at all (the root cause of Phase 11's "ablation 6/7/8
  show zero actions" finding above). Fixed with
  `taker_sizing_boundaries`, generating the exact partial quantity that
  fits the tightest of the marginal-edge/position/spend/risk-budget
  boundaries.
- **(Phase 12B, item 10)** Every taker candidate (alpha and hedge) was
  evaluated with `limit_price=1.0` - not real worst-price protection,
  since prices are probabilities in `[0,1]` and 1.0 is unconstraining.
  Fixed: `taker_sizing_boundaries` now derives the worst acceptable price
  as the last book level whose own marginal edge still clears
  `min_marginal_edge`, threaded through as `CandidateAction.max_execution_price`
  and used as the real limit price at dispatch.
- **(Phase 12B, item 9)** `evaluate_constraints`'s `spend_cap` check
  compared one candidate's own incremental cost against the full cap,
  not cumulative round spend - a sequence of individually-legal orders
  could exceed the intended round budget by an unbounded multiple. Fixed
  by comparing `portfolio_after.C` (the real cumulative cost, since every
  caller constructs a fresh `PortfolioState()` per round) against the cap
  directly.
- **(Phase 12B, item A)** `generate_synthetic_dataset()` always restarted
  round numbering at index 0, so any two separate calls (e.g. "train on
  15 rounds" then "evaluate on 6 rounds," the pattern used in every
  Phase 8-12 demo script and two test-suite fixtures) generated
  byte-identical market content for overlapping indices - every "held-out"
  evaluation performed since Phase 8 had actually evaluated on training
  data. Fixed with an `id_offset` parameter; all 9 identified call sites
  updated to use disjoint ranges.
- **(Phase 12B, item C)** Every Phase 8-12 demo/harness `train_q_model()`
  returned the raw `fit_logistic_regression` output directly, skipping
  the Platt calibration step Phase 5's own reference demo already
  implements correctly - an uncalibrated `q` can make
  `DeltaEV_U(x) = q*x - K_U(x)` look positive when the true (calibrated)
  probability would make it negative. Fixed with a shared
  `CalibratedModel`/`fit_calibrated_model` helper
  (`model/calibrated.py`), duck-type-compatible with the existing
  `LogisticModel.predict_proba` interface so no downstream caller needed
  to change beyond how the model is constructed.
- **(Phase 12B, item 13/E/L)** Every Phase 8-12 controller path (and the
  baseline) converted a chosen taker candidate directly into a `Fill`
  using that candidate's own pre-evaluation walk estimate, never actually
  submitting through Phase 7's `OrderState`/delay/revalidation lifecycle
  (`ExecutionSimulator.submit_taker_order`/`resolve_pending`, both
  already built and tested since Phase 7, just never called from any
  downstream controller). Inert-by-coincidence to date
  (`taker_delay_ms=0.0` everywhere, since nothing yet reads a market's
  real delay from its config - Phase 12B Tranche 3), but would have
  silently ignored a real delay the moment one was wired in. The
  baseline additionally had no depth-walk realism at all (always a full
  fill at the quoted limit price), a materially easier execution
  assumption than every V2 arm. Fixed with a single shared
  `ExecutionSimulator.execute_taker()` used identically by every arm
  (baseline included) in `walkforward/ablations.py`, `shadow/runner.py`,
  and the two demo scripts that had their own inline dispatch
  (`run_one_step_controller_demo.py`, `run_order_supervisor_demo.py`).

- **(Phase 5)** The L2 regularization gradient was divided by `n` twice
  (once as part of the averaged data gradient, again explicitly on the
  regularization term). For datasets in the thousands of rows, this made
  effective regularization strength ~1/n of what was requested - e.g.
  `l2=1.0` behaved like `l2=0.0004` - letting the model run to extreme,
  overconfident weights on the near-separable synthetic data. This is what
  caused the isotonic-calibration blowup below, not a bug in isotonic
  itself. Fixed by not dividing the regularization term by `n`.
- **(Phase 5)** Once the above was fixed, an *unrelated* second bug
  surfaced: the explicit gradient-descent step on the (now correctly-
  scaled) L2 term diverges to +-inf whenever `lr * l2 >= 2` (e.g. `l2=10`
  with the default `lr=0.3` produced a weight of `1.25e+89`). Fixed by
  switching to a proximal/shrinkage update for the L2 term
  (`w *= 1/(1+2*lr*l2)`), which is unconditionally stable for any `l2 >= 0`.
- **(Phase 5, not a bug)** Isotonic calibration fit on the validation split
  produced a Brier score far *worse* on test (0.318) than the model's raw,
  uncalibrated output (0.0996) - initially looked like a calibration bug,
  but PAVA was behaving correctly (in-sample Brier improved from 0.039 to
  0.001, exactly what pooling adjacent violators should do); the real
  issue was that near-deterministic synthetic data leaves isotonic almost
  nothing to pool, so it memorizes validation noise instead of
  generalizing. See "Known reconstruction gaps" #8.
- **(Phase 7)** `submit_taker_order` originally called `reconcile_fill`
  immediately at construction time regardless of `taker_delay_ms`, so a
  delayed order was already `FILLED` the instant it was submitted. This
  silently defeated the entire pending-delay mechanic: `can_cancel(now_ts)`
  checks `state is PENDING_DELAY`, but the state had already moved past
  that before any `now_ts` could be checked - the exact behavior Roadmap
  Phase 7's "Pending-delay no-cancel test" exists to catch, and it was
  caught by hand-testing the delayed-order path immediately after writing
  it, before a single automated test existed for it. Fixed by leaving the
  order in `PENDING_DELAY` at submission and adding `resolve_pending()`,
  which only applies the fill once the caller's simulated clock reaches
  `matched_ts`.
- **(Phase 8, test-design issue, not a controller bug)** The first version
  of the `test_optimizer_never_violates_g_min_across_random_states`
  Hypothesis test failed on `u=d=c=0.0, g_min=1.0`: the *chosen* candidate
  was WAIT with `g_after=0.0 < g_min=1.0`. This isn't the optimizer
  violating anything - WAIT never changes G, so if the starting portfolio
  is already below an unreachable `g_min` (impossible to avoid when
  `g_min > 0` from a flat start, since G can never exceed 0 with zero
  exposure), no action can fix that, and WAIT is correctly the best
  available choice. Fixed the test's invariant to compare against
  `min(g_min, starting_G)` rather than `g_min` alone - the real guarantee
  is "never worse than the floor or the starting point, whichever is
  looser," not "always at or above an arbitrary configured floor
  regardless of where you started."
- **(Phase 9)** `SupervisorDecision.reason` had no default value, so the
  `REPLACE` branch of `review_order` - which constructs a decision with
  only `order_id` and `action` (a replace has no `CancelReason`) - raised
  `TypeError: missing 1 required positional argument: 'reason'` the first
  time a book-displacement scenario actually exercised that code path
  (`test_tick_size_change_with_open_order_replace_uses_new_grid`). Every
  `CANCEL` branch happened to pass all three positional args, so this went
  unnoticed until a test hit the one branch that didn't. Fixed by giving
  `reason` a default of `None`.
- **(Phase 10)** Continuation rollouts re-walked the *original, undepleted*
  order book at every step, so a large taker fill's continuation value
  double-counted the exact liquidity it had just consumed - it looked like
  a modeling nuance ("book held fixed across the horizon") until a smoke
  test showed two candidates with clearly different immediate EVs (91.25
  vs 213.15) landing on the *exact same* total sequence value (304.396):
  the continuation step was re-buying the same cheap depth a second time,
  as if it refilled with zero latency inside a single 1-3s horizon step -
  precisely what Phase 7's delay/depth-walking machinery exists to rule
  out. Fixed by depleting the relevant side's book by the chosen
  candidate's `expected_fill` before recursing into the next level, with
  `test_continuation_does_not_double_count_consumed_liquidity` guarding
  against a regression. After the fix, splitting a purchase across "now"
  and "later" correctly ties in total value under zero churn cost (a
  separate, genuine finding - see "Known reconstruction gaps" and the demo
  finding above), rather than the split option being inflated.
- **(Phase 11)** `generate_hedge_candidate`'s sizing (SS17's
  `min_hedge_quantity`) used the raw best-ask price, but the taker
  candidate it feeds into then charges the *actual* taker fee on top - so
  the hedge was always sized just short of restoring `G` to `g_min`
  (`g_after=-30.000000000000004` against a floor of `-30.0` in the
  smoke test that caught it - a few ULPs short after the fee fix, versus
  visibly short before it, confirming the fee omission was the real gap
  and not just floating-point noise). Fixed by sizing against
  `c_effective = price + fee_config.taker_fee(1.0, price)` instead of raw
  price; a small `x_min *= 1.0001` safety margin (scoped to this function
  only, not touching Phase 3's already-tested general constraint logic)
  absorbs the remaining floating-point boundary case.
- **(Phase 11)** SS18's actual objective (`J = E[PnL] + lambda_G*G - ...`)
  needed a nonzero `lambda_g` for `enable_portfolio_repair` to ever
  functionally matter (at `lambda_g=0` a hedge's negative standalone EV
  always loses to WAIT), and tuning it hit the same class of scale
  mismatch as Phase 10's `churn_penalty=5.0` finding, twice: `lambda_g=0.5`
  made every directional candidate's selection score negative and
  suppressed *all* trading, because `g_after` for a meaningful position
  runs in the hundreds while `ev_after` runs in the tens (confirmed by
  dumping the candidate table - every non-WAIT candidate scored negative).
  Backing off to `lambda_g=0.05` fixed taker but then suppressed *maker*
  candidates specifically, for a subtler reason: maker's `ev_after` is
  properly fill-probability-weighted (`rho`-scaled expectation) but the
  `g_after` used in the *same* selection score is Phase 8's deliberately
  pessimistic if-filled value, unweighted - so the same `lambda_g`
  penalizes maker's risk far more than its true expected contribution, an
  inherent inconsistency between an expectation term and a conditional-
  worst-case term added together that a deeper fix (a separate
  probability-weighted "selection_g") would need to resolve properly.
  Given time constraints, re-tuned to `lambda_g=0.01` instead, verified via
  a trace script to restore the same 8 non-wait decisions per round as the
  `lambda_g=0.0` baseline. Left as a documented, empirically-tuned
  constant for this dataset (see `walkforward/ablations.py`'s
  `MANDATORY_ABLATIONS` comment) rather than a formally resolved design -
  flagging the maker EV/G inconsistency here so it isn't rediscovered from
  scratch later.

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

- **(Phase 12B Tranche 1.1)** The per-window calibration split
  (`walkforward/pipeline.py::fit_window_artifacts`) ran an example-count
  fraction split (`time_ordered_split`) over a window's own flat example
  list, which could and did place some of a round's own decision-point
  examples in the raw-model fit set and the rest of that *same round* in
  the calibration set - a round-level leak, since every example from one
  round shares that round's single settlement outcome. `sweep_parameter`
  was also evaluated directly on TEST rounds in the demo script (test-
  period tuning), and `parameter_stability_across_windows` reused window
  0's model for every window instead of fitting one per window.
  `taker_sizing_boundaries`'s risk-budget boundary used a flat
  `G_current - g_min` budget that is only exact when the purchased side
  is already the non-minimum side - buying the currently-underrepresented
  side can raise `min(U,D)` faster than it costs, making the flat budget
  badly conservative. Fixed: round-granular fit/calibration split
  (`_split_rounds_for_fit_and_calibration`), the sweep now uses
  `window.validate_round_ids`, stability fits a real per-window model via
  `WindowArtifacts`, and `directional_projected_g` replaces the flat
  budget with the exact side-aware kernel evaluation. Full detail:
  `docs/PHASE_12B_AUDIT.md`'s Tranche 1.1 section.
- **(Phase 12B Tranche 1.2)** `ExecutionSimulator.execute_taker()` could
  be handed the actual future book at submission time
  (`revalidation_asks`), so a delayed order's eventual fill was fully
  computed and stored before `matched_ts` ever arrived - one caller
  already read `.walk` immediately, silently bypassing the delay
  entirely. Three more execution paths (`shadow/runner.py`,
  `run_order_supervisor_demo.py`, `_run_baseline_round`) still mutated
  portfolio state from a delayed result without waiting for `matched_ts`.
  Taker candidate sizing had no candidate-specific worst-price limit
  (only the shared, depth/marginal-edge-only `p_max`), so an adverse
  repricing during the delay window could still breach `g_min`/
  `spend_cap`. A one-round calibration holdout (the common case at
  default window sizes) risked fitting Platt on a single-class `y`,
  silently producing a degenerate constant calibrator. Fixed: the
  `submit_taker`/`resolve_taker`/`PendingTakerOrder` redesign (no future
  fill information exists until `resolve_taker` is called with the real
  causal book), a shared `TakerOrderQueue` used by every execution path,
  `taker_max_execution_price` (exact, candidate-specific, tick-floored),
  and an `IdentityCalibrator` fallback with backward-expanding
  calibration-window search. Full detail: `docs/PHASE_12B_AUDIT.md`'s
  Tranche 1.2 section.
- **(Phase 12B Tranche 2A)** `PendingExposure.worst_case_portfolio()`
  assumed every active order filling *simultaneously* was the worst case
  for `G = min(U,D) - C` - false in general, since a fill on the
  currently-underrepresented side can *raise* `min(U,D)` enough to more
  than offset its own cost (a two-sided fill can be the BEST case, not
  the worst; a one-sided fill of just one leg can be worse than both
  filling). Pending exposure also omitted fees from committed cost, no
  `RiskView` combined confirmed + pending-taker + open-maker exposure for
  admission, open maker orders were never checked jointly before placing
  another one, `ShadowRunner`'s delayed-order resolution book was gated
  on `recv_ts` (this system's own wire-arrival time) instead of
  `event_time`/`source_ts` (the real exchange's own book at `matched_ts`
  - the two are genuinely different clocks), and the per-window
  calibration split could satisfy calibration-side class diversity while
  leaving the raw model fit on a single-class prefix. Fixed: an exact
  (bounded, `2^n`-subset) scenario envelope with a conservative
  analytical fallback beyond the cap (`PendingExposure.worst_case_g`),
  fee-inclusive `ActiveOrderExposure`, a `RiskView.admits()` hard-
  admission gate wired into every maker-placement path that tracks open
  orders, `event_time`-gated exchange execution-truth resolution, and a
  two-sided (fit AND calibration) diversity search that returns "no
  model" rather than fake calibration when unreachable. Full detail:
  `docs/PHASE_12B_AUDIT.md`'s Tranche 2A section.
- **(Phase 12B Tranche 2B-2E)** Four architectural additions on top of
  2A's aggregate risk admission, all gated behind default-`False`/`0.0`
  config so pre-existing behavior is unchanged unless explicitly enabled:
  `OrderPurpose.BUFFER_BUILD` candidates that may trade negative EV for
  positive `delta_g` (`cfg.enable_buffer_build`); soft regime control,
  where a disallowed candidate family is generated with a selection
  penalty instead of never existing (`cfg.soft_regime`); dynamic maker
  quantity/TTL/price-offset grids derived from `q, tau, sigma, G`
  instead of fixed constants (`cfg.dynamic_maker`); and a redesigned
  `OrderSupervisor.review_order` that replaced the old unconditional
  "regime flip -> cancel" rule with an `argmax(V_hold, V_cancel,
  V_replace)` economic policy plus a hysteresis margin against
  thrashing, keeping only feed-staleness and risk-breach as
  unconditional hard-safety cancels. Full detail (formulas, files, test
  list): `docs/PHASE_12B_AUDIT.md`'s Tranche 2B-2E section.
- **(Phase 12B Tranche 2.1)** Thirteen integration/math corrections
  closing gaps left in Tranche 2's architecture: hard safety overrides
  (feed-stale/risk-breach) now precede the supervisor's rate limiter
  rather than following it; `RiskView.admits()` gained `position_limit`
  enforcement and is now the universal pre-dispatch gate for every
  executable order type (taker, maker, REPLACE), not just maker
  placement; `OneStepController.decide()` checks aggregate risk *before*
  selection so a rejected top candidate reranks to the next-best legal
  one instead of vanishing; HEDGE/BUFFER_BUILD taker candidates get a
  purpose-aware worst-price ceiling instead of an unconstrained
  `limit_price=1.0`; breach-recovery semantics allow a partial repair to
  be admitted once already below `g_min` (prohibiting new ALPHA risk,
  requiring `G_after > G_before`); the selection formula's maker
  soft-risk term is now fill-probability-weighted (`expected_delta_g`)
  instead of using the absolute if-filled `G` level; BUFFER_BUILD
  generates a bounded multi-quantity candidate set instead of a single
  peak-quantity candidate; `dynamic_maker_sizing`'s dimensionally
  invalid dollar-vs-share comparison was replaced with an exact
  per-price feasible-quantity walk; REPLACE now actually reaches
  `OrderSupervisor.apply_replace` through both integration harnesses via
  a real re-evaluated `ReplacementPlan`, itself RiskView-gated; and
  `V_cancel` no longer double-counts `edge_min` (moved to a proper
  `hold_eligible` threshold). Full detail (formulas, files, test list):
  `docs/PHASE_12B_AUDIT.md`'s Tranche 2.1 section.
- **(Phase 12B Tranche 2.2 - final synthetic correctness closure)** Six
  remaining gaps closed before synthetic strategy work stopped: the
  aggregate `RiskView.admits()` g_min floor contradicted Tranche 2.1's
  own breach-recovery allowance (every recovery candidate was
  aggregate-invalid once already breached, since the worst-case
  envelope's own empty-fill subset reproduces the pre-candidate G
  exactly) - fixed with a two-mode envelope
  (`is_recovery_candidate`/`is_recovery_purpose()`), matching
  `_finalize`'s own `G_after > G_before` rule, and the identical
  contradiction in `purpose_aware_max_execution_price`'s HEDGE/
  BUFFER_BUILD price ceiling; `MPCController.decide()` now threads
  `risk_view` into its immediate one-step call and shares one
  `candidate_selection_score()` function with `OneStepController` (no
  more silently duplicated/drifted objectives), falling back to the
  plain risk-aware one-step decision whenever real active exposure
  exists rather than exploring a hypothetical rollout tree with no model
  for it; a new shared `TradingSession` (`execution/session.py`) now
  owns confirmed portfolio state, taker/maker order lifecycle, and
  RiskView-gated dispatch, with `ShadowRunner` fully migrated onto it
  (previously the one execution engine still missing RiskView,
  aggregate maker exposure, `OrderSupervisor`, and real open-order
  tracking - makers resolved via an immediate Bernoulli draw instead);
  BUFFER_BUILD's float-safety margin no longer discards a genuine
  exact-exchange-minimum candidate; `tau=0` (a real "no time left") is
  now distinguished from `tau=None` ("not supplied"), both in
  `dynamic_maker_candidates` and `OneStepController.decide`'s own
  signature; and the cancellation-hysteresis contract ("hysteresis
  permits a slightly inferior HOLD, it does not make V_hold numerically
  competitive with V_cancel") is now pinned down by an explicit test.
  Full detail (formulas, files, test list): `docs/PHASE_12B_AUDIT.md`'s
  Tranche 2.2 section.

**Phase 7 demo limitation worth flagging (not a bug)**: `run_execution_simulator_demo.py`
always reports 0 repriced orders. The synthetic generator emits book events
on a 1-second tick; the 250ms delay window is entirely inside a single tick,
so the book queried at `submit_ts` and at `submit_ts + 0.25s` is always
identical in this dataset - there's never actually a chance for the book to
move within the window. The repricing logic itself is verified directly in
`tests/test_execution.py` with hand-constructed book snapshots that do
differ between submission and revalidation. Sub-second synthetic tick
granularity would be needed to see repricing in the demo's own output.

## Live adapter confidence

**Superseded for the market/reference adapters by Phase 12C
(2026-08-15).** The table below was written when endpoints were taken from
docs.polymarket.com (2026-08-13) without a live call. Phase 12C verified
them against the live services and found several doc-vs-reality
discrepancies - see `docs/REAL_RECORDER_ARCHITECTURE.md` SS1 for the full
verification log.

| Adapter | Confidence | Caveat |
|---|---|---|
| `realtime/clob_ws.py` + `feeds/polymarket_clob.py` - market channel + in-memory book | **Verified live (2026-08-15)** | Subscribe message, `custom_feature_enabled` gating, and `book`/`price_change`/`last_trade_price`/`best_bid_ask`/`new_market`/`tick_size_change` shapes all confirmed against the live socket. Fixed a routing bug that discarded 100% of book deltas (`price_change` has no top-level `asset_id`). |
| `realtime/discovery.py` - market metadata | **Verified live (2026-08-15)** | Gamma + CLOB market-info; UP/DOWN from explicit `tokens[].outcome` labels cross-checked against Gamma `outcomes`. The old `NotImplementedError` mapping gap is closed. Found that Gamma `startDate` is the row-creation time, ~24h before the true round start. |
| `realtime/rtds.py` - Chainlink reference / TWAP-30 / TWAP-60 / Binance | **Verified live (2026-08-15)** | One shared connection, no credentials. Found that the documented dotted topic aliases are rejected and that `filters` suppresses live delivery; subscribes unfiltered and filters client-side. |
| `feeds/chainlink_twap.py` | Low - **superseded, do not use** | Chainlink Data Streams direct integration with an unverified HMAC handshake. Polymarket's own documentation recommends RTDS for production Chainlink TWAP, and `realtime/rtds.py` implements that with no credentials required. Kept only so the `TWAPFeed` interface has a direct-vendor reference implementation. |
| `feeds/polymarket_user.py` - order/fill WSS stream | High (WSS) / Unconfirmed (REST) | **Not exercised in Phase 12C** - no authenticated trading occurs (item 14). Still unverified against a live authenticated session. |
| `feeds/spot_composite.py` | High | Coinbase/Binance public spot-price REST endpoints. Superseded as the BTC leading signal by RTDS Binance, which arrives on the same socket as the reference feeds with a provider timestamp. |

## Before real historical data or live trading

- Replace `synthetic/rounds.py`'s dataset with real recorded rounds before
  trusting Phase 0's baseline performance report or any later phase's
  backtest - the synthetic generator only proves the pipeline runs and
  reconciles, not that any strategy has edge.
- Re-run Phase 11's 8-ablation matrix and parameter sensitivity/stability
  sweeps against real historical rounds before trusting any of them - the
  ablations 6-8 "zero actions" finding (fixed in Phase 12B Tranche 1 -
  see "Notable bugs" above; all 8 arms trade on synthetic data now) and
  the `edge_min=0.0`/`lambda_g=0.01` tuning are both dataset-specific
  artifacts of this synthetic generator's regime-flip frequency and
  book-depth sizing, not general conclusions. See
  `docs/PHASE_12B_AUDIT.md`'s Tranche 1/1.1/1.2/2A status for everything
  fixed since the original audit.
- Before trusting Phase 12's parity numbers against a real feed, re-tune
  `FeatureConfig.canonical_horizon_s` with a safety margin above measured
  real feed-arrival latency - see the Phase 12 finding above; this
  synthetic dataset's exact 1.0s-tick/1.0s-horizon coincidence is a
  synthetic-data artifact, but the underlying fragility (zero margin
  against nonzero arrival lag) is real and must be checked against actual
  live latency, not assumed away.
- Resolve the `_map_tokens_to_sides` gap in `polymarket_clob.py`.
- Confirm the Chainlink Data Streams auth handshake against Chainlink's own
  docs (not Polymarket's).
- Phases 7, 12, and 13 all have hard exit gates in the Roadmap doc
  (execution-model error tolerance, live/replay parity, reconciliation with
  no unexplained ledger discrepancies) that must pass before any real
  capital is at risk - none of those gates have been exercised yet.
