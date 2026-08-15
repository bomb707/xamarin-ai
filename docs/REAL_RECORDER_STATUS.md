# Real Recorder — Status (Phase 12C)

What is built, what was captured, what is verified, and what is known to be
wrong. Design rationale is in
[REAL_RECORDER_ARCHITECTURE.md](REAL_RECORDER_ARCHITECTURE.md); the storage
format is in [REAL_DATA_SCHEMA.md](REAL_DATA_SCHEMA.md).

**No orders were placed. No private key exists in this codebase. No model
was fitted and no strategy parameter was tuned.**

---

## 1. Item-by-item status

| # | Requirement | Status |
|---|---|---|
| 1 | Fix the Tranche 2.2 test-hygiene issue | Done — fresh `forced_decide` closure per invocation |
| 2 | Harden market discovery and metadata | Done, live-verified |
| 3 | Replace/fix the CLOB market WebSocket adapter | Done, live-verified |
| 4 | Use RTDS for BTC reference signals | Done, live-verified |
| 5 | Do not confuse settlement reference with TWAP | Done — **and a brief-vs-live conflict was found; see §5** |
| 6 | Raw recorder layer before normalized features | Done |
| 7 | Preserve pre-round history | Done |
| 8 | Explicit round lifecycle | Done |
| 9 | Carry Tranche 2.2 into ShadowRunner (`tau`/`sigma`) | Done |
| 10 | Replace hardcoded freshness with real freshness | Done |
| 11 | Do not pretend the maker simulator is fill truth | Done — counterfactual capture only, no model fitted |
| 12 | Fix replacement-order provenance | Done |
| 13 | Review event ordering for own-order events | Done |
| 14 | No authenticated trading | Enforced, and asserted by a test |
| 15 | Real integration acceptance gate | Done — see §3, §4 |

### Item 1 — test hygiene

`test_shadow_runner_dispatch_is_gated_by_aggregate_risk_view` reused one
`_forced_maker_then_wait_decide` closure across the `admits=True` and
`admits=False` runs. That closure carries a stateful `already_submitted`
flag, so the second run returned WAIT at every decision point and would have
"passed" with zero registrations regardless of what `RiskView` did. A fresh
closure is now built per invocation, so both branches genuinely attempt the
same candidate. Test correction only — no synthetic strategy design was
reopened.

### Item 9 — `tau` / `sigma`

`ShadowRunner` now passes `tau=fv.tau, sigma=fv.realized_vol` explicitly
into `OneStepController.decide`, from the same `FeatureVector` that produced
`q`. Previously both were omitted, so `decide()` fell back to `tau=None,
sigma=0.0` and dynamic maker mode — when enabled — silently operated as if
remaining time were unknown and realized volatility were zero while both
real values were sitting on `fv`.

### Item 10 — real freshness

Behavioural consequences, all now covered by tests:

* A decision point where `compute()` returns `InvalidFeatureState` is
  **recorded with its reason** instead of vanishing from the stream. On the
  synthetic evaluation round this surfaced 6 previously-invisible decision
  points (`INSUFFICIENT_VOLATILITY_HISTORY` at round start).
* A disconnect is recorded as an explicit not-fresh, suppressed decision
  **and** its resting orders are reviewed against a stale view — it no
  longer skips the decision while pretending orders remain supervised.
* Parity now excludes freshness-suppressed decisions for the same reason it
  excludes deadline-missed ones: their WAIT reflects the gate, not the
  causal-view difference the report isolates.

---

## 2. Tests

```
479 -> 493 tests, all passing
```

New Phase 12C suites:

| File | Covers |
|---|---|
| `tests/test_real_discovery.py` | item 2 — verbatim live payload fixtures, including the `startDate` trap, JSON-string `outcomes`, and refusal to infer UP/DOWN from index order |
| `tests/test_real_clob_ws.py` | item 3 — verbatim live frames; the `price_change` routing regression guard |
| `tests/test_rtds.py` | item 4 — four clocks kept separate, E18 decoding, unfiltered subscription |
| `tests/test_raw_recorder.py` | item 6 — exact ns conversion, verbatim payloads, non-blocking reader, bounded-drop accounting |
| `tests/test_round_lifecycle.py` | items 7, 8 |
| `tests/test_label_reconstruction.py` | items 5, 8 — both settlement bases |
| `tests/test_real_freshness.py` | item 10 |
| `tests/test_maker_counterfactual.py` | item 11 — asserts no fill/rho/q_fill is claimed |
| `tests/test_event_ordering_and_provenance.py` | items 12, 13 |
| `tests/test_real_service.py` | orchestration end-to-end, offline; item 14 enforcement |

Two defects in the new code were caught by these tests before any live run:
a first-touch comparison inverted in direction (a *higher* best bid means
queued behind, not at the touch), and `price_change`/unknown-type events
built but never emitted.

---

## 3. Live verification

Every endpoint and payload shape was confirmed against the live services on
2026-08-15. Full transcript in REAL_RECORDER_ARCHITECTURE.md §1. The
findings that changed the implementation:

| Finding | Impact |
|---|---|
| Gamma `startDate` is the row-creation time, ~24 h before the round | Round windows were being taken from it; now from `eventStartTime`/`endDate` |
| Gamma timestamps are ISO-8601 strings | `float()` on them raises; now parsed properly |
| `price_change` has **no top-level `asset_id`** | The old adapter silently discarded **100% of book deltas** (8,090 frames/60 s measured) |
| `custom_feature_enabled` gates `best_bid_ask` / `new_market` / `market_resolved` | Enabled by default |
| RTDS dotted topic aliases are rejected (`Invalid request body`) | Raw topic names used exclusively |
| RTDS `filters` returns one backfill then **zero** live updates | Subscribe unfiltered, filter client-side |
| BTC 5-minute markets settle on the Chainlink **TWAP-60**, not the plain reference | See §5 |
| Resolution lands 3–8 min after the round closes | Post-capture resolution sweep instead of a long tail |
| Gamma `?slug=` returns `[]` for settled markets | Retry with `closed=true` |

---

## 4. Captured integration sample

Two captures were taken. The first is reported in full because it is what
**found three real defects**; the second validates the fixes.

### 4.1 Capture 1 — 3 consecutive rounds, 2026-08-15 06:05–06:20 UTC

`btc-updown-5m-1786773900`, `-1786774200`, `-1786774500`

| Metric | Value |
|---|---|
| Events received | 539,529 |
| Events persisted | 538,761 |
| Duplicates suppressed | 768 |
| **Dropped events** | **0** |
| **Parse failures** | **0** |
| Queue high-water | 472 / 50,000 |
| Reconnects | 0 |
| REST resnapshots | 6 |
| Book-integrity checks | **30/30 matched** |
| `source -> recv` latency | p50 13.7 ms, p95 21.5 ms, p99 147.9 ms, max 2918 ms |
| `publisher -> recv` latency | p50 298.7 ms, p95 439.6 ms, p99 477.0 ms |
| Token/outcome mapping | 3/3 from explicit labels |
| Market IDs / windows / fees / delay | captured for all 3 rounds |

Per-round event counts (round 1 / 2 / 3):

```
price_change     167,147 / 147,434 / 198,220
book (WS)          2,246 /   2,330 /   3,254
best_bid_ask       3,282 /   3,242 /   5,044
last_trade_price   1,097 /   1,121 /   1,582
tick_size_change       2 /       2 /       2
REST snapshots         2 /       2 /       2
```

`tick_size_change` events did occur and were handled — the old adapter would
have raised `NotImplementedError` on them.

**Three defects this capture exposed:**

1. **RTDS stalled silently after ~688 s.** All four reference streams
   stopped, `recv` never raised, and the recorder reported
   `reconnect_count=0`, `parse_failures=0`. Rounds 2 and 3 therefore had
   **no reference data covering their settlement boundary** and their labels
   could not be reconstructed — inside a capture that otherwise looked
   perfect. Fixed with a stall watchdog on both stream adapters.
2. **RTDS events carried `round_id = NULL`.** `current_round_id` was never
   assigned, so all 2,647 reference rows were unattributable in the raw log,
   while in-memory attribution kept working and hid it. Fixed.
3. **Round 1 got 94 s of pre-round history against a 420 s lead**, because
   discovery took "the next round". Fixed: discovery now starts at the first
   round whose full pre-round window can be covered.

Label reconstruction, after the post-capture resolution sweep:

```
btc-updown-5m-1786773900  venue UP   declared UP (TWAP-60)  reference UP   agreement 1
btc-updown-5m-1786774200  venue DOWN declared —             reference —    (RTDS stall)
btc-updown-5m-1786774500  venue —    declared —             reference —    (RTDS stall)
```

The one round with complete reference data reconstructed **correctly**, with
both boundary observations landing at offset 0.00 s from the exact round
boundaries — the mechanism works; the stall was a data-availability failure,
not a reconstruction failure.

### 4.2 Capture 2 — 3 consecutive rounds, 2026-08-15 06:35–06:50 UTC

`btc-updown-5m-1786775700`, `-1786776000`, `-1786776300`
(`captures/phase12c_final_report.txt`)

| Metric | Value |
|---|---|
| Events received | 510,353 |
| Events persisted | 509,578 |
| Duplicates suppressed | 775 |
| **Dropped events** | **0** |
| Queue high-water | 270 / 50,000 |
| **Book-integrity checks** | **30/30 matched** |
| Stalls detected and recovered | 8 (7 CLOB, 1 RTDS) |
| Reconnects | 9 |
| `source -> recv` latency | p50 13.5 ms, p95 27.1 ms, p99 1549 ms, max 11.5 s |
| `publisher -> recv` latency | p50 279.7 ms, p95 434.9 ms, p99 472.8 ms |
| Parse failures | 49 — **all post-round teardown, now fixed; see below** |

**Item 7 satisfied on every round** (previously `SHORT` on round 1):

```
round 1: 465 pre-round Chainlink obs, earliest  510s before open  [OK]
round 2: 739 pre-round Chainlink obs, earliest  810s before open  [OK]
round 3: 1024 pre-round Chainlink obs, earliest 1110s before open [OK]
```

**Item 8 label reconstruction — 3/3 correct on BOTH bases:**

```
round                     venue   declared(TWAP-60)  reference(Chainlink)  offsets
btc-updown-5m-1786775700  DOWN    DOWN  ✓            DOWN  ✓               0.00s / 0.00s
btc-updown-5m-1786776000  UP      UP    ✓            UP    ✓               0.00s / 0.00s
btc-updown-5m-1786776300  UP      UP    ✓            UP    ✓               0.00s / 0.00s

declared_basis_agreement_rate   1.000
reference_basis_agreement_rate  1.000
bases_agreement_rate            1.000
labels_reproducible             True
```

Both boundary observations landed at **offset 0.00 s** from the exact round
boundaries on every round, so the reconstruction is not resting on a distant
proxy.

**The three capture-1 defects are confirmed fixed**: pre-round coverage is
`OK` everywhere, RTDS events carry a `round_id`, and the stall watchdog
fired 8 times and recovered every time (a stall that would previously have
silently voided the reference data).

**A fourth defect this capture exposed — and its fix.** The verdict came
back `training-grade: False`, disqualified by `parse_failures=49`. Every one
of those was **post-round teardown noise**, not data loss:

* Once every subscribed token settles, Polymarket closes the market channel
  cleanly: `ConnectionClosedOK: received 1000 (OK) all subscribed assets
  resolved`. That is the venue ending the subscription.
* The adapter treated it as a fault, reconnected, and — because the markets
  were now settled — got `404 Not Found` from `/book` for every token.
* This repeated for the whole 12-minute resolution sweep, producing the 7
  CLOB "stalls", most of the 9 reconnects, and all 49 parse failures.

So a capture with **zero dropped events and 30/30 integrity checks** was
being marked unusable for training by its own teardown. Fixed three ways: a
clean venue close is recorded as `stream_closed_by_venue` rather than a
failure and ends the subscription; a `/book` 404 on a settled token is
recorded as `book_unavailable_settled`; and the service now stops both
streams *before* the resolution sweep, since a finalized round has nothing
left to capture.

### 4.3 Capture 3 — teardown fix verification, 2026-08-15 07:10 UTC

One round, `btc-updown-5m-1786777800`
(`captures/phase12c_verify_report.txt`)

| Metric | Value |
|---|---|
| Events received | 190,167 |
| Events persisted | 189,922 |
| Duplicates suppressed | 245 |
| **Dropped events** | **0** |
| **Parse failures** | **0** (was 49) |
| Queue high-water | 428 / 50,000 |
| **Book-integrity checks** | **10/10 matched** |
| Stalls detected and recovered | 1 (RTDS, pre-round) |
| Reconnects | 1 |
| `source -> recv` latency | p50 12.7 ms, p95 35.9 ms, p99 1562 ms, max 2.8 s |
| `publisher -> recv` latency | p50 277.2 ms, p95 432.1 ms, p99 483.0 ms |

```
ws_snapshots=3142  rest_snapshots=2  deltas=177570  trades=1487  best_bid_ask=4404
pre-round: 426 Chainlink obs, earliest 474.0s before open  [OK]
```

Label reconstruction:

```
venue UP   declared(TWAP-60) UP ✓   reference(Chainlink) UP ✓
P_start 63010.5350 -> P_end 63061.8821   boundary offsets 0.00s / 0.00s
```

**Verdict:**

```
training-grade interval: True
labels reproducible:     True
```

This is the acceptance evidence for item 15: a real BTC 5-minute round
captured end-to-end with zero dropped events, zero parse failures, every
book-integrity check passing, full pre-round coverage, and a settlement
label independently reconstructed and matching Polymarket's own resolution
on both bases.

---

## 5. The settlement-rule discrepancy

The brief states:

> UP ⟺ ChainlinkBTC_end ≥ ChainlinkBTC_start … Do not use Binance or TWAP as
> a substitute settlement source.

Every live BTC 5-minute market says otherwise:

```
cryptoMarketConfigId : "btc-5m-twap-60"
cryptoMarketConfig   : {"twapEnabled": true, "twapLookbackSeconds": 60}
resolutionSource     : ".../btc-usd-twap-60s-streams"
description          : "...resolve to Up if the time-weighted average price
                        (TWAP) of Bitcoin, generated by Chainlink, ... is
                        greater than or equal to the price at the beginning
                        of that range."
```

**This is not resolved unilaterally.** Every round is reconstructed under
both bases and both agreement rates are reported:

* `declared` — the basis the market's own metadata declares (TWAP-60 today)
* `reference` — the brief's plain-Chainlink-reference rule

Binance is never a settlement basis under either reading — it is recorded as
a leading signal only, and `topic_for_basis` cannot return it.

On the rounds captured so far the two bases **agreed with each other**, so
they were not discriminating. A round where they disagree is the observation
that settles the question, and the recorder flags exactly that case
(`bases_agree = False`).

---

## 6. Known data-quality limitations

1. **`market_resolved` was never observed** on the market channel despite
   `custom_feature_enabled`. Instead the venue closes the whole subscription
   with "all subscribed assets resolved". Resolution is therefore read from
   Gamma `outcomePrices` (primary) and CLOB `tokens[].winner` (cross-check,
   lags by minutes).
2. **Resolution is not available at finalization time.** Rounds finalize
   with a reconstructed label and the venue outcome is filled in by a later
   sweep. A capture is therefore not fully labelled until the sweep (or
   `scripts/resolve_capture_labels.py`) has run.
3. **Sockets stall silently and recurrently.** Capture 2 recorded 8 stalls
   across ~25 minutes. The watchdog detects and recovers each one, but every
   stall is a real (short) data gap, and the root cause is server-side and
   outside our control. Any interval containing a stall deserves scrutiny
   before it is used for high-fidelity work.
4. **Sample size is an integration sample, not a training sample.** Three
   consecutive rounds validate plumbing and integrity, nothing statistical.
5. **`source -> recv` tail latency is large**: p99 1.55 s, max 11.5 s in
   capture 2 (p50 only 13.5 ms). This is dominated by the RTDS reference
   feeds, not the book. It bears directly on the `canonical_horizon_s`
   zero-margin fragility already flagged in PHASE_STATUS.md and must be
   re-measured over a longer capture before that parameter is trusted.
6. **Duplicates occur** (~0.15% in both captures). They are detected and
   suppressed, but their cause has not been investigated.
7. **Only the raw layer is built.** No normalized-event projection from
   `raw_events` into the Phase-2 `EventStore` shape exists yet, so the
   captured data is not yet consumable by the feature engine.
8. **`shadow/runner.py` still runs off the Phase-2 `EventStore`**, not the
   raw log. Real freshness is implemented and wired, but the real recorder
   and the shadow runner are not yet joined end-to-end.
9. **Hypothetical maker quotes were not generated during these captures.**
   The counterfactual tracker is implemented, wired to the live book/trade
   stream, and unit-tested, but no strategy drove it — so no real
   counterfactual dataset exists yet. That needs point 8 first.

---

## 7. Explicitly not done in this phase

* No `q` model fitted.
* No maker fill probability or `q_fill` calibrated — item 11 records the
  evidence for later estimation and nothing more.
* No strategy parameter tuned (`lambda_g`, regime penalties, maker
  parameters, edge thresholds, hysteresis, BUFFER_BUILD).
* No profitability claim.
* No authenticated trading, no private key, no order/cancel/replacement sent.

The next approval gate decides whether the captured dataset is trustworthy
enough for **real walk-forward model calibration and economic shadow
evaluation** — not whether synthetic PnL improved.
