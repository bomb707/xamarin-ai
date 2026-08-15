# Data Provenance & Runtime Market Constraints (Phase 12C.1)

Two invariants, made structurally true rather than conventionally observed:

> **Production/live code can never accidentally consume synthetic data.**
>
> **All executable market constraints come from the current Polymarket
> market, not static guesses.**

Both are enforced by tests that fail on an `import` statement or a missing
runtime object, not by naming conventions.

---

## 1. The three provenance modes

`src/xamarinbot/provenance.py`

| Mode | Meaning |
|---|---|
| `REAL_LIVE` | Observed from the live venue in real time by the recorder. |
| `REAL_REPLAY` | Replayed from a real capture. Every observation genuinely occurred; only the clock is replayed. |
| `SYNTHETIC_TEST` | Fabricated by a generator. Says nothing about market behaviour, edge, or profitability. |

**`SYNTHETIC_TEST` is the default** for a bare `EventStore`. That is the
safety property: an unlabelled store is treated as fabricated until
something proves otherwise, so *forgetting* to set provenance downgrades a
run to test-only rather than silently promoting fabricated data to
production evidence.

Provenance is persisted in the store's own `store_meta` table, so a
projected real capture does not become unlabelled — and therefore refused —
merely by being closed and reopened. Relabelling an existing store raises.

`require_real(provenance, context, allow_synthetic=False)` gates economic
evaluation. The escape hatch must be passed **at the call site** by
something that has deliberately opted in — a unit test or a
`scripts/dev_synthetic/` demo — never defaulted on by a library.

---

## 2. Where things live now

```
src/xamarinbot/
  provenance.py          DataProvenance, require_real
  rounds.py              RoundLabel  (neutral, provenance-tagged)
  market/
    constraints.py       MarketConstraints  (item 12)
    order_request.py     BUY dollar-encoding boundary (item 14)
  replay/
    feeds.py             was feeds/mock.py - Replay* adapters
    projection.py        REAL raw -> normalized (item 8)
  realtime/
    feed_adapter.py      was feeds/polymarket_clob.py - THE canonical
                         real-market Phase-1 adapter
    clob_ws.py rtds.py discovery.py …

devtools/synthetic/      the data FABRICATOR, outside the shipped package
scripts/                 real-market entry points ONLY
scripts/dev_synthetic/   run_synthetic_*  demos
```

### Renames that carry meaning

`feeds/mock.py` → `replay/feeds.py`, `Mock*` → `Replay*`. The module
contains no RNG and fabricates no values: it reads an `EventStore` that
something else populated and reconstructs feed-interface views of it. It is
what replays **real captured market data** into the shadow runner, and
calling it "Mock" is what made the synthetic-vs-real boundary hard to see.
No back-compat aliases were kept — they would defeat the guard below.

### Deleted / deprecated (item 7)

| Module | Fate | Why |
|---|---|---|
| `feeds/chainlink_twap.py` | **deleted** | Superseded by `realtime/rtds.py`, which is what Polymarket's own docs recommend and needs no credentials. Its auth handshake was never verified live. Zero callers. |
| `feeds/spot_composite.py` | **deleted** | Superseded by the RTDS Binance stream. Zero callers. |
| `feeds/polymarket_clob.py` | **moved** → `realtime/feed_adapter.py` | It is the canonical real-market Phase-1 bridge; `realtime/*` is the SSOT. |
| `feeds/polymarket_user.py` | **deprecated, retained** | Authenticated order/fill stream with no `realtime/` equivalent; Phase 13 needs it. Must keep zero callers. |

---

## 3. The permanent guard

`tests/test_import_boundaries.py` (the repo has no `.github/workflows`;
`pytest` is CI). It AST-parses every shipped module and asserts:

- nothing under `src/xamarinbot/**` imports `xamarinbot.synthetic`,
  `devtools*`, or `tests*`
- no shipped module references any `Mock*` name
- the three real-market scripts import no fabricated data and reference no
  order-placing symbol
- the retired adapters have no importers and their files are gone
- every `scripts/dev_synthetic/*.py` is named `run_synthetic_*`, and
  `scripts/` holds only real-market entry points

The one genuine blocker to the strong form (*all* of `src/`, not just
`realtime/` and `shadow/`) was that `walkforward/pipeline.py` and
`model/dataset.py` imported `SyntheticRoundResult` from the generator purely
as a typed carrier. That is now the neutral `xamarinbot.rounds.RoundLabel`.

---

## 4. REAL raw → normalized projection (item 8)

`src/xamarinbot/replay/projection.py`

```
RawEventStore_real  ->  NormalizedEventStore_real  ->  FeatureEngine
```

| Normalized event | Real source |
|---|---|
| `MARKET_CONFIG` | the `rounds` row the recorder persisted |
| `TWAP` | the round's **declared settlement basis**, chosen from its own `settlement_kind`/`twap_window_s` |
| `SPOT` | Binance BTCUSDT (leading signal) |
| `BOOK_SNAPSHOT` | `clob_market:book` + REST bootstrap/resync snapshots |
| `BOOK_DELTA` | each `clob_market:price_change` element |
| `SETTLEMENT` | the venue's resolved outcome, when captured |

**Nothing is ever synthesized.** Not a TWAP, spot, book observation, `p0`,
or timestamp. `p0` is the settlement-basis observation at or **before** the
round open — a real observation, or the round is refused. Missing streams
become counted, reported gaps and the feature engine's existing
`InvalidFeatureState` reasons downstream.

Plain Chainlink and TWAP-30 are deliberately **not** projected into
`EventType.TWAP`: `features/engine.py` builds one TWAP series, so emitting
two different quantities into it would silently interleave them and corrupt
`gap_twap_bp`/`z_gap`. Both remain in the raw log for diagnostics (item 15).

### Timestamps

`source_ts` and `recv_ts` become normalized columns — both are load-bearing
(`Event.sort_key` orders on `event_time`; `ShadowRunner` gates visibility on
`recv_ts`). Publisher time and the local monotonic clock have no column, so
they ride in a reserved `_provenance` payload block together with the raw
event's identity, its `condition_id`/`token_id`/side, the reconnect
generation, and a SHA-256 of the original wire bytes.

`EventStore.append_many()` exists because one real round is ~180,000
normalized events and `append()` commits per call.

---

## 5. Runtime market constraints (items 11–14)

`src/xamarinbot/market/constraints.py`

```python
MarketConstraints(
    condition_id, up_token_id, down_token_id,
    min_order_shares,        # SHARES, from CLOB mos / min_order_size
    tick_size, fee_configuration, taker_delay_ms,
    settlement_kind, twap_window_s,
    captured_at, source, provenance,
)
```

`OneStepConfig.taker_min_size = 1.0` is **deleted**. It was documented as a
placeholder for the real venue minimum "once wired in", while
`MarketConfig.min_order_size` already carried the real value from discovery
and dead-ended — nothing in the optimizer or execution layer read it. Every
sampled live BTC five-minute market reports **5.0 shares**, so the static
1.0 was not merely un-wired but wrong by 5×, in the direction that generates
orders the venue rejects.

`OneStepController.decide` / `MPCController.decide` now take
`constraints: MarketConstraints` in place of a bare `tick_size: float`, and
thread it into every candidate generator.

### One enforcement point

`optimizer/candidates.py::_finalize` — every candidate of every purpose
flows through it, so `qty < min_order_shares` adds `"min_order_shares"` to
`violated_constraints` there rather than relying on five generators each
remembering to check. Before this, `generate_hedge_candidate` and the
fixed-quantity maker path checked **no** minimum at all.

A sub-minimum candidate is marked **invalid, never rounded up**: its size
came from a risk or economic boundary, and rounding it up to reach the
minimum would breach exactly the constraint that produced it (item 13).

Minimum **shares** is never conflated with minimum **USDC notional**: 5
shares at $0.10 is $0.50 of notional and is a legal order.

### BUY encoding (item 14)

`market/order_request.py` is a translation boundary only. Polymarket's
FAK/FOK **BUY** takes a dollar `amount` while **SELL** takes share quantity;
the strategy optimizes shares throughout. The encoding is confined to this
one module and `x >= min_order_shares` is asserted on the SHARE quantity,
independently of the dollar encoding. **Nothing submits it** — there is no
client, no signing, no network call, and a test asserts as much structurally.

---

## 6. No fabricated numbers on the real path

**Item 10 — `q = 0.5` is gone from real runs.** When no probability model,
feature set, or design vector is available and the store is `REAL_*`, the
runner records `DecisionBlockReason.MODEL_UNAVAILABLE`, produces no ALPHA,
and dispatches nothing. The fallback survives only under `SYNTHETIC_TEST`.
The same treatment is applied in `shadow/parity.py` and
`walkforward/ablations.py`.

**Item 16 — `draw_maker_fill()` cannot run on real data.** It is an
uncalibrated Bernoulli; calling it on a real replay and reporting the result
as a maker fill would fabricate the single most consequential unknown in the
strategy. It now raises `SyntheticExecutionRefused` on `REAL_*`, and
`TradingSession` closes a real expiring quote as **unresolved**
(`n_maker_expired_unresolved`), leaving the Phase 12C counterfactual stream
as the only evidence — which is the true state of knowledge.

**Item 9 — freshness.** `REPLAY_FRESHNESS_POLICY` →
`SYNTHETIC_REPLAY_FRESHNESS_POLICY` (its budgets come from the generator's
declared cadence). New `REAL_REPLAY_FRESHNESS_POLICY` uses the measured real
cadences: book 2.0s (~130 msg/s), reference feeds 5.0s (~1 Hz).
`freshness_policy_for(provenance)` selects between them, so a real replay
can never inherit synthetic thresholds.

---

## 7. Settlement basis stays market-specific (item 15)

Neither plain-Chainlink nor TWAP-60 is hardcoded anywhere. The projection
and the label reconstructor both read the round's **persisted**
`settlement_kind` / `twap_window_s`.

`realtime/label.py` adds `LabelStatus{CONFIRMED, LABEL_AMBIGUOUS,
UNRESOLVED}`. A round is **`LABEL_AMBIGUOUS`** — and excluded from training —
when the declared basis reconstructs a different outcome from the one the
venue published, or when the market's own rule text contradicts its
structured configuration. Text that says nothing identifiable is treated as
*no evidence*, not as disagreement.

---

## 8. Capture artifacts (item 17)

`captures/VERIFICATION_MANIFEST.json`, regenerated by
`scripts/write_capture_manifest.py`, records the recorder commit, capture
timestamp, DB SHA-256, round IDs, per-round results, the data-quality
verdict, and known gaps. The `.db` files are gitignored (~500MB each), so
**the manifest is the committed evidence**.

The two flawed runs moved to `captures/archive/phase12c_pre_teardown_fix_*`;
`captures/phase12c_verify.db` remains canonical.

The manifest's verdict deliberately checks **more** than the recorder's own
health counters: a capture is training-grade only if it is clean *and* every
round's settlement label was actually reconstructed. The archived `sample`
capture reported `dropped=0` / `parse_failures=0` and would otherwise read
as healthy, while a silent RTDS stall had left two of its three rounds with
no reference data covering their settlement boundary. It is now correctly
reported `NOT_TRAINING_GRADE - unlabellable rounds`.

---

## 9. Running things

```bash
# real market, read-only
python scripts/run_market_discovery.py --rounds 2
python scripts/run_real_recorder.py --rounds 1 --db captures/run.db
python scripts/resolve_capture_labels.py captures/run.db
python scripts/write_capture_manifest.py

# REAL_REPLAY acceptance evidence
python scripts/run_real_replay_smoke.py captures/phase12c_verify.db

# synthetic demos (unmistakably named)
python scripts/dev_synthetic/run_synthetic_shadow_demo.py 2
```
