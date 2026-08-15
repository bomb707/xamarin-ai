# Real Data Schema (Phase 12C)

The recorder writes one SQLite file per capture session. Three tables:

| Table | Mutability | Purpose |
|---|---|---|
| `raw_events` | **append-only, never updated or deleted** | The immutable wire record |
| `rounds` | rewritten in place | Per-round metadata projection |
| `round_results` | rewritten in place | Per-round outcome + verdict |

`rounds` and `round_results` are *projections* of the immutable log, written
for convenient querying. They are never a substitute for `raw_events`: if
they disagree with it, `raw_events` is right.

---

## 1. `raw_events`

```sql
CREATE TABLE raw_events (
    session_id             TEXT    NOT NULL,
    recorder_sequence      INTEGER NOT NULL,
    topic                  TEXT    NOT NULL,
    event_type             TEXT    NOT NULL,
    round_id               TEXT,
    condition_id           TEXT,
    token_id               TEXT,
    source_timestamp_ns    INTEGER,
    publisher_timestamp_ns INTEGER,
    recv_wall_timestamp_ns INTEGER NOT NULL,
    recv_monotonic_ns      INTEGER NOT NULL,
    payload_json           TEXT    NOT NULL,
    normalized_side        TEXT,
    reconnect_generation   INTEGER NOT NULL,
    PRIMARY KEY (session_id, recorder_sequence)
);
CREATE INDEX idx_raw_round ON raw_events(round_id);
CREATE INDEX idx_raw_topic ON raw_events(round_id, topic);
CREATE INDEX idx_raw_token ON raw_events(token_id, source_timestamp_ns);
CREATE INDEX idx_raw_recv  ON raw_events(recv_wall_timestamp_ns);
```

### Column semantics

| Column | Notes |
|---|---|
| `session_id` | One per reader stream (`<capture>-clob`, `<capture>-rtds`). Each reader owns its own sequence counter, so a CLOB reconnect never renumbers RTDS events. |
| `recorder_sequence` | Monotonic **per session**, assigned at ingestion — not by SQLite. Survives batching: two events in one batch keep wire order. |
| `topic` | The transport stream (see §2). Named after the real subscription so a row traces back to what produced it. |
| `event_type` | The **original** wire event-type string, unmapped. |
| `round_id` | Market slug, e.g. `btc-updown-5m-1786773900`. NULL for events not attributable to one round (`new_market` for other assets). |
| `condition_id` | Polymarket condition id. |
| `token_id` | CLOB token (asset) id. NULL for frame-level events. |
| `source_timestamp_ns` | Origin's stamp. **NULL when absent — never 0.** |
| `publisher_timestamp_ns` | Relay's stamp (RTDS outer `timestamp`). NULL for CLOB events, which have no separate relay. |
| `recv_wall_timestamp_ns` | Local wall clock at frame read. |
| `recv_monotonic_ns` | Local monotonic clock. **Not an epoch** — only differences are meaningful. |
| `payload_json` | **Verbatim wire text**, key order and number formatting intact. |
| `normalized_side` | `"UP"` / `"DOWN"` / NULL. A convenience projection; NULL whenever the mapping is not known with certainty. |
| `reconnect_generation` | Increments on each reconnect of that stream. Events either side of a gap are distinguishable. |

### Invariants

* Append-only. `INSERT OR IGNORE` on `(session_id, recorder_sequence)` makes
  a replayed batch after a crash idempotent.
* `payload_json` is never rewritten by normalization. Item 6: *"Do not
  discard the original wire payload after normalization."*
* Timestamps are integer nanoseconds. Never floats — see
  REAL_RECORDER_ARCHITECTURE.md §4.
* Rows are returned in `(session_id, recorder_sequence)` order. That is the
  only ordering the raw layer claims; causal ordering is a downstream
  concern applied deliberately, not baked in here.

---

## 2. Topics and their event types

| `topic` | `event_type` values | Source |
|---|---|---|
| `clob_market` | `book`, `price_change`, `last_trade_price`, `tick_size_change`, `best_bid_ask`, `new_market`, `market_resolved`, *(any unrecognized type, verbatim)* | CLOB market WebSocket |
| `clob_rest` | `book_snapshot_rest`, `book_snapshot_rest_reconcile` | CLOB REST `/book` |
| `rtds_binance` | `update`, `backfill` | RTDS `crypto_prices`, symbol `btcusdt` |
| `rtds_chainlink` | `update`, `backfill` | RTDS `crypto_prices_chainlink`, symbol `btc/usd` |
| `rtds_twap_30` | `update`, `backfill` | RTDS `crypto_prices_twap_thirty` |
| `rtds_twap_60` | `update`, `backfill` | RTDS `crypto_prices_twap_sixty` |
| `market_metadata` | `market_metadata_discovered`, `market_metadata_poll`, `market_metadata_final`, `market_resolved_final` | Gamma + CLOB market-info |
| `recorder_control` | `lifecycle_transition`, `reconnect`, `book_integrity_check`, `label_reconstruction`, `rtds_error` | The recorder itself |
| `paper_quote` | hypothetical maker quotes (item 11) | Never sent to the venue |

An **unrecognized** market-channel event type is still recorded verbatim.
The raw log must be the wire, not a filtered view of it.

### `price_change` decomposition

A live `price_change` frame carries multiple tokens. Each element becomes
**its own row**, so a per-token query returns exactly that token's deltas.
The frame's `market` and `timestamp` are copied onto each element, plus
`_frame_change_count`, so the original grouping is recoverable:

```json
{"asset_id":"…","price":"0.99","size":"2315.29","side":"SELL",
 "hash":"…","best_bid":"0.01","best_ask":"0.02",
 "market":"0x…","timestamp":"1786771382835","_frame_change_count":2}
```

`side` refers to the side of the **book** the level lives on: `BUY` → bids,
`SELL` → asks. `size: 0` removes the level.

---

## 3. `rounds`

```sql
CREATE TABLE rounds (
    round_id           TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    condition_id       TEXT,
    question_id        TEXT,
    slug               TEXT,
    question           TEXT,
    description        TEXT,     -- full rules text
    resolution_source  TEXT,
    start_ts_ns        INTEGER,  -- TRUE round open (eventStartTime)
    end_ts_ns          INTEGER,  -- TRUE round close (endDate)
    up_token_id        TEXT,
    down_token_id      TEXT,
    tick_size          REAL,
    min_order_size     REAL,
    fee_config_json    TEXT,     -- full fee block, verbatim
    taker_delay_ms     REAL,
    twap_window_s      INTEGER,
    settlement_kind    TEXT,     -- 'chainlink_twap' | 'chainlink_reference'
    state              TEXT,     -- lifecycle state at last write
    raw_metadata_json  TEXT      -- {"gamma": …, "clob": …, "warnings": [...]}
);
```

This is item 2's required list. Every field is fetched per round; nothing is
hardcoded.

* `start_ts_ns` is **not** Gamma's `startDate` — see
  REAL_RECORDER_ARCHITECTURE.md §1.2.
* `fee_config_json` retains `makerBaseFee`, `takerBaseFee`, `feeSchedule`,
  `feeType`, `feesEnabled`, `makerRebatesFeeShareBps` verbatim. The
  integer `1000` base fees are **not** interpreted as a rate; the executable
  number is `feeSchedule.rate` (observed `0.07`).
* `raw_metadata_json` retains both complete API payloads.

---

## 4. `round_results`

```sql
CREATE TABLE round_results (
    round_id                  TEXT PRIMARY KEY,
    reported_outcome          TEXT,     -- Polymarket's own resolution
    reconstructed_outcome     TEXT,     -- ours, under the DECLARED basis
    reconstruction_basis      TEXT,     -- e.g. 'declared:crypto_prices_twap_sixty'
    label_agreement           INTEGER,  -- 1 / 0 / NULL if not comparable
    start_reference_value     REAL,
    end_reference_value       REAL,
    start_reference_ts_ns     INTEGER,
    end_reference_ts_ns       INTEGER,
    metrics_json              TEXT,
    is_training_grade         INTEGER,
    notes                     TEXT
);
```

`reconstructed_outcome` is the **declared** basis. The full two-basis
comparison (declared *and* plain-Chainlink-reference) lives in the
`recorder_control:label_reconstruction` raw event and in the capture report.

`is_training_grade` is 0 if the session recorded any dropped event, parse
failure, or book-integrity mismatch.

---

## 5. Useful queries

```sql
-- events per stream for one round
SELECT topic, event_type, COUNT(*)
FROM raw_events WHERE round_id = ?
GROUP BY topic, event_type ORDER BY 3 DESC;

-- source->receive latency percentile input, one stream
SELECT (recv_wall_timestamp_ns - source_timestamp_ns) / 1e6 AS ms
FROM raw_events
WHERE topic = 'rtds_twap_60' AND source_timestamp_ns IS NOT NULL
ORDER BY ms;

-- pre-round reference coverage (item 7)
SELECT r.round_id,
       COUNT(*) FILTER (WHERE e.source_timestamp_ns < r.start_ts_ns) AS pre_round,
       (r.start_ts_ns - MIN(e.source_timestamp_ns)) / 1e9 AS lead_s
FROM rounds r JOIN raw_events e ON e.round_id = r.round_id
WHERE e.topic = 'rtds_chainlink'
GROUP BY r.round_id;

-- did any stream reconnect mid-round?
SELECT round_id, topic, MIN(reconnect_generation), MAX(reconnect_generation)
FROM raw_events GROUP BY round_id, topic;

-- rebuild a token's book from the raw log
SELECT recorder_sequence, event_type, source_timestamp_ns, payload_json
FROM raw_events
WHERE token_id = ? AND topic IN ('clob_market','clob_rest')
ORDER BY session_id, recorder_sequence;
```

---

## 6. Relationship to the Phase-2 `EventStore`

`events/store.py` is **unchanged and still used** for deterministic replay of
normalized events (features, backtests, `ShadowRunner`). It is not a rival to
this schema.

| | `EventStore` | `RawEventStore` |
|---|---|---|
| commit granularity | per event | per batch |
| journal mode | default | WAL |
| timestamps | REAL seconds | INTEGER nanoseconds |
| source vs receive | collapsed by `event_time` | four clocks kept separate |
| payload | normalized dict | verbatim wire text + normalized projections |
| ordering on read | causal (`causal_sort`) | wire order |

The intended pipeline is: raw log → normalized event view → features. The
normalized view is derivable; the raw log is not.
