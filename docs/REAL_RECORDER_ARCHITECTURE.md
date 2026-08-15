# Real Market Recorder — Architecture (Phase 12C)

Phase 12C changes the character of the project: it stops designing around
synthetic behavior and starts measuring the actual market. Everything in
`src/xamarinbot/realtime/` talks to the real Polymarket / Chainlink /
Binance public data plane.

**Nothing in this phase places an order.** No private key, no authenticated
session, no maker order, taker order, cancel, or replacement is ever sent
(item 14). This is a recorder plus a paper shadow.

---

## 1. Live verification log

Every endpoint, topic, subscription shape and payload field below was
confirmed against the live services on **2026-08-15**, not taken from
documentation. The findings that *changed the implementation* are recorded
here because several of them contradict the docs.

### 1.1 Endpoints confirmed working

| Purpose | Endpoint | Result |
|---|---|---|
| Market catalogue | `GET https://gamma-api.polymarket.com/markets?slug=…` | 200, full payload |
| Executable market config | `GET https://clob.polymarket.com/markets/{condition_id}` | 200, explicit `tokens[].outcome` |
| Book snapshot (bootstrap/resync) | `GET https://clob.polymarket.com/book?token_id=…` | 200 |
| Market channel | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | connected, streaming |
| Reference data | `wss://ws-live-data.polymarket.com` | connected, streaming |

### 1.2 Finding — Gamma `startDate` is **not** the round start

For `btc-updown-5m-1786856400`:

```
startDate       2026-08-15T05:08:42.831334Z   <- row CREATION, ~24h early
eventStartTime  2026-08-16T05:00:00Z          <- true round open
endDate         2026-08-16T05:05:00Z          <- true round close
slug timestamp  1786856400 = 2026-08-16T05:00:00Z
```

The observed gap is ~85,900 seconds. Using `startDate` as the round start —
which the previous adapter did — mislabels `t` and `tau` for every feature
by roughly a day. `discovery.py` uses `eventStartTime` / `events[0].startTime`,
cross-checks it against the slug's unix timestamp, and emits a warning
recording that `startDate` was rejected.

### 1.3 Finding — Gamma timestamps are ISO-8601 strings

`"2026-08-16T05:05:00Z"`, and `"2026-08-15T05:08:42.831334Z"` with six
fractional digits. The previous adapter's `float(payload["endDate"])`
raises `ValueError` on every real payload. `discovery.parse_iso8601`
normalizes the `Z` suffix and the fractional part.

### 1.4 Finding — the CLOB `price_change` routing bug

A live `price_change` frame:

```json
{"market": "0x0778…2807", "timestamp": "1786771382835",
 "event_type": "price_change",
 "price_changes": [
   {"asset_id": "5440…0164", "price": "0.01", "size": "2315.29",
    "side": "BUY", "hash": "2692…bed2", "best_bid": "0.98", "best_ask": "0.99"},
   {"asset_id": "1010…4342", "price": "0.99", "size": "2315.29",
    "side": "SELL", "hash": "f0ef…0fc", "best_bid": "0.01", "best_ask": "0.02"}]}
```

There is **no top-level `asset_id`**. The old handler read
`msg.get("asset_id")` for every message, so for `price_change` it got
`None`, `_side_for_token(None)` returned `None`, and the handler returned
early — **silently discarding 100% of book deltas** while the feed looked
healthy. Measured rate during the probe: **8,090 `price_change` frames in
60 seconds**. The book would have been frozen at its bootstrap snapshot
forever.

`clob_ws.py` processes every element of `price_changes` independently and
writes one raw event per element.

### 1.5 Finding — `custom_feature_enabled` genuinely gates extra streams

A/B over 60 seconds on the same tokens:

```
{"assets_ids":[…],"type":"market"}                          -> book, price_change, last_trade_price
{"assets_ids":[…],"type":"market","custom_feature_enabled":true}
                                                            -> + best_bid_ask, new_market
```

Counts (custom, 60s): `price_change` 7747, `book` 138, `best_bid_ask` 134,
`last_trade_price` 66, `new_market` 14. `market_resolved` is documented and
handled but did not occur inside the probe window.

### 1.6 Finding — RTDS dotted topic aliases do not exist

Subscribing to `prices.crypto.chainlink.twap` (with
`{"symbol":"btc/usd","window":30}`) or `prices.crypto.chainlink` was
answered with:

```json
{"message": "Invalid request body", "connectionId": "…", "requestId": "…"}
```

and delivered nothing. Only the raw topic names work:

```
crypto_prices              Binance      (symbol "btcusdt")
crypto_prices_chainlink    Chainlink    (symbol "btc/usd")
crypto_prices_twap_thirty  TWAP-30      (symbol "btc/usd", window_s 30)
crypto_prices_twap_sixty   TWAP-60      (symbol "btc/usd", window_s 60)
```

The brief permitted this — "or the corresponding raw RTDS topics" — and
`rtds.py` uses the raw names exclusively.

### 1.7 Finding — RTDS `filters` suppresses live delivery

| Subscription | Result over 45–50 s |
|---|---|
| `{"topic":"crypto_prices_chainlink","type":"update","filters":"{\"symbol\":\"btc/usd\"}"}` | 1 `type:"subscribe"` backfill frame (56 historical points), then **zero updates** |
| `{"topic":"crypto_prices_chainlink","type":"update"}` | **48 btc/usd updates** (~1 Hz), sustained |

Unfiltered, all four topics delivered 48–49 BTC updates each per 50 s.
So `rtds.py` **subscribes unfiltered and filters client-side** on
`payload.symbol`. The cost is discarding other assets' ticks (~8× volume,
still only ~30 msg/s); the benefit is receiving BTC data at all.

The subscribe-time backfill is parsed and recorded when one arrives, but is
**not** relied on for pre-round history — item 7 is satisfied by starting
the recorder before the round opens (`PRE_ROUND`), not by a dump that costs
the live stream.

### 1.8 Finding — the settlement rule contradicts the brief

Live metadata on every BTC 5-minute market:

```json
"cryptoMarketConfigId": "btc-5m-twap-60",
"cryptoMarketConfig": {"asset":"btc","duration":"5m",
                       "twapEnabled":true,"twapLookbackSeconds":60},
"resolutionSource": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
"description": "…resolve to \"Up\" if the time-weighted average price (TWAP)
   of Bitcoin, generated by Chainlink, of the time range specified in the
   title is greater than or equal to the price at the beginning of that range…"
```

The brief states the rule as `UP ⟺ ChainlinkBTC_end ≥ ChainlinkBTC_start`
using the **plain reference price**, and says TWAP is a predictive feature
only. The live markets say the settlement reference **is** the Chainlink
60-second TWAP stream.

The recorder does not pick a side. `label.py` reconstructs the label under
**both** bases for every round and reports both agreement rates:

* `declared` — the basis the market's own metadata declares
* `reference` — the brief's plain-Chainlink-reference rule

Whichever basis actually reproduces Polymarket's resolutions becomes an
empirical result of the capture rather than an assumption baked into the
recorder. Binance is never a settlement basis under either reading.

### 1.9 Observed reference payload shape

```json
{"connection_id": "gXTNwy43YWeIKEhCjA==",
 "payload": {"full_accuracy_value": "63086406212239293939712",
             "symbol": "btc/usd",
             "timestamp": 1786771253000,      <- Chainlink observation (ms)
             "value": 63086.40621223929,
             "window_s": 60},
 "timestamp": 1786771254918,                  <- RTDS publisher (ms)
 "topic": "crypto_prices_twap_sixty", "type": "update"}
```

The two `timestamp` fields are ~1.9 s apart. Collapsing them would
attribute nearly two seconds of *oracle* latency to our own network hop.
Chainlink topics carry E18 fixed-point `full_accuracy_value`; Binance
carries a plain decimal (`"63151.96000000"`).

Binance BTCUSDT read 63151.96 while Chainlink BTC/USD read 63086.22 at the
same instant — ~65 USD, ~10 bp apart. They are genuinely different sources.

### 1.10 Finding — resolution lands 3–8 minutes after the round closes

Measured directly on settled rounds:

| Time since round close | State |
|---|---|
| 127 s | `closed=false`, still `acceptingOrders=true`, prices `["0.775","0.225"]` |
| 159 s | not closed |
| 459 s | `closed=true`, `outcomePrices ["1","0"]`, CLOB `winner` flags still all false |
| 759 s | `closed=true`, CLOB `winner=["Up"]` |

Two consequences:

* **No per-round tail window can capture the resolution** without idling the
  recorder for longer than the round itself and eating into the next round's
  `PRE_ROUND` capture. Rounds are therefore finalized on schedule with their
  reconstructed label, and the venue's outcome is filled in afterwards by a
  post-capture sweep (`resolve_pending`, or `resolve_from_store` /
  `scripts/resolve_capture_labels.py` for an already-exited capture).
* **CLOB `tokens[].winner` lags Gamma's `outcomePrices`** by minutes, so it
  is an independent cross-check rather than the primary source.

### 1.11 Finding — Gamma hides settled markets by default

`GET /markets?slug=btc-updown-5m-1786772100` returns `[]` once the round has
resolved; the same slug with `closed=true` returns the full settled payload.
This is exactly backwards from what settlement reading needs, so
`fetch_gamma_market` retries with `closed=true` before concluding the market
does not exist. Without that retry every finalization lookup for a round
that had actually settled would silently see "no market".

### 1.12 Finding — a WebSocket can stop delivering without ever raising

During a real 3-round capture the RTDS socket **stopped delivering after
~688 seconds and never raised**: `recv` simply kept timing out on an
apparently healthy connection. The CLOB stream in the same process ran to
completion (1084 s), so it was specific to RTDS.

The consequence was the worst possible failure mode for a recorder: a
clean-looking capture — `dropped_events=0`, `parse_failures=0`,
`reconnect_count=0`, 30/30 book-integrity checks passing — with **no
settlement reference data at all for the last two of three rounds**, so
their labels could not be reconstructed.

Both stream adapters now run a **stall watchdog** (default 30 s, against a
measured ~1 Hz reference cadence and ~130 msg/s book cadence). A `PONG`
counts as socket liveness but **not** as data: a server answering pings
while having silently dropped our subscriptions is precisely the case this
guards. On a stall the adapter writes a `stream_stalled` control event and
forces a reconnect, and the capture report surfaces both the stall and any
resulting reference-data gap at the settlement boundary.

Confirmed working in a subsequent capture:

```
1786775582 stream_stalled: {"stream":"rtds","silent_for_s":30.007,"stall_timeout_s":30.0}
1786775582 reconnect:      {"generation":1,"stream":"rtds"}
```

Reference data resumed immediately after the reconnect. Note this fired
within the first few minutes, so **RTDS stalls are recurring rather than a
one-off** — which is exactly why the watchdog, not a longer timeout, is the
right answer. The root cause is server-side and outside our control; what is
in our control is detecting it, recording it, and recovering.

The threshold is a genuine trade-off: too low churns connections on a quiet
market, too high loses data. 30 s is ~120x the per-stream publication
interval and ~4000x the book's, so a false positive would require every one
of the subscribed streams to go simultaneously silent for half a minute.

### 1.13 Finding — "the next round" is not far enough ahead

The same capture gave its first round only **94 seconds** of pre-round
reference history against a configured 420 s lead, because discovery took
"the next round" and that round opened 94 s later. Rounds 2 and 3 were fine.

Discovery now starts at the first round opening **at least
`pre_round_lead_s` from now**. A round whose item 7 lookback cannot be fully
covered should not be in the dataset at all rather than be quietly short.

### 1.14 Finding — the venue ends the subscription, and that is not an error

When every subscribed token settles, Polymarket closes the market channel
cleanly:

```
ConnectionClosedOK: received 1000 (OK) all subscribed assets resolved;
                    then sent 1000 (OK) all subscribed assets resolved
```

and `GET /book?token_id=…` then returns `404 Not Found` for those tokens,
because the book no longer exists.

Treating either as a fault caused a real, and badly wrong, outcome: a
capture with **zero dropped events and 30/30 book-integrity checks passing**
was marked `training-grade: False`, disqualified by 49 "parse failures"
that were entirely its own teardown. The adapter had reconnected into
settled markets for the whole 12-minute resolution sweep.

Three corrections:

* A clean close (status 1000) is recorded as `stream_closed_by_venue`; if it
  carries "all subscribed assets resolved", the reader exits rather than
  reconnecting into nothing.
* A `/book` 404 is recorded as `book_unavailable_settled`.
* The service stops both streams **before** the resolution sweep, since a
  finalized round has nothing left to capture.

The general principle: `is_training_grade()` gates model training, so
anything counted against it must be genuine evidence of missing or corrupt
market data. Normal end-of-life protocol events are not.

---

## 2. Component map

```
                    ┌───────────────────────────────────────────┐
                    │            RealRecorderService            │
                    │  (round lifecycle, orchestration, report) │
                    └───┬──────────────┬─────────────┬──────────┘
                        │              │             │
          ┌─────────────▼───┐   ┌──────▼──────┐  ┌───▼────────────┐
          │ MarketDiscovery │   │Polymarket   │  │  RTDSClient    │
          │ Gamma + CLOB    │   │MarketStream │  │ one shared WS  │
          │ metadata        │   │ CLOB WS +   │  │ 4 BTC ref      │
          │                 │   │ in-mem book │  │ streams        │
          └─────────────────┘   └──────┬──────┘  └───┬────────────┘
                                       │             │
                                  RawEvent      RawEvent
                                       │             │
                                  ┌────▼─────────────▼────┐
                                  │  RawRecorder.submit() │  non-blocking
                                  │  bounded queue (50k)  │  put_nowait
                                  └───────────┬───────────┘
                                              │
                                   ┌──────────▼──────────┐
                                   │  writer thread      │  batches of 500
                                   │  RawEventStore      │  one COMMIT/batch
                                   │  SQLite WAL         │  ns integer times
                                   └─────────────────────┘
```

Cross-cutting: `RecorderMetrics` (health + latency), `MakerCounterfactual‑
Tracker` (item 11), `label.py` (items 5/8), `freshness.py` (item 10).

---

## 3. The ingestion path

**Reader threads do three things per event**: build the immutable record,
bump counters, `put_nowait`. Nothing else. A SQLite commit is never in the
WebSocket reader's path.

**The queue is bounded (50,000) on purpose.** An unbounded queue does not
solve back-pressure; it converts a persistence stall into unbounded memory
growth and eventually loses the whole capture instead of a countable prefix.
A bounded queue that drops **and counts** is strictly more honest — and the
count is exactly what disqualifies the interval:

> A dropped-event counter greater than zero must make that capture interval
> unsuitable for high-fidelity model training unless explicitly repaired.

This is encoded as `RecorderMetrics.is_training_grade()`, not as a log line.
Parse failures and book-integrity mismatches disqualify equally.

At the measured ~130 events/s, 50,000 slots is ~6 minutes of slack — longer
than a whole round — while bounding memory to a few tens of MB.

**Writer**: batches of up to 500, or whatever has accumulated after 0.5 s,
in one transaction. `journal_mode=WAL` so the periodic integrity check can
read without stalling ingestion. `synchronous=NORMAL` is a deliberate
durability trade: it survives process crash, risking only the last
transactions on power failure, for data that can simply be recaptured.

---

## 4. Timestamps

Four clocks, kept separate, stored as **integer nanoseconds**:

| Field | Meaning |
|---|---|
| `source_timestamp_ns` | Stamped by the origin — CLOB matching engine, Chainlink observation time, Binance trade time. **The only timestamp freshness may be judged against.** |
| `publisher_timestamp_ns` | Stamped by the relay (RTDS outer `timestamp`). The gap to `source` is the oracle hop we do not control. |
| `recv_wall_timestamp_ns` | Local wall clock at frame read. Same epoch as the above, but subject to NTP steps. |
| `recv_monotonic_ns` | Local monotonic clock. **Not an epoch** — only differences are meaningful. Inter-event deltas use this so a wall-clock step cannot reorder events. |

`EventStore` (Phase 2) stores float seconds and collapses source/receive via
`event_time`. That is fine for deterministic replay and is unchanged. It is
wrong for the immutable wire record, which is why `RawEventStore` exists
alongside it rather than replacing it.

Millisecond→nanosecond conversion is **integer arithmetic**.
`float(1786771254918) * 1e6` evaluates to `1786771254918000128` — 128 ns of
fabrication, because float64 has 53 mantissa bits and the product needs 61.
Small, but it is precisely the defect this layer exists to remove.

---

## 5. Book maintenance

* `book` frames replace the ladder wholesale (snapshot).
* `price_change` elements apply per level: `side: "BUY"` → bids,
  `"SELL"` → asks; `size: 0` removes the level.
* A delta arriving with **no snapshot beneath it** is not applied. The book
  records `has_gap = True` and reports `is_quotable = False` rather than
  inventing a ladder from deltas alone.
* `tick_size_change` updates the book's tick size on the same connection.
* Reconnect increments `reconnect_generation` (stamped on every subsequent
  raw event), marks every book as awaiting a snapshot, and re-bootstraps
  from REST before applying further deltas — a gap can never be mistaken for
  a quiet book.

**REST is used for exactly three things**: bootstrap snapshot, reconnect
resync, periodic integrity check. `get_snapshot()` is a pure in-memory read.

**Integrity check** compares in-memory top-of-book against a fresh REST
snapshot. Deep-ladder differences are expected against a book updating ~130
times/second read at two different instants; a **top-of-book** disagreement
is the signal that matters for execution. A mismatch marks the interval
suspect, resnapshots from the response already in hand, and writes a
reconciliation event — it never silently continues.

---

## 6. Round lifecycle

```
DISCOVERED ──> PRE_ROUND ──> ACTIVE ──> ENDED ──┬──> RESOLVED ──> FINALIZED
                                                └──────────────────────┘
```

`ENDED → FINALIZED` directly is legal: a round can be finalized without the
venue having published a resolution, and that fact is recorded rather than
waited on indefinitely. No other state can be skipped — the transition log
must remain a truthful history.

**PRE_ROUND exists for item 7.** Default lead is **420 s**: the largest
window any current feature spans is the 300 s round itself, plus 120 s of
margin covering the 60 s TWAP's own warm-up and a slow subscribe. It is a
config field, not a constant, because "largest feature window" is a property
of the feature set and will move. Early-round momentum and TWAP history are
computed from actual pre-round observations, never approximated from `p0`.

The service discovers starting at the **next** round, not the current one:
the current round has already opened, so it could never have pre-round
history, and capturing a knowingly deficient interval would pollute the
dataset.

Tokens for every round are subscribed up front, so round N+1's PRE_ROUND
capture is already running while round N is ACTIVE. Subscribing at the
rollover instant would guarantee a gap exactly where the next round's
opening book matters most.

**On finalization**, in order: persist final market metadata → persist the
`market_resolved` event → reconstruct the label and record the boundary
Chainlink observations → close hypothetical quotes → flush buffered events →
write round metrics and the training-grade verdict.

---

## 7. Event ordering (item 13)

The Phase-2 tie-break table gave every event type its own rank, which had
two defects:

1. It ranked `ORDER_STATUS`(5) / `FILL`(6) / `CANCEL`(7) **before**
   `ORDER_SUBMIT`(8). On a timestamp tie that literally sorted an order's
   fill ahead of the submit that created it.
2. It imposed a total order on genuinely **external** observations
   (`TWAP < SPOT < BOOK_SNAPSHOT < BOOK_DELTA`). Two external events sharing
   a source timestamp are not causally ordered; "TWAP happened before the
   book update" is a causal fact fabricated from an arbitrary constant.

Now:

```
MARKET_CONFIG              rank 0   round precondition, not an observation
TWAP/SPOT/BOOK_*           rank 1   ALL EQUAL — ties fall through
ORDER_SUBMIT               rank 2   caused by data visible at that instant
ORDER_STATUS/FILL/CANCEL   rank 3   caused by their own submit
SETTLEMENT                 rank 4
```

`sort_key = (event_time, rank, recv_ts, sequence)` — when external events tie
on source time and rank, the next most defensible discriminator is which one
we actually received first, then insertion order.

`causal_sort()` additionally repairs **structurally, per `order_id`**: a
venue can stamp a fill at or before its own acknowledged submit (matching
engine vs order gateway clock skew). Ranking cannot fix that — timestamp
order would still emit the fill first — so the submit is moved to
immediately before the earliest reaction that jumped ahead of it.

`ordering_ambiguities()` reports adjacent external pairs whose order rests on
nothing stronger than arrival, so ambiguity is counted rather than hidden.

---

## 8. Freshness (item 10)

`ShadowRunner` used to pass a literal `is_fresh=True`, which made SS21's
freshness gate and SS16's `FEED_STALE` cancel trigger unreachable. Freshness
is now computed from each input's **source** timestamp.

Using receive time for the age would report a dead feed as fresh for as long
as our own socket kept delivering anything at all. Visibility is still gated
on `recv_ts` (a live system cannot act on bytes it has not received); the two
timestamps do different jobs.

Live budgets, derived from measured publication cadence:

| Feed | Budget | Why |
|---|---|---|
| market book | 2.0 s | ~130 msg/s stream; 2 s of silence is already anomalous |
| chainlink reference | 5.0 s | ~1 Hz; allows a few missed observations |
| chainlink TWAP 30/60 | 5.0 s | same cadence |
| binance | 5.0 s | same cadence; a leading signal, not a settlement source |

Replay budgets (`REPLAY_FRESHNESS_POLICY`) are derived the same way from the
synthetic generator's declared cadence (`tick_interval_s=1.0`,
`resnapshot_interval_s=5.0`), each plus one interval of margin — not tuned
against outcomes.

**MISSING is distinct from STALE.** "Never received a Chainlink observation"
and "the last one was 6 seconds ago" must not look alike.

When a required input is missing or stale: no new ALPHA; resting paper
orders are **reviewed against an explicitly stale view** (which is what
reaches `FEED_STALE`, checked before every other cancel predicate); an
explicit reason is recorded; the failure is counted. A disconnect no longer
skips the decision point while pretending resting orders remain supervised.

---

## 9. Maker counterfactuals (item 11)

`ExecutionSimulator.draw_maker_fill` answers "did this fill during its TTL?"
with one Bernoulli draw against an uncalibrated `rho`, then applies the whole
remaining quantity at expiry. That is fine as a synthetic stand-in and is
**not** carried into real-data evaluation as ground truth.

`counterfactual.py` records **no fills at all**. Per hypothetical quote it
captures: submit timestamp, side, price, quantity, TTL, best bid/ask at
submit, queue-ahead estimate, subsequent book deltas, subsequent trade
prints, first-touch timestamp, first-cross timestamp, cumulative traded
volume at/through the quote, book size ahead over time, cancel/replace
timestamp, `q` at submit and over the lifetime.

Definitions:

* **first-touch** — best bid **≤** quote price: our price is now top of the
  visible bid ladder. A best bid *higher* than ours means we are queued
  behind someone better, which is the opposite of at-the-touch.
* **first-cross** — best ask ≤ quote price: visible liquidity is offered at
  or below our bid, so a resting order here was marketable.

`cross_observed` is **evidence a fill was achievable**, not an assertion one
happened. The summary deliberately contains no `fill_rate`, `rho` or
`q_fill`. Estimating `ρ = P(fill | state, price, queue, horizon)` and
`q_fill` from this data is explicitly **out of scope for Phase 12C**.

This also supplies the real `distance_to_touch_ticks` and
`queue_ahead_shares` that replace the hardcoded zeros.

---

## 10. Replacement provenance (item 12)

A replacement is a **new order with a new thesis**. `ReplacementPlan` now
carries `g_after_if_fill`, `fair_value` and `q` from its own re-evaluation,
and `TradingSession` builds the replacement `TrackedOrder` from those.

Previously `fair_value_at_submit` was copied from the canceled order — a
stale fair value priced against a `q` that had since moved, which is exactly
the drift that triggered the REPLACE — and `g_after_if_fill_at_submit` was
the re-evaluation of the *old* order's price/size. Every downstream cancel
predicate (edge failure, risk breach, fair-value drift) was therefore judged
against a thesis the new order never had.

---

## 11. What Phase 12C deliberately does not do

* No authenticated trading; no private key; no order, cancel or replacement
  sent to Polymarket (item 14).
* No `q` model fitted.
* No maker fill probability or `q_fill` calibrated.
* No strategy parameter tuned (`lambda_g`, regime penalties, maker
  parameters, edge thresholds, hysteresis, BUFFER_BUILD).
* No profitability claim.

The next approval gate decides whether the captured dataset is trustworthy
enough for **real walk-forward model calibration and economic shadow
evaluation** — not whether synthetic PnL improved.
