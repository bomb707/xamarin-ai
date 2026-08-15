"""Phase 12C item 6: the immutable raw-event layer.

Covers the three properties the brief actually names: nanosecond-integer
timestamps kept separate, the verbatim wire payload retained, and a bounded
async queue with a batched writer that does not put a SQLite commit in the
reader's path.
"""
from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from xamarinbot.realtime.metrics import LatencyStats, RecorderMetrics
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic, ms_to_ns
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.realtime.recorder import RawRecorder, RecorderConfig


def make_event(builder, seq_payload="x", **kwargs):
    return builder.build(
        Topic.CLOB_MARKET, "book", {"v": seq_payload},
        round_id="r1", token_id="t1", source_timestamp_ns=1_000_000_000, **kwargs,
    )


# ------------------------------------------------------------- timestamps

def test_millisecond_to_nanosecond_conversion_is_exact():
    """Integer arithmetic, not float - see ms_to_ns's docstring."""
    assert ms_to_ns(1786771254918) == 1786771254918_000_000
    assert ms_to_ns("1786771382832") == 1786771382832_000_000
    # the float path would have produced ...000128 here
    assert ms_to_ns(1786771254918) % 1_000_000 == 0
    assert ms_to_ns(None) is None
    assert ms_to_ns("") is None


def test_all_four_clocks_are_stored_separately():
    b = RawEventBuilder(session_id="s")
    ev = b.build(
        Topic.RTDS_CHAINLINK, "update", {"v": 1},
        source_timestamp_ns=1_000_000_000, publisher_timestamp_ns=1_500_000_000,
    )
    assert ev.source_timestamp_ns == 1_000_000_000
    assert ev.publisher_timestamp_ns == 1_500_000_000
    assert isinstance(ev.recv_wall_timestamp_ns, int)
    assert isinstance(ev.recv_monotonic_ns, int)
    assert ev.source_to_recv_latency_ns == ev.recv_wall_timestamp_ns - 1_000_000_000
    assert ev.publisher_to_recv_latency_ns == ev.recv_wall_timestamp_ns - 1_500_000_000


def test_missing_source_timestamp_gives_none_latency_not_zero():
    b = RawEventBuilder(session_id="s")
    ev = b.build(Topic.CLOB_MARKET, "book", {"v": 1})
    assert ev.source_timestamp_ns is None
    assert ev.source_to_recv_latency_ns is None  # NOT 0, which would read as zero latency


def test_raw_wire_payload_is_kept_verbatim():
    """Item 6: "Do not discard the original wire payload after
    normalization." Byte-for-byte, including key order."""
    b = RawEventBuilder(session_id="s")
    wire = '{"z":1,"a":2,"nested":{"k":"v"}}'
    ev = b.build(Topic.CLOB_MARKET, "book", None, raw_json=wire, normalized_side="UP")
    assert ev.payload_json == wire       # exact bytes, key order intact
    assert ev.normalized_side == "UP"    # normalization sits ALONGSIDE it
    assert ev.payload == {"z": 1, "a": 2, "nested": {"k": "v"}}


def test_recorder_sequence_is_monotonic_within_a_session():
    b = RawEventBuilder(session_id="s")
    seqs = [b.build(Topic.CLOB_MARKET, "book", {"i": i}).recorder_sequence for i in range(5)]
    assert seqs == [1, 2, 3, 4, 5]


def test_reconnect_generation_is_stamped_on_every_event():
    b = RawEventBuilder(session_id="s")
    assert b.build(Topic.CLOB_MARKET, "book", {}).reconnect_generation == 0
    b.reconnect_generation += 1
    assert b.build(Topic.CLOB_MARKET, "book", {}).reconnect_generation == 1


# ------------------------------------------------------------------ store

def test_batch_write_persists_every_field_and_round_trips():
    store = RawEventStore(":memory:")
    b = RawEventBuilder(session_id="s")
    events = [
        b.build(Topic.CLOB_MARKET, "price_change", {"i": i}, round_id="r1",
                condition_id="c1", token_id="t1", source_timestamp_ns=1_000 + i,
                publisher_timestamp_ns=2_000 + i, normalized_side="UP")
        for i in range(10)
    ]
    assert store.write_batch(events) == 10
    back = store.events(round_id="r1")
    assert len(back) == 10
    assert [e.recorder_sequence for e in back] == list(range(1, 11))
    first = back[0]
    assert first.topic is Topic.CLOB_MARKET
    assert first.condition_id == "c1"
    assert first.normalized_side == "UP"
    assert first.source_timestamp_ns == 1_000
    assert first.publisher_timestamp_ns == 2_000
    store.close()


def test_store_is_append_only_and_idempotent_on_replay():
    store = RawEventStore(":memory:")
    b = RawEventBuilder(session_id="s")
    events = [make_event(b, i) for i in range(5)]
    store.write_batch(events)
    store.write_batch(events)  # replayed batch after a crash
    assert store.count() == 5
    store.close()


def test_counts_by_topic_reports_per_stream_event_counts():
    store = RawEventStore(":memory:")
    b = RawEventBuilder(session_id="s")
    store.write_batch(
        [b.build(Topic.CLOB_MARKET, "book", {}, round_id="r1") for _ in range(3)]
        + [b.build(Topic.CLOB_MARKET, "price_change", {}, round_id="r1") for _ in range(7)]
        + [b.build(Topic.RTDS_TWAP_60, "update", {}, round_id="r1") for _ in range(2)]
    )
    counts = store.counts_by_topic("r1")
    assert counts["clob_market:book"] == 3
    assert counts["clob_market:price_change"] == 7
    assert counts["rtds_twap_60:update"] == 2
    store.close()


# -------------------------------------------------------------- recorder

def test_submit_never_blocks_the_reader_on_a_commit():
    """The architectural requirement: a slow store must not stall the
    reader. With a writer that sleeps 50ms per batch, 200 submits still
    return in well under the 10s a per-event commit would have cost."""
    store = RawEventStore(":memory:")
    real_write = store.write_batch

    def slow_write(events):
        time.sleep(0.05)
        return real_write(events)

    store.write_batch = slow_write
    recorder = RawRecorder(store, cfg=RecorderConfig(batch_size=50, batch_timeout_s=0.05))
    recorder.start()
    b = RawEventBuilder(session_id="s")
    t0 = time.perf_counter()
    for i in range(200):
        assert recorder.submit(make_event(b, i)) is True
    submit_elapsed = time.perf_counter() - t0
    assert submit_elapsed < 1.0, f"reader path blocked for {submit_elapsed:.2f}s"
    recorder.stop()
    assert store.count() == 200
    store.close()


def test_full_queue_drops_and_counts_rather_than_growing_without_bound():
    store = RawEventStore(":memory:")
    metrics = RecorderMetrics()
    # No writer thread started, so the queue can only fill.
    recorder = RawRecorder(store, metrics, RecorderConfig(queue_maxsize=5))
    b = RawEventBuilder(session_id="s")
    results = [recorder.submit(make_event(b, i)) for i in range(10)]
    assert results.count(True) == 5
    assert results.count(False) == 5
    assert metrics.dropped_events == 5
    assert metrics.events_received == 10
    store.close()


def test_a_single_dropped_event_disqualifies_the_interval_for_training():
    """Item 6's rule, as data: "A dropped-event counter greater than zero
    must make that capture interval unsuitable for high-fidelity model
    training unless explicitly repaired.\""""
    m = RecorderMetrics()
    assert m.is_training_grade() is True
    m.record_dropped()
    assert m.is_training_grade() is False
    assert "dropped_events=1" in m.disqualifiers()


def test_parse_failures_and_integrity_mismatches_also_disqualify():
    m = RecorderMetrics()
    m.record_parse_failure()
    assert m.is_training_grade() is False
    m2 = RecorderMetrics()
    m2.record_integrity_check(matched=False)
    assert m2.is_training_grade() is False
    assert m2.suspect_intervals == 1


def test_duplicate_events_are_detected_and_not_double_persisted():
    store = RawEventStore(":memory:")
    metrics = RecorderMetrics()
    recorder = RawRecorder(store, metrics)
    b = RawEventBuilder(session_id="s")
    ev = b.build(Topic.CLOB_MARKET, "book", {"same": 1}, token_id="t1",
                 source_timestamp_ns=123, raw_json='{"same":1}')
    # Same wire observation redelivered on a new connection - different
    # recorder_sequence and receive time, same underlying event.
    dup = b.build(Topic.CLOB_MARKET, "book", {"same": 1}, token_id="t1",
                  source_timestamp_ns=123, raw_json='{"same":1}')
    assert ev.dedupe_key == dup.dedupe_key
    recorder.submit(ev)
    recorder.submit(dup)
    recorder.flush()
    assert metrics.duplicate_events == 1
    assert store.count() == 1
    store.close()


def test_distinct_observations_sharing_a_timestamp_are_not_duplicates():
    b = RawEventBuilder(session_id="s")
    a = b.build(Topic.CLOB_MARKET, "price_change", None, token_id="t1",
                source_timestamp_ns=123, raw_json='{"price":"0.40"}')
    c = b.build(Topic.CLOB_MARKET, "price_change", None, token_id="t1",
                source_timestamp_ns=123, raw_json='{"price":"0.41"}')
    assert a.dedupe_key != c.dedupe_key


def test_queue_high_water_mark_is_tracked():
    store = RawEventStore(":memory:")
    metrics = RecorderMetrics()
    recorder = RawRecorder(store, metrics, RecorderConfig(queue_maxsize=100))
    b = RawEventBuilder(session_id="s")
    for i in range(30):
        recorder.submit(make_event(b, i))
    assert metrics.queue_high_water >= 25
    store.close()


def test_flush_drains_before_finalization():
    store = RawEventStore(":memory:")
    recorder = RawRecorder(store, cfg=RecorderConfig(batch_size=10, batch_timeout_s=0.05))
    recorder.start()
    b = RawEventBuilder(session_id="s")
    for i in range(137):
        recorder.submit(make_event(b, i))
    assert recorder.flush(timeout_s=10.0) is True
    assert store.count() == 137
    recorder.stop()
    store.close()


def test_concurrent_readers_do_not_lose_or_corrupt_events():
    """Two feed threads share one recorder, as CLOB and RTDS do."""
    store = RawEventStore(":memory:")
    metrics = RecorderMetrics()
    recorder = RawRecorder(store, metrics, RecorderConfig(batch_size=64))
    recorder.start()

    def produce(name, n):
        b = RawEventBuilder(session_id=name)
        for i in range(n):
            recorder.submit(b.build(Topic.CLOB_MARKET, "price_change", {"i": i},
                                    round_id="r1", token_id=name,
                                    source_timestamp_ns=1_000_000 + i))

    threads = [threading.Thread(target=produce, args=(f"s{k}", 500)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    recorder.stop()
    assert store.count("r1") == 1000
    assert metrics.dropped_events == 0
    assert metrics.is_training_grade() is True
    store.close()


# --------------------------------------------------------------- metrics

def test_latency_percentiles_are_ordered_and_reproducible():
    a = LatencyStats("x")
    b = LatencyStats("x")
    for i in range(1, 1001):
        a.record(i * 1_000_000)
        b.record(i * 1_000_000)
    assert a.percentile_ns(50) <= a.percentile_ns(95) <= a.percentile_ns(99) <= a.max_ns
    assert a.as_dict() == b.as_dict()  # deterministic reservoir


def test_latency_stats_ignore_missing_samples():
    s = LatencyStats("x")
    s.record(None)
    assert s.count == 0
    assert s.mean_ns is None
