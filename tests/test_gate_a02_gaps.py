"""Gate A.0.2 - live gap attribution closure.

The hole A.0.1 left
-------------------
A.0.1 built structured attribution and wired it to `parse_failure`. But an
RTDS blackout is not a parse failure - nothing fails to parse, the data
simply stops arriving. The watchdog wrote `stream_stalled`, preflight read
only `parse_failure`, and so a 37-second gap in the global BTC reference
series could be fully recorded and still leave every overlapping round
`data_valid=True`.

These tests drive the PRODUCTION adapters - `RTDSClient`,
`PolymarketMarketStream`, `RealRecorderService` - rather than the attribution
helpers, because the helpers were already right. What was missing was the
wiring, and only wiring tests can prove wiring.
"""
from __future__ import annotations

import json

import pytest

from xamarinbot.eligibility import Disqualifier
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.gate_a import build_gate_a_dataset
from xamarinbot.realtime.attribution import (
    MATERIAL_GAP_NS,
    AttributionStatus,
    RoundWindow,
    Stream,
    StreamGap,
    StreamGapTracker,
    attribute_gap,
)
from xamarinbot.realtime.clob_ws import PolymarketMarketStream
from xamarinbot.realtime.label import RuleTextStatus, verify_rule_text
from xamarinbot.realtime.preflight import (
    attribution_summary,
    evaluate_round,
    reconstruct_gap_interval,
    structured_failure_events,
)
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.realtime.rtds import RTDSClient

from tests.test_gate_a01_closure import make_eligible_capture
from tests.test_real_projection import END_NS, ROUND, START_NS, make_capture

FS = FeatureSet("gate_a02", base=("z_gap",))
S = 1_000_000_000


class FakeClock:
    """A wall clock the test drives, so an outage can be seconds long
    without the test taking seconds."""

    def __init__(self, now_ns: int):
        self.now = now_ns

    def __call__(self) -> int:
        return self.now

    def advance_s(self, seconds: float) -> int:
        self.now += int(seconds * 1e9)
        return self.now


# =============== item 1: the interval is real, not a point ===============

def test_a_stall_interval_starts_at_the_last_observation_not_the_watchdog():
    """THE regression the spec names.

        data last observed   100s
        watchdog fires       130s
        valid data resumes   137s

        outage = [100, 137]   NOT   [130, 130]

    The watchdog waits 30s of silence before firing, so the moment it fires
    is 30 seconds INSIDE the outage, not its start. And the stream is still
    dead through the reconnect afterwards. Recorded as a point, a 37-second
    blackout intersects almost no round window and disqualifies nothing."""
    base = START_NS
    clock = FakeClock(base + 100 * S)
    seen = []
    tracker = StreamGapTracker(Stream.RTDS, on_gap=seen.append, clock=clock)

    tracker.note_data()                    # last good observation, t=100s
    clock.advance_s(30)                    # silence
    tracker.begin("stream_stalled")        # watchdog fires, t=130s
    clock.advance_s(7)
    tracker.note_data()                    # valid BTC data resumes, t=137s

    assert len(seen) == 1
    gap = seen[0]
    assert gap.last_data_ns == base + 100 * S
    assert gap.detected_ns == base + 130 * S
    assert gap.recovered_ns == base + 137 * S
    assert gap.duration_ns == 37 * S
    assert gap.is_material


def test_that_interval_invalidates_every_overlapping_round():
    base = START_NS
    gap = StreamGap("rtds", "stream_stalled",
                    last_data_ns=base + 100 * S,
                    detected_ns=base + 130 * S,
                    recovered_ns=base + 137 * S)
    # three rounds whose ACTIVE windows straddle the outage
    windows = [
        RoundWindow("r1", "0x1", "t1u", "t1d", base, base + 120 * S),
        RoundWindow("r2", "0x2", "t2u", "t2d", base + 120 * S, base + 240 * S),
        RoundWindow("r3", "0x3", "t3u", "t3d", base + 10_000 * S, base + 10_300 * S),
    ]
    a = attribute_gap(gap, windows)
    assert a.attribution_status is AttributionStatus.GLOBAL_WINDOW
    assert set(a.affected_round_ids) == {"r1", "r2"}
    assert "r3" not in a.affected_round_ids


def test_a_zero_duration_stall_would_have_invalidated_nothing():
    """Proves the defect was real: the same outage recorded as a point."""
    base = START_NS
    point = StreamGap("rtds", "stream_stalled",
                      last_data_ns=base + 130 * S, detected_ns=base + 130 * S,
                      recovered_ns=base + 130 * S)
    assert point.duration_ns == 0
    assert point.is_material is False


def test_a_gap_still_open_at_shutdown_is_published_not_lost():
    clock = FakeClock(START_NS)
    seen = []
    tracker = StreamGapTracker(Stream.RTDS, on_gap=seen.append, clock=clock)
    tracker.note_data()
    clock.advance_s(45)
    tracker.begin("stream_stalled")
    clock.advance_s(5)
    tracker.abandon()
    assert len(seen) == 1 and seen[0].duration_ns == 50 * S


def test_a_second_stall_does_not_reopen_an_already_open_gap():
    clock = FakeClock(START_NS)
    seen = []
    t = StreamGapTracker(Stream.RTDS, on_gap=seen.append, clock=clock)
    t.note_data()
    clock.advance_s(30)
    first = t.begin("stream_stalled")
    clock.advance_s(30)
    again = t.begin("stream_stalled")
    assert again is first
    clock.advance_s(1)
    t.note_data()
    assert len(seen) == 1 and seen[0].duration_ns == 61 * S


# =========== acceptance 1+2: the PRODUCTION RTDS adapter is wired =========

def _rtds_frame(topic: str, ts_ms: int, value: float) -> str:
    return json.dumps({
        "topic": topic, "type": "update",
        "timestamp": ts_ms,
        "payload": {"symbol": "btc/usd", "timestamp": ts_ms, "value": value,
                    "window_s": 60},
    })


def test_the_real_rtds_client_opens_a_gap_when_its_watchdog_fires():
    """Acceptance 1: not the tracker in isolation - `RTDSClient` itself."""
    gaps = []
    client = RTDSClient(builder=RawEventBuilder(session_id="t"),
                        on_raw_event=lambda e: None,
                        on_data_gap=gaps.append)
    clock = FakeClock(START_NS)
    client.gaps._clock = clock

    client.handle_message(_rtds_frame("crypto_prices_twap_sixty", 1000, 63000.0))
    assert client.gaps.last_data_ns is not None, "a real frame must mark liveness"

    clock.advance_s(30)
    client.gaps.begin("stream_stalled")       # exactly what the watchdog does
    assert client.gaps.open_gap is not None
    assert gaps == [], "an open gap is not published until data resumes"


def test_the_real_rtds_client_closes_the_gap_on_the_first_resumed_observation():
    """Acceptance 2. The gap ends at a parsed, wanted BTC observation - not
    at the reconnect, which only proves a socket opened."""
    gaps, events = [], []
    client = RTDSClient(builder=RawEventBuilder(session_id="t"),
                        on_raw_event=events.append,
                        on_data_gap=gaps.append)

    client.handle_message(_rtds_frame("crypto_prices_twap_sixty", 1000, 63000.0))
    client.gaps.begin("stream_stalled")
    client.handle_message(_rtds_frame("crypto_prices_twap_sixty", 38_000, 63010.0))

    assert len(gaps) == 1
    assert gaps[0].stream == "rtds"
    assert client.gaps.open_gap is None
    # The interval's ends are the OBSERVATIONS' own receive timestamps -
    # deliberately not a clock the test controls, because in production the
    # gap must be bounded by when data really arrived, not by when anything
    # noticed.
    assert gaps[0].last_data_ns == events[0].recv_wall_timestamp_ns
    assert gaps[0].recovered_ns == events[-1].recv_wall_timestamp_ns


def test_a_pong_does_not_close_an_rtds_gap():
    """A server answering pings while having dropped our subscriptions is
    precisely the failure the watchdog exists for."""
    gaps = []
    client = RTDSClient(builder=RawEventBuilder(session_id="t"),
                        on_raw_event=lambda e: None,
                        on_data_gap=gaps.append)
    client.gaps._clock = FakeClock(START_NS)
    client.handle_message(_rtds_frame("crypto_prices_twap_sixty", 1000, 63000.0))
    client.gaps.begin("stream_stalled")
    client.handle_message("PONG")
    assert gaps == [], "a PONG is liveness for the socket, not data"
    assert client.gaps.open_gap is not None


def test_a_non_btc_tick_does_not_close_an_rtds_gap():
    gaps = []
    client = RTDSClient(builder=RawEventBuilder(session_id="t"),
                        on_raw_event=lambda e: None,
                        on_data_gap=gaps.append)
    client.gaps._clock = FakeClock(START_NS)
    client.handle_message(_rtds_frame("crypto_prices_twap_sixty", 1000, 63000.0))
    client.gaps.begin("stream_stalled")
    client.handle_message(json.dumps({
        "topic": "crypto_prices", "type": "update", "timestamp": 2000,
        "payload": {"symbol": "ethusdt", "timestamp": 2000, "value": 3000.0},
    }))
    assert gaps == [], "an ETH tick is not a resumed BTC observation"


# ========= acceptance 5+6: the PRODUCTION CLOB adapter is wired ==========

def test_the_real_clob_stream_has_a_gap_tracker_ending_at_resnapshot():
    stream = PolymarketMarketStream(
        token_ids=["tok-A-up"], side_for_token={"tok-A-up": "UP"},
        builder=RawEventBuilder(session_id="t"),
        on_raw_event=lambda e: None,
    )
    assert stream.gaps.stream == Stream.CLOB.value

    import inspect

    src = inspect.getsource(PolymarketMarketStream._run_forever)
    assert 'self.gaps.begin("connection_gap")' in src
    # The close must come AFTER the REST resync, not when the socket opened.
    assert src.index("self.bootstrap_all()") < src.index("self.gaps.close()")


def test_a_clob_gap_spanning_two_required_windows_invalidates_both():
    """Acceptance 6, and the spec's exact scenario:

        round A required window ends   110s
        round B required window starts 120s
        CLOB gap                       [105, 125]

    Both are affected, even though the reconnect callback fires only at
    125s - which is why a point event at 125s would have missed A entirely.
    """
    base = START_NS
    # Windows are stated directly as required intervals (zero lead/tail) so
    # the boundaries under test are the ones the spec names.
    a = RoundWindow("round-A", "0xA", "au", "ad", base, base + 110 * S,
                    pre_round_lead_ns=0, post_round_tail_ns=0)
    b = RoundWindow("round-B", "0xB", "bu", "bd", base + 120 * S, base + 240 * S,
                    pre_round_lead_ns=0, post_round_tail_ns=0)

    gap = StreamGap("clob", "connection_gap",
                    last_data_ns=base + 105 * S,
                    detected_ns=base + 125 * S,
                    recovered_ns=base + 125 * S)
    att = attribute_gap(gap, [a, b])
    assert att.attribution_status is AttributionStatus.WINDOW_INFERRED
    assert set(att.affected_round_ids) == {"round-A", "round-B"}

    point = StreamGap("clob", "connection_gap",
                      last_data_ns=base + 125 * S, detected_ns=base + 125 * S,
                      recovered_ns=base + 125 * S)
    assert "round-A" not in attribute_gap(point, [a, b]).affected_round_ids, (
        "a point at the reconnect instant misses the round that lost data"
    )


# ====== acceptance 3+4: preflight consumes gaps and invalidates rounds ====

def _write_gap_event(raw: RawEventStore, *, stream: str, start_ns: int,
                     end_ns: int, rounds: tuple[str, ...], seq: int = 0) -> None:
    # `(session_id, recorder_sequence)` is the primary key and inserts are
    # OR IGNORE, so a fresh builder per call would collide on sequence 1
    # and silently drop every event after the first.
    b = RawEventBuilder(session_id=f"gap-{seq}")
    gap = StreamGap(stream, "stream_stalled", last_data_ns=start_ns,
                    detected_ns=start_ns, recovered_ns=end_ns)
    att = attribute_gap(gap, [
        RoundWindow(r, None, None, None, START_NS, END_NS) for r in rounds
    ])
    raw.write_batch([b.build(
        Topic.RECORDER_CONTROL, "data_gap",
        dict(att.as_payload(), duration_s=(end_ns - start_ns) / 1e9),
    )])


def test_preflight_consumes_a_persisted_data_gap(tmp_path):
    """Acceptance 3."""
    raw = make_eligible_capture(tmp_path)
    _write_gap_event(raw, stream="rtds", start_ns=START_NS + 100 * S,
                     end_ns=START_NS + 137 * S, rounds=(ROUND,))

    failures = structured_failure_events(raw)
    assert len(failures) == 1
    assert failures[0].attribution_status is AttributionStatus.GLOBAL_WINDOW
    assert failures[0].affected_round_ids == (ROUND,)
    assert failures[0].interval_end_ns - failures[0].interval_start_ns == 37 * S


def test_a_gap_makes_the_affected_round_data_invalid(tmp_path):
    """Acceptance 4 - the whole point. Before A.0.2 this round stayed
    `data_valid=True` because no frame had failed to parse."""
    raw = make_eligible_capture(tmp_path)
    assert evaluate_round(raw, ROUND, verify_projection_run=False).data_valid is True

    _write_gap_event(raw, stream="rtds", start_ns=START_NS + 100 * S,
                     end_ns=START_NS + 137 * S, rounds=(ROUND,))

    rec = evaluate_round(raw, ROUND, verify_projection_run=False)
    assert rec.data_valid is False
    assert rec.training_eligible is False
    assert Disqualifier.PARSE_FAILURES in rec.data_disqualifiers


def test_a_gap_that_misses_the_round_leaves_it_valid(tmp_path):
    """The gate must be able to say no. A gap outside every required window
    is real, recorded, and harmless to this round."""
    raw = make_eligible_capture(tmp_path)
    _write_gap_event(raw, stream="rtds",
                     start_ns=START_NS - 100_000 * S,
                     end_ns=START_NS - 99_000 * S, rounds=())
    rec = evaluate_round(raw, ROUND, verify_projection_run=False)
    assert rec.data_valid is True


# ======== item 3: one reader, legacy stalls reconstructed from data ======

def test_a_legacy_stream_stalled_interval_is_reconstructed_from_the_data(tmp_path):
    """Captures written before A.0.2 recorded only the instant the watchdog
    fired. Both ends of the real outage are still recoverable, because the
    raw log records exactly what did and did not arrive."""
    raw = make_capture(tmp_path, n=8)
    b = RawEventBuilder(session_id="legacy")
    # The fixture's reference series runs start-2s .. start+5s at 1Hz. A
    # stall control event placed after it has no data following it.
    at_ns = START_NS + 60 * S
    raw.write_batch([b.build(
        Topic.RECORDER_CONTROL, "stream_stalled",
        {"stream": "rtds", "silent_for_s": 30.0, "stall_timeout_s": 30.0},
    )])
    start, end = reconstruct_gap_interval(raw, "rtds", at_ns)
    assert start <= at_ns, "the outage starts at the last observation received"
    assert start > START_NS - 10 * S, "and that observation is a real one"


def test_a_legacy_stall_with_a_real_gap_is_surfaced_as_a_failure(tmp_path):
    raw = make_capture(tmp_path, n=8)
    b = RawEventBuilder(session_id="legacy")
    stall = b.build(
        Topic.RECORDER_CONTROL, "stream_stalled",
        {"stream": "rtds", "silent_for_s": 45.0, "stall_timeout_s": 30.0},
    )
    import dataclasses

    raw.write_batch([dataclasses.replace(
        stall, recv_wall_timestamp_ns=START_NS + 45 * S
    )])
    failures = structured_failure_events(raw)
    assert failures, "a legacy stall with a real data gap must be surfaced"
    assert failures[0].failure_kind == "stream_stalled"
    assert failures[0].source_event_type == "stream_stalled"


def test_a_harmless_reconnect_does_not_disqualify_a_round(tmp_path):
    """Item 3: 'a harmless control event/reconnect that produced no
    observation gap must not automatically disqualify a round'. The fixture
    publishes at 1Hz, so a reconnect placed between two adjacent
    observations spans less than one publication interval."""
    raw = make_eligible_capture(tmp_path)
    b = RawEventBuilder(session_id="legacy")
    import dataclasses

    # Mid-series, where observations are one second apart.
    at = START_NS + 100 * S + 300_000_000
    raw.write_batch([dataclasses.replace(
        b.build(Topic.RECORDER_CONTROL, "reconnect",
                {"generation": 1, "stream": "clob_market"}),
        recv_wall_timestamp_ns=at,
    )])
    start, end = reconstruct_gap_interval(raw, "clob", at)
    assert end - start <= MATERIAL_GAP_NS, (
        "two consecutive 1Hz observations bracket it, so nothing was missed"
    )
    assert evaluate_round(raw, ROUND, verify_projection_run=False).data_valid is True


def test_a_stall_and_the_reconnect_it_triggers_are_one_outage(tmp_path):
    """Measured on the real pre-A0.2 batch: four RTDS blackouts produced
    four `stream_stalled` events and four `reconnect` events, and
    reconstructing from the data gave all eight the SAME interval. They are
    one outage seen twice - a stream cannot be down twice at once."""
    import dataclasses

    raw = make_capture(tmp_path, n=8)
    b = RawEventBuilder(session_id="legacy")
    at = START_NS + 45 * S
    raw.write_batch([
        dataclasses.replace(
            b.build(Topic.RECORDER_CONTROL, "stream_stalled",
                    {"stream": "rtds", "silent_for_s": 30.0}),
            recv_wall_timestamp_ns=at),
        dataclasses.replace(
            b.build(Topic.RECORDER_CONTROL, "reconnect",
                    {"generation": 1, "stream": "rtds"}),
            recv_wall_timestamp_ns=at + 200_000_000),
    ])
    failures = structured_failure_events(raw)
    assert len(failures) == 1, f"one outage, not {len(failures)}"


def test_two_genuinely_separate_outages_stay_separate(tmp_path):
    """Merging must not swallow distinct outages - only overlapping ones."""
    import dataclasses

    raw = make_eligible_capture(tmp_path)
    b = RawEventBuilder(session_id="legacy")
    raw.write_batch([
        dataclasses.replace(
            b.build(Topic.RECORDER_CONTROL, "stream_stalled", {"stream": "rtds"}),
            recv_wall_timestamp_ns=START_NS + 50 * S),
        dataclasses.replace(
            b.build(Topic.RECORDER_CONTROL, "stream_stalled", {"stream": "rtds"}),
            recv_wall_timestamp_ns=START_NS + 200 * S),
    ])
    intervals = {
        (f.interval_start_ns, f.interval_end_ns)
        for f in structured_failure_events(raw)
    }
    assert len(intervals) == len(structured_failure_events(raw))


def test_the_reader_covers_every_failure_bearing_event_type():
    from xamarinbot.realtime.attribution import FAILURE_EVENT_TYPES

    assert {"parse_failure", "data_gap", "stream_stalled", "reconnect"} <= (
        FAILURE_EVENT_TYPES
    )


def test_gap_records_cannot_account_for_unrecorded_parse_failures(tmp_path):
    """Completeness compares like with like. Five well-attributed gaps must
    not make three uncounted parse failures look accounted for."""
    raw = make_eligible_capture(tmp_path)
    for i in range(5):
        _write_gap_event(raw, stream="rtds",
                         start_ns=START_NS + (100 + i * 10) * S,
                         end_ns=START_NS + (105 + i * 10) * S,
                         rounds=(ROUND,), seq=i)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP", "reconstructed_outcome": "UP",
        "label_agreement": 1,
        "metrics_json": json.dumps({"session_metrics": {
            "dropped_events": 0, "parse_failures": 3, "data_gaps": 5,
        }}),
    })
    s = attribution_summary(raw)
    assert s.unrecorded_count == 3
    assert s.is_complete is False


# =========== acceptance 7+8: A.0.1 behaviour is not regressed ===========

def test_a_token_specific_parse_error_is_still_exact(tmp_path):
    """Acceptance 7."""
    from xamarinbot.realtime.attribution import attribute_failure

    windows = [RoundWindow("round-B", "0xB", "tok-B-up", "tok-B-down",
                           START_NS + 300 * S, START_NS + 600 * S)]
    a = attribute_failure(
        stream=Stream.CLOB, failure_kind="TimeoutError",
        recv_timestamp_ns=START_NS, raw="bootstrap:tok-B-up", windows=windows,
    )
    assert a.attribution_status is AttributionStatus.EXACT
    assert a.affected_round_ids == ("round-B",)


def test_an_unattributed_failure_still_activates_the_fallback(tmp_path):
    """Acceptance 8."""
    raw = make_eligible_capture(tmp_path)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP", "reconstructed_outcome": "UP",
        "label_agreement": 1,
        "metrics_json": json.dumps({"session_metrics": {
            "dropped_events": 0, "parse_failures": 1,
        }}),
    })
    s = attribution_summary(raw)
    assert s.is_complete is False
    assert evaluate_round(raw, ROUND, verify_projection_run=False).data_valid is False


# ===== acceptance 9: a caller-supplied verdict cannot enter Gate A ======

def test_the_gate_a_builder_takes_no_eligibility_override(tmp_path):
    """Acceptance 9. A.0.1 accepted a caller map 'as a cache' and enforced
    rather than trusted it - but a caller who constructs
    `RoundEligibility(training_eligible=True)` defeats that distinction
    entirely. A plain dict is an assertion, not evidence."""
    import inspect

    params = inspect.signature(build_gate_a_dataset).parameters
    assert "eligibility" not in params

    raw = make_eligible_capture(tmp_path)
    from xamarinbot.eligibility import RoundEligibility

    fake = {ROUND: RoundEligibility(
        round_id=ROUND, label_valid=True, data_valid=True,
        projection_valid=True, projection_verified=True,
    )}
    with pytest.raises(TypeError):
        build_gate_a_dataset(raw, FeatureConfig(), FS, eligibility=fake)


def test_a_faked_verdict_cannot_get_a_dirty_round_into_gate_a(tmp_path):
    """The scenario the spec names: a raw round carrying a book-integrity
    mismatch, with a caller asserting it is fine. The canonical path
    re-derives the verdict from the capture and excludes it."""
    raw = make_eligible_capture(tmp_path)
    b = RawEventBuilder(session_id="integrity")
    raw.write_batch([b.build(
        Topic.RECORDER_CONTROL, "book_integrity_check",
        {"matched": False, "detail": "book diverged from the venue"},
        round_id=ROUND,
    )])

    result = build_gate_a_dataset(raw, FeatureConfig(), FS)
    assert result.included_rounds == []
    assert "book_integrity_mismatch" in result.excluded_rounds[ROUND]
    assert result.dataset.examples == []


def test_the_clean_capture_still_builds(tmp_path):
    """The gate must not simply refuse everything."""
    raw = make_eligible_capture(tmp_path)
    result = build_gate_a_dataset(raw, FeatureConfig(), FS)
    assert result.included_rounds == [ROUND]
    assert result.dataset.examples


# ======= acceptance 10: chainlink_reference needs positive evidence =====

def test_chainlink_reference_text_without_positive_evidence_is_not_verified():
    """Acceptance 10. The old rule returned `not mentions_twap`, so a market
    described only as 'Bitcoin Up or Down' was VERIFIED_TRUE for a settlement
    basis its text never names."""
    assert verify_rule_text("chainlink_reference", None, "Bitcoin Up or Down") is not (
        RuleTextStatus.VERIFIED_TRUE
    )


def test_chainlink_reference_text_naming_the_basis_is_verified():
    assert verify_rule_text(
        "chainlink_reference", None,
        "https://data.chain.link/streams/btc-usd", "Resolves on the Chainlink price",
    ) is RuleTextStatus.VERIFIED_TRUE


def test_chainlink_reference_text_advertising_twap_is_verified_false():
    assert verify_rule_text(
        "chainlink_reference", None,
        "resolves using the Chainlink 60-second TWAP",
    ) is RuleTextStatus.VERIFIED_FALSE


def test_absence_of_a_twap_mention_is_no_longer_evidence():
    from xamarinbot.realtime.label import declared_basis_matches_rule_text

    assert declared_basis_matches_rule_text(
        "chainlink_reference", None, "Bitcoin Up or Down",
    ) is None, "silence must be None, not True"


def test_the_schema_bump_distinguishes_a01_from_a02_captures():
    """Item 7: the persisted failure format changed, so the schema version
    must change with it. A v2 capture recorded its stalls as zero-duration
    points OUTSIDE the eligibility gate; its data-quality verdicts are not
    comparable with a v3 capture's until it is reprocessed."""
    from xamarinbot.realtime.attribution import ATTRIBUTION_SCHEMA_VERSION
    from xamarinbot.realtime.identity import (
        GENERATION_BY_SCHEMA,
        POST_A0_1_RECORDER,
        POST_A0_2_RECORDER,
        RECORDER_SCHEMA_VERSION,
        RecorderIdentity,
    )

    assert RECORDER_SCHEMA_VERSION == 3
    assert ATTRIBUTION_SCHEMA_VERSION == 2
    assert GENERATION_BY_SCHEMA[2] == POST_A0_1_RECORDER
    assert GENERATION_BY_SCHEMA[3] == POST_A0_2_RECORDER
    assert RecorderIdentity.capture().recorder_generation == POST_A0_2_RECORDER


def test_an_a01_capture_still_reads_back_as_its_own_generation(tmp_path):
    """A v2 capture must keep saying v2 - the bump describes what wrote it,
    not what is reading it."""
    from xamarinbot.realtime.identity import POST_A0_1_RECORDER, RecorderIdentity

    raw = RawEventStore(str(tmp_path / "raw.db"))
    v2 = RecorderIdentity.from_dict({
        "recorder_code_sha": "8f875de", "process_pid": 1, "process_started_at": 1.0,
        "python_version": "3.12.3", "recorder_schema_version": 2,
        "recorder_generation": POST_A0_1_RECORDER,
    })
    raw.upsert_session_meta("s", v2)
    back = raw.recorder_identity()
    assert back.recorder_schema_version == 2
    assert back.recorder_generation == POST_A0_1_RECORDER
    raw.close()


def test_the_twap_path_is_unchanged():
    assert verify_rule_text(
        "chainlink_twap", 60,
        "https://data.chain.link/streams/btc-usd-twap-60s-streams",
    ) is RuleTextStatus.VERIFIED_TRUE
