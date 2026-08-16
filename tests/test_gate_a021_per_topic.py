"""Gate A.0.2.1 - per-topic RTDS continuity closure.

The hole A.0.2 left
-------------------
A.0.2 made feed outages real intervals inside the eligibility gate, but it
asked the question of the SOCKET. The RTDS subscription is deliberately
unfiltered - filtered subscriptions do not deliver - so the socket carries
every asset the venue publishes. ETH and SOL ticks at 20/s keep a
socket-level liveness mark perfectly fresh while every BTC observation has
vanished.

The strategy does not consume "the RTDS socket". It consumes four
independent series, and reconstructs the label from one specific one of
them. A TWAP-60 blackout with the other three healthy is invisible in the
aggregate and fatal to the label.

Every test here drives the production `RTDSClient` or the production
eligibility path.
"""
from __future__ import annotations

import json

import pytest

from xamarinbot.eligibility import Disqualifier
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.gate_a import build_gate_a_dataset
from xamarinbot.realtime import continuity
from xamarinbot.realtime.attribution import AttributionStatus, RoundWindow, attribute_gap
from xamarinbot.realtime.preflight import evaluate_round, topic_gap_labels
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.realtime.rtds import (
    TOPIC_BINANCE,
    TOPIC_CHAINLINK,
    TOPIC_TWAP_30,
    TOPIC_TWAP_60,
    RTDSClient,
)

from tests.test_gate_a01_closure import make_eligible_capture
from tests.test_real_projection import END_NS, ROUND, START_NS, make_capture

FS = FeatureSet("gate_a021", base=("z_gap",))
S = 1_000_000_000
ALL_TOPICS = (TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30, TOPIC_TWAP_60)


def frame(topic: str, ts_ms: int, value: float = 63000.0, symbol: str | None = None) -> str:
    """A real RTDS reference update, in the wire shape captured live."""
    sym = symbol or ("btcusdt" if topic == TOPIC_BINANCE else "btc/usd")
    return json.dumps({
        "topic": topic, "type": "update", "timestamp": ts_ms,
        "payload": {"symbol": sym, "timestamp": ts_ms, "value": value,
                    "window_s": 60 if topic == TOPIC_TWAP_60 else 30},
    })


def client(gaps: list | None = None) -> RTDSClient:
    return RTDSClient(builder=RawEventBuilder(session_id="t"),
                      on_raw_event=lambda e: None,
                      on_data_gap=(gaps.append if gaps is not None else None))


def freshness(c: RTDSClient) -> dict[str, int | None]:
    return {t: tr.last_data_ns for t, tr in c.topic_gaps.items()}


# ===== 1: unrelated-asset traffic cannot reset BTC topic freshness =====

def test_unrelated_asset_traffic_cannot_reset_btc_topic_freshness():
    """THE bug. The socket is unfiltered, so ETH/SOL ticks arrive
    constantly. Under A.0.2 each one refreshed the single aggregate
    liveness mark; the BTC feeds could all be dead and the watchdog would
    never fire."""
    c = client()
    before = freshness(c)
    assert all(v is None for v in before.values())

    for i in range(50):
        c.handle_message(frame(TOPIC_BINANCE, 1000 + i, symbol="ethusdt"))
        c.handle_message(frame(TOPIC_CHAINLINK, 1000 + i, symbol="sol/usd"))

    assert freshness(c) == before, (
        "not one BTC series may be marked fresh by another asset's ticks"
    )


def test_a_pong_cannot_reset_any_btc_topic_freshness():
    c = client()
    c.handle_message("PONG")
    c.handle_message("PING")
    assert all(v is None for v in freshness(c).values())


# ===== 2+3: one BTC topic cannot vouch for another =====

def test_binance_btc_traffic_cannot_reset_twap60_freshness():
    c = client()
    c.handle_message(frame(TOPIC_BINANCE, 1000))
    marks = freshness(c)
    assert marks[TOPIC_BINANCE] is not None
    assert marks[TOPIC_TWAP_60] is None
    assert marks[TOPIC_TWAP_30] is None
    assert marks[TOPIC_CHAINLINK] is None


def test_twap30_traffic_cannot_close_a_twap60_gap():
    gaps = []
    c = client(gaps)
    c.handle_message(frame(TOPIC_TWAP_60, 1000))
    c.handle_message(frame(TOPIC_TWAP_30, 1000))
    c.topic_gaps[TOPIC_TWAP_60].begin("topic_stalled")

    for i in range(20):
        c.handle_message(frame(TOPIC_TWAP_30, 2000 + i))

    assert gaps == [], "TWAP-30 activity says nothing about TWAP-60"
    assert c.topic_gaps[TOPIC_TWAP_60].open_gap is not None
    assert c.topic_gaps[TOPIC_TWAP_30].open_gap is None


def test_note_data_on_one_topic_closes_only_that_topics_gap():
    """The required invariant, stated directly."""
    gaps = []
    c = client(gaps)
    for t in ALL_TOPICS:
        c.handle_message(frame(t, 1000))
    for t in ALL_TOPICS:
        c.topic_gaps[t].begin("topic_stalled")

    c.handle_message(frame(TOPIC_CHAINLINK, 5000))

    assert len(gaps) == 1
    assert gaps[0].wire_topic == TOPIC_CHAINLINK
    still_open = [t for t in ALL_TOPICS if c.topic_gaps[t].open_gap is not None]
    assert set(still_open) == {TOPIC_BINANCE, TOPIC_TWAP_30, TOPIC_TWAP_60}


# ===== 4+5: a topic gap spans that topic's own observations =====

def test_a_twap60_gap_opens_from_twap60s_own_last_valid_observation():
    gaps, events = [], []
    c = RTDSClient(builder=RawEventBuilder(session_id="t"),
                   on_raw_event=events.append, on_data_gap=gaps.append)
    c.handle_message(frame(TOPIC_TWAP_60, 1000))
    twap60_recv = events[-1].recv_wall_timestamp_ns
    # other series keep publishing throughout
    for i in range(10):
        c.handle_message(frame(TOPIC_BINANCE, 2000 + i))

    c.topic_gaps[TOPIC_TWAP_60].begin("topic_stalled")
    c.handle_message(frame(TOPIC_TWAP_60, 40_000))

    assert len(gaps) == 1
    assert gaps[0].last_data_ns == twap60_recv, (
        "the outage starts at TWAP-60's own last observation, not at any "
        "other series' activity"
    )
    assert gaps[0].expected_symbol == "btc/usd"


def test_a_twap60_gap_closes_only_on_a_new_valid_twap60_observation():
    gaps = []
    c = client(gaps)
    c.handle_message(frame(TOPIC_TWAP_60, 1000))
    c.topic_gaps[TOPIC_TWAP_60].begin("topic_stalled")

    c.handle_message("PONG")
    c.handle_message(frame(TOPIC_BINANCE, 2000))
    c.handle_message(frame(TOPIC_TWAP_60, 3000, symbol="ethusdt"))   # wrong symbol
    assert gaps == []

    c.handle_message(frame(TOPIC_TWAP_60, 4000))
    assert len(gaps) == 1 and gaps[0].wire_topic == TOPIC_TWAP_60


def test_a_frame_with_an_unusable_value_does_not_close_a_gap():
    gaps = []
    c = client(gaps)
    c.handle_message(frame(TOPIC_TWAP_60, 1000))
    c.topic_gaps[TOPIC_TWAP_60].begin("topic_stalled")
    c.handle_message(json.dumps({
        "topic": TOPIC_TWAP_60, "type": "update", "timestamp": 2000,
        "payload": {"symbol": "btc/usd", "timestamp": 2000, "value": "not-a-number"},
    }))
    assert gaps == [], "an unparseable value is not a resumed observation"


# ===== 3 (watchdog) + 6: four independent stalls =====

def test_the_watchdog_evaluates_each_topic_against_its_own_last_observation():
    """The spec's first regression:

        other assets      20 msg/s, continuing
        Binance BTC       continuing
        Chainlink BTC     continuing
        TWAP-30           continuing
        TWAP-60           stops for 35 seconds

    Expected: TWAP-60 gap detected, other three healthy, socket untouched.
    """
    gaps = []
    c = client(gaps)
    now = START_NS
    for t in ALL_TOPICS:
        c.topic_gaps[t].last_data_ns = now
    # three series stay current; TWAP-60 does not
    later = now + 35 * S
    for t in (TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30):
        c.topic_gaps[t].last_data_ns = later

    stale = c._check_topic_liveness(now_ns=later)

    assert stale == [TOPIC_TWAP_60]
    assert c.topic_gaps[TOPIC_TWAP_60].open_gap is not None
    for t in (TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30):
        assert c.topic_gaps[t].open_gap is None, f"{t} must stay healthy"


def test_all_four_btc_topics_stall_while_unrelated_assets_continue():
    """The spec's second regression: continuous raw socket traffic must not
    prevent four BTC outages from opening."""
    gaps = []
    c = client(gaps)
    now = START_NS
    for t in ALL_TOPICS:
        c.topic_gaps[t].last_data_ns = now
    # unrelated assets flood the socket for 40s - and change nothing
    for i in range(800):
        c.handle_message(frame(TOPIC_BINANCE, 1000 + i, symbol="ethusdt"))

    stale = c._check_topic_liveness(now_ns=now + 40 * S)
    assert set(stale) == set(ALL_TOPICS)


def test_four_independently_stalled_topics_produce_four_identifiable_records():
    """Acceptance 6: four distinct gap records, each naming its series - not
    one undifferentiated 'RTDS outage'."""
    gaps = []
    c = client(gaps)
    now = START_NS
    for t in ALL_TOPICS:
        c.topic_gaps[t].last_data_ns = now
    c._check_topic_liveness(now_ns=now + 40 * S)
    # each recovers separately
    for i, t in enumerate(ALL_TOPICS):
        c.handle_message(frame(t, 50_000 + i))

    assert len(gaps) == 4
    assert sorted(g.wire_topic for g in gaps) == sorted(ALL_TOPICS)
    for g in gaps:
        assert g.expected_symbol in ("btc/usd", "btcusdt")


def test_a_topic_stall_does_not_drop_the_socket():
    """A TWAP-60 outage must not interrupt three healthy feeds. Only the
    socket-liveness watchdog breaks the connection."""
    import inspect

    src = inspect.getsource(RTDSClient._run_forever)
    check = src.index("self._check_topic_liveness()")
    socket_break = src.index("if time.monotonic() - last_data > self._stall_timeout:")
    assert check < socket_break
    # the per-topic check is not followed by a break of its own
    between = src[check:socket_break]
    assert "break" not in between


def test_a_never_seen_topic_is_not_immediately_stale():
    c = client()
    assert c._check_topic_liveness(now_ns=START_NS + 10_000 * S) == []
    c.mark_connected(at_ns=START_NS)
    assert c._check_topic_liveness(now_ns=START_NS + 1 * S) == []
    assert set(c._check_topic_liveness(now_ns=START_NS + 40 * S)) == set(ALL_TOPICS)


# ===== 4: the topic is recorded on the gap =====

def test_the_gap_record_names_the_series_and_symbol():
    gaps = []
    c = client(gaps)
    c.handle_message(frame(TOPIC_TWAP_60, 1000))
    c.topic_gaps[TOPIC_TWAP_60].begin("topic_stalled")
    c.handle_message(frame(TOPIC_TWAP_60, 40_000))

    att = attribute_gap(gaps[0], [
        RoundWindow("r1", "0x1", "u", "d", START_NS, END_NS),
    ])
    payload = att.as_payload()
    assert payload["stream"] == "rtds"
    assert payload["wire_topic"] == TOPIC_TWAP_60
    assert payload["expected_symbol"] == "btc/usd"
    assert payload["failure_kind"] == "topic_stalled"
    assert payload["attribution_status"] == AttributionStatus.GLOBAL_WINDOW.value
    assert payload["interval_start_ns"] and payload["interval_end_ns"]


# ===== 7: an isolated topic gap invalidates the right rounds =====

def test_an_isolated_twap60_gap_invalidates_only_overlapping_rounds():
    gap_list = []
    c = client(gap_list)
    c.handle_message(frame(TOPIC_TWAP_60, 1000))
    c.topic_gaps[TOPIC_TWAP_60].begin("topic_stalled")
    c.handle_message(frame(TOPIC_TWAP_60, 40_000))
    gap = gap_list[0]

    near = RoundWindow("near", "0x1", "u", "d",
                       gap.last_data_ns, gap.last_data_ns + 300 * S)
    far = RoundWindow("far", "0x2", "u", "d",
                      gap.last_data_ns + 100_000 * S,
                      gap.last_data_ns + 100_300 * S)
    att = attribute_gap(gap, [near, far])
    assert "near" in att.affected_round_ids
    assert "far" not in att.affected_round_ids


# ===== 8: a healthy topic does not inherit another's gap =====

def test_a_healthy_topic_does_not_inherit_another_topics_gap():
    gaps = []
    c = client(gaps)
    for t in ALL_TOPICS:
        c.handle_message(frame(t, 1000))
    c.topic_gaps[TOPIC_TWAP_60].begin("topic_stalled")

    for i in range(30):
        for t in (TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30):
            c.handle_message(frame(t, 2000 + i))

    assert gaps == []
    for t in (TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30):
        assert c.topic_gaps[t].open_gap is None
    assert c.topic_gaps[TOPIC_TWAP_60].open_gap is not None


# ===== 9: offline revalidation of pre-existing captures =====

def _sparse_topic_capture(tmp_path, *, hole_topic: Topic, hole_s: float):
    """A capture where every series publishes at 1 Hz except one, which has
    a hole. No watchdog event is written - the point is that the audit finds
    the outage from the DATA, as it must for captures recorded before the
    per-topic watchdog existed."""
    raw = RawEventStore(str(tmp_path / "raw.db"))
    raw.upsert_round({
        "round_id": ROUND, "session_id": "s", "condition_id": "0xc",
        "start_ts_ns": START_NS, "end_ts_ns": END_NS,
        "up_token_id": "u", "down_token_id": "d", "state": "FINALIZED",
        "settlement_kind": "chainlink_twap", "twap_window_s": 60,
    })
    b = RawEventBuilder(session_id="s")
    events = []
    import dataclasses

    for wire, topic in continuity.REQUIRED_TOPICS.items():
        t = 0.0
        while t < 300.0:
            at = START_NS + int(t * 1e9)
            ev = b.build(topic, "update", {"payload": {"value": 1.0}},
                         round_id=ROUND, source_timestamp_ns=at)
            events.append(dataclasses.replace(ev, recv_wall_timestamp_ns=at))
            # the hole applies to one series only, mid-round
            t += hole_s if (topic is hole_topic and 100.0 <= t < 100.0 + 1.0) else 1.0
    raw.write_batch(events)
    return raw


def test_the_offline_audit_finds_a_twap60_only_gap_with_no_watchdog_event(tmp_path):
    """Acceptance 9. This is how LEGACY / POST_A0_1 / POST_A0_2 captures are
    revalidated: the per-topic timelines were always stored separately, only
    the analysis was aggregate."""
    raw = _sparse_topic_capture(tmp_path, hole_topic=Topic.RTDS_TWAP_60, hole_s=35.0)
    audit = continuity.audit_capture(raw)

    assert audit[TOPIC_TWAP_60].gap_count == 1
    assert audit[TOPIC_TWAP_60].longest_gap_s == pytest.approx(35.0, abs=0.1)
    for other in (TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30):
        assert audit[other].gap_count == 0, f"{other} must be reported healthy"


def test_the_offline_audit_attributes_that_gap_to_the_round(tmp_path):
    raw = _sparse_topic_capture(tmp_path, hole_topic=Topic.RTDS_TWAP_60, hole_s=35.0)
    affected = continuity.rounds_affected_by_topic_gaps(raw)
    assert affected == {ROUND: [f"rtds:{TOPIC_TWAP_60}"]}
    assert topic_gap_labels(raw, ROUND) == [f"rtds:{TOPIC_TWAP_60}"]


def test_the_offline_audit_reports_both_clocks_separately(tmp_path):
    raw = _sparse_topic_capture(tmp_path, hole_topic=Topic.RTDS_TWAP_60, hole_s=35.0)
    c = continuity.audit_capture(raw)[TOPIC_BINANCE]
    assert c.source is not None and c.recv is not None
    assert c.source.clock == "source" and c.recv.clock == "recv"
    for ia in (c.source, c.recv):
        for field in ("median", "p95", "p99", "p999", "max"):
            assert getattr(ia, field) == pytest.approx(1.0, abs=0.01)


def test_a_clean_capture_reports_no_topic_gaps(tmp_path):
    """The audit must be able to say no."""
    raw = _sparse_topic_capture(tmp_path, hole_topic=Topic.RTDS_TWAP_60, hole_s=1.0)
    assert continuity.rounds_affected_by_topic_gaps(raw) == {}


# ===== 6: the threshold is documented, sensitivity is reported =====

def test_the_threshold_is_reported_across_a_range_not_asserted(tmp_path):
    raw = _sparse_topic_capture(tmp_path, hole_topic=Topic.RTDS_TWAP_60, hole_s=35.0)
    sens = continuity.threshold_sensitivity(raw)
    assert set(sens) == set(continuity.SENSITIVITY_THRESHOLDS_S)
    # the 35s hole is found at every threshold below it and at none above
    assert sens[10.0]["gaps_by_topic"][TOPIC_TWAP_60] == 1
    assert sens[60.0]["gaps_by_topic"][TOPIC_TWAP_60] == 0


def test_the_default_threshold_sits_in_the_measured_empty_band():
    """Item 6: the constant is a documented data-quality ASSUMPTION, and it
    happens to be non-load-bearing. Measured over ~191k real interarrivals,
    nothing at all falls between 10s and 25s on any series, so every
    threshold in that band gives identical verdicts."""
    assert 10.0 <= continuity.DEFAULT_GAP_THRESHOLD_S < 25.0
    assert continuity.DEFAULT_GAP_THRESHOLD_S in continuity.SENSITIVITY_THRESHOLDS_S
    doc = continuity.__doc__ or ""
    assert "never selected by how many rounds it leaves eligible" in doc


def test_quantiles_are_real_observations_not_interpolations():
    xs = [1.0, 1.0, 1.0, 9.0]
    assert continuity.quantile(sorted(xs), 0.5) in xs
    assert continuity.quantile(sorted(xs), 0.99) in xs


# ===== 10: DATA_GAP and PARSE_FAILURE are separate reasons =====

def test_data_gap_and_parse_failure_are_distinct_eligibility_reasons(tmp_path):
    """Item 7. A.0.2 routed gaps through `parse_failure_count`, so a round
    excluded for a 32-second TWAP-60 blackout reported `parse_failures` -
    when nothing had failed to parse."""
    raw = make_eligible_capture(tmp_path)
    b = RawEventBuilder(session_id="pf")
    from xamarinbot.realtime.attribution import Stream, attribute_failure

    att = attribute_failure(
        stream=Stream.CLOB, failure_kind="ValueError",
        recv_timestamp_ns=START_NS + 10 * S, raw="{}",
        windows=[RoundWindow(ROUND, "0xcond", "up-token", "down-token",
                             START_NS, END_NS)],
    )
    raw.write_batch([b.build(
        Topic.RECORDER_CONTROL, "parse_failure", att.as_payload(),
    )])
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP", "reconstructed_outcome": "UP",
        "label_agreement": 1,
        "metrics_json": json.dumps({"session_metrics": {
            "dropped_events": 0, "parse_failures": 1,
        }}),
    })
    rec = evaluate_round(raw, ROUND, verify_projection_run=False)
    assert Disqualifier.PARSE_FAILURES in rec.data_disqualifiers
    assert Disqualifier.DATA_GAP not in rec.data_disqualifiers


def test_a_topic_outage_is_reported_as_data_gap_naming_the_topic(tmp_path):
    raw = _sparse_topic_capture(tmp_path, hole_topic=Topic.RTDS_TWAP_60, hole_s=35.0)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP", "reconstructed_outcome": "UP",
        "label_agreement": 1,
        "metrics_json": json.dumps({"session_metrics": {
            "dropped_events": 0, "parse_failures": 0,
        }}),
    })
    rec = evaluate_round(raw, ROUND, verify_projection_run=False)
    assert Disqualifier.DATA_GAP in rec.data_disqualifiers
    assert Disqualifier.PARSE_FAILURES not in rec.data_disqualifiers
    assert TOPIC_TWAP_60 in rec.detail[Disqualifier.DATA_GAP.value]
    assert rec.data_valid is False


def test_the_two_reasons_are_both_data_disqualifiers():
    from xamarinbot.eligibility import DATA_DISQUALIFIERS

    assert Disqualifier.DATA_GAP in DATA_DISQUALIFIERS
    assert Disqualifier.PARSE_FAILURES in DATA_DISQUALIFIERS


# ===== 11: the canonical builder still cannot be fed a fake verdict =====

def test_the_gate_a_builder_still_refuses_fabricated_eligibility(tmp_path):
    import inspect

    assert "eligibility" not in inspect.signature(build_gate_a_dataset).parameters

    raw = _sparse_topic_capture(tmp_path, hole_topic=Topic.RTDS_TWAP_60, hole_s=35.0)
    from xamarinbot.eligibility import RoundEligibility

    fake = {ROUND: RoundEligibility(
        round_id=ROUND, label_valid=True, data_valid=True,
        projection_valid=True, projection_verified=True,
    )}
    with pytest.raises(TypeError):
        build_gate_a_dataset(raw, FeatureConfig(), FS, eligibility=fake)

    result = build_gate_a_dataset(raw, FeatureConfig(), FS)
    assert result.included_rounds == []
    assert ROUND in result.excluded_rounds


# ===== 12: nothing in this pass fits a model =====

def test_no_q_model_is_fitted_anywhere_in_this_pass():
    """The recorder/eligibility path must not reach a fitter. Checked
    structurally rather than by inspection, since this is a standing
    constraint across the whole data-plumbing chapter."""
    import ast
    import pathlib

    banned = {"fit_logistic_regression", "fit_platt", "fit_isotonic",
              "build_real_examples", "build_gate_a_dataset"}
    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "xamarinbot"
    offenders = []
    for path in (root / "realtime").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name in banned:
                        offenders.append(f"{path.name}: {alias.name}")
    assert not offenders, (
        "the recorder/eligibility path must never fit or assemble a model: "
        + ", ".join(offenders)
    )
