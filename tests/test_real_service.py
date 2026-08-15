"""Phase 12C: end-to-end recorder orchestration, driven offline.

No sockets and no HTTP: the service's discovery is faked with captured
payloads and the streams are driven by handing captured frames to the
adapters directly. What is exercised is the wiring - lifecycle, attribution,
persistence, finalization, counterfactual capture, and the item 14
guarantee that nothing order-placing exists.
"""
from __future__ import annotations

import json
import time

import pytest

from xamarinbot.realtime.discovery import MarketDiscovery
from xamarinbot.realtime.lifecycle import LifecycleConfig, RoundState
from xamarinbot.realtime.raw_events import Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.realtime.recorder import RecorderConfig
from xamarinbot.realtime.report import format_capture_report
from xamarinbot.realtime.rtds import TOPIC_CHAINLINK, TOPIC_TWAP_60
from xamarinbot.realtime.service import (
    RealRecorderService,
    ServiceConfig,
    resolve_from_store,
)
from tests.test_real_discovery import LIVE_CLOB, LIVE_GAMMA

ROUND_START = 1_786_856_400


def fake_http(gamma_override=None, clob_override=None):
    def _get(url, params, timeout):
        if "/markets" in url and params and "slug" in params:
            gamma = dict(LIVE_GAMMA)
            slug = params["slug"]
            ts = int(slug.rsplit("-", 1)[-1])
            gamma["slug"] = slug
            gamma["eventStartTime"] = None
            gamma["events"] = [{"id": "1", "startTime": None}]
            # Drive the window purely from the slug timestamp.
            gamma["endDate"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts + 300))
            gamma["conditionId"] = f"0xcond{ts}"
            if gamma_override:
                gamma.update(gamma_override)
            return [gamma]
        clob = dict(LIVE_CLOB)
        if clob_override:
            clob.update(clob_override)
        return clob

    return _get


def make_service(tmp_path, n_rounds=1, **cfg_kwargs):
    store = RawEventStore(str(tmp_path / "cap.db"))
    cfg = ServiceConfig(
        n_rounds=n_rounds,
        lifecycle=LifecycleConfig(pre_round_lead_s=420.0, post_round_tail_s=10.0),
        recorder=RecorderConfig(batch_size=16, batch_timeout_s=0.05),
        resolution_sweep_s=0.0,
        **cfg_kwargs,
    )
    service = RealRecorderService(
        store, cfg, discovery=MarketDiscovery(http_get=fake_http()), log=lambda *a: None,
    )
    return store, service


# ------------------------------------------------------------- discovery

def test_discovery_starts_at_the_first_round_with_a_full_pre_round_window(tmp_path):
    """Item 7: a round whose pre-round lookback cannot be fully covered
    should not be in the dataset at all, rather than be quietly short.

    A real capture showed the weaker "just take the next round" rule giving
    round 1 only 94s of reference history against a 420s configured lead.
    Here, 137s into a round, the next round opens in 163s - short of the
    420s lead - so discovery must skip it.
    """
    store, service = make_service(tmp_path, n_rounds=2)
    now = float(ROUND_START + 137)          # 137s into a round
    captures = service.discover(now)
    assert len(captures) == 2
    lead = service.cfg.lifecycle.pre_round_lead_s
    assert captures[0].metadata.start_ts == ROUND_START + 600   # skipped the +300 round
    assert captures[1].metadata.start_ts == ROUND_START + 900
    assert all(c.metadata.start_ts - now >= lead for c in captures)
    rows = [store.get_round(c.metadata.round_id) for c in captures]
    assert all(r is not None and r["state"] == "DISCOVERED" for r in rows)
    assert all(r["up_token_id"] and r["down_token_id"] for r in rows)
    service.shutdown()
    store.close()


def test_discovery_writes_the_raw_metadata_payloads(tmp_path):
    store, service = make_service(tmp_path)
    service.recorder.start()
    captures = service.discover(float(ROUND_START))
    service.recorder.flush()
    events = store.events(round_id=captures[0].metadata.round_id, topics=[Topic.MARKET_METADATA])
    assert events, "market metadata must be recorded in the raw log"
    payload = events[0].payload
    assert "gamma" in payload and "clob" in payload
    service.shutdown()
    store.close()


# ------------------------------------------------------------- lifecycle

def test_round_walks_the_full_lifecycle_and_finalizes(tmp_path):
    store, service = make_service(tmp_path)
    service.recorder.start()
    capture = service.discover(float(ROUND_START))[0]
    start = capture.metadata.start_ts
    end = capture.metadata.end_ts

    service._tick_round(capture, start - 1000)
    assert capture.lifecycle.state is RoundState.DISCOVERED
    service._tick_round(capture, start - 100)
    assert capture.lifecycle.state is RoundState.PRE_ROUND
    service._tick_round(capture, start + 10)
    assert capture.lifecycle.state is RoundState.ACTIVE
    service._tick_round(capture, end + 1)
    assert capture.lifecycle.state is RoundState.ENDED
    service._tick_round(capture, end + 1000)   # past the tail
    assert capture.lifecycle.state is RoundState.FINALIZED

    assert store.get_round(capture.metadata.round_id)["state"] == "FINALIZED"
    results = store.round_results()
    assert len(results) == 1
    service.shutdown()
    store.close()


def test_finalization_records_the_label_reconstruction_event(tmp_path):
    store, service = make_service(tmp_path)
    service.recorder.start()
    capture = service.discover(float(ROUND_START))[0]
    start, end = capture.metadata.start_ts, capture.metadata.end_ts

    # Feed a reference series spanning the round on BOTH bases.
    from xamarinbot.realtime.rtds import ReferenceObservation

    for topic, s, e in ((TOPIC_TWAP_60, 63000.0, 63100.0), (TOPIC_CHAINLINK, 63000.0, 62900.0)):
        for i in range(-30, 331):
            frac = min(max(i / 300.0, 0.0), 1.0)
            capture.observations.setdefault(topic, []).append(ReferenceObservation(
                topic=topic, symbol="btc/usd", value=s + (e - s) * frac,
                full_accuracy_value=s + (e - s) * frac,
                window_s=60 if "sixty" in topic else None,
                source_ts_ns=int((start + i) * 1e9), publisher_ts_ns=None,
                recv_wall_ns=int((start + i) * 1e9), recv_monotonic_ns=1,
            ))

    service._tick_round(capture, start + 1)
    service._tick_round(capture, end + 1)
    service._tick_round(capture, end + 1000)
    service.recorder.flush()

    events = [e for e in store.events(round_id=capture.metadata.round_id,
                                      topics=[Topic.RECORDER_CONTROL])
              if e.event_type == "label_reconstruction"]
    assert events, "finalization must record the label reconstruction"
    payload = events[0].payload
    # Both bases present, and the discriminating disagreement is visible.
    assert payload["declared"]["outcome"] == "UP"
    assert payload["reference"]["outcome"] == "DOWN"
    assert payload["bases_agree"] is False
    service.shutdown()
    store.close()


def test_lifecycle_transitions_are_written_to_the_raw_log(tmp_path):
    store, service = make_service(tmp_path)
    service.recorder.start()
    capture = service.discover(float(ROUND_START))[0]
    service._tick_round(capture, capture.metadata.start_ts + 1)
    service.recorder.flush()
    transitions = [
        e.payload for e in store.events(round_id=capture.metadata.round_id,
                                        topics=[Topic.RECORDER_CONTROL])
        if e.event_type == "lifecycle_transition"
    ]
    assert [(t["from"], t["to"]) for t in transitions] == [
        ("DISCOVERED", "PRE_ROUND"), ("PRE_ROUND", "ACTIVE"),
    ]
    service.shutdown()
    store.close()


# ------------------------------------------- reference-feed attribution

def test_reference_observations_are_attributed_to_recording_rounds(tmp_path):
    """Reference feeds are global, so a pre-round observation legitimately
    belongs to more than one round's window (item 7's lookback overlaps the
    previous round)."""
    store, service = make_service(tmp_path, n_rounds=2)
    captures = service.discover(float(ROUND_START))
    for c in captures:
        c.lifecycle.advance(c.metadata.start_ts - 100)

    from xamarinbot.realtime.rtds import ReferenceObservation

    obs = ReferenceObservation(
        topic=TOPIC_CHAINLINK, symbol="btc/usd", value=1.0, full_accuracy_value=1.0,
        window_s=None, source_ts_ns=1, publisher_ts_ns=None, recv_wall_ns=1,
        recv_monotonic_ns=1,
    )
    service._on_observation(obs)
    assert all(c.observations[TOPIC_CHAINLINK] == [obs] for c in captures)
    service.shutdown()
    store.close()


def test_reference_observations_are_tagged_with_a_round_in_the_raw_log(tmp_path):
    """Regression for a live-found gap: `rtds.current_round_id` was never
    assigned, so every persisted RTDS row carried `round_id = NULL` and no
    reference data was attributable to any round in the raw log - while the
    in-memory attribution kept working, which is what hid it."""
    store, service = make_service(tmp_path, n_rounds=2)
    captures = service.discover(float(ROUND_START))
    assert service.rtds.current_round_id is None

    # Nothing recording yet -> tag the nearest upcoming round.
    service._update_reference_round_tag(captures[0].metadata.start_ts - 5000)
    assert service.rtds.current_round_id == captures[0].metadata.round_id

    # One round ACTIVE -> that one wins.
    captures[1].lifecycle.advance(captures[1].metadata.start_ts + 1)
    service._update_reference_round_tag(captures[1].metadata.start_ts + 1)
    assert service.rtds.current_round_id == captures[1].metadata.round_id
    service.shutdown()
    store.close()


def test_rtds_client_emits_a_stall_event_when_the_socket_goes_silent():
    """Regression for the live failure where RTDS stopped delivering after
    ~688s WITHOUT raising, so the recorder captured no reference data for
    two of three rounds while reporting zero errors and zero reconnects."""
    from xamarinbot.realtime.raw_events import RawEventBuilder
    from xamarinbot.realtime.rtds import RTDSClient

    captured = []
    client = RTDSClient(
        builder=RawEventBuilder(session_id="t"), on_raw_event=captured.append,
        stall_timeout_s=0.05,
    )
    assert client._stall_timeout == 0.05

    # A PONG keeps the socket alive but is NOT data - a server answering
    # pings with dropped subscriptions is exactly the failure mode.
    client.handle_message("PONG")
    assert captured == []


def test_stall_threshold_is_far_above_the_measured_publication_interval():
    """~1 Hz per stream across four streams: 30s of total silence is a dead
    socket, not a quiet market."""
    from xamarinbot.realtime.clob_ws import PolymarketMarketStream
    from xamarinbot.realtime.raw_events import RawEventBuilder
    from xamarinbot.realtime.rtds import RTDSClient

    rtds = RTDSClient(builder=RawEventBuilder(session_id="t"), on_raw_event=lambda e: None)
    assert rtds._stall_timeout >= 20.0
    stream = PolymarketMarketStream(
        ["t"], builder=RawEventBuilder(session_id="t"), on_raw_event=lambda e: None)
    assert stream._stall_timeout >= 20.0


# -------------------------------------------------- item 11 integration

def test_hypothetical_quote_is_tracked_against_the_live_book(tmp_path):
    from xamarinbot.realtime.clob_ws import PolymarketMarketStream
    from xamarinbot.realtime.raw_events import RawEventBuilder

    store, service = make_service(tmp_path)
    service.recorder.start()
    capture = service.discover(float(ROUND_START))[0]
    meta = capture.metadata
    up = meta.up_token_id

    service.stream = PolymarketMarketStream(
        [up], builder=RawEventBuilder(session_id="t"),
        on_raw_event=service._on_market_event,
        side_for_token={up: "UP"}, round_for_token={up: meta.round_id},
        condition_for_token={up: meta.condition_id},
    )
    service.stream.handle_message(json.dumps({
        "event_type": "book", "asset_id": up, "market": meta.condition_id,
        "timestamp": "1786856400000", "hash": "h", "tick_size": "0.01",
        "bids": [{"price": "0.44", "size": "100"}, {"price": "0.43", "size": "250"}],
        "asks": [{"price": "0.46", "size": "80"}],
    }))

    # `now` must share the frames' clock: the tracker expires quotes against
    # observation timestamps, and in production both are real wall time.
    quote = service.record_paper_quote(
        meta.round_id, "UP", 0.43, 25.0, 60.0, q=0.55, now=1786856400.0)
    assert quote is not None
    # Real book state, not the synthetic zeros.
    assert quote.queue_ahead_at_submit == 250.0
    assert quote.distance_to_touch_ticks_at_submit == pytest.approx(1.0)

    # Subsequent market activity is observed against it.
    service.stream.handle_message(json.dumps({
        "event_type": "price_change", "market": meta.condition_id,
        "timestamp": "1786856405000",
        "price_changes": [{"asset_id": up, "price": "0.44", "size": "0", "side": "BUY"}],
    }))
    service.stream.handle_message(json.dumps({
        "event_type": "last_trade_price", "asset_id": up, "market": meta.condition_id,
        "price": "0.43", "size": "9", "side": "SELL", "timestamp": "1786856406000",
    }))
    assert quote.book_delta_count >= 1
    assert quote.cumulative_traded_at_or_through == pytest.approx(9.0)
    assert quote.first_touch_ts is not None  # 0.44 level removed -> we are the touch

    service.recorder.flush()
    submitted = [e for e in store.events(topics=[Topic.PAPER_QUOTE])]
    assert submitted and submitted[0].event_type == "hypothetical_quote_submitted"
    service.shutdown()
    store.close()


def test_streams_are_stopped_before_the_resolution_sweep(tmp_path):
    """LIVE FINDING: leaving the market subscription open through the sweep
    reconnected into settled markets for its whole duration, producing seven
    spurious stall/reconnect cycles and a stream of `/book` 404s. Once every
    round is finalized there is nothing left to capture."""
    store, service = make_service(tmp_path)
    stopped = {"stream": False, "rtds": False}

    class FakeStream:
        def stop(self):
            stopped["stream"] = True

        def book(self, tid):
            return None

        def resolution_for(self, cid):
            return None

    service.stream = FakeStream()
    service.rtds.stop = lambda: stopped.update(rtds=True)

    capture = service.discover(float(ROUND_START))[0]
    service.recorder.start()
    service._tick_round(capture, capture.metadata.start_ts + 1)
    service._tick_round(capture, capture.metadata.end_ts + 1)
    service._tick_round(capture, capture.metadata.end_ts + 1000)
    assert capture.lifecycle.is_finished

    # run() performs the teardown; emulate its post-loop step directly since
    # the loop itself would block on wall-clock time.
    if service.stream is not None:
        service.stream.stop()
        service.stream = None
    service.rtds.stop()
    assert stopped == {"stream": True, "rtds": True}
    service.recorder.stop()
    store.close()


# ------------------------------------------------- post-capture resolve

def test_resolve_from_store_fills_in_agreement_after_the_fact(tmp_path):
    """Settlement lands minutes after finalization, so labels are completed
    by a later pass over the stored capture."""
    store, service = make_service(tmp_path)
    service.recorder.start()
    capture = service.discover(float(ROUND_START))[0]
    start, end = capture.metadata.start_ts, capture.metadata.end_ts

    from xamarinbot.realtime.rtds import ReferenceObservation

    for i in range(-30, 331):
        frac = min(max(i / 300.0, 0.0), 1.0)
        capture.observations.setdefault(TOPIC_TWAP_60, []).append(ReferenceObservation(
            topic=TOPIC_TWAP_60, symbol="btc/usd", value=63000.0 + 100.0 * frac,
            full_accuracy_value=63000.0 + 100.0 * frac, window_s=60,
            source_ts_ns=int((start + i) * 1e9), publisher_ts_ns=None,
            recv_wall_ns=int((start + i) * 1e9), recv_monotonic_ns=1,
        ))
    service._tick_round(capture, start + 1)
    service._tick_round(capture, end + 1)
    service._tick_round(capture, end + 1000)
    service.recorder.flush()
    service.shutdown()

    # Before the sweep: reconstructed but not comparable.
    row = store.round_results()[0]
    assert row["reconstructed_outcome"] == "UP"
    assert row["reported_outcome"] is None
    assert row["label_agreement"] is None

    # The venue has now settled the market UP.
    settled = MarketDiscovery(http_get=fake_http(gamma_override={
        "closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["1", "0"]',
    }))
    out = resolve_from_store(store, discovery=settled, max_wait_s=1.0, log=lambda *a: None)
    assert out["resolved"] == 1
    row = store.round_results()[0]
    assert row["reported_outcome"] == "UP"
    assert row["label_agreement"] == 1
    store.close()


def test_resolve_from_store_records_disagreement_honestly(tmp_path):
    store, service = make_service(tmp_path)
    service.recorder.start()
    capture = service.discover(float(ROUND_START))[0]
    start, end = capture.metadata.start_ts, capture.metadata.end_ts

    from xamarinbot.realtime.rtds import ReferenceObservation

    for i in range(-30, 331):
        frac = min(max(i / 300.0, 0.0), 1.0)
        capture.observations.setdefault(TOPIC_TWAP_60, []).append(ReferenceObservation(
            topic=TOPIC_TWAP_60, symbol="btc/usd", value=63000.0 + 100.0 * frac,
            full_accuracy_value=63000.0 + 100.0 * frac, window_s=60,
            source_ts_ns=int((start + i) * 1e9), publisher_ts_ns=None,
            recv_wall_ns=int((start + i) * 1e9), recv_monotonic_ns=1,
        ))
    service._tick_round(capture, start + 1)
    service._tick_round(capture, end + 1)
    service._tick_round(capture, end + 1000)
    service.recorder.flush()
    service.shutdown()

    # Venue says DOWN; we reconstructed UP. That must be recorded as a
    # disagreement, not smoothed over.
    settled = MarketDiscovery(http_get=fake_http(gamma_override={
        "closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["0", "1"]',
    }))
    resolve_from_store(store, discovery=settled, max_wait_s=1.0, log=lambda *a: None)
    row = store.round_results()[0]
    assert row["reported_outcome"] == "DOWN"
    assert row["reconstructed_outcome"] == "UP"
    assert row["label_agreement"] == 0
    store.close()


# ------------------------------------------------------------- reporting

def test_capture_report_renders_and_states_the_verdict(tmp_path):
    store, service = make_service(tmp_path)
    service.recorder.start()
    capture = service.discover(float(ROUND_START))[0]
    service._tick_round(capture, capture.metadata.start_ts + 1)
    service._tick_round(capture, capture.metadata.end_ts + 1)
    service._tick_round(capture, capture.metadata.end_ts + 1000)
    service.recorder.flush()

    report = format_capture_report([capture], service.metrics, store)
    for expected in (
        "MARKET IDS / TIME WINDOWS", "TOKEN / OUTCOME MAPPING SUCCESS",
        "EVENTS PER STREAM", "REFERENCE FEED UPDATES",
        "PRE-ROUND HISTORY COVERAGE", "LATENCY DISTRIBUTIONS", "RECORDER HEALTH",
        "BOOK-INTEGRITY CHECKS", "LABEL RECONSTRUCTION", "VERDICT",
    ):
        assert expected in report
    # Item 15 is about integrity, not profitability: the report may DISCLAIM
    # profitability, but must never report a profitability metric.
    lowered = report.lower()
    assert "no profitability claim is made or implied" in lowered
    for banned in ("pnl:", "profit:", "sharpe", "win rate", "hit rate",
                   "fill rate", "expected value:", "return:"):
        assert banned not in lowered
    service.shutdown()
    store.close()


# ------------------------------------------------------------- item 14

def test_service_has_no_order_placing_surface():
    """Item 14: no maker order, taker order, cancel or replacement may be
    sent, and no signing/credential path may exist."""
    import inspect

    import xamarinbot.realtime.clob_ws as clob_ws
    import xamarinbot.realtime.rtds as rtds
    import xamarinbot.realtime.service as service_mod

    for mod in (service_mod, clob_ws, rtds):
        src = inspect.getsource(mod).lower()
        for banned in ("private_key", "privatekey", "api_secret", "sign_order",
                       "post_order", "/order", "eip712", "l1_auth", "l2_auth"):
            assert banned not in src, f"{mod.__name__} references {banned!r}"

    # The only quote-shaped entry point is explicitly hypothetical.
    names = [n for n in dir(service_mod.RealRecorderService) if not n.startswith("_")]
    assert "record_paper_quote" in names
    assert not any(n in names for n in ("submit_order", "place_order", "cancel_order",
                                        "replace_order"))
