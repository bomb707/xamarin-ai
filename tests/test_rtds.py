"""Phase 12C item 4: shared RTDS connection for BTC reference signals.

Frames are verbatim live captures from 2026-08-15.
"""
from __future__ import annotations

import json

import pytest

from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.realtime.rtds import (
    TOPIC_BINANCE,
    TOPIC_CHAINLINK,
    TOPIC_TWAP_30,
    TOPIC_TWAP_60,
    RTDSClient,
    decode_full_accuracy,
)

CHAINLINK_BTC = json.dumps({
    "connection_id": "gXTNwy43YWeIKEhCjA==",
    "payload": {"full_accuracy_value": "63086218888020438000000", "symbol": "btc/usd",
                "timestamp": 1786771253000, "value": 63086.21888802044},
    "timestamp": 1786771254796, "topic": "crypto_prices_chainlink", "type": "update",
})
TWAP60_BTC = json.dumps({
    "connection_id": "gXTNwy43YWeIKEhCjA==",
    "payload": {"full_accuracy_value": "63086406212239293939712", "symbol": "btc/usd",
                "timestamp": 1786771253000, "value": 63086.40621223929, "window_s": 60},
    "timestamp": 1786771254918, "topic": "crypto_prices_twap_sixty", "type": "update",
})
TWAP30_BTC = json.dumps({
    "payload": {"full_accuracy_value": "63086396983844760715264", "symbol": "btc/usd",
                "timestamp": 1786771253000, "value": 63086.39698384476, "window_s": 30},
    "timestamp": 1786771254921, "topic": "crypto_prices_twap_thirty", "type": "update",
})
BINANCE_BTC = json.dumps({
    "payload": {"full_accuracy_value": "63151.96000000", "symbol": "btcusdt",
                "timestamp": 1786771255000, "value": 63151.96},
    "timestamp": 1786771255161, "topic": "crypto_prices", "type": "update",
})
# Another asset on the same unfiltered socket - must be dropped client-side.
ZEC_TWAP30 = json.dumps({
    "payload": {"full_accuracy_value": "492659710962142674944", "symbol": "zec/usd",
                "timestamp": 1786771129000, "value": 492.6597109621427, "window_s": 30},
    "timestamp": 1786771130588, "topic": "crypto_prices_twap_thirty", "type": "update",
})
# The control-plane rejection the dotted topic aliases actually produce.
INVALID_BODY = json.dumps({"message": "Invalid request body", "connectionId": "x", "requestId": "y"})


def make_client(**kwargs):
    captured = []
    client = RTDSClient(
        builder=RawEventBuilder(session_id="test"),
        on_raw_event=captured.append,
        **kwargs,
    )
    return client, captured


def test_subscribes_unfiltered_to_the_raw_topics():
    """Live-verified: `filters` suppresses all live delivery, and the dotted
    aliases are rejected outright. So the subscription must be unfiltered
    and must use the raw topic names."""
    client, _ = make_client()
    msg = json.loads(client.subscribe_message())
    assert msg["action"] == "subscribe"
    topics = [s["topic"] for s in msg["subscriptions"]]
    assert topics == [TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30, TOPIC_TWAP_60]
    # no filters anywhere - that is the whole point
    assert all("filters" not in s for s in msg["subscriptions"])
    # and no dotted alias, which the live socket rejects
    assert not any("." in t for t in topics)


def test_all_four_btc_reference_signals_are_captured():
    client, captured = make_client()
    for frame in (CHAINLINK_BTC, BINANCE_BTC, TWAP30_BTC, TWAP60_BTC):
        client.handle_message(frame)
    assert client.latest(TOPIC_CHAINLINK).value == pytest.approx(63086.21888802044)
    assert client.latest(TOPIC_BINANCE).value == pytest.approx(63151.96)
    assert client.latest(TOPIC_TWAP_30).window_s == 30
    assert client.latest(TOPIC_TWAP_60).window_s == 60
    assert {e.topic for e in captured} == {
        Topic.RTDS_CHAINLINK, Topic.RTDS_BINANCE, Topic.RTDS_TWAP_30, Topic.RTDS_TWAP_60,
    }


def test_other_assets_are_filtered_client_side():
    client, captured = make_client()
    client.handle_message(ZEC_TWAP30)
    assert client.latest(TOPIC_TWAP_30) is None
    assert captured == []


def test_chainlink_observation_time_is_kept_separate_from_publisher_and_receive():
    """Item 4: "Do not replace Chainlink observation time with local receive
    time." All four clocks must survive independently."""
    client, captured = make_client()
    client.handle_message(TWAP60_BTC)
    ev = captured[-1]
    assert ev.source_timestamp_ns == 1786771253000 * 1_000_000      # Chainlink observation
    assert ev.publisher_timestamp_ns == 1786771254918 * 1_000_000   # RTDS publisher
    assert ev.recv_wall_timestamp_ns > 0                            # local wall
    assert ev.recv_monotonic_ns > 0                                 # local monotonic
    assert ev.source_timestamp_ns != ev.publisher_timestamp_ns != ev.recv_wall_timestamp_ns

    obs = client.latest(TOPIC_TWAP_60)
    assert obs.source_ts_ns == 1786771253000 * 1_000_000
    assert obs.publisher_ts_ns == 1786771254918 * 1_000_000
    # the oracle hop (~1.9s here) is genuinely separable from our own
    assert (obs.publisher_ts_ns - obs.source_ts_ns) / 1e9 == pytest.approx(1.918, abs=1e-3)


def test_latency_split_does_not_attribute_oracle_delay_to_our_network():
    client, captured = make_client()
    client.handle_message(TWAP60_BTC)
    ev = captured[-1]
    assert ev.source_to_recv_latency_ns > ev.publisher_to_recv_latency_ns


def test_e18_decoding_is_applied_only_to_chainlink_topics():
    assert decode_full_accuracy(TOPIC_CHAINLINK, "63086218888020438000000") == pytest.approx(63086.218888, abs=1e-4)
    assert decode_full_accuracy(TOPIC_TWAP_60, "63086406212239293939712") == pytest.approx(63086.406212, abs=1e-4)
    # Binance sends a plain decimal, not E18
    assert decode_full_accuracy(TOPIC_BINANCE, "63151.96000000") == pytest.approx(63151.96)
    assert decode_full_accuracy(TOPIC_CHAINLINK, None) is None


def test_binance_and_chainlink_are_genuinely_different_sources():
    """They disagreed by ~65 USD on the captured frames; treating one as a
    proxy for the other would be a ~10bp error on the settlement input."""
    client, _ = make_client()
    client.handle_message(CHAINLINK_BTC)
    client.handle_message(BINANCE_BTC)
    diff = abs(client.latest(TOPIC_BINANCE).value - client.latest(TOPIC_CHAINLINK).value)
    assert diff > 10.0


def test_out_of_order_observation_does_not_overwrite_the_newer_latest():
    client, _ = make_client()
    client.handle_message(CHAINLINK_BTC)
    stale = json.loads(CHAINLINK_BTC)
    stale["payload"]["timestamp"] = 1786771200000  # older
    stale["payload"]["value"] = 1.0
    client.handle_message(json.dumps(stale))
    assert client.latest(TOPIC_CHAINLINK).value == pytest.approx(63086.21888802044)
    # ...but it is still recorded in history, not discarded
    assert len(client.history(TOPIC_CHAINLINK)) == 2


def test_boundary_lookup_uses_source_timestamps():
    client, _ = make_client()
    for ts, val in ((1786771250000, 100.0), (1786771253000, 200.0), (1786771256000, 300.0)):
        frame = json.loads(CHAINLINK_BTC)
        frame["payload"]["timestamp"] = ts
        frame["payload"]["value"] = val
        frame["payload"]["full_accuracy_value"] = str(int(val * 10**18))
        client.handle_message(json.dumps(frame))
    boundary = 1786771254000 * 1_000_000
    assert client.observation_at_or_before(TOPIC_CHAINLINK, boundary).value == 200.0
    assert client.observation_at_or_after(TOPIC_CHAINLINK, boundary).value == 300.0


def test_control_plane_rejection_is_recorded_and_counted():
    """An "Invalid request body" reply is why a capture could be silently
    empty; it must be visible after the fact."""
    failures = []
    client, captured = make_client(on_parse_failure=lambda raw, exc: failures.append(exc))
    client.handle_message(INVALID_BODY)
    assert len(failures) == 1
    assert captured[-1].topic is Topic.RECORDER_CONTROL
    assert captured[-1].event_type == "rtds_error"


def test_update_without_a_source_timestamp_is_rejected_not_defaulted():
    failures = []
    client, captured = make_client(on_parse_failure=lambda raw, exc: failures.append(exc))
    frame = json.loads(CHAINLINK_BTC)
    del frame["payload"]["timestamp"]
    client.handle_message(json.dumps(frame))
    assert len(failures) == 1
    assert client.latest(TOPIC_CHAINLINK) is None
    assert captured == []


def test_raw_wire_payload_is_preserved_verbatim():
    client, captured = make_client()
    client.handle_message(TWAP60_BTC)
    assert json.loads(captured[-1].payload_json) == json.loads(TWAP60_BTC)
