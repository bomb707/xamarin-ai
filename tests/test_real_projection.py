"""Phase 12C.1 item 8: REAL raw -> normalized event projection.

Fixtures below use the real wire shapes captured in Phase 12C, so the
payload-key contract asserted here is the one `features/engine.py` and
`replay/feeds.py` actually require.
"""
from __future__ import annotations

import json

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.market.constraints import MarketConstraints
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.replay.feeds import ReplayBookFeed, ReplayCursor, market_config_from_payload
from xamarinbot.replay.projection import (
    PROVENANCE_KEY,
    ProjectionError,
    project_capture,
    project_round,
    settlement_topic_for,
)

ROUND = "btc-updown-5m-1786777800"
START_NS = 1_786_777_800_000_000_000
END_NS = 1_786_778_100_000_000_000
UP = "up-token"
DOWN = "down-token"


#: When the recorder actually learned this market's parameters. In the real
#: verified capture this is 473.5s BEFORE the round opens, and it is what
#: MARKET_CONFIG's visibility timestamp must be (Phase 12C.2 item 3).
METADATA_RECV_NS = START_NS - 470_000_000_000
#: When the venue's outcome was actually observed - 92s AFTER the round
#: closed in the real capture.
RESOLUTION_RECV_NS = END_NS + 92_000_000_000


def make_capture(tmp_path, *, twap_window=60, settlement="chainlink_twap",
                 with_spot=True, twap_covers_open=True, n=8, data_topic=None,
                 with_metadata=True, tick_change_at=None, with_resolution=True):
    raw = RawEventStore(str(tmp_path / "raw.db"))
    raw.upsert_round({
        "round_id": ROUND, "session_id": "s", "condition_id": "0xcond",
        "slug": ROUND, "question": "Bitcoin Up or Down",
        "resolution_source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
        "start_ts_ns": START_NS, "end_ts_ns": END_NS,
        "up_token_id": UP, "down_token_id": DOWN,
        "tick_size": 0.01, "min_order_size": 5.0,
        "fee_config_json": json.dumps({"feeSchedule": {"rate": 0.07}}),
        "taker_delay_ms": 0.0, "twap_window_s": twap_window,
        "settlement_kind": settlement, "state": "FINALIZED",
    })

    b = RawEventBuilder(session_id="s")
    events = []

    # The REST metadata observation: no external source timestamp, a real
    # receive timestamp. This is what MARKET_CONFIG's visibility must come
    # from.
    if with_metadata:
        events.append(b.build(
            Topic.MARKET_METADATA, "market_metadata_discovered",
            {"gamma": {"conditionId": "0xcond"}, "clob": {}, "warnings": []},
            round_id=ROUND, condition_id="0xcond",
            source_timestamp_ns=None,
        ))
        events[-1] = _at_recv(events[-1], METADATA_RECV_NS)
    if with_resolution:
        events.append(b.build(
            Topic.MARKET_METADATA, "market_metadata_resolution",
            {"closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["1", "0"]'},
            round_id=ROUND, condition_id="0xcond", source_timestamp_ns=None,
        ))
        events[-1] = _at_recv(events[-1], RESOLUTION_RECV_NS)
    # `data_topic` lets a test put reference data on a DIFFERENT stream from
    # the one the market declares, which is how "the capture has data, but
    # not the declared basis" is exercised.
    twap_topic = data_topic or ({30: Topic.RTDS_TWAP_30, 60: Topic.RTDS_TWAP_60}.get(
        twap_window, Topic.RTDS_CHAINLINK
    ) if settlement == "chainlink_twap" else Topic.RTDS_CHAINLINK)

    # reference series, one per second, starting before the open
    first = -2 if twap_covers_open else 2
    for i in range(first, first + n):
        src_ns = START_NS + i * 1_000_000_000
        events.append(b.build(
            twap_topic, "update",
            {"payload": {"symbol": "btc/usd", "timestamp": src_ns // 1_000_000,
                         "value": 63000.0 + i, "full_accuracy_value": str(int((63000.0 + i) * 10**18)),
                         "window_s": twap_window},
             "timestamp": src_ns // 1_000_000 + 1500, "topic": twap_topic.value, "type": "update"},
            round_id=ROUND, source_timestamp_ns=src_ns,
            publisher_timestamp_ns=src_ns + 1_500_000_000,
        ))
        if with_spot:
            events.append(b.build(
                Topic.RTDS_BINANCE, "update",
                {"payload": {"symbol": "btcusdt", "timestamp": src_ns // 1_000_000,
                             "value": 63070.0 + i},
                 "timestamp": src_ns // 1_000_000, "topic": "crypto_prices", "type": "update"},
                round_id=ROUND, source_timestamp_ns=src_ns,
            ))

    # a two-sided book per token: one snapshot, then a delta every second so
    # the book stays fresh across the whole span the reference series covers
    for token, side in ((UP, "UP"), (DOWN, "DOWN")):
        src_ns = START_NS + first * 1_000_000_000
        events.append(b.build(
            Topic.CLOB_MARKET, "book",
            {"asset_id": token, "market": "0xcond", "timestamp": str(src_ns // 1_000_000),
             "hash": "h0", "tick_size": "0.01",
             "bids": [{"price": "0.44", "size": "100"}, {"price": "0.43", "size": "250"}],
             "asks": [{"price": "0.46", "size": "80"}]},
            round_id=ROUND, token_id=token, normalized_side=side,
            source_timestamp_ns=src_ns,
        ))
        for i in range(first + 1, first + n):
            d_ns = START_NS + i * 1_000_000_000
            events.append(b.build(
                Topic.CLOB_MARKET, "price_change",
                {"asset_id": token, "market": "0xcond",
                 "timestamp": str(d_ns // 1_000_000),
                 "price": "0.45", "size": str(60 + i), "side": "BUY", "hash": f"h{i}"},
                round_id=ROUND, token_id=token, normalized_side=side,
                source_timestamp_ns=d_ns,
            ))
        # a non-book event that has no normalized counterpart
        events.append(b.build(
            Topic.CLOB_MARKET, "last_trade_price",
            {"asset_id": token, "price": "0.45", "size": "5",
             "timestamp": str(src_ns // 1_000_000)},
            round_id=ROUND, token_id=token, normalized_side=side,
            source_timestamp_ns=src_ns,
        ))

    # A real tick_size_change, announced once per token exactly as the venue
    # does it.
    if tick_change_at is not None:
        at_ns = START_NS + int(tick_change_at * 1e9)
        for token, side in ((UP, "UP"), (DOWN, "DOWN")):
            events.append(b.build(
                Topic.CLOB_MARKET, "tick_size_change",
                {"asset_id": token, "market": "0xcond",
                 "timestamp": str(at_ns // 1_000_000),
                 "old_tick_size": "0.01", "new_tick_size": "0.001"},
                round_id=ROUND, token_id=token, normalized_side=side,
                source_timestamp_ns=at_ns,
            ))

    raw.write_batch(_realistic_recv(events))
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP", "reconstructed_outcome": "UP",
        "reconstruction_basis": "declared:crypto_prices_twap_sixty",
        "label_agreement": 1, "end_reference_value": 63005.0,
    })
    return raw


#: Measured source->recv p50 in the real captures was ~13.5ms. The builder
#: stamps the wall clock at construction, which for a fixture would put every
#: receive time at "now" - decades of apparent latency, and enough to make a
#: recv_ts-gated consumer (ShadowRunner) see nothing at all. Fixtures
#: therefore stamp a realistic wire latency.
WIRE_LATENCY_NS = 15_000_000


def _at_recv(event, recv_ns: int):
    """Override an event's receive timestamp (the builder stamps 'now')."""
    import dataclasses

    return dataclasses.replace(event, recv_wall_timestamp_ns=recv_ns)


def _realistic_recv(events: list) -> list:
    """Give every event a receive timestamp just after its source timestamp,
    leaving events that already have an explicit one alone."""
    import dataclasses

    out = []
    for e in events:
        if e.source_timestamp_ns is not None:
            e = dataclasses.replace(
                e, recv_wall_timestamp_ns=e.source_timestamp_ns + WIRE_LATENCY_NS
            )
        out.append(e)
    return out


def new_out(tmp_path, provenance=DataProvenance.REAL_REPLAY) -> EventStore:
    return EventStore(str(tmp_path / "out.db"), provenance=provenance)


# --------------------------------------------------------------- basics

def test_projection_writes_the_normalized_event_vocabulary(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.provenance is DataProvenance.REAL_REPLAY
    assert res.counts["MARKET_CONFIG"] == 1
    assert res.counts["TWAP"] == 8
    assert res.counts["SPOT"] == 8
    assert res.counts["BOOK_SNAPSHOT"] == 2      # one per token
    assert res.counts["BOOK_DELTA"] == 2 * 7
    assert "SETTLEMENT" not in res.counts, "the label must stay off the causal stream"


def test_a_synthetic_destination_store_is_refused(tmp_path):
    """Real observations must not be written into a store that would then
    claim to be synthetic - or worse, be trusted as synthetic-safe."""
    raw = make_capture(tmp_path)
    out = new_out(tmp_path, DataProvenance.SYNTHETIC_TEST)
    with pytest.raises(ProjectionError, match="REAL_REPLAY"):
        project_round(raw, ROUND, out)


# ------------------------------------------------ item 15: declared basis

def test_settlement_basis_comes_from_the_market_not_a_constant():
    assert settlement_topic_for("chainlink_twap", 60) is Topic.RTDS_TWAP_60
    assert settlement_topic_for("chainlink_twap", 30) is Topic.RTDS_TWAP_30
    assert settlement_topic_for("chainlink_reference", None) is Topic.RTDS_CHAINLINK
    with pytest.raises(ProjectionError, match="not a stream RTDS publishes"):
        settlement_topic_for("chainlink_twap", 45)


def test_a_30s_market_projects_from_the_30s_stream(tmp_path):
    raw, out = make_capture(tmp_path, twap_window=30), new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.settlement_topic == Topic.RTDS_TWAP_30.value
    twap = [e for e in out.all_events(ROUND) if e.event_type is EventType.TWAP]
    assert all(e.payload["window_seconds"] == 30 for e in twap)


def test_only_the_declared_basis_becomes_the_twap_series(tmp_path):
    """Projecting plain-Chainlink AND TWAP into one `EventType.TWAP` series
    would silently interleave two different quantities and corrupt
    `gap_twap_bp`/`z_gap`."""
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    windows = {e.payload["window_seconds"]
               for e in out.all_events(ROUND) if e.event_type is EventType.TWAP}
    assert windows == {60}


# ------------------------------------------ never synthesize what is missing

def test_a_missing_settlement_stream_refuses_rather_than_substituting(tmp_path):
    """The capture holds TWAP-60 data, but this market declares plain
    Chainlink reference as its basis. The projection must refuse rather than
    quietly substituting the series it happens to have."""
    raw = make_capture(tmp_path, settlement="chainlink_reference", twap_window=60,
                       data_topic=Topic.RTDS_TWAP_60)
    out = new_out(tmp_path)
    with pytest.raises(ProjectionError, match="no rtds_chainlink observations"):
        project_round(raw, ROUND, out)


def test_p0_missing_at_the_open_refuses_rather_than_defaulting(tmp_path):
    """`p0` is a real observation at or before the round open. If the
    capture has none, the round cannot be projected - it must not be
    invented."""
    raw = make_capture(tmp_path, twap_covers_open=False)
    out = new_out(tmp_path)
    with pytest.raises(ProjectionError, match="p0 cannot be established"):
        project_round(raw, ROUND, out)


def test_p0_is_the_real_observation_at_or_before_the_open(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    # series is 63000.0 + i for i in -2..5; the last at/before the open is i=0
    assert res.p0 == pytest.approx(63000.0)


def test_a_missing_spot_stream_is_reported_not_filled_in(tmp_path):
    raw, out = make_capture(tmp_path, with_spot=False), new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert "SPOT" not in res.counts
    assert any("MISSING_SPOT" in w for w in res.warnings)


def test_events_with_no_normalized_counterpart_are_counted_not_forced(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.skipped["no_normalized_event_for:last_trade_price"] == 2


def test_missing_market_metadata_refuses(tmp_path):
    raw = RawEventStore(str(tmp_path / "raw.db"))
    out = new_out(tmp_path)
    with pytest.raises(ProjectionError, match="no round metadata"):
        project_round(raw, "nope", out)


# ------------------------------------------------ preserved provenance

def test_every_projected_event_carries_its_raw_provenance(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    for e in out.all_events(ROUND):
        block = e.payload[PROVENANCE_KEY]
        assert block["provenance"] == "REAL_REPLAY"


def test_all_four_timestamps_survive_the_projection(tmp_path):
    """`source_ts`/`recv_ts` become normalized columns; publisher time and
    the monotonic clock have no column, so they ride in the provenance
    block rather than being dropped."""
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    twap = [e for e in out.all_events(ROUND) if e.event_type is EventType.TWAP][0]
    block = twap.payload[PROVENANCE_KEY]
    assert twap.source_ts is not None
    assert twap.recv_ts is not None
    assert twap.recv_ts >= twap.source_ts
    assert block["publisher_timestamp_ns"] is not None
    assert block["recv_monotonic_ns"] is not None
    assert block["recv_wall_timestamp_ns"] is not None
    # identity back to the raw row, plus a hash of the original wire bytes
    assert block["raw_event_id"][0] == "s"
    assert len(block["payload_sha256"]) == 64
    assert block["token_id"] is None or isinstance(block["token_id"], str)


def test_book_events_carry_side_token_and_hash(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    snaps = [e for e in out.all_events(ROUND) if e.event_type is EventType.BOOK_SNAPSHOT]
    assert {e.payload["side"] for e in snaps} == {"UP", "DOWN"}
    assert all(e.payload[PROVENANCE_KEY]["token_id"] in (UP, DOWN) for e in snaps)


# --------------------------------------------- the downstream contract

def test_projected_payloads_satisfy_the_feature_engine_contract(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    events = out.all_events(ROUND)

    config = next(e for e in events if e.event_type is EventType.MARKET_CONFIG)
    for key in ("start_ts", "end_ts", "twap_window_seconds", "tick_size",
                "min_order_size", "fee_rate", "taker_delay_ms",
                "up_token_id", "down_token_id", "market_id"):
        assert key in config.payload

    twap = next(e for e in events if e.event_type is EventType.TWAP)
    assert "value" in twap.payload and "window_seconds" in twap.payload

    spot = next(e for e in events if e.event_type is EventType.SPOT)
    assert "value" in spot.payload and "provider" in spot.payload

    snap = next(e for e in events if e.event_type is EventType.BOOK_SNAPSHOT)
    assert snap.payload["side"] in ("UP", "DOWN")
    assert all(len(lvl) == 2 for lvl in snap.payload["bids"] + snap.payload["asks"])

    delta = next(e for e in events if e.event_type is EventType.BOOK_DELTA)
    assert delta.payload["book"] in ("bids", "asks")
    assert isinstance(delta.payload["price"], float)
    assert isinstance(delta.payload["size"], float)


def test_market_config_rebuilds_despite_the_extra_provenance_key(tmp_path):
    """`MarketConfig(**payload)` would raise on the `_provenance` block, so
    reconstruction selects the declared fields."""
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    payload = next(e.payload for e in out.all_events(ROUND)
                   if e.event_type is EventType.MARKET_CONFIG)
    cfg = market_config_from_payload(payload)
    assert cfg.min_order_size == 5.0
    assert cfg.tick_size == 0.01

    constraints = MarketConstraints.from_market_config(
        cfg, provenance=DataProvenance.REAL_REPLAY, source="projected"
    )
    assert constraints.min_order_shares == 5.0
    assert constraints.tick_size == 0.01


def test_market_config_reconstruction_names_a_missing_parameter(tmp_path):
    with pytest.raises(KeyError, match="min_order_size"):
        market_config_from_payload({"market_id": "x", "tick_size": 0.01})


def test_replay_feeds_reconstruct_the_book_from_projected_events(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    from xamarinbot.portfolio.state import Side

    cursor = ReplayCursor(out, ROUND)
    cursor.advance_to(END_NS / 1e9)
    book = ReplayBookFeed(cursor).get_snapshot(ROUND, Side.UP)
    assert book is not None
    assert book.best_bid is not None and book.best_ask is not None
    # the BUY delta at 0.45 became the new best bid
    assert book.best_bid.price == pytest.approx(0.45)


def test_projection_produces_causal_feature_vectors(tmp_path):
    raw, out = make_capture(tmp_path, n=90), new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    events = out.all_events(ROUND)
    fv = compute(events, ROUND, START_NS / 1e9 + 70.0, res.p0, FeatureConfig())
    assert isinstance(fv, FeatureVector), getattr(fv, "reason", None)
    assert fv.p0 == res.p0
    assert fv.twap > 0 and fv.spot > 0
    assert 0.0 < fv.clob_mid < 1.0


def test_project_capture_reports_a_skipped_round_rather_than_dropping_it(tmp_path):
    raw = make_capture(tmp_path, twap_covers_open=False)
    out = new_out(tmp_path)
    logged: list[str] = []
    results = project_capture(raw, out, log=logged.append)
    assert results == []
    assert any("SKIPPED" in m for m in logged)
