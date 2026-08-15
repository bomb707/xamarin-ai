"""Phase 12C item 3: the market WebSocket adapter.

Every frame below is a VERBATIM capture from the live market channel on
2026-08-15 (token ids and hashes intact), so these tests exercise the real
wire shapes rather than a guessed schema. The headline case is
`test_price_change_updates_every_token_in_the_frame`, which is the
regression guard for the routing bug that silently discarded 100% of book
deltas.
"""
from __future__ import annotations

import json

import pytest

from xamarinbot.realtime.clob_ws import BookState, PolymarketMarketStream
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic

UP = "101037353461995830192466008033313618285191559257644968767055811116648440504342"
DOWN = "54402477137234758446057675219311287183330325133648353120744222317943221410164"
MARKET = "0x07785d9e652d7397d8d22279cbd7c4fef3cfc9700b0cbfe0e338e0514d882807"

BOOK_FRAME = json.dumps({
    "market": MARKET, "asset_id": UP, "timestamp": "1786771382832",
    "hash": "a76507f75a7dbb4a7171fb3b2c57c565005606ca",
    "bids": [{"price": "0.44", "size": "100"}, {"price": "0.43", "size": "250"}],
    "asks": [{"price": "0.46", "size": "80"}, {"price": "0.47", "size": "300"}],
    "tick_size": "0.01", "last_trade_price": "0.45", "event_type": "book",
})

# The real shape: NO top-level asset_id; a price_changes array whose
# elements each carry their own asset_id.
PRICE_CHANGE_FRAME = json.dumps({
    "market": MARKET, "timestamp": "1786771382835", "event_type": "price_change",
    "price_changes": [
        {"asset_id": DOWN, "price": "0.55", "size": "2315.29", "side": "BUY",
         "hash": "2692b9c45c9cd673ce61b4a0e7358c8a0401bed2",
         "best_bid": "0.55", "best_ask": "0.56"},
        {"asset_id": UP, "price": "0.44", "size": "0", "side": "BUY",
         "hash": "f0ef587a6f7884e90895fc3caccdceb0f20230fc",
         "best_bid": "0.43", "best_ask": "0.46"},
    ],
})

LAST_TRADE_FRAME = json.dumps({
    "market": MARKET, "asset_id": DOWN, "price": "0.99", "size": "50",
    "fee_rate_bps": "0", "side": "BUY", "timestamp": "1786771382979",
    "event_type": "last_trade_price",
    "transaction_hash": "0xb92d08b9ed45d42e15a9ad4788beae030be313ccf71c3e3a6bfbdd483d8966aa",
})

BEST_BID_ASK_FRAME = json.dumps({
    "market": MARKET, "asset_id": UP, "best_bid": "0.43", "best_ask": "0.46",
    "spread": "0.03", "timestamp": "1786771382861", "event_type": "best_bid_ask",
})

TICK_SIZE_FRAME = json.dumps({
    "market": MARKET, "asset_id": UP, "old_tick_size": "0.01",
    "new_tick_size": "0.001", "timestamp": "1786771382900",
    "event_type": "tick_size_change",
})


def make_stream(**kwargs):
    captured: list = []
    stream = PolymarketMarketStream(
        [UP, DOWN],
        builder=RawEventBuilder(session_id="test"),
        on_raw_event=captured.append,
        side_for_token={UP: "UP", DOWN: "DOWN"},
        round_for_token={UP: "r1", DOWN: "r1"},
        condition_for_token={UP: MARKET, DOWN: MARKET},
        **kwargs,
    )
    return stream, captured


def seed_books(stream):
    """Both tokens need a snapshot before deltas are meaningful."""
    stream.handle_message(BOOK_FRAME)
    stream.handle_message(json.dumps({
        **json.loads(BOOK_FRAME), "asset_id": DOWN,
        "bids": [{"price": "0.54", "size": "90"}],
        "asks": [{"price": "0.56", "size": "70"}],
    }))


# --------------------------------------------------------------- routing

def test_price_change_updates_every_token_in_the_frame():
    """THE regression test for item 3's headline bug.

    The old adapter read a top-level `asset_id` from a price_change frame.
    Live frames have none, so `_side_for_token(None)` returned None and the
    handler returned early - discarding every book delta while the feed
    looked healthy. Both tokens in this one frame must be applied.
    """
    stream, captured = make_stream()
    seed_books(stream)
    captured.clear()

    stream.handle_message(PRICE_CHANGE_FRAME)

    # DOWN: a new/updated bid level at 0.55
    down = stream.book(DOWN)
    assert down.bids[0.55] == pytest.approx(2315.29)
    # UP: size 0 removes the 0.44 bid level entirely
    up = stream.book(UP)
    assert 0.44 not in up.bids
    assert up.bids[0.43] == pytest.approx(250.0)

    # each change became its own raw event, correctly attributed
    assert len(captured) == 2
    assert {e.token_id for e in captured} == {UP, DOWN}
    assert all(e.event_type == "price_change" for e in captured)
    assert {e.normalized_side for e in captured} == {"UP", "DOWN"}


def test_price_change_frame_has_no_top_level_asset_id():
    """Guard the premise of the test above against a schema change."""
    assert "asset_id" not in json.loads(PRICE_CHANGE_FRAME)
    assert all("asset_id" in c for c in json.loads(PRICE_CHANGE_FRAME)["price_changes"])


def test_book_snapshot_builds_a_sorted_two_sided_book():
    stream, _ = make_stream()
    stream.handle_message(BOOK_FRAME)
    book = stream.book(UP)
    assert book.best_bid == (0.44, 100.0)
    assert book.best_ask == (0.46, 80.0)
    assert [p for p, _ in book.sorted_bids()] == [0.44, 0.43]   # descending
    assert [p for p, _ in book.sorted_asks()] == [0.46, 0.47]   # ascending
    assert book.tick_size == 0.01
    assert book.is_quotable


def test_deltas_before_any_snapshot_are_recorded_as_a_gap_not_applied():
    """A delta with no snapshot beneath it cannot be applied to anything
    meaningful; the book must report a gap rather than inventing a ladder."""
    stream, _ = make_stream()
    stream.handle_message(PRICE_CHANGE_FRAME)
    up = stream.book(UP)
    assert up.awaiting_snapshot is True
    assert up.has_gap is True
    assert up.is_quotable is False
    assert not up.bids and not up.asks


def test_tick_size_change_is_handled_on_this_connection():
    stream, captured = make_stream()
    stream.handle_message(BOOK_FRAME)
    captured.clear()
    stream.handle_message(TICK_SIZE_FRAME)
    assert stream.book(UP).tick_size == 0.001
    assert captured[-1].event_type == "tick_size_change"


def test_last_trade_price_and_best_bid_ask_are_recorded():
    stream, captured = make_stream()
    seed_books(stream)
    captured.clear()
    stream.handle_message(LAST_TRADE_FRAME)
    stream.handle_message(BEST_BID_ASK_FRAME)
    kinds = [e.event_type for e in captured]
    assert "last_trade_price" in kinds and "best_bid_ask" in kinds
    assert stream.book(DOWN).last_trade_price == pytest.approx(0.99)


def test_unknown_event_types_are_still_recorded_verbatim():
    """The raw log must be the wire, not a filtered view of it."""
    stream, captured = make_stream()
    frame = json.dumps({"event_type": "something_new", "market": MARKET, "timestamp": "1786771382999"})
    stream.handle_message(frame)
    assert captured[-1].event_type == "something_new"
    assert captured[-1].payload["market"] == MARKET


def test_parse_failures_are_counted_not_swallowed():
    failures = []
    stream, _ = make_stream(on_parse_failure=lambda raw, exc: failures.append(exc))
    stream.handle_message("{not json")
    assert len(failures) == 1


def test_timestamps_are_millisecond_strings_converted_to_nanoseconds():
    stream, captured = make_stream()
    stream.handle_message(BOOK_FRAME)
    ev = captured[-1]
    assert ev.source_timestamp_ns == 1786771382832 * 1_000_000
    assert ev.recv_wall_timestamp_ns > 0
    assert ev.recv_monotonic_ns > 0
    # the verbatim wire payload survives normalization
    assert json.loads(ev.payload_json)["hash"] == "a76507f75a7dbb4a7171fb3b2c57c565005606ca"


# ---------------------------------------------------------- subscription

def test_subscription_enables_custom_market_features():
    """Live A/B showed `custom_feature_enabled` is what gates best_bid_ask /
    new_market / market_resolved."""
    stream, _ = make_stream()
    msg = json.loads(stream._subscribe_message())
    assert msg["type"] == "market"
    assert msg["custom_feature_enabled"] is True
    assert set(msg["assets_ids"]) == {UP, DOWN}


def test_custom_features_can_be_disabled():
    stream, _ = make_stream(custom_features=False)
    assert "custom_feature_enabled" not in json.loads(stream._subscribe_message())


# ------------------------------------------------------------- integrity

def test_integrity_mismatch_resnapshots_and_records_a_reconciliation_event():
    """Item 15: a mismatch must mark, resnapshot and record - never
    silently continue."""
    stream, captured = make_stream()
    stream.handle_message(BOOK_FRAME)
    resnapshots = []
    stream._on_resnapshot = resnapshots.append

    # REST disagrees with the in-memory top of book.
    stream._rest_book = lambda token_id: {
        "timestamp": "1786771390000", "hash": "restdiff",
        "bids": [{"price": "0.40", "size": "10"}],
        "asks": [{"price": "0.50", "size": "10"}],
    }
    captured.clear()
    result = stream.check_integrity(UP)

    assert result.matched is False
    assert result.ws_best_bid == 0.44 and result.rest_best_bid == 0.40
    kinds = [e.event_type for e in captured]
    assert "book_integrity_check" in kinds
    assert "book_snapshot_rest_reconcile" in kinds
    assert resnapshots == [UP]
    # the book was actually repaired from the REST response
    assert stream.book(UP).best_bid == (0.40, 10.0)


def test_integrity_match_leaves_the_book_alone():
    stream, captured = make_stream()
    stream.handle_message(BOOK_FRAME)
    stream._rest_book = lambda token_id: {
        "timestamp": "1786771390000", "hash": "same",
        "bids": [{"price": "0.44", "size": "100"}, {"price": "0.43", "size": "250"}],
        "asks": [{"price": "0.46", "size": "80"}],
    }
    captured.clear()
    result = stream.check_integrity(UP)
    assert result.matched is True
    assert "book_snapshot_rest_reconcile" not in [e.event_type for e in captured]


# ------------------------------------------------- end-of-round teardown

def test_clean_venue_close_is_not_a_parse_failure():
    """LIVE FINDING: once every subscribed token settles, Polymarket closes
    the market channel with `ConnectionClosedOK: received 1000 (OK) all
    subscribed assets resolved`. That is the venue ending the subscription,
    not a data-integrity fault - counting it as one disqualified an
    otherwise flawless capture (0 dropped, 30/30 integrity) for training."""
    from xamarinbot.realtime.clob_ws import _all_assets_resolved, _is_normal_close

    class ConnectionClosedOK(Exception):
        pass

    exc = ConnectionClosedOK(
        "received 1000 (OK) all subscribed assets resolved; "
        "then sent 1000 (OK) all subscribed assets resolved"
    )
    assert _is_normal_close(exc) is True
    assert _all_assets_resolved(exc) is True
    # A genuine failure must still be treated as one.
    assert _is_normal_close(OSError("connection reset by peer")) is False
    assert _all_assets_resolved(OSError("connection reset by peer")) is False


def test_rest_book_404_on_a_settled_token_is_recorded_not_counted_as_failure():
    """`/book?token_id=…` 404s once the market has settled; the book no
    longer exists. Expected at end of round."""
    from xamarinbot.realtime.clob_ws import _is_settled_token_404

    exc = RuntimeError(
        "Client error '404 Not Found' for url "
        "'https://clob.polymarket.com/book?token_id=8423308476593277449'"
    )
    assert _is_settled_token_404(exc) is True
    assert _is_settled_token_404(RuntimeError("500 Server Error for /book")) is False

    failures = []
    stream, captured = make_stream(on_parse_failure=lambda raw, e: failures.append(e))

    def boom(token_id):
        raise exc

    stream._rest_book = boom
    stream.bootstrap_all()
    assert failures == [], "a settled-token 404 must not count as a parse failure"
    assert any(e.event_type == "book_unavailable_settled" for e in captured)


# ----------------------------------------------------------- book state

def test_zero_size_removes_a_level_and_nonzero_replaces_it():
    book = BookState("t")
    book.apply_snapshot([{"price": "0.4", "size": "5"}], [{"price": "0.6", "size": "5"}],
                        book_hash="h", source_ts_ns=1, recv_ts_ns=2)
    book.apply_price_change(0.4, 9.0, "BUY", book_hash="h2", source_ts_ns=3, recv_ts_ns=4)
    assert book.bids[0.4] == 9.0
    book.apply_price_change(0.4, 0.0, "BUY", book_hash="h3", source_ts_ns=5, recv_ts_ns=6)
    assert 0.4 not in book.bids


def test_buy_side_is_bids_and_sell_side_is_asks():
    book = BookState("t")
    book.apply_snapshot([], [], book_hash=None, source_ts_ns=1, recv_ts_ns=1)
    book.apply_price_change(0.30, 7.0, "BUY", book_hash=None, source_ts_ns=2, recv_ts_ns=2)
    book.apply_price_change(0.70, 3.0, "SELL", book_hash=None, source_ts_ns=3, recv_ts_ns=3)
    assert book.bids == {0.30: 7.0}
    assert book.asks == {0.70: 3.0}
