"""Phase 1 verification: disconnect/reconnect, stale timestamp rejection,
tick-size-change test, out-of-order event test, cross-feed clock skew test."""
from __future__ import annotations

from xamarinbot.clock import ClockSync, RoundClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.feeds.freshness import FeedFreshnessMonitor, Freshness
from xamarinbot.feeds.mock import (
    MockBookFeed,
    MockFeedCursor,
    MockMarketConfigProvider,
    MockSpotFeed,
    MockTWAPFeed,
    MockUserOrderFeed,
)
from xamarinbot.portfolio.state import Side


def test_round_clock_t_and_tau():
    clock = RoundClock(start_ts=1000.0, end_ts=1300.0)
    assert clock.t(1040.0) == 40.0
    assert clock.tau(1040.0) == 260.0
    assert clock.is_within_round(1000.0) and clock.is_within_round(1300.0)
    assert not clock.is_within_round(999.0)
    assert not clock.is_within_round(1301.0)


def test_clock_sync_offset_and_skew_tracking():
    sync = ClockSync(window=8)
    sync.record(source_ts=10.0, recv_ts=10.2)
    sync.record(source_ts=11.0, recv_ts=11.5)
    sync.record(source_ts=12.0, recv_ts=12.1)
    summary = sync.summary()
    assert summary["sample_count"] == 3
    assert abs(summary["mean_offset_s"] - ((0.2 + 0.5 + 0.1) / 3)) < 1e-9
    assert abs(summary["max_offset_s"] - 0.5) < 1e-9


def test_feed_freshness_monitor_unknown_fresh_stale():
    monitor = FeedFreshnessMonitor(default_max_staleness_s=5.0)
    assert monitor.status("twap", now=100.0) is Freshness.UNKNOWN

    monitor.record("twap", event_time=100.0)
    assert monitor.status("twap", now=102.0) is Freshness.FRESH
    assert monitor.status("twap", now=110.0) is Freshness.STALE  # 10s > 5s threshold


def test_stale_timestamp_is_not_silently_treated_as_fresh():
    monitor = FeedFreshnessMonitor(max_staleness_s={"spot": 2.0})
    monitor.record("spot", event_time=50.0)
    assert monitor.is_fresh("spot", now=51.0)
    assert not monitor.is_fresh("spot", now=53.0)
    assert not monitor.all_fresh(["spot", "twap"], now=53.0)  # twap never recorded either


def _round_id_store_with_config_changes() -> tuple[EventStore, str]:
    store = EventStore(":memory:")
    round_id = "r1"
    store.append(
        EventType.MARKET_CONFIG, round_id, recv_ts=0.0, source_ts=0.0,
        payload=dict(market_id=round_id, up_token_id="U", down_token_id="D", start_ts=0.0, end_ts=300.0,
                     tick_size=0.01, min_order_size=1.0, fee_rate=0.07, taker_delay_ms=0.0, twap_window_seconds=30),
    )
    store.append(
        EventType.MARKET_CONFIG, round_id, recv_ts=120.0, source_ts=120.0,
        payload=dict(market_id=round_id, up_token_id="U", down_token_id="D", start_ts=0.0, end_ts=300.0,
                     tick_size=0.001, min_order_size=1.0, fee_rate=0.07, taker_delay_ms=0.0, twap_window_seconds=30),
    )
    return store, round_id


def test_tick_size_change_is_detected():
    store, round_id = _round_id_store_with_config_changes()
    cursor = MockFeedCursor(store, round_id, now=200.0)
    provider = MockMarketConfigProvider(cursor)

    seen = []
    provider.subscribe_tick_size_changes(round_id, lambda tick: seen.append(tick))
    assert seen == [0.001]

    config = provider.get_market_config(round_id)
    assert config.tick_size == 0.001  # latest visible config wins


def test_disconnect_reconnect_is_a_safe_noop_for_mock_feeds():
    store, round_id = _round_id_store_with_config_changes()
    cursor = MockFeedCursor(store, round_id, now=0.0)
    for feed in (MockTWAPFeed(cursor), MockSpotFeed(cursor), MockBookFeed(cursor), MockUserOrderFeed(cursor)):
        feed.reconnect()  # must not raise


def test_book_snapshot_plus_deltas_reconstruct_correctly_and_resnapshot_matches():
    store = EventStore(":memory:")
    round_id = "r1"
    store.append(EventType.BOOK_SNAPSHOT, round_id, recv_ts=0.0, source_ts=0.0,
                 payload={"side": "UP", "bids": [], "asks": [[0.50, 100.0]], "book_hash": "h0"})
    store.append(EventType.BOOK_DELTA, round_id, recv_ts=1.0, source_ts=1.0,
                 payload={"side": "UP", "book": "asks", "price": 0.50, "size": 0, "book_hash": "h1"})
    store.append(EventType.BOOK_DELTA, round_id, recv_ts=1.0, source_ts=1.0,
                 payload={"side": "UP", "book": "asks", "price": 0.49, "size": 80.0, "book_hash": "h1"})

    cursor = MockFeedCursor(store, round_id, now=0.5)
    feed = MockBookFeed(cursor)
    snap_before = feed.get_snapshot(round_id, Side.UP)
    assert snap_before.best_ask.price == 0.50

    cursor.advance_to(1.0)
    snap_after = feed.get_snapshot(round_id, Side.UP)
    assert snap_after.best_ask.price == 0.49
    assert snap_after.book_hash == "h1"

    resnap = feed.resnapshot(round_id, Side.UP)
    assert resnap.best_ask.price == 0.49


def test_book_feed_cache_rebuilds_correctly_if_cursor_moves_backward():
    store = EventStore(":memory:")
    round_id = "r1"
    store.append(EventType.BOOK_SNAPSHOT, round_id, recv_ts=0.0, source_ts=0.0,
                 payload={"side": "UP", "bids": [], "asks": [[0.5, 10.0]]})
    store.append(EventType.BOOK_DELTA, round_id, recv_ts=5.0, source_ts=5.0,
                 payload={"side": "UP", "book": "asks", "price": 0.5, "size": 0})
    store.append(EventType.BOOK_DELTA, round_id, recv_ts=5.0, source_ts=5.0,
                 payload={"side": "UP", "book": "asks", "price": 0.6, "size": 20.0})

    cursor = MockFeedCursor(store, round_id, now=10.0)
    feed = MockBookFeed(cursor)
    assert feed.get_snapshot(round_id, Side.UP).best_ask.price == 0.6

    cursor.advance_to(0.0)  # rewind past the delta
    assert feed.get_snapshot(round_id, Side.UP).best_ask.price == 0.5


def test_cross_feed_clock_skew_detectable_via_freshness_monitor():
    """Two feeds with the same wall-clock 'now' but very different last
    event times should show different freshness - i.e. skew is observable
    rather than silently averaged away."""
    monitor = FeedFreshnessMonitor(default_max_staleness_s=3.0)
    monitor.record("twap", event_time=100.0)
    monitor.record("spot", event_time=97.0)
    assert monitor.is_fresh("twap", now=101.0)
    assert not monitor.is_fresh("spot", now=101.0)
