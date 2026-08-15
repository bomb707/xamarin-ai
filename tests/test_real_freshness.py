"""Phase 12C item 10: real freshness replacing the hardcoded is_fresh=True."""
from __future__ import annotations

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.realtime.freshness import (
    FeedKind,
    FeedStatus,
    FreshnessPolicy,
    evaluate_feed,
    evaluate_freshness,
)
from xamarinbot.shadow.runner import REPLAY_FRESHNESS_POLICY, freshness_from_events

NOW = 1000.0


def test_fresh_when_every_required_input_is_recent():
    r = evaluate_freshness(NOW, {
        FeedKind.BOOK: NOW - 0.5,
        FeedKind.CHAINLINK_REFERENCE: NOW - 1.0,
        FeedKind.CHAINLINK_TWAP_60: NOW - 1.0,
        FeedKind.BINANCE: NOW - 1.0,
    })
    assert r.is_fresh is True
    assert r.reason is None


def test_one_stale_required_input_makes_the_whole_state_not_fresh():
    r = evaluate_freshness(NOW, {
        FeedKind.BOOK: NOW - 30.0,
        FeedKind.CHAINLINK_REFERENCE: NOW - 1.0,
        FeedKind.CHAINLINK_TWAP_60: NOW - 1.0,
        FeedKind.BINANCE: NOW - 1.0,
    })
    assert r.is_fresh is False
    assert "book:STALE" in r.reason
    assert r.feeds[FeedKind.BOOK].status is FeedStatus.STALE


def test_missing_is_distinct_from_stale():
    """"never observed" and "last seen 6 seconds ago" must not look alike."""
    r = evaluate_freshness(NOW, {FeedKind.BOOK: NOW - 0.1})
    assert r.feeds[FeedKind.CHAINLINK_REFERENCE].status is FeedStatus.MISSING
    assert r.feeds[FeedKind.CHAINLINK_REFERENCE].age_s is None
    assert r.is_fresh is False
    assert "MISSING" in r.reason


def test_a_gapped_book_is_unusable_even_if_recent():
    r = evaluate_freshness(
        NOW,
        {FeedKind.BOOK: NOW - 0.1, FeedKind.CHAINLINK_REFERENCE: NOW - 1.0,
         FeedKind.CHAINLINK_TWAP_60: NOW - 1.0, FeedKind.BINANCE: NOW - 1.0},
        unusable=frozenset({FeedKind.BOOK}),
    )
    assert r.feeds[FeedKind.BOOK].status is FeedStatus.UNUSABLE
    assert r.is_fresh is False


def test_a_source_timestamp_far_ahead_of_local_time_is_not_trusted():
    """Clock skew must not read as impossibly-fresh data."""
    f = evaluate_feed(FeedKind.BOOK, NOW + 10.0, NOW, FreshnessPolicy())
    assert f.status is FeedStatus.UNUSABLE
    assert "ahead of local clock" in f.detail


def test_the_30s_twap_is_not_required():
    """A market configured for a 60s window must not be blocked on a feed
    it does not use."""
    policy = FreshnessPolicy()
    assert FeedKind.CHAINLINK_TWAP_30 not in policy.required
    r = evaluate_freshness(NOW, {
        FeedKind.BOOK: NOW - 0.1, FeedKind.CHAINLINK_REFERENCE: NOW - 1.0,
        FeedKind.CHAINLINK_TWAP_60: NOW - 1.0, FeedKind.BINANCE: NOW - 1.0,
        FeedKind.CHAINLINK_TWAP_30: NOW - 999.0,
    })
    assert r.is_fresh is True


def test_live_book_budget_is_much_tighter_than_the_reference_budgets():
    """Derived from measured rates: the book is a ~130 msg/s stream, the
    reference feeds publish at ~1 Hz."""
    p = FreshnessPolicy()
    assert p.limit_for(FeedKind.BOOK) < p.limit_for(FeedKind.CHAINLINK_REFERENCE)


# ------------------------------------------------ replay-side derivation

def _store_with(events):
    store = EventStore(":memory:")
    for et, recv, src in events:
        store.append(et, "r1", recv_ts=recv, source_ts=src, payload={})
    return store


def test_age_is_measured_from_source_time_not_receive_time():
    """A feed that stopped publishing long ago is stale however promptly
    its last message arrived. Here the last TWAP was OBSERVED at t=10 but
    RECEIVED at t=99, so a receive-time age would report it fresh."""
    store = _store_with([
        (EventType.BOOK_SNAPSHOT, 99.5, 99.5),
        (EventType.SPOT, 99.5, 99.5),
        (EventType.TWAP, 99.5, 10.0),
    ])
    events = store.all_events("r1")
    r = freshness_from_events(events, 100.0, REPLAY_FRESHNESS_POLICY)
    assert r.feeds[FeedKind.CHAINLINK_TWAP_60].age_s == pytest.approx(90.0)
    assert r.is_fresh is False
    assert "chainlink_twap_60:STALE" in r.reason


def test_only_arrived_events_contribute():
    """Visibility is gated on recv_ts even though age is measured on
    source_ts - a message not yet received cannot make a feed fresh."""
    store = _store_with([(EventType.TWAP, 200.0, 99.9)])
    events = store.all_events("r1")
    r = freshness_from_events([e for e in events if e.recv_ts <= 100.0], 100.0,
                              REPLAY_FRESHNESS_POLICY)
    assert r.feeds[FeedKind.CHAINLINK_TWAP_60].status is FeedStatus.MISSING


def test_replay_policy_does_not_require_a_feed_the_replay_cannot_supply():
    """The Phase-2 event vocabulary has no Chainlink-reference event, so
    requiring one would make every replay decision permanently not-fresh -
    and mapping the synthetic SPOT series to it would have manufactured a
    settlement reference out of a leading-price series."""
    assert FeedKind.CHAINLINK_REFERENCE not in REPLAY_FRESHNESS_POLICY.required
    assert REPLAY_FRESHNESS_POLICY.required == {
        FeedKind.BOOK, FeedKind.CHAINLINK_TWAP_60, FeedKind.BINANCE,
    }


def test_replay_budgets_match_the_generators_declared_cadence():
    """Derived from `synthetic/rounds.py`'s own `tick_interval_s=1.0` and
    `resnapshot_interval_s=5.0`, each plus one interval of margin - not
    tuned against results."""
    p = REPLAY_FRESHNESS_POLICY
    assert p.limit_for(FeedKind.BOOK) == 6.0
    assert p.limit_for(FeedKind.CHAINLINK_TWAP_60) == 2.0
    assert p.limit_for(FeedKind.BINANCE) == 2.0
