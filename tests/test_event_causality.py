"""Phase 2 exit gate: causality tests pass; replayed portfolio ledger
matches recorded fills exactly when fills are known."""
from __future__ import annotations

from xamarinbot.events.replay import ReplayClock, seeded_random
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType


def _store_with_events() -> EventStore:
    store = EventStore(":memory:")
    store.append(EventType.SPOT, "r1", recv_ts=1.0, source_ts=1.0, payload={"value": 100.0})
    store.append(EventType.SPOT, "r1", recv_ts=5.0, source_ts=5.0, payload={"value": 101.0})
    store.append(EventType.SPOT, "r1", recv_ts=10.0, source_ts=10.0, payload={"value": 102.0})
    return store


def test_future_events_are_inaccessible_at_earlier_decision_time():
    store = _store_with_events()
    clock = ReplayClock(store, "r1")

    visible = clock.events_up_to(5.0)
    assert len(visible) == 2
    assert all(e.event_time <= 5.0 for e in visible)
    assert all(e.payload["value"] != 102.0 for e in visible)  # the t=10 event must not leak


def test_events_up_to_is_monotonically_non_decreasing_in_size():
    store = _store_with_events()
    clock = ReplayClock(store, "r1")
    sizes = [len(clock.events_up_to(t)) for t in (0.0, 1.0, 4.9, 5.0, 9.9, 10.0, 100.0)]
    assert sizes == sorted(sizes)
    assert sizes[-1] == 3


def test_out_of_order_insertion_is_still_ordered_by_event_time_on_read():
    """Events may arrive/be appended out of source-time order (e.g. a
    resnapshot catching up after a disconnect); the store must still expose
    them in causal order."""
    store = EventStore(":memory:")
    store.append(EventType.SPOT, "r1", recv_ts=10.0, source_ts=10.0, payload={"value": "late-arriving-but-old"})
    store.append(EventType.SPOT, "r1", recv_ts=11.0, source_ts=2.0, payload={"value": "arrived-late-source-early"})
    store.append(EventType.SPOT, "r1", recv_ts=12.0, source_ts=1.0, payload={"value": "earliest-source-ts"})

    ordered = store.all_events("r1")
    assert [e.payload["value"] for e in ordered] == [
        "earliest-source-ts",
        "arrived-late-source-early",
        "late-arriving-but-old",
    ]


def test_replay_survives_disconnect_and_resnapshot_style_gap():
    """A disconnect followed by a BOOK_SNAPSHOT resnapshot should not make
    earlier events reappear or later events leak - the replay clock only
    ever cares about event_time <= decision_time, regardless of connection
    gaps in between."""
    store = EventStore(":memory:")
    store.append(EventType.BOOK_SNAPSHOT, "r1", recv_ts=0.0, source_ts=0.0, payload={"side": "UP", "bids": [], "asks": [[0.5, 10]]})
    store.append(EventType.BOOK_DELTA, "r1", recv_ts=1.0, source_ts=1.0, payload={"side": "UP", "book": "asks", "price": 0.5, "size": 0})
    # simulated disconnect gap between t=1 and t=8 (no events)
    store.append(EventType.BOOK_SNAPSHOT, "r1", recv_ts=8.0, source_ts=8.0, payload={"side": "UP", "bids": [], "asks": [[0.6, 5]]})

    clock = ReplayClock(store, "r1")
    assert len(clock.events_up_to(0.5)) == 1
    assert len(clock.events_up_to(1.0)) == 2
    assert len(clock.events_up_to(7.9)) == 2  # still in the gap
    assert len(clock.events_up_to(8.0)) == 3


def test_seeded_random_is_reproducible_across_instances():
    r1 = seeded_random("round-1", 42)
    r2 = seeded_random("round-1", 42)
    assert [r1.random() for _ in range(10)] == [r2.random() for _ in range(10)]


def test_seeded_random_differs_across_keys():
    r1 = seeded_random("round-1", 1)
    r2 = seeded_random("round-1", 2)
    assert [r1.random() for _ in range(5)] != [r2.random() for _ in range(5)]


def test_partial_fill_then_cancel_remainder_is_ordered_causally():
    store = EventStore(":memory:")
    store.append(EventType.ORDER_SUBMIT, "r1", recv_ts=1.0, source_ts=1.0, payload={"order_id": "o1", "size": 10})
    store.append(EventType.FILL, "r1", recv_ts=2.0, source_ts=2.0, payload={"order_id": "o1", "size": 4})
    store.append(EventType.CANCEL, "r1", recv_ts=3.0, source_ts=3.0, payload={"order_id": "o1", "remaining": 6})

    clock = ReplayClock(store, "r1")
    visible_after_fill = clock.events_up_to(2.0)
    kinds = [e.event_type for e in visible_after_fill]
    assert EventType.CANCEL not in kinds
    assert EventType.FILL in kinds

    visible_after_cancel = clock.events_up_to(3.0)
    assert EventType.CANCEL in [e.event_type for e in visible_after_cancel]
