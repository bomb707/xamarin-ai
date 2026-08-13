"""Causal replay engine (Roadmap Phase 2).

"Build replay clock that exposes only events with event_time <= decision_time."
"Support event-driven decisions plus a configurable heartbeat."
"Add reproducible random seeds for stochastic fill models used only when
historical queue data is insufficient."
"""
from __future__ import annotations

import hashlib
import random

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import Event


class ReplayClock:
    """Exposes only the causal prefix of the event log visible at a given
    decision time. Never returns an event with event_time > decision_time -
    this is the single invariant that makes replay a valid stand-in for live
    execution (verified by test_event_causality.py)."""

    def __init__(self, store: EventStore, round_id: str):
        self._round_id = round_id
        self._events = store.all_events(round_id=round_id)

    def events_up_to(self, decision_time: float) -> list[Event]:
        """All events with event_time <= decision_time, in deterministic
        order. O(n) scan is fine at this data scale; the event list is
        already sorted so this could be bisected if it becomes a hot path."""
        return [e for e in self._events if e.event_time <= decision_time]

    def last_event_up_to(self, decision_time: float) -> Event | None:
        visible = self.events_up_to(decision_time)
        return visible[-1] if visible else None

    @property
    def all_events(self) -> list[Event]:
        return list(self._events)

    def decision_points(self, heartbeat: float | None = None) -> list[float]:
        """Merge event-driven decision times (one per event) with an
        optional fixed-interval heartbeat between the first and last event,
        deduplicated and sorted - "event-driven decisions plus a
        configurable heartbeat."""
        if not self._events:
            return []
        times = {e.event_time for e in self._events}
        if heartbeat is not None and heartbeat > 0:
            start = self._events[0].event_time
            end = self._events[-1].event_time
            t = start
            while t <= end:
                times.add(t)
                t += heartbeat
        return sorted(times)


class ReplayRunner:
    """Drives a decision function over a round's causal replay clock,
    calling it once per decision point with only the causally-visible event
    prefix."""

    def __init__(self, store: EventStore, round_id: str, heartbeat: float | None = None):
        self.round_id = round_id
        self.clock = ReplayClock(store, round_id)
        self.heartbeat = heartbeat

    def run(self, decision_fn):
        """decision_fn(decision_time, visible_events) -> Any. Returns the
        list of (decision_time, result) pairs in order."""
        results = []
        for decision_time in self.clock.decision_points(self.heartbeat):
            visible = self.clock.events_up_to(decision_time)
            results.append((decision_time, decision_fn(decision_time, visible)))
        return results


def seeded_random(round_id: str, event_id: int | str) -> random.Random:
    """A reproducible Random instance keyed by (round_id, event_id), used
    for stochastic fill models when historical queue data is insufficient.
    Uses a stable SHA-256-derived seed rather than Python's process-salted
    hash(), so the same key always yields the same stream across processes
    and runs."""
    digest = hashlib.sha256(f"{round_id}:{event_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)
