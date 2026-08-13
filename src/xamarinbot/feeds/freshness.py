"""Feed freshness monitor (Roadmap Phase 1 deliverable).

"Do not submit orders when state freshness is uncertain." Every feed must
report an explicit freshness status rather than being assumed live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Freshness(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"  # never observed


@dataclass
class FeedFreshnessMonitor:
    max_staleness_s: dict[str, float] = field(default_factory=dict)
    default_max_staleness_s: float = 5.0
    _last_seen: dict[str, float] = field(default_factory=dict)

    def record(self, feed_name: str, event_time: float) -> None:
        prev = self._last_seen.get(feed_name)
        if prev is None or event_time > prev:
            self._last_seen[feed_name] = event_time

    def status(self, feed_name: str, now: float) -> Freshness:
        last = self._last_seen.get(feed_name)
        if last is None:
            return Freshness.UNKNOWN
        threshold = self.max_staleness_s.get(feed_name, self.default_max_staleness_s)
        return Freshness.FRESH if (now - last) <= threshold else Freshness.STALE

    def is_fresh(self, feed_name: str, now: float) -> bool:
        return self.status(feed_name, now) is Freshness.FRESH

    def all_fresh(self, feed_names: list[str], now: float) -> bool:
        return all(self.is_fresh(name, now) for name in feed_names)
