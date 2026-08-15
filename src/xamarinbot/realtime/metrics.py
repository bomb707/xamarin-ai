"""Recorder health metrics (Phase 12C item 6).

The brief's non-negotiable rule is encoded here as `is_training_grade`:

    "A dropped-event counter greater than zero must make that capture
     interval unsuitable for high-fidelity model training unless explicitly
     repaired."

That is a property of the *data*, not a warning to be logged and forgotten,
so it is a method on the metrics object that the capture report reads and
prints, and it defaults to False for anything the recorder could not prove
clean.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    """Streaming latency summary. Keeps a bounded reservoir of samples for
    percentiles rather than every sample, because a multi-round capture at
    ~130 events/second would otherwise accumulate millions of floats purely
    to compute a p99."""

    name: str
    count: int = 0
    total_ns: int = 0
    min_ns: int | None = None
    max_ns: int | None = None
    _reservoir: list[int] = field(default_factory=list)
    _reservoir_limit: int = 20000
    _rng_state: int = 0

    def record(self, latency_ns: int | None) -> None:
        if latency_ns is None:
            return
        self.count += 1
        self.total_ns += latency_ns
        if self.min_ns is None or latency_ns < self.min_ns:
            self.min_ns = latency_ns
        if self.max_ns is None or latency_ns > self.max_ns:
            self.max_ns = latency_ns
        if len(self._reservoir) < self._reservoir_limit:
            self._reservoir.append(latency_ns)
        else:
            # Deterministic reservoir sampling (Vitter R) with an LCG, so a
            # capture's reported percentiles are reproducible from its own
            # event sequence rather than depending on process randomness.
            self._rng_state = (self._rng_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
            j = self._rng_state % self.count
            if j < self._reservoir_limit:
                self._reservoir[j] = latency_ns

    @property
    def mean_ns(self) -> float | None:
        return self.total_ns / self.count if self.count else None

    def percentile_ns(self, p: float) -> int | None:
        """Nearest-rank percentile over the reservoir. `p` in [0, 100]."""
        if not self._reservoir:
            return None
        ordered = sorted(self._reservoir)
        rank = max(1, math.ceil(p / 100.0 * len(ordered)))
        return ordered[min(rank, len(ordered)) - 1]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": (self.mean_ns / 1e6) if self.mean_ns is not None else None,
            "min_ms": (self.min_ns / 1e6) if self.min_ns is not None else None,
            "p50_ms": self._pct_ms(50),
            "p95_ms": self._pct_ms(95),
            "p99_ms": self._pct_ms(99),
            "max_ms": (self.max_ns / 1e6) if self.max_ns is not None else None,
            "sampled": len(self._reservoir),
        }

    def _pct_ms(self, p: float) -> float | None:
        v = self.percentile_ns(p)
        return (v / 1e6) if v is not None else None


@dataclass
class RecorderMetrics:
    """Every counter the brief names, plus the two latency distributions.

    Thread-safe: the WebSocket reader threads increment `events_received`
    and the drop/parse counters, while the writer thread increments
    `events_persisted`, so the counters genuinely are shared state.
    """

    events_received: int = 0
    events_persisted: int = 0
    queue_high_water: int = 0
    dropped_events: int = 0
    parse_failures: int = 0
    duplicate_events: int = 0
    reconnect_count: int = 0
    resnapshot_count: int = 0
    book_integrity_checks: int = 0
    book_integrity_mismatches: int = 0
    freshness_failures: int = 0
    suspect_intervals: int = 0
    source_to_recv: LatencyStats = field(default_factory=lambda: LatencyStats("source->recv"))
    publisher_to_recv: LatencyStats = field(default_factory=lambda: LatencyStats("publisher->recv"))
    per_topic_received: dict[str, int] = field(default_factory=dict)
    per_event_type_received: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_received(self, topic: str, event_type: str, source_latency_ns: int | None, publisher_latency_ns: int | None) -> None:
        with self._lock:
            self.events_received += 1
            self.per_topic_received[topic] = self.per_topic_received.get(topic, 0) + 1
            key = f"{topic}:{event_type}"
            self.per_event_type_received[key] = self.per_event_type_received.get(key, 0) + 1
            self.source_to_recv.record(source_latency_ns)
            self.publisher_to_recv.record(publisher_latency_ns)

    def record_persisted(self, n: int) -> None:
        with self._lock:
            self.events_persisted += n

    def record_dropped(self, n: int = 1) -> None:
        with self._lock:
            self.dropped_events += n

    def record_parse_failure(self) -> None:
        with self._lock:
            self.parse_failures += 1

    def record_duplicate(self) -> None:
        with self._lock:
            self.duplicate_events += 1

    def record_reconnect(self) -> None:
        with self._lock:
            self.reconnect_count += 1

    def record_resnapshot(self) -> None:
        with self._lock:
            self.resnapshot_count += 1

    def record_integrity_check(self, matched: bool) -> None:
        with self._lock:
            self.book_integrity_checks += 1
            if not matched:
                self.book_integrity_mismatches += 1
                self.suspect_intervals += 1

    def record_freshness_failure(self) -> None:
        with self._lock:
            self.freshness_failures += 1

    def observe_queue_depth(self, depth: int) -> None:
        with self._lock:
            if depth > self.queue_high_water:
                self.queue_high_water = depth

    def is_training_grade(self) -> bool:
        """Item 6's rule, as data rather than as a log line: any dropped
        event, any parse failure, or any book-integrity mismatch makes the
        interval unsuitable for high-fidelity model training unless a human
        explicitly repairs and re-marks it."""
        return (
            self.dropped_events == 0
            and self.parse_failures == 0
            and self.book_integrity_mismatches == 0
        )

    def disqualifiers(self) -> list[str]:
        """Why `is_training_grade()` is False, stated explicitly rather than
        leaving the caller to re-derive it."""
        out: list[str] = []
        if self.dropped_events:
            out.append(f"dropped_events={self.dropped_events}")
        if self.parse_failures:
            out.append(f"parse_failures={self.parse_failures}")
        if self.book_integrity_mismatches:
            out.append(f"book_integrity_mismatches={self.book_integrity_mismatches}")
        return out

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "events_received": self.events_received,
                "events_persisted": self.events_persisted,
                "queue_high_water": self.queue_high_water,
                "dropped_events": self.dropped_events,
                "parse_failures": self.parse_failures,
                "duplicate_events": self.duplicate_events,
                "reconnect_count": self.reconnect_count,
                "resnapshot_count": self.resnapshot_count,
                "book_integrity_checks": self.book_integrity_checks,
                "book_integrity_mismatches": self.book_integrity_mismatches,
                "freshness_failures": self.freshness_failures,
                "suspect_intervals": self.suspect_intervals,
                "source_to_recv": self.source_to_recv.as_dict(),
                "publisher_to_recv": self.publisher_to_recv.as_dict(),
                "per_topic_received": dict(self.per_topic_received),
                "per_event_type_received": dict(self.per_event_type_received),
                "is_training_grade": self.is_training_grade(),
                "disqualifiers": self.disqualifiers(),
            }
