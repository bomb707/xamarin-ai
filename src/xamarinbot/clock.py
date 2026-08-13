"""Round clock and cross-feed clock synchronization (Roadmap Phase 1).

"Implement market discovery and round clock with explicit start_ts, end_ts,
t and tau."
"Synchronize host clocks and continuously record clock offset/latency
metrics."
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoundClock:
    """t = elapsed seconds in round, tau = remaining seconds
    (Strategy doc SS4: 0 <= t <= 300, tau = 300 - t)."""

    start_ts: float
    end_ts: float

    def t(self, now: float) -> float:
        return now - self.start_ts

    def tau(self, now: float) -> float:
        return self.end_ts - now

    def is_within_round(self, now: float) -> bool:
        return self.start_ts <= now <= self.end_ts

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


@dataclass
class ClockSync:
    """Tracks per-feed offset (recv_ts - source_ts) and latency, over a
    rolling window, so staleness/skew can be monitored continuously rather
    than assumed."""

    window: int = 512
    _samples: deque[tuple[float, float]] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=self.window)

    def record(self, source_ts: float, recv_ts: float) -> None:
        self._samples.append((source_ts, recv_ts))

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def mean_offset(self) -> float | None:
        if not self._samples:
            return None
        return sum(recv - src for src, recv in self._samples) / len(self._samples)

    @property
    def max_offset(self) -> float | None:
        if not self._samples:
            return None
        return max(recv - src for src, recv in self._samples)

    def summary(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "mean_offset_s": self.mean_offset,
            "max_offset_s": self.max_offset,
        }
