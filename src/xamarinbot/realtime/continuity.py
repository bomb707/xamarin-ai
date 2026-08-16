"""Offline per-topic RTDS continuity audit (Gate A.0.2.1 items 5 and 6).

Why this exists
---------------
A.0.2 closed the gap where a feed outage left a round looking clean, but it
tracked continuity at the SOCKET level. The RTDS subscription is
deliberately unfiltered (filtered subscriptions do not deliver), so the
socket carries every asset the venue publishes. ETH and SOL ticks at 20/s
keep a socket-level liveness mark perfectly fresh while every BTC
observation has vanished.

The strategy does not consume "the RTDS socket". It consumes four
independent series - Binance BTCUSDT, Chainlink btc/usd, and the 30s and
60s TWAPs - and a label reconstructed from one specific one of them. A
TWAP-60 blackout with the other three healthy is invisible to a socket-level
watchdog and fatal to the label.

This module answers that question from the raw log alone, so captures
recorded before the per-topic watchdog existed can still be audited: the
topic timelines were always stored separately, only the analysis was
aggregate.

On thresholds
-------------
`gap_threshold_s` is a DATA-QUALITY ASSUMPTION, not a tuned parameter. It is
never selected by how many rounds it leaves eligible. `interarrival_summary`
reports the empirical distribution per topic and per clock so the assumption
can be checked against what the feeds actually do, and
`threshold_sensitivity` reports how the verdict moves across a range so a
single number is never load-bearing on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.realtime.attribution import RoundWindow
from xamarinbot.realtime.raw_events import Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.realtime.rtds import (
    TOPIC_BINANCE,
    TOPIC_CHAINLINK,
    TOPIC_TWAP_30,
    TOPIC_TWAP_60,
)

#: The four BTC series the strategy and the label actually depend on, with
#: the raw topic each is stored under and the symbol it must carry.
REQUIRED_TOPICS: dict[str, Topic] = {
    TOPIC_BINANCE: Topic.RTDS_BINANCE,
    TOPIC_CHAINLINK: Topic.RTDS_CHAINLINK,
    TOPIC_TWAP_30: Topic.RTDS_TWAP_30,
    TOPIC_TWAP_60: Topic.RTDS_TWAP_60,
}

#: Nominal publication cadence: every one of these series publishes at
#: ~1 Hz. Used only to state the assumption explicitly - never to decide it.
NOMINAL_CADENCE_S = 1.0

#: A missing-observation gap for a ~1 Hz series, as a DATA-QUALITY
#: ASSUMPTION - documented, and deliberately not tuned.
#:
#: It happens not to be load-bearing, which is the best case available.
#: Measured over every capture to date (~191,000 interarrivals across the
#: four series), the receive-time distribution is sharply bimodal:
#:
#:     topic                        0-2s    2-5s   5-10s  10-25s  25-40s
#:     crypto_prices               50067       6       0       0      45
#:     crypto_prices_chainlink     46877     694      25       0      45
#:     crypto_prices_twap_thirty   46850     725      26       0      45
#:     crypto_prices_twap_sixty    46865     717      26       0      45
#:
#: There is an EMPTY BAND from 10s to 25s on all four series: normal
#: operation never exceeds ~9s, and a real outage never comes in under 25s.
#: Every threshold in that band therefore yields identical verdicts, so no
#: choice within it can be accused of having been picked for its answer.
#: `threshold_sensitivity` reports the curve so this stays checkable as
#: more data arrives rather than being taken on trust.
#:
#: 10s is also well below the 30s the live watchdog needs before it can fire
#: at all, so the offline audit can see outages the live path structurally
#: cannot.
DEFAULT_GAP_THRESHOLD_S = 10.0

#: The range reported for sensitivity. Chosen to bracket the assumption by
#: an order of magnitude either way, BEFORE looking at what it does to the
#: eligible count.
SENSITIVITY_THRESHOLDS_S = (2.0, 5.0, 10.0, 20.0, 30.0, 60.0)


def quantile(sorted_xs: list[float], q: float) -> float:
    """Nearest-rank quantile. No interpolation: an interarrival that is
    reported as the p99 should be an interarrival that actually happened."""
    if not sorted_xs:
        return float("nan")
    idx = min(len(sorted_xs) - 1, max(0, int(round(q * len(sorted_xs))) - 1))
    return sorted_xs[idx]


@dataclass(frozen=True)
class Interarrivals:
    """The empirical spacing of one series on one clock."""

    clock: str  # "source" | "recv"
    n: int
    median: float
    p95: float
    p99: float
    p999: float
    max: float

    @classmethod
    def of(cls, deltas_s: list[float], clock: str) -> "Interarrivals":
        xs = sorted(deltas_s)
        return cls(
            clock=clock, n=len(xs),
            median=quantile(xs, 0.5), p95=quantile(xs, 0.95),
            p99=quantile(xs, 0.99), p999=quantile(xs, 0.999),
            max=xs[-1] if xs else float("nan"),
        )

    def as_dict(self) -> dict:
        return {
            "clock": self.clock, "n": self.n, "median": self.median,
            "p95": self.p95, "p99": self.p99, "p999": self.p999, "max": self.max,
        }


@dataclass(frozen=True)
class TopicGap:
    """One interval on one topic with no observation in it."""

    wire_topic: str
    start_ns: int
    end_ns: int

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def overlaps(self, window: RoundWindow) -> bool:
        return window.overlaps(self.start_ns, self.end_ns)


@dataclass
class TopicContinuity:
    """One BTC series' continuity across a whole capture."""

    wire_topic: str
    expected_symbol: str
    observations: int = 0
    #: Both clocks, separately: `source` is the publisher's own stamp and
    #: `recv` is when it reached us. They fail differently - a stalled
    #: publisher and a stalled socket look identical on one clock and
    #: nothing alike on both.
    source: Interarrivals | None = None
    recv: Interarrivals | None = None
    gaps: list[TopicGap] = field(default_factory=list)
    first_recv_ns: int | None = None
    last_recv_ns: int | None = None

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    @property
    def longest_gap_s(self) -> float:
        return max((g.duration_s for g in self.gaps), default=0.0)

    def gaps_overlapping(self, window: RoundWindow) -> list[TopicGap]:
        return [g for g in self.gaps if g.overlaps(window)]

    def as_dict(self) -> dict:
        return {
            "wire_topic": self.wire_topic,
            "expected_symbol": self.expected_symbol,
            "observations": self.observations,
            "gap_count": self.gap_count,
            "longest_gap_s": self.longest_gap_s,
            "source_interarrival": self.source.as_dict() if self.source else None,
            "recv_interarrival": self.recv.as_dict() if self.recv else None,
        }


def topic_timeline(raw: RawEventStore, topic: Topic) -> tuple[list[int], list[int]]:
    """`(recv_ns, source_ns)` for every observation on one series, in order.

    Read straight from the raw log, which stored each topic separately from
    the beginning - which is what makes captures written before the
    per-topic watchdog auditable at all.
    """
    recvs: list[int] = []
    sources: list[int] = []
    for e in raw.events(topics=[topic]):
        recvs.append(e.recv_wall_timestamp_ns)
        if e.source_timestamp_ns is not None:
            sources.append(e.source_timestamp_ns)
    recvs.sort()
    sources.sort()
    return recvs, sources


def _deltas_s(xs: list[int]) -> list[float]:
    return [(b - a) / 1e9 for a, b in zip(xs, xs[1:])]


def audit_topic(
    raw: RawEventStore,
    wire_topic: str,
    topic: Topic,
    *,
    expected_symbol: str = "btc/usd",
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
) -> TopicContinuity:
    """Continuity of ONE BTC series, measured from its own observations.

    A gap is a spacing in RECEIVE time longer than the threshold: receive
    time is when a live system could actually have acted on the
    observation, which is the same clock the causal feature gate uses.
    """
    recvs, sources = topic_timeline(raw, topic)
    result = TopicContinuity(
        wire_topic=wire_topic, expected_symbol=expected_symbol,
        observations=len(recvs),
        first_recv_ns=recvs[0] if recvs else None,
        last_recv_ns=recvs[-1] if recvs else None,
    )
    if len(recvs) >= 2:
        result.recv = Interarrivals.of(_deltas_s(recvs), "recv")
    if len(sources) >= 2:
        result.source = Interarrivals.of(_deltas_s(sources), "source")

    threshold_ns = int(gap_threshold_s * 1e9)
    for a, b in zip(recvs, recvs[1:]):
        if b - a > threshold_ns:
            result.gaps.append(TopicGap(wire_topic, a, b))
    return result


def audit_capture(
    raw: RawEventStore, *, gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S
) -> dict[str, TopicContinuity]:
    """Per-topic continuity for every required BTC series in a capture."""
    from xamarinbot.realtime.rtds import BTC_SYMBOLS

    return {
        wire: audit_topic(
            raw, wire, topic,
            expected_symbol=BTC_SYMBOLS.get(wire, "btc/usd"),
            gap_threshold_s=gap_threshold_s,
        )
        for wire, topic in REQUIRED_TOPICS.items()
    }


def rounds_affected_by_topic_gaps(
    raw: RawEventStore, *, gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S
) -> dict[str, list[str]]:
    """`round_id -> ["rtds:<wire_topic>", ...]` for every round whose
    required window overlaps a gap on that series.

    This is the offline equivalent of the live per-topic watchdog, and it
    finds outages the live watchdog structurally could not: a topic that
    stalls for 12 seconds never reaches the 30-second stall timeout, and a
    topic that stalls while other assets keep the socket busy never reached
    the A.0.2 watchdog at all.
    """
    from xamarinbot.realtime.preflight import round_windows

    windows = round_windows(raw)
    audit = audit_capture(raw, gap_threshold_s=gap_threshold_s)
    out: dict[str, list[str]] = {}
    for wire, continuity in audit.items():
        for window in windows:
            if continuity.gaps_overlapping(window):
                out.setdefault(window.round_id, []).append(f"rtds:{wire}")
    return out


def threshold_sensitivity(
    raw: RawEventStore, thresholds_s: tuple[float, ...] = SENSITIVITY_THRESHOLDS_S
) -> dict[float, dict]:
    """How the verdict moves with the assumption.

    Item 6: a threshold must never be selected by the answer it produces.
    Reporting the whole curve is what keeps that honest - if the eligible
    count is stable across an order of magnitude, the constant is not
    load-bearing; if it swings, the audit says so out loud rather than
    presenting one number as a fact.
    """
    out: dict[float, dict] = {}
    for t in thresholds_s:
        affected = rounds_affected_by_topic_gaps(raw, gap_threshold_s=t)
        audit = audit_capture(raw, gap_threshold_s=t)
        out[t] = {
            "rounds_affected": len(affected),
            "gaps_by_topic": {w: c.gap_count for w, c in audit.items()},
            "longest_gap_s": {w: c.longest_gap_s for w, c in audit.items()},
        }
    return out
