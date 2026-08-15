"""Explicit round lifecycle (Phase 12C item 8).

    DISCOVERED -> PRE_ROUND -> ACTIVE -> ENDED -> RESOLVED -> FINALIZED

Making these states explicit is what item 7's requirement rests on:
PRE_ROUND is not a formality, it is the window in which BTC reference /
TWAP / Binance data is recorded BEFORE the round opens, so that early-round
momentum and TWAP history are computed from actual observations instead of
being approximated from `p0`.

    "Preserve enough lookback for the largest feature window plus a safety
     margin."

`pre_round_lead_s` defaults to 420 seconds. The largest window any current
feature uses is the 300-second round itself (`FeatureConfig`'s canonical
horizons and volatility window all sit inside it), and 120 seconds of
margin covers the 60-second TWAP lookback needing its own warm-up plus a
slow subscribe. It is a config field, not a constant, precisely because the
"largest feature window" is a property of the feature set and will move.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class RoundState(str, Enum):
    DISCOVERED = "DISCOVERED"   # metadata fetched; nothing recorded yet
    PRE_ROUND = "PRE_ROUND"     # recording reference history before the open
    ACTIVE = "ACTIVE"           # inside [start_ts, end_ts)
    ENDED = "ENDED"             # past end_ts, awaiting the venue's resolution
    RESOLVED = "RESOLVED"       # venue reported an outcome
    FINALIZED = "FINALIZED"     # metadata + label + metrics persisted, buffers flushed


#: The only transitions allowed. A recorder that could skip ENDED would be
#: able to finalize a round it never saw close.
_ALLOWED: dict[RoundState, frozenset[RoundState]] = {
    RoundState.DISCOVERED: frozenset({RoundState.PRE_ROUND, RoundState.ACTIVE}),
    RoundState.PRE_ROUND: frozenset({RoundState.ACTIVE}),
    RoundState.ACTIVE: frozenset({RoundState.ENDED}),
    # RESOLVED is optional: a round can be finalized without the venue
    # having published a resolution yet, and that fact is itself recorded
    # rather than being waited on indefinitely.
    RoundState.ENDED: frozenset({RoundState.RESOLVED, RoundState.FINALIZED}),
    RoundState.RESOLVED: frozenset({RoundState.FINALIZED}),
    RoundState.FINALIZED: frozenset(),
}


class LifecycleError(RuntimeError):
    pass


@dataclass(frozen=True)
class LifecycleConfig:
    #: How long before `start_ts` to enter PRE_ROUND and begin recording.
    pre_round_lead_s: float = 420.0
    #: How long after `end_ts` to keep recording before finalizing, so the
    #: settling book, the closing Chainlink observations and any
    #: `market_resolved` event are all captured.
    post_round_tail_s: float = 90.0
    #: How long to keep polling the venue for a resolution before
    #: finalizing without one.
    resolution_wait_s: float = 60.0


@dataclass
class RoundLifecycle:
    """One round's state machine, plus the transition log.

    Every transition is recorded with both wall and monotonic clocks, and is
    emitted to the raw log through `on_transition` so the capture's own
    history is replayable next to the market data.
    """

    round_id: str
    start_ts: float
    end_ts: float
    cfg: LifecycleConfig = field(default_factory=LifecycleConfig)
    state: RoundState = RoundState.DISCOVERED
    transitions: list[tuple[RoundState, float, int]] = field(default_factory=list)
    on_transition: Callable[[str, RoundState, RoundState, float], None] | None = None

    def __post_init__(self) -> None:
        self.transitions.append((self.state, time.time(), time.monotonic_ns()))

    # ---------------------------------------------------------- transition

    def transition_to(self, new_state: RoundState, now: float | None = None) -> None:
        if new_state not in _ALLOWED[self.state]:
            raise LifecycleError(
                f"{self.round_id}: illegal transition {self.state.value} -> {new_state.value}"
            )
        old = self.state
        self.state = new_state
        now = time.time() if now is None else now
        self.transitions.append((new_state, now, time.monotonic_ns()))
        if self.on_transition is not None:
            self.on_transition(self.round_id, old, new_state, now)

    # ------------------------------------------------------------ windows

    @property
    def pre_round_start_ts(self) -> float:
        return self.start_ts - self.cfg.pre_round_lead_s

    @property
    def finalize_after_ts(self) -> float:
        return self.end_ts + self.cfg.post_round_tail_s

    def target_state(self, now: float) -> RoundState:
        """The state the clock alone implies. The service uses this to drive
        DISCOVERED->PRE_ROUND->ACTIVE->ENDED; RESOLVED and FINALIZED are
        driven by events and by the service's own completion work, not by
        the clock."""
        if now < self.pre_round_start_ts:
            return RoundState.DISCOVERED
        if now < self.start_ts:
            return RoundState.PRE_ROUND
        if now < self.end_ts:
            return RoundState.ACTIVE
        return RoundState.ENDED

    def advance(self, now: float | None = None) -> list[RoundState]:
        """Step the machine forward to whatever the clock implies, one legal
        transition at a time. Returns the states entered."""
        now = time.time() if now is None else now
        target = self.target_state(now)
        entered: list[RoundState] = []
        order = [RoundState.DISCOVERED, RoundState.PRE_ROUND, RoundState.ACTIVE, RoundState.ENDED]
        if self.state not in order or target not in order:
            return entered
        while order.index(self.state) < order.index(target):
            nxt = order[order.index(self.state) + 1]
            self.transition_to(nxt, now)
            entered.append(nxt)
        return entered

    @property
    def is_recording(self) -> bool:
        """Whether this round's market data should be being captured now.
        True from PRE_ROUND through ENDED - item 7's whole point is that
        recording starts before ACTIVE."""
        return self.state in (RoundState.PRE_ROUND, RoundState.ACTIVE, RoundState.ENDED)

    @property
    def is_finished(self) -> bool:
        return self.state is RoundState.FINALIZED

    def elapsed(self, now: float) -> float:
        """`t` - seconds since the round opened. Negative during PRE_ROUND,
        which is meaningful and must not be clamped: a feature computed at
        t<0 is pre-round context, not an error."""
        return now - self.start_ts

    def remaining(self, now: float) -> float:
        """`tau` - seconds until the round closes, floored at 0."""
        return max(0.0, self.end_ts - now)
