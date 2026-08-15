"""Phase 12C items 7 and 8: round lifecycle and pre-round history."""
from __future__ import annotations

import pytest

from xamarinbot.realtime.lifecycle import (
    LifecycleConfig,
    LifecycleError,
    RoundLifecycle,
    RoundState,
)

START = 1_786_772_100.0
END = START + 300.0


def make(cfg=None, **kwargs):
    return RoundLifecycle(round_id="btc-updown-5m-1786772100", start_ts=START, end_ts=END,
                          cfg=cfg or LifecycleConfig(), **kwargs)


def test_states_are_all_present():
    assert [s.value for s in RoundState] == [
        "DISCOVERED", "PRE_ROUND", "ACTIVE", "ENDED", "RESOLVED", "FINALIZED",
    ]


def test_clock_drives_discovered_through_ended_in_order():
    lc = make()
    lc.advance(START - 1000)
    assert lc.state is RoundState.DISCOVERED
    lc.advance(START - 100)          # inside the 420s pre-round lead
    assert lc.state is RoundState.PRE_ROUND
    lc.advance(START + 10)
    assert lc.state is RoundState.ACTIVE
    lc.advance(END + 1)
    assert lc.state is RoundState.ENDED
    assert [s for s, _, _ in lc.transitions] == [
        RoundState.DISCOVERED, RoundState.PRE_ROUND, RoundState.ACTIVE, RoundState.ENDED,
    ]


def test_no_state_can_be_skipped_even_by_a_late_start():
    """Advancing straight from DISCOVERED past the round end must still
    walk through PRE_ROUND and ACTIVE rather than jumping to ENDED - the
    transition log has to remain a truthful history."""
    lc = make()
    lc.advance(END + 60)
    assert lc.state is RoundState.ENDED
    assert [s for s, _, _ in lc.transitions] == [
        RoundState.DISCOVERED, RoundState.PRE_ROUND, RoundState.ACTIVE, RoundState.ENDED,
    ]


def test_illegal_transitions_raise():
    lc = make()
    with pytest.raises(LifecycleError):
        lc.transition_to(RoundState.FINALIZED)
    lc.advance(END + 1)
    with pytest.raises(LifecycleError):
        lc.transition_to(RoundState.ACTIVE)


def test_a_round_may_finalize_without_a_venue_resolution():
    lc = make()
    lc.advance(END + 1)
    lc.transition_to(RoundState.FINALIZED)   # ENDED -> FINALIZED is legal
    assert lc.is_finished


def test_recording_starts_before_the_round_opens():
    """Item 7: pre-round history is the reason PRE_ROUND exists."""
    lc = make()
    lc.advance(START - 1000)
    assert lc.is_recording is False
    lc.advance(START - 300)
    assert lc.state is RoundState.PRE_ROUND
    assert lc.is_recording is True


def test_pre_round_lead_exceeds_the_largest_feature_window_plus_margin():
    """The largest window any current feature spans is the 300s round
    itself; the default lead must clear it with real margin."""
    cfg = LifecycleConfig()
    assert cfg.pre_round_lead_s >= 300.0 + 60.0
    lc = make(cfg)
    assert lc.pre_round_start_ts == START - cfg.pre_round_lead_s


def test_elapsed_is_negative_before_the_open_and_is_not_clamped():
    """A feature computed at t<0 is pre-round context, not an error."""
    lc = make()
    assert lc.elapsed(START - 120) == pytest.approx(-120.0)
    assert lc.elapsed(START + 30) == pytest.approx(30.0)


def test_remaining_is_tau_and_floors_at_zero():
    lc = make()
    assert lc.remaining(START) == pytest.approx(300.0)
    assert lc.remaining(START + 250) == pytest.approx(50.0)
    assert lc.remaining(END + 99) == 0.0


def test_capture_continues_after_the_round_ends():
    """The settling book, the closing Chainlink observations and any
    market_resolved event all arrive after end_ts."""
    lc = make()
    lc.advance(END + 1)
    assert lc.state is RoundState.ENDED
    assert lc.is_recording is True
    assert lc.finalize_after_ts > END


def test_transitions_are_reported_to_the_callback():
    seen = []
    lc = make(on_transition=lambda rid, old, new, ts: seen.append((old, new)))
    lc.advance(START + 1)
    assert seen == [
        (RoundState.DISCOVERED, RoundState.PRE_ROUND),
        (RoundState.PRE_ROUND, RoundState.ACTIVE),
    ]
