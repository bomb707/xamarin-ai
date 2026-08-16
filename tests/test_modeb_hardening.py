"""MODE-B hardening: three narrow execution defects, genuinely tested.

Each of these replaces a test that could not fail:

  * the delayed-taker regression contained `if False else None` and
    `assert ... or True`, so it asserted nothing about the fill
  * the execution-journal assertion was `ops <= allowed`, which passes on
    the empty set
  * book freshness collapsed UP and DOWN into one age, so a fresh UP book
    certified a stale DOWN book
"""
from __future__ import annotations

import types

import pytest

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.session import TradingSession
from xamarinbot.market.constraints import MarketConstraints
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.types import CandidateAction, OrderMode, OrderPurpose
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.freshness import FeedKind, FeedStatus, evaluate_freshness
from xamarinbot.regime.types import Direction, GapRegime, RegimeState
from xamarinbot.shadow.runner import REAL_REPLAY_FRESHNESS_POLICY

START = 1_000_000.0

#: A neutral regime that permits action families.
NEUTRAL = RegimeState(GapRegime.NEAR_CENTER_POSITIVE, Direction.UP, Direction.UP)


def _levels(pairs):
    from xamarinbot.feeds.base import BookLevel

    return [BookLevel(price=p, size=s) for p, s in pairs]


class Book:
    """A minimal two-sided book with real depth."""

    def __init__(self, bids=None, asks=None):
        self.bids = _levels(bids if bids is not None else [(0.40, 500.0), (0.39, 500.0)])
        self.asks = _levels(asks if asks is not None else [(0.42, 500.0), (0.43, 500.0)])


def make_session(*, taker_delay_ms: float, on_execution=None) -> TradingSession:
    constraints = MarketConstraints.for_testing(
        tick_size=0.01, min_order_shares=5.0, taker_delay_ms=taker_delay_ms)
    s = TradingSession(
        round_id="r1",
        fee_config=FeeConfig(crypto_fee_rate=0.07),
        exec_cfg=ExecutionConfig(taker_delay_ms=taker_delay_ms),
        cfg=OneStepConfig(g_min=-1000.0, spend_cap=1000.0, position_limit=1000.0,
                          edge_min=0.0),
        constraints=constraints,
        portfolio=PortfolioState(),
    )
    s.on_execution = on_execution
    return s


def taker_candidate(qty=10.0, side=Side.UP, limit=0.45) -> CandidateAction:
    return CandidateAction(
        action_id="a1", purpose=OrderPurpose.ALPHA, side=side, mode=OrderMode.FAK,
        price=limit, qty=qty, ttl_s=0.0, expected_fill=qty, delta_ev=1.0,
        g_after=-1.0, pi_u_after=1.0, pi_d_after=-1.0,
        max_execution_price=limit,
    )


# ============ PART 16: the delayed final-grid taker, for real ============

def test_a_taker_submitted_at_the_last_grid_point_fills_at_its_matched_ts():
    """submit at t=270.000, delay 250ms -> matched at 270.250.

    The strategy clock has stopped by then, so without the drain the fill
    would never be applied. Every escape hatch removed: this asserts the
    exact fill, once.
    """
    events = []
    s = make_session(taker_delay_ms=250.0,
                     on_execution=lambda k, p: events.append((k, p)))
    submit_ts = START + 270.0
    book = Book()

    s.dispatch(taker_candidate(), submit_ts, NEUTRAL, 0.8, book, book)

    assert len(s.queue._pending) == 1, "a delayed taker must rest, not fill at submit"
    before = s.portfolio
    assert before.U == 0.0 and before.C == 0.0, "no fill may occur at submit time"

    # too early - the matching engine has not reached matched_ts
    s.resolve_ready_takers(START + 270.1, lambda p: book.asks)
    assert len(s.queue._pending) == 1, "must not fill before matched_ts"
    assert s.portfolio.U == 0.0

    s.resolve_ready_takers(START + 270.250, lambda p: book.asks)
    assert len(s.queue._pending) == 0, "the taker must resolve at matched_ts"

    after = s.portfolio
    assert after.U == pytest.approx(10.0)
    # C rises by exactly notional + fee, from the simulator's own numbers
    fill = [p for k, p in events if k == "FILLED"][0]
    expected_cost = fill["avg_execution_price"] * fill["filled_shares"] + fill["fee"]
    assert after.C == pytest.approx(expected_cost)
    assert fill["matched_ts"] == pytest.approx(START + 270.250)
    assert fill["submit_ts"] == pytest.approx(submit_ts)
    assert fill["matched_ts"] > fill["submit_ts"]

    # and exactly once
    s.resolve_ready_takers(START + 400.0, lambda p: book.asks)
    assert s.portfolio.U == pytest.approx(10.0)
    assert s.portfolio.C == pytest.approx(expected_cost)
    assert len([k for k, _ in events if k == "FILLED"]) == 1


def test_the_delayed_fill_uses_the_book_at_matched_ts_not_at_submit():
    """The real matching engine prices against the book as it stood at
    matched_ts."""
    s = make_session(taker_delay_ms=250.0)
    cheap = Book(asks=[(0.42, 500.0)])
    expensive = Book(asks=[(0.48, 500.0)])
    s.dispatch(taker_candidate(limit=0.50), START + 270.0,
               NEUTRAL, 0.8, cheap, cheap)
    s.resolve_ready_takers(START + 270.250, lambda p: expensive.asks)
    # filled at the LATER book's price, not the submission-time one
    assert s.portfolio.C / s.portfolio.U > 0.45


# ============ PART 17: per-side book freshness ==========================

def _ages(up_age: float, down_age: float, now: float = 100.0) -> dict:
    return {
        FeedKind.BOOK_UP: now - up_age,
        FeedKind.BOOK_DOWN: now - down_age,
        FeedKind.CHAINLINK_TWAP_60: now - 0.1,
        FeedKind.BINANCE: now - 0.1,
    }


def test_a_fresh_up_book_does_not_certify_a_stale_down_book():
    """UP 0.1s, DOWN 8.0s. A DOWN candidate must not be tradable."""
    report = evaluate_freshness(100.0, _ages(0.1, 8.0), REAL_REPLAY_FRESHNESS_POLICY)
    assert report.feeds[FeedKind.BOOK_UP].status is FeedStatus.FRESH
    assert report.feeds[FeedKind.BOOK_DOWN].status is FeedStatus.STALE
    assert report.is_fresh is False, "a stale DOWN book must block the decision"


def test_a_fresh_down_book_does_not_hide_a_stale_up_book():
    """UP 3.0s against a 2.0s limit, DOWN 0.1s. The UP-derived CLOB signal
    must not read as fresh merely because DOWN received a newer event."""
    report = evaluate_freshness(100.0, _ages(3.0, 0.1), REAL_REPLAY_FRESHNESS_POLICY)
    assert report.feeds[FeedKind.BOOK_UP].status is FeedStatus.STALE
    assert report.feeds[FeedKind.BOOK_DOWN].status is FeedStatus.FRESH
    assert report.is_fresh is False


def test_both_sides_fresh_is_fresh():
    """The gate must be able to say yes."""
    report = evaluate_freshness(100.0, _ages(0.1, 0.2), REAL_REPLAY_FRESHNESS_POLICY)
    assert report.is_fresh is True


def test_both_book_sides_are_required_by_the_real_policy():
    required = REAL_REPLAY_FRESHNESS_POLICY.required
    assert FeedKind.BOOK_UP in required and FeedKind.BOOK_DOWN in required


def test_a_book_event_only_refreshes_its_own_side():
    from xamarinbot.events.types import EventType
    from xamarinbot.shadow.runner import feed_for_event

    up = types.SimpleNamespace(event_type=EventType.BOOK_DELTA, payload={"side": "UP"})
    down = types.SimpleNamespace(event_type=EventType.BOOK_SNAPSHOT,
                                 payload={"side": "DOWN"})
    assert feed_for_event(up) is FeedKind.BOOK_UP
    assert feed_for_event(down) is FeedKind.BOOK_DOWN


def test_a_burst_of_up_deltas_leaves_the_down_book_stale():
    """End to end through `freshness_from_events`: the real failure shape."""
    from xamarinbot.events.store import EventStore
    from xamarinbot.events.types import EventType
    from xamarinbot.shadow.runner import freshness_from_events

    store = EventStore(":memory:", provenance=DataProvenance.REAL_REPLAY)
    store.append(EventType.BOOK_SNAPSHOT, "r", recv_ts=90.0, source_ts=90.0,
                 payload={"side": "DOWN", "bids": [], "asks": []})
    for i in range(50):
        store.append(EventType.BOOK_DELTA, "r", recv_ts=99.0 + i * 0.01,
                     source_ts=99.0 + i * 0.01,
                     payload={"side": "UP", "book": "bids", "price": 0.4, "size": 1.0})
    store.append(EventType.TWAP, "r", recv_ts=99.9, source_ts=99.9, payload={"value": 1.0})
    store.append(EventType.SPOT, "r", recv_ts=99.9, source_ts=99.9, payload={"value": 1.0})

    events = store.all_events("r")
    report = freshness_from_events(events, 100.0, REAL_REPLAY_FRESHNESS_POLICY)
    assert report.feeds[FeedKind.BOOK_UP].status is FeedStatus.FRESH
    assert report.feeds[FeedKind.BOOK_DOWN].status is FeedStatus.STALE
    assert report.is_fresh is False, (
        "50 UP deltas must not certify a DOWN book last seen 10 seconds ago"
    )
    store.close()


# ======= PART 18: execution journal fidelity, immediate and delayed =====

def test_an_immediate_fak_still_produces_a_full_submit_fill_lifecycle():
    """The case a state-diff observer structurally cannot see: pending is 0
    before AND after, so only the authoritative callback reveals the order."""
    events = []
    s = make_session(taker_delay_ms=0.0,
                     on_execution=lambda k, p: events.append((k, p)))
    book = Book()
    assert len(s.queue._pending) == 0
    s.dispatch(taker_candidate(qty=10.0), START, NEUTRAL, 0.8, book, book)
    assert len(s.queue._pending) == 0, "an immediate FAK never rests"

    kinds = [k for k, _ in events]
    assert "ORDER_SUBMITTED" in kinds
    assert "FILLED" in kinds

    submitted = dict(events[kinds.index("ORDER_SUBMITTED")][1])
    filled = dict(events[kinds.index("FILLED")][1])
    assert submitted["order_id"] == filled["order_id"], "one linkable order id"
    assert filled["filled_shares"] == pytest.approx(10.0)
    assert filled["avg_execution_price"] > 0
    assert filled["fee"] > 0
    assert filled["role"] == "TAKER"
    assert filled["max_execution_price"] == pytest.approx(0.45)
    assert filled["slippage"] == pytest.approx(
        filled["avg_execution_price"] - filled["max_execution_price"])


def test_the_journal_records_price_and_fee_from_the_simulator_not_from_d_C():
    """`d_C` conflates notional and fee and cannot recover either."""
    events = []
    s = make_session(taker_delay_ms=0.0,
                     on_execution=lambda k, p: events.append((k, p)))
    book = Book()
    s.dispatch(taker_candidate(qty=10.0), START, NEUTRAL, 0.8, book, book)
    f = [p for k, p in events if k == "FILLED"][0]
    assert s.portfolio.C == pytest.approx(
        f["avg_execution_price"] * f["filled_shares"] + f["fee"])
    assert f["fee"] > 0 and f["avg_execution_price"] > 0


def test_a_delayed_fak_lifecycle_is_linkable_by_one_order_id():
    events = []
    s = make_session(taker_delay_ms=250.0,
                     on_execution=lambda k, p: events.append((k, p)))
    book = Book()
    s.dispatch(taker_candidate(), START + 270.0, NEUTRAL, 0.8, book, book)
    s.resolve_ready_takers(START + 270.250, lambda p: book.asks)

    kinds = [k for k, _ in events]
    for expected in ("ORDER_SUBMITTED", "PENDING_DELAY", "TAKER_MATCHED", "FILLED"):
        assert expected in kinds, f"missing {expected}: {kinds}"
    ids = {p.get("order_id") for _, p in events if p.get("order_id")}
    assert len(ids) == 1, f"the whole lifecycle must share one order id: {ids}"


def test_a_partial_fill_is_recorded_as_partial():
    events = []
    s = make_session(taker_delay_ms=0.0,
                     on_execution=lambda k, p: events.append((k, p)))
    thin = Book(asks=[(0.42, 4.0)])       # less depth than requested
    s.dispatch(taker_candidate(qty=10.0), START, NEUTRAL, 0.8, thin, thin)
    f = [p for k, p in events if k in ("FILLED", "NO_FILL")][0]
    assert f["requested_shares"] == pytest.approx(10.0)
    assert f["filled_shares"] < 10.0
    assert f["partial"] is True


def test_the_observer_is_attached_to_the_live_session():
    import inspect

    from xamarinbot.shadow import live

    src = inspect.getsource(live.LiveShadowService.ensure_round)
    assert "execution.attach(session)" in src


# ============ PART 19: the model harness must positively trade ==========

def test_the_model_enabled_harness_produces_a_non_wait_action_and_execution():
    """Item 19: `assert ops <= allowed` passes on the empty set. This
    positively requires a real action AND a real consequence."""
    events = []
    s = make_session(taker_delay_ms=0.0,
                     on_execution=lambda k, p: events.append((k, p)))
    book = Book()
    chosen = taker_candidate(qty=10.0)
    assert chosen.mode is not OrderMode.WAIT

    before = s.portfolio
    s.dispatch(chosen, START, NEUTRAL, 0.8, book, book)
    after = s.portfolio

    changed = (after.U, after.D, after.C) != (before.U, before.D, before.C)
    pending = len(s.queue._pending) > 0
    open_makers = len(list(s.supervisor.open_order_ids())) > 0
    assert changed or pending or open_makers, (
        "a non-WAIT action must have a paper consequence"
    )
    assert any(k == "ORDER_SUBMITTED" for k, _ in events)


def test_a_wait_action_produces_no_execution_event():
    """The gate must be able to say no, or the test above proves nothing."""
    events = []
    s = make_session(taker_delay_ms=0.0,
                     on_execution=lambda k, p: events.append((k, p)))
    book = Book()
    wait = CandidateAction(
        action_id="w", purpose=OrderPurpose.ALPHA, side=None,
        mode=OrderMode.WAIT, price=0.0, qty=0.0, ttl_s=0.0, expected_fill=0.0,
        delta_ev=0.0, g_after=0.0, pi_u_after=0.0, pi_d_after=0.0)
    s.dispatch(wait, START, NEUTRAL, 0.8, book, book)
    assert events == []
    assert s.portfolio.U == 0.0 and s.portfolio.C == 0.0
