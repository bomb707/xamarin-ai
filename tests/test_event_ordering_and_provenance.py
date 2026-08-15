"""Phase 12C items 12 and 13.

item 12 - a replacement order records the REPLACEMENT's thesis, never the
          canceled order's.
item 13 - own-order reactions can never precede their own submit, and no
          causal fact is fabricated between two external events that merely
          share a timestamp.
"""
from __future__ import annotations

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import (
    Event,
    EventType,
    causal_sort,
    ordering_ambiguities,
)
from xamarinbot.market.constraints import MarketConstraints

# Phase 12C.1 item 12: executable market parameters are a runtime object read
# from the market, not static config. `for_testing` is explicitly stamped
# SYNTHETIC_TEST and defaults min_order_shares to 1.0, so this file's sizing
# arithmetic is unchanged - but it can no longer be inherited by a live path.
CONSTRAINTS = MarketConstraints.for_testing()


def ev(seq, etype, source_ts, recv_ts=None, **payload):
    return Event(
        sequence=seq, event_type=etype, round_id="r1",
        recv_ts=recv_ts if recv_ts is not None else source_ts,
        source_ts=source_ts, payload=payload,
    )


# ============================== item 13 ==============================

def test_fill_never_sorts_before_its_own_submit_on_a_timestamp_tie():
    """The exact pathology named in item 13: "do not construct an ordering
    that could logically place FILL / CANCEL / ORDER_STATUS before their
    corresponding ORDER_SUBMIT merely because timestamps tie"."""
    events = [
        ev(1, EventType.FILL, 10.0, order_id="o1", size=4),
        ev(2, EventType.ORDER_SUBMIT, 10.0, order_id="o1", size=10),
        ev(3, EventType.CANCEL, 10.0, order_id="o1"),
        ev(4, EventType.ORDER_STATUS, 10.0, order_id="o1", status="OPEN"),
    ]
    ordered = causal_sort(events)
    kinds = [e.event_type for e in ordered]
    assert kinds.index(EventType.ORDER_SUBMIT) < kinds.index(EventType.FILL)
    assert kinds.index(EventType.ORDER_SUBMIT) < kinds.index(EventType.CANCEL)
    assert kinds.index(EventType.ORDER_SUBMIT) < kinds.index(EventType.ORDER_STATUS)


def test_submit_precedence_survives_a_skewed_fill_timestamp():
    """A venue can stamp a fill fractionally BEFORE the submit it answers
    (matching engine vs order gateway clock skew). Ranking alone cannot fix
    that - ordering by timestamp would still emit the fill first - so the
    guarantee is structural, per order_id."""
    events = [
        ev(1, EventType.FILL, 9.5, order_id="o1"),
        ev(2, EventType.ORDER_SUBMIT, 10.0, order_id="o1"),
    ]
    ordered = causal_sort(events)
    assert [e.event_type for e in ordered] == [EventType.ORDER_SUBMIT, EventType.FILL]


def test_repair_is_per_order_and_does_not_disturb_other_orders():
    events = [
        ev(1, EventType.FILL, 9.5, order_id="o1"),
        ev(2, EventType.ORDER_SUBMIT, 9.7, order_id="o2"),
        ev(3, EventType.ORDER_SUBMIT, 10.0, order_id="o1"),
        ev(4, EventType.FILL, 10.5, order_id="o2"),
    ]
    ordered = causal_sort(events)
    ids = [(e.event_type, e.payload["order_id"]) for e in ordered]
    assert ids.index((EventType.ORDER_SUBMIT, "o1")) < ids.index((EventType.FILL, "o1"))
    assert ids.index((EventType.ORDER_SUBMIT, "o2")) < ids.index((EventType.FILL, "o2"))
    assert len(ordered) == 4
    assert {id(e) for e in ordered} == {id(e) for e in events}


def test_a_reaction_with_no_recorded_submit_is_left_alone_not_invented():
    events = [ev(1, EventType.FILL, 10.0, order_id="orphan")]
    assert causal_sort(events) == events


def test_external_events_sharing_a_timestamp_get_no_fabricated_priority():
    """Item 13: "Do not rely on an arbitrary event-type priority to create
    causal facts when two external events have identical timestamps."
    TWAP, SPOT and BOOK_DELTA must share one rank so the tie falls through
    to arrival order rather than to a constant."""
    from xamarinbot.events.types import _TYPE_PRIORITY

    ranks = {
        _TYPE_PRIORITY[t]
        for t in (EventType.TWAP, EventType.SPOT, EventType.BOOK_SNAPSHOT, EventType.BOOK_DELTA)
    }
    assert len(ranks) == 1, "external observations must not be ranked against each other"


def test_tied_external_events_break_on_arrival_order_then_sequence():
    a = ev(1, EventType.BOOK_DELTA, 10.0, recv_ts=10.9)
    b = ev(2, EventType.TWAP, 10.0, recv_ts=10.2)
    ordered = causal_sort([a, b])
    # b arrived first, so b comes first - not "TWAP outranks BOOK_DELTA"
    assert ordered == [b, a]

    c = ev(3, EventType.TWAP, 20.0, recv_ts=20.5)
    d = ev(4, EventType.SPOT, 20.0, recv_ts=20.5)
    # identical source AND arrival time -> insertion sequence, the final
    # always-unique key
    assert causal_sort([d, c]) == [c, d]


def test_ordering_ambiguities_are_reported_rather_than_hidden():
    events = [
        ev(1, EventType.TWAP, 10.0, recv_ts=10.0),
        ev(2, EventType.SPOT, 10.0, recv_ts=10.0),
        ev(3, EventType.BOOK_DELTA, 11.0, recv_ts=11.0),
    ]
    ambiguous = ordering_ambiguities(events)
    assert len(ambiguous) == 1
    assert {e.event_type for e in ambiguous[0]} == {EventType.TWAP, EventType.SPOT}


def test_market_config_precedes_and_settlement_follows_the_round():
    events = [
        ev(3, EventType.SETTLEMENT, 5.0),
        ev(2, EventType.BOOK_SNAPSHOT, 5.0),
        ev(1, EventType.MARKET_CONFIG, 5.0),
    ]
    kinds = [e.event_type for e in causal_sort(events)]
    assert kinds[0] is EventType.MARKET_CONFIG
    assert kinds[-1] is EventType.SETTLEMENT


def test_store_reads_apply_the_causal_ordering():
    store = EventStore(":memory:")
    store.append(EventType.FILL, "r1", recv_ts=10.0, source_ts=10.0, payload={"order_id": "o1"})
    store.append(EventType.ORDER_SUBMIT, "r1", recv_ts=10.0, source_ts=10.0, payload={"order_id": "o1"})
    kinds = [e.event_type for e in store.all_events("r1")]
    assert kinds == [EventType.ORDER_SUBMIT, EventType.FILL]
    store.close()


# ============================== item 12 ==============================

def test_replacement_plan_carries_its_own_thesis():
    from xamarinbot.execution.config import ExecutionConfig
    from xamarinbot.optimizer.candidates import evaluate_replacement_plan
    from xamarinbot.optimizer.config import OneStepConfig
    from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side

    plan = evaluate_replacement_plan(
        Side.UP, remaining_shares=10.0, best_bid=0.44, best_ask=0.46,
        constraints=CONSTRAINTS, offsets_ticks=(0, 1, 2), horizon_s=30.0,
        portfolio=PortfolioState(), q=0.62,
        exec_cfg=ExecutionConfig(), fee_config=FeeConfig(),
        cfg=OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0),
    )
    assert plan is not None
    # the replacement's OWN q and fair value, priced now
    assert plan.q == pytest.approx(0.62)
    assert plan.fair_value == pytest.approx(0.62)   # UP -> q
    # and its own if-filled G, distinct from the marginal risk contribution
    assert isinstance(plan.g_after_if_fill, float)


def test_down_side_fair_value_is_one_minus_q():
    from xamarinbot.execution.config import ExecutionConfig
    from xamarinbot.optimizer.candidates import evaluate_replacement_plan
    from xamarinbot.optimizer.config import OneStepConfig
    from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side

    # q must favour DOWN, or every grid price is negative-EV and no plan
    # is offered at all (which is itself correct behaviour, just not what
    # this test is about).
    plan = evaluate_replacement_plan(
        Side.DOWN, remaining_shares=10.0, best_bid=0.44, best_ask=0.46,
        constraints=CONSTRAINTS, offsets_ticks=(0, 1, 2), horizon_s=30.0,
        portfolio=PortfolioState(), q=0.30,
        exec_cfg=ExecutionConfig(), fee_config=FeeConfig(),
        cfg=OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0),
    )
    assert plan is not None
    assert plan.q == pytest.approx(0.30)
    assert plan.fair_value == pytest.approx(0.70)


def test_replacement_tracked_order_does_not_inherit_the_canceled_orders_thesis():
    """The integration check: drive a REPLACE through TradingSession and
    confirm the new TrackedOrder's provenance fields come from the
    replacement, not from the order that was just torn up."""
    from unittest.mock import patch

    from xamarinbot.execution.config import ExecutionConfig
    from xamarinbot.execution.session import TradingSession
    from xamarinbot.feeds.base import BookLevel, BookSnapshot
    from xamarinbot.optimizer.config import OneStepConfig
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.portfolio.state import FeeConfig, Side
    from xamarinbot.regime.types import Direction, GapRegime, RegimeState
    from xamarinbot.supervisor.supervisor import OrderSupervisor
    from xamarinbot.supervisor.types import (
        CancelReason,
        SupervisorActionType,
        SupervisorDecision,
    )

    cfg = OneStepConfig(g_min=-1000.0, spend_cap=10_000.0, position_limit=10_000.0)
    session = TradingSession("r1", FeeConfig(), ExecutionConfig(), cfg, CONSTRAINTS)
    state = RegimeState(GapRegime.UPPER_MIDDLE, Direction.UP, Direction.UP)

    def book(side, bid, ask):
        return BookSnapshot(side=side, bids=(BookLevel(bid, 500.0),),
                            asks=(BookLevel(ask, 500.0),), ts=0.0, recv_ts=0.0)

    # Place a maker at a STALE thesis: q was 0.30 when it was submitted.
    order_state = session.sim.submit_maker_order("old", Side.UP, 10.0, 0.30, 0.0)
    from xamarinbot.supervisor.types import TrackedOrder

    stale = TrackedOrder(order_state, state, OrderPurpose.ALPHA,
                         q_at_submit=0.30, fair_value_at_submit=0.30,
                         g_after_if_fill_at_submit=-11.11, ev_at_submit=-9.99,
                         ttl_s=600.0, submit_ts=0.0, last_action_ts=0.0)
    session.supervisor.register(stale)

    # Force a REPLACE at a decision point where q has moved to 0.70.
    with patch.object(
        OrderSupervisor, "review_order",
        lambda self, tracked, *a, **k: SupervisorDecision(
            tracked.order_id, SupervisorActionType.REPLACE, CancelReason.EDGE_FAILURE),
    ):
        session.review_open_orders(
            10.0, state, 0.70, book(Side.UP, 0.44, 0.46), book(Side.DOWN, 0.54, 0.56),
            120.0, True,
        )

    new_orders = [t for oid, t in session.supervisor.orders.items() if oid != "old"]
    assert new_orders, "expected the replacement to be registered"
    new = new_orders[0]
    # The replacement's thesis is its own, priced against the CURRENT q...
    assert new.q_at_submit == pytest.approx(0.70)
    assert new.fair_value_at_submit == pytest.approx(0.70)
    # ...and nothing was inherited from the canceled order.
    assert new.fair_value_at_submit != stale.fair_value_at_submit
    assert new.g_after_if_fill_at_submit != stale.g_after_if_fill_at_submit
    assert new.ev_at_submit != stale.ev_at_submit
    assert new.submit_ts == 10.0
    # Every field item 12 enumerates is carried from the replacement's own
    # re-evaluation, including the marginal risk contribution.
    # `review_open_orders` prices the replacement over the ORDER's own TTL
    # (600s here), not the config default - recomputed rather than
    # hardcoded so the assertion tracks the model, not a magic number.
    expected = replacement_plan_expected_delta_g(session, 0.70, horizon_s=600.0)
    assert new.expected_delta_g_at_submit == pytest.approx(expected)
    assert new.expected_delta_g_at_submit != 0.0


def replacement_plan_expected_delta_g(session, q, horizon_s):
    from xamarinbot.optimizer.candidates import evaluate_replacement_plan
    from xamarinbot.portfolio.state import Side

    plan = evaluate_replacement_plan(
        Side.UP, 10.0, 0.44, 0.46, session.constraints,
        session.cfg.maker_price_offsets_ticks, horizon_s,
        session.portfolio, q, session.exec_cfg, session.fee_config, session.cfg,
    )
    return plan.expected_delta_g
