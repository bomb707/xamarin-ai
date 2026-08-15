"""Phase 12C.1 items 10 and 16: no fabricated numbers on the real path.

Two different fabrications are removed here, and they fail in the same way -
by producing a plausible number where the honest answer is "unknown":

  item 10  `q = 0.5` when no probability model exists, i.e. turning "we have
           no estimate" into "the true probability is exactly 50%".
  item 16  `draw_maker_fill()`, an uncalibrated Bernoulli, reporting a
           synthetic outcome as though it were a real maker fill.
"""
from __future__ import annotations

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.order_state import OrderState
from xamarinbot.execution.simulator import ExecutionSimulator, SyntheticExecutionRefused
from xamarinbot.features.config import FeatureConfig
from xamarinbot.market.constraints import MarketConstraints
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, Side
from xamarinbot.provenance import DataProvenance
from xamarinbot.shadow.config import ShadowConfig
from xamarinbot.shadow.runner import ShadowRunner
from xamarinbot.shadow.types import DecisionBlockReason

from devtools.synthetic.rounds import populate_synthetic_round


def _round(provenance: DataProvenance) -> tuple[EventStore, str, float]:
    """One generated round, written into a store LABELLED as `provenance`.

    The events are the same either way - what differs is only what the store
    claims to be, which is exactly the axis these tests exercise.
    """
    store = EventStore(":memory:", provenance=provenance)
    label = populate_synthetic_round(store, "r1", start_ts=0.0, round_length_s=60.0)
    return store, label.round_id, label.p0


def _run(store: EventStore, round_id: str, p0: float):
    return ShadowRunner(
        store, round_id, p0, FeatureConfig(), FeeConfig(), ExecutionConfig(),
        OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0),
        model=None, feature_set=None, cfg=ShadowConfig(),
    ).run()


# ------------------------------------------------------------- item 10

def test_real_replay_without_a_model_produces_no_alpha():
    """`model=None` on real data must block the decision, not invent q=0.5."""
    store, rid, p0 = _round(DataProvenance.REAL_REPLAY)
    result = _run(store, rid, p0)

    assert result.provenance is DataProvenance.REAL_REPLAY
    assert result.n_model_unavailable > 0
    blocked = [r for r in result.records
               if r.blocked_reason is DecisionBlockReason.MODEL_UNAVAILABLE]
    assert blocked, "a modelless real run must record MODEL_UNAVAILABLE"
    assert all(r.action_id == "wait" and r.qty == 0.0 for r in blocked)
    # nothing was dispatched, so the portfolio never moved
    assert result.final_portfolio.U == 0.0 and result.final_portfolio.D == 0.0
    store.close()


def test_synthetic_run_may_still_use_the_q_fallback():
    """The fallback survives for SYNTHETIC_TEST, whose whole purpose is to
    exercise the pipeline - never to produce a number anyone acts on."""
    store, rid, p0 = _round(DataProvenance.SYNTHETIC_TEST)
    result = _run(store, rid, p0)
    assert result.provenance is DataProvenance.SYNTHETIC_TEST
    assert result.n_model_unavailable == 0
    assert not any(r.blocked_reason is DecisionBlockReason.MODEL_UNAVAILABLE
                   for r in result.records)
    store.close()


def test_block_reasons_distinguish_why_a_decision_produced_nothing():
    """A blocked decision must not be indistinguishable from a WAIT the
    optimizer chose on the economics."""
    assert {r.value for r in DecisionBlockReason} == {
        "MODEL_UNAVAILABLE", "INVALID_FEATURES", "FEED_STALE", "FEED_DISCONNECTED",
    }


def test_shadow_result_carries_its_provenance():
    store, rid, p0 = _round(DataProvenance.SYNTHETIC_TEST)
    assert _run(store, rid, p0).provenance is DataProvenance.SYNTHETIC_TEST
    store.close()


# ------------------------------------------------------------- item 16

def _maker_order() -> OrderState:
    return OrderState(
        order_id="o1", side=Side.UP, role=LiquidityRole.MAKER,
        limit_price=0.40, requested_shares=10.0, submit_ts=0.0,
    )


def test_draw_maker_fill_refuses_real_data():
    """A REAL_REPLAY evaluation must not call the synthetic Bernoulli and
    report the outcome as a real maker fill."""
    sim = ExecutionSimulator("r1", FeeConfig(), ExecutionConfig())
    for p in (DataProvenance.REAL_REPLAY, DataProvenance.REAL_LIVE):
        with pytest.raises(SyntheticExecutionRefused, match="counterfactual"):
            sim.draw_maker_fill(_maker_order(), 0.0, 0.0, 10.0, provenance=p)


def test_draw_maker_fill_still_works_for_simulation():
    sim = ExecutionSimulator("r1", FeeConfig(), ExecutionConfig())
    draw = sim.draw_maker_fill(_maker_order(), 1.0, 10.0, 10.0)
    assert 0.0 <= draw.fill_probability <= 1.0
    assert isinstance(draw.filled, bool)


def test_real_session_expires_a_maker_unresolved_instead_of_drawing_a_fill():
    """On real data the outcome of a resting quote is genuinely unknown until
    a fill model is calibrated, so the session records it as unresolved
    rather than fabricating a fill."""
    from xamarinbot.execution.session import TradingSession
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.regime.types import Direction, GapRegime, RegimeState
    from xamarinbot.supervisor.types import TrackedOrder

    real = MarketConstraints.for_testing()
    real = type(real)(**{**real.__dict__, "provenance": DataProvenance.REAL_REPLAY})
    session = TradingSession(
        "r1", FeeConfig(), ExecutionConfig(),
        OneStepConfig(g_min=-1000.0, spend_cap=1e6, position_limit=1e6), real,
    )
    state = RegimeState(GapRegime.UPPER_MIDDLE, Direction.UP, Direction.UP)
    order = session.sim.submit_maker_order("o1", Side.UP, 10.0, 0.40, 0.0)
    session.supervisor.register(TrackedOrder(
        order, state, OrderPurpose.ALPHA, 0.6, 0.6, 0.0, 1.0, 5.0, 0.0, 0.0,
    ))

    from xamarinbot.feeds.base import BookLevel, BookSnapshot

    def book(side):
        return BookSnapshot(side=side, bids=(BookLevel(0.44, 100.0),),
                            asks=(BookLevel(0.46, 100.0),), ts=0.0, recv_ts=0.0)

    # decision_ts past the 5s TTL -> expiry path
    session.review_open_orders(100.0, state, 0.6, book(Side.UP), book(Side.DOWN), 10.0, True)

    assert session.n_maker_expired_unresolved == 1
    assert session.n_maker_expired_filled == 0
    assert session.portfolio.U == 0.0, "no fabricated fill may reach the portfolio"


def test_synthetic_session_still_draws_a_fill_at_expiry():
    from xamarinbot.execution.session import TradingSession
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.regime.types import Direction, GapRegime, RegimeState
    from xamarinbot.supervisor.types import TrackedOrder
    from xamarinbot.feeds.base import BookLevel, BookSnapshot

    session = TradingSession(
        "r1", FeeConfig(), ExecutionConfig(),
        OneStepConfig(g_min=-1000.0, spend_cap=1e6, position_limit=1e6),
        MarketConstraints.for_testing(),
    )
    state = RegimeState(GapRegime.UPPER_MIDDLE, Direction.UP, Direction.UP)
    order = session.sim.submit_maker_order("o1", Side.UP, 10.0, 0.40, 0.0)
    session.supervisor.register(TrackedOrder(
        order, state, OrderPurpose.ALPHA, 0.6, 0.6, 0.0, 1.0, 5.0, 0.0, 0.0,
    ))

    def book(side):
        return BookSnapshot(side=side, bids=(BookLevel(0.44, 100.0),),
                            asks=(BookLevel(0.46, 100.0),), ts=0.0, recv_ts=0.0)

    session.review_open_orders(100.0, state, 0.6, book(Side.UP), book(Side.DOWN), 10.0, True)
    assert session.n_maker_expired_unresolved == 0
    assert session.n_maker_expired_filled + session.n_maker_expired_unfilled == 1
