"""Phase 12 verification: 24/7 reconnect stability, no timestamp leakage
(the recv_ts vs event_time causal gate), controller deadlines met (SS21
Latency gate fallback), and live-vs-replay parity (Roadmap Phase 12
verification, named explicitly)."""
from __future__ import annotations

import math

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.replay.feeds import ReplayCursor, ReplayTWAPFeed
from xamarinbot.model.calibrated import fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG
from xamarinbot.model.walkforward import round_ordered_split
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.shadow.config import ShadowConfig
from xamarinbot.shadow.parity import compare_live_vs_replay
from xamarinbot.shadow.runner import FaultInjector, ShadowRunner
from devtools.synthetic.rounds import generate_synthetic_dataset

HEARTBEAT_S = 10.0


# --------------------------------------------------------------------------
# time_attr="recv_ts": no timestamp leakage relative to true wire-arrival
# --------------------------------------------------------------------------


def _store_with_source_recv_gap() -> EventStore:
    store = EventStore(":memory:")
    # source_ts=2.0 but doesn't actually arrive (recv_ts) until t=5.0 - a
    # live system must not act on it before t=5.0, even though its source
    # timestamp has already passed at t=3.0.
    store.append(EventType.SPOT, "r1", recv_ts=5.0, source_ts=2.0, payload={"value": 999.0, "provider": "test"})
    store.append(EventType.SPOT, "r1", recv_ts=1.0, source_ts=1.0, payload={"value": 100.0, "provider": "test"})
    return store


def test_recv_ts_gated_cursor_does_not_leak_future_arriving_event():
    store = _store_with_source_recv_gap()
    live_cursor = ReplayCursor(store, "r1", time_attr="recv_ts")
    live_cursor.advance_to(3.0)
    feed = ReplayTWAPFeed  # not used - SPOT via generic events_up_to instead
    visible = live_cursor.events_up_to(EventType.SPOT)
    assert len(visible) == 1
    assert visible[0].payload["value"] == 100.0  # the recv_ts=5.0 event must not leak at t=3.0


def test_recv_ts_gated_cursor_exposes_event_once_actually_received():
    store = _store_with_source_recv_gap()
    live_cursor = ReplayCursor(store, "r1", time_attr="recv_ts")
    live_cursor.advance_to(5.0)
    visible = live_cursor.events_up_to(EventType.SPOT)
    assert len(visible) == 2


def test_default_event_time_gated_cursor_is_optimistic_relative_to_recv_ts():
    """Documents the exact gap Phase 12 exists to catch: the default
    (Phase 2) cursor considers the event visible as soon as its *source*
    timestamp passes, even though it hasn't actually arrived yet."""
    store = _store_with_source_recv_gap()
    offline_cursor = ReplayCursor(store, "r1")  # default time_attr="event_time"
    offline_cursor.advance_to(3.0)
    visible = offline_cursor.events_up_to(EventType.SPOT)
    assert len(visible) == 2  # the recv_ts=5.0 event IS visible offline at t=3.0 (event_time=2.0 <= 3.0)


# --------------------------------------------------------------------------
# Phase 12B Tranche 2A item 5: exchange execution time (event_time/source_ts)
# vs. local strategy information time (recv_ts) - two genuinely different
# clocks for a delayed taker order's resolution.
# --------------------------------------------------------------------------


def test_exchange_execution_book_uses_event_time_not_recv_ts_for_delayed_resolution():
    """A book change with source_ts < matched_ts but recv_ts > matched_ts
    must be visible to an event_time-gated cursor (the real exchange's
    own book at matched_ts) even though a recv_ts-gated cursor (this
    system's own wire-arrival view) does not see it yet."""
    store = EventStore(":memory:")
    matched_ts = 10.25
    store.append(
        EventType.BOOK_DELTA, "r1", recv_ts=matched_ts + 1.0, source_ts=matched_ts - 0.05,
        payload={"side": "UP", "book": "asks", "price": 0.60, "size": 100.0},
    )

    execution_truth_cursor = ReplayCursor(store, "r1")  # default time_attr="event_time"
    execution_truth_cursor.advance_to(matched_ts)
    assert len(execution_truth_cursor.events_up_to(EventType.BOOK_DELTA)) == 1  # exchange truth: already happened

    strategy_view_cursor = ReplayCursor(store, "r1", time_attr="recv_ts")
    strategy_view_cursor.advance_to(matched_ts)
    assert len(strategy_view_cursor.events_up_to(EventType.BOOK_DELTA)) == 0  # this system: hasn't arrived yet


def test_shadow_runner_delayed_resolution_uses_exchange_event_time_not_recv_ts():
    """End-to-end proof through ShadowRunner itself: inject a BOOK_DELTA
    with source_ts just before matched_ts but recv_ts well after it, force
    a delayed FAK decision, and confirm the resolved fill reflects the
    injected (exchange-truth) price - not the round's original book, and
    not the strategy's own recv_ts-gated view (which never even sees this
    delta before the round replay moves on)."""
    from unittest.mock import patch

    from xamarinbot.optimizer.controller import OneStepController, OneStepDecision
    from xamarinbot.optimizer.types import CandidateAction, OrderMode
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.portfolio.state import Side

    import xamarinbot.execution.simulator as simulator_mod

    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=1)
    r = results[0]
    feature_cfg = FeatureConfig()
    fee_config = FeeConfig()
    exec_cfg = ExecutionConfig(taker_delay_ms=250.0)
    permissive_cfg = OneStepConfig(g_min=-1_000_000.0, spend_cap=None, position_limit=None, edge_min=-1_000_000.0)

    forced = CandidateAction(
        action_id="forced_taker_up", purpose=OrderPurpose.ALPHA, side=Side.UP, mode=OrderMode.FAK,
        price=0.99, qty=5.0, ttl_s=0.0, expected_fill=5.0, delta_ev=1.0, g_after=0.0,
        pi_u_after=0.0, pi_d_after=0.0, violated_constraints=(), max_execution_price=0.99,
    )
    already_submitted = {"flag": False}

    def forced_decide(self, round_id, decision_ts, portfolio, q, permitted_actions, book_up, book_down, tick_size, is_fresh, tau=None, sigma=0.0, risk_view=None):
        # Submit exactly one delayed taker order (at the first decision
        # point real feature data is available) so there is exactly one
        # fill to check the resolution price of - every later decision
        # point just WAITs.
        if already_submitted["flag"]:
            wait = CandidateAction(
                action_id="wait", purpose=OrderPurpose.ALPHA, side=None, mode=OrderMode.WAIT, price=None,
                qty=0.0, ttl_s=None, expected_fill=0.0, delta_ev=0.0, g_after=portfolio.G,
                pi_u_after=portfolio.Pi_U, pi_d_after=portfolio.Pi_D, violated_constraints=(),
            )
            return OneStepDecision(round_id, decision_ts, wait, (wait,), skip_reason=None)
        already_submitted["flag"] = True
        return OneStepDecision(round_id, decision_ts, forced, (forced,), skip_reason=None)

    def _run() -> "ShadowRoundResult":
        with patch.object(OneStepController, "decide", forced_decide):
            runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, permissive_cfg, None, None, ShadowConfig())
            return runner.run()

    # Phase 1 (dry run): discover the REAL matched_ts of the one order
    # that gets submitted - depends on exactly which decision point has
    # valid feature data first, which this test must not hardcode/guess.
    captured: dict = {}
    real_submit_taker = simulator_mod.ExecutionSimulator.submit_taker

    def spy_submit_taker(self, *a, **kw):
        pending = real_submit_taker(self, *a, **kw)
        if "matched_ts" not in captured and pending.was_delayed:
            captured["matched_ts"] = pending.matched_ts
        return pending

    with patch.object(simulator_mod.ExecutionSimulator, "submit_taker", spy_submit_taker):
        _run()
    assert "matched_ts" in captured, "expected the forced FAK decision to submit exactly one delayed order"
    matched_ts = captured["matched_ts"]

    # Phase 2: inject the exchange-truth book change - genuinely happened
    # at the exchange just before matched_ts, but doesn't arrive over this
    # system's own wire until well after it - then rerun for real.
    store.append(
        EventType.BOOK_DELTA, r.round_id, recv_ts=matched_ts + 5.0, source_ts=matched_ts - 0.01,
        payload={"side": "UP", "book": "asks", "price": 0.10, "size": 1000.0},
    )
    already_submitted["flag"] = False
    result = _run()

    assert math.isclose(result.final_portfolio.U, 5.0)
    expected_cost = 5.0 * 0.10 + fee_config.taker_fee(5.0, 0.10)
    assert math.isclose(result.final_portfolio.C, expected_cost, rel_tol=1e-6), (
        "resolved fill must reflect the injected exchange-truth (event_time-gated) book, not the original/recv_ts-gated one"
    )


# --------------------------------------------------------------------------
# ShadowRunner - shared fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_cfg():
    return FeatureConfig()


@pytest.fixture(scope="module")
def fee_config():
    return FeeConfig()


@pytest.fixture(scope="module")
def exec_cfg():
    return ExecutionConfig()


@pytest.fixture(scope="module")
def one_step_cfg():
    return OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0, edge_min=0.0)


@pytest.fixture(scope="module")
def trained_model(feature_cfg):
    # Phase 12B audit item 5/C: calibrated, not raw.
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=8)
    by_fs = build_examples_multi(store, results, feature_cfg, [COMBINED_LEAD_LAG], heartbeat_s=HEARTBEAT_S)
    examples = by_fs[COMBINED_LEAD_LAG.name]
    split = round_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    return fit_calibrated_model(split.train, split.validation, COMBINED_LEAD_LAG)


@pytest.fixture(scope="module")
def eval_dataset():
    store = EventStore(":memory:")
    # id_offset=8: disjoint from trained_model's training rounds [0, 8)
    # (Phase 12B audit Addendum A).
    results = generate_synthetic_dataset(store, n_rounds=3, id_offset=8)
    return store, results


# --------------------------------------------------------------------------
# ShadowRunner behavior
# --------------------------------------------------------------------------


def test_shadow_runner_produces_decisions_and_never_crashes(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    store, results = eval_dataset
    r = results[0]
    runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
    result = runner.run()
    assert len(result.records) > 0
    assert result.n_reconnects == 0
    for rec in result.records:
        assert math.isfinite(rec.delta_ev)
        assert math.isfinite(rec.g_after)
        assert rec.decide_elapsed_ms >= 0.0


def test_shadow_runner_24_7_reconnect_stability_survives_simulated_outage(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    """"24/7 reconnect stability" (Roadmap Phase 12 verification): inject
    a simulated feed outage mid-round and confirm the runner completes the
    round cleanly - no crash, no wedge, and decisions resume normally
    after the outage - rather than merely assuming reconnect() works."""
    store, results = eval_dataset
    r = results[0]
    baseline_runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
    baseline = baseline_runner.run()
    all_ts = [rec.decision_ts for rec in baseline.records]
    assert len(all_ts) > 10
    disconnect_ts = frozenset(all_ts[len(all_ts) // 2 : len(all_ts) // 2 + 3])

    fault = FaultInjector(disconnect_at=disconnect_ts)
    runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig(), fault=fault)
    result = runner.run()

    assert result.n_reconnects == len(disconnect_ts)
    recorded_ts = {rec.decision_ts for rec in result.records}

    # Phase 12C item 10: an outage is RECORDED, not silently skipped - "a
    # disconnect must not simply skip the decision while pretending existing
    # resting orders remain safely supervised". Each disconnected decision
    # point produces an explicit not-fresh, suppressed record placing no
    # order, rather than vanishing from the decision stream.
    assert disconnect_ts <= recorded_ts
    outage_records = [rec for rec in result.records if rec.decision_ts in disconnect_ts]
    assert len(outage_records) == len(disconnect_ts)
    assert all(not rec.is_fresh for rec in outage_records)
    assert all(rec.suppressed_by_freshness for rec in outage_records)
    assert all(rec.action_id == "wait" and rec.qty == 0.0 for rec in outage_records)

    # decisions resume after the outage - not wedged, not truncated
    later_ts = [t for t in all_ts if t > max(disconnect_ts)]
    assert any(t in recorded_ts for t in later_ts)
    # the first decision made on a RESTORED feed is flagged reconnected=True
    # (the outage records themselves are the outage, not the recovery)
    restored = [
        rec for rec in sorted(result.records, key=lambda x: x.decision_ts)
        if rec.decision_ts > max(disconnect_ts) and not rec.suppressed_by_freshness
    ]
    assert restored, "expected at least one normal decision after the outage"
    assert restored[0].reconnected is True


def test_shadow_runner_resolves_delayed_taker_orders_via_matched_ts_not_submit_ts(eval_dataset, feature_cfg, fee_config, one_step_cfg, trained_model):
    """Phase 12B Tranche 1.2 items 1/12: ShadowRunner previously still
    misused delayed results (called execute_taker and immediately
    consumed .walk regardless of delay). Proven by spying on
    ExecutionSimulator.resolve_taker and asserting every call only ever
    happens at or after that order's own matched_ts.

    This dataset/model combination's regime classifier only ever permits
    WAIT under ShadowRunner's strict recv_ts gate (a known, pre-existing,
    documented artifact of this synthetic generator - see
    test_recv_ts_gate_collapses_z_spot_when_canonical_horizon_matches_tick_spacing
    above), so OneStepController.decide is monkeypatched here to force a
    FAK decision every call - isolating the dispatch/lifecycle wiring
    under test from the emergent (and here, degenerate) regime behavior."""
    from unittest.mock import patch

    import xamarinbot.execution.simulator as simulator_mod
    from xamarinbot.optimizer.controller import OneStepController, OneStepDecision
    from xamarinbot.optimizer.types import CandidateAction, OrderMode
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.portfolio.state import Side

    store, results = eval_dataset
    delayed_exec_cfg = ExecutionConfig(taker_delay_ms=250.0)

    def forced_decide(self, round_id, decision_ts, portfolio, q, permitted_actions, book_up, book_down, tick_size, is_fresh, tau=None, sigma=0.0, risk_view=None):
        forced = CandidateAction(
            action_id="forced_taker_up", purpose=OrderPurpose.ALPHA, side=Side.UP, mode=OrderMode.FAK,
            price=0.99, qty=5.0, ttl_s=0.0, expected_fill=5.0, delta_ev=1.0, g_after=0.0,
            pi_u_after=0.0, pi_d_after=0.0, violated_constraints=(), max_execution_price=0.99,
        )
        return OneStepDecision(round_id, decision_ts, forced, (forced,), skip_reason=None)

    calls = []
    real_resolve = simulator_mod.ExecutionSimulator.resolve_taker

    def spy(self, pending, asks_at_match=None, now_ts=None):
        if not pending.is_resolved and asks_at_match is not None and now_ts is not None:
            calls.append((pending.submit_ts, pending.matched_ts, now_ts))
        return real_resolve(self, pending, asks_at_match, now_ts)

    with patch.object(simulator_mod.ExecutionSimulator, "resolve_taker", spy), \
         patch.object(OneStepController, "decide", forced_decide):
        for r in results:
            runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, delayed_exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
            runner.run()

    assert calls, "expected at least one delayed taker order to be resolved via resolve_taker"
    for submit_ts, matched_ts, now_ts in calls:
        assert matched_ts > submit_ts
        assert now_ts >= matched_ts


# --------------------------------------------------------------------------
# Phase 12B Tranche 2.2 item 3: ShadowRunner is unified onto the same
# shared TradingSession execution/session engine (RiskView-gated dispatch,
# open makers tracked via OrderSupervisor, cancel/replace review) the
# supervisor-enabled walk-forward ablation arm uses - no longer a
# separately-maintained, simplified execution engine (no RiskView, no
# aggregate maker exposure, instant Bernoulli-draw makers).
# --------------------------------------------------------------------------


def _forced_maker_then_wait_decide(forced_maker):
    """Shared helper: a OneStepController.decide patch that returns
    `forced_maker` exactly once, then WAIT forever after - isolating a
    single maker order's lifecycle from the emergent (and often
    WAIT-only) regime behavior on this synthetic dataset."""
    from xamarinbot.optimizer.controller import OneStepDecision
    from xamarinbot.optimizer.types import CandidateAction, OrderMode
    from xamarinbot.portfolio.math import OrderPurpose

    already_submitted = {"flag": False}

    def forced_decide(self, round_id, decision_ts, portfolio, q, permitted_actions, book_up, book_down, tick_size, is_fresh, tau=None, sigma=0.0, risk_view=None):
        if already_submitted["flag"]:
            wait = CandidateAction(
                action_id="wait", purpose=OrderPurpose.ALPHA, side=None, mode=OrderMode.WAIT, price=None,
                qty=0.0, ttl_s=None, expected_fill=0.0, delta_ev=0.0, g_after=portfolio.G,
                pi_u_after=portfolio.Pi_U, pi_d_after=portfolio.Pi_D, violated_constraints=(),
            )
            return OneStepDecision(round_id, decision_ts, wait, (wait,), skip_reason=None)
        already_submitted["flag"] = True
        return OneStepDecision(round_id, decision_ts, forced_maker, (forced_maker,), skip_reason=None)

    return forced_decide


def test_shadow_runner_maker_order_is_tracked_as_open_not_instant_filled(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    """A POST_ONLY decision must be registered as an OPEN order (tracked
    by OrderSupervisor, reviewable/cancelable on later decision points)
    rather than resolved via an immediate Bernoulli draw at submission
    time - the exact gap ShadowRunner had before TradingSession
    unification."""
    from unittest.mock import patch

    import xamarinbot.supervisor.supervisor as supervisor_mod
    from xamarinbot.optimizer.controller import OneStepController
    from xamarinbot.optimizer.types import CandidateAction, OrderMode
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.portfolio.state import Side

    store, results = eval_dataset
    r = results[0]

    forced_maker = CandidateAction(
        action_id="forced_maker_up", purpose=OrderPurpose.ALPHA, side=Side.UP, mode=OrderMode.POST_ONLY,
        price=0.40, qty=5.0, ttl_s=600.0, expected_fill=2.5, delta_ev=1.0, g_after=0.0,
        pi_u_after=0.0, pi_d_after=0.0, violated_constraints=(),
    )
    forced_decide = _forced_maker_then_wait_decide(forced_maker)

    calls = {"n": 0}
    real_register = supervisor_mod.OrderSupervisor.register

    def spy_register(self, tracked):
        calls["n"] += 1
        return real_register(self, tracked)

    with patch.object(supervisor_mod.OrderSupervisor, "register", spy_register), \
         patch.object(OneStepController, "decide", forced_decide):
        runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
        runner.run()

    assert calls["n"] > 0, "the maker candidate must be registered as an open order with OrderSupervisor, not instant-filled"


def test_shadow_runner_reviews_open_maker_orders_for_cancel_replace(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    """Once a maker order is open, later decision points must actually
    run it through OrderSupervisor.review_order (the cancel/replace/
    hold policy) - not just leave it resting unexamined until TTL."""
    from unittest.mock import patch

    import xamarinbot.supervisor.supervisor as supervisor_mod
    from xamarinbot.optimizer.controller import OneStepController
    from xamarinbot.optimizer.types import CandidateAction, OrderMode
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.portfolio.state import Side

    store, results = eval_dataset
    r = results[0]

    # A long TTL so the order stays open across several decision points,
    # giving review_order real chances to fire before expiry.
    forced_maker = CandidateAction(
        action_id="forced_maker_up", purpose=OrderPurpose.ALPHA, side=Side.UP, mode=OrderMode.POST_ONLY,
        price=0.40, qty=5.0, ttl_s=10_000.0, expected_fill=2.5, delta_ev=1.0, g_after=0.0,
        pi_u_after=0.0, pi_d_after=0.0, violated_constraints=(),
    )
    forced_decide = _forced_maker_then_wait_decide(forced_maker)

    calls = {"n": 0}
    real_review = supervisor_mod.OrderSupervisor.review_order

    def spy_review(self, *a, **kw):
        calls["n"] += 1
        return real_review(self, *a, **kw)

    with patch.object(supervisor_mod.OrderSupervisor, "review_order", spy_review), \
         patch.object(OneStepController, "decide", forced_decide):
        runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
        runner.run()

    assert calls["n"] > 0, "an open maker order must actually be reviewed (cancel/replace/hold) on later decision points"


def test_shadow_runner_dispatch_is_gated_by_aggregate_risk_view(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    """Every dispatch (taker or maker) must be checked against the
    aggregate RiskView (confirmed + pending takers + open makers) before
    it can mutate state - forcing RiskView.admits() to always return
    False must suppress every registration/fill ShadowRunner would
    otherwise make."""
    from unittest.mock import patch

    import xamarinbot.portfolio.exposure as exposure_mod
    import xamarinbot.supervisor.supervisor as supervisor_mod
    from xamarinbot.optimizer.controller import OneStepController
    from xamarinbot.optimizer.types import CandidateAction, OrderMode
    from xamarinbot.portfolio.math import OrderPurpose
    from xamarinbot.portfolio.state import Side

    store, results = eval_dataset
    r = results[0]

    forced_maker = CandidateAction(
        action_id="forced_maker_up", purpose=OrderPurpose.ALPHA, side=Side.UP, mode=OrderMode.POST_ONLY,
        price=0.40, qty=5.0, ttl_s=600.0, expected_fill=2.5, delta_ev=1.0, g_after=0.0,
        pi_u_after=0.0, pi_d_after=0.0, violated_constraints=(),
    )

    def count_registrations(admits_return: bool) -> int:
        calls = {"n": 0}
        real_register = supervisor_mod.OrderSupervisor.register

        def spy_register(self, tracked):
            calls["n"] += 1
            return real_register(self, tracked)

        # Phase 12C item 1 (test hygiene): build a FRESH forced-decision
        # closure per invocation. `_forced_maker_then_wait_decide` carries
        # a stateful `already_submitted` flag, so reusing one closure
        # across the admits=True and admits=False runs made the second run
        # return WAIT at every decision point - it would have "passed"
        # with zero registrations no matter what RiskView did, testing
        # nothing. Both branches must genuinely attempt the same candidate.
        forced_decide = _forced_maker_then_wait_decide(forced_maker)

        with patch.object(exposure_mod.RiskView, "admits", return_value=admits_return), \
             patch.object(supervisor_mod.OrderSupervisor, "register", spy_register), \
             patch.object(OneStepController, "decide", forced_decide):
            runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
            runner.run()
        return calls["n"]

    n_admitted = count_registrations(True)
    n_blocked = count_registrations(False)
    assert n_admitted > 0, "expected at least one maker registration when admission is forced True"
    assert n_blocked == 0, "RiskView.admits()==False must block every dispatch ShadowRunner makes"


def test_shadow_runner_controller_deadline_missed_falls_back_to_wait(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    """SS21 Latency gate: "Controller misses deadline -> Fallback to
    simpler one-step/WAIT policy." A deadline of 0ms guarantees every
    decision "misses" it (decide() always takes > 0ms), so every recorded
    decision must be forced to WAIT."""
    store, results = eval_dataset
    r = results[0]
    cfg = ShadowConfig(decision_deadline_ms=0.0)
    runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, cfg)
    result = runner.run()
    # Phase 12C item 10: the record stream now also contains decision points
    # suppressed before the controller ever ran (missing/stale inputs), and
    # those cannot "miss a deadline" they never started. The latency gate is
    # asserted over exactly the decisions that DID run the controller.
    ran = [rec for rec in result.records if not rec.suppressed_by_freshness]
    assert len(ran) > 0
    assert result.n_missed_deadlines == len(ran)
    assert all(rec.missed_deadline and rec.action_id == "wait" for rec in ran)
    assert all(rec.action_id == "wait" for rec in result.records)


def test_shadow_runner_generous_deadline_never_misses(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    store, results = eval_dataset
    r = results[0]
    cfg = ShadowConfig(decision_deadline_ms=10_000.0)
    runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, cfg)
    result = runner.run()
    assert result.n_missed_deadlines == 0
    assert all(not rec.missed_deadline for rec in result.records)


# --------------------------------------------------------------------------
# live-vs-replay parity
# --------------------------------------------------------------------------


def test_compare_live_vs_replay_produces_bounded_parity_rate(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    store, results = eval_dataset
    r = results[0]
    runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
    result = runner.run()
    report = compare_live_vs_replay(result.records, store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG)
    assert report.n_compared > 0
    assert 0.0 <= report.parity_rate <= 1.0
    assert report.n_mismatches == len(report.mismatches)


def test_compare_live_vs_replay_skips_missed_deadline_decisions(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    """A decision forced to WAIT by the latency fallback reflects the
    deadline, not the causal-view difference parity is meant to isolate -
    comparing it against offline's real (non-deadline-limited) choice
    would conflate two unrelated effects."""
    store, results = eval_dataset
    r = results[0]
    cfg = ShadowConfig(decision_deadline_ms=0.0)
    runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, cfg)
    result = runner.run()
    report = compare_live_vs_replay(result.records, store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG)
    assert report.n_compared == 0


def test_recv_ts_gate_collapses_z_spot_when_canonical_horizon_matches_tick_spacing(feature_cfg):
    """Pins down the dominant, fully-diagnosed Phase 12 finding (see
    docs/PHASE_STATUS.md): this synthetic generator ticks on whole
    seconds, exactly matching FeatureConfig.canonical_horizon_s (1.0s), so
    under strict recv_ts gating the "now" and "1-horizon-ago" spot lookups
    resolve to the identical prior tick at every whole-second decision
    point (the true "now" tick hasn't arrived yet), making Z_spot's log
    return exactly log(x/x) = 0.0 - not a bug in the gate itself, but a
    real fragility worth guarding against a silent regression either way."""
    from xamarinbot.events.store import EventStore
    from xamarinbot.features.engine import compute
    from xamarinbot.features.types import FeatureVector
    from devtools.synthetic.rounds import generate_synthetic_dataset

    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=1)
    r = results[0]
    events = store.all_events(r.round_id)
    decision_ts = 100.0  # well past round start, an exact whole second

    causal_live = [e for e in events if e.recv_ts <= decision_ts]
    fv_live = compute(causal_live, r.round_id, decision_ts, r.p0, feature_cfg)
    assert isinstance(fv_live, FeatureVector)
    assert fv_live.z_spot == 0.0

    fv_offline = compute(events, r.round_id, decision_ts, r.p0, feature_cfg)
    assert isinstance(fv_offline, FeatureVector)
    assert fv_offline.z_spot != 0.0  # offline's event_time gate does not have this collapse


def test_identical_gate_would_give_perfect_parity_sanity_check(eval_dataset, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model):
    """Sanity check on the comparison mechanism itself, independent of the
    real recv_ts-vs-event_time question: if the "shadow" records are
    fabricated to just echo the offline decision at every timestamp, the
    parity checker must report 100% parity - proves compare_live_vs_replay
    isn't spuriously flagging mismatches."""
    from dataclasses import replace

    store, results = eval_dataset
    r = results[0]
    runner = ShadowRunner(store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG, ShadowConfig())
    result = runner.run()
    from xamarinbot.shadow.parity import _offline_decisions_at

    ts_set = {rec.decision_ts for rec in result.records}
    offline_map = _offline_decisions_at(ts_set, store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG)
    echoed_records = tuple(
        replace(rec, action_id=offline_map[rec.decision_ts]) for rec in result.records if rec.decision_ts in offline_map
    )
    report = compare_live_vs_replay(echoed_records, store, r.round_id, r.p0, feature_cfg, fee_config, exec_cfg, one_step_cfg, trained_model, COMBINED_LEAD_LAG)
    assert report.n_mismatches == 0
    assert report.parity_rate == 1.0
