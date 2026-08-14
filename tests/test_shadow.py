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
from xamarinbot.feeds.mock import MockFeedCursor, MockTWAPFeed
from xamarinbot.model.calibrated import fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG
from xamarinbot.model.walkforward import round_ordered_split
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.shadow.config import ShadowConfig
from xamarinbot.shadow.parity import compare_live_vs_replay
from xamarinbot.shadow.runner import FaultInjector, ShadowRunner
from xamarinbot.synthetic.rounds import generate_synthetic_dataset

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
    live_cursor = MockFeedCursor(store, "r1", time_attr="recv_ts")
    live_cursor.advance_to(3.0)
    feed = MockTWAPFeed  # not used - SPOT via generic events_up_to instead
    visible = live_cursor.events_up_to(EventType.SPOT)
    assert len(visible) == 1
    assert visible[0].payload["value"] == 100.0  # the recv_ts=5.0 event must not leak at t=3.0


def test_recv_ts_gated_cursor_exposes_event_once_actually_received():
    store = _store_with_source_recv_gap()
    live_cursor = MockFeedCursor(store, "r1", time_attr="recv_ts")
    live_cursor.advance_to(5.0)
    visible = live_cursor.events_up_to(EventType.SPOT)
    assert len(visible) == 2


def test_default_event_time_gated_cursor_is_optimistic_relative_to_recv_ts():
    """Documents the exact gap Phase 12 exists to catch: the default
    (Phase 2) cursor considers the event visible as soon as its *source*
    timestamp passes, even though it hasn't actually arrived yet."""
    store = _store_with_source_recv_gap()
    offline_cursor = MockFeedCursor(store, "r1")  # default time_attr="event_time"
    offline_cursor.advance_to(3.0)
    visible = offline_cursor.events_up_to(EventType.SPOT)
    assert len(visible) == 2  # the recv_ts=5.0 event IS visible offline at t=3.0 (event_time=2.0 <= 3.0)


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
    # decisions resume after the outage - not wedged, not truncated
    recorded_ts = {rec.decision_ts for rec in result.records}
    assert recorded_ts.isdisjoint(disconnect_ts)
    later_ts = [t for t in all_ts if t > max(disconnect_ts)]
    assert any(t in recorded_ts for t in later_ts)
    # the first decision right after an outage is flagged reconnected=True
    first_after = min(t for t in recorded_ts if t > max(disconnect_ts))
    assert next(rec for rec in result.records if rec.decision_ts == first_after).reconnected is True


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

    def forced_decide(self, round_id, decision_ts, portfolio, q, permitted_actions, book_up, book_down, tick_size, is_fresh):
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
    assert len(result.records) > 0
    assert result.n_missed_deadlines == len(result.records)
    assert all(rec.missed_deadline and rec.action_id == "wait" for rec in result.records)


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
    from xamarinbot.synthetic.rounds import generate_synthetic_dataset

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
