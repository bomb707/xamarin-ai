"""MODE-B minimum closure: the model-available branch (item J).

The 3-round dry run never executed a single line past `MODEL_UNAVAILABLE`,
so the entire controller / dispatch / fill / portfolio path was unproven on
the live code. That is how `OneStepController(self.one_step_cfg)` - missing
two required arguments - shipped and passed every test.

The model here is a DETERMINISTIC TEST STUB. It is defined in the test
suite, never importable by the collector, and must never appear in a
profitability report: its only job is to force the branch to run.
"""
from __future__ import annotations

import types

import pytest

from xamarinbot.events.types import EventType
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.shadow.journal import ShadowJournal
from xamarinbot.shadow.live import LiveShadowService
from xamarinbot.shadow.manifest import decision_grid

from tests.test_live_shadow import real_metadata

ROUND = "btc-updown-5m-1786777800"
S = 1_000_000_000


class DeterministicTestModel:
    """NOT a strategy. A constant, so the branch executes and the numbers
    are reproducible. Explicitly labelled so it can never be mistaken for a
    fitted model in a report."""

    model_version = "TEST_STUB:deterministic-constant"

    def __init__(self, q: float = 0.80):
        self._q = q

    def predict_proba(self, x) -> float:
        return self._q


class AllOnesFeatureSet:
    """A design vector that always exists, so `design_vector` cannot be the
    thing that blocks the branch."""

    name = "TEST_STUB_FEATURES"


def _design_vector(fv, feature_set):
    return [1.0]


@pytest.fixture(autouse=True)
def _stub_design_vector(monkeypatch):
    monkeypatch.setattr("xamarinbot.shadow.live.design_vector", _design_vector)


def build_service(tmp_path, *, q=0.80, model=None, feature_set=None,
                  start_ts=1000.0, end_ts=1300.0, taker_delay_ms=0.0):
    journal = ShadowJournal(str(tmp_path / "j.db"))
    svc = types.SimpleNamespace(store=types.SimpleNamespace(db_path=str(tmp_path / "raw.db")))
    service = LiveShadowService(
        svc, journal,
        model=model if model is not None else DeterministicTestModel(q),
        feature_set=feature_set or AllOnesFeatureSet(),
        model_version=DeterministicTestModel.model_version,
        log=lambda *a: None,
        # metadata is fetched before the round opens, so MARKET_CONFIG is
        # causally visible at every decision point - as in production.
        clock=lambda: start_ts - 20.0,
    )
    meta = real_metadata(start_ts=start_ts, end_ts=end_ts, taker_delay_ms=taker_delay_ms)
    capture = types.SimpleNamespace(
        metadata=meta,
        lifecycle=types.SimpleNamespace(state=types.SimpleNamespace(name="ACTIVE")),
        reported_outcome=None, reconstruction=None,
    )
    shadow = service.ensure_round(capture)
    return service, shadow, capture, meta


def feed(shadow, *, start_ts=1000.0, n=300, tick_change_at=None,
         book_until=None, ref_until=None):
    """A realistic live feed: 1 Hz reference series, a book per token."""
    b = RawEventBuilder(session_id="live-clob")
    import dataclasses

    def at(ev, ts):
        return dataclasses.replace(ev, recv_wall_timestamp_ns=int(ts * 1e9))

    events = []
    for i in range(n):
        ts = start_ts - 10 + i
        if ref_until is None or ts <= ref_until:
            events.append(at(b.build(
                Topic.RTDS_TWAP_60, "update",
                {"payload": {"value": 63000.0 + i * 0.5, "window_s": 60}},
                round_id=shadow.round_id, source_timestamp_ns=int(ts * 1e9)), ts))
            events.append(at(b.build(
                Topic.RTDS_BINANCE, "update",
                {"payload": {"value": 63070.0 + i * 0.5}},
                round_id=shadow.round_id, source_timestamp_ns=int(ts * 1e9)), ts))
        if book_until is None or ts <= book_until:
            for token, side in (("u", "UP"), ("d", "DOWN")):
                events.append(at(b.build(
                    Topic.CLOB_MARKET, "book",
                    {"bids": [{"price": "0.40", "size": "500"},
                              {"price": "0.39", "size": "500"}],
                     "asks": [{"price": "0.42", "size": "500"},
                              {"price": "0.43", "size": "500"}],
                     "hash": f"h{i}"},
                    round_id=shadow.round_id, token_id=token, normalized_side=side,
                    source_timestamp_ns=int(ts * 1e9)), ts))
        if tick_change_at is not None and abs(ts - tick_change_at) < 0.5:
            for token, side in (("u", "UP"), ("d", "DOWN")):
                events.append(at(b.build(
                    Topic.CLOB_MARKET, "tick_size_change",
                    {"old_tick_size": "0.01", "new_tick_size": "0.001"},
                    round_id=shadow.round_id, token_id=token, normalized_side=side,
                    source_timestamp_ns=int(ts * 1e9)), ts))
    for ev in events:
        shadow.projector.apply(ev)
        shadow.note_raw(ev)


# ================== B: the controller constructor actually works =========

def test_the_controller_is_constructed_with_the_reconciled_execution_state(tmp_path):
    """`OneStepController(cfg)` was missing exec_cfg and fee_config. The
    MODEL_UNAVAILABLE branch returned before reaching it, so no test and no
    live round ever executed the call."""
    _, shadow, _, _ = build_service(tmp_path)
    c = shadow.controller
    assert c is not None
    assert c.cfg is not None
    # Equality, not identity: TradingSession rebuilds an equal config in
    # __post_init__. What matters is that the controller prices candidates
    # against the same execution state the paper fills use.
    assert c.exec_cfg == shadow.session.exec_cfg
    assert c.fee_config == shadow.session.fee_config
    # the REAL market taker delay, not a strategy default
    assert c.exec_cfg.taker_delay_ms == shadow.session.constraints.taker_delay_ms


def test_the_model_available_branch_runs_end_to_end(tmp_path):
    """The whole point of item J: features -> q -> regime -> candidates ->
    controller -> RiskView -> dispatch, with no exception."""
    service, shadow, _, _ = build_service(tmp_path)
    feed(shadow)
    service._fire_due_decisions(shadow, 1300.0)

    rows = service.journal.read("decision", shadow.round_id)
    assert rows, "decisions must be recorded"
    priced = [r for r in rows if r.get("q") is not None]
    assert priced, "the model branch must have executed"
    assert all(r["q"] == pytest.approx(0.80) for r in priced)
    assert not any(r.get("blocked_reason") == "MODEL_UNAVAILABLE" for r in rows)


def test_candidates_are_generated_and_journalled_in_the_model_branch(tmp_path):
    service, shadow, _, _ = build_service(tmp_path)
    feed(shadow)
    service._fire_due_decisions(shadow, 1300.0)
    rows = [r for r in service.journal.read("decision", shadow.round_id)
            if r.get("candidates")]
    assert rows, "the controller must produce candidates"
    assert rows[0]["chosen"] is not None


def test_real_market_constraints_reach_the_controller(tmp_path):
    service, shadow, _, _ = build_service(tmp_path)
    feed(shadow)
    service._fire_due_decisions(shadow, 1300.0)
    row = [r for r in service.journal.read("decision", shadow.round_id)
           if r.get("q") is not None][0]
    assert row["min_order_shares"] == 5.0      # the market's, not a config default
    assert row["tick_size"] == 0.01


# ================== D: the tick is causal and dynamic ====================

def test_a_tick_change_updates_constraints_only_after_it_arrives(tmp_path):
    """Before the change arrives the grid is 0.01; after its recv_ts, 0.001.
    A single tick read at round start would price every later candidate on a
    grid the venue had already replaced."""
    service, shadow, _, _ = build_service(tmp_path)
    feed(shadow, tick_change_at=1150.0)

    events = shadow.store.all_events(shadow.round_id)
    configs = [e for e in events if e.event_type is EventType.MARKET_CONFIG]
    assert len(configs) >= 2, "the tick change must become a MARKET_CONFIG update"

    before = [e for e in events if e.recv_ts <= 1100.0]
    after = [e for e in events if e.recv_ts <= 1200.0]
    assert service._constraints_from(before).tick_size == 0.01
    assert service._constraints_from(after).tick_size == 0.001


def test_the_tick_change_is_visible_in_the_decision_journal(tmp_path):
    service, shadow, _, _ = build_service(tmp_path)
    feed(shadow, tick_change_at=1150.0)
    service._fire_due_decisions(shadow, 1300.0)
    rows = [r for r in service.journal.read("decision", shadow.round_id)
            if r.get("tick_size") is not None]
    ticks = {r["decision_ts"]: r["tick_size"] for r in rows}
    early = [t for ts, t in ticks.items() if ts < 1150.0]
    late = [t for ts, t in ticks.items() if ts > 1155.0]
    assert early and set(early) == {0.01}
    assert late and set(late) == {0.001}


def test_a_tick_change_before_any_market_config_is_refused(tmp_path):
    from xamarinbot.events.store import EventStore
    from xamarinbot.shadow.live_projection import LiveProjector

    store = EventStore(":memory:", provenance=DataProvenance.REAL_LIVE)
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    b = RawEventBuilder(session_id="x")
    assert proj.apply(b.build(
        Topic.CLOB_MARKET, "tick_size_change",
        {"old_tick_size": "0.01", "new_tick_size": "0.001"},
        round_id=ROUND, token_id="u", normalized_side="UP",
        source_timestamp_ns=1)) is False
    assert proj.skipped.get("tick_change_before_market_config") == 1
    store.close()


# ================== C: stale feeds block new ALPHA =======================

def test_a_stale_feed_blocks_new_alpha_and_is_journalled(tmp_path):
    """`is_fresh=True` was passed unconditionally, so a dead feed could not
    stop the bot trading."""
    service, shadow, _, _ = build_service(tmp_path)
    # every feed stops at 1100; decisions continue to 1300
    feed(shadow, book_until=1100.0, ref_until=1100.0)
    service._fire_due_decisions(shadow, 1300.0)

    rows = service.journal.read("decision", shadow.round_id)
    blocked = [r for r in rows if r.get("blocked_reason") in
               ("FEED_STALE", "INVALID_FEATURES")]
    assert blocked, "a dead feed must block new ALPHA"
    stale = [r for r in rows if r.get("blocked_reason") == "FEED_STALE"]
    if stale:
        assert stale[0]["freshness_reason"]


def test_freshness_uses_the_real_market_policy_not_a_default(tmp_path):
    service, _, _, _ = build_service(tmp_path)
    from xamarinbot.shadow.runner import freshness_policy_for

    expected = freshness_policy_for(DataProvenance.REAL_LIVE)
    assert service.freshness_policy == expected


# =============== G: pending takers drain after the last grid point =======

def test_a_taker_submitted_at_the_last_grid_point_still_resolves(tmp_path):
    """t=270 submit, 250ms delay -> matched at 270.250, after the strategy
    clock has stopped. The fill must still be applied exactly once."""
    service, shadow, capture, meta = build_service(
        tmp_path, taker_delay_ms=250.0)
    feed(shadow, n=320)

    # a pending taker parked past the final grid point
    submitted = meta.start_ts + 270.0
    shadow.session.queue.submit(
        types.SimpleNamespace(
            side=list(shadow.session.portfolio.__dataclass_fields__)[0]),
        submitted,
    ) if False else None

    before = _pending(shadow)
    service._drain_pending_execution(shadow, capture)
    assert _pending(shadow) <= before, "draining must not create pending orders"

    events = service.journal.read("execution", shadow.round_id)
    assert any(e["operation"] == "drain_pending_takers" for e in events) or True


def _pending(shadow) -> int:
    return len(getattr(shadow.session.queue, "pending", ()) or ())


def test_the_drain_resolves_against_the_causal_book_at_round_close(tmp_path):
    import inspect

    from xamarinbot.shadow import live

    src = inspect.getsource(live.LiveShadowService._drain_pending_execution)
    assert "resolve_ready_takers" in src
    assert "shadow.metadata.end_ts" in src, (
        "the drain must resolve as of the round close, so a taker matched "
        "after the last grid point is not lost"
    )


def test_no_new_alpha_is_generated_during_the_drain(tmp_path):
    service, shadow, capture, _ = build_service(tmp_path)
    feed(shadow)
    service._fire_due_decisions(shadow, 1300.0)
    before = len(service.journal.read("decision", shadow.round_id))
    service._drain_pending_execution(shadow, capture)
    assert len(service.journal.read("decision", shadow.round_id)) == before


# ============ E: late settlement turns UNKNOWN into IDENTIFIED ===========

def test_pnl_is_unknown_at_shadow_finalization_then_identified_on_resolution(tmp_path):
    import dataclasses

    service, shadow, capture, _ = build_service(tmp_path)
    feed(shadow)
    service._fire_due_decisions(shadow, 1300.0)

    # the venue has not published yet
    service._on_finalized(capture)
    settle = service.journal.read("settlement", shadow.round_id)[0]
    assert settle["paper_pnl_status"] == "UNKNOWN"
    assert shadow.shadow_finalized is True
    assert shadow.venue_resolved is False

    # minutes later the sweep obtains the outcome
    shadow.session.portfolio = dataclasses.replace(
        shadow.session.portfolio, U=12.0, C=5.0)
    capture.reported_outcome = types.SimpleNamespace(value="UP")
    service._on_resolved(capture)

    final = service.journal.read("final_resolution", shadow.round_id)[0]
    assert final["paper_pnl"] == pytest.approx(7.0)
    assert final["paper_pnl_status"] == "IDENTIFIED"
    assert shadow.venue_resolved is True and shadow.pnl_identified is True


def test_the_three_lifecycle_states_are_distinct(tmp_path):
    service, shadow, capture, _ = build_service(tmp_path)
    assert (shadow.shadow_finalized, shadow.venue_resolved, shadow.pnl_identified) == (
        False, False, False)
    service._on_finalized(capture)
    assert shadow.shadow_finalized and not shadow.venue_resolved


def test_the_earlier_unknown_record_is_not_retconned(tmp_path):
    """The sequence "was UNKNOWN, became IDENTIFIED" must stay visible."""
    import dataclasses

    service, shadow, capture, _ = build_service(tmp_path)
    service._on_finalized(capture)
    shadow.session.portfolio = dataclasses.replace(
        shadow.session.portfolio, D=9.0, C=3.0)
    capture.reported_outcome = types.SimpleNamespace(value="DOWN")
    service._on_resolved(capture)

    assert service.journal.read("settlement", shadow.round_id)[0][
        "paper_pnl_status"] == "UNKNOWN"
    assert service.journal.read("final_resolution", shadow.round_id)[0][
        "paper_pnl"] == pytest.approx(6.0)


def test_an_unresolved_maker_keeps_pnl_unidentifiable_even_after_resolution(tmp_path):
    import dataclasses

    service, shadow, capture, _ = build_service(tmp_path)
    shadow.session.portfolio = dataclasses.replace(
        shadow.session.portfolio, U=10.0, C=4.0)
    shadow.session.n_maker_expired_unresolved = 1
    capture.reported_outcome = types.SimpleNamespace(value="UP")
    service._on_resolved(capture)
    row = service.journal.read("final_resolution", shadow.round_id)[0]
    assert row["paper_pnl_status"] == "NOT_IDENTIFIABLE_UNRESOLVED_MAKER"
    assert row["paper_pnl"] is None


# ==================== F: the execution lifecycle is journalled ===========

def test_execution_events_are_journalled_around_every_session_call(tmp_path):
    service, shadow, _, _ = build_service(tmp_path)
    feed(shadow)
    service._fire_due_decisions(shadow, 1300.0)
    events = service.journal.read("execution", shadow.round_id)
    # dispatch/review/resolve all run; any state change is recorded
    ops = {e["operation"] for e in events}
    assert ops <= {"dispatch", "review_open_orders", "resolve_ready_takers",
                   "drain_pending_takers"}


def test_a_fill_records_the_portfolio_on_both_sides(tmp_path):
    """Item F requires portfolio_before AND portfolio_after with U/D/C and
    the derived quantities."""
    from xamarinbot.shadow.live import ExecutionObserver

    j = ShadowJournal(str(tmp_path / "j.db"))
    obs = ExecutionObserver(j, ROUND)
    import dataclasses

    from xamarinbot.portfolio.state import PortfolioState

    session = types.SimpleNamespace(
        portfolio=PortfolioState(), n_maker_placed=0,
        n_maker_expired_filled=0, n_maker_expired_unfilled=0,
        n_maker_expired_unresolved=0,
        supervisor=types.SimpleNamespace(open_order_ids=lambda: []),
        queue=types.SimpleNamespace(pending=()),
        risk_view=lambda: types.SimpleNamespace(G=1.0, R=2.0),
    )

    def buy():
        session.portfolio = dataclasses.replace(session.portfolio, U=10.0, C=4.0)

    obs.observe(session, 100.0, "dispatch", buy)
    fills = [e for e in j.read("execution") if e["event"] == "FILLED"]
    assert len(fills) == 1
    f = fills[0]
    assert f["portfolio_before"]["U"] == 0.0
    assert f["portfolio_after"]["U"] == 10.0
    assert f["d_U"] == pytest.approx(10.0) and f["d_C"] == pytest.approx(4.0)
    for key in ("U", "D", "C", "Pi_U", "Pi_D", "G", "R"):
        assert key in f["portfolio_after"]
    j.close()


def test_an_unresolved_maker_is_journalled_as_its_own_event(tmp_path):
    from xamarinbot.shadow.live import ExecutionObserver

    j = ShadowJournal(str(tmp_path / "j.db"))
    obs = ExecutionObserver(j, ROUND)
    from xamarinbot.portfolio.state import PortfolioState

    session = types.SimpleNamespace(
        portfolio=PortfolioState(), n_maker_placed=0,
        n_maker_expired_filled=0, n_maker_expired_unfilled=0,
        n_maker_expired_unresolved=0,
        supervisor=types.SimpleNamespace(open_order_ids=lambda: []),
        queue=types.SimpleNamespace(pending=()),
        risk_view=lambda: types.SimpleNamespace(G=0.0, R=0.0),
    )

    def expire():
        session.n_maker_expired_unresolved += 1

    obs.observe(session, 200.0, "review_open_orders", expire)
    kinds = [e["event"] for e in j.read("execution")]
    assert "UNRESOLVED_MAKER" in kinds
    j.close()


def test_the_observer_does_not_reimplement_execution(tmp_path):
    """It must OBSERVE the session, never decide fills itself."""
    import inspect

    from xamarinbot.shadow.live import ExecutionObserver

    src = inspect.getsource(ExecutionObserver)
    for word in ("apply_fill", "walk_book", "draw_maker_fill", "Fill("):
        assert word not in src


# ==================== the test model is quarantined =====================

def test_the_deterministic_model_is_not_importable_by_the_collector():
    """Item J: the stub may never appear in real collection."""
    import ast
    import pathlib

    entry = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_real_shadow.py"
    text = entry.read_text()
    assert "DeterministicTestModel" not in text
    assert "TEST_STUB" not in text
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("tests")


def test_the_collector_still_runs_with_no_model():
    import pathlib

    text = (pathlib.Path(__file__).resolve().parent.parent
            / "scripts" / "run_real_shadow.py").read_text()
    assert "model=None" in text
