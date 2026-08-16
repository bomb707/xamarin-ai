"""Live real-market shadow service - readiness audit blockers.

These cover the four things the audit found missing on the critical path:

  * live feeds reach the real FeatureEngine WITHOUT a replay round-trip
  * the strategy clock is frozen, not set by market message rate
  * every decision is permanently journalled, including the ones that were
    blocked and every candidate that was rejected
  * the live entrypoint cannot reach an order-submitting code path at all
"""
from __future__ import annotations

import ast
import json
import pathlib
import types

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.shadow.journal import ShadowJournal
from xamarinbot.shadow.live_projection import LiveProjector
from xamarinbot.shadow.manifest import (
    DECISION_GRID_END_S,
    DECISION_GRID_START_S,
    STRATEGY_VERSION,
    build_manifest,
    decision_grid,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUND = "btc-updown-5m-1786777800"
START_NS = 1_786_777_800_000_000_000


# ===================== the frozen strategy clock (item 10) ================

def test_the_strategy_clock_is_frozen_not_market_driven():
    grid = decision_grid()
    assert grid[0] == DECISION_GRID_START_S == 15.0
    assert grid[-1] == DECISION_GRID_END_S == 270.0
    assert all(round(b - a, 6) == 3.0 for a, b in zip(grid, grid[1:]))
    assert len(grid) == 86


def test_the_live_grid_matches_the_training_grid():
    """If the model is fitted on a different schedule from the one the bot
    fires on, it is being asked at decision points that never occur."""
    from xamarinbot.model.real_dataset import decision_grid as training_grid

    assert decision_grid() == training_grid()


def test_no_pre_round_decision_point():
    assert min(decision_grid()) > 0.0


# ========================= Strategy V0 manifest (item 6) =================

def _manifest():
    from xamarinbot.execution.config import ExecutionConfig
    from xamarinbot.features.config import FeatureConfig
    from xamarinbot.optimizer.config import OneStepConfig
    from xamarinbot.regime.config import RegimeConfig

    return build_manifest(FeatureConfig(), OneStepConfig(g_min=0.0), RegimeConfig(),
                          ExecutionConfig(), None, "NONE:test")


def test_the_manifest_pins_every_component_of_the_strategy():
    m = _manifest()
    d = m.as_dict()
    for key in ("strategy_version", "config_hash", "controller", "model_version",
                "feature_set", "feature_config_hash", "one_step_config_hash",
                "regime_config_hash", "execution_config_hash", "maker_enabled",
                "hedge_enabled", "buffer_enabled", "mpc_enabled", "decision_grid_s"):
        assert key in d
    assert d["strategy_version"] == STRATEGY_VERSION


def test_the_config_hash_is_stable_across_runs():
    assert _manifest().config_hash == _manifest().config_hash


def test_changing_a_strategy_parameter_changes_the_hash():
    """The whole point: a silent config change must be visible as a
    different id, so records from before and after are not pooled."""
    import dataclasses

    from xamarinbot.execution.config import ExecutionConfig
    from xamarinbot.features.config import FeatureConfig
    from xamarinbot.optimizer.config import OneStepConfig
    from xamarinbot.regime.config import RegimeConfig

    base = OneStepConfig(g_min=0.0)
    changed = dataclasses.replace(base, g_min=base.g_min + 0.01)
    a = build_manifest(FeatureConfig(), base, RegimeConfig(), ExecutionConfig(),
                       None, "NONE:test")
    b = build_manifest(FeatureConfig(), changed, RegimeConfig(), ExecutionConfig(),
                       None, "NONE:test")
    assert a.config_hash != b.config_hash


def test_the_model_version_is_part_of_the_identity():
    from xamarinbot.execution.config import ExecutionConfig
    from xamarinbot.features.config import FeatureConfig
    from xamarinbot.optimizer.config import OneStepConfig
    from xamarinbot.regime.config import RegimeConfig

    a = build_manifest(FeatureConfig(), OneStepConfig(g_min=0.0), RegimeConfig(),
                       ExecutionConfig(), None, "modelA")
    b = build_manifest(FeatureConfig(), OneStepConfig(g_min=0.0), RegimeConfig(),
                       ExecutionConfig(), None, "modelB")
    assert a.config_hash != b.config_hash


# ================ live projection: same contract as replay (item 3) ======

def _live_store() -> EventStore:
    return EventStore(":memory:", provenance=DataProvenance.REAL_LIVE)


def real_metadata(*, start_ts: float = 1.0, end_ts: float = 301.0,
                  taker_delay_ms: float = 250.0, schedule_rate=0.07,
                  twap_window_s=60):
    """The ACTUAL `RealMarketMetadata` dataclass, not a SimpleNamespace.

    A loose stand-in is why `metadata.fee_config` - a field that does not
    exist - passed every test and then failed on the live wire. A fixture
    that accepts any attribute cannot test an attribute contract.
    """
    from xamarinbot.realtime.discovery import FeeConfiguration, RealMarketMetadata

    return RealMarketMetadata(
        round_id=ROUND, market_id="m", event_id="e", condition_id="0xc",
        question_id="q", slug=ROUND, question="Bitcoin Up or Down",
        description="Chainlink 60-second TWAP",
        resolution_source="https://data.chain.link/streams/btc-usd-twap-60s-streams",
        start_ts=start_ts, end_ts=end_ts, up_token_id="u", down_token_id="d",
        tick_size=0.01, min_order_size=5.0,
        fees=FeeConfiguration(
            maker_base_fee=0.0, taker_base_fee=0.0, fees_enabled=True,
            fee_type="schedule", schedule_rate=schedule_rate,
            schedule_exponent=None, schedule_taker_only=None,
            schedule_rebate_rate=None, maker_rebates_fee_share_bps=None,
            raw={"feeSchedule": {"rate": schedule_rate}},
        ),
        taker_delay_ms=taker_delay_ms, settlement_kind="chainlink_twap",
        twap_window_s=twap_window_s, is_executable=True,
        outcome_label_source="outcomes", raw_gamma={}, raw_clob={}, warnings=[],
    )


def test_a_market_with_no_fee_schedule_is_refused_not_defaulted():
    """A defaulted fee silently changes every candidate's economics. The
    offline projection refuses; the live path must refuse identically."""
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    with pytest.raises(ValueError, match="fee"):
        proj.emit_market_config(real_metadata(schedule_rate=None), START_NS)
    store.close()


def test_a_market_with_no_twap_window_is_refused():
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    with pytest.raises(ValueError, match="TWAP window"):
        proj.emit_market_config(real_metadata(twap_window_s=None), START_NS)
    store.close()


def test_market_config_is_built_from_the_real_metadata_dataclass():
    """Guards the exact failure that reached the live wire: an attribute
    that does not exist on `RealMarketMetadata`."""
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    proj.emit_market_config(real_metadata(), START_NS)
    p = store.all_events(ROUND)[0].payload
    assert p["fee_rate"] == pytest.approx(0.07)
    assert p["taker_delay_ms"] == pytest.approx(250.0)
    assert p["min_order_size"] == 5.0
    store.close()


def test_live_projection_refuses_a_synthetic_destination():
    synthetic = EventStore(":memory:", provenance=DataProvenance.SYNTHETIC_TEST)
    with pytest.raises(ValueError, match="synthetic"):
        LiveProjector(synthetic, ROUND, Topic.RTDS_TWAP_60)
    synthetic.close()


def test_a_live_book_delta_projects_with_the_replay_payload_contract():
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    b = RawEventBuilder(session_id="live")
    ev = b.build(
        Topic.CLOB_MARKET, "price_change",
        {"market": "0xc", "timestamp": "1786777800000",
         "price_changes": [{"asset_id": "up", "price": "0.45", "size": "60",
                            "side": "BUY", "hash": "h1"}]},
        round_id=ROUND, token_id="up", normalized_side="UP",
        source_timestamp_ns=START_NS,
    )
    assert proj.apply(ev) is True
    events = store.all_events(ROUND)
    delta = [e for e in events if e.event_type is EventType.BOOK_DELTA]
    assert len(delta) == 1
    p = delta[0].payload
    # the exact keys features/engine.py and replay/feeds.py require
    assert p["side"] == "UP" and p["book"] == "bids"
    assert p["price"] == 0.45 and p["size"] == 60.0
    store.close()


def test_a_sell_price_change_becomes_an_ask():
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    b = RawEventBuilder(session_id="live")
    proj.apply(b.build(
        Topic.CLOB_MARKET, "price_change",
        {"price_changes": [{"asset_id": "d", "price": "0.55", "size": "10",
                            "side": "SELL"}]},
        round_id=ROUND, token_id="d", normalized_side="DOWN",
        source_timestamp_ns=START_NS,
    ))
    p = [e for e in store.all_events(ROUND) if e.event_type is EventType.BOOK_DELTA][0].payload
    assert p["book"] == "asks" and p["side"] == "DOWN"
    store.close()


def test_only_the_declared_basis_becomes_the_twap_series():
    """Mixing TWAP-30 and TWAP-60 into one series would silently corrupt
    `gap_twap_bp` and `z_gap`."""
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    b = RawEventBuilder(session_id="live")
    for topic in (Topic.RTDS_TWAP_60, Topic.RTDS_TWAP_30, Topic.RTDS_CHAINLINK):
        proj.apply(b.build(
            topic, "update", {"payload": {"value": 63000.0, "window_s": 60}},
            round_id=ROUND, source_timestamp_ns=START_NS,
        ))
    twap = [e for e in store.all_events(ROUND) if e.event_type is EventType.TWAP]
    assert len(twap) == 1
    assert proj.skipped.get("non_declared_reference") == 2
    store.close()


def test_binance_becomes_spot_not_twap():
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    b = RawEventBuilder(session_id="live")
    proj.apply(b.build(Topic.RTDS_BINANCE, "update",
                       {"payload": {"value": 63070.0}},
                       round_id=ROUND, source_timestamp_ns=START_NS))
    kinds = {e.event_type for e in store.all_events(ROUND)}
    assert EventType.SPOT in kinds and EventType.TWAP not in kinds
    store.close()


def test_events_with_no_normalized_counterpart_are_counted_not_dropped():
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    b = RawEventBuilder(session_id="live")
    proj.apply(b.build(Topic.CLOB_MARKET, "last_trade_price", {"price": "0.5"},
                       round_id=ROUND, token_id="up", normalized_side="UP",
                       source_timestamp_ns=START_NS))
    assert store.all_events(ROUND) == []
    assert sum(proj.skipped.values()) == 1


def test_the_projection_preserves_both_clocks():
    """`recv_ts` is the live causal gate; `source_ts` is the exchange's."""
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    b = RawEventBuilder(session_id="live")
    import dataclasses

    ev = b.build(Topic.RTDS_TWAP_60, "update", {"payload": {"value": 1.0}},
                 round_id=ROUND, source_timestamp_ns=START_NS)
    ev = dataclasses.replace(ev, recv_wall_timestamp_ns=START_NS + 15_000_000)
    proj.apply(ev)
    e = store.all_events(ROUND)[0]
    assert e.source_ts == pytest.approx(START_NS / 1e9)
    assert e.recv_ts == pytest.approx((START_NS + 15_000_000) / 1e9)
    assert e.recv_ts > e.source_ts
    store.close()


def test_market_config_carries_the_full_replay_contract():
    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    proj.emit_market_config(real_metadata(), START_NS)
    from xamarinbot.replay.feeds import market_config_from_payload

    cfg = market_config_from_payload(store.all_events(ROUND)[0].payload)
    assert cfg.min_order_size == 5.0 and cfg.tick_size == 0.01
    store.close()


def test_live_projection_feeds_the_real_feature_engine_directly(tmp_path):
    """The headline blocker: live feeds -> causal FeatureVector, with NO
    raw capture, no normalized replay DB and no ReplayCursor round-trip."""
    from xamarinbot.features.config import FeatureConfig
    from xamarinbot.features.engine import compute
    from xamarinbot.features.types import FeatureVector

    store = _live_store()
    proj = LiveProjector(store, ROUND, Topic.RTDS_TWAP_60)
    proj.emit_market_config(
        real_metadata(start_ts=1000.0, end_ts=1300.0, taker_delay_ms=0.0),
        int(990 * 1e9),
    )

    b = RawEventBuilder(session_id="live")
    import dataclasses

    def at(ev, ts: float):
        return dataclasses.replace(
            ev, recv_wall_timestamp_ns=int(ts * 1e9) + 10_000_000)

    for i in range(120):
        ts = 990.0 + i
        proj.apply(at(b.build(Topic.RTDS_TWAP_60, "update",
                              {"payload": {"value": 63000.0 + i, "window_s": 60}},
                              round_id=ROUND, source_timestamp_ns=int(ts * 1e9)), ts))
        proj.apply(at(b.build(Topic.RTDS_BINANCE, "update",
                              {"payload": {"value": 63070.0 + i}},
                              round_id=ROUND, source_timestamp_ns=int(ts * 1e9)), ts))
        for token, side in (("u", "UP"), ("d", "DOWN")):
            proj.apply(at(b.build(
                Topic.CLOB_MARKET, "book",
                {"bids": [{"price": "0.44", "size": "100"}],
                 "asks": [{"price": "0.46", "size": "80"}], "hash": f"h{i}"},
                round_id=ROUND, token_id=token, normalized_side=side,
                source_timestamp_ns=int(ts * 1e9)), ts))

    decision_ts = 1015.0            # t = +15s, the first grid point
    events = store.all_events(ROUND)
    causal = [e for e in events if e.recv_ts <= decision_ts]
    fv = compute(causal, ROUND, decision_ts, 63000.0, FeatureConfig())
    assert isinstance(fv, FeatureVector), getattr(fv, "reason", None)
    assert fv.tau == pytest.approx(285.0)
    store.close()


# ================= the shadow journal (items 12 and 13) ==================

def _fake_shadow(round_id=ROUND):
    from xamarinbot.portfolio.state import PortfolioState

    session = types.SimpleNamespace(
        portfolio=PortfolioState(),
        n_maker_expired_unresolved=0,
        supervisor=types.SimpleNamespace(open_order_ids=lambda: []),
    )
    return types.SimpleNamespace(
        round_id=round_id,
        metadata=types.SimpleNamespace(
            condition_id="0xc", up_token_id="u", down_token_id="d",
            start_ts=1000.0, end_ts=1300.0, settlement_kind="chainlink_twap",
            twap_window_s=60, tick_size=0.01, min_order_size=5.0,
            taker_delay_ms=0.0,
        ),
        session=session, p0=63000.0, fired={15.0}, decisions=1,
        blocked={}, missed_deadlines=0,
        projector=types.SimpleNamespace(counts={"TWAP": 3}, skipped={}),
        raw_seq_by_session={"cap-clob": (1, 99), "cap-rtds": (1, 40)},
        raw_recv_first_ns=START_NS, raw_recv_last_ns=START_NS + 300_000_000_000,
    )


def test_every_decision_is_permanently_recorded(tmp_path):
    j = ShadowJournal(str(tmp_path / "j.db"))
    j.write_manifest(_manifest())
    s = _fake_shadow()
    j.write_decision(s, 1015.0, 15.0, None, None, None, None,
                     blocked_reason="MODEL_UNAVAILABLE", manifest=_manifest())
    j.close()

    reopened = ShadowJournal(str(tmp_path / "j.db"))
    rows = reopened.read("decision")
    assert len(rows) == 1, "the record must survive process exit"
    assert rows[0]["blocked_reason"] == "MODEL_UNAVAILABLE"
    reopened.close()


def test_a_blocked_decision_is_recorded_not_skipped(tmp_path):
    """A silently skipped decision point is indistinguishable from a WAIT."""
    j = ShadowJournal(str(tmp_path / "j.db"))
    s = _fake_shadow()
    for reason in ("NO_P0", "INVALID_FEATURES", "MODEL_UNAVAILABLE"):
        j.write_decision(s, 1015.0, 15.0, None, None, None, None,
                         blocked_reason=reason, manifest=_manifest())
    assert [r["blocked_reason"] for r in j.read("decision")] == [
        "NO_P0", "INVALID_FEATURES", "MODEL_UNAVAILABLE"]
    j.close()


def test_every_candidate_is_journalled_not_only_the_winner(tmp_path):
    """Strategy research needs the rejected candidates - "why did it not do
    the other thing" is unanswerable from the chosen action alone."""
    j = ShadowJournal(str(tmp_path / "j.db"))
    s = _fake_shadow()
    decision = types.SimpleNamespace(candidates=[
        {"action_id": "a1", "purpose": "ALPHA", "is_valid": False,
         "violated_constraints": ["MIN_ORDER_SHARES"]},
        {"action_id": "a2", "purpose": "WAIT", "is_valid": True},
    ])
    j.write_decision(s, 1015.0, 15.0, None, 0.55, None, decision,
                     chosen={"action_id": "a2"}, manifest=_manifest())
    row = j.read("decision")[0]
    assert len(row["candidates"]) == 2
    assert row["candidates"][0]["violated_constraints"] == ["MIN_ORDER_SHARES"]
    assert row["chosen"]["action_id"] == "a2"
    j.close()


def test_the_decision_record_carries_strategy_and_model_identity(tmp_path):
    j = ShadowJournal(str(tmp_path / "j.db"))
    m = _manifest()
    j.write_decision(_fake_shadow(), 1015.0, 15.0, None, None, None, None,
                     manifest=m)
    row = j.read("decision")[0]
    assert row["strategy_version"] == STRATEGY_VERSION
    assert row["config_hash"] == m.config_hash
    assert row["model_version"] == "NONE:test"
    j.close()


def test_the_linkage_row_maps_a_decision_back_to_its_raw_capture(tmp_path):
    """Audit item 13: one losing decision weeks later must be traceable to
    the exact capture, event range, code SHA and config."""
    j = ShadowJournal(str(tmp_path / "j.db"))
    j.write_round_opened(_fake_shadow(), _manifest(), "captures/shadow/raw.db")
    row = j.read("round_opened")[0]
    assert row["raw_capture_db"] == "captures/shadow/raw.db"
    assert row["code_sha"] and len(row["code_sha"]) == 40
    assert row["config_hash"] and row["process_pid"]
    j.close()


def test_the_decision_record_carries_a_per_stream_raw_event_range(tmp_path):
    """Each stream's builder owns its own sequence counter - the CLOB and
    RTDS readers both start at 1. A single min/max across them is not a
    range: the first live run reported `123041..6006`, which describes
    nothing. Ranges are per session, plus globally-comparable wall bounds."""
    j = ShadowJournal(str(tmp_path / "j.db"))
    j.write_decision(_fake_shadow(), 1015.0, 15.0, None, None, None, None,
                     manifest=_manifest())
    row = j.read("decision")[0]
    assert row["raw_seq_by_session"] == {"cap-clob": [1, 99], "cap-rtds": [1, 40]}
    assert row["raw_recv_first_ns"] < row["raw_recv_last_ns"]
    j.close()


def test_a_per_session_range_is_never_inverted():
    """The defect the live run exposed, at the source."""
    import types as _t

    from xamarinbot.shadow.live import LiveRoundShadow

    s = LiveRoundShadow(round_id=ROUND, metadata=None, store=None,
                        projector=None, session=None, regime=None)
    for session_id, seq in (("clob", 5), ("clob", 2), ("rtds", 900), ("clob", 9)):
        s.note_raw(_t.SimpleNamespace(
            session_id=session_id, recorder_sequence=seq,
            recv_wall_timestamp_ns=START_NS + seq))
    assert s.raw_seq_by_session == {"clob": (2, 9), "rtds": (900, 900)}
    for lo, hi in s.raw_seq_by_session.values():
        assert lo <= hi
    j = None


def test_settlement_pnl_is_identified_when_no_maker_is_outstanding(tmp_path):
    j = ShadowJournal(str(tmp_path / "j.db"))
    import dataclasses

    s = _fake_shadow()
    s.session.portfolio = dataclasses.replace(s.session.portfolio, U=10.0, C=4.0)
    capture = types.SimpleNamespace(
        reported_outcome=types.SimpleNamespace(value="UP"),
        reconstruction=None,
    )
    j.write_settlement(s, capture, _manifest())
    row = j.read("settlement")[0]
    assert row["paper_pnl"] == pytest.approx(6.0)
    assert row["paper_pnl_status"] == "IDENTIFIED"
    j.close()


def test_an_unresolved_maker_makes_round_pnl_not_identifiable(tmp_path):
    """Audit item 8: with maker fills unknown, PnL must be reported as NOT
    IDENTIFIABLE rather than silently counting the maker as unfilled."""
    j = ShadowJournal(str(tmp_path / "j.db"))
    import dataclasses

    s = _fake_shadow()
    s.session.portfolio = dataclasses.replace(s.session.portfolio, U=10.0, C=4.0)
    s.session.n_maker_expired_unresolved = 1
    capture = types.SimpleNamespace(
        reported_outcome=types.SimpleNamespace(value="UP"), reconstruction=None)
    j.write_settlement(s, capture, _manifest())
    row = j.read("settlement")[0]
    assert row["paper_pnl_status"] == "NOT_IDENTIFIABLE_UNRESOLVED_MAKER"
    assert row["unresolved_makers"] == 1
    j.close()


def test_an_unsettled_round_reports_unknown_not_zero(tmp_path):
    j = ShadowJournal(str(tmp_path / "j.db"))
    capture = types.SimpleNamespace(reported_outcome=None, reconstruction=None)
    j.write_settlement(_fake_shadow(), capture, _manifest())
    row = j.read("settlement")[0]
    assert row["paper_pnl"] is None
    assert row["paper_pnl_status"] == "UNKNOWN"
    j.close()


# ============ item 16: no real order can be sent, structurally ===========

def _transitive_imports(entry: pathlib.Path, seen: set[str] | None = None) -> set[str]:
    """Every `xamarinbot.*` module reachable from an entrypoint."""
    seen = seen if seen is not None else set()
    tree = ast.parse(entry.read_text(), filename=str(entry))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    for mod in sorted(modules):
        if not mod.startswith("xamarinbot") or mod in seen:
            continue
        seen.add(mod)
        path = REPO_ROOT / "src" / pathlib.Path(*mod.split("."))
        for candidate in (path.with_suffix(".py"), path / "__init__.py"):
            if candidate.exists():
                _transitive_imports(candidate, seen)
                break
    return seen


def test_the_live_shadow_entrypoint_cannot_reach_an_order_client():
    """Audit item 16, over the WHOLE transitive import graph - not just the
    entrypoint's own imports."""
    entry = REPO_ROOT / "scripts" / "run_real_shadow.py"
    reachable = _transitive_imports(entry)
    assert reachable, "the entrypoint must import the package at all"

    banned = ("xamarinbot.feeds.polymarket_user",)
    for mod in banned:
        assert mod not in reachable, f"live shadow must not reach {mod}"


def test_no_module_reachable_from_live_shadow_signs_or_submits_orders():
    entry = REPO_ROOT / "scripts" / "run_real_shadow.py"
    banned = ("post_order", "create_order", "sign_order", "private_key",
              "eip712", "ApiCreds", "signing_key")
    offenders = []
    for mod in sorted(_transitive_imports(entry)):
        path = REPO_ROOT / "src" / pathlib.Path(*mod.split("."))
        for candidate in (path.with_suffix(".py"), path / "__init__.py"):
            if not candidate.exists():
                continue
            text = candidate.read_text().lower()
            for word in banned:
                if word.lower() in text:
                    offenders.append(f"{mod}: {word}")
    assert not offenders, (
        "no module reachable from the live shadow entrypoint may reference an "
        "order-signing or order-submitting symbol:\n  " + "\n  ".join(offenders)
    )


def test_the_entrypoint_itself_holds_no_credentials():
    text = (REPO_ROOT / "scripts" / "run_real_shadow.py").read_text().lower()
    for word in ("private_key", "api_key", "secret", "passphrase", "signer"):
        assert word not in text, f"live shadow entrypoint references {word!r}"


def test_the_live_service_dispatches_only_into_the_paper_session():
    """The only 'dispatch' must mutate TradingSession, never a venue."""
    import inspect

    from xamarinbot.shadow import live

    src = inspect.getsource(live.LiveShadowService)
    assert "session.dispatch(" in src
    # substring checks would match `design_vector`; these are the real ones
    for word in ("requests.post", "httpx.post", "post_order", "sign_order",
                 "private_key", "eip712"):
        assert word not in src


# ============ thread safety: readers buffer, decisions drain =============

def test_live_events_are_buffered_not_written_on_the_reader_thread():
    """SQLite connections are not shareable across threads, and the feed
    readers are separate threads from the decision loop. Buffering is what
    makes that safe; causality is unaffected because `recv_wall_timestamp_ns`
    is stamped when the bytes arrived, not when they are drained."""
    import types as _t

    from xamarinbot.shadow.live import LiveShadowService

    svc = _t.SimpleNamespace(store=_t.SimpleNamespace(db_path="x.db"))
    s = LiveShadowService(svc, ShadowJournal(":memory:"), model=None,
                          feature_set=None, log=lambda *a: None)
    b = RawEventBuilder(session_id="live")
    ev = b.build(Topic.RTDS_TWAP_60, "update", {"payload": {"value": 1.0}},
                 round_id=ROUND, source_timestamp_ns=START_NS)
    s._on_live_event(ev)
    assert len(s._inbox) == 1, "the reader thread must only enqueue"


def test_draining_projects_buffered_events_into_the_round():
    import types as _t

    from xamarinbot.shadow.live import LiveShadowService

    svc = _t.SimpleNamespace(store=_t.SimpleNamespace(db_path="x.db"))
    s = LiveShadowService(svc, ShadowJournal(":memory:"), model=None,
                          feature_set=None, log=lambda *a: None)
    meta = real_metadata(start_ts=1000.0, end_ts=1300.0)
    cap = _t.SimpleNamespace(
        metadata=meta,
        lifecycle=_t.SimpleNamespace(state=_t.SimpleNamespace(name="ACTIVE")),
    )
    shadow = s.ensure_round(cap)
    b = RawEventBuilder(session_id="live")
    s._on_live_event(b.build(
        Topic.RTDS_TWAP_60, "update", {"payload": {"value": 63000.0}},
        round_id=meta.round_id, source_timestamp_ns=int(1010 * 1e9)))
    assert s._drain_inbox() == 1
    assert shadow.projector.counts.get("TWAP") == 1
    assert shadow.raw_seq_by_session, 'the raw range must be recorded per stream'
    assert shadow.raw_recv_first_ns is not None
