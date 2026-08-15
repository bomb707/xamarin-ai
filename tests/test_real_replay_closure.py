"""Phase 12C.2: final real-replay correctness closure.

Four fabrications are removed here. Each one produced a *plausible* replay
that was quietly wrong in a way no aggregate metric would have shown:

  item 1  the simulation could charge a fee, or wait a taker delay, that the
          market never reported
  item 2  one immutable tick for the whole round, though the venue changed it
          mid-round
  item 3  a MARKET_CONFIG timestamp invented to win a sort, and a SETTLEMENT
          stamped at the round end though the outcome was learned 92s later
  item 4  a guessed settlement rule when none was recorded
"""
from __future__ import annotations

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig, MakerFillConfig
from xamarinbot.feeds.base import MarketConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.market.constraints import (
    ExecutionStateConflict,
    MarketConstraintError,
    MarketConstraints,
    reconcile_execution_state,
)
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.raw_events import Topic
from xamarinbot.replay.feeds import market_config_from_payload
from xamarinbot.replay.projection import ProjectionError, project_round
from xamarinbot.shadow.config import ShadowConfig
from xamarinbot.shadow.runner import ShadowRunner

from tests.test_real_projection import END_NS, ROUND, START_NS, make_capture, new_out

MARKET_FEE = FeeConfig(crypto_fee_rate=0.07)


def real_constraints(**kw) -> MarketConstraints:
    base = MarketConstraints.for_testing(**kw)
    return type(base)(**{**base.__dict__, "provenance": DataProvenance.REAL_REPLAY})


# ===================== item 1: MarketConstraints is the SSOT =============

def test_real_replay_derives_fee_and_delay_from_the_market():
    """`FeeUsed = FeeReportedByMarket`, `TakerDelayUsed = DelayReportedByMarket`
    hold by construction, not by a caller remembering to match them."""
    c = real_constraints(fee_rate=0.07, taker_delay_ms=250.0)
    fee, exec_cfg = reconcile_execution_state(c, None, None)
    assert fee.crypto_fee_rate == 0.07
    assert exec_cfg.taker_delay_ms == 250.0
    assert fee is c.fee_configuration, "the market's own object, not a copy"


def test_a_contradicting_fee_fails_loudly_on_real_data():
    c = real_constraints(fee_rate=0.07)
    with pytest.raises(ExecutionStateConflict, match="contradicts the market"):
        reconcile_execution_state(c, FeeConfig(crypto_fee_rate=0.02), None)


def test_a_contradicting_taker_delay_fails_loudly_on_real_data():
    c = real_constraints(taker_delay_ms=250.0)
    with pytest.raises(ExecutionStateConflict, match="taker delay"):
        reconcile_execution_state(c, None, ExecutionConfig(taker_delay_ms=0.0))


def test_maker_simulation_knobs_are_not_market_facts():
    """The maker fill model is a simulation parameter, not something the venue
    reports, so a caller keeps it while the delay is still overridden."""
    c = real_constraints(taker_delay_ms=250.0)
    maker = MakerFillConfig(base_fill_rate_per_s=0.99)
    _, exec_cfg = reconcile_execution_state(
        c, None, ExecutionConfig(taker_delay_ms=250.0, maker=maker)
    )
    assert exec_cfg.maker is maker
    assert exec_cfg.taker_delay_ms == 250.0


def test_synthetic_runs_may_still_vary_fee_and_delay():
    """Being able to vary them is the point of a generated round."""
    c = MarketConstraints.for_testing(fee_rate=0.07, taker_delay_ms=0.0)
    fee, exec_cfg = reconcile_execution_state(
        c, FeeConfig(crypto_fee_rate=0.02), ExecutionConfig(taker_delay_ms=999.0)
    )
    assert fee.crypto_fee_rate == 0.02
    assert exec_cfg.taker_delay_ms == 999.0


def test_the_market_fee_reaches_candidate_and_execution_math():
    """Not just stored - actually used by the EV/cost arithmetic and by the
    simulator that resolves fills."""
    from xamarinbot.execution.session import TradingSession

    c = real_constraints(fee_rate=0.07, taker_delay_ms=250.0)
    session = TradingSession(
        "r1", FeeConfig(crypto_fee_rate=0.07), ExecutionConfig(taker_delay_ms=250.0),
        OneStepConfig(g_min=-1000.0), c,
    )
    assert session.fee_config.crypto_fee_rate == 0.07
    assert session.sim.fee_config is session.fee_config
    # the simulator the fills actually run through carries the market's delay
    assert session.sim.cfg.taker_delay_ms == 250.0
    # and the fee formula uses that rate
    assert session.fee_config.taker_fee(10.0, 0.5) == pytest.approx(10 * 0.07 * 0.5 * 0.5)


def test_a_session_built_with_a_wrong_fee_on_real_data_raises():
    from xamarinbot.execution.session import TradingSession

    with pytest.raises(ExecutionStateConflict):
        TradingSession(
            "r1", FeeConfig(crypto_fee_rate=0.02), ExecutionConfig(),
            OneStepConfig(g_min=-1000.0), real_constraints(fee_rate=0.07),
        )


# ===================== item 2: causal, dynamic tick size =================

def test_tick_size_change_is_projected_as_a_later_market_config(tmp_path):
    raw = make_capture(tmp_path, n=40, tick_change_at=20.0)
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)

    configs = [e for e in out.all_events(ROUND) if e.event_type is EventType.MARKET_CONFIG]
    assert len(configs) == 2, "the initial config plus one tick update"
    ticks = [e.payload["tick_size"] for e in sorted(configs, key=lambda e: e.event_time)]
    assert ticks == [0.01, 0.001]
    # the update retains every other market constraint
    later = max(configs, key=lambda e: e.event_time)
    assert later.payload["min_order_size"] == 5.0
    assert later.payload["settlement_kind"] == "chainlink_twap"
    assert later.payload["twap_window_seconds"] == 60
    # one config per change, not one per token
    # the second token's announcement of the SAME tick is not a change
    assert res.skipped.get("tick_size_change_repeat_same_value") == 1
    assert res.tick_timeline[-1][1] == 0.001


def test_tick_is_the_latest_value_visible_by_decision_time(tmp_path):
    """`tick(t) = latestRecordedTickVisibleByDecisionTime`. No future tick may
    leak backward."""
    raw = make_capture(tmp_path, n=40, tick_change_at=20.0)
    out = new_out(tmp_path)
    project_round(raw, ROUND, out)
    events = out.all_events(ROUND)
    change_ts = START_NS / 1e9 + 20.0

    def tick_at(decision_ts: float) -> float:
        visible = [e for e in events
                   if e.event_type is EventType.MARKET_CONFIG and e.recv_ts <= decision_ts]
        latest = max(visible, key=lambda e: (e.event_time, e.sequence))
        return market_config_from_payload(latest.payload).tick_size

    assert tick_at(change_ts - 1.0) == 0.01     # before the event
    assert tick_at(change_ts + 1.0) == 0.001    # after the event
    assert tick_at(change_ts - 0.001) == 0.01, "no backward leak"


def test_the_verified_capture_has_a_real_mid_round_tick_change(tmp_path):
    """Documents that this is not a hypothetical: the canonical capture
    genuinely changes tick from 0.01 to 0.001 partway through the round."""
    import pathlib

    from xamarinbot.realtime.raw_store import RawEventStore

    capture = pathlib.Path("captures/phase12c_verify.db")
    if not capture.exists():
        pytest.skip("capture databases are gitignored; run scripts/run_real_recorder.py")

    store = RawEventStore(str(capture))
    rid = store.round_ids()[0]
    row = store.get_round(rid)
    changes = [
        e for e in store.events(round_id=rid, topics=[Topic.CLOB_MARKET])
        if e.event_type == "tick_size_change"
    ]
    assert changes, "the canonical capture must contain the tick change this item is about"
    p = changes[0].payload
    assert float(p["old_tick_size"]) == 0.01
    assert float(p["new_tick_size"]) == 0.001
    rel = changes[0].source_timestamp_ns / 1e9 - row["start_ts_ns"] / 1e9
    assert 0 < rel < 300, f"the change is mid-round at t=+{rel:.1f}s"
    store.close()


def test_shadow_runner_passes_the_current_tick_into_every_decision(tmp_path):
    """Behavioural, not textual: capture the `MarketConstraints` the runner
    actually hands the controller at each decision point and confirm the tick
    switches from 0.01 to 0.001 exactly when the venue's change becomes
    visible - and never before."""
    from unittest.mock import patch

    from xamarinbot.optimizer.controller import OneStepController, OneStepDecision
    from xamarinbot.optimizer.candidates import wait_candidate

    raw = make_capture(tmp_path, n=200, tick_change_at=100.0)
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    change_ts = START_NS / 1e9 + 100.0

    seen: list[tuple[float, float]] = []

    def spy_decide(self, round_id, decision_ts, portfolio, q, permitted,
                   book_up, book_down, constraints, is_fresh, **kw):
        seen.append((decision_ts, constraints.tick_size))
        wait = wait_candidate("wait", portfolio)
        return OneStepDecision(round_id, decision_ts, wait, (wait,), skip_reason=None)

    # a stand-in model so the runner reaches the decision (12C.1 item 10
    # otherwise blocks a modelless REAL run before any tick is read)
    class ConstantModel:
        def predict_proba(self, x):
            return 0.55

    from xamarinbot.model.features import FeatureSet

    fs = FeatureSet("const", base=("z_gap",))
    with patch.object(OneStepController, "decide", spy_decide):
        ShadowRunner(
            out, ROUND, res.p0, FeatureConfig(), MARKET_FEE, ExecutionConfig(),
            OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0),
            model=ConstantModel(), feature_set=fs, cfg=ShadowConfig(),
        ).run()

    assert seen, "the runner must have reached at least one decision"
    before = [tick for ts, tick in seen if ts < change_ts]
    after = [tick for ts, tick in seen if ts > change_ts + 1.0]
    assert before and all(t == 0.01 for t in before), (
        f"no future tick may leak backward; saw {sorted(set(before))} before the change"
    )
    assert after and all(t == 0.001 for t in after), (
        f"the new tick must apply after the change; saw {sorted(set(after))}"
    )


# ============ item 3: no fabricated timestamps in normalized REAL =========

def test_market_config_uses_the_real_metadata_receive_timestamp(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    project_round(raw, ROUND, out)
    config = next(e for e in out.all_events(ROUND)
                  if e.event_type is EventType.MARKET_CONFIG)

    # REST metadata has no external source timestamp - None is the honest value
    assert config.source_ts is None
    assert config.recv_ts == (START_NS - 470_000_000_000) / 1e9
    # and NOT the old fabricated `min(earliest, start_ts) - 1e-6`
    fabricated = START_NS / 1e9 - 1e-6
    assert abs(config.recv_ts - fabricated) > 400.0, (
        "the config must be visible at the moment the metadata was really "
        "received, not at a timestamp invented to sort before the round"
    )


def test_a_capture_without_metadata_refuses_rather_than_inventing_a_timestamp(tmp_path):
    raw = make_capture(tmp_path, with_metadata=False)
    out = new_out(tmp_path)
    with pytest.raises(ProjectionError, match="Refusing to fabricate one"):
        project_round(raw, ROUND, out)


def test_the_label_is_not_in_the_causal_event_stream(tmp_path):
    """"causal market events -> FeatureEngine, eventual RoundLabel ->
    supervised target"."""
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    res = project_round(raw, ROUND, out)

    assert not [e for e in out.all_events(ROUND) if e.event_type is EventType.SETTLEMENT]
    # the target travels on its own path instead
    assert res.label is not None
    assert res.label.outcome.value == "UP"
    assert res.label.provenance is DataProvenance.REAL_REPLAY


def test_the_label_could_not_be_observed_before_it_was_known(tmp_path):
    """The decisive test: the outcome was learned 92s after the round closed,
    so nothing in the feature/controller stream may see it before then."""
    raw = make_capture(tmp_path)
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out, include_settlement=True)

    end_ts = END_NS / 1e9
    assert res.label_observed_at is not None
    assert res.label_observed_at > end_ts, "the outcome was learned after the close"

    settlements = [e for e in out.all_events(ROUND) if e.event_type is EventType.SETTLEMENT]
    assert settlements, "include_settlement=True should emit it"
    s = settlements[0]
    assert s.recv_ts == pytest.approx(res.label_observed_at)
    assert s.event_time > end_ts, (
        "a label must never become causally visible merely because the "
        "five-minute clock ended"
    )

    # and it is invisible to any decision point inside the round
    for t in (0.0, 150.0, 299.0, 300.0):
        visible = [e for e in out.all_events(ROUND)
                   if e.event_type is EventType.SETTLEMENT and e.event_time <= END_NS / 1e9 - 300.0 + t]
        assert not visible, f"the outcome leaked at t=+{t}s"


def test_settlement_is_omitted_when_no_resolution_was_ever_observed(tmp_path):
    raw = make_capture(tmp_path, with_resolution=False)
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out, include_settlement=True)
    assert not [e for e in out.all_events(ROUND) if e.event_type is EventType.SETTLEMENT]
    assert any("omitting SETTLEMENT" in w for w in res.warnings)


# ================== item 4: no settlement-rule fallbacks ==================

def test_a_missing_settlement_rule_fails_the_projection_closed(tmp_path):
    raw = make_capture(tmp_path)
    row = raw.get_round(ROUND)
    row["settlement_kind"] = None
    raw.upsert_round(row)
    out = new_out(tmp_path)
    with pytest.raises(ProjectionError, match="Refusing to guess one"):
        project_round(raw, ROUND, out)


def test_market_constraints_will_not_default_a_settlement_rule():
    cfg = MarketConfig(
        market_id="m", up_token_id="U", down_token_id="D", start_ts=0.0, end_ts=300.0,
        tick_size=0.01, min_order_size=5.0, fee_rate=0.07, taker_delay_ms=0.0,
        twap_window_seconds=60, settlement_kind="",
    )
    with pytest.raises(MarketConstraintError, match="refusing to guess"):
        MarketConstraints.from_market_config(cfg)


def test_settlement_rule_survives_raw_metadata_to_market_constraints(tmp_path):
    """Raw round metadata -> MARKET_CONFIG -> MarketConstraints."""
    raw, out = make_capture(tmp_path, twap_window=30), new_out(tmp_path)
    project_round(raw, ROUND, out)
    payload = next(e.payload for e in out.all_events(ROUND)
                   if e.event_type is EventType.MARKET_CONFIG)
    assert payload["settlement_kind"] == "chainlink_twap"

    c = MarketConstraints.from_market_config(
        market_config_from_payload(payload),
        provenance=DataProvenance.REAL_REPLAY, source="projected",
    )
    assert c.settlement_kind == "chainlink_twap"
    assert c.twap_window_s == 30


def test_market_config_payload_must_carry_the_settlement_rule():
    with pytest.raises(KeyError, match="settlement_kind"):
        market_config_from_payload({
            "market_id": "m", "up_token_id": "U", "down_token_id": "D",
            "start_ts": 0.0, "end_ts": 300.0, "tick_size": 0.01,
            "min_order_size": 5.0, "fee_rate": 0.07, "taker_delay_ms": 0.0,
            "twap_window_seconds": 60,
        })


# ============================ end to end =================================

def test_a_real_replay_round_runs_with_market_derived_execution_state(tmp_path):
    """The whole chain: projected real events -> ShadowRunner, with fee,
    delay, tick and settlement rule all coming from the market."""
    raw = make_capture(tmp_path, n=120, tick_change_at=60.0)
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)

    result = ShadowRunner(
        out, ROUND, res.p0, FeatureConfig(), MARKET_FEE, ExecutionConfig(),
        OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0),
        model=None, feature_set=None, cfg=ShadowConfig(),
    ).run()

    assert result.provenance is DataProvenance.REAL_REPLAY
    # no model -> no ALPHA on real data (12C.1 item 10 still holds)
    assert result.n_model_unavailable > 0
    assert result.final_portfolio.U == 0.0
