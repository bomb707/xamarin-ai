"""Gate A.0 - real training-dataset integrity preflight.

Every test here is a regression against a way the dataset could look larger,
cleaner or more significant than it is. They fall into three groups:

  labels      a target must be independently verified, not taken on the
              venue's word (items 1, 2, 3, 8)
  examples    example count must be set by the strategy's cadence, not by
              market noise, and nothing may be visible before it arrived
              (items 4, 5, 9)
  statistics  one round is one observation, and adjacent rounds are not
              independent (items 6, 7)
"""
from __future__ import annotations

import json

import pytest

from xamarinbot.eligibility import Disqualifier, RoundEligibility, build, summarize
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model import blockstats
from xamarinbot.model.calibration import fit_platt
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.logistic import fit_logistic_regression, _fit_standardization
from xamarinbot.model.real_dataset import (
    DECISION_GRID_END_S,
    DECISION_GRID_START_S,
    build_real_examples,
    chronological_split,
    decision_grid,
    visible_events,
)
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.label import UnsupportedSettlementKind, topic_for_basis
from xamarinbot.realtime.preflight import evaluate_round, preflight
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.replay.projection import (
    ProjectionError,
    project_round,
    settlement_topic_for,
)

from tests.test_real_projection import (
    END_NS, ROUND, START_NS, UP, DOWN, make_capture, new_out, _realistic_recv,
)

FS = FeatureSet("gate_a0", base=("z_gap",))


# =================== item 1: three independent gates ======================

def test_confirmed_label_alone_does_not_make_a_round_trainable():
    """The headline error: `LabelStatus.CONFIRMED == trainable`. A perfectly
    verified label sitting on a round whose book drifted from the venue is
    not a usable training round."""
    rec = build(
        "r1",
        label_status="CONFIRMED", reconstructed_outcome="UP",
        reported_outcome="UP", declared_agrees=True,
        metrics={"dropped_events": 0, "parse_failures": 0},
        round_integrity_mismatches=1,          # <- the book went out of sync
    )
    assert rec.label_valid is True
    assert rec.data_valid is False
    assert rec.training_eligible is False
    assert Disqualifier.BOOK_INTEGRITY_MISMATCH in rec.data_disqualifiers


def test_training_eligible_is_the_conjunction_of_all_three():
    def rec(label=True, data=True, proj=True) -> RoundEligibility:
        return build(
            "r",
            label_status="CONFIRMED" if label else "UNRESOLVED",
            reconstructed_outcome="UP" if label else None,
            reported_outcome="UP", declared_agrees=label,
            metrics={"dropped_events": 0 if data else 3, "parse_failures": 0},
            round_integrity_mismatches=0,
            projection_problems=[] if proj else [Disqualifier.PROJECTION_FAILED],
        )

    assert rec().training_eligible is True
    assert rec(label=False).training_eligible is False
    assert rec(data=False).training_eligible is False
    assert rec(proj=False).training_eligible is False


def test_every_data_quality_failure_is_reported_separately():
    rec = build(
        "r1", label_status="CONFIRMED", reconstructed_outcome="UP",
        reported_outcome="UP", declared_agrees=True,
        metrics={"dropped_events": 5, "parse_failures": 2},
        round_integrity_mismatches=1,
    )
    assert set(rec.data_disqualifiers) == {
        Disqualifier.DROPPED_EVENTS,
        Disqualifier.PARSE_FAILURES,
        Disqualifier.BOOK_INTEGRITY_MISMATCH,
    }


def test_missing_recorder_metrics_is_not_treated_as_clean():
    """Absence of evidence is not evidence of cleanliness."""
    rec = build(
        "r1", label_status="CONFIRMED", reconstructed_outcome="UP",
        reported_outcome="UP", declared_agrees=True,
        metrics=None, round_integrity_mismatches=0,
    )
    assert rec.data_valid is False
    assert Disqualifier.NO_RECORDER_METRICS in rec.data_disqualifiers


def test_index_fields_carry_the_full_breakdown():
    fields = build(
        "r1", label_status="CONFIRMED", reconstructed_outcome="UP",
        reported_outcome="UP", declared_agrees=True,
        metrics={"dropped_events": 0, "parse_failures": 0},
        round_integrity_mismatches=1,
    ).as_index_fields()
    for key in ("label_valid", "data_training_grade", "projection_valid",
                "training_eligible", "data_disqualifiers"):
        assert key in fields
    assert fields["training_eligible"] is False
    assert "book_integrity_mismatch" in fields["data_disqualifiers"]


def test_summary_reports_the_five_counts_separately():
    records = [
        build("a", label_status="CONFIRMED", reconstructed_outcome="UP",
              reported_outcome="UP", declared_agrees=True,
              metrics={"dropped_events": 0, "parse_failures": 0},
              round_integrity_mismatches=0),
        build("b", label_status="CONFIRMED", reconstructed_outcome="UP",
              reported_outcome="UP", declared_agrees=True,
              metrics={"dropped_events": 0, "parse_failures": 0},
              round_integrity_mismatches=1),
    ]
    s = summarize(records)
    assert s == {
        "captured": 2, "label_valid": 2, "data_training_grade": 1,
        "projection_valid": 2, "training_eligible": 1,
        "disqualifiers_by_reason": {"book_integrity_mismatch": 1},
        "eligible_round_ids": ["a"],
    }


def test_preflight_over_a_real_capture_separates_the_gates(tmp_path):
    raw = make_capture(tmp_path)
    report = preflight(raw)
    assert report.counts["captured"] == 1
    text = report.format()
    for line in ("captured rounds", "label CONFIRMED", "data-quality clean",
                 "projection valid", "FINAL training eligible"):
        assert line in text


# ============ item 2: a RoundLabel needs a CONFIRMED reconstruction =======

def test_unresolved_reported_outcome_cannot_become_a_round_label(tmp_path):
    """`reported=UP, reconstructed=None, status=UNRESOLVED` used to emit
    `RoundLabel(outcome=UP)` because projection checked only the venue's
    word - discarding the entire point of reconstructing the rule ourselves."""
    raw = make_capture(tmp_path)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP",
        "reconstructed_outcome": None, "label_agreement": None,
    })
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.label is None
    assert any("could not be independently reconstructed" in w for w in res.warnings)


def test_an_ambiguous_label_cannot_become_a_round_label(tmp_path):
    raw = make_capture(tmp_path)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP",
        "reconstructed_outcome": "DOWN", "label_agreement": 0,
    })
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.label is None
    assert any("disagrees with the venue" in w for w in res.warnings)


def test_a_confirmed_reconstruction_does_produce_a_label(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.label is not None
    assert res.label.outcome.value == "UP"


def test_an_ineligible_round_never_reaches_the_model_dataset(tmp_path):
    """End to end: no label -> no examples, whatever the venue said."""
    raw = make_capture(tmp_path, n=200)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP",
        "reconstructed_outcome": None, "label_agreement": None,
    })
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.label is None
    ds = build_real_examples(out, [l for l in [res.label] if l], FeatureConfig(), FS)
    assert ds.examples == []


# ================ item 3 / 8: settlement rule validation ==================

def test_rule_text_disagreement_makes_the_label_ambiguous():
    from xamarinbot.realtime.label import (
        BasisReconstruction, LabelReconstruction, LabelStatus, Outcome,
    )

    basis = BasisReconstruction(
        "declared", "crypto_prices_twap_sixty", 1.0, 2.0, 1, 2, 0.0, 0.0,
        Outcome.UP, "up",
    )
    rec = LabelReconstruction(
        round_id="r", declared=basis, reference=basis,
        reported_outcome=Outcome.UP, reported_source="gamma",
        rule_text_agrees=False,
    )
    assert rec.status is LabelStatus.AMBIGUOUS
    assert rec.is_trainable is False


def test_rule_text_silence_is_not_disagreement():
    from xamarinbot.realtime.label import (
        BasisReconstruction, LabelReconstruction, LabelStatus, Outcome,
    )

    basis = BasisReconstruction(
        "declared", "t", 1.0, 2.0, 1, 2, 0.0, 0.0, Outcome.UP, "up",
    )
    rec = LabelReconstruction(
        round_id="r", declared=basis, reference=basis,
        reported_outcome=Outcome.UP, reported_source="gamma",
        rule_text_agrees=None,
    )
    assert rec.status is LabelStatus.CONFIRMED


def test_the_recorder_passes_the_market_rule_text_into_reconstruction():
    """Item 3: `reconstruct_label` always accepted the text; nothing passed
    it, so `rule_text_agrees` was permanently None."""
    import inspect

    from xamarinbot.realtime import service as mod

    src = inspect.getsource(mod.RealRecorderService._finalize)
    assert "resolution_source=m.resolution_source" in src
    assert "description=m.description" in src


@pytest.mark.parametrize("kind", ["foo", "twap", "", "CHAINLINK_TWAP", "chainlink-twap"])
def test_unknown_settlement_kind_fails_closed(kind):
    """Item 8: both helpers used to fall through to the plain Chainlink
    reference for ANY unrecognized value, silently choosing the wrong price
    series as label truth."""
    with pytest.raises(UnsupportedSettlementKind):
        topic_for_basis(kind, 60)
    with pytest.raises(ProjectionError):
        settlement_topic_for(kind, 60)


def test_supported_settlement_kinds_still_resolve():
    assert topic_for_basis("chainlink_twap", 60) == "crypto_prices_twap_sixty"
    assert topic_for_basis("chainlink_twap", 30) == "crypto_prices_twap_thirty"
    assert topic_for_basis("chainlink_reference", None) == "crypto_prices_chainlink"


def test_an_unsupported_twap_window_also_fails_closed():
    with pytest.raises(UnsupportedSettlementKind):
        topic_for_basis("chainlink_twap", 45)


# ============ item 4: preregistered grid, immune to event rate ============

def test_the_decision_grid_is_the_declared_strategy_cadence():
    grid = decision_grid()
    assert grid[0] == DECISION_GRID_START_S == 15.0
    assert grid[-1] == DECISION_GRID_END_S == 270.0
    assert all(round(b - a, 6) == 3.0 for a, b in zip(grid, grid[1:]))
    assert len(grid) == 86


def test_no_pre_round_decision_points():
    assert min(decision_grid()) > 0.0


def test_event_rate_inflation_cannot_increase_the_example_count(tmp_path):
    """The decisive regression. A busier market must not produce a bigger
    training set - event rate is not information about the outcome."""
    quiet_dir, busy_dir = tmp_path / "quiet", tmp_path / "busy"
    quiet_dir.mkdir()
    busy_dir.mkdir()

    quiet = make_capture(quiet_dir, n=300)
    out_q = new_out(quiet_dir)
    res_q = project_round(quiet, ROUND, out_q)
    ds_q = build_real_examples(out_q, [res_q.label], FeatureConfig(), FS)

    # the same round plus 100k additional intermediate book deltas
    busy = make_capture(busy_dir, n=300)
    b = RawEventBuilder(session_id="flood")
    flood = []
    for i in range(100_000):
        at_ns = START_NS + int((15.0 + (i % 250)) * 1e9) + (i % 1000) * 1_000_000
        flood.append(b.build(
            Topic.CLOB_MARKET, "price_change",
            {"asset_id": UP, "market": "0xcond", "timestamp": str(at_ns // 1_000_000),
             "price": "0.45", "size": str(50 + (i % 7)), "side": "BUY", "hash": f"f{i}"},
            round_id=ROUND, token_id=UP, normalized_side="UP",
            source_timestamp_ns=at_ns,
        ))
    busy.write_batch(_realistic_recv(flood))
    out_b = new_out(busy_dir)
    res_b = project_round(busy, ROUND, out_b)
    ds_b = build_real_examples(out_b, [res_b.label], FeatureConfig(), FS)

    assert res_b.counts["BOOK_DELTA"] > res_q.counts["BOOK_DELTA"] + 90_000, (
        "the flood must really have reached the projection"
    )
    assert ds_q.examples, "the quiet round must produce examples at all"
    assert [e.decision_ts for e in ds_q.examples] == [e.decision_ts for e in ds_b.examples]
    assert len(ds_q.examples) == len(ds_b.examples)


# ================= item 5: receive-time causal visibility =================

def test_an_event_not_yet_received_is_invisible():
    """`source_ts=100.0, recv_ts=101.5` must not be visible at t=100.5, even
    though its source timestamp has passed."""
    store = EventStore(":memory:", provenance=DataProvenance.REAL_REPLAY)
    store.append(EventType.SPOT, "r", recv_ts=101.5, source_ts=100.0, payload={"value": 1.0})
    events = store.all_events("r")

    assert visible_events(events, 100.5) == []
    assert len(visible_events(events, 102.0)) == 1
    store.close()


def test_a_usable_observation_must_satisfy_both_clocks():
    """`recv_ts <= t` AND `source_ts <= t`. The caller enforces the first,
    `compute()` the second."""
    store = EventStore(":memory:", provenance=DataProvenance.REAL_REPLAY)
    # arrived early, but stamped in the future (clock skew)
    store.append(EventType.SPOT, "r", recv_ts=99.0, source_ts=105.0, payload={"value": 1.0})
    events = store.all_events("r")
    vis = visible_events(events, 100.0)
    assert len(vis) == 1                       # recv gate passes
    assert [e for e in vis if e.event_time <= 100.0] == []   # source gate does not
    store.close()


def test_the_real_dataset_uses_receive_time_gating(tmp_path):
    """A late-arriving observation must not appear in an earlier example."""
    import inspect

    from xamarinbot.model import real_dataset as mod

    src = inspect.getsource(mod.build_real_examples)
    assert "visible_events(events, decision_ts)" in src


# ============ item 6: round-balanced weights, no pseudo-replication =======

def test_each_round_contributes_equal_total_weight(tmp_path):
    """A quiet round and a busy round must carry the same statistical
    weight: the target is one settlement outcome per round."""
    raw = make_capture(tmp_path, n=400)
    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    ds = build_real_examples(out, [res.label], FeatureConfig(), FS)
    assert ds.examples
    assert ds.total_weight(ROUND) == pytest.approx(1.0)
    assert all(e.weight == pytest.approx(1.0 / len(ds.examples)) for e in ds.examples)


def test_quiet_and_busy_rounds_get_equal_weight_despite_unequal_counts():
    """Explicitly the item 6 test: 40 vs 80 valid decisions, weight 1 each."""
    from xamarinbot.model.real_dataset import RealDatasetResult, RealExample

    def rows(round_id: str, n: int) -> list[RealExample]:
        return [
            RealExample(round_id, float(i), float(i), None, [0.0], 1, 1.0 / n)
            for i in range(n)
        ]

    ds = RealDatasetResult(feature_set="f")
    ds.examples = rows("quiet", 40) + rows("busy", 80)
    ds.valid_per_round = {"quiet": 40, "busy": 80}
    assert ds.total_weight("quiet") == pytest.approx(1.0)
    assert ds.total_weight("busy") == pytest.approx(1.0)


def test_weighted_standardization_matches_the_specified_formula():
    X = [[1.0], [2.0], [10.0]]
    w = [0.5, 0.5, 0.0]         # the outlier carries no weight
    means, stds = _fit_standardization(X, w)
    assert means[0] == pytest.approx(1.5)
    assert stds[0] == pytest.approx(0.5)


def test_sample_weights_actually_change_the_fit():
    """A zero-weighted block must not influence the model."""
    X = [[0.0], [1.0]] * 10 + [[5.0]] * 50
    y = [0, 1] * 10 + [0] * 50
    w_all = [1.0] * len(X)
    w_ignore_tail = [1.0] * 20 + [0.0] * 50

    m_all = fit_logistic_regression(X, y, "f", ("x",), sample_weight=w_all)
    m_tail = fit_logistic_regression(X, y, "f", ("x",), sample_weight=w_ignore_tail)
    m_none = fit_logistic_regression(X[:20], y[:20], "f", ("x",))

    assert m_all.weights != m_tail.weights
    assert m_tail.weights[0] == pytest.approx(m_none.weights[0], abs=1e-6)


def test_scaling_all_weights_leaves_the_fit_unchanged():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    a = fit_logistic_regression(X, y, "f", ("x",), sample_weight=[1.0] * 4)
    b = fit_logistic_regression(X, y, "f", ("x",), sample_weight=[7.0] * 4)
    assert a.weights[0] == pytest.approx(b.weights[0], abs=1e-9)
    assert a.bias == pytest.approx(b.bias, abs=1e-9)


def test_platt_calibration_accepts_weights():
    q = [0.1, 0.2, 0.8, 0.9] * 5
    y = [0, 0, 1, 1] * 5
    cal = fit_platt(q, y, sample_weight=[1.0 / 20] * 20)
    assert 0.0 <= cal.transform(0.5) <= 1.0


# ================ item 7: serial correlation, not IID rounds ==============

def test_autocorrelation_detects_persistence():
    persistent = [float(i % 20) for i in range(200)]
    acf = blockstats.autocorrelation(persistent, 5)
    assert acf[0] > 0.5, "a persistent series must show positive lag-1 autocorrelation"


def test_effective_sample_size_is_below_n_for_correlated_series():
    persistent = []
    x = 0.0
    rng_state = 12345
    for _ in range(300):
        rng_state = (rng_state * 1103515245 + 12345) % (1 << 31)
        x = 0.9 * x + (rng_state / (1 << 31) - 0.5)
        persistent.append(x)
    n_eff = blockstats.effective_sample_size(persistent)
    assert 1.0 <= n_eff < len(persistent), "correlated rounds are worth fewer observations"


def test_block_bootstrap_is_deterministic_and_brackets_the_point():
    xs = [float(i % 7) - 3.0 for i in range(120)]
    a = blockstats.moving_block_bootstrap(xs, 6, n_resamples=300, seed_key="t")
    b = blockstats.moving_block_bootstrap(xs, 6, n_resamples=300, seed_key="t")
    assert (a.lo, a.hi, a.point) == (b.lo, b.hi, b.point)
    assert a.lo <= a.point <= a.hi


def test_block_sensitivity_reports_every_block_length_not_the_best_one():
    xs = [float(i % 5) for i in range(200)]
    res = blockstats.analyze_series(xs, "pnl_per_round")
    assert set(res.by_block) == {6, 12, 24}          # 30 / 60 / 120 minutes
    d = res.as_dict()
    assert set(d["blocks"]) == {"30min", "60min", "120min"}
    assert d["effective_sample_size"] <= res.n_rounds


def test_the_status_no_longer_claims_rounds_are_independent():
    """Item 7's language correction. The old status line called consecutive
    rounds "chronologically independent"; they are non-overlapping, which is
    a strictly weaker property."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    text = (root / "scripts" / "run_continuous_capture.py").read_text()
    status = text[text.index("def print_status"):text.index("def run_batch")]
    assert "NOT automatically" in status
    assert "independent real rounds" not in status
    assert "block bootstrap" in status and "effective sample size" in status


# ==================== chronological, round-disjoint splits ================

def test_chronological_splits_are_round_disjoint_and_ordered():
    from xamarinbot.model.real_dataset import RealDatasetResult, RealExample

    ds = RealDatasetResult(feature_set="f")
    for r in range(10):
        for k in range(5):
            ds.examples.append(
                RealExample(f"r{r:02d}", 15.0 + k, r * 1000.0 + k, None, [0.0], r % 2, 0.2)
            )
        ds.valid_per_round[f"r{r:02d}"] = 5

    split = chronological_split(ds)
    assert split.is_round_disjoint
    assert max(e.decision_ts for e in split.train) < min(e.decision_ts for e in split.calibrate)
    assert max(e.decision_ts for e in split.calibrate) < min(e.decision_ts for e in split.test)


# ================ item 9: no future tick at initialization ================

def test_initial_session_constraints_come_from_the_opening_config():
    import inspect

    from xamarinbot.shadow import runner as mod

    src = inspect.getsource(mod.ShadowRunner.run)
    assert "earliest_config = min(opening_configs" in src
    assert "constraints = constraints_at(events)" not in src, (
        "the initializer must not be able to select a future tick update"
    )


def test_the_opening_constraints_use_the_first_tick_not_a_later_one(tmp_path):
    raw = make_capture(tmp_path, n=200, tick_change_at=60.0)
    out = new_out(tmp_path)
    project_round(raw, ROUND, out)

    events = out.all_events(ROUND)
    configs = [e for e in events if e.event_type is EventType.MARKET_CONFIG]
    earliest = min(configs, key=lambda e: (e.event_time, e.sequence))
    latest = max(configs, key=lambda e: (e.event_time, e.sequence))
    assert earliest.payload["tick_size"] == 0.01
    assert latest.payload["tick_size"] == 0.001


# ============ item 1: reference feeds are global, not per-round ===========

def test_reference_observations_are_selected_by_time_not_round_tag(tmp_path):
    """A real multi-round batch tags each reference observation with whichever
    round was ACTIVE. Filtering by that tag would leave rounds 2..N with no
    observation at their own boundaries, and the projection would refuse to
    label rounds whose data is present but filed under a neighbour."""
    import dataclasses

    source = make_capture(tmp_path, n=40)
    reference = (Topic.RTDS_TWAP_60, Topic.RTDS_TWAP_30, Topic.RTDS_CHAINLINK,
                 Topic.RTDS_BINANCE)
    rebuilt = [
        dataclasses.replace(e, round_id="a-neighbouring-round")
        if e.topic in reference else e
        for e in source.events()
    ]
    assert any(e.round_id == "a-neighbouring-round" for e in rebuilt)

    raw = RawEventStore(str(tmp_path / "retagged.db"))
    raw.upsert_round(source.get_round(ROUND))
    for row in source.round_results():
        raw.upsert_round_result(row)
    raw.write_batch(rebuilt)

    out = new_out(tmp_path)
    res = project_round(raw, ROUND, out)
    assert res.counts["TWAP"] > 0, "reference data must be found despite the round tag"
    assert res.p0 > 0

    # ... and the preflight must reach exactly the same verdict on both
    # stores. Whose round a global reference observation happens to be filed
    # under cannot change whether a round is trainable.
    assert (
        evaluate_round(raw, ROUND).disqualifiers
        == evaluate_round(source, ROUND).disqualifiers
    )
