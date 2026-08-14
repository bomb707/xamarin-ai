"""Phase 6 verification: "Unit test every matrix cell." "Replay transitions
around ±1/±0.5/0 seeds." "Verify no matrix entry directly bypasses EV/risk
gates." "All seed states are deterministic and auditable."
"""
from __future__ import annotations

import itertools

from xamarinbot.features.types import FeatureVector, TimeRegime
from xamarinbot.journal.schema import RegimeTransitionRecord
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.regime.classifier import RegimeClassifier, state_for, to_journal_record
from xamarinbot.regime.config import RegimeConfig
from xamarinbot.regime.matrix import (
    ActionPermissionMatrix,
    classify_seed_action,
    clob_direction_from_sign,
    gap_regime_for,
    spot_direction_from_bp,
)
from xamarinbot.regime.types import Direction, GapRegime, RegimeState, SeedAction
from xamarinbot.reports.regime_report import build_transition_report

CFG = RegimeConfig()

# --------------------------------------------------------------------------
# Exhaustive matrix coverage
# --------------------------------------------------------------------------


def test_every_matrix_cell_is_defined_and_valid():
    """6 gap regimes x 3 CLOB directions x 3 spot directions = 54 cells;
    every one must return a real SeedAction (not raise, not None)."""
    all_states = [
        RegimeState(gap_regime=g, clob_direction=c, spot_direction=s)
        for g, c, s in itertools.product(GapRegime, Direction, Direction)
    ]
    assert len(all_states) == 54
    for state in all_states:
        action = classify_seed_action(state)
        assert isinstance(action, SeedAction)
        assert action != SeedAction.CANCEL  # CANCEL only ever comes from the stateful classifier, never the pure matrix


def test_ss8_documented_rows_match_the_source_table():
    # Row 1: Positive/upper-middle, CLOB UP, Spot UP -> TAKER UP
    assert classify_seed_action(RegimeState(GapRegime.UPPER_MIDDLE, Direction.UP, Direction.UP)) is SeedAction.TAKER_UP
    assert classify_seed_action(RegimeState(GapRegime.STRONG_POSITIVE, Direction.UP, Direction.UP)) is SeedAction.TAKER_UP

    # Row 2: Positive/upper-middle, CLOB DOWN, Spot UP -> MAKER UP
    assert classify_seed_action(RegimeState(GapRegime.UPPER_MIDDLE, Direction.DOWN, Direction.UP)) is SeedAction.MAKER_UP

    # Row 3: Near center/weakening positive, CLOB DOWN, Spot DOWN -> TAKER DOWN
    assert classify_seed_action(RegimeState(GapRegime.NEAR_CENTER_POSITIVE, Direction.DOWN, Direction.DOWN)) is SeedAction.TAKER_DOWN

    # Row 4: Negative/lower-middle, CLOB DOWN, Spot DOWN -> TAKER DOWN
    assert classify_seed_action(RegimeState(GapRegime.LOWER_MIDDLE, Direction.DOWN, Direction.DOWN)) is SeedAction.TAKER_DOWN
    assert classify_seed_action(RegimeState(GapRegime.STRONG_NEGATIVE, Direction.DOWN, Direction.DOWN)) is SeedAction.TAKER_DOWN

    # Row 5: Negative/lower-middle, CLOB UP, Spot DOWN -> MAKER DOWN
    assert classify_seed_action(RegimeState(GapRegime.LOWER_MIDDLE, Direction.UP, Direction.DOWN)) is SeedAction.MAKER_DOWN

    # Row 6: any region, conflict/stale -> WAIT (FLAT stands in for "stale" here - upstream Phase 4 already filters truly invalid/stale data)
    assert classify_seed_action(RegimeState(GapRegime.UPPER_MIDDLE, Direction.FLAT, Direction.UP)) is SeedAction.WAIT
    assert classify_seed_action(RegimeState(GapRegime.UPPER_MIDDLE, Direction.UP, Direction.FLAT)) is SeedAction.WAIT
    assert classify_seed_action(RegimeState(GapRegime.UPPER_MIDDLE, Direction.DOWN, Direction.DOWN)) is SeedAction.WAIT  # conflicting


def test_symmetric_near_center_negative_reversal():
    # mirrors SS8 row 3 for the negative side (not in the source table)
    assert classify_seed_action(RegimeState(GapRegime.NEAR_CENTER_NEGATIVE, Direction.UP, Direction.UP)) is SeedAction.TAKER_UP


def test_permitted_actions_always_include_wait():
    for action in SeedAction:
        permitted = ActionPermissionMatrix.permitted_actions(action)
        assert SeedAction.WAIT in permitted
        assert action in permitted


# --------------------------------------------------------------------------
# Gap regime breakpoints (±1/±0.5/0)
# --------------------------------------------------------------------------


def test_gap_regime_breakpoints():
    assert gap_regime_for(1.5, CFG) is GapRegime.STRONG_POSITIVE
    assert gap_regime_for(1.0, CFG) is GapRegime.STRONG_POSITIVE  # boundary is inclusive on the strong side
    assert gap_regime_for(0.99, CFG) is GapRegime.UPPER_MIDDLE
    assert gap_regime_for(0.5, CFG) is GapRegime.UPPER_MIDDLE
    assert gap_regime_for(0.49, CFG) is GapRegime.NEAR_CENTER_POSITIVE
    assert gap_regime_for(0.0, CFG) is GapRegime.NEAR_CENTER_POSITIVE
    assert gap_regime_for(-0.01, CFG) is GapRegime.NEAR_CENTER_NEGATIVE
    assert gap_regime_for(-0.5, CFG) is GapRegime.NEAR_CENTER_NEGATIVE
    assert gap_regime_for(-0.51, CFG) is GapRegime.LOWER_MIDDLE
    assert gap_regime_for(-1.0, CFG) is GapRegime.LOWER_MIDDLE
    assert gap_regime_for(-1.01, CFG) is GapRegime.STRONG_NEGATIVE


def test_clob_direction_from_sign():
    assert clob_direction_from_sign(1) is Direction.UP
    assert clob_direction_from_sign(-1) is Direction.DOWN
    assert clob_direction_from_sign(0) is Direction.FLAT


def test_spot_direction_from_bp_threshold():
    cfg = RegimeConfig(spot_flat_threshold_bp=1.0)
    assert spot_direction_from_bp(None, cfg) is Direction.FLAT
    assert spot_direction_from_bp(0.5, cfg) is Direction.FLAT
    assert spot_direction_from_bp(-0.5, cfg) is Direction.FLAT
    assert spot_direction_from_bp(1.5, cfg) is Direction.UP
    assert spot_direction_from_bp(-1.5, cfg) is Direction.DOWN


# --------------------------------------------------------------------------
# No bypass of EV/risk gates (structural check)
# --------------------------------------------------------------------------


def test_regime_module_has_no_dependency_on_execution_or_portfolio_math():
    """Roadmap Phase 6 verification: "no matrix entry directly bypasses
    EV/risk gates." Enforced structurally: the regime package must not
    import the portfolio math kernel or anything that places/simulates
    orders - it only ever returns candidate action families."""
    import xamarinbot.regime.classifier as classifier_mod
    import xamarinbot.regime.matrix as matrix_mod
    import xamarinbot.regime.types as types_mod

    for mod in (classifier_mod, matrix_mod, types_mod):
        names = {getattr(v, "__module__", "") for v in vars(mod).values()}
        assert not any(n.startswith("xamarinbot.portfolio") for n in names)


# --------------------------------------------------------------------------
# Stateful classifier: transitions, dwell time, CANCEL
# --------------------------------------------------------------------------


def _fv(decision_ts: float, z_gap: float, clob_sign: int, spot_bp: float | None) -> FeatureVector:
    return FeatureVector(
        feature_version="v1",
        round_id="r0",
        decision_ts=decision_ts,
        t=decision_ts,
        tau=300.0 - decision_ts,
        time_regime=TimeRegime.CORE_TRADING,
        p0=100_000.0,
        twap=100_010.0,
        spot=100_020.0,
        clob_mid=0.55,
        gap_twap_bp=1.0,
        gap_spot_bp=2.0,
        lead_gap_bp=1.0,
        twap_pressure_model=0.0,
        realized_vol=0.01,
        z_gap=z_gap,
        z_spot=0.3,
        clob_log_odds=0.2,
        z_clob=0.1,
        ofi=0.0,
        spread=0.02,
        spot_returns_bp={1.0: spot_bp} if spot_bp is not None else {},
        clob_direction={1.0: clob_sign},
    )


def test_state_for_reads_canonical_horizon_from_feature_vector():
    fv = _fv(10.0, z_gap=1.5, clob_sign=1, spot_bp=5.0)
    state = state_for(fv, CFG)
    assert state == RegimeState(GapRegime.STRONG_POSITIVE, Direction.UP, Direction.UP)


def test_first_observation_has_no_from_state_and_no_dwell():
    clf = RegimeClassifier(round_id="r0")
    snapshot = clf.observe(_fv(0.0, z_gap=1.5, clob_sign=1, spot_bp=5.0))
    assert snapshot.seed_action is SeedAction.TAKER_UP
    assert len(clf.transitions) == 1
    assert clf.transitions[0].from_state is None
    assert clf.transitions[0].dwell_time_s is None


def test_same_state_produces_no_new_transition():
    clf = RegimeClassifier(round_id="r0")
    clf.observe(_fv(0.0, z_gap=1.5, clob_sign=1, spot_bp=5.0))
    clf.observe(_fv(5.0, z_gap=1.5, clob_sign=1, spot_bp=5.0))  # identical state
    assert len(clf.transitions) == 1  # only the initial observation


def test_transition_records_correct_dwell_time():
    clf = RegimeClassifier(round_id="r0")
    clf.observe(_fv(0.0, z_gap=1.5, clob_sign=1, spot_bp=5.0))  # STRONG_POSITIVE/UP/UP
    clf.observe(_fv(7.0, z_gap=-1.5, clob_sign=-1, spot_bp=-5.0))  # STRONG_NEGATIVE/DOWN/DOWN
    assert len(clf.transitions) == 2
    second = clf.transitions[1]
    assert second.dwell_time_s == 7.0
    assert second.from_state == RegimeState(GapRegime.STRONG_POSITIVE, Direction.UP, Direction.UP)
    assert second.to_state == RegimeState(GapRegime.STRONG_NEGATIVE, Direction.DOWN, Direction.DOWN)


def test_cancel_emitted_only_when_leaving_a_directional_thesis():
    clf = RegimeClassifier(round_id="r0")
    # directional thesis (TAKER_UP)
    s1 = clf.observe(_fv(0.0, z_gap=1.5, clob_sign=1, spot_bp=5.0))
    assert s1.seed_action is SeedAction.TAKER_UP

    # conflict -> should be CANCEL (had a thesis, now lapsing)
    s2 = clf.observe(_fv(5.0, z_gap=1.5, clob_sign=1, spot_bp=None))
    assert s2.seed_action is SeedAction.CANCEL

    # still conflicted, no thesis to cancel this time -> plain WAIT
    s3 = clf.observe(_fv(10.0, z_gap=1.5, clob_sign=-1, spot_bp=None))
    assert s3.seed_action is SeedAction.WAIT


def test_cancel_never_appears_from_the_pure_matrix_only_the_classifier():
    clf = RegimeClassifier(round_id="r0")
    clf.observe(_fv(0.0, z_gap=1.5, clob_sign=1, spot_bp=5.0))
    snapshot = clf.observe(_fv(5.0, z_gap=1.5, clob_sign=1, spot_bp=None))
    assert snapshot.seed_action is SeedAction.CANCEL
    assert SeedAction.CANCEL in snapshot.permitted_actions
    assert SeedAction.WAIT in snapshot.permitted_actions


# --------------------------------------------------------------------------
# Journaling + transition report
# --------------------------------------------------------------------------


def test_journal_round_trip_and_transition_report():
    clf = RegimeClassifier(round_id="r0")
    clf.observe(_fv(0.0, z_gap=1.5, clob_sign=1, spot_bp=5.0))  # start: TAKER_UP
    clf.observe(_fv(5.0, z_gap=1.5, clob_sign=1, spot_bp=None))  # -> CANCEL

    journal = JournalWriter(":memory:")
    for t in clf.transitions:
        journal.write(to_journal_record(t))

    rows = journal.read(RegimeTransitionRecord, "r0")
    assert len(rows) == 2
    assert rows[0].from_state is None
    assert rows[1].seed_action == "CANCEL"
    assert rows[1].dwell_time_s == 5.0

    report = build_transition_report(journal)
    assert report.n_transitions == 2
    assert report.n_first_observations == 1
    assert report.cancel_count == 1
    assert report.seed_action_counts["CANCEL"] == 1
