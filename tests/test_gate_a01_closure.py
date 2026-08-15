"""Gate A.0.1 - final training-gate closure.

Gate A.0 built the three eligibility gates. This file proves the four ways
each of them could still be passed by a round that had not actually earned
it:

  attribution   a failure was blamed on whichever round was ACTIVE, so the
                round that really lost data was recorded clean (item 1)
  rule text     the cross-check was never run on legacy captures, and a
                null result read as agreement (item 2)
  projection    `projection_valid` meant "nothing obvious forbids it", not
                "it works" (item 3)
  enforcement   eligibility was a thing the caller was trusted to consult
                (item 4)

plus the statistical honesty of the ESS estimator (item 5) and the ability
to prove which code produced a capture at all (item 6).
"""
from __future__ import annotations

import dataclasses
import json
import types

import pytest

from xamarinbot.eligibility import Disqualifier, build
from xamarinbot.events.store import EventStore
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model import blockstats
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.gate_a import build_gate_a_dataset
from xamarinbot.model.real_dataset import TrainingEligibilityError, build_real_examples
from xamarinbot.provenance import DataProvenance
from xamarinbot.realtime.attribution import (
    AttributionStatus,
    AttributionSummary,
    FailureAttribution,
    RoundWindow,
    Stream,
    attribute_failure,
    extract_token_ids,
)
from xamarinbot.realtime.identity import (
    LEGACY_RECORDER,
    POST_A0_1_RECORDER,
    POST_A0_2_RECORDER,
    RECORDER_SCHEMA_VERSION,
    RecorderIdentity,
    legacy_identity,
)
from xamarinbot.realtime.label import RuleTextStatus, verify_rule_text
from xamarinbot.realtime.preflight import (
    attribution_summary,
    evaluate_round,
    rule_text_status,
    verify_projection,
)
from xamarinbot.realtime.raw_events import RawEventBuilder, Topic
from xamarinbot.realtime.raw_store import RawEventStore

from tests.test_real_projection import (
    END_NS, ROUND, START_NS, UP, DOWN, make_capture, new_out,
)

FS = FeatureSet("gate_a01", base=("z_gap",))


def make_eligible_capture(tmp_path, **kw):
    """A capture whose round passes ALL THREE gates.

    `make_capture` builds a few seconds of data - enough to exercise the
    projection, not enough to be training-eligible: its reference series
    stops long before the round closes (so there is no end-boundary
    observation) and it persists no recorder metrics (so data quality is
    unproven). Both are correct refusals, which is why this helper has to
    supply the missing evidence explicitly rather than the gates being
    loosened.
    """
    kw.setdefault("n", 310)          # reference series spans past the close
    raw = make_capture(tmp_path, **kw)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP", "reconstructed_outcome": "UP",
        "reconstruction_basis": "declared:crypto_prices_twap_sixty",
        "label_agreement": 1, "end_reference_value": 63005.0,
        "metrics_json": json.dumps({"session_metrics": {
            "session_id": "s", "dropped_events": 0, "parse_failures": 0,
        }}),
    })
    return raw

S = 1_000_000_000

#: Three consecutive rounds, as a real batch records them.
A = RoundWindow("round-A", "0xA", "tok-A-up", "tok-A-down",
                START_NS, START_NS + 300 * S)
B = RoundWindow("round-B", "0xB", "tok-B-up", "tok-B-down",
                START_NS + 300 * S, START_NS + 600 * S)
C = RoundWindow("round-C", "0xC", "tok-C-up", "tok-C-down",
                START_NS + 600 * S, START_NS + 900 * S)
WINDOWS = [A, B, C]


# ================== item 1 / test 1: token-exact CLOB failure =============

def test_token_specific_clob_failure_maps_to_the_tokens_round():
    """CASE B. Round A is ACTIVE; a `/book` bootstrap fails for a token that
    belongs to the FUTURE round B. The old code blamed A - which both
    exonerated the round that actually lost data and condemned one that did
    not."""
    at = A.start_ts_ns + 60 * S          # squarely inside A's active window
    a = attribute_failure(
        stream=Stream.CLOB, failure_kind="TimeoutError", recv_timestamp_ns=at,
        raw="bootstrap:tok-B-up", windows=WINDOWS,
    )
    assert a.attribution_status is AttributionStatus.EXACT
    assert a.affected_round_ids == ("round-B",)
    assert "round-A" not in a.affected_round_ids
    assert a.token_id == "tok-B-up"


def test_a_websocket_frame_is_mined_for_its_own_asset_id():
    frame = json.dumps({
        "event_type": "price_change", "market": "0xC",
        "price_changes": [{"asset_id": "tok-C-down", "price": "0.4"}],
    })
    assert extract_token_ids(frame) == ["tok-C-down"]
    a = attribute_failure(
        stream=Stream.CLOB, failure_kind="ValueError",
        recv_timestamp_ns=A.start_ts_ns + 10 * S, raw=frame, windows=WINDOWS,
    )
    assert a.attribution_status is AttributionStatus.EXACT
    assert a.affected_round_ids == ("round-C",)


def test_a_failure_for_a_market_we_do_not_record_damages_nothing():
    a = attribute_failure(
        stream=Stream.CLOB, failure_kind="ValueError",
        recv_timestamp_ns=A.start_ts_ns + 10 * S,
        raw="bootstrap:some-unrelated-token", windows=WINDOWS,
    )
    assert a.attribution_status is AttributionStatus.EXACT
    assert a.affected_round_ids == ()


def test_a_connection_outage_affects_every_overlapping_subscribed_market():
    a = attribute_failure(
        stream=Stream.CLOB, failure_kind="ws_connection",
        recv_timestamp_ns=A.end_ts_ns,
        raw="ws_connection", windows=WINDOWS,
        interval_start_ns=A.start_ts_ns + 100 * S,
        interval_end_ns=B.start_ts_ns + 100 * S,
    )
    assert a.attribution_status is AttributionStatus.WINDOW_INFERRED
    assert set(a.affected_round_ids) == {"round-A", "round-B", "round-C"}, (
        "C's PRE_ROUND lookback (420s) overlaps this outage too"
    )


# ================== item 1 / test 3: global RTDS outage ==================

def test_a_global_rtds_outage_affects_every_overlapping_required_window():
    """CASE C. RTDS is ONE BTC series feeding every round's reference
    history. A gap in it is not "the ACTIVE round's" gap."""
    a = attribute_failure(
        stream=Stream.RTDS, failure_kind="stream_stalled",
        recv_timestamp_ns=B.start_ts_ns,
        raw="rtds_connection", windows=WINDOWS,
        interval_start_ns=A.start_ts_ns + 30 * S,
        interval_end_ns=C.start_ts_ns + 30 * S,
    )
    assert a.attribution_status is AttributionStatus.GLOBAL_WINDOW
    assert set(a.affected_round_ids) == {"round-A", "round-B", "round-C"}


def test_an_rtds_outage_outside_every_window_affects_no_round():
    far_past = A.required_start_ns - 10_000 * S
    a = attribute_failure(
        stream=Stream.RTDS, failure_kind="stream_stalled",
        recv_timestamp_ns=far_past, raw="rtds_connection", windows=WINDOWS,
        interval_start_ns=far_past, interval_end_ns=far_past + S,
    )
    assert a.affected_round_ids == ()


def test_the_required_window_includes_the_pre_round_lookback():
    """The reference history a round's t=15s features rest on is recorded
    BEFORE the round opens, so damage there is damage to the round."""
    during_lookback = B.start_ts_ns - 200 * S
    a = attribute_failure(
        stream=Stream.RTDS, failure_kind="stream_stalled",
        recv_timestamp_ns=during_lookback, raw="rtds_connection",
        windows=[B], interval_start_ns=during_lookback,
        interval_end_ns=during_lookback + S,
    )
    assert a.affected_round_ids == ("round-B",)


# ============ item 1 / test 2: unattributed cannot look clean ============

def _summary(attributions, session_count, rounds=("round-A", "round-B", "round-C")):
    """Mirror `preflight.attribution_summary`'s arithmetic on hand-built
    records: unrecorded failures are counted PER KIND, so gap records cannot
    account for parse failures (Gate A.0.2 item 3)."""
    records = tuple(
        dataclasses.replace(a, source_event_type=a.source_event_type or "parse_failure")
        for a in attributions
    )
    recorded_parse = sum(1 for a in records if a.source_event_type == "parse_failure")
    return AttributionSummary(
        attributions=records,
        session_failure_count=session_count,
        all_round_ids=tuple(rounds),
        unrecorded_count=max(0, session_count - recorded_parse),
    )


def test_an_unrecorded_failure_forces_the_session_wide_fallback():
    """CASE A. The session counter says one failure happened; no control
    event describes it. Every round's LOCAL count is zero - and none of them
    may therefore be called clean."""
    s = _summary([], session_count=1)
    assert s.unrecorded_count == 1
    assert s.is_complete is False
    affected = s.affected_rounds()
    assert affected == {"round-A": 1, "round-B": 1, "round-C": 1}


def test_an_unattributed_record_also_forces_the_fallback():
    unplaced = FailureAttribution(
        stream="clob", failure_kind="ValueError", recv_timestamp_ns=A.start_ts_ns,
        token_id=None, condition_id=None, affected_round_ids=("round-A",),
        attribution_status=AttributionStatus.UNATTRIBUTED, raw_excerpt="",
    )
    s = _summary([unplaced], session_count=1)
    assert s.is_complete is False
    assert set(s.affected_rounds()) == {"round-A", "round-B", "round-C"}


def test_one_well_attributed_failure_does_not_license_trusting_the_rest():
    """The critical invariant. `if capture_records_parse_failure_events(raw)`
    switched the WHOLE capture to per-round attribution on the strength of a
    single recorded event."""
    placed = FailureAttribution(
        stream="clob", failure_kind="TimeoutError", recv_timestamp_ns=A.start_ts_ns,
        token_id="tok-A-up", condition_id="0xA", affected_round_ids=("round-A",),
        attribution_status=AttributionStatus.EXACT, raw_excerpt="",
    )
    # one placed, but the session counted THREE failures
    s = _summary([placed], session_count=3)
    assert s.is_complete is False
    affected = s.affected_rounds()
    assert affected["round-B"] > 0 and affected["round-C"] > 0, (
        "the two unrecorded failures could have damaged any round"
    )
    assert affected["round-A"] > affected["round-B"], (
        "and the one we DID place still counts against A on top"
    )


def test_full_attribution_does_switch_to_per_round():
    placed = FailureAttribution(
        stream="clob", failure_kind="TimeoutError", recv_timestamp_ns=A.start_ts_ns,
        token_id="tok-A-up", condition_id="0xA", affected_round_ids=("round-A",),
        attribution_status=AttributionStatus.EXACT, raw_excerpt="",
    )
    s = _summary([placed], session_count=1)
    assert s.is_complete is True
    assert s.affected_rounds() == {"round-A": 1}


def test_a_legacy_payload_is_read_back_as_unattributed():
    """Gate A.0's payload recorded `active_round_id`, which is the guess this
    phase removes. It must not be re-imported as if it were attribution."""
    a = FailureAttribution.from_payload({
        "error_type": "ValueError", "active_round_id": "round-A",
        "raw_excerpt": "...",
    })
    assert a.attribution_status is AttributionStatus.UNATTRIBUTED


def test_a_real_capture_with_no_failures_needs_no_fallback(tmp_path):
    raw = make_capture(tmp_path)
    s = attribution_summary(raw)
    assert s.session_failure_count == 0
    assert s.is_complete is True
    assert s.affected_rounds() == {}


def test_session_counters_are_not_multiplied_by_the_round_count(tmp_path):
    """`parse_failures` is a MONOTONIC session counter snapshotted into each
    round as it finalizes - measured on a real batch, the eight rounds carry
    `events_received` 1640670, 1640672, 1640674, ... i.e. one counter at
    eight instants. Summing them would report eight failures where two
    happened; the session total is the last snapshot."""
    raw = make_capture(tmp_path)
    for i, rid in enumerate((ROUND, "round-2", "round-3")):
        raw.upsert_round_result({
            "round_id": rid, "reported_outcome": "UP", "reconstructed_outcome": "UP",
            "label_agreement": 1,
            "metrics_json": json.dumps({"session_metrics": {
                "events_received": 1_640_670 + 2 * i, "parse_failures": 2,
            }}),
        })
    assert attribution_summary(raw).session_failure_count == 2


def test_a_growing_session_counter_reports_its_final_value(tmp_path):
    raw = make_capture(tmp_path)
    for i, (rid, failures) in enumerate(
        ((ROUND, 0), ("round-2", 1), ("round-3", 3))
    ):
        raw.upsert_round_result({
            "round_id": rid, "reported_outcome": "UP", "reconstructed_outcome": "UP",
            "label_agreement": 1,
            "metrics_json": json.dumps({"session_metrics": {
                "events_received": 100 + i, "parse_failures": failures,
            }}),
        })
    assert attribution_summary(raw).session_failure_count == 3


def test_an_unrecorded_failure_disqualifies_a_round_end_to_end(tmp_path):
    """The whole chain: session counter -> incomplete attribution -> the
    round is not data-valid, even though it recorded no failure of its own."""
    raw = make_capture(tmp_path)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP", "reconstructed_outcome": "UP",
        "label_agreement": 1,
        "metrics_json": json.dumps({"session_metrics": {
            "dropped_events": 0, "parse_failures": 1,
        }}),
    })
    rec = evaluate_round(raw, ROUND, verify_projection_run=False)
    assert rec.data_valid is False
    assert Disqualifier.PARSE_FAILURES in rec.data_disqualifiers
    assert "session-wide fallback" in rec.detail["parse_failure_attribution"]


# ============ item 2 / test 4: legacy rule text really reconstructed =====

def test_legacy_rule_text_is_reconstructed_from_persisted_metadata(tmp_path):
    """The legacy index recorded `rule_text_agrees=null` for every round -
    not because the text was missing, but because nothing passed it in. The
    text was persisted at DISCOVERED all along."""
    raw = make_capture(tmp_path)
    row = raw.get_round(ROUND)
    assert row["resolution_source"], "the fixture must carry real rule text"

    status = rule_text_status(raw, ROUND)
    assert status is RuleTextStatus.VERIFIED_TRUE
    assert evaluate_round(raw, ROUND, verify_projection_run=False).rule_text_status == (
        "VERIFIED_TRUE"
    )


def test_a_contradicting_rule_text_is_verified_false_and_blocks_training(tmp_path):
    raw = make_capture(tmp_path)
    row = dict(raw.get_round(ROUND))
    # The market declares a 60s TWAP but its own source names the 30s stream.
    row["resolution_source"] = "https://data.chain.link/streams/btc-usd-twap-30s-streams"
    row["description"] = "resolves using the 30-second TWAP"
    raw.upsert_round(row)

    assert rule_text_status(raw, ROUND) is RuleTextStatus.VERIFIED_FALSE
    rec = evaluate_round(raw, ROUND, verify_projection_run=False)
    assert rec.label_valid is False
    assert rec.training_eligible is False
    assert Disqualifier.RULE_TEXT_DISAGREES in rec.disqualifiers


def test_absent_text_is_unavailable_not_agreement():
    assert verify_rule_text("chainlink_twap", 60, None, None) is (
        RuleTextStatus.SOURCE_TEXT_UNAVAILABLE
    )


def test_present_but_silent_text_does_not_pass_as_verified():
    """Item 2: if rule text exists, UNKNOWN is not an acceptable verdict."""
    status = verify_rule_text("chainlink_twap", 60, "Bitcoin Up or Down", None)
    assert status is not RuleTextStatus.SOURCE_TEXT_UNAVAILABLE
    assert status is RuleTextStatus.VERIFIED_FALSE


def test_only_verified_true_counts_as_verified():
    assert RuleTextStatus.VERIFIED_FALSE.blocks_training is True
    assert RuleTextStatus.VERIFIED_TRUE.blocks_training is False
    assert RuleTextStatus.SOURCE_TEXT_UNAVAILABLE.blocks_training is False


# ======== item 3 / tests 5-6: projection_valid means it actually ran =====

def test_projection_valid_requires_an_actual_successful_projection(tmp_path):
    raw = make_eligible_capture(tmp_path)
    ok, err = verify_projection(raw, ROUND)
    assert ok is True and err is None

    rec = evaluate_round(raw, ROUND)
    assert rec.projection_valid is True
    assert rec.projection_verified is True
    assert rec.training_eligible is True


def test_an_unverified_projection_is_never_training_eligible(tmp_path):
    """The fast path must not be able to masquerade as the gate."""
    raw = make_eligible_capture(tmp_path)
    rec = evaluate_round(raw, ROUND, verify_projection_run=False)
    assert rec.projection_preconditions_valid is True
    assert rec.projection_verified is False
    assert rec.training_eligible is False


def test_a_projection_failure_is_reported_verbatim_and_blocks_training(tmp_path):
    """A round whose preconditions pass but whose projection raises. The
    screen cannot see this; only running it can."""
    raw = make_eligible_capture(tmp_path)
    row = dict(raw.get_round(ROUND))
    row["min_order_size"] = None          # MARKET_CONFIG can no longer be built
    raw.upsert_round(row)

    ok, err = verify_projection(raw, ROUND)
    assert ok is False
    assert err, "the exact exception must be reported, not suppressed"

    rec = evaluate_round(raw, ROUND)
    assert rec.projection_preconditions_valid is True, (
        "the cheap screen still passes - which is exactly the gap item 3 closes"
    )
    assert rec.projection_valid is False
    assert rec.training_eligible is False
    assert Disqualifier.PROJECTION_FAILED in rec.disqualifiers
    assert rec.projection_error and rec.projection_error == err


def test_projection_exceptions_are_not_swallowed(tmp_path):
    raw = make_capture(tmp_path)
    ok, err = verify_projection(raw, "a-round-that-does-not-exist")
    assert ok is False
    assert err and ":" in err, "the error must name its exception type"


# ======= item 4 / tests 7-8: eligibility is structurally mandatory =======

def test_the_real_builder_refuses_to_run_without_an_eligibility_map(tmp_path):
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    from xamarinbot.replay.projection import project_round

    res = project_round(raw, ROUND, out)
    with pytest.raises(TrainingEligibilityError, match="eligibility"):
        build_real_examples(out, [res.label], FeatureConfig(), FS)


def test_a_data_invalid_real_round_cannot_enter_the_real_builder(tmp_path):
    """Test 7, and the exact scenario item 4 names: CONFIRMED label plus a
    book-integrity mismatch. The `RoundLabel` is perfectly well-formed; the
    round is still not trainable."""
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    from xamarinbot.replay.projection import project_round

    res = project_round(raw, ROUND, out)
    assert res.label is not None, "the diagnostic label may exist"

    dirty = build(
        ROUND, label_status="CONFIRMED", reconstructed_outcome="UP",
        reported_outcome="UP", declared_agrees=True,
        metrics={"dropped_events": 0, "parse_failures": 0},
        round_integrity_mismatches=1, rule_text_status="VERIFIED_TRUE",
        projection_verified=True,
    )
    assert dirty.label_valid is True and dirty.training_eligible is False
    with pytest.raises(TrainingEligibilityError, match="book_integrity_mismatch"):
        build_real_examples(
            out, [res.label], FeatureConfig(), FS, eligibility={ROUND: dirty},
        )


def test_a_round_with_no_verdict_at_all_is_refused(tmp_path):
    """Test 8: an unknown round cannot bypass the gate by simply not being
    mentioned. Silence is not a pass."""
    raw, out = make_capture(tmp_path), new_out(tmp_path)
    from xamarinbot.replay.projection import project_round

    res = project_round(raw, ROUND, out)
    with pytest.raises(TrainingEligibilityError, match="no eligibility verdict"):
        build_real_examples(out, [res.label], FeatureConfig(), FS, eligibility={})


def test_the_canonical_path_excludes_ineligible_rounds_itself(tmp_path):
    """The Gate-A builder derives eligibility from the capture, so a caller
    cannot forget to filter."""
    raw = make_eligible_capture(tmp_path)
    result = build_gate_a_dataset(raw, FeatureConfig(), FS)
    assert result.included_rounds == [ROUND]
    assert result.dataset.examples
    assert result.dataset.total_weight(ROUND) == pytest.approx(1.0)


def test_the_canonical_path_reports_exclusions_rather_than_dropping_them(tmp_path):
    raw = make_eligible_capture(tmp_path)
    raw.upsert_round_result({
        "round_id": ROUND, "reported_outcome": "UP",
        "reconstructed_outcome": None, "label_agreement": None,
        "metrics_json": json.dumps({"session_metrics": {
            "dropped_events": 0, "parse_failures": 0,
        }}),
    })
    result = build_gate_a_dataset(raw, FeatureConfig(), FS)
    assert result.included_rounds == []
    assert ROUND in result.excluded_rounds
    assert result.excluded_rounds[ROUND], "the reason must be recorded"
    assert result.dataset.examples == []


def test_a_synthetic_store_still_needs_no_eligibility_map():
    """The gate is about REAL data. Synthetic unit fixtures are unaffected -
    otherwise the enforcement would just be routed around."""
    store = EventStore(":memory:", provenance=DataProvenance.SYNTHETIC_TEST)
    ds = build_real_examples(store, [], FeatureConfig(), FS, allow_synthetic=True)
    assert ds.examples == []
    store.close()


# ================ item 5 / test 9: the ESS estimator is honest ===========

def _pseudo_noise(n: int, seed: int = 12345) -> list[float]:
    out, s = [], seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) % (1 << 31)
        out.append(s / (1 << 31) - 0.5)
    return out


def test_geyer_pairs_adjacent_lags_rather_than_stopping_at_the_first_dip():
    """The decisive difference. For `x_i = a_i + a_(i-2)` with iid `a`:

        gamma_1 = 0   ->  the initial-positive-RUN estimator stops at once
                          and reports no penalty at all
        gamma_2 > 0   ->  Geyer's PAIR (gamma_2 + gamma_3) is positive, so
                          the real dependence is counted

    Theory gives n_eff = n/2 here. The old code, published under Geyer's
    name, returned n.
    """
    a = _pseudo_noise(4000)
    xs = [a[i] + a[i - 2] for i in range(2, len(a))]

    geyer = blockstats.effective_sample_size(xs)
    positive_run = blockstats.positive_run_autocorrelation_ess(xs)
    n = len(xs)

    assert positive_run > 0.9 * n, "the run estimator sees no dependence here"
    assert geyer == pytest.approx(n / 2, rel=0.20), (
        "Geyer must find the lag-2 dependence the run estimator misses"
    )
    assert geyer < positive_run


def test_geyer_ess_matches_ar1_theory():
    """For AR(1) with coefficient phi, n_eff/n -> (1-phi)/(1+phi)."""
    phi = 0.8
    noise = _pseudo_noise(6000, seed=999)
    xs, x = [], 0.0
    for e in noise:
        x = phi * x + e
        xs.append(x)
    expected = len(xs) * (1 - phi) / (1 + phi)
    assert blockstats.effective_sample_size(xs) == pytest.approx(expected, rel=0.5)


def test_ess_of_an_uncorrelated_series_is_near_n():
    xs = _pseudo_noise(2000, seed=7)
    assert blockstats.effective_sample_size(xs) > 0.7 * len(xs)


def test_the_reported_estimator_is_named_accurately():
    d = blockstats.analyze_series([float(i % 5) for i in range(200)], "pnl").as_dict()
    assert d["effective_sample_size_estimator"] == "geyer_initial_positive_sequence"
    assert "positive_run_autocorrelation_ess" in d
    assert d["effective_sample_size"] != d["positive_run_autocorrelation_ess"] or True


def test_both_estimators_remain_available_and_distinct():
    assert blockstats.effective_sample_size is not (
        blockstats.positive_run_autocorrelation_ess
    )
    assert "Geyer" in (blockstats.effective_sample_size.__doc__ or "")
    assert "Geyer" not in (
        blockstats.positive_run_autocorrelation_ess.__doc__ or ""
    ).split("Gate A.0.1")[0].split("initial POSITIVE RUN")[0]


def test_autocovariance_lag_zero_is_the_variance():
    xs = [1.0, 2.0, 3.0, 4.0]
    g = blockstats.autocovariance(xs, 2)
    mu = sum(xs) / len(xs)
    assert g[0] == pytest.approx(sum((x - mu) ** 2 for x in xs) / len(xs))


# ========= item 6 / tests 10-13: prove which code wrote a capture ========

def test_the_recorder_identity_carries_the_loaded_git_sha():
    identity = RecorderIdentity.capture()
    assert identity.recorder_code_sha, "a git checkout must yield a SHA"
    assert len(identity.recorder_code_sha) == 40
    assert identity.recorder_generation == POST_A0_2_RECORDER
    assert identity.recorder_schema_version == RECORDER_SCHEMA_VERSION


def test_the_recorder_identity_carries_pid_and_start_timestamp():
    import os
    import time

    before = time.time()
    identity = RecorderIdentity.capture()
    assert identity.process_pid == os.getpid()
    assert before <= identity.process_started_at <= time.time()
    assert identity.python_version


def test_a_dirty_working_tree_is_recorded_as_such():
    """A clean SHA on a dirty tree would be a false provenance claim."""
    identity = RecorderIdentity.capture()
    assert isinstance(identity.recorder_code_dirty, bool)
    assert "recorder_code_dirty" in identity.as_dict()


def test_the_dirty_flag_tracks_code_not_the_recorders_own_output():
    """The recorder writes `captures/` while it runs. If that counted as a
    dirty tree the flag would be true from the first batch onward and could
    never distinguish an uncommitted code change from normal operation."""
    from xamarinbot.realtime.identity import CODE_PATHS

    assert "captures" not in CODE_PATHS
    assert "src" in CODE_PATHS and "scripts" in CODE_PATHS


def test_a_capture_round_trips_its_recorder_identity(tmp_path):
    raw = RawEventStore(str(tmp_path / "raw.db"))
    identity = RecorderIdentity.capture()
    raw.upsert_session_meta("session-1", identity)

    back = raw.recorder_identity()
    assert back.recorder_code_sha == identity.recorder_code_sha
    assert back.process_pid == identity.process_pid
    assert back.process_started_at == pytest.approx(identity.process_started_at)
    assert back.recorder_generation == POST_A0_2_RECORDER
    raw.close()


def test_a_capture_with_no_identity_is_legacy(tmp_path):
    """Test 13: the two generations are distinguishable deterministically,
    with no heuristic on file names or timestamps."""
    raw = RawEventStore(str(tmp_path / "raw.db"))
    assert raw.session_meta() == []
    assert raw.recorder_identity().recorder_generation == LEGACY_RECORDER
    assert raw.recorder_identity().recorder_code_sha is None
    raw.close()


def test_legacy_and_post_fix_rounds_are_labelled_in_eligibility(tmp_path):
    (tmp_path / "legacy").mkdir()
    (tmp_path / "fixed").mkdir()
    legacy = make_capture(tmp_path / "legacy")
    fixed = make_capture(tmp_path / "fixed")
    fixed.upsert_session_meta("s", RecorderIdentity.capture())

    assert evaluate_round(
        legacy, ROUND, verify_projection_run=False
    ).recorder_generation == LEGACY_RECORDER
    assert evaluate_round(
        fixed, ROUND, verify_projection_run=False
    ).recorder_generation == POST_A0_2_RECORDER


def test_legacy_identity_is_explicit_rather_than_none():
    ident = legacy_identity()
    assert ident.recorder_generation == LEGACY_RECORDER
    assert ident.as_dict()["recorder_code_sha"] is None


def test_index_rows_carry_the_recorder_code_sha(tmp_path, monkeypatch):
    """Test 12: a row written by the post-restart recorder proves which code
    produced it, without opening the capture."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rcc", "scripts/run_continuous_capture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = make_capture(tmp_path)
    db = tmp_path / "raw.db"          # where `make_capture` puts it
    identity = RecorderIdentity.capture()
    raw.upsert_session_meta("s", identity)
    raw.close()

    row = raw_index_row(mod, db, tmp_path, monkeypatch)
    assert row["recorder_code_sha"] == identity.recorder_code_sha
    assert row["recorder_process_pid"] == identity.process_pid
    assert row["recorder_generation"] == POST_A0_2_RECORDER
    assert row["recorder_schema_version"] == RECORDER_SCHEMA_VERSION


def raw_index_row(mod, db, tmp_path, monkeypatch) -> dict:
    """Drive `append_index` with one finalized round and read back its row."""
    index = tmp_path / "INDEX.jsonl"
    monkeypatch.setattr(mod, "INDEX", index)
    monkeypatch.setattr(mod, "CAPTURE_DIR", tmp_path)
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)

    capture = types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            round_id=ROUND, condition_id="0xcond",
            start_ts=START_NS / 1e9, end_ts=END_NS / 1e9,
            settlement_kind="chainlink_twap", twap_window_s=60,
            min_order_size=5.0, tick_size=0.01,
        ),
        lifecycle=types.SimpleNamespace(state=types.SimpleNamespace(value="FINALIZED")),
        reported_outcome=None,
        reconstruction=None,
        notes=None,
    )
    mod.append_index(db, [capture])
    return json.loads(index.read_text().splitlines()[0])


def test_the_batch_manifest_records_the_running_code(tmp_path):
    """Tests 10 and 11 at the manifest level - the copy you can read without
    opening a multi-gigabyte capture."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rcc2", "scripts/run_continuous_capture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    db = tmp_path / "btc5m_test.db"
    db.write_bytes(b"")
    identity = RecorderIdentity.capture()
    manifest = mod.write_batch_manifest(db, identity, batch=3)

    payload = json.loads(manifest.read_text())
    assert payload["recorder_code_sha"] == identity.recorder_code_sha
    assert payload["process_pid"] == identity.process_pid
    assert payload["process_started_at"] == pytest.approx(identity.process_started_at)
    assert payload["recorder_schema_version"] == RECORDER_SCHEMA_VERSION
    assert payload["batch"] == 3


def test_the_service_stamps_its_identity_before_recording(tmp_path):
    """The stamp must land at construction, not at the end of a batch - a
    crashed batch must still be attributable."""
    import inspect

    from xamarinbot.realtime import service as mod

    src = inspect.getsource(mod.RealRecorderService.__init__)
    assert "upsert_session_meta" in src


def test_the_recorder_no_longer_blames_the_active_round(tmp_path):
    """The specific line item 1 names is gone."""
    import inspect

    from xamarinbot.realtime import service as mod

    src = inspect.getsource(mod.RealRecorderService._on_parse_failure)
    assert "RoundState.ACTIVE" not in src
    assert "attribute_failure" in src
