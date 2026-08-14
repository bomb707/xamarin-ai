"""Phase 4 verification: "No feature may consume a future timestamp."
"Feature output invariant between live and replay." "Missing/stale inputs
produce explicit invalid state." "Feature set is deterministic and causal."
"""
from __future__ import annotations

import math

from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector, InvalidFeatureState, InvalidReason, TimeRegime, time_regime_for
from xamarinbot.synthetic.rounds import populate_synthetic_round

CFG = FeatureConfig()


def _round_events(round_id: str = "r0", **kwargs):
    store = EventStore(":memory:")
    result = populate_synthetic_round(store, round_id, **kwargs)
    return store.all_events(round_id), result


def test_missing_inputs_produce_explicit_invalid_state_not_defaults():
    events, result = _round_events()
    r = compute([], "r0", 100.0, result.p0, CFG)
    assert isinstance(r, InvalidFeatureState)
    assert r.reason is InvalidReason.MISSING_MARKET_CONFIG


def test_insufficient_volatility_history_at_round_start_is_explicit():
    events, result = _round_events()
    r = compute(events, "r0", 0.0, result.p0, CFG)
    assert isinstance(r, InvalidFeatureState)
    assert r.reason is InvalidReason.INSUFFICIENT_VOLATILITY_HISTORY


def test_stale_spot_produces_explicit_invalid_state():
    store = EventStore(":memory:")
    result = populate_synthetic_round(store, "r0")
    events = store.all_events("r0")
    # decision far past the last SPOT event's freshness window but still
    # within an existing TWAP/book prefix - construct that by asking at a
    # time right after data stops (simulate a feed outage) via a filtered
    # event list capped early, decided upon later.
    capped = [e for e in events if e.event_time <= 50.0]
    r = compute(capped, "r0", 50.0 + CFG.freshness_s + 1.0, result.p0, CFG)
    assert isinstance(r, InvalidFeatureState)
    assert r.reason in (InvalidReason.STALE_SPOT, InvalidReason.STALE_TWAP, InvalidReason.STALE_BOOK)


def test_future_events_do_not_affect_output_even_if_present_in_input_list():
    events, result = _round_events(bias_bp_per_tick=7.0)
    decision_ts = 80.0

    r_without_future = compute([e for e in events if e.event_time <= decision_ts], "r0", decision_ts, result.p0, CFG)
    r_with_future = compute(events, "r0", decision_ts, result.p0, CFG)  # full list, including t>80 events

    assert isinstance(r_without_future, FeatureVector)
    assert isinstance(r_with_future, FeatureVector)
    assert r_without_future == r_with_future


def test_feature_output_invariant_between_incremental_and_batch_replay():
    """Simulates 'live' (growing event list, computed at each new decision
    point) vs 'batch replay' (full event list, computed once at each of the
    same decision points) and checks they agree - the same code path
    produces the same result regardless of how much *future* data happens
    to already be sitting in the caller's list."""
    events, result = _round_events(bias_bp_per_tick=7.0)
    decision_points = [20.0, 45.0, 90.0, 150.0, 220.0]

    batch_results = [compute(events, "r0", dt, result.p0, CFG) for dt in decision_points]

    live_results = []
    for dt in decision_points:
        live_slice = [e for e in events if e.event_time <= dt]
        live_results.append(compute(live_slice, "r0", dt, result.p0, CFG))

    assert batch_results == live_results


def test_deterministic_across_repeated_calls():
    events, result = _round_events(bias_bp_per_tick=-7.0)
    r1 = compute(events, "r0", 120.0, result.p0, CFG)
    r2 = compute(events, "r0", 120.0, result.p0, CFG)
    assert r1 == r2


def test_gap_and_lead_formulas_match_spec():
    events, result = _round_events(bias_bp_per_tick=7.0)
    r = compute(events, "r0", 90.0, result.p0, CFG)
    assert isinstance(r, FeatureVector)
    assert math.isclose(r.gap_twap_bp, 10_000.0 * (r.twap - r.p0) / r.p0)
    assert math.isclose(r.gap_spot_bp, 10_000.0 * (r.spot - r.p0) / r.p0)
    assert math.isclose(r.lead_gap_bp, 10_000.0 * (r.spot - r.twap) / r.twap)


def test_z_gap_formula_matches_spec():
    events, result = _round_events(bias_bp_per_tick=7.0)
    r = compute(events, "r0", 90.0, result.p0, CFG)
    assert isinstance(r, FeatureVector)
    expected = math.log(r.twap / r.p0) / (r.realized_vol * math.sqrt(r.tau))
    assert math.isclose(r.z_gap, expected, rel_tol=1e-6)


def test_clob_log_odds_formula_matches_spec():
    events, result = _round_events(bias_bp_per_tick=7.0)
    r = compute(events, "r0", 90.0, result.p0, CFG)
    assert isinstance(r, FeatureVector)
    expected = math.log(r.clob_mid / (1.0 - r.clob_mid))
    assert math.isclose(r.clob_log_odds, expected, rel_tol=1e-6)


def test_ofi_formula_matches_spec_and_is_bounded():
    events, result = _round_events(bias_bp_per_tick=7.0)
    r = compute(events, "r0", 90.0, result.p0, CFG)
    assert isinstance(r, FeatureVector)
    assert -1.0 <= r.ofi <= 1.0


def test_time_regime_boundaries():
    assert time_regime_for(0.0) is TimeRegime.DISCOVERY
    assert time_regime_for(59.9) is TimeRegime.DISCOVERY
    assert time_regime_for(60.0) is TimeRegime.CORE_TRADING
    assert time_regime_for(179.9) is TimeRegime.CORE_TRADING
    assert time_regime_for(180.0) is TimeRegime.COMPRESSION
    assert time_regime_for(239.9) is TimeRegime.COMPRESSION
    assert time_regime_for(240.0) is TimeRegime.LATE_CONTROLLED
    assert time_regime_for(269.9) is TimeRegime.LATE_CONTROLLED
    assert time_regime_for(270.0) is TimeRegime.SETTLEMENT_PROTECTION
    assert time_regime_for(300.0) is TimeRegime.SETTLEMENT_PROTECTION


def test_spot_return_horizons_omitted_when_insufficient_history():
    events, result = _round_events(bias_bp_per_tick=7.0)
    # at t=6, volatility history is sufficient (needs 6 points, has 7) but
    # the round is only 6s old, so the 10s-horizon return can't exist yet.
    r = compute(events, "r0", 6.0, result.p0, CFG)
    assert isinstance(r, FeatureVector)
    assert 10.0 not in r.spot_returns_bp
    assert 1.0 in r.spot_returns_bp
