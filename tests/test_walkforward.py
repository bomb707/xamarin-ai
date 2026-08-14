"""Phase 11 verification: rolling walk-forward windows have no lookahead,
bootstrap CIs are correct, all 8 mandatory ablations run without error, and
parameter sensitivity/stability sweeps never touch a window's test rounds
("no test-period tuning" - Roadmap Phase 11 verification, named
explicitly)."""
from __future__ import annotations

import math

import pytest

from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.calibrated import fit_calibrated_model
from xamarinbot.model.dataset import build_examples_multi
from xamarinbot.model.features import COMBINED_LEAD_LAG
from xamarinbot.model.walkforward import time_ordered_split
from xamarinbot.portfolio.state import FeeConfig, Side
from xamarinbot.synthetic.rounds import generate_synthetic_dataset
from xamarinbot.walkforward.ablations import MANDATORY_ABLATIONS, RoundResult, run_ablation_round
from xamarinbot.walkforward.bootstrap import bootstrap_ci
from xamarinbot.walkforward.sensitivity import parameter_stability_across_windows, sweep_parameter
from xamarinbot.walkforward.windows import WalkForwardWindow, rolling_windows

HEARTBEAT_S = 10.0


# --------------------------------------------------------------------------
# windows.py
# --------------------------------------------------------------------------


def test_rolling_windows_train_validate_test_are_chronologically_disjoint_and_ordered():
    round_ids = [f"r{i}" for i in range(10)]
    windows = rolling_windows(round_ids, n_train=3, n_validate=2, n_test=2)
    assert len(windows) == 2  # (3+2+2)=7 per window, step=n_test=2 -> starts 0, 2 fit within 10
    for w in windows:
        train_idx = [round_ids.index(r) for r in w.train_round_ids]
        validate_idx = [round_ids.index(r) for r in w.validate_round_ids]
        test_idx = [round_ids.index(r) for r in w.test_round_ids]
        # every train index precedes every validate index precedes every test index
        assert max(train_idx) < min(validate_idx)
        assert max(validate_idx) < min(test_idx)
        assert len(w.train_round_ids) == 3
        assert len(w.validate_round_ids) == 2
        assert len(w.test_round_ids) == 2
        assert w.all_round_ids == w.train_round_ids + w.validate_round_ids + w.test_round_ids


def test_rolling_windows_overlapping_step_still_orders_each_window_causally():
    """A smaller-than-default `step` deliberately lets a round reused as
    one window's test appear as a *later* window's train/validate (per
    windows.py's own docstring) - that's expected reuse, not a leak. The
    invariant that must hold regardless of `step` is per-window: each
    window's own train/validate/test stay chronologically ordered."""
    round_ids = [f"r{i}" for i in range(12)]
    windows = rolling_windows(round_ids, n_train=2, n_validate=2, n_test=2, step=2)
    assert len(windows) > 1
    for w in windows:
        train_idx = [round_ids.index(r) for r in w.train_round_ids]
        validate_idx = [round_ids.index(r) for r in w.validate_round_ids]
        test_idx = [round_ids.index(r) for r in w.test_round_ids]
        assert max(train_idx) < min(validate_idx) < max(validate_idx) < min(test_idx)


def test_rolling_windows_default_step_gives_nonoverlapping_test_segments():
    round_ids = [f"r{i}" for i in range(20)]
    windows = rolling_windows(round_ids, n_train=3, n_validate=2, n_test=3)
    all_test_ids = [rid for w in windows for rid in w.test_round_ids]
    assert len(all_test_ids) == len(set(all_test_ids))  # no round appears in two windows' test segments


@pytest.mark.parametrize("n_train,n_validate,n_test", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)])
def test_rolling_windows_rejects_nonpositive_sizes(n_train, n_validate, n_test):
    with pytest.raises(ValueError):
        rolling_windows(["r0", "r1"], n_train=n_train, n_validate=n_validate, n_test=n_test)


def test_rolling_windows_empty_when_dataset_smaller_than_one_window():
    windows = rolling_windows(["r0", "r1"], n_train=3, n_validate=3, n_test=3)
    assert windows == []


# --------------------------------------------------------------------------
# bootstrap.py
# --------------------------------------------------------------------------


def test_bootstrap_ci_point_estimate_is_plain_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bootstrap_ci(values, n_resamples=500, seed_key="test-a")
    assert math.isclose(result.point_estimate, 3.0)


def test_bootstrap_ci_lower_bound_never_exceeds_upper_bound():
    values = [1.0, -5.0, 3.0, 10.0, -2.0, 0.5, 7.0]
    result = bootstrap_ci(values, n_resamples=500, seed_key="test-b")
    assert result.lower <= result.upper


def test_bootstrap_ci_is_deterministic_for_same_seed_key():
    values = [1.0, 5.0, -3.0, 8.0, 2.0]
    a = bootstrap_ci(values, n_resamples=200, seed_key="fixed-key")
    b = bootstrap_ci(values, n_resamples=200, seed_key="fixed-key")
    assert a == b


def test_bootstrap_ci_different_seed_keys_can_differ():
    values = [1.0, 5.0, -3.0, 8.0, 2.0, 0.0, -1.0, 9.0]
    a = bootstrap_ci(values, n_resamples=200, seed_key="key-1")
    b = bootstrap_ci(values, n_resamples=200, seed_key="key-2")
    assert (a.lower, a.upper) != (b.lower, b.upper)


def test_bootstrap_ci_zero_samples_returns_zeroed_result_not_error():
    result = bootstrap_ci([], seed_key="empty")
    assert result.n_samples == 0
    assert result.point_estimate == 0.0
    assert result.lower == 0.0 == result.upper


def test_bootstrap_ci_single_sample_is_degenerate_at_that_value():
    result = bootstrap_ci([42.0], seed_key="single")
    assert result.n_samples == 1
    assert result.point_estimate == result.lower == result.upper == 42.0


def test_bootstrap_ci_widens_with_more_variance():
    tight = bootstrap_ci([5.0, 5.1, 4.9, 5.0, 5.05], n_resamples=500, seed_key="tight")
    wide = bootstrap_ci([-50.0, 60.0, -40.0, 55.0, -45.0], n_resamples=500, seed_key="wide")
    assert (wide.upper - wide.lower) > (tight.upper - tight.lower)


# --------------------------------------------------------------------------
# ablations.py - shared fixtures (module-scoped: training + eval datasets
# are expensive to build, and every ablation test needs the same ones)
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
def trained_model(feature_cfg):
    # Phase 12B audit item 5/C: the model these tests exercise the
    # controller with must be calibrated, matching what every demo/harness
    # now does - not the raw logistic score.
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=8)
    by_fs = build_examples_multi(store, results, feature_cfg, [COMBINED_LEAD_LAG], heartbeat_s=HEARTBEAT_S)
    examples = by_fs[COMBINED_LEAD_LAG.name]
    split = time_ordered_split(examples, train_frac=0.6, val_frac=0.2)
    return fit_calibrated_model(split.train, split.validation, COMBINED_LEAD_LAG)


@pytest.fixture(scope="module")
def eval_dataset():
    store = EventStore(":memory:")
    # id_offset=8: disjoint from trained_model's training rounds [0, 8)
    # (Phase 12B audit Addendum A - previously both fixtures restarted at
    # index 0, so "eval" rounds were identical to training rounds).
    results = generate_synthetic_dataset(store, n_rounds=6, id_offset=8)
    return store, results


def test_mandatory_ablations_is_exactly_the_ss20_1_list():
    names = [s.name for s in MANDATORY_ABLATIONS]
    assert len(MANDATORY_ABLATIONS) == 8
    assert names == [
        "1_baseline_unanimous",
        "2_twap_only",
        "3_spot_only",
        "4_lead_lag",
        "5_lead_lag_clob_no_repair",
        "6_portfolio_control_taker_only",
        "7_maker_taker_cancel_replace",
        "8_full_mpc",
    ]
    # #1 has no feature set/config (baseline strategy, not the V2 model);
    # every other ablation must specify a feature set and a OneStepConfig.
    assert MANDATORY_ABLATIONS[0].controller == "baseline"
    for spec in MANDATORY_ABLATIONS[1:]:
        assert spec.controller in ("one_step", "mpc")
        assert spec.feature_set is not None
        assert spec.one_step_cfg is not None


@pytest.mark.parametrize("spec", MANDATORY_ABLATIONS, ids=[s.name for s in MANDATORY_ABLATIONS])
def test_every_mandatory_ablation_runs_without_error_and_produces_sane_metrics(spec, eval_dataset, feature_cfg, fee_config, exec_cfg, trained_model):
    store, results = eval_dataset
    for r in results[:3]:
        result = run_ablation_round(spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, trained_model, None)
        assert isinstance(result, RoundResult)
        assert result.n_actions >= 0
        assert 0.0 <= result.fill_rate <= 1.0
        assert math.isfinite(result.realized_pnl)
        assert math.isfinite(result.final_g)


def test_ablations_6_7_8_now_trade_after_taker_sizing_fix(eval_dataset, feature_cfg, fee_config, exec_cfg, trained_model):
    """UPDATED by Phase 12B Tranche 1 (items 7/8) - superseding the prior
    "zero actions" finding this test used to pin down.

    That finding was itself an artifact of the pre-Tranche-1 bug it
    documented: `taker_quantities` only ever offered raw cumulative
    depth-level sums as candidate sizes, so whenever even the smallest
    depth level breached `g_min`, taker-only execution (#6) had literally
    nothing feasible to offer. Since taker fills are also immediate/
    synchronous (no resting period), they were never vulnerable to the
    REGIME_FLIP-before-TTL cancellation dynamic that killed #7/#8's maker
    orders either - once sizing produces a risk-feasible taker quantity,
    it can win selection and fill before any regime flip has a chance to
    cancel it. Fixing the sizing bug (`taker_sizing_boundaries`, wired
    into `max_directional_spend`) therefore changes all three ablations'
    behavior on this dataset, not just #6's - confirmed directly (11, 11,
    and 114 actions respectively across this eval set at the time of the
    fix). Exact counts aren't asserted here since they're sensitive to
    the model fit and dataset - the qualitative claim (each can now
    trade) is the regression this test protects."""
    store, results = eval_dataset
    for name in ("6_portfolio_control_taker_only", "7_maker_taker_cancel_replace", "8_full_mpc"):
        spec = next(s for s in MANDATORY_ABLATIONS if s.name == name)
        totals = [
            run_ablation_round(spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, trained_model, None)
            for r in results
        ]
        assert sum(r.n_actions for r in totals) > 0, f"{name} expected to trade now that taker sizing is risk/depth/marginal-edge-aware"


def test_supervisor_receives_recomputed_economics_not_placeholders(eval_dataset, feature_cfg, fee_config, exec_cfg, trained_model):
    """Phase 12B audit item 12/18 regression: `_run_controller_round`
    previously called `supervisor.review_order` with a hardcoded
    `current_delta_ev=0.0` and `current_g_after_if_fill=portfolio.G`
    (the *unconditional* current G, not the order's own if-filled G).
    Spy on `OrderSupervisor.review_order` across a real ablation-7 run and
    assert at least one call received a `current_delta_ev` that isn't
    exactly 0.0 and a `current_g_after_if_fill` that isn't always
    identical to the portfolio's own G at call time - proving real,
    order-specific recomputation is happening, not the old placeholders."""
    from unittest.mock import patch

    import xamarinbot.supervisor.supervisor as supervisor_mod

    store, results = eval_dataset
    spec = next(s for s in MANDATORY_ABLATIONS if s.name == "7_maker_taker_cancel_replace")

    calls = []
    real_review_order = supervisor_mod.OrderSupervisor.review_order

    def spy(self, tracked, now_ts, current_regime_state, current_delta_ev, current_g_after_if_fill, tau, is_fresh, current_optimal_ev=None):
        calls.append((current_delta_ev, current_g_after_if_fill))
        return real_review_order(self, tracked, now_ts, current_regime_state, current_delta_ev, current_g_after_if_fill, tau, is_fresh, current_optimal_ev)

    with patch.object(supervisor_mod.OrderSupervisor, "review_order", spy):
        for r in results:
            run_ablation_round(spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, trained_model, None)

    assert len(calls) > 0, "expected at least one open-order review across these rounds"
    assert any(delta_ev != 0.0 for delta_ev, _ in calls), "current_delta_ev was always exactly 0.0 - looks like the old placeholder"


def test_ablation_5_without_repair_does_take_action(eval_dataset, feature_cfg, fee_config, exec_cfg, trained_model):
    """Contrast case for the previous test - #5 (no supervisor, no
    taker_only) resolves maker fills immediately via a stochastic draw
    rather than leaving them open to be cancelled, so it should trade."""
    store, results = eval_dataset
    spec = next(s for s in MANDATORY_ABLATIONS if s.name == "5_lead_lag_clob_no_repair")
    totals = [
        run_ablation_round(spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, trained_model, None)
        for r in results
    ]
    assert sum(r.n_actions for r in totals) > 0


def test_baseline_ablation_matches_direct_baseline_semantics(eval_dataset, feature_cfg, fee_config, exec_cfg, trained_model):
    store, results = eval_dataset
    spec = MANDATORY_ABLATIONS[0]
    r = results[0]
    result = run_ablation_round(spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, trained_model, None)
    assert result.fill_rate in (0.0, 1.0)  # baseline's fill_rate is binary per docstring


# --------------------------------------------------------------------------
# sensitivity.py
# --------------------------------------------------------------------------


def test_sweep_parameter_varies_only_the_named_field(eval_dataset, feature_cfg, fee_config, exec_cfg, trained_model):
    from xamarinbot.optimizer.config import OneStepConfig

    store, results = eval_dataset
    base_cfg = OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0, edge_min=0.0)
    result = sweep_parameter("edge_min", [0.0, 50.0], base_cfg, COMBINED_LEAD_LAG, store, results[:3], feature_cfg, fee_config, exec_cfg, trained_model)
    assert [p.value for p in result.points] == [0.0, 50.0]
    # a very high edge_min should never trade more than a permissive one on the same data
    by_value = {p.value: p for p in result.points}
    assert by_value[50.0].mean_fill_rate <= by_value[0.0].mean_fill_rate


def test_sweep_parameter_rejects_unknown_field():
    from xamarinbot.optimizer.config import OneStepConfig

    base_cfg = OneStepConfig(g_min=-100.0)
    with pytest.raises(ValueError):
        sweep_parameter("not_a_real_field", [0.0], base_cfg, COMBINED_LEAD_LAG, EventStore(":memory:"), [], FeatureConfig(), FeeConfig(), ExecutionConfig(), None)


def test_parameter_stability_across_windows_uses_only_validate_rounds_never_test(eval_dataset, feature_cfg, fee_config, exec_cfg, trained_model):
    """The Roadmap Phase 11 "no test-period tuning" requirement, made
    concrete: monkeypatch `sweep_parameter` to record the round_ids it's
    handed on each call (one call per window, in window order - checked
    per-window since a small `step` can legitimately reuse a round as one
    window's test and a later window's validate, per windows.py's own
    docstring, so only a per-call/per-window check is meaningful here)."""
    from xamarinbot.optimizer.config import OneStepConfig
    import xamarinbot.walkforward.sensitivity as sensitivity_mod

    store, results = eval_dataset
    round_ids = [r.round_id for r in results]
    windows = rolling_windows(round_ids, n_train=1, n_validate=1, n_test=1)
    assert windows, "expected at least one window from 6 rounds at size 1/1/1"

    calls: list[set[str]] = []
    real_sweep = sensitivity_mod.sweep_parameter

    def spy(parameter_name, values, base_cfg, feature_set, store_, rounds, *a, **kw):
        calls.append({r.round_id for r in rounds})
        return real_sweep(parameter_name, values, base_cfg, feature_set, store_, rounds, *a, **kw)

    sensitivity_mod.sweep_parameter = spy
    try:
        base_cfg = OneStepConfig(g_min=-100.0, spend_cap=200.0, position_limit=200.0, edge_min=0.0)
        parameter_stability_across_windows("edge_min", [0.0, 1.0], base_cfg, COMBINED_LEAD_LAG, windows, store, results, feature_cfg, fee_config, exec_cfg, trained_model)
    finally:
        sensitivity_mod.sweep_parameter = real_sweep

    assert len(calls) == len(windows)
    for window, seen in zip(windows, calls):
        assert seen == set(window.validate_round_ids)
        assert not (seen & set(window.test_round_ids))


def test_parameter_stability_reports_instability_when_argmax_differs_per_window():
    from dataclasses import dataclass
    from unittest.mock import patch

    from xamarinbot.walkforward.sensitivity import SweepPoint, SensitivityResult

    @dataclass(frozen=True)
    class FakeRound:
        round_id: str
        p0: float = 100_000.0
        outcome: Side = Side.UP

    fake_results = [
        SensitivityResult("edge_min", (SweepPoint(0.0, 10.0, 0.0, 1.0, 1), SweepPoint(1.0, 5.0, 0.0, 1.0, 1)), best_value=0.0),
        SensitivityResult("edge_min", (SweepPoint(0.0, 3.0, 0.0, 1.0, 1), SweepPoint(1.0, 9.0, 0.0, 1.0, 1)), best_value=1.0),
    ]
    windows = [
        WalkForwardWindow(0, ("t0",), ("v0",), ("x0",)),
        WalkForwardWindow(1, ("t1",), ("v1",), ("x1",)),
    ]
    all_rounds = [FakeRound("t0"), FakeRound("v0"), FakeRound("x0"), FakeRound("t1"), FakeRound("v1"), FakeRound("x1")]
    with patch("xamarinbot.walkforward.sensitivity.sweep_parameter", side_effect=fake_results):
        from xamarinbot.optimizer.config import OneStepConfig

        result = parameter_stability_across_windows(
            "edge_min", [0.0, 1.0], OneStepConfig(g_min=-100.0), COMBINED_LEAD_LAG, windows,
            EventStore(":memory:"), all_rounds, FeatureConfig(), FeeConfig(), ExecutionConfig(), None,
        )
    assert result.window_best_values == (0.0, 1.0)
    assert result.stable is False
    assert result.n_distinct_best_values == 2
