"""Parameter sensitivity/stability analysis (Roadmap Phase 11: "Track
parameter stability and sensitivity, not only best parameter values.").

Sweeps one `OneStepConfig` field at a time across a set of rounds and
reports the resulting metric at every value, not only the argmax - the
roadmap phrasing draws an explicit contrast with reporting a single "best"
number, since a sharp, narrow optimum found on one dataset slice is exactly
what a walk-forward process (`windows.py`) is meant to catch overfitting
to. `parameter_stability_across_windows` makes that concrete: it re-runs
the sweep independently per walk-forward window and reports whether the
argmax value drifts window to window - instability there is itself the
finding, not a bug to fix.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.features.config import FeatureConfig
from xamarinbot.model.features import FeatureSet
from xamarinbot.model.calibrated import CalibratedModel
from xamarinbot.model.logistic import LogisticModel
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.portfolio.state import FeeConfig, Side
from xamarinbot.walkforward.ablations import AblationSpec, RoundResult, run_ablation_round
from xamarinbot.walkforward.windows import WalkForwardWindow


@dataclass(frozen=True)
class SweepPoint:
    value: float
    mean_pnl: float
    mean_g: float
    mean_fill_rate: float
    n_rounds: int


@dataclass(frozen=True)
class SensitivityResult:
    parameter_name: str
    points: tuple[SweepPoint, ...]
    best_value: float  # argmax mean_pnl among points

    @property
    def pnl_range(self) -> float:
        """Spread between the best and worst mean PnL across swept values -
        a flat range means the parameter barely matters in this regime;
        a wide one means it does and is worth tuning carefully."""
        pnls = [p.mean_pnl for p in self.points]
        return max(pnls) - min(pnls) if pnls else 0.0


def _rounds_for_ids(all_rounds: list, round_ids: tuple[str, ...]) -> list:
    wanted = set(round_ids)
    return [r for r in all_rounds if r.round_id in wanted]


def sweep_parameter(
    parameter_name: str,
    values: list[float],
    base_cfg: OneStepConfig,
    feature_set: FeatureSet,
    store: EventStore,
    rounds: list,  # list[SyntheticRoundResult], pre-filtered to the evaluation slice
    feature_cfg: FeatureConfig,
    fee_config: FeeConfig,
    exec_cfg: ExecutionConfig,
    model: LogisticModel | CalibratedModel | None,
) -> SensitivityResult:
    """Runs `run_ablation_round` once per (value, round) pair, varying only
    `parameter_name` on `base_cfg` via `dataclasses.replace` - every other
    field, and the underlying replay data, held fixed, so the swept
    parameter is the only thing that can explain a metric change."""
    if not hasattr(base_cfg, parameter_name):
        raise ValueError(f"OneStepConfig has no field {parameter_name!r}")

    points: list[SweepPoint] = []
    for value in values:
        cfg = replace(base_cfg, **{parameter_name: value})
        spec = AblationSpec(f"sweep_{parameter_name}={value}", f"sensitivity sweep", controller="one_step", feature_set=feature_set, one_step_cfg=cfg)
        results: list[RoundResult] = [
            run_ablation_round(spec, store, r.round_id, r.p0, r.outcome, feature_cfg, fee_config, exec_cfg, model, None)
            for r in rounds
        ]
        n = len(results)
        mean_pnl = sum(r.realized_pnl for r in results) / n if n else 0.0
        mean_g = sum(r.final_g for r in results) / n if n else 0.0
        mean_fill_rate = sum(r.fill_rate for r in results) / n if n else 0.0
        points.append(SweepPoint(value, mean_pnl, mean_g, mean_fill_rate, n))

    best = max(points, key=lambda p: p.mean_pnl) if points else None
    return SensitivityResult(parameter_name, tuple(points), best_value=best.value if best else 0.0)


@dataclass(frozen=True)
class StabilityResult:
    parameter_name: str
    values_swept: tuple[float, ...]
    window_best_values: tuple[float, ...]  # one per walk-forward window, in window order
    stable: bool  # True iff every window agrees on the same best value
    n_distinct_best_values: int


def parameter_stability_across_windows(
    parameter_name: str,
    values: list[float],
    base_cfg: OneStepConfig,
    feature_set: FeatureSet,
    windows: list[WalkForwardWindow],
    store: EventStore,
    all_rounds: list,  # list[SyntheticRoundResult] covering every round_id referenced by `windows`
    feature_cfg: FeatureConfig,
    fee_config: FeeConfig,
    exec_cfg: ExecutionConfig,
    model: LogisticModel | CalibratedModel | None,
) -> StabilityResult:
    """Per Roadmap Phase 11: "Lock parameters before each test segment" -
    the sweep here uses each window's *validate* rounds only (never test),
    matching that discipline; `test_walkforward.py`'s "no test-period
    tuning" check asserts this same separation holds."""
    window_best: list[float] = []
    for window in windows:
        validate_rounds = _rounds_for_ids(all_rounds, window.validate_round_ids)
        if not validate_rounds:
            continue
        result = sweep_parameter(parameter_name, values, base_cfg, feature_set, store, validate_rounds, feature_cfg, fee_config, exec_cfg, model)
        window_best.append(result.best_value)

    distinct = set(window_best)
    return StabilityResult(
        parameter_name=parameter_name,
        values_swept=tuple(values),
        window_best_values=tuple(window_best),
        stable=len(distinct) <= 1,
        n_distinct_best_values=len(distinct),
    )
