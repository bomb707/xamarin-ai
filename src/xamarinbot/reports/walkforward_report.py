"""Walk-forward / ablation-matrix report formatting (Roadmap Phase 11
deliverable: a report comparing all 8 mandatory ablations with bootstrap
confidence intervals, plus parameter sensitivity/stability tables)."""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.walkforward.ablations import RoundResult
from xamarinbot.walkforward.bootstrap import BootstrapResult, bootstrap_ci
from xamarinbot.walkforward.sensitivity import SensitivityResult, StabilityResult


@dataclass(frozen=True)
class AblationSummary:
    name: str
    description: str
    n_rounds: int
    mean_pnl: float
    pnl_ci: BootstrapResult
    mean_fill_rate: float
    mean_actions: float


def summarize_ablation(name: str, description: str, results: list[RoundResult]) -> AblationSummary:
    n = len(results)
    pnls = [r.realized_pnl for r in results]
    mean_pnl = sum(pnls) / n if n else 0.0
    mean_fill_rate = sum(r.fill_rate for r in results) / n if n else 0.0
    mean_actions = sum(r.n_actions for r in results) / n if n else 0.0
    return AblationSummary(name, description, n, mean_pnl, bootstrap_ci(pnls, seed_key=name), mean_fill_rate, mean_actions)


def format_ablation_matrix(summaries: list[AblationSummary]) -> str:
    lines = ["=== Ablation Matrix (Roadmap Phase 11 / SS20.1) ==="]
    header = f"{'name':<32} {'n':>4} {'mean_pnl':>10} {'95% CI':>22} {'fill_rate':>10} {'actions/rd':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        ci = f"[{s.pnl_ci.lower:.2f}, {s.pnl_ci.upper:.2f}]"
        lines.append(f"{s.name:<32} {s.n_rounds:>4} {s.mean_pnl:>10.3f} {ci:>22} {s.mean_fill_rate:>10.2f} {s.mean_actions:>11.2f}")
    return "\n".join(lines)


def format_sensitivity(result: SensitivityResult) -> str:
    lines = [f"=== Sensitivity: {result.parameter_name} ==="]
    for p in result.points:
        lines.append(f"  {result.parameter_name}={p.value:<10} mean_pnl={p.mean_pnl:>10.3f}  mean_g={p.mean_g:>10.3f}  fill_rate={p.mean_fill_rate:.2f}  (n={p.n_rounds})")
    lines.append(f"  best (by mean_pnl): {result.best_value}  |  pnl_range across sweep: {result.pnl_range:.3f}")
    return "\n".join(lines)


def format_stability(result: StabilityResult) -> str:
    lines = [f"=== Stability across windows: {result.parameter_name} ==="]
    lines.append(f"  values swept: {result.values_swept}")
    lines.append(f"  per-window best: {result.window_best_values}")
    lines.append(f"  stable across windows: {result.stable}  ({result.n_distinct_best_values} distinct best value(s))")
    if not result.stable:
        lines.append("  NOTE: the argmax parameter value changed between walk-forward windows -")
        lines.append("  this is a genuine finding (overfitting risk to any single window), not a bug.")
    return "\n".join(lines)
