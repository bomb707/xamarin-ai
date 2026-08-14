"""Transition statistics report (Roadmap Phase 6 deliverable)."""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.journal.schema import RegimeTransitionRecord
from xamarinbot.journal.writer import JournalWriter

_START = "START"


def _key(state: dict | None) -> str:
    if state is None:
        return _START
    return f"{state['gap_regime']}/{state['clob_direction']}/{state['spot_direction']}"


@dataclass
class RegimeTransitionReport:
    n_transitions: int = 0
    n_first_observations: int = 0
    cancel_count: int = 0
    seed_action_counts: dict[str, int] = field(default_factory=dict)
    transition_pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    avg_dwell_by_state: dict[str, float] = field(default_factory=dict)


def build_transition_report(journal: JournalWriter) -> RegimeTransitionReport:
    records = journal.read(RegimeTransitionRecord)
    report = RegimeTransitionReport(n_transitions=len(records))
    dwell_by_from_state: dict[str, list[float]] = {}

    for r in records:
        report.seed_action_counts[r.seed_action] = report.seed_action_counts.get(r.seed_action, 0) + 1
        if r.seed_action == "CANCEL":
            report.cancel_count += 1

        from_key, to_key = _key(r.from_state), _key(r.to_state)
        pair = (from_key, to_key)
        report.transition_pair_counts[pair] = report.transition_pair_counts.get(pair, 0) + 1

        if r.from_state is None:
            report.n_first_observations += 1
        elif r.dwell_time_s is not None:
            dwell_by_from_state.setdefault(from_key, []).append(r.dwell_time_s)

    report.avg_dwell_by_state = {k: sum(v) / len(v) for k, v in dwell_by_from_state.items()}
    return report


def format_transition_report(report: RegimeTransitionReport, top_n: int = 15) -> str:
    lines = [
        "=== Regime Transition Statistics (Phase 6) ===",
        f"Total transitions: {report.n_transitions}  |  Round starts: {report.n_first_observations}  |  CANCEL emitted: {report.cancel_count}",
        f"Seed action counts: {report.seed_action_counts}",
        "",
        f"Top {top_n} transition pairs (from -> to):",
    ]
    top_pairs = sorted(report.transition_pair_counts.items(), key=lambda kv: -kv[1])[:top_n]
    for (from_key, to_key), n in top_pairs:
        lines.append(f"  {n:>5}  {from_key} -> {to_key}")

    lines.append("")
    lines.append("Average dwell time (s) before leaving each state:")
    for state, avg_dwell in sorted(report.avg_dwell_by_state.items(), key=lambda kv: -kv[1])[:top_n]:
        lines.append(f"  {avg_dwell:>7.2f}s  {state}")

    return "\n".join(lines)
