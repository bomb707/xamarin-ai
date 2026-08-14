"""Daily shadow report + live-vs-replay parity report (Roadmap Phase 12
deliverables)."""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.reports.mpc_report import percentile
from xamarinbot.shadow.parity import ParityReport
from xamarinbot.shadow.types import ShadowRoundResult


@dataclass(frozen=True)
class DailyShadowReport:
    n_rounds: int
    n_decisions: int
    n_non_wait_decisions: int
    n_reconnects: int
    n_missed_deadlines: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_hypothetical_g: float

    @property
    def missed_deadline_rate(self) -> float:
        return self.n_missed_deadlines / self.n_decisions if self.n_decisions else 0.0


def build_daily_shadow_report(round_results: list[ShadowRoundResult]) -> DailyShadowReport:
    all_records = [r for result in round_results for r in result.records]
    elapsed = sorted(r.decide_elapsed_ms for r in all_records)
    n_decisions = len(all_records)
    n_non_wait = sum(1 for r in all_records if r.action_id != "wait")
    n_reconnects = sum(result.n_reconnects for result in round_results)
    n_missed = sum(result.n_missed_deadlines for result in round_results)
    n = len(round_results)
    mean_g = sum(result.final_portfolio.G for result in round_results) / n if n else 0.0

    return DailyShadowReport(
        n_rounds=n, n_decisions=n_decisions, n_non_wait_decisions=n_non_wait,
        n_reconnects=n_reconnects, n_missed_deadlines=n_missed,
        p50_ms=percentile(elapsed, 50), p95_ms=percentile(elapsed, 95),
        p99_ms=percentile(elapsed, 99), max_ms=elapsed[-1] if elapsed else 0.0,
        mean_hypothetical_g=mean_g,
    )


def format_daily_shadow_report(report: DailyShadowReport) -> str:
    return (
        "=== Daily Shadow Report (Phase 12) ===\n"
        f"Rounds: {report.n_rounds}  |  Decisions: {report.n_decisions}  |  Non-WAIT: {report.n_non_wait_decisions}\n"
        f"Reconnects handled: {report.n_reconnects}  |  Missed deadlines: {report.n_missed_deadlines} ({report.missed_deadline_rate:.1%})\n"
        f"Decision latency  p50: {report.p50_ms:.3f}ms  p95: {report.p95_ms:.3f}ms  p99: {report.p99_ms:.3f}ms  max: {report.max_ms:.3f}ms\n"
        f"Mean hypothetical G at round end: {report.mean_hypothetical_g:.3f}"
    )


def format_parity_report(reports: list[ParityReport]) -> str:
    lines = ["=== Live-vs-Replay Parity Report (Phase 12) ==="]
    total_compared = sum(r.n_compared for r in reports)
    total_mismatches = sum(r.n_mismatches for r in reports)
    overall_rate = 1.0 - (total_mismatches / total_compared) if total_compared else 1.0
    for r in reports:
        lines.append(f"  {r.round_id}: {r.n_compared - r.n_mismatches}/{r.n_compared} match ({r.parity_rate:.1%})")
    lines.append(f"Overall parity: {total_compared - total_mismatches}/{total_compared} ({overall_rate:.1%})")
    if total_mismatches:
        lines.append(f"  ({total_mismatches} mismatch(es) - see 'Notable bugs'/Phase 12 finding in docs/PHASE_STATUS.md: expected from the recv_ts-vs-event_time causal gate, not a bug)")
    return "\n".join(lines)
