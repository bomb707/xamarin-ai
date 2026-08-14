"""Taker slippage/delay report (Roadmap Phase 7 deliverable).

Operates directly on the `TakerOrderResult` objects a simulation run
collects, rather than round-tripping through the journal - Phase 7 is about
validating the *simulator's* behavior (the Phase 7 exit gate: "Execution
model error is quantified and within research tolerance"), which is most
directly measured from its own outputs. A strategy integration would
additionally journal fills through the existing FillRecord/OrderEventRecord
entities, as Phase 0's baseline replay already does for its simpler
same-tick-fill assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.execution.taker import TakerOrderResult


@dataclass
class ExecutionReport:
    n_orders: int = 0
    n_delayed: int = 0
    n_repriced: int = 0
    n_fully_filled: int = 0
    n_partially_filled: int = 0
    n_unfilled: int = 0
    total_requested: float = 0.0
    total_filled: float = 0.0
    total_fees: float = 0.0
    avg_fill_rate: float = 0.0
    avg_delay_slippage_per_share: float = 0.0  # (walk.avg_price - walk_at_submission.avg_price), delayed+repriced+filled orders only


def build_execution_report(results: list[TakerOrderResult]) -> ExecutionReport:
    report = ExecutionReport(n_orders=len(results))
    fill_rates: list[float] = []
    delay_slippages: list[float] = []

    for r in results:
        report.total_requested += r.walk.requested_shares
        report.total_filled += r.walk.filled_shares
        report.total_fees += r.walk.total_fee
        fill_rates.append(r.walk.filled_shares / r.walk.requested_shares if r.walk.requested_shares > 0 else 0.0)

        if r.walk.fully_filled:
            report.n_fully_filled += 1
        elif r.walk.filled_shares > 0:
            report.n_partially_filled += 1
        else:
            report.n_unfilled += 1

        if r.was_delayed:
            report.n_delayed += 1
            if r.repriced:
                report.n_repriced += 1
            if r.repriced and r.walk_at_submission is not None and r.walk.filled_shares > 0 and r.walk_at_submission.filled_shares > 0:
                delay_slippages.append(r.walk.avg_price - r.walk_at_submission.avg_price)

    report.avg_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0.0
    report.avg_delay_slippage_per_share = sum(delay_slippages) / len(delay_slippages) if delay_slippages else 0.0
    return report


def format_execution_report(report: ExecutionReport) -> str:
    lines = [
        "=== Taker Slippage/Delay Report (Phase 7) ===",
        f"Orders: {report.n_orders}  |  Fully filled: {report.n_fully_filled}  |  Partial: {report.n_partially_filled}  |  Unfilled: {report.n_unfilled}",
        f"Delayed (250ms window): {report.n_delayed}  |  Repriced during delay: {report.n_repriced}",
        f"Total requested: {report.total_requested:.2f}  |  Total filled: {report.total_filled:.2f}  |  Avg fill rate: {report.avg_fill_rate:.1%}",
        f"Total fees: {report.total_fees:.4f}",
        f"Avg delay-induced slippage per share (repriced+filled only): {report.avg_delay_slippage_per_share:+.5f}",
    ]
    return "\n".join(lines)
