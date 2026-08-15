"""Cancel/replace analytics (Roadmap Phase 9 deliverable).

"Measure cancel regret: canceled orders that would have filled profitably
versus kept orders that became adversely selected." Cancel regret can only
be measured *after the fact*, by looking at what the market actually did
following a cancel - this is post-hoc replay analysis, never a live
decision input (the supervisor's own decisions only ever use causal data up
to the decision time, per every earlier phase's causality invariant).

`would_have_filled` is a simplified proxy (did the touch price cross the
canceled order's price within a lookback window afterward), not a full
counterfactual fill simulation with queue position - documented as an
approximation, in the same spirit as Phase 7's uncalibrated maker fill
model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.events.store import EventStore
from xamarinbot.replay.feeds import ReplayBookFeed, ReplayCursor
from xamarinbot.journal.schema import SupervisorDecisionRecord
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.portfolio.state import Side


@dataclass(frozen=True)
class CancelRegretRecord:
    order_id: str
    cancel_ts: float
    side: Side
    price: float
    reason: str | None
    would_have_filled: bool


@dataclass
class SupervisorReport:
    action_counts: dict[str, int] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)
    n_cancels_checked_for_regret: int = 0
    n_cancel_regret: int = 0  # canceled orders that likely would have filled

    @property
    def cancel_regret_rate(self) -> float:
        return self.n_cancel_regret / self.n_cancels_checked_for_regret if self.n_cancels_checked_for_regret else 0.0


def _would_have_filled(store: EventStore, round_id: str, side: Side, order_price: float, cancel_ts: float, lookback_s: float) -> bool:
    """Approximation: did the best ask on `side`'s book touch or cross
    `order_price` (i.e. the resting bid's price) at any point in
    (cancel_ts, cancel_ts + lookback_s]? A real fill also needs queue
    priority and size, which this doesn't model."""
    events = store.all_events(round_id)
    cursor = ReplayCursor(store, round_id, preloaded=events, now=cancel_ts)
    book_feed = ReplayBookFeed(cursor)
    times = sorted({e.event_time for e in events if cancel_ts < e.event_time <= cancel_ts + lookback_s})
    for t in times:
        cursor.advance_to(t)
        snap = book_feed.get_snapshot(round_id, side)
        if snap is not None and snap.best_ask is not None and snap.best_ask.price <= order_price:
            return True
    return False


def build_supervisor_report(journal: JournalWriter, store: EventStore, regret_lookback_s: float = 15.0) -> SupervisorReport:
    records = journal.read(SupervisorDecisionRecord)
    report = SupervisorReport()

    for r in records:
        report.action_counts[r.action] = report.action_counts.get(r.action, 0) + 1
        if r.reason:
            report.reason_counts[r.reason] = report.reason_counts.get(r.reason, 0) + 1

        if r.action == "CANCEL":
            report.n_cancels_checked_for_regret += 1
            side = Side.UP if r.side == "UP" else Side.DOWN
            if _would_have_filled(store, r.round_id, side, r.price, r.decision_ts, regret_lookback_s):
                report.n_cancel_regret += 1

    return report


def format_supervisor_report(report: SupervisorReport) -> str:
    lines = [
        "=== OrderSupervisor Cancel/Replace Analytics (Phase 9) ===",
        f"Action counts: {report.action_counts}",
        f"Cancel reason counts: {report.reason_counts}",
        f"Cancel regret: {report.n_cancel_regret}/{report.n_cancels_checked_for_regret} "
        f"canceled orders ({report.cancel_regret_rate:.1%}) likely would have filled (approximate, see module docstring)",
    ]
    return "\n".join(lines)
