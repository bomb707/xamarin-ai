"""Baseline performance report + failure taxonomy (Roadmap Phase 0).

"Run a baseline replay set and export win rate, average win/loss, profit
factor, largest single loss, max drawdown, fees, fill rate, and PnL by
entry price/time." "Create a baseline failure taxonomy: high-price loss,
late entry, false TWAP confirmation, reversal, stale data, missed/partial
fill, oversizing."

The source docs name every failure-taxonomy category but do not define
exact classification rules for a baseline whose full execution model
(Phase 7) doesn't exist yet. The heuristics below are this reconstruction's
best-effort rules, documented inline; `missed_partial_fill` is always 0
because partial fills aren't modeled until the Phase 7 execution simulator
- every Phase-0 fill is currently assumed to fill completely at decision
time (see synthetic/rounds.py and scripts/run_baseline_replay.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.journal.schema import AuditRecord, FillRecord, SettlementRecord
from xamarinbot.journal.writer import JournalWriter


@dataclass
class BaselineReport:
    n_rounds: int = 0
    n_rounds_with_position: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = float("nan")
    largest_single_loss: float = 0.0
    max_drawdown: float = 0.0
    total_fees: float = 0.0
    fill_rate: float = 0.0
    net_pnl: float = 0.0
    pnl_by_entry_price_bucket: dict[float, float] = field(default_factory=dict)
    pnl_by_time_bucket: dict[float, float] = field(default_factory=dict)
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    failure_taxonomy: dict[str, int] = field(default_factory=dict)


def _bucket(value: float, size: float) -> float:
    return round(value / size) * size


def build_baseline_report(journal: JournalWriter, cfg: BaselineConfig) -> BaselineReport:
    round_ids = journal.round_ids()
    report = BaselineReport(n_rounds=len(round_ids))

    price_buckets: dict[float, list[float]] = {}
    time_buckets: dict[float, list[float]] = {}
    pnls: list[float] = []
    wins: list[float] = []
    losses: list[float] = []
    taxonomy = {
        "high_price_loss": 0,
        "late_entry": 0,
        "false_twap_confirmation": 0,
        "reversal": 0,
        "stale_data": 0,
        "missed_partial_fill": 0,
        "oversizing": 0,
    }
    skip_counts: dict[str, int] = {}
    n_order_attempts = 0
    n_filled_attempts = 0

    for round_id in round_ids:
        audits: list[AuditRecord] = journal.read(AuditRecord, round_id)
        fills: list[FillRecord] = journal.read(FillRecord, round_id)
        settlements: list[SettlementRecord] = journal.read(SettlementRecord, round_id)

        for a in audits:
            if a.skip_reason:
                skip_counts[a.skip_reason] = skip_counts.get(a.skip_reason, 0) + 1
                if a.skip_reason == "STALE_DATA":
                    taxonomy["stale_data"] += 1
            else:
                n_order_attempts += 1
                n_filled_attempts += 1  # Phase 0 assumes full same-tick fill

        report.total_fees += sum(f.fee for f in fills)

        if not fills or not settlements:
            continue

        report.n_rounds_with_position += 1
        settlement = settlements[0]
        pnl = settlement.realized_pnl
        pnls.append(pnl)

        first_fill = min(fills, key=lambda f: f.fill_ts)
        weighted_price = sum(f.price * f.size for f in fills) / sum(f.size for f in fills)
        total_qty = sum(f.size for f in fills)

        price_buckets.setdefault(_bucket(weighted_price, 0.05), []).append(pnl)
        time_buckets.setdefault(_bucket(first_fill.fill_ts, 30.0), []).append(pnl)

        entry_audit = next(
            (a for a in audits if not a.skip_reason and abs(a.decision_ts - first_fill.fill_ts) < 1e-6),
            None,
        )

        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(pnl)
            if weighted_price >= cfg.avg_price_guard - 0.05:
                taxonomy["high_price_loss"] += 1
            if first_fill.fill_ts >= 0.7 * cfg.decision_window_end_s:
                taxonomy["late_entry"] += 1
            if total_qty > cfg.clip:
                taxonomy["oversizing"] += 1
            if entry_audit is not None:
                if abs(entry_audit.diagnostics.get("gap_bp", 0.0)) < 2 * cfg.minimum_gap_bp:
                    taxonomy["false_twap_confirmation"] += 1
                entry_twap_dir = entry_audit.diagnostics.get("twap_direction", 0)
                outcome_sign = 1 if settlement.outcome == "UP" else -1
                if entry_twap_dir != 0 and entry_twap_dir != outcome_sign:
                    taxonomy["reversal"] += 1

    report.n_wins = len(wins)
    report.n_losses = len(losses)
    report.win_rate = report.n_wins / report.n_rounds_with_position if report.n_rounds_with_position else 0.0
    report.avg_win = sum(wins) / len(wins) if wins else 0.0
    report.avg_loss = sum(abs(l) for l in losses) / len(losses) if losses else 0.0
    loss_sum = sum(abs(l) for l in losses)
    report.profit_factor = (sum(wins) / loss_sum) if loss_sum > 0 else float("inf")
    report.largest_single_loss = min(pnls) if pnls else 0.0
    report.net_pnl = sum(pnls)
    report.fill_rate = (n_filled_attempts / n_order_attempts) if n_order_attempts else 0.0
    report.skip_reason_counts = skip_counts
    report.failure_taxonomy = taxonomy

    # max drawdown over the round-order cumulative PnL curve
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cum += pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    report.max_drawdown = max_dd

    report.pnl_by_entry_price_bucket = {k: sum(v) / len(v) for k, v in sorted(price_buckets.items())}
    report.pnl_by_time_bucket = {k: sum(v) / len(v) for k, v in sorted(time_buckets.items())}

    return report


def format_report(report: BaselineReport) -> str:
    lines = [
        "=== Xamarinbot V2 - Baseline (Phase 0) Report ===",
        "(SYNTHETIC data - see synthetic/rounds.py; not evidence of real edge)",
        f"Rounds: {report.n_rounds}  |  Rounds with a position: {report.n_rounds_with_position}",
        f"Win rate: {report.win_rate:.1%}  ({report.n_wins}W / {report.n_losses}L)",
        f"Avg win: {report.avg_win:.4f}  |  Avg loss: {report.avg_loss:.4f}  |  Profit factor: {report.profit_factor:.3f}",
        f"Largest single loss: {report.largest_single_loss:.4f}  |  Max drawdown: {report.max_drawdown:.4f}",
        f"Total fees: {report.total_fees:.4f}  |  Fill rate: {report.fill_rate:.1%}  |  Net PnL: {report.net_pnl:.4f}",
        f"Skip reasons: {report.skip_reason_counts}",
        f"Failure taxonomy: {report.failure_taxonomy}",
        f"PnL by entry-price bucket: {report.pnl_by_entry_price_bucket}",
        f"PnL by entry-time bucket: {report.pnl_by_time_bucket}",
    ]
    return "\n".join(lines)
