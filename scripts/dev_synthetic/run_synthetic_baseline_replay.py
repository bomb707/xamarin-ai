#!/usr/bin/env python3
"""Phase 0 end-to-end demo: generate the synthetic dataset, replay the
reconstructed baseline strategy over it causally, journal every
decision/fill/settlement, and print the baseline performance report.

Fills are assumed to execute completely at the decision-time limit price
(no partial fills, no depth walk, no delay) - that realism arrives with the
Phase 7 ExecutionSimulator. This script only proves Phase 0-3 are wired
together correctly and deterministically; see docs/PHASE_STATUS.md.

Usage: python scripts/run_baseline_replay.py [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
# `devtools` (the synthetic data fabricator) lives at the repo root,
# deliberately outside the shipped package - see Phase 12C.1 item 4.
sys.path.insert(0, str(_REPO_ROOT))

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.baseline.inputs import elapsed_t
from xamarinbot.baseline.strategy import BaselineInputs, decide
from xamarinbot.config import config_hash
from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.feeds.freshness import FeedFreshnessMonitor
from xamarinbot.replay.feeds import (
    ReplayCursor,
    ReplayMarketConfigProvider,
    ReplaySpotFeed,
    ReplayTWAPFeed,
    ReplayBookFeed,
)
from xamarinbot.journal.schema import (
    AuditRecord,
    FillRecord,
    MarketConfigRecord,
    OrderEventRecord,
    PortfolioStateRecord,
    SettlementRecord,
)
from xamarinbot.journal.writer import JournalWriter
from xamarinbot.portfolio.state import Fill, FeeConfig, LiquidityRole, PortfolioState, apply_fill
from xamarinbot.reports.baseline_report import build_baseline_report, format_report
from devtools.synthetic.rounds import generate_synthetic_dataset


def _midpoint(book_snapshot):
    if book_snapshot.best_bid and book_snapshot.best_ask:
        return (book_snapshot.best_bid.price + book_snapshot.best_ask.price) / 2.0
    if book_snapshot.best_ask:
        return book_snapshot.best_ask.price
    return 0.5


class _TypeIndex:
    """One-time-built, bisectable index into a preloaded, sorted event list
    for a single event type - avoids re-querying/re-scanning the store at
    every decision point."""

    def __init__(self, events: list, event_type: EventType):
        self._events = [e for e in events if e.event_type is event_type]
        self._times = [e.event_time for e in self._events]

    def latest_payload_before(self, cutoff: float):
        import bisect

        idx = bisect.bisect_right(self._times, cutoff)
        return self._events[idx - 1].payload if idx else None


def run_round(store: EventStore, round_id: str, outcome_side, cfg: BaselineConfig, journal: JournalWriter) -> None:
    clock = ReplayClock(store, round_id)
    round_events = store.all_events(round_id)
    cursor = ReplayCursor(store, round_id, preloaded=round_events)
    twap_feed = ReplayTWAPFeed(cursor)
    spot_feed = ReplaySpotFeed(cursor)
    book_feed = ReplayBookFeed(cursor)
    config_provider = ReplayMarketConfigProvider(cursor)
    freshness = FeedFreshnessMonitor(default_max_staleness_s=cfg.freshness_s)

    # Separate cursor/feed for the "clob_lookback_s ago" reads, advanced in
    # lock-step (monotonically) with the main cursor each iteration, so
    # ReplayBookFeed's incremental cache benefits here too instead of
    # rebuilding a fresh cursor (and re-scanning the whole round) at every
    # decision point.
    prev_cursor = ReplayCursor(store, round_id, preloaded=round_events)
    prev_book_feed = ReplayBookFeed(prev_cursor)
    spot_index = _TypeIndex(round_events, EventType.SPOT)

    portfolio = PortfolioState()
    fee_config = FeeConfig(crypto_fee_rate=cfg.fee_fallback_rate)
    cfg_hash = config_hash(cfg)
    p0: float | None = None
    market_config_journaled = False

    from xamarinbot.portfolio.state import Side  # local import to avoid unused at module scope

    for decision_time in clock.decision_points():
        cursor.advance_to(decision_time)

        market_config = config_provider.get_market_config(round_id)
        if not market_config_journaled:
            journal.write(
                MarketConfigRecord(
                    round_id=round_id,
                    up_token_id=market_config.up_token_id,
                    down_token_id=market_config.down_token_id,
                    start_ts=market_config.start_ts,
                    end_ts=market_config.end_ts,
                    tick_size=market_config.tick_size,
                    min_order_size=market_config.min_order_size,
                    fee_rate=market_config.fee_rate,
                    taker_delay_ms=market_config.taker_delay_ms,
                    twap_window_seconds=market_config.twap_window_seconds,
                )
            )
            market_config_journaled = True

        twap_obs = twap_feed.get_latest(round_id)
        spot_obs = spot_feed.get_latest(round_id)
        if twap_obs is None or spot_obs is None:
            continue

        if p0 is None:
            p0 = spot_obs.value

        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)

        freshness.record("twap", twap_obs.observation_ts)
        freshness.record("spot", spot_obs.recv_ts)
        if book_up is not None:
            freshness.record("book", book_up.ts)
        is_fresh = freshness.all_fresh(["twap", "spot", "book"], decision_time)

        prev_cursor.advance_to(decision_time - cfg.clob_lookback_s)
        prev_book_up = prev_book_feed.get_snapshot(round_id, Side.UP)
        # true UP midpoint (Strategy doc SS4 M_t), not the raw ask - using
        # the ask directly would bake the spread into clob_mid and throw
        # off both direction-vs-noise sensitivity and the DOWN-side
        # ask_range reference (its complement 1-clob_mid).
        clob_mid = _midpoint(book_up) if book_up else 0.5
        clob_mid_prev = _midpoint(prev_book_up) if prev_book_up else clob_mid

        spot_prev_payload = spot_index.latest_payload_before(decision_time - cfg.spot_lookback_s)
        spot_prev = spot_prev_payload["value"] if spot_prev_payload else spot_obs.value

        inputs = BaselineInputs(
            t=elapsed_t(decision_time, market_config.start_ts),
            p0=p0,
            twap=twap_obs.value,
            clob_mid=clob_mid,
            clob_mid_prev=clob_mid_prev,
            spot=spot_obs.value,
            spot_prev=spot_prev,
            best_ask_up=book_up.best_ask.price if book_up and book_up.best_ask else None,
            best_ask_down=book_down.best_ask.price if book_down and book_down.best_ask else None,
            is_fresh=is_fresh,
        )

        decision = decide(inputs, portfolio.U, portfolio.D, portfolio.C, cfg)

        journal.write(
            AuditRecord(
                round_id=round_id,
                decision_ts=decision_time,
                strategy_version=cfg.strategy_version,
                config_hash=cfg_hash,
                skip_reason=decision.skip_reason.value if decision.skip_reason else None,
                diagnostics={
                    "clob_direction": decision.clob_direction,
                    "spot_direction": decision.spot_direction,
                    "twap_direction": decision.twap_direction,
                    "gap_bp": decision.gap_bp,
                },
            )
        )

        if decision.order is None:
            continue

        order = decision.order
        order_id = f"{round_id}-{decision_time:.3f}"
        fee = fee_config.fee_for(order.role, order.quantity, order.price)
        fill = Fill(side=order.side, price=order.price, shares=order.quantity, role=order.role, fee=fee)
        portfolio = apply_fill(portfolio, fill)

        journal.write(
            OrderEventRecord(
                round_id=round_id,
                order_id=order_id,
                decision_ts=decision_time,
                side=order.side.value,
                role=order.role.value,
                price=order.price,
                quantity=order.quantity,
                event="SUBMIT",
                status="FILLED",
            )
        )
        journal.write(
            FillRecord(
                round_id=round_id,
                order_id=order_id,
                fill_ts=decision_time,
                side=order.side.value,
                price=order.price,
                size=order.quantity,
                fee=fee,
                liquidity_role=order.role.value,
            )
        )
        journal.write(
            PortfolioStateRecord(
                round_id=round_id,
                decision_ts=decision_time,
                U=portfolio.U,
                D=portfolio.D,
                C=portfolio.C,
                Pi_U=portfolio.Pi_U,
                Pi_D=portfolio.Pi_D,
                G=portfolio.G,
                R=portfolio.R,
            )
        )

    if portfolio.U > 0 or portfolio.D > 0:
        realized_pnl = portfolio.Pi_U if outcome_side.value == "UP" else portfolio.Pi_D
        payout = portfolio.U if outcome_side.value == "UP" else portfolio.D
        journal.write(
            SettlementRecord(round_id=round_id, outcome=outcome_side.value, payout=payout, realized_pnl=realized_pnl)
        )


def main() -> None:
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=n_rounds)

    cfg = BaselineConfig()
    journal = JournalWriter(":memory:")

    for result in results:
        run_round(store, result.round_id, result.outcome, cfg, journal)

    report = build_baseline_report(journal, cfg)
    print(format_report(report))


if __name__ == "__main__":
    main()
