"""Taker execution simulation (Roadmap Phase 7).

"Build taker cost curve by walking current ask depth for requested size."
"Apply current fee parameters to simulated taker fills."
"Model 250 ms taker delay on markets where enabled, including
revalidation/repricing risk."
"Implement FAK partial fill semantics."

Reuses `feeds.base.BookLevel`/`BookSnapshot` (Phase 1) rather than a new
book representation, and `portfolio.state.FeeConfig` (Phase 3) for the fee
formula rather than reimplementing it.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.feeds.base import BookLevel
from xamarinbot.portfolio.state import FeeConfig


@dataclass(frozen=True)
class DepthWalkResult:
    """K(x) evaluated at one requested size: walking `levels` (already
    sorted best-first) up to `requested_shares`, only consuming levels at
    or better than `limit_price` - a marketable limit order never crosses
    its own limit, and FAK semantics fall out naturally: whatever isn't
    filled here (`requested_shares - filled_shares`) is simply never
    filled, no resting order created."""

    requested_shares: float
    filled_shares: float
    total_cost: float  # notional only
    total_fee: float
    avg_price: float
    legs: tuple[tuple[float, float], ...]  # (price, size) actually consumed
    limited_by_price: bool  # True if the walk stopped because of limit_price, not depth
    limited_by_depth: bool  # True if the walk stopped because the book ran out (partial FAK fill)

    @property
    def total_paid(self) -> float:
        return self.total_cost + self.total_fee

    @property
    def fully_filled(self) -> bool:
        return abs(self.filled_shares - self.requested_shares) < 1e-9


def walk_depth(levels: tuple[BookLevel, ...], requested_shares: float, limit_price: float, fee_config: FeeConfig) -> DepthWalkResult:
    remaining = requested_shares
    legs: list[tuple[float, float]] = []
    total_cost = 0.0
    total_fee = 0.0
    limited_by_price = False

    for level in levels:  # asks sorted ascending = best (cheapest) first
        if remaining <= 1e-12:
            break
        if level.price > limit_price + 1e-12:
            limited_by_price = True
            break
        take = min(remaining, level.size)
        if take <= 0:
            continue
        total_cost += take * level.price
        total_fee += fee_config.taker_fee(take, level.price)
        legs.append((level.price, take))
        remaining -= take

    filled = requested_shares - remaining
    avg_price = (total_cost / filled) if filled > 1e-12 else 0.0
    limited_by_depth = remaining > 1e-9 and not limited_by_price

    return DepthWalkResult(
        requested_shares=requested_shares,
        filled_shares=filled,
        total_cost=total_cost,
        total_fee=total_fee,
        avg_price=avg_price,
        legs=tuple(legs),
        limited_by_price=limited_by_price,
        limited_by_depth=limited_by_depth,
    )


@dataclass(frozen=True)
class TakerOrderResult:
    submit_ts: float
    matched_ts: float
    was_delayed: bool
    walk: DepthWalkResult
    repriced: bool  # book at revalidation differed from book at submission (delayed orders only)
    walk_at_submission: DepthWalkResult | None  # what would have filled with no delay, for slippage reporting


def simulate_taker_order(
    asks_at_submission: tuple[BookLevel, ...],
    requested_shares: float,
    limit_price: float,
    fee_config: FeeConfig,
    submit_ts: float,
    taker_delay_ms: float = 0.0,
    asks_at_revalidation: tuple[BookLevel, ...] | None = None,
) -> TakerOrderResult:
    """Simulates one FAK taker order. If `taker_delay_ms > 0` (crypto/
    finance up-down markets with the 250ms delay enabled, Strategy doc
    SS2.2), the *caller* is responsible for supplying the book as it stood
    `taker_delay_ms` later (`asks_at_revalidation`) - obtained from replay
    by querying the causal event store at submit_ts + delay, exactly like
    the real exchange's matching engine would use the book at that later
    instant. This is not a causality violation of the *strategy's*
    decision (which only ever sees data up to submit_ts) - it's the
    simulated *exchange* correctly modeling what actually determines the
    fill, per "revalidation/repricing risk."

    "the order is pending and cannot be canceled" during the delay window
    is enforced by order_state.py, not here - this function only computes
    the eventual fill.
    """
    walk_at_submission = walk_depth(asks_at_submission, requested_shares, limit_price, fee_config)

    if taker_delay_ms <= 0:
        return TakerOrderResult(
            submit_ts=submit_ts,
            matched_ts=submit_ts,
            was_delayed=False,
            walk=walk_at_submission,
            repriced=False,
            walk_at_submission=None,
        )

    matched_ts = submit_ts + taker_delay_ms / 1000.0
    effective_asks = asks_at_revalidation if asks_at_revalidation is not None else asks_at_submission
    walk_final = walk_depth(effective_asks, requested_shares, limit_price, fee_config)
    repriced = effective_asks != asks_at_submission

    return TakerOrderResult(
        submit_ts=submit_ts,
        matched_ts=matched_ts,
        was_delayed=True,
        walk=walk_final,
        repriced=repriced,
        walk_at_submission=walk_at_submission,
    )
