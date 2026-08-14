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
    """The FINAL, actual fill outcome of a taker order. Phase 12B Tranche
    1.2 item 2: this must never be constructed from a future book at
    submission time for a delayed order - see
    `execution/simulator.py::ExecutionSimulator.resolve_taker`, the only
    place a `TakerOrderResult` for a delayed order is ever produced, and
    only once the actual causal book at `matched_ts` is genuinely known
    (i.e. the caller's own simulated clock has reached `matched_ts`)."""

    submit_ts: float
    matched_ts: float
    was_delayed: bool
    walk: DepthWalkResult
    repriced: bool  # book at resolution differed from book at submission (delayed orders only)
    walk_at_submission: DepthWalkResult | None  # what would have filled with no delay, for slippage reporting


def walk_at_submission(
    asks_at_submission: tuple[BookLevel, ...], requested_shares: float, limit_price: float, fee_config: FeeConfig
) -> DepthWalkResult:
    """What would fill right now, against the book already visible at
    `submit_ts` - always legitimate, present-time information, computed
    the same way regardless of whether this order will end up delayed.
    Phase 12B Tranche 1.2 item 2: this is deliberately the ONLY fill
    computation `ExecutionSimulator.submit_taker` performs for a delayed
    order - it must never be treated as the order's actual eventual fill,
    only used as a diagnostic (`TakerOrderResult.walk_at_submission`, for
    slippage reporting) once the order is later resolved."""
    return walk_depth(asks_at_submission, requested_shares, limit_price, fee_config)
