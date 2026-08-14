"""Aggregate risk exposure from not-yet-confirmed active orders (Phase 12B
Tranche 1.2 items 5/6).

`PortfolioState` (portfolio/state.py) represents CONFIRMED fills only, and
must stay that way - it is the exact settlement-payoff kernel every
identity in `tests/test_portfolio_math.py` is proven against. But a risk
admission check that only ever looks at confirmed state is blind to
orders that are still "in flight" and may yet turn into more fills after
the current decision:

  - a taker order in `PENDING_DELAY` (Phase 7's 250ms delay window)
  - a resting maker order (not yet filled, not yet expired/canceled)
  - a REPLACE's new order, mid-flight and not yet terminal

A new candidate must not be admitted merely because
`confirmed + candidate` is safe while `confirmed + pending_exposure +
candidate` is not. `PendingExposure` is a separate, additive view used
only for admission checks - it is never merged into `PortfolioState`
itself, and no code in this module mutates confirmed state.

This module is the interface Tranche 2's more dynamic maker
architecture will build on (item 6) - it does not yet redesign maker
pricing/TTL, and the maker-side builder below is deliberately a thin,
conservative accounting pass, not a new maker economics model.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.portfolio.state import PortfolioState, Side


@dataclass(frozen=True)
class PendingExposure:
    """Worst-case aggregate exposure from every active (not-yet-terminal)
    order this decision hasn't already accounted for via confirmed fills.

    `max_committed_spend` is the worst-case additional `C` if every
    pending order fills in full at its own worst-case (limit) price;
    `potential_up_shares`/`potential_down_shares` are the worst-case
    additional `U`/`D`. These are deliberately upper bounds (using each
    order's `limit_price`/`requested_shares`, not an expected value) -
    exposure accounting for an admission gate should be conservative, not
    a best estimate."""

    max_committed_spend: float = 0.0
    potential_up_shares: float = 0.0
    potential_down_shares: float = 0.0
    n_pending_delayed_takers: int = 0
    n_open_maker_orders: int = 0

    def worst_case_portfolio(self, confirmed: PortfolioState) -> PortfolioState:
        """`confirmed + this exposure`, applied as if every pending order
        fills in full, simultaneously - the conservative state a new
        candidate must remain risk-safe against (rather than only against
        `confirmed` alone)."""
        return PortfolioState(
            U=confirmed.U + self.potential_up_shares,
            D=confirmed.D + self.potential_down_shares,
            C=confirmed.C + self.max_committed_spend,
        )

    def __add__(self, other: "PendingExposure") -> "PendingExposure":
        return PendingExposure(
            max_committed_spend=self.max_committed_spend + other.max_committed_spend,
            potential_up_shares=self.potential_up_shares + other.potential_up_shares,
            potential_down_shares=self.potential_down_shares + other.potential_down_shares,
            n_pending_delayed_takers=self.n_pending_delayed_takers + other.n_pending_delayed_takers,
            n_open_maker_orders=self.n_open_maker_orders + other.n_open_maker_orders,
        )


NO_EXPOSURE = PendingExposure()


def exposure_from_pending_takers(pending_orders) -> PendingExposure:
    """Builds a `PendingExposure` from a list of
    `execution.simulator.PendingTakerOrder` objects still awaiting
    resolution (`not p.is_resolved`). Worst case per order: it fills in
    full at its own `limit_price` (the most it could possibly cost / the
    most shares it could possibly add) - `TakerOrderQueue`'s own
    conservative admission gate (Phase 12B Tranche 1.2 item 5: at most
    one `PENDING_DELAY` taker outstanding at a time) means this will
    almost always sum over zero or one order in practice today, but the
    accounting itself does not assume that - it sums correctly over any
    number, for when a fuller concurrent-order model exists."""
    spend = 0.0
    up = 0.0
    down = 0.0
    n = 0
    for p in pending_orders:
        if p.is_resolved:
            continue
        worst_cost = p.requested_shares * p.limit_price
        spend += worst_cost
        if p.side is Side.UP:
            up += p.requested_shares
        else:
            down += p.requested_shares
        n += 1
    return PendingExposure(max_committed_spend=spend, potential_up_shares=up, potential_down_shares=down, n_pending_delayed_takers=n)


def exposure_from_open_maker_orders(tracked_orders) -> PendingExposure:
    """Builds a `PendingExposure` from a list of open
    `supervisor.types.TrackedOrder` (or anything exposing the same
    `.order_state.side`/`.order_state.remaining_shares`/
    `.order_state.limit_price` shape). Same worst-case-full-fill
    convention as `exposure_from_pending_takers`.

    Phase 12B Tranche 1.2 item 6: this is the accounting primitive
    Tranche 2's dynamic maker placement will need before it can safely
    generate more concurrent maker orders than today's harnesses do -
    "several currently open maker orders filling" is not yet checked
    against jointly by any admission path in this codebase (each order
    was only ever checked individually-safe at its own placement time).
    Wiring this into an actual admission gate for maker candidates is
    explicitly deferred to Tranche 2, per that item's own instruction not
    to redesign maker pricing/TTL yet - this function exists so that
    wiring has a ready, tested building block to call."""
    spend = 0.0
    up = 0.0
    down = 0.0
    n = 0
    for tracked in tracked_orders:
        order_state = tracked.order_state
        remaining = order_state.remaining_shares
        if remaining <= 0:
            continue
        worst_cost = remaining * order_state.limit_price
        spend += worst_cost
        if order_state.side is Side.UP:
            up += remaining
        else:
            down += remaining
        n += 1
    return PendingExposure(max_committed_spend=spend, potential_up_shares=up, potential_down_shares=down, n_open_maker_orders=n)
