"""Aggregate risk exposure from not-yet-confirmed active orders (Phase 12B
Tranche 1.2 items 5/6, corrected and completed in Tranche 2A).

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
candidate` is not. `PendingExposure`/`RiskView` are separate, additive
views used only for admission checks - never merged into `PortfolioState`
itself, and no code in this module mutates confirmed state.

Phase 12B Tranche 2A item 1 correction: an earlier version of this module
computed worst-case G by assuming every active order fills SIMULTANEOUSLY
is the worst case. That is false for `G = min(U,D) - C`: a two-sided fill
(one order on each side) can *raise* min(U,D) enough to more than offset
its own cost, so "everything fills" is sometimes the BEST case, not the
worst. Concretely, at `U=D=C=0` with a pending 100 UP@0.40 and a pending
100 DOWN@0.40: both filling gives `U=D=100, C=80, G=+20`; only the UP one
filling gives `U=100, D=0, C=40, G=-40` - strictly worse, even though
fewer orders filled. The true worst case is `min` over every FILL SUBSET,
not the all-fill subset alone - see `PendingExposure.worst_case_g`.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.portfolio.state import FeeConfig, LiquidityRole, PortfolioState, Side

# Above this many active orders, exact 2^n subset enumeration is skipped in
# favor of the conservative analytical bound (see worst_case_g) - kept
# small since every extra order doubles the enumeration cost, and no
# admission path in this codebase should ever have more than a handful of
# simultaneously active orders (TakerOrderQueue's own gate already limits
# pending takers to at most one; open makers are bounded by the maker
# concurrency the controller actually generates).
DEFAULT_EXACT_SUBSET_CAP = 10


@dataclass(frozen=True)
class ActiveOrderExposure:
    """One active (not-yet-terminal) order's own worst-case contribution
    if it alone fills in full - the atomic unit
    `PendingExposure.worst_case_g`'s scenario envelope enumerates subsets
    over. `max_cost` must already be fee-inclusive (Phase 12B Tranche 2A
    item 2) - the same all-in definition of cost `apply_fill` uses for
    confirmed fills (`price*shares + fee`), so hard-risk admission never
    under-counts what a fill would actually cost."""

    side: Side
    max_fill_qty: float
    max_cost: float


@dataclass(frozen=True)
class PendingExposure:
    """Aggregate exposure from every active order this decision hasn't
    already accounted for via confirmed fills. `orders` is the underlying
    per-order list the scenario envelope (`worst_case_g`) enumerates over;
    the flat `max_committed_spend`/`potential_up_shares`/
    `potential_down_shares` fields are plain sums, kept for cheap
    reporting/summary use where the full envelope isn't needed - unlike
    worst-case G, worst-case cumulative SPEND genuinely is just "every
    order fills" (cost never decreases when more orders fill), so summing
    `max_cost` is exact for spend_cap admission, not an approximation."""

    max_committed_spend: float = 0.0
    potential_up_shares: float = 0.0
    potential_down_shares: float = 0.0
    n_pending_delayed_takers: int = 0
    n_open_maker_orders: int = 0
    orders: tuple[ActiveOrderExposure, ...] = ()

    def worst_case_g(self, confirmed: PortfolioState, exact_subset_cap: int = DEFAULT_EXACT_SUBSET_CAP) -> float:
        """`G_reserved = min over subsets S of self.orders of
        G(confirmed + fills(S))` (Phase 12B Tranche 2A item 1) - the
        mathematically correct replacement for the old (wrong)
        "everything fills simultaneously" assumption. Exact subset
        enumeration (`2**n` scenarios) while `len(orders) <=
        exact_subset_cap`; beyond that, falls back to the conservative
        analytical bound

            G_reserved = min(U,D) - (C + sum(K_i_max))

        which charges every order's worst-case cost but credits NO
        protective share contribution from any of them - always a valid
        lower bound on the true worst case (never claims a pending order
        has already protected the portfolio), just not necessarily tight."""
        n = len(self.orders)
        if n == 0:
            return confirmed.G
        if n <= exact_subset_cap:
            worst: float | None = None
            for mask in range(1 << n):
                u, d, c = confirmed.U, confirmed.D, confirmed.C
                for i, order in enumerate(self.orders):
                    if mask & (1 << i):
                        if order.side is Side.UP:
                            u += order.max_fill_qty
                        else:
                            d += order.max_fill_qty
                        c += order.max_cost
                g = min(u, d) - c
                if worst is None or g < worst:
                    worst = g
            assert worst is not None
            return worst
        total_cost = sum(o.max_cost for o in self.orders)
        return min(confirmed.U, confirmed.D) - (confirmed.C + total_cost)

    def __add__(self, other: "PendingExposure") -> "PendingExposure":
        return PendingExposure(
            max_committed_spend=self.max_committed_spend + other.max_committed_spend,
            potential_up_shares=self.potential_up_shares + other.potential_up_shares,
            potential_down_shares=self.potential_down_shares + other.potential_down_shares,
            n_pending_delayed_takers=self.n_pending_delayed_takers + other.n_pending_delayed_takers,
            n_open_maker_orders=self.n_open_maker_orders + other.n_open_maker_orders,
            orders=self.orders + other.orders,
        )


NO_EXPOSURE = PendingExposure()


def exposure_from_pending_takers(pending_orders, fee_config: FeeConfig) -> PendingExposure:
    """Builds a `PendingExposure` from a list of
    `execution.simulator.PendingTakerOrder` objects still awaiting
    resolution (`not p.is_resolved`). Worst case per order: it fills in
    full at its own `limit_price`, fee included (Phase 12B Tranche 2A
    item 2 - `K_taker_max = x*p_limit + Fee(x, p_limit)`, not just
    `x*p_limit`) - `TakerOrderQueue`'s own conservative admission gate
    (Phase 12B Tranche 1.2 item 5: at most one `PENDING_DELAY` taker
    outstanding at a time) means this will almost always sum over zero or
    one order in practice today, but the accounting itself does not
    assume that - it sums correctly over any number, for when a fuller
    concurrent-order model exists."""
    orders = []
    spend = 0.0
    up = 0.0
    down = 0.0
    n = 0
    for p in pending_orders:
        if p.is_resolved:
            continue
        fee = fee_config.taker_fee(p.requested_shares, p.limit_price)
        worst_cost = p.requested_shares * p.limit_price + fee
        spend += worst_cost
        if p.side is Side.UP:
            up += p.requested_shares
        else:
            down += p.requested_shares
        n += 1
        orders.append(ActiveOrderExposure(side=p.side, max_fill_qty=p.requested_shares, max_cost=worst_cost))
    return PendingExposure(
        max_committed_spend=spend, potential_up_shares=up, potential_down_shares=down,
        n_pending_delayed_takers=n, orders=tuple(orders),
    )


def exposure_from_open_maker_orders(tracked_orders, fee_config: FeeConfig) -> PendingExposure:
    """Builds a `PendingExposure` from a list of open
    `supervisor.types.TrackedOrder` (or anything exposing the same
    `.order_state.side`/`.order_state.remaining_shares`/
    `.order_state.limit_price` shape). Same worst-case-full-fill, fee-
    inclusive convention as `exposure_from_pending_takers` - maker fees
    are always 0 today (`FeeConfig.fee_for(MAKER, ...)`), but routing
    through `fee_config` keeps this the same all-in cost definition
    `apply_fill` uses, not a hardcoded assumption that stays correct only
    by coincidence.

    Phase 12B Tranche 2A item 4: wired into actual admission via
    `RiskView` (see `optimizer/controller.py` / the ablations/order-
    supervisor round loops) - "several currently open maker orders
    filling" is now checked jointly before a new maker candidate is
    admitted, not just individually-safe-at-placement-time."""
    orders = []
    spend = 0.0
    up = 0.0
    down = 0.0
    n = 0
    for tracked in tracked_orders:
        order_state = tracked.order_state
        remaining = order_state.remaining_shares
        if remaining <= 0:
            continue
        fee = fee_config.fee_for(LiquidityRole.MAKER, remaining, order_state.limit_price)
        worst_cost = remaining * order_state.limit_price + fee
        spend += worst_cost
        if order_state.side is Side.UP:
            up += remaining
        else:
            down += remaining
        n += 1
        orders.append(ActiveOrderExposure(side=order_state.side, max_fill_qty=remaining, max_cost=worst_cost))
    return PendingExposure(
        max_committed_spend=spend, potential_up_shares=up, potential_down_shares=down,
        n_open_maker_orders=n, orders=tuple(orders),
    )


@dataclass(frozen=True)
class RiskView:
    """The single source of truth for HARD risk admission decisions
    (Phase 12B Tranche 2A item 3) - distinct from the probabilistic EV/
    expected-fill model candidates use for ranking.

        ExpectedState != HardRiskState.

    `confirmed_portfolio` (+ a candidate's own expected/probabilistic
    fill) is what EV/ranking calculations should keep using;
    `worst_case_g`/`committed_spend` (+ a candidate's own worst-case
    fill, via `admits`) is what hard `g_min`/`spend_cap` gates must use.
    Never conflate the two - a candidate can be EV-ranked using its
    expected fill while still being hard-rejected using its worst case."""

    confirmed_portfolio: PortfolioState
    pending_taker_exposure: PendingExposure = NO_EXPOSURE
    open_maker_exposure: PendingExposure = NO_EXPOSURE

    @property
    def combined_exposure(self) -> PendingExposure:
        return self.pending_taker_exposure + self.open_maker_exposure

    @property
    def committed_spend(self) -> float:
        """Confirmed C plus every active order's worst-case (fee-
        inclusive) cost, were all of them to fill - exact for spend_cap
        admission (see `PendingExposure.worst_case_g`'s docstring on why
        spend, unlike G, has "all fill" as its genuine worst case)."""
        return self.confirmed_portfolio.C + self.combined_exposure.max_committed_spend

    @property
    def potential_up_position(self) -> float:
        return self.confirmed_portfolio.U + self.combined_exposure.potential_up_shares

    @property
    def potential_down_position(self) -> float:
        return self.confirmed_portfolio.D + self.combined_exposure.potential_down_shares

    @property
    def worst_case_g(self) -> float:
        return self.combined_exposure.worst_case_g(self.confirmed_portfolio)

    def admits(
        self, candidate: ActiveOrderExposure, g_min: float, spend_cap: float | None,
        position_limit: float | None = None, is_recovery_candidate: bool = False,
        exact_subset_cap: int = DEFAULT_EXACT_SUBSET_CAP,
    ) -> bool:
        """Would admitting `candidate` (as one more active order,
        alongside every order already tracked in this view) ever risk a
        `g_min`, `spend_cap`, or `position_limit` breach under some fill
        subset? Runs the full scenario envelope over `existing orders
        UNION {candidate}` - not "candidate always fills," since the
        worst subset might or might not include it, exactly like any
        other order (Phase 12B Tranche 2A item 1's fix applies here too).

        `position_limit` (Phase 12B Tranche 2.1 item 2 - previously
        exposed via `potential_up_position`/`potential_down_position` but
        never actually enforced here) uses a plain sum over every UP/DOWN
        order, not the subset envelope `worst_g` needs: unlike `G`, a
        side's potential position can only ever grow as more orders on
        that side fill (no non-monotonic min() interaction), so "every
        order on this side fills" genuinely is the worst (and only
        relevant) case for a position ceiling.

        `is_recovery_candidate` (Phase 12B Tranche 2.2 item 1) fixes a
        real contradiction with `_finalize`'s breach-recovery semantics
        (`optimizer/candidates.py`): the aggregate worst-case envelope's
        own subset enumeration includes the empty-fill subset, so once
        `worst_case_g` (BEFORE this candidate) is already below `g_min`,
        `worst_g >= g_min` can never hold again no matter what the
        candidate does - literally every recovery candidate would be
        aggregate-rejected, contradicting `_finalize`'s explicit
        allowance for a partial repair that improves G without fully
        restoring it. Two modes, matching `_finalize` exactly:

          - SAFE (worst_g_base >= g_min): preserve the ordinary hard
            rule, worst_g_new >= g_min, for every candidate regardless
            of purpose.
          - RECOVERY (worst_g_base < g_min): a non-recovery candidate
            (`is_recovery_candidate=False` - the caller's own ALPHA-vs-
            HEDGE/BUFFER_BUILD purpose check) is rejected outright -
            "ALPHA remains prohibited while breached" applies at the
            aggregate level too, not just the single-candidate one. A
            recovery candidate is admitted iff it does not WORSEN the
            aggregate envelope: worst_g_new >= worst_g_base."""
        combined_orders = self.combined_exposure.orders + (candidate,)
        worst_g_new = PendingExposure(orders=combined_orders).worst_case_g(self.confirmed_portfolio, exact_subset_cap)
        worst_g_base = self.worst_case_g

        if worst_g_base >= g_min:
            if worst_g_new < g_min:
                return False
        else:
            if not is_recovery_candidate:
                return False
            if worst_g_new < worst_g_base:
                return False
        if spend_cap is not None:
            total_worst_spend = self.confirmed_portfolio.C + sum(o.max_cost for o in combined_orders)
            if total_worst_spend > spend_cap:
                return False
        if position_limit is not None:
            up_potential = self.confirmed_portfolio.U + sum(o.max_fill_qty for o in combined_orders if o.side is Side.UP)
            down_potential = self.confirmed_portfolio.D + sum(o.max_fill_qty for o in combined_orders if o.side is Side.DOWN)
            if up_potential > position_limit or down_potential > position_limit:
                return False
        return True
