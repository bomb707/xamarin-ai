"""Order-request translation boundary (Phase 12C.1 item 14).

Polymarket's market-order API encodes the two sides differently: a FAK/FOK
**BUY** is expressed as a dollar `amount`, while a **SELL** is expressed as a
share quantity. The strategy, meanwhile, optimizes desired filled shares `x`
throughout - every EV, risk and constraint expression in `portfolio/math.py`
and `optimizer/candidates.py` is written in shares.

The temptation is to let the dollar encoding leak inward, so sizing starts
happening in notional. That would quietly break the minimum-order rule,
because the venue's minimum is a SHARE quantity (item 13): a request that is
comfortably above a dollar threshold can still be below the share minimum.

So the encoding is confined to exactly one place - here - and the invariant

    x >= min_order_shares

is asserted on the SHARE quantity, independently of how the request happens
to be encoded on the wire.

NOTHING IN PHASE 12C.1 SUBMITS THESE. This module builds request objects and
validates them. There is no client, no signing, no network call, and
`tests/test_import_boundaries.py` asserts no real entry point references any
order-placing symbol.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.market.constraints import MarketConstraints
from xamarinbot.portfolio.state import Side


class OrderRequestError(ValueError):
    """Raised instead of emitting a request the venue would reject."""


@dataclass(frozen=True)
class MarketBuyRequest:
    """A FAK/FOK BUY, in the venue's dollar encoding.

    `amount_usdc` is the wire field. `desired_shares` and
    `max_execution_price` are retained alongside it precisely so the share
    intent survives the translation and can be re-checked downstream - the
    dollar figure alone cannot tell you whether the order meets the venue's
    share minimum.
    """

    token_id: str
    side: Side
    amount_usdc: float
    desired_shares: float
    max_execution_price: float
    max_all_in_cost: float

    @property
    def implied_shares_at_cap(self) -> float:
        """Shares this amount buys if every share fills at the worst
        acceptable price - the conservative reading, and the one that must
        clear the venue minimum."""
        return self.amount_usdc / self.max_execution_price


@dataclass(frozen=True)
class MarketSellRequest:
    """A SELL, in the venue's share encoding - no translation needed."""

    token_id: str
    side: Side
    shares: float
    min_execution_price: float


def build_market_buy_request(
    side: Side,
    desired_shares: float,
    max_execution_price: float,
    max_all_in_cost: float,
    constraints: MarketConstraints,
) -> MarketBuyRequest:
    """Translate a share-denominated intent into the venue's dollar encoding.

    Raises rather than rounding: item 13 is explicit that an order below the
    market minimum is not an executable candidate, and that generating none
    is correct where rounding upward would break the risk/economic constraint
    the size was derived from.
    """
    if desired_shares <= 0:
        raise OrderRequestError(f"desired_shares must be positive, got {desired_shares}")
    if not 0.0 < max_execution_price <= 1.0:
        raise OrderRequestError(
            f"max_execution_price must be in (0, 1], got {max_execution_price}"
        )
    if not constraints.admits_size(desired_shares):
        raise OrderRequestError(
            f"{desired_shares} shares is below the market minimum of "
            f"{constraints.min_order_shares} shares; generate no order rather than "
            "rounding up to the minimum, which would breach the risk/economic "
            "constraint this size was derived from"
        )

    # The dollar amount that buys `desired_shares` even in the worst
    # acceptable execution, capped by the caller's all-in budget.
    amount = min(desired_shares * max_execution_price, max_all_in_cost)
    if amount <= 0:
        raise OrderRequestError(
            f"translated BUY amount is {amount}; max_all_in_cost={max_all_in_cost} "
            "leaves nothing to spend"
        )

    request = MarketBuyRequest(
        token_id=constraints.token_for(side),
        side=side,
        amount_usdc=amount,
        desired_shares=desired_shares,
        max_execution_price=max_execution_price,
        max_all_in_cost=max_all_in_cost,
    )

    # The share invariant must survive the dollar encoding. If the all-in
    # budget clipped the amount below what the minimum share size costs, the
    # request is not executable no matter how it is encoded.
    if not constraints.admits_size(request.implied_shares_at_cap):
        raise OrderRequestError(
            f"a {amount:.6f} USDC BUY buys only "
            f"{request.implied_shares_at_cap:.6f} shares at the worst acceptable "
            f"price {max_execution_price}, below the market minimum of "
            f"{constraints.min_order_shares} shares"
        )
    return request


def build_market_sell_request(
    side: Side,
    shares: float,
    min_execution_price: float,
    constraints: MarketConstraints,
) -> MarketSellRequest:
    """SELL is already share-denominated; only the minimum is checked."""
    if shares <= 0:
        raise OrderRequestError(f"shares must be positive, got {shares}")
    if not constraints.admits_size(shares):
        raise OrderRequestError(
            f"{shares} shares is below the market minimum of "
            f"{constraints.min_order_shares} shares"
        )
    return MarketSellRequest(
        token_id=constraints.token_for(side),
        side=side,
        shares=shares,
        min_execution_price=min_execution_price,
    )
