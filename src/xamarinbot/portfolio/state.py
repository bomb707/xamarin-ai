"""Exact portfolio-payoff state and fill application.

Implements Roadmap Phase 3 step "Create PortfolioState with U, D, C, Pi_U,
Pi_D, G, R" and "Update C from actual fill price, size and actual taker fee;
maker fee is zero under current rules but read fee configuration", plus the
identities from Strategy doc SS10 (Exact Portfolio-Control Mathematics).

This module has no dependency on a predictor, exchange client, or optimizer,
per the Phase 3 exit gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class LiquidityRole(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


@dataclass(frozen=True)
class FeeConfig:
    """Runtime fee parameters. crypto_fee_rate is a fallback default (0.07)
    per Strategy doc SS2.3 - production code must read the market's actual
    fee configuration rather than relying on this fallback beyond
    testing/degraded-data scenarios."""

    crypto_fee_rate: float = 0.07

    def taker_fee(self, shares: float, price: float) -> float:
        """fee = shares * feeRate * p * (1-p) [Strategy doc SS2.3]."""
        return shares * self.crypto_fee_rate * price * (1.0 - price)

    def fee_for(self, role: LiquidityRole, shares: float, price: float) -> float:
        """Makers are not charged trading fees; takers are [P1]."""
        if role is LiquidityRole.MAKER:
            return 0.0
        return self.taker_fee(shares, price)


@dataclass(frozen=True)
class Fill:
    """A single confirmed fill used to update exact portfolio state."""

    side: Side
    price: float
    shares: float
    role: LiquidityRole
    fee: float  # actual fee from the fill/market, not recomputed from a formula


@dataclass(frozen=True)
class PortfolioState:
    """Exact economic state: U, D, C, and the derived identities
    Pi_U, Pi_D, G, R (Strategy doc SS4, SS10)."""

    U: float = 0.0
    D: float = 0.0
    C: float = 0.0

    @property
    def Pi_U(self) -> float:
        """Settlement PnL if UP wins = U - C."""
        return self.U - self.C

    @property
    def Pi_D(self) -> float:
        """Settlement PnL if DOWN wins = D - C."""
        return self.D - self.C

    @property
    def G(self) -> float:
        """Worst-case settlement margin = min(U,D) - C = min(Pi_U, Pi_D)."""
        return min(self.U, self.D) - self.C

    @property
    def R(self) -> float:
        """Directional share imbalance = U - D = Pi_U - Pi_D."""
        return self.U - self.D

    @property
    def m(self) -> float:
        """Matched level: m = (U+D)/2 (Strategy doc SS10.1)."""
        return (self.U + self.D) / 2.0

    @property
    def d(self) -> float:
        """Directional half-imbalance: d = (U-D)/2 (Strategy doc SS10.1)."""
        return (self.U - self.D) / 2.0


def apply_fill(state: PortfolioState, fill: Fill) -> PortfolioState:
    """Pure, non-mutating update of portfolio state from one confirmed fill.

    Cost of a fill is its notional (price * shares) plus fee; UP fills add to
    U, DOWN fills add to D, and every fill's total acquisition cost adds to C
    regardless of side, per Strategy doc SS4/SS10 ("C = total actual
    acquisition cost including taker fees").
    """
    cost = fill.price * fill.shares + fill.fee
    if fill.side is Side.UP:
        return PortfolioState(U=state.U + fill.shares, D=state.D, C=state.C + cost)
    return PortfolioState(U=state.U, D=state.D + fill.shares, C=state.C + cost)


def apply_fills(state: PortfolioState, fills: list[Fill]) -> PortfolioState:
    for fill in fills:
        state = apply_fill(state, fill)
    return state
