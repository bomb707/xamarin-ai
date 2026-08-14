"""CandidateAction (Roadmap Phase 8 deliverable; the record shape is
Strategy doc SS23.3's `CandidateAction` interface almost verbatim: "action
record: {purpose, side, mode, price, qty, TTL, expectedFill, EV_after,
G_after, Pi_U_after, Pi_D_after}")."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import Side


class OrderMode(str, Enum):
    """Strategy doc SS12 control vector domain: "POST-ONLY / GTC / GTD /
    FAK / FOK / WAIT / CANCEL." Phase 8's one-step candidates only ever
    generate FAK (taker), POST_ONLY (maker), and WAIT - the others are
    declared for interface completeness but not produced here; GTC/GTD/FOK/
    CANCEL are Phase 9 (order-lifecycle) territory."""

    FAK = "FAK"
    POST_ONLY = "POST_ONLY"
    WAIT = "WAIT"
    GTC = "GTC"
    GTD = "GTD"
    FOK = "FOK"
    CANCEL = "CANCEL"


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    purpose: OrderPurpose
    side: Side | None  # None for WAIT
    mode: OrderMode
    price: float | None
    qty: float
    ttl_s: float | None
    expected_fill: float  # expected filled shares (qty for taker if depth suffices, qty*rho for maker)
    ev_after: float
    g_after: float
    pi_u_after: float
    pi_d_after: float
    violated_constraints: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return not self.violated_constraints
