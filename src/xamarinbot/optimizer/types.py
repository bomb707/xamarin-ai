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
    price: float | None  # average execution price from this candidate's own evaluation-time walk (FAK) or quoted price (maker)
    qty: float  # desired_shares, per Phase 12B audit item 10's execution contract
    ttl_s: float | None
    expected_fill: float  # expected filled shares (qty for taker if depth suffices, qty*rho for maker)
    delta_ev: float
    g_after: float
    pi_u_after: float
    pi_d_after: float
    violated_constraints: tuple[str, ...] = field(default_factory=tuple)
    # Phase 12B audit item 10: the worst acceptable execution price this
    # candidate was evaluated against (FAK only; None for maker/WAIT,
    # where it doesn't apply the same way) - `price` above is the
    # resulting *average* fill price from evaluation, not this cap, so a
    # caller re-executing the chosen candidate through the real order
    # lifecycle (item 13) needs this field to preserve the same
    # worst-price protection rather than re-deriving or guessing a limit.
    max_execution_price: float | None = None
    # Phase 12B Tranche 2C: soft-regime selection penalty - nonzero only
    # when `cfg.soft_regime=True` and this candidate's own family was NOT
    # in the regime classifier's hard-permitted set. Subtracted from the
    # selection score (never from `delta_ev` itself, which must keep
    # meaning "expected PnL") so a regime-disfavored-but-still-valid
    # candidate can still win selection if its edge is large enough,
    # rather than never existing at all (the old hard-gate behavior,
    # still available as `cfg.soft_regime=False`, the default).
    selection_penalty: float = 0.0

    @property
    def is_valid(self) -> bool:
        return not self.violated_constraints
