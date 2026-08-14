"""Candidate-action simulation, risk constraints, and derived sizing formulas.

Implements the remaining Roadmap Phase 3 deliverables:
  - simulateFill(action) without mutating live state
  - alpha-order EV and hedge-order risk contribution separately
  - G_min, P_min, spend and position constraints
  - derived maximum directional spend and hedge quantity formulas

Formulas are taken from Strategy doc SS13 (taker economics), SS17
(hedge/portfolio-repair). This module stays prediction-free: it never
consumes a probability `q`. Upstream EV calculations that need `q` belong to
later phases (Phase 5 probability model, Phase 8 optimizer).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xamarinbot.portfolio.state import Fill, FeeConfig, LiquidityRole, PortfolioState, Side, apply_fill


class OrderPurpose(str, Enum):
    ALPHA = "ALPHA"
    HEDGE = "HEDGE"


@dataclass(frozen=True)
class CandidateFill:
    """A hypothetical fill to simulate against the current portfolio state."""

    side: Side
    price: float
    shares: float
    role: LiquidityRole
    purpose: OrderPurpose


@dataclass(frozen=True)
class FillSimulationResult:
    """Result of simulating one candidate fill without mutating live state."""

    purpose: OrderPurpose
    delta_U: float
    delta_D: float
    delta_C: float
    portfolio_before: PortfolioState
    portfolio_after: PortfolioState
    risk_contribution: float  # G_after - G_before


def simulate_fill(
    state: PortfolioState, candidate: CandidateFill, fee_config: FeeConfig
) -> FillSimulationResult:
    """Simulate a candidate fill against `state` without mutating it.

    Alpha orders and hedge orders are simulated identically at the kernel
    level (the math is purpose-agnostic); the `purpose` tag and
    `risk_contribution` field let upstream callers separate an alpha order's
    expected value from a hedge order's risk-contribution role, per the
    Phase 3 exit gate that the kernel not depend on the predictor.
    """
    fee = fee_config.fee_for(candidate.role, candidate.shares, candidate.price)
    fill = Fill(side=candidate.side, price=candidate.price, shares=candidate.shares, role=candidate.role, fee=fee)
    after = apply_fill(state, fill)

    delta_U = after.U - state.U
    delta_D = after.D - state.D
    delta_C = after.C - state.C

    return FillSimulationResult(
        purpose=candidate.purpose,
        delta_U=delta_U,
        delta_D=delta_D,
        delta_C=delta_C,
        portfolio_before=state,
        portfolio_after=after,
        risk_contribution=after.G - state.G,
    )


# --------------------------------------------------------------------------
# Risk gates: G_min, P_min, spend and position constraints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConstraints:
    """Configured risk gates a candidate action must pass (Strategy doc SS18,
    Roadmap Phase 3 step "Implement G_min, P_min, spend and position
    constraints.")."""

    g_min: float  # worst-case settlement PnL floor
    p_min: float | None = None  # favored-side profit floor, if used
    spend_cap: float | None = None  # B_max: max additional spend this round
    position_limit: float | None = None  # max shares per side


@dataclass(frozen=True)
class ConstraintCheck:
    passed: bool
    violated: tuple[str, ...]


def evaluate_constraints(
    result: FillSimulationResult,
    constraints: RiskConstraints,
    favored_side: Side | None = None,
) -> ConstraintCheck:
    """Reject hard-constraint violations for a simulated candidate action.

    `spend_cap` is a round-level budget (`RiskConstraints.spend_cap`
    docstring: "max additional spend this round"), so it must be checked
    against *cumulative* cost this round, not one candidate's incremental
    cost alone (Phase 12B audit item 9: comparing `result.delta_C` - a
    single fill's own cost - against the full cap let any number of
    individually-legal orders exceed the intended round budget in total).
    Every caller in this codebase constructs a fresh `PortfolioState()` at
    the start of each round and never carries one across rounds (verified
    across every call site), so `result.portfolio_after.C` already *is*
    cumulative round spend including this candidate - no separate
    `round_start_C` needs to be threaded through for that invariant to
    hold. If a future caller ever reuses one `PortfolioState` across
    multiple rounds/markets, this check would need `C` measured relative
    to that round's own starting `C`, not zero.
    """
    violated: list[str] = []
    after = result.portfolio_after

    if after.G < constraints.g_min:
        violated.append("g_min")

    if constraints.p_min is not None and favored_side is not None:
        pi_favored = after.Pi_U if favored_side is Side.UP else after.Pi_D
        if pi_favored < constraints.p_min:
            violated.append("p_min")

    if constraints.spend_cap is not None and after.C > constraints.spend_cap:
        violated.append("spend_cap")

    if constraints.position_limit is not None:
        if after.U > constraints.position_limit or after.D > constraints.position_limit:
            violated.append("position_limit")

    return ConstraintCheck(passed=not violated, violated=tuple(violated))


# --------------------------------------------------------------------------
# Derived maximum directional spend and hedge quantity formulas
# (Strategy doc SS13, SS17)
# --------------------------------------------------------------------------


def max_directional_spend(g_current: float, g_min: float) -> float:
    """Maximum directional risk-budget spend: K(x) <= G_current - G_min.

    Phase 12B Tranche 1.1 item 5: this is an EXACT boundary only in the
    special case where buying more of `side` does not change
    `min(U, D)` - i.e. `side` is already the non-minimum ("overrepresented")
    side, so `G_side(x) = G_current - K_side(x)` collapses to this flat
    per-dollar budget. It is NOT a universal quantity boundary: when
    `side` is currently the *minimum* side, buying it raises `min(U, D)`
    itself as `x` grows, and this flat formula understates how much can
    safely be bought (see the regression case in
    `test_taker_sizing_uses_exact_side_aware_g_when_buying_the_underrepresented_side`:
    U=0, D=100, C=50, g_min=-100 - the flat budget here is
    G_current - g_min = (min(0,100)-50) - (-100) = $50, capping spend at
    ~96.6 shares, although buying 100 UP shares at ~$0.5175/share all-in
    is actually risk-feasible, since it also raises `min(U,D)` from 0
    toward 100).
    Use `directional_projected_g` for the exact, side-aware formula that
    holds in every case; `optimizer/candidates.py::taker_sizing_boundaries`
    is the source of truth for taker quantity feasibility and does not use
    this function."""
    return g_current - g_min


def directional_projected_g(u: float, d: float, c: float, side: Side, x: float, k_x: float) -> float:
    """Exact side-aware projected worst-case settlement margin after
    buying `x` additional shares on `side` at cumulative (fee-inclusive)
    cost `k_x`, replacing `max_directional_spend`'s flat
    `G_current - g_min` budget (only exact when `side` is already the
    non-minimum side - see that function's docstring) with the general
    formula, exact for every portfolio state (Phase 12B Tranche 1.1
    item 5):

        G_U(x) = min(U+x, D) - [C + K_U(x)]
        G_D(x) = min(U, D+x) - [C + K_D(x)]

    Proof this matches the existing, tested `G = min(U,D) - C` kernel: for
    a one-sided UP fill, `U' = U+x`, `D' = D`, `C' = C + K_U(x)`, so
    `G' = min(U',D') - C' = min(U+x,D) - (C+K_U(x))` exactly, by
    definition - no new kernel, just the existing one evaluated at the
    post-fill state (same method already used for `directional_projected_g`'s
    one-sided BUFFER_BUILD sibling, `ΔG_U(x)`, in the Phase 12B audit's
    Addendum F-H)."""
    if side is Side.UP:
        return min(u + x, d) - (c + k_x)
    return min(u, d + x) - (c + k_x)


def min_hedge_quantity(l_max: float, pi_down: float, c_d: float) -> float:
    """Minimum DOWN hedge to restore loss floor:
    x_min_hedge = max(0, (-L_max - Pi_DOWN) / (1 - c_D))."""
    return max(0.0, (-l_max - pi_down) / (1.0 - c_d))


def max_hedge_quantity(pi_up: float, p_min: float, c_d: float) -> float:
    """Maximum hedge before violating favored-side profit floor:
    x_max_hedge = (Pi_UP - P_min) / c_D."""
    return (pi_up - p_min) / c_d


def hedge_efficiency(c: float) -> float:
    """Downside protection gained per unit favored-side profit sacrificed:
    HedgeEfficiency(c) = (1-c)/c."""
    return (1.0 - c) / c
