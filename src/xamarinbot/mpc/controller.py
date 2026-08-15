"""MPCController v1 (Roadmap Phase 10 deliverable).

"At each event, it predicts a short sequence of possible market/portfolio
states, optimizes a sequence of actions, executes only the first action,
then re-solves on the next event" (Strategy doc SS15). Continuation value
at each future state is estimated by the *greedy one-step policy itself*
(Phase 8's `OneStepController`, called recursively) rather than a second,
independent search - a standard rollout-policy approximation, and exactly
what "Recommended first implementation: one-step / short-horizon MPC with
discrete candidate actions" describes.

Simplifications, all documented at point of use: continuation projections
hold CLOB/spot direction fixed, evolving only `GapRegime` via Phase 10's
transition model; maker candidates' continuation uses their *if-filled*
portfolio uniformly (their own `delta_ev`, used for selection, already
correctly probability-weights the fill itself - this only affects the
continuation estimate, not the immediate decision). The order book *depth*
is NOT held fully fixed - see `_deplete_book` below; only holding depth
fixed turned out to double-count consumed liquidity (see docs/PHASE_STATUS.md).

Phase 12B Tranche 2.2 item 2: the hypothetical rollout tree (`_rollout`)
has no exposure-transition model of its own - when real active exposure
(pending takers, open makers) exists, `decide()` falls back to the plain
risk-aware one-step decision entirely rather than exploring hypothetical
future states without knowing how that exposure evolves. This keeps MPC
EXPERIMENTAL/NON-PRODUCTION until real data validates whether the deeper
rollout is worth a proper exposure-propagation model - see item 2's
"prefer the simpler safe solution" instruction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from xamarinbot.feeds.base import BookLevel, BookSnapshot
from xamarinbot.mpc.config import MPCConfig
from xamarinbot.mpc.scenario import TransitionModel
from xamarinbot.mpc.types import MPCDecision
from xamarinbot.optimizer.candidates import candidate_selection_score
from xamarinbot.market.constraints import MarketConstraints
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import CandidateAction, OrderMode
from xamarinbot.portfolio.exposure import RiskView
from xamarinbot.portfolio.state import PortfolioState, Side
from xamarinbot.regime.matrix import ActionPermissionMatrix, classify_seed_action
from xamarinbot.regime.types import RegimeState


class _DeadlineExceeded(Exception):
    pass


def _portfolio_after_candidate(portfolio: PortfolioState, candidate: CandidateAction) -> PortfolioState:
    """Reconstructs the post-fill PortfolioState from a CandidateAction's
    reported g_after/pi_u_after/pi_d_after (the SS23.3 interface doesn't
    carry raw U/D/C deltas) - exact, not approximate: each candidate only
    ever moves one side, so (side, Pi_U_after, Pi_D_after) plus the
    unchanged other side fully determines (U, D, C)."""
    if candidate.side is None:
        return portfolio
    if candidate.side is Side.UP:
        d_after = portfolio.D
        c_after = d_after - candidate.pi_d_after
        u_after = candidate.pi_u_after + c_after
        return PortfolioState(U=u_after, D=d_after, C=c_after)
    u_after = portfolio.U
    c_after = u_after - candidate.pi_u_after
    d_after = candidate.pi_d_after + c_after
    return PortfolioState(U=u_after, D=d_after, C=c_after)


def _deplete_book(book: BookSnapshot | None, shares_consumed: float) -> BookSnapshot | None:
    """Removes `shares_consumed` from the front of `book`'s ask depth.
    Without this, a continuation rollout re-walks the *original* book every
    step, so a large taker fill's continuation value would double-count
    exactly the liquidity it just consumed - effectively assuming the same
    cheap depth refills with zero latency within the same horizon step,
    which is precisely the kind of unrealistic fill Phase 7's depth-walking
    and 250ms-delay machinery exists to rule out."""
    if book is None or shares_consumed <= 1e-12:
        return book
    remaining = shares_consumed
    new_asks: list[BookLevel] = []
    for level in book.asks:
        if remaining <= 1e-9:
            new_asks.append(level)
            continue
        take = min(remaining, level.size)
        remaining -= take
        new_size = level.size - take
        if new_size > 1e-9:
            new_asks.append(BookLevel(level.price, new_size))
    return BookSnapshot(side=book.side, bids=book.bids, asks=tuple(new_asks), ts=book.ts, recv_ts=book.recv_ts, book_hash=book.book_hash)


def _deplete_books_for_candidate(
    book_up: BookSnapshot | None, book_down: BookSnapshot | None, candidate: CandidateAction
) -> tuple[BookSnapshot | None, BookSnapshot | None]:
    """Only FAK (taker) fills consume book depth - WAIT and resting maker
    orders leave the ask side untouched."""
    if candidate.mode is not OrderMode.FAK or candidate.side is None or candidate.expected_fill <= 0:
        return book_up, book_down
    if candidate.side is Side.UP:
        return _deplete_book(book_up, candidate.expected_fill), book_down
    return book_up, _deplete_book(book_down, candidate.expected_fill)


@dataclass
class MPCController:
    cfg: MPCConfig
    transition_model: TransitionModel
    one_step: OneStepController

    def decide(
        self,
        round_id: str,
        decision_ts: float,
        portfolio: PortfolioState,
        q: float,
        regime_state: RegimeState,
        book_up: BookSnapshot | None,
        book_down: BookSnapshot | None,
        constraints: MarketConstraints,
        is_fresh: bool,
        risk_view: RiskView | None = None,
    ) -> MPCDecision:
        """`risk_view` (Phase 12B Tranche 2.2 item 2) is forwarded into the
        underlying one-step call for the IMMEDIATE action, exactly like
        `OneStepController.decide` itself - the action MPC ultimately
        returns must already be aggregate-risk-admissible, not merely
        rejected by dispatch after winning selection here too. If real
        active exposure is tracked in `risk_view` (pending takers or open
        makers), the deeper rollout is skipped entirely and this risk-aware
        one-step decision is returned directly (`used_fallback=True`) -
        the hypothetical future states `_rollout` explores have no model
        for how that exposure evolves, so exploring them without one would
        silently ignore it exactly where it matters most."""
        start = time.monotonic()
        permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(regime_state))
        base = self.one_step.decide(round_id, decision_ts, portfolio, q, permitted, book_up, book_down, constraints, is_fresh, risk_view=risk_view)

        if self.cfg.horizon_steps <= 1:
            # Degenerate horizon (Phase 10 verification: "MPC action
            # equals one-step action in degenerate horizon") - no
            # continuation term, so total sequence value is just the
            # candidate's own selection score, the same ranking Phase 8
            # already computed.
            sequence_values = {c.action_id: candidate_selection_score(c, self.one_step.cfg) for c in base.candidates}
            return MPCDecision(round_id, decision_ts, base.chosen, base.candidates, sequence_values, used_fallback=False, elapsed_ms=(time.monotonic() - start) * 1000.0)

        if risk_view is not None and risk_view.combined_exposure.orders:
            return MPCDecision(round_id, decision_ts, base.chosen, base.candidates, {}, used_fallback=True, elapsed_ms=(time.monotonic() - start) * 1000.0)

        deadline = start + self.cfg.time_budget_ms / 1000.0
        sequence_values: dict[str, float] = {}
        try:
            for candidate in base.candidates:
                if not candidate.is_valid:
                    sequence_values[candidate.action_id] = float("-inf")
                    continue
                if time.monotonic() > deadline:
                    raise _DeadlineExceeded()
                portfolio_after = _portfolio_after_candidate(portfolio, candidate)
                book_up_after, book_down_after = _deplete_books_for_candidate(book_up, book_down, candidate)
                continuation = 0.0
                for next_gap, prob in self.transition_model.next_states(regime_state.gap_regime, self.cfg.transition_top_k):
                    next_state = RegimeState(next_gap, regime_state.clob_direction, regime_state.spot_direction)
                    continuation += prob * self._rollout(
                        round_id, portfolio_after, next_state, q, book_up_after, book_down_after, constraints, is_fresh,
                        self.cfg.horizon_steps - 1, deadline,
                    )
                sequence_values[candidate.action_id] = candidate_selection_score(candidate, self.one_step.cfg) + continuation
        except _DeadlineExceeded:
            # "Bound computation time; fall back to one-step controller if
            # deadline exceeded" - the plain Phase 8 decision, unmodified.
            return MPCDecision(round_id, decision_ts, base.chosen, base.candidates, {}, used_fallback=True, elapsed_ms=(time.monotonic() - start) * 1000.0)

        best_id = max(sequence_values, key=lambda action_id: sequence_values[action_id])
        chosen = next(c for c in base.candidates if c.action_id == best_id)
        return MPCDecision(round_id, decision_ts, chosen, base.candidates, sequence_values, used_fallback=False, elapsed_ms=(time.monotonic() - start) * 1000.0)

    def _rollout(
        self,
        round_id: str,
        portfolio: PortfolioState,
        regime_state: RegimeState,
        q: float,
        book_up: BookSnapshot | None,
        book_down: BookSnapshot | None,
        constraints: MarketConstraints,
        is_fresh: bool,
        depth: int,
        deadline: float,
    ) -> float:
        """Expected continuation value from this hypothetical future state
        with `depth` steps remaining, using the greedy one-step policy as
        the rollout policy at every level."""
        if depth <= 0:
            return 0.0
        if time.monotonic() > deadline:
            raise _DeadlineExceeded()

        permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(regime_state))
        decision = self.one_step.decide(round_id, 0.0, portfolio, q, permitted, book_up, book_down, constraints, is_fresh)
        value = candidate_selection_score(decision.chosen, self.one_step.cfg)
        if depth > 1:
            portfolio_after = _portfolio_after_candidate(portfolio, decision.chosen)
            book_up_after, book_down_after = _deplete_books_for_candidate(book_up, book_down, decision.chosen)
            for next_gap, prob in self.transition_model.next_states(regime_state.gap_regime, self.cfg.transition_top_k):
                next_state = RegimeState(next_gap, regime_state.clob_direction, regime_state.spot_direction)
                value += prob * self._rollout(round_id, portfolio_after, next_state, q, book_up_after, book_down_after, constraints, is_fresh, depth - 1, deadline)
        return value
