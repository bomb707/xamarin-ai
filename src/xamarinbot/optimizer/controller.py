"""OneStepController v1 (Roadmap Phase 8 deliverable).

"Select the highest-value action; include WAIT as a candidate with value 0
/ continuation estimate." "Record the complete candidate table for each
decision." Ties together Phase 3 (portfolio kernel), Phase 5 (q), Phase 6
(regime candidate families via `permitted_actions`), and Phase 7 (execution
cost/fill estimates) - this module owns none of that logic itself, only the
candidate generation/selection loop.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.feeds.base import BookSnapshot
from xamarinbot.optimizer.candidates import (
    dynamic_maker_sizing,
    evaluate_maker_candidate,
    evaluate_taker_candidate,
    generate_buffer_build_candidates,
    generate_hedge_candidate,
    maker_price_grid,
    taker_sizing_boundaries,
    wait_candidate,
)
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.types import CandidateAction
from xamarinbot.portfolio.math import OrderPurpose
from xamarinbot.portfolio.state import FeeConfig, PortfolioState, Side
from xamarinbot.regime.types import SeedAction


@dataclass(frozen=True)
class OneStepDecision:
    round_id: str
    decision_ts: float
    chosen: CandidateAction
    candidates: tuple[CandidateAction, ...]
    skip_reason: str | None  # set (and chosen forced to WAIT) when is_fresh is False


class OneStepController:
    def __init__(self, cfg: OneStepConfig, exec_cfg: ExecutionConfig, fee_config: FeeConfig):
        self.cfg = cfg
        self.exec_cfg = exec_cfg
        self.fee_config = fee_config

    def _family_gate(self, family: SeedAction, permitted_actions: frozenset[SeedAction]) -> tuple[bool, float]:
        """Returns `(should_generate, selection_penalty)` for one
        directional candidate family (Phase 12B Tranche 2C). Hard mode
        (`cfg.soft_regime=False`, the default - unchanged from Phases
        6-12B behavior): only generate the family if the regime
        classifier's `permitted_actions` includes it; no candidate exists
        at all otherwise, so it can never be chosen no matter how
        economically attractive. Soft mode (`cfg.soft_regime=True`):
        ALWAYS generate the family, applying `cfg.regime_prior_penalty`
        to the selection score (not to `delta_ev` itself) when the regime
        did not permit it - an economically strong candidate can still
        win selection, just penalized rather than structurally invisible
        to the optimizer's own `argmax`. Portfolio-repair (HEDGE) and
        BUFFER_BUILD are unaffected by either mode - they are generated
        regardless of `permitted_actions` already (a portfolio-level
        decision, not a directional thesis the seed regime classifier has
        an opinion on), matching the reviewer's own instruction that
        "portfolio repair and buffer actions must not be blocked simply
        because the directional seed state says WAIT.\""""
        permitted = family in permitted_actions
        if self.cfg.soft_regime:
            return True, (0.0 if permitted else self.cfg.regime_prior_penalty)
        return permitted, 0.0

    def decide(
        self,
        round_id: str,
        decision_ts: float,
        portfolio: PortfolioState,
        q: float,
        permitted_actions: frozenset[SeedAction],
        book_up: BookSnapshot | None,
        book_down: BookSnapshot | None,
        tick_size: float,
        is_fresh: bool,
        tau: float = 0.0,
        sigma: float = 0.0,
    ) -> OneStepDecision:
        """`tau`/`sigma` (Phase 12B Tranche 2D) are optional and default
        to 0.0 - existing callers that don't pass them are unaffected,
        since they're only consulted when `cfg.dynamic_maker=True`
        (itself defaulting False). Callers that want dynamic maker
        sizing should pass `tau=fv.tau, sigma=fv.realized_vol` from the
        same `FeatureVector` already computed for `q`."""
        wait = wait_candidate("wait", portfolio)
        candidates: list[CandidateAction] = [wait]

        if not is_fresh:
            # Data gate (Roadmap SS21 risk gates): "No new alpha" when
            # freshness is uncertain - only WAIT is ever offered, full stop.
            return OneStepDecision(round_id, decision_ts, wait, tuple(candidates), skip_reason="stale_data")

        idx = 0
        # Phase 12B audit items 7/8/10, Tranche 1.2 item 3: candidate
        # quantities and each quantity's OWN hard, risk-safe worst-price
        # limit are derived from the actual book + current risk/position/
        # spend budgets (taker_sizing_boundaries), not raw depth-level
        # sums evaluated against an unconstraining limit_price=1.0 or a
        # single shared depth/marginal-edge boundary. p_max is None only
        # when no level clears min_marginal_edge at all, in which case
        # there is nothing to generate for that side this decision.
        taker_up_ok, taker_up_penalty = self._family_gate(SeedAction.TAKER_UP, permitted_actions)
        if taker_up_ok and book_up is not None and book_up.asks:
            sizing = taker_sizing_boundaries(book_up.asks, q, self.fee_config, self.cfg, portfolio, Side.UP, tick_size)
            if sizing.p_max is not None:
                for qty in sizing.quantities:
                    idx += 1
                    limit_price = sizing.max_execution_price_by_qty.get(qty, sizing.p_max)
                    c = evaluate_taker_candidate(
                        f"taker_up_{idx}", Side.UP, OrderPurpose.ALPHA, qty, limit_price=limit_price,
                        asks=book_up.asks, portfolio=portfolio, q=q, fee_config=self.fee_config, cfg=self.cfg,
                    )
                    candidates.append(replace(c, selection_penalty=taker_up_penalty) if taker_up_penalty else c)

        taker_down_ok, taker_down_penalty = self._family_gate(SeedAction.TAKER_DOWN, permitted_actions)
        if taker_down_ok and book_down is not None and book_down.asks:
            sizing = taker_sizing_boundaries(book_down.asks, 1.0 - q, self.fee_config, self.cfg, portfolio, Side.DOWN, tick_size)
            if sizing.p_max is not None:
                for qty in sizing.quantities:
                    idx += 1
                    limit_price = sizing.max_execution_price_by_qty.get(qty, sizing.p_max)
                    c = evaluate_taker_candidate(
                        f"taker_down_{idx}", Side.DOWN, OrderPurpose.ALPHA, qty, limit_price=limit_price,
                        asks=book_down.asks, portfolio=portfolio, q=q, fee_config=self.fee_config, cfg=self.cfg,
                    )
                    candidates.append(replace(c, selection_penalty=taker_down_penalty) if taker_down_penalty else c)

        if self.cfg.enable_portfolio_repair:
            # Hedge candidates are evaluated regardless of the regime's
            # permitted_actions - portfolio repair (SS17) is a portfolio-
            # level decision about existing imbalance, not a directional
            # thesis the seed regime classifier has an opinion on.
            asks_up = book_up.asks if book_up is not None else ()
            asks_down = book_down.asks if book_down is not None else ()
            hedge = generate_hedge_candidate("hedge", portfolio, asks_up, asks_down, q, self.fee_config, self.cfg)
            if hedge is not None:
                candidates.append(hedge)

        if self.cfg.enable_buffer_build:
            # Phase 12B Tranche 2B: same regime-independence as HEDGE
            # above - a portfolio-geometry decision, not a directional one.
            asks_up = book_up.asks if book_up is not None else ()
            asks_down = book_down.asks if book_down is not None else ()
            candidates.extend(generate_buffer_build_candidates("buffer_build", portfolio, asks_up, asks_down, q, self.fee_config, self.cfg))

        # Roadmap Phase 11 / SS20.1 ablation #6 "taker-only execution":
        # skip maker candidate generation entirely rather than generating
        # and then discarding them, so the candidate table itself reflects
        # what the ablation is testing.
        if not self.cfg.taker_only:
            if self.cfg.dynamic_maker:
                up_qty, up_ttl, up_offsets = dynamic_maker_sizing(q, Side.UP, tau, sigma, portfolio, self.cfg)
                down_qty, down_ttl, down_offsets = dynamic_maker_sizing(q, Side.DOWN, tau, sigma, portfolio, self.cfg)
            else:
                up_qty = down_qty = self.cfg.maker_quantity
                up_ttl = down_ttl = self.cfg.maker_horizon_s
                up_offsets = down_offsets = self.cfg.maker_price_offsets_ticks

            maker_up_ok, maker_up_penalty = self._family_gate(SeedAction.MAKER_UP, permitted_actions)
            if maker_up_ok and book_up is not None and book_up.best_bid and book_up.best_ask:
                for price, offset in maker_price_grid(book_up.best_bid.price, book_up.best_ask.price, tick_size, up_offsets):
                    idx += 1
                    queue_ahead = book_up.best_bid.size if offset == 0 else 0.0
                    c = evaluate_maker_candidate(
                        f"maker_up_{idx}", Side.UP, OrderPurpose.ALPHA, price, up_qty,
                        distance_to_touch_ticks=float(offset), queue_ahead_shares=queue_ahead, horizon_s=up_ttl,
                        portfolio=portfolio, q=q, exec_cfg=self.exec_cfg, cfg=self.cfg,
                    )
                    candidates.append(replace(c, selection_penalty=maker_up_penalty) if maker_up_penalty else c)

            maker_down_ok, maker_down_penalty = self._family_gate(SeedAction.MAKER_DOWN, permitted_actions)
            if maker_down_ok and book_down is not None and book_down.best_bid and book_down.best_ask:
                for price, offset in maker_price_grid(book_down.best_bid.price, book_down.best_ask.price, tick_size, down_offsets):
                    idx += 1
                    queue_ahead = book_down.best_bid.size if offset == 0 else 0.0
                    c = evaluate_maker_candidate(
                        f"maker_down_{idx}", Side.DOWN, OrderPurpose.ALPHA, price, down_qty,
                        distance_to_touch_ticks=float(offset), queue_ahead_shares=queue_ahead, horizon_s=down_ttl,
                        portfolio=portfolio, q=q, exec_cfg=self.exec_cfg, cfg=self.cfg,
                    )
                    candidates.append(replace(c, selection_penalty=maker_down_penalty) if maker_down_penalty else c)

        # WAIT and CANCEL permitted-actions generate nothing beyond the
        # WAIT candidate already present - neither is a "new position"
        # proposal (CANCEL concerns existing open orders, Phase 9's job).

        valid = [c for c in candidates if c.is_valid]
        # SS18: J = E[PnL_T] + lambda_G*G_T - ...; lambda_g=0 (the default)
        # reduces this to plain delta_ev ranking, identical to Phases
        # 8-10. selection_penalty (Tranche 2C soft-regime mode) is 0.0 for
        # every candidate unless cfg.soft_regime=True, so this is a no-op
        # in the default/hard-gate configuration.
        chosen = max(valid, key=lambda c: c.delta_ev + self.cfg.lambda_g * c.g_after - c.selection_penalty) if valid else wait
        return OneStepDecision(round_id, decision_ts, chosen, tuple(candidates), skip_reason=None)
