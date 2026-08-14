"""OneStepController v1 (Roadmap Phase 8 deliverable).

"Select the highest-value action; include WAIT as a candidate with value 0
/ continuation estimate." "Record the complete candidate table for each
decision." Ties together Phase 3 (portfolio kernel), Phase 5 (q), Phase 6
(regime candidate families via `permitted_actions`), and Phase 7 (execution
cost/fill estimates) - this module owns none of that logic itself, only the
candidate generation/selection loop.
"""
from __future__ import annotations

from dataclasses import dataclass

from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.feeds.base import BookSnapshot
from xamarinbot.optimizer.candidates import (
    evaluate_maker_candidate,
    evaluate_taker_candidate,
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
    ) -> OneStepDecision:
        wait = wait_candidate("wait", portfolio)
        candidates: list[CandidateAction] = [wait]

        if not is_fresh:
            # Data gate (Roadmap SS21 risk gates): "No new alpha" when
            # freshness is uncertain - only WAIT is ever offered, full stop.
            return OneStepDecision(round_id, decision_ts, wait, tuple(candidates), skip_reason="stale_data")

        idx = 0
        # Phase 12B audit items 7/8/10: candidate quantities and the
        # worst-price cap are now both derived from the actual book +
        # current risk/position/spend budgets (taker_sizing_boundaries),
        # not raw depth-level sums evaluated against an unconstraining
        # limit_price=1.0. p_max is None only when no level clears
        # min_marginal_edge at all, in which case there is nothing to
        # generate for that side this decision.
        if SeedAction.TAKER_UP in permitted_actions and book_up is not None and book_up.asks:
            sizing = taker_sizing_boundaries(book_up.asks, q, self.fee_config, self.cfg, portfolio, portfolio.U)
            if sizing.p_max is not None:
                for qty in sizing.quantities:
                    idx += 1
                    candidates.append(
                        evaluate_taker_candidate(
                            f"taker_up_{idx}", Side.UP, OrderPurpose.ALPHA, qty, limit_price=sizing.p_max,
                            asks=book_up.asks, portfolio=portfolio, q=q, fee_config=self.fee_config, cfg=self.cfg,
                        )
                    )

        if SeedAction.TAKER_DOWN in permitted_actions and book_down is not None and book_down.asks:
            sizing = taker_sizing_boundaries(book_down.asks, 1.0 - q, self.fee_config, self.cfg, portfolio, portfolio.D)
            if sizing.p_max is not None:
                for qty in sizing.quantities:
                    idx += 1
                    candidates.append(
                        evaluate_taker_candidate(
                            f"taker_down_{idx}", Side.DOWN, OrderPurpose.ALPHA, qty, limit_price=sizing.p_max,
                            asks=book_down.asks, portfolio=portfolio, q=q, fee_config=self.fee_config, cfg=self.cfg,
                        )
                    )

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

        # Roadmap Phase 11 / SS20.1 ablation #6 "taker-only execution":
        # skip maker candidate generation entirely rather than generating
        # and then discarding them, so the candidate table itself reflects
        # what the ablation is testing.
        if not self.cfg.taker_only:
            if SeedAction.MAKER_UP in permitted_actions and book_up is not None and book_up.best_bid and book_up.best_ask:
                for price, offset in maker_price_grid(book_up.best_bid.price, book_up.best_ask.price, tick_size, self.cfg.maker_price_offsets_ticks):
                    idx += 1
                    queue_ahead = book_up.best_bid.size if offset == 0 else 0.0
                    candidates.append(
                        evaluate_maker_candidate(
                            f"maker_up_{idx}", Side.UP, OrderPurpose.ALPHA, price, self.cfg.maker_quantity,
                            distance_to_touch_ticks=float(offset), queue_ahead_shares=queue_ahead, horizon_s=self.cfg.maker_horizon_s,
                            portfolio=portfolio, q=q, exec_cfg=self.exec_cfg, cfg=self.cfg,
                        )
                    )

            if SeedAction.MAKER_DOWN in permitted_actions and book_down is not None and book_down.best_bid and book_down.best_ask:
                for price, offset in maker_price_grid(book_down.best_bid.price, book_down.best_ask.price, tick_size, self.cfg.maker_price_offsets_ticks):
                    idx += 1
                    queue_ahead = book_down.best_bid.size if offset == 0 else 0.0
                    candidates.append(
                        evaluate_maker_candidate(
                            f"maker_down_{idx}", Side.DOWN, OrderPurpose.ALPHA, price, self.cfg.maker_quantity,
                            distance_to_touch_ticks=float(offset), queue_ahead_shares=queue_ahead, horizon_s=self.cfg.maker_horizon_s,
                            portfolio=portfolio, q=q, exec_cfg=self.exec_cfg, cfg=self.cfg,
                        )
                    )

        # WAIT and CANCEL permitted-actions generate nothing beyond the
        # WAIT candidate already present - neither is a "new position"
        # proposal (CANCEL concerns existing open orders, Phase 9's job).

        valid = [c for c in candidates if c.is_valid]
        # SS18: J = E[PnL_T] + lambda_G*G_T - ...; lambda_g=0 (the default)
        # reduces this to plain delta_ev ranking, identical to Phases 8-10.
        chosen = max(valid, key=lambda c: c.delta_ev + self.cfg.lambda_g * c.g_after) if valid else wait
        return OneStepDecision(round_id, decision_ts, chosen, tuple(candidates), skip_reason=None)
