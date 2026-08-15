"""Mandatory ablation matrix (Roadmap Phase 11 / Strategy doc SS20.1,
"Mandatory ablations" - the list below is that section almost verbatim):

  1. Baseline three-way unanimous strategy.
  2. V2 TWAP-only probability model.
  3. V2 current-BTC-only model.
  4. V2 TWAP + current-BTC lead-lag model.
  5. Lead-lag + CLOB without portfolio repair.
  6. Full model with portfolio control but taker-only execution.
  7. Full model with maker/taker timing and cancel/replace.
  8. Full MPC versus one-step optimizer.

Each ablation is a runnable config over the *same* causal replay, not a
separate strategy implementation - #2-8 differ only in which `FeatureSet`
feeds `q`, which `OneStepConfig` flags are on, and whether Phase 9's
`OrderSupervisor` or Phase 10's `MPCController` wraps the decision. This is
the first place all ten phases run side by side against one shared harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.baseline.inputs import elapsed_t
from xamarinbot.baseline.strategy import BaselineInputs, decide as baseline_decide
from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.simulator import ExecutionSimulator, TakerOrderQueue
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor, MockSpotFeed, MockTWAPFeed
from xamarinbot.model.features import COMBINED_LEAD_LAG, LEAD_LAG_ONLY, SPOT_ONLY, TWAP_ONLY, FeatureSet, design_vector
from xamarinbot.model.calibrated import CalibratedModel
from xamarinbot.model.logistic import LogisticModel
from xamarinbot.mpc.config import MPCConfig
from xamarinbot.mpc.controller import MPCController
from xamarinbot.mpc.scenario import TransitionModel
from xamarinbot.optimizer.candidates import candidate_exposure, evaluate_maker_candidate, evaluate_replacement_plan, is_recovery_purpose
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import OrderMode
from xamarinbot.portfolio.exposure import ActiveOrderExposure, RiskView, exposure_from_open_maker_orders
from xamarinbot.portfolio.state import Fill, FeeConfig, LiquidityRole, PortfolioState, Side, apply_fill
from xamarinbot.regime.classifier import RegimeClassifier
from xamarinbot.regime.matrix import ActionPermissionMatrix, classify_seed_action
from xamarinbot.supervisor.config import SupervisorConfig
from xamarinbot.supervisor.supervisor import OrderSupervisor
from xamarinbot.supervisor.types import SupervisorActionType, TrackedOrder

HEARTBEAT_S = 10.0


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    controller: str  # "baseline" | "one_step" | "mpc"
    feature_set: FeatureSet | None = None  # None only for "baseline"
    one_step_cfg: OneStepConfig | None = None
    use_supervisor: bool = False
    mpc_cfg: MPCConfig | None = None


def _default_one_step_cfg(**overrides) -> OneStepConfig:
    base = dict(g_min=-100.0, spend_cap=200.0, position_limit=200.0, edge_min=0.0)
    base.update(overrides)
    return OneStepConfig(**base)


# Phase 12B Tranche 2.1 item 12: TEST_ONLY_LAMBDA_G is a coefficient
# selected by observing THIS repository's own synthetic dataset behavior
# (originally to keep both taker and maker candidates viable under the
# pre-Tranche-2.1 selection formula - see docs/PHASE_STATUS.md's "Notable
# bugs" for the original magnitude analysis, superseded by item 6's
# expected_delta_g correction). It is an ablation-harness knob to exercise
# the enable_portfolio_repair/lambda_g code paths in THIS synthetic
# dataset, not a value with any claim to being a correct or calibrated
# production setting - never promote it into a real OneStepConfig outside
# this ablation matrix.
TEST_ONLY_LAMBDA_G = 0.01


MANDATORY_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("1_baseline_unanimous", "Baseline three-way unanimous strategy (Phase 0)", controller="baseline"),
    AblationSpec("2_twap_only", "V2 TWAP-only probability model", controller="one_step", feature_set=TWAP_ONLY, one_step_cfg=_default_one_step_cfg()),
    AblationSpec("3_spot_only", "V2 current-BTC-only model", controller="one_step", feature_set=SPOT_ONLY, one_step_cfg=_default_one_step_cfg()),
    AblationSpec("4_lead_lag", "V2 TWAP + current-BTC lead-lag model", controller="one_step", feature_set=LEAD_LAG_ONLY, one_step_cfg=_default_one_step_cfg()),
    AblationSpec("5_lead_lag_clob_no_repair", "Lead-lag + CLOB without portfolio repair", controller="one_step", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=False)),
    AblationSpec("6_portfolio_control_taker_only", "Full model with portfolio control but taker-only execution", controller="one_step", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=True, taker_only=True, lambda_g=TEST_ONLY_LAMBDA_G)),
    AblationSpec("7_maker_taker_cancel_replace", "Full model with maker/taker timing and cancel/replace", controller="one_step", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=True, lambda_g=TEST_ONLY_LAMBDA_G), use_supervisor=True),
    AblationSpec("8_full_mpc", "Full MPC versus one-step optimizer", controller="mpc", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=True, lambda_g=TEST_ONLY_LAMBDA_G), use_supervisor=True, mpc_cfg=MPCConfig(horizon_steps=2, time_budget_ms=50.0)),
)


@dataclass
class RoundResult:
    round_id: str
    realized_pnl: float
    n_actions: int
    final_g: float
    fill_rate: float  # actions filled / actions attempted (1.0 for baseline/taker-certain fills)


def run_ablation_round(
    spec: AblationSpec,
    store: EventStore,
    round_id: str,
    p0: float,
    outcome: Side,
    feature_cfg: FeatureConfig,
    fee_config: FeeConfig,
    exec_cfg: ExecutionConfig,
    model: LogisticModel | CalibratedModel | None,
    transition_model: TransitionModel | None,
    baseline_cfg: BaselineConfig | None = None,
) -> RoundResult:
    if spec.controller == "baseline":
        portfolio, n_actions, n_attempts = _run_baseline_round(store, round_id, p0, fee_config, exec_cfg, baseline_cfg or BaselineConfig())
        pnl = portfolio.Pi_U if outcome is Side.UP else portfolio.Pi_D
        fill_rate = (n_actions / n_attempts) if n_attempts else 0.0
        return RoundResult(round_id, pnl, n_actions, portfolio.G, fill_rate)

    portfolio, n_actions, n_attempts = _run_controller_round(spec, store, round_id, p0, feature_cfg, fee_config, exec_cfg, model, transition_model)
    pnl = portfolio.Pi_U if outcome is Side.UP else portfolio.Pi_D
    fill_rate = (n_actions / n_attempts) if n_attempts else 0.0
    return RoundResult(round_id, pnl, n_actions, portfolio.G, fill_rate)


def _run_baseline_round(
    store: EventStore, round_id: str, p0: float, fee_config: FeeConfig, exec_cfg: ExecutionConfig, cfg: BaselineConfig
) -> tuple[PortfolioState, int, int]:
    """Phase 12B Tranche 1 items 2/D: this function previously (a) passed
    the same value for `spot` and `spot_prev` (making `spot_direction`
    always 0, so the unanimity check could never pass) and (b) passed
    absolute `decision_ts` as `t` instead of elapsed round time (making
    `OUTSIDE_DECISION_WINDOW` fire for the entire duration of every round
    whose `start_ts != 0`). Both are fixed together below, mirroring the
    already-correct pattern in `scripts/run_baseline_replay.py` (which
    never had either bug) via the shared `elapsed_t()` helper and a
    second lookback cursor for spot, matching the one already used here
    for `clob_mid_prev`.

    Items 13/E/L, Tranche 1.2 item 1: fills are now routed through the
    same `ExecutionSimulator.submit_taker`/`resolve_taker` real depth-walk
    lifecycle every V2 arm uses (`_run_controller_round` below), via the
    shared `TakerOrderQueue`, instead of assuming the baseline's full
    requested quantity always fills completely at its quoted limit price
    - a materially easier execution assumption than every other ablation
    arm had, confirmed and fixed together with the other two baseline-
    harness bugs above so all arms share one execution engine per item L
    ("same book + same latency + same depth + same fee model... only
    strategy decisions should differ"). Baseline and V2 sharing the exact
    same chronological delay/resolution lifecycle (not just the same
    depth-walk math) matters specifically because they are compared
    head-to-head."""
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)
    twap_feed, spot_feed, book_feed = MockTWAPFeed(cursor), MockSpotFeed(cursor), MockBookFeed(cursor)
    prev_clob_cursor = MockFeedCursor(store, round_id, preloaded=events)
    prev_book_feed = MockBookFeed(prev_clob_cursor)
    prev_spot_cursor = MockFeedCursor(store, round_id, preloaded=events)
    prev_spot_feed = MockSpotFeed(prev_spot_cursor)
    # Dedicated cursor/book_feed for fetching the actual causal book at a
    # delayed taker order's matched_ts, only at resolve time (Phase 12B
    # Tranche 1.2 item 2) - kept separate from the decision-time cursor
    # above since it advances to a different timestamp.
    revalidation_cursor = MockFeedCursor(store, round_id, preloaded=events)
    revalidation_book_feed = MockBookFeed(revalidation_cursor)
    portfolio = PortfolioState()
    sim = ExecutionSimulator(round_id, fee_config, exec_cfg)
    queue = TakerOrderQueue(sim)
    n_actions = 0
    n_attempts = 0
    order_seq = 0

    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    round_start_ts = market_config["start_ts"]

    def _book_at(pending) -> tuple:
        revalidation_cursor.advance_to(pending.matched_ts)
        book = revalidation_book_feed.get_snapshot(round_id, pending.side)
        return book.asks if book is not None else ()

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)

        for pending, taker_result in queue.resolve_ready(decision_ts, _book_at):
            if taker_result.walk.filled_shares > 0:
                portfolio = apply_fill(portfolio, Fill(pending.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))
                n_actions += 1

        twap_obs, spot_obs = twap_feed.get_latest(round_id), spot_feed.get_latest(round_id)
        if twap_obs is None or spot_obs is None:
            continue
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)
        if book_up is None or not book_up.best_bid or not book_up.best_ask:
            continue
        prev_clob_cursor.advance_to(decision_ts - cfg.clob_lookback_s)
        prev_book_up = prev_book_feed.get_snapshot(round_id, Side.UP)
        mid = (book_up.best_bid.price + book_up.best_ask.price) / 2.0
        mid_prev = ((prev_book_up.best_bid.price + prev_book_up.best_ask.price) / 2.0) if prev_book_up and prev_book_up.best_bid and prev_book_up.best_ask else mid

        prev_spot_cursor.advance_to(decision_ts - cfg.spot_lookback_s)
        prev_spot_obs = prev_spot_feed.get_latest(round_id)
        spot_prev = prev_spot_obs.value if prev_spot_obs is not None else spot_obs.value

        inputs = BaselineInputs(
            t=elapsed_t(decision_ts, round_start_ts), p0=p0, twap=twap_obs.value, clob_mid=mid, clob_mid_prev=mid_prev,
            spot=spot_obs.value, spot_prev=spot_prev,
            best_ask_up=book_up.best_ask.price, best_ask_down=book_down.best_ask.price if book_down and book_down.best_ask else None,
            is_fresh=True,
        )
        result = baseline_decide(inputs, portfolio.U, portfolio.D, portfolio.C, cfg)
        if result.order is not None and not queue.has_pending:
            n_attempts += 1
            order_seq += 1
            book = book_up if result.order.side is Side.UP else book_down
            asks = book.asks if book is not None else ()
            pending = queue.try_submit(f"{round_id}-o{order_seq}", result.order.side, result.order.quantity, result.order.price, asks, decision_ts)
            if pending is not None and not pending.was_delayed:
                taker_result = sim.resolve_taker(pending)
                if taker_result.walk.filled_shares > 0:
                    portfolio = apply_fill(portfolio, Fill(result.order.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, result.order.role, taker_result.walk.total_fee))
                    n_actions += 1
    return portfolio, n_actions, n_attempts


def _run_controller_round(
    spec: AblationSpec,
    store: EventStore,
    round_id: str,
    p0: float,
    feature_cfg: FeatureConfig,
    fee_config: FeeConfig,
    exec_cfg: ExecutionConfig,
    model: LogisticModel | CalibratedModel | None,
    transition_model: TransitionModel | None,
) -> tuple[PortfolioState, int, int]:
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)
    book_feed = MockBookFeed(cursor)
    # Dedicated cursor/book_feed for fetching the actual causal book at a
    # delayed taker order's matched_ts, only at resolve time (Phase 12B
    # Tranche 1.2 item 2) - kept separate from the main decision-time
    # cursor above since it advances to a different timestamp, mirroring
    # the prev_cursor pattern already used for baseline lookback in this
    # same module.
    revalidation_cursor = MockFeedCursor(store, round_id, preloaded=events)
    revalidation_book_feed = MockBookFeed(revalidation_cursor)
    regime_clf = RegimeClassifier(round_id=round_id)
    one_step = OneStepController(spec.one_step_cfg, exec_cfg, fee_config)
    mpc = MPCController(spec.mpc_cfg, transition_model or TransitionModel(probabilities={}), one_step) if spec.controller == "mpc" else None
    supervisor = OrderSupervisor(SupervisorConfig(g_min=spec.one_step_cfg.g_min, edge_min=spec.one_step_cfg.edge_min)) if spec.use_supervisor else None
    sim = ExecutionSimulator(round_id, fee_config, exec_cfg)
    queue = TakerOrderQueue(sim)
    portfolio = PortfolioState()
    n_actions = 0
    n_attempts = 0
    order_seq = 0

    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    tick_size = market_config["tick_size"]

    def _book_at(pending) -> tuple:
        revalidation_cursor.advance_to(pending.matched_ts)
        book = revalidation_book_feed.get_snapshot(round_id, pending.side)
        return book.asks if book is not None else ()

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)

        # Resolve any delayed taker orders whose matched_ts has arrived -
        # never mutate portfolio for a fill whose matched_ts is still in
        # the future (Phase 12B Tranche 1.1 item 7 / Tranche 1.2 item 2).
        for pending, taker_result in queue.resolve_ready(decision_ts, _book_at):
            if taker_result.walk.filled_shares > 0:
                portfolio = apply_fill(portfolio, Fill(pending.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))
                n_actions += 1

        fv = compute(events, round_id, decision_ts, p0, feature_cfg)
        if not isinstance(fv, FeatureVector):
            continue
        snapshot = regime_clf.observe(fv)
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)
        vec = design_vector(fv, spec.feature_set) if model is not None else None
        q = model.predict_proba(vec) if (model is not None and vec is not None) else 0.5

        if supervisor is not None:
            for order_id in list(supervisor.open_order_ids()):
                tracked = supervisor.orders[order_id]
                book = book_up if tracked.order_state.side is Side.UP else book_down
                if book is None or not book.best_bid or not book.best_ask:
                    continue
                horizon = tracked.ttl_s or spec.one_step_cfg.maker_horizon_s
                if decision_ts - tracked.submit_ts >= horizon:
                    draw = sim.draw_maker_fill(tracked.order_state, 0.0, 0.0, horizon)
                    n_attempts += 1
                    if draw.filled:
                        shares = tracked.order_state.remaining_shares
                        tracked.order_state.reconcile_fill(decision_ts, shares)
                        portfolio = apply_fill(portfolio, Fill(tracked.order_state.side, tracked.order_state.limit_price, shares, LiquidityRole.MAKER, 0.0))
                        n_actions += 1
                    else:
                        tracked.order_state.cancel(decision_ts)
                    del supervisor.orders[order_id]
                    continue
                # Phase 12B audit item 12/18: recompute this order's actual
                # current economics (remaining shares, current q, current
                # portfolio, current if-filled G, current delta_ev) instead
                # of the hardcoded current_delta_ev=0.0 / current portfolio.G
                # placeholders this call used before - mirrors the already-
                # correct pattern in scripts/run_order_supervisor_demo.py.
                current = evaluate_maker_candidate(
                    f"review-{order_id}", tracked.order_state.side, tracked.purpose, tracked.order_state.limit_price,
                    tracked.order_state.remaining_shares, distance_to_touch_ticks=0.0, queue_ahead_shares=0.0,
                    horizon_s=horizon, portfolio=portfolio, q=q, exec_cfg=exec_cfg, cfg=spec.one_step_cfg,
                )
                # Phase 12B Tranche 2.1 item 9: a real ReplacementPlan,
                # re-evaluated against the current book/portfolio, so
                # REPLACE can actually fire through this harness instead of
                # `current_optimal_ev` staying permanently None.
                replacement = evaluate_replacement_plan(
                    tracked.order_state.side, tracked.order_state.remaining_shares, book.best_bid.price, book.best_ask.price,
                    tick_size, spec.one_step_cfg.maker_price_offsets_ticks, horizon, portfolio, q, exec_cfg, fee_config, spec.one_step_cfg,
                )
                current_optimal_ev = replacement.delta_ev if replacement is not None else None
                decision = supervisor.review_order(tracked, decision_ts, snapshot.state, current.delta_ev, current.g_after, fv.tau, True, current_optimal_ev)
                if decision.action is SupervisorActionType.CANCEL:
                    supervisor.apply_cancel(decision, decision_ts)
                elif decision.action is SupervisorActionType.REPLACE and replacement is not None:
                    # Item 2/3: the replacement must clear the SAME
                    # aggregate hard-risk bar any other new order would -
                    # built EXCLUDING this order's own exposure (it is
                    # being torn up, not staying open alongside the
                    # replacement).
                    other_makers = [t for oid, t in supervisor.orders.items() if oid != order_id]
                    replace_risk_view = RiskView(
                        portfolio, pending_taker_exposure=queue.exposure,
                        open_maker_exposure=exposure_from_open_maker_orders(other_makers, fee_config),
                    )
                    if replace_risk_view.admits(replacement.exposure, spec.one_step_cfg.g_min, spec.one_step_cfg.spend_cap, spec.one_step_cfg.position_limit, is_recovery_candidate=is_recovery_purpose(tracked.purpose)):
                        order_seq += 1
                        result = supervisor.apply_replace(decision, decision_ts, f"{round_id}-o{order_seq}", replacement.price, replacement.qty)
                        if result and result.new_order is not None:
                            supervisor.register(TrackedOrder(
                                result.new_order, snapshot.state, tracked.purpose, q, tracked.fair_value_at_submit,
                                current.g_after, replacement.delta_ev, replacement.ttl_s, decision_ts, decision_ts,
                            ))
                    # else: the replacement itself would breach aggregate
                    # risk - hold the original order rather than tear it up
                    # with nothing safe to replace it with.

        # Phase 12B Tranche 2.1 items 2/3: one shared RiskView, combining
        # every currently pending taker and open maker order, threaded
        # into decide() itself (so an aggregate-unsafe candidate is
        # excluded from selection before argmax runs, not merely rejected
        # after winning it) AND re-checked at dispatch for both taker and
        # maker submission - "every new executable order must be admitted
        # against confirmed + pending + resting + candidate before
        # submission," not just maker placement.
        risk_view = RiskView(
            portfolio, pending_taker_exposure=queue.exposure,
            open_maker_exposure=exposure_from_open_maker_orders(list(supervisor.orders.values()) if supervisor is not None else [], fee_config),
        )

        permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(snapshot.state))
        if mpc is not None:
            # Phase 12B Tranche 2.2 item 2: risk_view threaded into MPC's
            # immediate decision the same way it is into one-step's -
            # MPC's own decide() forwards it into the underlying
            # OneStepController call for the immediate action AND falls
            # back to that risk-aware one-step decision entirely if real
            # active exposure exists (see MPCController.decide's own
            # docstring for why the deeper rollout doesn't get its own
            # exposure-transition model).
            mpc_decision = mpc.decide(round_id, decision_ts, portfolio, q, snapshot.state, book_up, book_down, tick_size, True, risk_view=risk_view)
            chosen = mpc_decision.chosen
        else:
            one_step_decision = one_step.decide(round_id, decision_ts, portfolio, q, permitted, book_up, book_down, tick_size, True, risk_view=risk_view)
            chosen = one_step_decision.chosen

        if chosen.mode is OrderMode.FAK and chosen.qty > 0 and not queue.has_pending:
            # Phase 12B audit items 13/E/L, Tranche 1.1 item 7, Tranche 1.2
            # items 1/2/5, Tranche 2.1 item 2: route through the real
            # submit->(delay)->resolve lifecycle via the shared
            # TakerOrderQueue instead of directly converting the
            # candidate's own pre-evaluation walk estimate into a Fill -
            # and, like maker dispatch below, re-check aggregate hard
            # admission immediately before submission (defense in depth on
            # top of decide()'s own pre-selection check above - nothing
            # changes between the two within one synchronous decision, but
            # every dispatch site must independently clear this gate per
            # the reviewer's own instruction, not rely solely on upstream
            # filtering). A genuinely delayed order is queued and only
            # mutates portfolio once this loop's own decision_ts reaches
            # its matched_ts (resolved at the top of the loop above),
            # using the actual causal book fetched only then - never the
            # submission book, and never computed/leaked at submit time.
            # The `not queue.has_pending` guard is the conservative item 5
            # admission gate: at most one PENDING_DELAY taker outstanding.
            limit_price = chosen.max_execution_price if chosen.max_execution_price is not None else chosen.price
            exposure = candidate_exposure(chosen, fee_config)
            if exposure is not None and risk_view.admits(exposure, spec.one_step_cfg.g_min, spec.one_step_cfg.spend_cap, spec.one_step_cfg.position_limit, is_recovery_candidate=is_recovery_purpose(chosen.purpose)):
                n_attempts += 1
                order_seq += 1
                asks = book_up.asks if chosen.side is Side.UP else book_down.asks
                new_pending = queue.try_submit(f"{round_id}-o{order_seq}", chosen.side, chosen.qty, limit_price, asks, decision_ts)
                if new_pending is not None and not new_pending.was_delayed:
                    taker_result = sim.resolve_taker(new_pending)
                    if taker_result.walk.filled_shares > 0:
                        portfolio = apply_fill(portfolio, Fill(chosen.side, taker_result.walk.avg_price, taker_result.walk.filled_shares, LiquidityRole.TAKER, taker_result.walk.total_fee))
                        n_actions += 1
        elif chosen.mode is OrderMode.POST_ONLY and supervisor is not None:
            # Phase 12B Tranche 2A item 4, Tranche 2.1 item 2: check the
            # new maker candidate against the same shared RiskView
            # (confirmed + pending takers + every currently open maker
            # order) before admitting it - several individually-safe open
            # makers filling together can still jointly breach
            # g_min/spend_cap/position_limit (item 1's own counterexample),
            # which per-placement-time-only checks miss.
            exposure = candidate_exposure(chosen, fee_config)
            if exposure is not None and risk_view.admits(exposure, spec.one_step_cfg.g_min, spec.one_step_cfg.spend_cap, spec.one_step_cfg.position_limit, is_recovery_candidate=is_recovery_purpose(chosen.purpose)):
                order_seq += 1
                order_state = sim.submit_maker_order(f"{round_id}-o{order_seq}", chosen.side, chosen.qty, chosen.price, decision_ts)
                supervisor.register(TrackedOrder(order_state, snapshot.state, chosen.purpose, q, q if chosen.side is Side.UP else 1 - q, chosen.g_after, chosen.delta_ev, chosen.ttl_s or spec.one_step_cfg.maker_horizon_s, decision_ts, decision_ts))
        elif chosen.mode is OrderMode.POST_ONLY:
            # no supervisor tracking a maker order still resolves as an
            # immediate probability-weighted draw, for ablations that
            # allow maker execution without cancel/replace (none currently
            # do among the 8, but this keeps the dispatch total).
            order_seq += 1
            order_state = sim.submit_maker_order(f"{round_id}-o{order_seq}", chosen.side, chosen.qty, chosen.price, decision_ts)
            draw = sim.draw_maker_fill(order_state, 0.0, 0.0, chosen.ttl_s or spec.one_step_cfg.maker_horizon_s)
            n_attempts += 1
            if draw.filled:
                portfolio = apply_fill(portfolio, Fill(chosen.side, chosen.price, chosen.qty, LiquidityRole.MAKER, 0.0))
                n_actions += 1

    return portfolio, n_actions, n_attempts
