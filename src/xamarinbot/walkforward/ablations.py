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
from xamarinbot.baseline.strategy import BaselineInputs, decide as baseline_decide
from xamarinbot.events.replay import ReplayClock
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.execution.simulator import ExecutionSimulator
from xamarinbot.features.config import FeatureConfig
from xamarinbot.features.engine import compute
from xamarinbot.features.types import FeatureVector
from xamarinbot.feeds.mock import MockBookFeed, MockFeedCursor, MockSpotFeed, MockTWAPFeed
from xamarinbot.model.features import COMBINED_LEAD_LAG, LEAD_LAG_ONLY, SPOT_ONLY, TWAP_ONLY, FeatureSet, design_vector
from xamarinbot.model.logistic import LogisticModel
from xamarinbot.mpc.config import MPCConfig
from xamarinbot.mpc.controller import MPCController
from xamarinbot.mpc.scenario import TransitionModel
from xamarinbot.optimizer.config import OneStepConfig
from xamarinbot.optimizer.controller import OneStepController
from xamarinbot.optimizer.types import OrderMode
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


# lambda_g=0.01 (ablations 6-8) was tuned against this dataset's typical
# magnitudes, not guessed - two rounds of it: g_after for a meaningful taker
# position runs in the hundreds (a one-sided fill necessarily drags G down
# by roughly its own cost) while ev_after runs in the tens, so an initial
# lambda_g=0.5 made every directional candidate's selection score negative
# and suppressed all trading (the same class of scale mismatch as Phase
# 10's churn_penalty=5.0 finding). Backing off to 0.05 fixed taker but then
# suppressed *maker* candidates specifically: their ev_after is properly
# fill-probability-weighted but the g_after used in the same score is the
# pessimistic if-filled value, unweighted, so the same lambda_g penalizes
# maker's risk far more than its true expected contribution. 0.01 is small
# enough to keep both viable in this dataset. See docs/PHASE_STATUS.md.


MANDATORY_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("1_baseline_unanimous", "Baseline three-way unanimous strategy (Phase 0)", controller="baseline"),
    AblationSpec("2_twap_only", "V2 TWAP-only probability model", controller="one_step", feature_set=TWAP_ONLY, one_step_cfg=_default_one_step_cfg()),
    AblationSpec("3_spot_only", "V2 current-BTC-only model", controller="one_step", feature_set=SPOT_ONLY, one_step_cfg=_default_one_step_cfg()),
    AblationSpec("4_lead_lag", "V2 TWAP + current-BTC lead-lag model", controller="one_step", feature_set=LEAD_LAG_ONLY, one_step_cfg=_default_one_step_cfg()),
    AblationSpec("5_lead_lag_clob_no_repair", "Lead-lag + CLOB without portfolio repair", controller="one_step", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=False)),
    AblationSpec("6_portfolio_control_taker_only", "Full model with portfolio control but taker-only execution", controller="one_step", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=True, taker_only=True, lambda_g=0.01)),
    AblationSpec("7_maker_taker_cancel_replace", "Full model with maker/taker timing and cancel/replace", controller="one_step", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=True, lambda_g=0.01), use_supervisor=True),
    AblationSpec("8_full_mpc", "Full MPC versus one-step optimizer", controller="mpc", feature_set=COMBINED_LEAD_LAG, one_step_cfg=_default_one_step_cfg(enable_portfolio_repair=True, lambda_g=0.01), use_supervisor=True, mpc_cfg=MPCConfig(horizon_steps=2, time_budget_ms=50.0)),
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
    model: LogisticModel | None,
    transition_model: TransitionModel | None,
    baseline_cfg: BaselineConfig | None = None,
) -> RoundResult:
    if spec.controller == "baseline":
        portfolio = _run_baseline_round(store, round_id, p0, fee_config, baseline_cfg or BaselineConfig())
        pnl = portfolio.Pi_U if outcome is Side.UP else portfolio.Pi_D
        n_actions = 1 if (portfolio.U > 0 or portfolio.D > 0) else 0
        return RoundResult(round_id, pnl, n_actions, portfolio.G, fill_rate=1.0 if n_actions else 0.0)

    portfolio, n_actions, n_attempts = _run_controller_round(spec, store, round_id, p0, feature_cfg, fee_config, exec_cfg, model, transition_model)
    pnl = portfolio.Pi_U if outcome is Side.UP else portfolio.Pi_D
    fill_rate = (n_actions / n_attempts) if n_attempts else 0.0
    return RoundResult(round_id, pnl, n_actions, portfolio.G, fill_rate)


def _run_baseline_round(store: EventStore, round_id: str, p0: float, fee_config: FeeConfig, cfg: BaselineConfig) -> PortfolioState:
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)
    twap_feed, spot_feed, book_feed = MockTWAPFeed(cursor), MockSpotFeed(cursor), MockBookFeed(cursor)
    prev_cursor = MockFeedCursor(store, round_id, preloaded=events)
    prev_book_feed = MockBookFeed(prev_cursor)
    portfolio = PortfolioState()

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)
        twap_obs, spot_obs = twap_feed.get_latest(round_id), spot_feed.get_latest(round_id)
        if twap_obs is None or spot_obs is None:
            continue
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        book_down = book_feed.get_snapshot(round_id, Side.DOWN)
        if book_up is None or not book_up.best_bid or not book_up.best_ask:
            continue
        prev_cursor.advance_to(decision_ts - cfg.clob_lookback_s)
        prev_book_up = prev_book_feed.get_snapshot(round_id, Side.UP)
        mid = (book_up.best_bid.price + book_up.best_ask.price) / 2.0
        mid_prev = ((prev_book_up.best_bid.price + prev_book_up.best_ask.price) / 2.0) if prev_book_up and prev_book_up.best_bid and prev_book_up.best_ask else mid

        inputs = BaselineInputs(
            t=decision_ts, p0=p0, twap=twap_obs.value, clob_mid=mid, clob_mid_prev=mid_prev,
            spot=spot_obs.value, spot_prev=spot_obs.value,
            best_ask_up=book_up.best_ask.price, best_ask_down=book_down.best_ask.price if book_down and book_down.best_ask else None,
            is_fresh=True,
        )
        result = baseline_decide(inputs, portfolio.U, portfolio.D, portfolio.C, cfg)
        if result.order is not None:
            fee = fee_config.fee_for(result.order.role, result.order.quantity, result.order.price)
            portfolio = apply_fill(portfolio, Fill(result.order.side, result.order.price, result.order.quantity, result.order.role, fee))
    return portfolio


def _run_controller_round(
    spec: AblationSpec,
    store: EventStore,
    round_id: str,
    p0: float,
    feature_cfg: FeatureConfig,
    fee_config: FeeConfig,
    exec_cfg: ExecutionConfig,
    model: LogisticModel | None,
    transition_model: TransitionModel | None,
) -> tuple[PortfolioState, int, int]:
    events = store.all_events(round_id)
    clock = ReplayClock(store, round_id)
    cursor = MockFeedCursor(store, round_id, preloaded=events)
    book_feed = MockBookFeed(cursor)
    regime_clf = RegimeClassifier(round_id=round_id)
    one_step = OneStepController(spec.one_step_cfg, exec_cfg, fee_config)
    mpc = MPCController(spec.mpc_cfg, transition_model or TransitionModel(probabilities={}), one_step) if spec.controller == "mpc" else None
    supervisor = OrderSupervisor(SupervisorConfig(g_min=spec.one_step_cfg.g_min, edge_min=spec.one_step_cfg.edge_min)) if spec.use_supervisor else None
    sim = ExecutionSimulator(round_id, fee_config, exec_cfg)
    portfolio = PortfolioState()
    n_actions = 0
    n_attempts = 0
    order_seq = 0

    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    tick_size = market_config["tick_size"]

    for decision_ts in clock.decision_points(heartbeat=HEARTBEAT_S):
        cursor.advance_to(decision_ts)
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
                decision = supervisor.review_order(tracked, decision_ts, snapshot.state, 0.0, portfolio.G, fv.tau, True)
                if decision.action is SupervisorActionType.CANCEL:
                    supervisor.apply_cancel(decision, decision_ts)

        permitted = ActionPermissionMatrix.permitted_actions(classify_seed_action(snapshot.state))
        if mpc is not None:
            mpc_decision = mpc.decide(round_id, decision_ts, portfolio, q, snapshot.state, book_up, book_down, tick_size, True)
            chosen = mpc_decision.chosen
        else:
            one_step_decision = one_step.decide(round_id, decision_ts, portfolio, q, permitted, book_up, book_down, tick_size, True)
            chosen = one_step_decision.chosen

        if chosen.mode is OrderMode.FAK and chosen.expected_fill > 0:
            n_attempts += 1
            fee = fee_config.taker_fee(chosen.expected_fill, chosen.price)
            portfolio = apply_fill(portfolio, Fill(chosen.side, chosen.price, chosen.expected_fill, LiquidityRole.TAKER, fee))
            n_actions += 1
        elif chosen.mode is OrderMode.POST_ONLY and supervisor is not None:
            order_seq += 1
            order_state = sim.submit_maker_order(f"{round_id}-o{order_seq}", chosen.side, chosen.qty, chosen.price, decision_ts)
            supervisor.register(TrackedOrder(order_state, snapshot.state, chosen.purpose, q, q if chosen.side is Side.UP else 1 - q, chosen.g_after, chosen.ev_after, chosen.ttl_s or spec.one_step_cfg.maker_horizon_s, decision_ts, decision_ts))
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
