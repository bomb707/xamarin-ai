"""Phase 12B Tranche 1 regression tests, items 2/D: `walkforward/ablations.py`'s
`_run_baseline_round` previously (a) passed `spot == spot_prev`, making
`spot_direction` always 0 and unanimity impossible, and (b) passed
absolute `decision_ts` as elapsed round time, making every round whose
`start_ts != 0` fail the decision-window gate for its entire duration.
`tests/test_baseline_strategy.py` already covers `baseline/strategy.py::decide()`
directly with correct inputs; these tests cover the *harness* that builds
those inputs from a causal replay, which is where both bugs actually were."""
from __future__ import annotations

import math

from xamarinbot.baseline.config import BaselineConfig
from xamarinbot.baseline.inputs import elapsed_t
from xamarinbot.events.store import EventStore
from xamarinbot.execution.config import ExecutionConfig
from xamarinbot.portfolio.state import FeeConfig, Side
from xamarinbot.synthetic.rounds import populate_synthetic_round
from xamarinbot.walkforward.ablations import _run_baseline_round


def test_elapsed_t_is_relative_not_absolute():
    assert elapsed_t(decision_ts=370.0, round_start_ts=360.0) == 10.0
    assert elapsed_t(decision_ts=10.0, round_start_ts=0.0) == 10.0
    assert elapsed_t(decision_ts=1_700_000_010.0, round_start_ts=1_700_000_000.0) == 10.0


def _run_at_start_ts(round_id: str, start_ts: float, bias_bp_per_tick: float, fee_config: FeeConfig, cfg: BaselineConfig):
    store = EventStore(":memory:")
    populate_synthetic_round(store, round_id, start_ts=start_ts, bias_bp_per_tick=bias_bp_per_tick)
    portfolio, _n_actions, _n_attempts = _run_baseline_round(store, round_id, p0=100_000.0, fee_config=fee_config, exec_cfg=ExecutionConfig(), cfg=cfg)
    return portfolio


def test_baseline_behavior_is_invariant_to_absolute_round_start_timestamp():
    """Three rounds with an identical *relative* 0-300s market path
    (populate_synthetic_round's RNG is seeded on round_id alone, not on
    start_ts, so using the same round_id at different start_ts values
    reproduces the identical relative path) but different absolute
    start_ts must produce identical baseline decisions - before the fix,
    only the start_ts=0 round could ever trade at all."""
    fee_config = FeeConfig()
    cfg = BaselineConfig()
    bias = 7.0  # a clear directional bias so the baseline has a real chance to trade

    p_zero = _run_at_start_ts("baseline-time-test", 0.0, bias, fee_config, cfg)
    p_mid = _run_at_start_ts("baseline-time-test", 360.0, bias, fee_config, cfg)
    p_large = _run_at_start_ts("baseline-time-test", 1_700_000_000.0, bias, fee_config, cfg)

    assert math.isclose(p_zero.U, p_mid.U) and math.isclose(p_zero.U, p_large.U)
    assert math.isclose(p_zero.D, p_mid.D) and math.isclose(p_zero.D, p_large.D)
    assert math.isclose(p_zero.C, p_mid.C) and math.isclose(p_zero.C, p_large.C)


def test_baseline_can_actually_trade_on_a_round_not_starting_at_zero():
    """Direct regression for the compounded bug: before the fix, every
    round with start_ts > 0 (i.e. every round except the very first in
    any multi-round dataset) could never place a single trade, regardless
    of how favorable the market path was."""
    fee_config = FeeConfig()
    cfg = BaselineConfig()
    portfolio = _run_at_start_ts("baseline-nonzero-start", 360.0, 7.0, fee_config, cfg)
    assert portfolio.U > 0 or portfolio.D > 0


def test_baseline_harness_spot_direction_is_not_always_flat():
    """A strongly-biased round must produce at least one decision where
    the harness's own spot lookback disagrees with the current spot value
    (i.e. spot_direction != 0 at least once) - before the fix this was
    impossible by construction (spot_prev was always exactly spot)."""
    from xamarinbot.events.replay import ReplayClock
    from xamarinbot.events.types import EventType
    from xamarinbot.baseline.strategy import BaselineInputs, decide
    from xamarinbot.feeds.mock import MockFeedCursor, MockSpotFeed, MockTWAPFeed, MockBookFeed
    from xamarinbot.baseline.inputs import elapsed_t

    store = EventStore(":memory:")
    round_id = "baseline-spot-direction-test"
    populate_synthetic_round(store, round_id, start_ts=0.0, bias_bp_per_tick=7.0)
    events = store.all_events(round_id)
    market_config = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    cfg = BaselineConfig()

    cursor = MockFeedCursor(store, round_id, preloaded=events)
    twap_feed, spot_feed, book_feed = MockTWAPFeed(cursor), MockSpotFeed(cursor), MockBookFeed(cursor)
    prev_spot_cursor = MockFeedCursor(store, round_id, preloaded=events)
    prev_spot_feed = MockSpotFeed(prev_spot_cursor)
    clock = ReplayClock(store, round_id)

    seen_nonzero_spot_direction = False
    for decision_ts in clock.decision_points(heartbeat=10.0):
        cursor.advance_to(decision_ts)
        twap_obs, spot_obs = twap_feed.get_latest(round_id), spot_feed.get_latest(round_id)
        if twap_obs is None or spot_obs is None:
            continue
        book_up = book_feed.get_snapshot(round_id, Side.UP)
        if book_up is None or not book_up.best_bid or not book_up.best_ask:
            continue
        prev_spot_cursor.advance_to(decision_ts - cfg.spot_lookback_s)
        prev_spot_obs = prev_spot_feed.get_latest(round_id)
        spot_prev = prev_spot_obs.value if prev_spot_obs is not None else spot_obs.value
        mid = (book_up.best_bid.price + book_up.best_ask.price) / 2.0

        inputs = BaselineInputs(
            t=elapsed_t(decision_ts, market_config["start_ts"]), p0=100_000.0, twap=twap_obs.value,
            clob_mid=mid, clob_mid_prev=mid, spot=spot_obs.value, spot_prev=spot_prev,
            best_ask_up=book_up.best_ask.price, best_ask_down=None, is_fresh=True,
        )
        decision = decide(inputs, 0.0, 0.0, 0.0, cfg)
        if decision.spot_direction != 0:
            seen_nonzero_spot_direction = True
            break

    assert seen_nonzero_spot_direction
