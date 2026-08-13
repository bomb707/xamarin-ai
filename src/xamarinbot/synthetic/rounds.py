"""SYNTHETIC regression dataset generator.

This is NOT real market data. No historical Polymarket/Chainlink/BTC data
was supplied with the source spec, so Roadmap Phase 0's "regression dataset
of representative rounds" is stood in with a deterministically-seeded random
walk that has the right *shape* (a spot path, a lagging TWAP, a noisy CLOB
midpoint, a settlement outcome) so the replay/journal/report pipeline can be
exercised end-to-end and tested for determinism. Any win-rate/PnL numbers
this produces are meaningless as trading-performance evidence - they only
demonstrate that the pipeline runs and reconciles. Replace with real
recorded rounds before trusting Phase 0's baseline performance report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from xamarinbot.events.replay import seeded_random
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.portfolio.state import Side


@dataclass(frozen=True)
class SyntheticRoundResult:
    round_id: str
    p0: float
    final_twap: float
    outcome: Side


def populate_synthetic_round(
    store: EventStore,
    round_id: str,
    start_ts: float = 0.0,
    round_length_s: float = 300.0,
    tick_interval_s: float = 1.0,
    p0: float = 100_000.0,
    vol_bp_per_tick: float = 5.0,
    bias_bp_per_tick: float = 0.0,
    twap_window_seconds: int = 30,
    tick_size: float = 0.01,
    min_order_size: float = 1.0,
    fee_rate: float = 0.07,
    book_liquidity: float = 500.0,
    half_spread: float = 0.01,
    resnapshot_interval_s: float = 5.0,
) -> SyntheticRoundResult:
    """Appends one synthetic round's worth of events to `store` and returns
    its ground-truth settlement outcome. Deterministic for a given
    `round_id` (uses seeded_random, not the process RNG)."""

    rng = seeded_random(round_id, "synthetic-round")
    end_ts = start_ts + round_length_s
    up_token_id = f"{round_id}-UP"
    down_token_id = f"{round_id}-DOWN"

    store.append(
        EventType.MARKET_CONFIG,
        round_id,
        recv_ts=start_ts,
        source_ts=start_ts,
        payload=dict(
            market_id=round_id,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            start_ts=start_ts,
            end_ts=end_ts,
            tick_size=tick_size,
            min_order_size=min_order_size,
            fee_rate=fee_rate,
            taker_delay_ms=0.0,
            twap_window_seconds=twap_window_seconds,
        ),
    )

    n_ticks = int(round_length_s / tick_interval_s) + 1
    spot_path: list[float] = []
    spot = p0
    for _ in range(n_ticks):
        drift = p0 * (bias_bp_per_tick / 10_000.0)
        shock = p0 * (rng.gauss(0.0, vol_bp_per_tick) / 10_000.0)
        spot = spot + drift + shock
        spot_path.append(spot)

    window_ticks = max(1, int(twap_window_seconds / tick_interval_s))
    prev_levels: dict[tuple[Side, str], float] | None = None
    last_snapshot_t: float = start_ts

    for i in range(n_ticks):
        t = start_ts + i * tick_interval_s
        spot_val = spot_path[i]
        window = spot_path[max(0, i - window_ticks + 1) : i + 1]
        twap_val = sum(window) / len(window)

        store.append(EventType.SPOT, round_id, recv_ts=t + 0.01, source_ts=t, payload={"value": spot_val, "provider": "synthetic"})
        store.append(EventType.TWAP, round_id, recv_ts=t + 0.05, source_ts=t, payload={"value": twap_val, "window_seconds": twap_window_seconds})

        gap_bp = 10_000.0 * (spot_val - twap_val) / twap_val
        # Small noise relative to the gap-driven signal: mid should mostly
        # track the same spot-vs-TWAP pressure that drives spot/TWAP
        # direction, or unanimous-direction agreement (required for the
        # baseline to ever trade) would be pure chance at every tick.
        mid = 1.0 / (1.0 + math.exp(-0.15 * gap_bp)) + rng.gauss(0.0, 0.002)
        mid = min(0.97, max(0.03, mid))
        ask_up = min(0.99, max(0.01, round(mid + half_spread, 2)))
        bid_up = min(0.99, max(0.01, round(mid - half_spread, 2)))
        ask_down = min(0.99, max(0.01, round((1.0 - mid) + half_spread, 2)))
        bid_down = min(0.99, max(0.01, round((1.0 - mid) - half_spread, 2)))

        current_levels = {
            (Side.UP, "asks"): ask_up,
            (Side.UP, "bids"): bid_up,
            (Side.DOWN, "asks"): ask_down,
            (Side.DOWN, "bids"): bid_down,
        }

        due_for_resnapshot = prev_levels is None or (t - last_snapshot_t) >= resnapshot_interval_s
        if due_for_resnapshot:
            # "periodic/safety resnapshot" (Roadmap Phase 1): also keeps the
            # book feed fresh even through ticks where no level actually
            # changed, matching a real feed's heartbeat/snapshot behavior.
            store.append(
                EventType.BOOK_SNAPSHOT,
                round_id,
                recv_ts=t + 0.02,
                source_ts=t,
                payload={
                    "side": Side.UP.value,
                    "bids": [[bid_up, book_liquidity]],
                    "asks": [[ask_up, book_liquidity]],
                    "book_hash": f"{round_id}-up-{i}",
                },
            )
            store.append(
                EventType.BOOK_SNAPSHOT,
                round_id,
                recv_ts=t + 0.02,
                source_ts=t,
                payload={
                    "side": Side.DOWN.value,
                    "bids": [[bid_down, book_liquidity]],
                    "asks": [[ask_down, book_liquidity]],
                    "book_hash": f"{round_id}-down-{i}",
                },
            )
            last_snapshot_t = t
        elif prev_levels is not None:
            # Remove/re-add against the *actual* previously-emitted price
            # for each (side, book) level, carried over from last
            # iteration - not recomputed from the mid formula, which would
            # drift out of sync with what was really written and leave
            # stale levels behind forever (best_bid/best_ask would then
            # reflect leftover garbage instead of the current top of book).
            for key, new_price in current_levels.items():
                side, book = key
                prev_price = prev_levels[key]
                if new_price != prev_price:
                    store.append(
                        EventType.BOOK_DELTA, round_id, recv_ts=t + 0.02, source_ts=t,
                        payload={"side": side.value, "book": book, "price": prev_price, "size": 0, "book_hash": f"{round_id}-{side.value}-{i}"},
                    )
                    store.append(
                        EventType.BOOK_DELTA, round_id, recv_ts=t + 0.02, source_ts=t,
                        payload={"side": side.value, "book": book, "price": new_price, "size": book_liquidity, "book_hash": f"{round_id}-{side.value}-{i}"},
                    )

        prev_levels = current_levels

    final_twap = sum(spot_path[-window_ticks:]) / len(spot_path[-window_ticks:])
    outcome = Side.UP if final_twap > p0 else Side.DOWN

    store.append(
        EventType.SETTLEMENT, round_id, recv_ts=end_ts, source_ts=end_ts,
        payload={"outcome": outcome.value, "final_twap": final_twap, "p0": p0},
    )

    return SyntheticRoundResult(round_id=round_id, p0=p0, final_twap=final_twap, outcome=outcome)


def generate_synthetic_dataset(store: EventStore, n_rounds: int, round_length_s: float = 300.0) -> list[SyntheticRoundResult]:
    results = []
    for i in range(n_rounds):
        round_id = f"synthetic-round-{i:04d}"
        # alternate/vary bias so the dataset isn't all one direction -
        # important so every baseline skip reason has a chance to fire.
        bias = [0.0, 7.0, -7.0, 0.0][i % 4]
        result = populate_synthetic_round(
            store, round_id, start_ts=i * (round_length_s + 60.0), round_length_s=round_length_s, bias_bp_per_tick=bias
        )
        results.append(result)
    return results
