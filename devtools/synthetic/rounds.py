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

from xamarinbot.events.replay import seeded_random
from xamarinbot.events.store import EventStore
from xamarinbot.events.types import EventType
from xamarinbot.portfolio.state import Side
from xamarinbot.provenance import DataProvenance
from xamarinbot.rounds import RoundLabel

#: Everything this module produces carries this provenance. Phase 12C.1
#: item 2: an economic-evaluation path refuses it unless the caller passes
#: `allow_synthetic=True` explicitly.
PROVENANCE = DataProvenance.SYNTHETIC_TEST


def new_synthetic_store(db_path: str = ":memory:") -> EventStore:
    """An `EventStore` correctly labelled as holding fabricated data.

    Prefer this over `EventStore(...)` in demos and tests that then call
    `generate_synthetic_dataset`, so the label is attached at construction
    rather than depending on the default.
    """
    return EventStore(db_path, provenance=PROVENANCE)


def _depth_levels(touch_price: float, is_ask: bool, tick_size: float, book_liquidity: float,
                   offsets_ticks: tuple[int, ...], size_multipliers: tuple[float, ...]) -> list[tuple[float, float]]:
    """Levels beyond the touch move in lockstep with it (same tick_size
    offset every tick) with increasing size further from the touch - a
    plausible, simple depth shape for depth-walking demos/tests. The touch
    level (offset 0) is always `touch_price` unchanged, so best_bid/
    best_ask and everything computed from them (clob_mid, Phase 4-6
    features) are unaffected by adding these extra levels."""
    sign = 1 if is_ask else -1
    out = []
    for offset, mult in zip(offsets_ticks, size_multipliers):
        price = round(touch_price + sign * offset * tick_size, 2) if offset else touch_price
        out.append((min(0.99, max(0.01, price)), book_liquidity * mult))
    return out


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
    settlement_kind: str = "chainlink_twap",
    tick_size: float = 0.01,
    min_order_size: float = 1.0,
    fee_rate: float = 0.07,
    book_liquidity: float = 500.0,
    half_spread: float = 0.01,
    resnapshot_interval_s: float = 5.0,
    depth_offsets_ticks: tuple[int, ...] = (0, 2, 4),
    depth_size_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> RoundLabel:
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
            # Phase 12C.2 item 4: settlement_kind is a required market
            # parameter with no default. The generator models a
            # TWAP-settled market, so it says so explicitly.
            settlement_kind=settlement_kind,
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
        # baseline to ever trade) would be pure chance at every tick. Slope
        # is deliberately gentle (0.15 saturates mid to ~1.0 within the
        # first few ticks of any biased round, at which point clob_dir is
        # stuck at 0 for the rest of the round and stops carrying
        # information - 0.001 keeps mid graduated across the full observed
        # gap_bp range, roughly 0.50-0.88 from gap=0 to gap=2000bp).
        mid = 1.0 / (1.0 + math.exp(-0.001 * gap_bp)) + rng.gauss(0.0, 0.002)
        mid = min(0.97, max(0.03, mid))
        ask_up = min(0.99, max(0.01, round(mid + half_spread, 2)))
        bid_up = min(0.99, max(0.01, round(mid - half_spread, 2)))
        ask_down = min(0.99, max(0.01, round((1.0 - mid) + half_spread, 2)))
        bid_down = min(0.99, max(0.01, round((1.0 - mid) - half_spread, 2)))

        # Each (side, book) maps to a list of (price, size) levels: the
        # touch level (offset 0, unchanged from the original single-level
        # design) plus depth_offsets_ticks[1:] additional levels beyond it,
        # for depth-walking (Phase 7). Levels move in lockstep with the
        # touch, so best_bid/best_ask - and everything computed from them
        # in Phases 4-6 - are unaffected by their presence.
        current_levels = {
            (Side.UP, "asks"): _depth_levels(ask_up, True, tick_size, book_liquidity, depth_offsets_ticks, depth_size_multipliers),
            (Side.UP, "bids"): _depth_levels(bid_up, False, tick_size, book_liquidity, depth_offsets_ticks, depth_size_multipliers),
            (Side.DOWN, "asks"): _depth_levels(ask_down, True, tick_size, book_liquidity, depth_offsets_ticks, depth_size_multipliers),
            (Side.DOWN, "bids"): _depth_levels(bid_down, False, tick_size, book_liquidity, depth_offsets_ticks, depth_size_multipliers),
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
                    "bids": [[p, s] for p, s in current_levels[(Side.UP, "bids")]],
                    "asks": [[p, s] for p, s in current_levels[(Side.UP, "asks")]],
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
                    "bids": [[p, s] for p, s in current_levels[(Side.DOWN, "bids")]],
                    "asks": [[p, s] for p, s in current_levels[(Side.DOWN, "asks")]],
                    "book_hash": f"{round_id}-down-{i}",
                },
            )
            last_snapshot_t = t
        elif prev_levels is not None:
            # Remove/re-add against the *actual* previously-emitted prices
            # for each (side, book), carried over from last iteration - not
            # recomputed from the mid formula, which would drift out of
            # sync with what was really written and leave stale levels
            # behind forever (best_bid/best_ask would then reflect
            # leftover garbage instead of the current top of book).
            for key, new_level_list in current_levels.items():
                side, book = key
                prev_level_list = prev_levels[key]
                if new_level_list != prev_level_list:
                    for price, _ in prev_level_list:
                        store.append(
                            EventType.BOOK_DELTA, round_id, recv_ts=t + 0.02, source_ts=t,
                            payload={"side": side.value, "book": book, "price": price, "size": 0, "book_hash": f"{round_id}-{side.value}-{i}"},
                        )
                    for price, size in new_level_list:
                        store.append(
                            EventType.BOOK_DELTA, round_id, recv_ts=t + 0.02, source_ts=t,
                            payload={"side": side.value, "book": book, "price": price, "size": size, "book_hash": f"{round_id}-{side.value}-{i}"},
                        )

        prev_levels = current_levels

    final_twap = sum(spot_path[-window_ticks:]) / len(spot_path[-window_ticks:])
    outcome = Side.UP if final_twap > p0 else Side.DOWN

    store.append(
        EventType.SETTLEMENT, round_id, recv_ts=end_ts, source_ts=end_ts,
        payload={"outcome": outcome.value, "final_twap": final_twap, "p0": p0},
    )

    return RoundLabel(
        round_id=round_id, p0=p0, final_reference=final_twap, outcome=outcome,
        provenance=PROVENANCE,
    )


def generate_synthetic_dataset(
    store: EventStore, n_rounds: int, round_length_s: float = 300.0, id_offset: int = 0
) -> list[RoundLabel]:
    """`id_offset` shifts the absolute round index used for `round_id`,
    `start_ts`, and the bias pattern - every one of `round_id`,
    `bias_bp_per_tick`, and `start_ts` is a pure function of that index
    (via `seeded_random(round_id, ...)` for the RNG), so two calls with
    disjoint `[id_offset, id_offset+n_rounds)` ranges are guaranteed to
    generate disjoint, independent market paths, never the same round
    twice under different labels. The default `id_offset=0` reproduces
    every existing call site's behavior unchanged - callers that need a
    genuinely held-out set (train vs. validate vs. test) must pass
    non-overlapping ranges explicitly (see Phase 12B audit Addendum A:
    every prior call site restarted at index 0, so "held-out" data was
    frequently identical to training data)."""
    results = []
    for i in range(id_offset, id_offset + n_rounds):
        round_id = f"synthetic-round-{i:04d}"
        # alternate/vary bias so the dataset isn't all one direction -
        # important so every baseline skip reason has a chance to fire.
        bias = [0.0, 7.0, -7.0, 0.0][i % 4]
        result = populate_synthetic_round(
            store, round_id, start_ts=i * (round_length_s + 60.0), round_length_s=round_length_s, bias_bp_per_tick=bias
        )
        results.append(result)
    return results
