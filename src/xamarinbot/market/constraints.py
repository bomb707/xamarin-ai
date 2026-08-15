"""Runtime `MarketConstraints` (Phase 12C.1 items 11-13).

The problem this replaces
-------------------------
`OneStepConfig.taker_min_size = 1.0` was the production minimum-order
fallback, carrying a comment that admitted it stood in for
`MarketConfig.min_order_size` "once wired in ... not yet threaded through
here". Meanwhile `MarketConfig.min_order_size` already carried the REAL value
all the way from `realtime/discovery.py` and then dead-ended - nothing in the
optimizer or execution layer ever read it.

All three BTC five-minute markets sampled in the Phase 12C captures report:

    min_order_size = 5.0 shares
    tick_size      = 0.01

so the static 1.0 was not merely un-wired, it was wrong by 5x, in the
direction that generates orders the venue would reject.

Units
-----
`min_order_shares` is a SHARE quantity, exactly as Polymarket reports it
(CLOB market-info `mos` / orderbook `min_order_size`). It is **not** a USDC
notional and must never be conflated with one (item 13): 5 shares at $0.10 is
$0.50 of notional and is a legal order; $1.00 of notional at $0.10 is 10
shares and is a different constraint entirely.

Why a runtime object rather than more config fields
---------------------------------------------------
Item 12: "Do not copy these values into unrelated static configs." A market
parameter copied into a frozen strategy config is a snapshot that silently
goes stale and becomes a second apparent source of truth. `MarketConstraints`
is constructed per round from that round's own metadata and passed down the
call chain, so there is exactly one place any executable parameter comes from.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from xamarinbot.feeds.base import MarketConfig
from xamarinbot.portfolio.state import FeeConfig
from xamarinbot.provenance import DataProvenance


class MarketConstraintError(ValueError):
    """Raised instead of defaulting an executable market parameter."""


@dataclass(frozen=True)
class MarketConstraints:
    """Immutable, per-round view of what the venue will actually accept."""

    condition_id: str
    up_token_id: str
    down_token_id: str

    #: Minimum order size in SHARES (never USDC notional).
    min_order_shares: float
    #: Minimum price increment.
    tick_size: float
    #: Fee parameters in the form the portfolio math consumes.
    fee_configuration: FeeConfig
    #: Venue-imposed taker delay, milliseconds. 0 when the market has none.
    taker_delay_ms: float

    #: How this market says it settles ("chainlink_twap" /
    #: "chainlink_reference"), and the declared TWAP lookback when it has
    #: one. Item 15: never globally hardcoded - each market declares its own.
    settlement_kind: str
    twap_window_s: int | None

    #: When these values were read, and from where. A constraint set is a
    #: measurement with a timestamp, not a constant.
    captured_at: float
    source: str
    provenance: DataProvenance = DataProvenance.SYNTHETIC_TEST

    def __post_init__(self) -> None:
        if self.min_order_shares <= 0:
            raise MarketConstraintError(
                f"min_order_shares must be positive, got {self.min_order_shares}"
            )
        if self.tick_size <= 0:
            raise MarketConstraintError(f"tick_size must be positive, got {self.tick_size}")

    # ----------------------------------------------------------- helpers

    def admits_size(self, shares: float, *, tolerance: float = 1e-9) -> bool:
        """Whether `shares` is an executable quantity at this venue.

        The tolerance exists for one specific reason, documented by an
        existing regression test: a boundary quantity computed as exactly the
        minimum can come back as `4.999999999` through floating-point
        arithmetic, and rejecting it would silently discard a genuinely legal
        order. It is NOT licence to round a too-small order upward.
        """
        return shares >= self.min_order_shares - tolerance

    def token_for(self, side) -> str:
        from xamarinbot.portfolio.state import Side

        return self.up_token_id if side is Side.UP else self.down_token_id

    def describe(self) -> str:
        return (
            f"{self.condition_id[:18]}… MOS={self.min_order_shares} shares "
            f"tick={self.tick_size} fee={self.fee_configuration.crypto_fee_rate} "
            f"delay={self.taker_delay_ms}ms settle={self.settlement_kind}"
            f"({self.twap_window_s}s) src={self.source} [{self.provenance.value}]"
        )

    # ------------------------------------------------------ constructors

    @classmethod
    def from_market_config(
        cls,
        config: MarketConfig,
        *,
        condition_id: str | None = None,
        provenance: DataProvenance = DataProvenance.SYNTHETIC_TEST,
        source: str = "market_config",
        captured_at: float | None = None,
    ) -> "MarketConstraints":
        """Build from a Phase-1 `MarketConfig`, which is what both the real
        adapter and a replayed MARKET_CONFIG event produce.

        Phase 12C.2 item 4 removed the `settlement_kind="chainlink_twap"`
        default this used to carry. A default is a guess, and a guessed
        settlement rule silently decides which reference series counts as
        truth for the label - the single most consequential parameter in the
        round. It now comes from `MarketConfig.settlement_kind`, which the
        projection populates from the recorder's persisted metadata or fails
        closed.
        """
        settlement_kind = config.settlement_kind
        if not settlement_kind:
            raise MarketConstraintError(
                f"{config.market_id}: no settlement rule recorded; refusing to guess "
                "one. A market whose settlement basis was never captured cannot be "
                "labelled or replayed."
            )
        return cls(
            condition_id=condition_id or config.market_id,
            up_token_id=config.up_token_id,
            down_token_id=config.down_token_id,
            min_order_shares=float(config.min_order_size),
            tick_size=float(config.tick_size),
            fee_configuration=FeeConfig(crypto_fee_rate=float(config.fee_rate)),
            taker_delay_ms=float(config.taker_delay_ms),
            settlement_kind=settlement_kind,
            twap_window_s=int(config.twap_window_seconds),
            captured_at=captured_at if captured_at is not None else config.start_ts,
            source=source,
            provenance=provenance,
        )

    @classmethod
    def for_testing(
        cls,
        *,
        min_order_shares: float = 1.0,
        tick_size: float = 0.01,
        fee_rate: float = 0.07,
        taker_delay_ms: float = 0.0,
        condition_id: str = "test-condition",
        up_token_id: str = "test-up",
        down_token_id: str = "test-down",
        settlement_kind: str = "chainlink_twap",
        twap_window_s: int | None = 60,
    ) -> "MarketConstraints":
        """Explicitly-synthetic constraints for unit tests.

        The default `min_order_shares=1.0` reproduces the old
        `OneStepConfig.taker_min_size` default so existing candidate-sizing
        tests keep asserting the same arithmetic - but it is now unmistakably
        a TEST fixture stamped `SYNTHETIC_TEST`, not a production fallback
        that a live path could silently inherit.
        """
        return cls(
            condition_id=condition_id,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            min_order_shares=min_order_shares,
            tick_size=tick_size,
            fee_configuration=FeeConfig(crypto_fee_rate=fee_rate),
            taker_delay_ms=taker_delay_ms,
            settlement_kind=settlement_kind,
            twap_window_s=twap_window_s,
            captured_at=0.0,
            source="for_testing",
            provenance=DataProvenance.SYNTHETIC_TEST,
        )


class ExecutionStateConflict(ValueError):
    """Raised when a caller supplies executable financial state that
    contradicts what the market reported (Phase 12C.2 item 1)."""


def reconcile_execution_state(
    constraints: MarketConstraints,
    fee_config: FeeConfig | None = None,
    exec_cfg: "ExecutionConfig | None" = None,
) -> "tuple[FeeConfig, ExecutionConfig]":
    """Return the authoritative `(FeeConfig, ExecutionConfig)` for a round.

    Phase 12C.2 item 1. `ShadowRunner`, `OneStepController` and
    `TradingSession` each accepted a `FeeConfig` and an `ExecutionConfig`
    independently of the round's `MarketConstraints`, which meant the system
    could run with

        Fee_simulation   != Fee_market
        Delay_simulation != Delay_market

    and nothing would notice. Both are financial state: the fee rate enters
    every EV, every all-in cost cap and every fill; the taker delay decides
    which book a delayed order is matched against. Two copies of either is
    one copy too many.

    The fix ELIMINATES the duplicate rather than asserting the copies agree:
    on REAL_LIVE / REAL_REPLAY the market's own values are returned, so
    `FeeUsed = FeeReportedByMarket` and `TakerDelayUsed = DelayReportedByMarket`
    hold by construction. A caller that supplied a *different* value is not
    silently overridden - that would hide a real disagreement - it raises.

    On SYNTHETIC_TEST the caller's values win, because the whole point of a
    generated round is to be able to vary them.
    """
    from xamarinbot.execution.config import ExecutionConfig

    market_fee = constraints.fee_configuration
    market_delay = constraints.taker_delay_ms

    if not constraints.provenance.is_real:
        return (
            fee_config if fee_config is not None else market_fee,
            exec_cfg if exec_cfg is not None else ExecutionConfig(taker_delay_ms=market_delay),
        )

    if fee_config is not None and fee_config.crypto_fee_rate != market_fee.crypto_fee_rate:
        raise ExecutionStateConflict(
            f"supplied FeeConfig(crypto_fee_rate={fee_config.crypto_fee_rate}) contradicts "
            f"the market's reported fee rate {market_fee.crypto_fee_rate} for "
            f"{constraints.condition_id}. On {constraints.provenance.value} data the "
            "market is the single source of truth for executable fees - pass None, or "
            "pass the market's own value."
        )
    if exec_cfg is not None and exec_cfg.taker_delay_ms != market_delay:
        raise ExecutionStateConflict(
            f"supplied ExecutionConfig(taker_delay_ms={exec_cfg.taker_delay_ms}) "
            f"contradicts the market's reported taker delay {market_delay}ms for "
            f"{constraints.condition_id}. On {constraints.provenance.value} data the "
            "market is the single source of truth for the taker delay."
        )

    # Derived, not merely checked: the returned objects ARE the market's.
    # The maker fill-model parameters are simulation knobs rather than market
    # facts, so a caller's ExecutionConfig keeps those.
    resolved_exec = (
        ExecutionConfig(taker_delay_ms=market_delay, maker=exec_cfg.maker)
        if exec_cfg is not None
        else ExecutionConfig(taker_delay_ms=market_delay)
    )
    return market_fee, resolved_exec
