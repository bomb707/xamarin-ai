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
        settlement_kind: str = "chainlink_twap",
        provenance: DataProvenance = DataProvenance.SYNTHETIC_TEST,
        source: str = "market_config",
        captured_at: float | None = None,
    ) -> "MarketConstraints":
        """Build from a Phase-1 `MarketConfig`, which is what both the real
        adapter and a replayed MARKET_CONFIG event produce."""
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
