"""Counterfactual capture for hypothetical maker quotes (Phase 12C item 11).

Why this module exists instead of reusing the maker simulator
-------------------------------------------------------------
`ExecutionSimulator.draw_maker_fill` answers "did this order fill sometime
during its TTL?" with ONE Bernoulli draw against an uncalibrated `rho`, and
the session then applies the whole remaining quantity at expiry. Nothing
about that is fill *truth*: not the probability, not the timing, not the
size. It was always labeled a placeholder, and it is fine as a synthetic
stand-in - but carrying it into real-data evaluation as ground truth would
silently convert an assumption into a measurement.

So this module records no fills at all. It records, for every hypothetical
quote, the raw market evidence from which `rho = P(fill | state, price,
queue, horizon)` and `q_fill` can LATER be estimated from real data. Fitting
those models is explicitly out of scope for Phase 12C.

The quantities captured are exactly item 11's list. Definitions, stated
because they are easy to get subtly wrong:

  A quote is a resting BUY of `qty` at `price` on `side`'s token (the only
  maker shape this strategy places - both UP and DOWN exposure is acquired
  by buying the corresponding token, never by selling).

  first-touch    The first moment the best bid on that token equals the
                 quote price: the quote is at the touch, so it is now at the
                 front of the visible ladder rather than behind other levels.
  first-cross    The first moment the best ASK on that token is at or below
                 the quote price. At that instant a real resting bid at this
                 price is marketable against visible liquidity - the
                 strongest available evidence that a fill was achievable.
  traded at/through
                 Cumulative size of trades printed at a price <= the quote
                 price. These are the trades that would have interacted with
                 the level.
  queue ahead    Size resting at the quote's own price level. At submit this
                 is the queue the order would join behind; sampled over the
                 lifetime it shows whether that queue drained (a fill
                 becomes likely) or grew.

Nothing here places, cancels, or replaces anything at the venue (item 14).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Side = Literal["UP", "DOWN"]


@dataclass(frozen=True)
class QueueSample:
    """One observation of the state around a resting quote."""

    ts: float
    best_bid: float | None
    best_ask: float | None
    size_at_quote_price: float
    size_ahead_better_prices: float
    q: float | None


@dataclass(frozen=True)
class TradePrint:
    ts: float
    price: float
    size: float
    side: str | None


@dataclass
class HypotheticalQuote:
    """A maker quote the strategy WOULD have submitted, and everything
    observed about the market around it. Never sent to the venue."""

    quote_id: str
    round_id: str
    token_id: str
    side: Side
    price: float
    qty: float
    ttl_s: float
    submit_ts: float

    # --- state at submit ---
    best_bid_at_submit: float | None = None
    best_ask_at_submit: float | None = None
    #: Size already resting at our price when we would have joined. This is
    #: the real `queue_ahead_shares` that item 11 says must stop being
    #: hardcoded to 0.
    queue_ahead_at_submit: float = 0.0
    #: Ticks between our price and the touch on our side. The other value
    #: item 11 says must stop being hardcoded to 0.
    distance_to_touch_ticks_at_submit: float = 0.0
    tick_size: float = 0.01
    q_at_submit: float | None = None
    fair_value_at_submit: float | None = None

    # --- observed over the quote's lifetime ---
    first_touch_ts: float | None = None
    first_cross_ts: float | None = None
    cumulative_traded_at_or_through: float = 0.0
    trades: list[TradePrint] = field(default_factory=list)
    queue_samples: list[QueueSample] = field(default_factory=list)
    book_delta_count: int = 0
    q_samples: list[tuple[float, float]] = field(default_factory=list)

    # --- lifecycle (paper only) ---
    cancel_ts: float | None = None
    cancel_reason: str | None = None
    replaced_by: str | None = None
    expired_ts: float | None = None

    @property
    def end_ts(self) -> float:
        for ts in (self.cancel_ts, self.expired_ts):
            if ts is not None:
                return ts
        return self.submit_ts + self.ttl_s

    @property
    def time_to_first_touch_s(self) -> float | None:
        return None if self.first_touch_ts is None else self.first_touch_ts - self.submit_ts

    @property
    def time_to_first_cross_s(self) -> float | None:
        return None if self.first_cross_ts is None else self.first_cross_ts - self.submit_ts

    @property
    def q_at_end(self) -> float | None:
        return self.q_samples[-1][1] if self.q_samples else self.q_at_submit

    def as_dict(self) -> dict:
        return {
            "quote_id": self.quote_id,
            "round_id": self.round_id,
            "token_id": self.token_id,
            "side": self.side,
            "price": self.price,
            "qty": self.qty,
            "ttl_s": self.ttl_s,
            "submit_ts": self.submit_ts,
            "best_bid_at_submit": self.best_bid_at_submit,
            "best_ask_at_submit": self.best_ask_at_submit,
            "queue_ahead_at_submit": self.queue_ahead_at_submit,
            "distance_to_touch_ticks_at_submit": self.distance_to_touch_ticks_at_submit,
            "tick_size": self.tick_size,
            "q_at_submit": self.q_at_submit,
            "fair_value_at_submit": self.fair_value_at_submit,
            "first_touch_ts": self.first_touch_ts,
            "time_to_first_touch_s": self.time_to_first_touch_s,
            "first_cross_ts": self.first_cross_ts,
            "time_to_first_cross_s": self.time_to_first_cross_s,
            "cumulative_traded_at_or_through": self.cumulative_traded_at_or_through,
            "n_trades_observed": len(self.trades),
            "book_delta_count": self.book_delta_count,
            "n_queue_samples": len(self.queue_samples),
            "queue_ahead_first": self.queue_samples[0].size_at_quote_price if self.queue_samples else None,
            "queue_ahead_last": self.queue_samples[-1].size_at_quote_price if self.queue_samples else None,
            "q_at_submit_recorded": self.q_at_submit,
            "q_at_end": self.q_at_end,
            "cancel_ts": self.cancel_ts,
            "cancel_reason": self.cancel_reason,
            "replaced_by": self.replaced_by,
            "expired_ts": self.expired_ts,
            "end_ts": self.end_ts,
            # Deliberately NOT a fill flag. `first_cross_ts is not None` is
            # evidence a fill was achievable, not an assertion that one
            # happened - the difference is the whole point of item 11.
            "cross_observed": self.first_cross_ts is not None,
            "touch_observed": self.first_touch_ts is not None,
        }


class MakerCounterfactualTracker:
    """Tracks every open hypothetical quote against the live book/trades.

    Fed by the recorder from the same stream that maintains the order book,
    so the observations are on exactly the data a real resting order would
    have experienced.
    """

    def __init__(self, sample_interval_s: float = 1.0):
        self._open: dict[str, HypotheticalQuote] = {}
        self._closed: list[HypotheticalQuote] = []
        self._last_sample_ts: dict[str, float] = {}
        self.sample_interval_s = sample_interval_s
        self._seq = 0

    # ---------------------------------------------------------- lifecycle

    def next_quote_id(self, round_id: str) -> str:
        self._seq += 1
        return f"{round_id}-hq{self._seq}"

    def register(
        self,
        *,
        round_id: str,
        token_id: str,
        side: Side,
        price: float,
        qty: float,
        ttl_s: float,
        submit_ts: float,
        bids: dict[float, float],
        asks: dict[float, float],
        tick_size: float,
        q: float | None = None,
        fair_value: float | None = None,
        quote_id: str | None = None,
    ) -> HypotheticalQuote:
        """Record a quote the strategy chose, with the REAL book state
        around it - not the `distance_to_touch_ticks=0, queue_ahead=0`
        placeholders the synthetic path used."""
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        queue_ahead = bids.get(price, 0.0)
        # Size resting at strictly better (higher) bid prices, which is what
        # a marketable seller consumes before reaching our level.
        ahead_better = sum(size for p, size in bids.items() if p > price)
        distance_ticks = (
            abs(best_bid - price) / tick_size if (best_bid is not None and tick_size > 0) else 0.0
        )
        quote = HypotheticalQuote(
            quote_id=quote_id or self.next_quote_id(round_id),
            round_id=round_id,
            token_id=token_id,
            side=side,
            price=price,
            qty=qty,
            ttl_s=ttl_s,
            submit_ts=submit_ts,
            best_bid_at_submit=best_bid,
            best_ask_at_submit=best_ask,
            queue_ahead_at_submit=queue_ahead,
            distance_to_touch_ticks_at_submit=distance_ticks,
            tick_size=tick_size,
            q_at_submit=q,
            fair_value_at_submit=fair_value,
        )
        quote.queue_samples.append(
            QueueSample(submit_ts, best_bid, best_ask, queue_ahead, ahead_better, q)
        )
        if q is not None:
            quote.q_samples.append((submit_ts, q))
        self._open[quote.quote_id] = quote
        self._last_sample_ts[quote.quote_id] = submit_ts
        return quote

    def cancel(self, quote_id: str, ts: float, reason: str | None = None, replaced_by: str | None = None) -> None:
        quote = self._open.pop(quote_id, None)
        if quote is None:
            return
        quote.cancel_ts = ts
        quote.cancel_reason = reason
        quote.replaced_by = replaced_by
        self._closed.append(quote)

    def expire_due(self, now: float) -> list[HypotheticalQuote]:
        """Close every quote whose TTL has elapsed. Returns them."""
        expired = []
        for qid, quote in list(self._open.items()):
            if now - quote.submit_ts >= quote.ttl_s:
                quote.expired_ts = now
                del self._open[qid]
                self._closed.append(quote)
                expired.append(quote)
        return expired

    # ------------------------------------------------------- observations

    def observe_book(
        self,
        token_id: str,
        ts: float,
        bids: dict[float, float],
        asks: dict[float, float],
        q: float | None = None,
    ) -> None:
        """Feed a book update. Detects first-touch and first-cross, and
        samples the queue at `sample_interval_s`."""
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        for qid, quote in self._open.items():
            if quote.token_id != token_id:
                continue
            quote.book_delta_count += 1

            if quote.first_touch_ts is None and best_bid is not None and best_bid <= quote.price:
                # Our price is now the top of the visible bid ladder: no
                # OTHER bid rests above us. Note the direction - a best bid
                # HIGHER than our price means someone is bidding better and
                # we are still queued behind them, which is the opposite of
                # being at the touch.
                quote.first_touch_ts = ts
            if quote.first_cross_ts is None and best_ask is not None and best_ask <= quote.price:
                # Visible liquidity is offered at or below our bid: a
                # resting order here was marketable at this instant.
                quote.first_cross_ts = ts

            if ts - self._last_sample_ts.get(qid, quote.submit_ts) >= self.sample_interval_s:
                quote.queue_samples.append(
                    QueueSample(
                        ts, best_bid, best_ask,
                        bids.get(quote.price, 0.0),
                        sum(size for p, size in bids.items() if p > quote.price),
                        q,
                    )
                )
                self._last_sample_ts[qid] = ts
            if q is not None and (not quote.q_samples or quote.q_samples[-1][1] != q):
                quote.q_samples.append((ts, q))

    def observe_trade(self, token_id: str, ts: float, price: float, size: float, side: str | None = None) -> None:
        """Feed a `last_trade_price` print. Accumulates the volume that
        traded at or through each open quote's price."""
        for quote in self._open.values():
            if quote.token_id != token_id:
                continue
            if price <= quote.price:
                quote.cumulative_traded_at_or_through += size
                quote.trades.append(TradePrint(ts, price, size, side))

    # ------------------------------------------------------------ reading

    @property
    def open_quotes(self) -> list[HypotheticalQuote]:
        return list(self._open.values())

    @property
    def closed_quotes(self) -> list[HypotheticalQuote]:
        return list(self._closed)

    def all_quotes(self) -> list[HypotheticalQuote]:
        return self.closed_quotes + self.open_quotes

    def close_all(self, ts: float, reason: str = "round_finalized") -> list[HypotheticalQuote]:
        for qid in list(self._open):
            self.cancel(qid, ts, reason=reason)
        return self._closed

    def summary(self, round_id: str | None = None) -> dict:
        quotes = [q for q in self.all_quotes() if round_id is None or q.round_id == round_id]
        touched = [q for q in quotes if q.first_touch_ts is not None]
        crossed = [q for q in quotes if q.first_cross_ts is not None]
        return {
            "n_quotes": len(quotes),
            "n_touch_observed": len(touched),
            "n_cross_observed": len(crossed),
            "mean_queue_ahead_at_submit": (
                sum(q.queue_ahead_at_submit for q in quotes) / len(quotes) if quotes else None
            ),
            "mean_distance_to_touch_ticks": (
                sum(q.distance_to_touch_ticks_at_submit for q in quotes) / len(quotes) if quotes else None
            ),
            "mean_traded_at_or_through": (
                sum(q.cumulative_traded_at_or_through for q in quotes) / len(quotes) if quotes else None
            ),
            # Explicitly NOT called a fill rate. It is the fraction of
            # quotes for which visible liquidity crossed the quote price at
            # least once - an upper-bound-ish indicator, not a measurement
            # of fills, and not a calibrated probability.
            "cross_observed_fraction": (len(crossed) / len(quotes)) if quotes else None,
            "note": (
                "Counterfactual evidence only. No fill probability or q_fill is "
                "estimated in Phase 12C; these fields exist so that estimation can "
                "later be done from real data instead of the synthetic Bernoulli."
            ),
        }
