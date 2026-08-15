"""Phase 12C item 11: counterfactual capture for hypothetical maker quotes.

The point of these tests is that nothing here claims a fill. They assert
that the raw evidence needed to LATER estimate rho and q_fill from real data
is captured, and that the synthetic Bernoulli's placeholders
(`distance_to_touch_ticks=0`, `queue_ahead_shares=0`) are replaced by real
observed state.
"""
from __future__ import annotations

import pytest

from xamarinbot.realtime.counterfactual import MakerCounterfactualTracker

BIDS = {0.44: 100.0, 0.43: 250.0, 0.42: 90.0}
ASKS = {0.46: 80.0, 0.47: 300.0}


def register(tracker, price=0.43, **kwargs):
    return tracker.register(
        round_id="r1", token_id="tok", side="UP", price=price, qty=25.0,
        ttl_s=60.0, submit_ts=100.0, bids=dict(BIDS), asks=dict(ASKS),
        tick_size=0.01, q=0.55, fair_value=0.55, **kwargs,
    )


def test_state_at_submit_is_real_not_zeroed():
    """Item 11: "stop re-evaluating resting maker economics with artificial
    distance_to_touch_ticks = 0 / queue_ahead_shares = 0"."""
    t = MakerCounterfactualTracker()
    q = register(t, price=0.43)
    assert q.best_bid_at_submit == 0.44
    assert q.best_ask_at_submit == 0.46
    assert q.queue_ahead_at_submit == 250.0        # real size resting at our price
    assert q.distance_to_touch_ticks_at_submit == pytest.approx(1.0)  # one tick off the touch
    assert q.queue_ahead_at_submit != 0.0
    assert q.distance_to_touch_ticks_at_submit != 0.0


def test_every_field_item_11_names_is_recorded():
    t = MakerCounterfactualTracker()
    q = register(t)
    t.observe_book("tok", 101.0, {0.43: 200.0, 0.42: 5.0}, {0.46: 80.0}, q=0.56)
    t.observe_trade("tok", 101.5, 0.43, 12.0, "SELL")
    t.observe_book("tok", 103.0, {0.43: 50.0}, {0.43: 10.0}, q=0.57)
    d = q.as_dict()
    for field in (
        "submit_ts", "side", "price", "qty", "ttl_s",
        "best_bid_at_submit", "best_ask_at_submit",
        "queue_ahead_at_submit", "distance_to_touch_ticks_at_submit",
        "first_touch_ts", "first_cross_ts", "cumulative_traded_at_or_through",
        "book_delta_count", "q_at_submit", "q_at_end",
    ):
        assert field in d, f"{field} not captured"
    assert d["book_delta_count"] == 2
    assert d["cumulative_traded_at_or_through"] == pytest.approx(12.0)


def test_first_touch_is_when_the_best_bid_reaches_our_price():
    t = MakerCounterfactualTracker()
    q = register(t, price=0.43)
    t.observe_book("tok", 101.0, {0.44: 10.0, 0.43: 250.0}, dict(ASKS))
    assert q.first_touch_ts is None       # still one level behind
    t.observe_book("tok", 102.0, {0.43: 250.0}, dict(ASKS))
    assert q.first_touch_ts == 102.0
    assert q.time_to_first_touch_s == pytest.approx(2.0)


def test_first_cross_is_when_visible_liquidity_is_offered_at_or_below_our_bid():
    t = MakerCounterfactualTracker()
    q = register(t, price=0.43)
    t.observe_book("tok", 101.0, dict(BIDS), {0.46: 80.0})
    assert q.first_cross_ts is None
    t.observe_book("tok", 104.0, dict(BIDS), {0.43: 5.0})
    assert q.first_cross_ts == 104.0
    assert q.time_to_first_cross_s == pytest.approx(4.0)


def test_cross_observed_is_evidence_not_a_fill_claim():
    """The tracker must never assert a fill happened - only that one was
    achievable. That distinction is the whole reason this module replaces
    the Bernoulli draw for real-data evaluation."""
    t = MakerCounterfactualTracker()
    q = register(t)
    t.observe_book("tok", 104.0, dict(BIDS), {0.43: 5.0})
    d = q.as_dict()
    assert d["cross_observed"] is True
    assert "filled" not in d
    assert "fill_probability" not in d
    s = t.summary("r1")
    assert "fill_rate" not in s
    assert "rho" not in s
    assert "Counterfactual evidence only" in s["note"]


def test_only_trades_at_or_through_our_price_accumulate():
    t = MakerCounterfactualTracker()
    q = register(t, price=0.43)
    t.observe_trade("tok", 101.0, 0.50, 100.0)   # far above our bid
    assert q.cumulative_traded_at_or_through == 0.0
    t.observe_trade("tok", 102.0, 0.43, 7.0)     # at our price
    t.observe_trade("tok", 103.0, 0.41, 3.0)     # through our price
    assert q.cumulative_traded_at_or_through == pytest.approx(10.0)
    assert len(q.trades) == 2


def test_queue_ahead_is_sampled_over_the_quotes_lifetime():
    t = MakerCounterfactualTracker(sample_interval_s=1.0)
    q = register(t, price=0.43)
    for i, size in enumerate([200.0, 150.0, 40.0], start=1):
        t.observe_book("tok", 100.0 + i, {0.43: size}, dict(ASKS))
    sizes = [s.size_at_quote_price for s in q.queue_samples]
    assert sizes[0] == 250.0     # at submit
    assert sizes[-1] == 40.0     # the queue drained over the lifetime
    assert q.as_dict()["queue_ahead_first"] == 250.0
    assert q.as_dict()["queue_ahead_last"] == 40.0


def test_q_is_tracked_at_submit_and_over_the_lifetime():
    t = MakerCounterfactualTracker()
    q = register(t)
    t.observe_book("tok", 101.0, dict(BIDS), dict(ASKS), q=0.61)
    t.observe_book("tok", 102.0, dict(BIDS), dict(ASKS), q=0.63)
    assert q.q_at_submit == 0.55
    assert q.q_at_end == 0.63


def test_cancel_and_replace_timestamps_are_recorded():
    t = MakerCounterfactualTracker()
    q = register(t)
    t.cancel(q.quote_id, 120.0, reason="EDGE_FAILURE", replaced_by="r1-hq2")
    assert q.cancel_ts == 120.0
    assert q.cancel_reason == "EDGE_FAILURE"
    assert q.replaced_by == "r1-hq2"
    assert q.end_ts == 120.0
    assert t.open_quotes == []


def test_ttl_expiry_closes_the_quote():
    t = MakerCounterfactualTracker()
    q = register(t)
    assert t.expire_due(150.0) == []          # ttl is 60s from 100.0
    expired = t.expire_due(161.0)
    assert expired == [q]
    assert q.expired_ts == 161.0


def test_observations_are_routed_by_token():
    t = MakerCounterfactualTracker()
    q = register(t)
    t.observe_book("other-token", 101.0, {0.99: 1.0}, {0.99: 1.0})
    t.observe_trade("other-token", 101.0, 0.01, 500.0)
    assert q.book_delta_count == 0
    assert q.cumulative_traded_at_or_through == 0.0
