"""Phase 12C items 5 and 8: independent settlement-label reconstruction.

Also pins the brief-vs-live discrepancy: the brief states the rule against
the plain Chainlink reference, while every live BTC 5-minute market's own
metadata declares a Chainlink 60-second TWAP settlement source. Both bases
are reconstructed and compared; nothing here silently picks one.
"""
from __future__ import annotations

import pytest

from xamarinbot.realtime.label import (
    Outcome,
    reconstruct_basis,
    reconstruct_label,
    reported_outcome_from_gamma,
    reported_outcome_from_market_resolved,
    summarize_agreement,
    topic_for_basis,
)
from xamarinbot.realtime.rtds import (
    TOPIC_BINANCE,
    TOPIC_CHAINLINK,
    TOPIC_TWAP_30,
    TOPIC_TWAP_60,
    ReferenceObservation,
)

START = 1_786_772_100.0
END = START + 300.0


def obs(topic, ts, value, window=None):
    return ReferenceObservation(
        topic=topic, symbol="btc/usd", value=value,
        full_accuracy_value=value, window_s=window,
        source_ts_ns=int(ts * 1e9), publisher_ts_ns=int((ts + 1.5) * 1e9),
        recv_wall_ns=int((ts + 1.6) * 1e9), recv_monotonic_ns=1,
    )


def series(topic, start_val, end_val, window=None):
    """One observation per second across the round, plus pre-round lead."""
    out = []
    for i in range(-60, 361):
        ts = START + i
        frac = min(max(i / 300.0, 0.0), 1.0)
        out.append(obs(topic, ts, start_val + (end_val - start_val) * frac, window))
    return out


# ------------------------------------------------------------ basis choice

def test_declared_basis_follows_the_market_metadata():
    assert topic_for_basis("chainlink_twap", 60) == TOPIC_TWAP_60
    assert topic_for_basis("chainlink_twap", 30) == TOPIC_TWAP_30
    assert topic_for_basis("chainlink_reference", None) == TOPIC_CHAINLINK


def test_binance_is_never_a_settlement_basis():
    """Item 5: "Do not use Binance or TWAP as a substitute settlement
    source." Binance is not reachable as a basis under any configuration."""
    for kind in ("chainlink_twap", "chainlink_reference", "anything_else"):
        for w in (None, 30, 60):
            assert topic_for_basis(kind, w) != TOPIC_BINANCE


# ------------------------------------------------------ the rule itself

def test_up_when_end_is_greater_than_start():
    r = reconstruct_basis("declared", TOPIC_TWAP_60, series(TOPIC_TWAP_60, 63000.0, 63100.0, 60),
                          START, END)
    assert r.outcome is Outcome.UP


def test_down_when_end_is_below_start():
    r = reconstruct_basis("declared", TOPIC_TWAP_60, series(TOPIC_TWAP_60, 63100.0, 63000.0, 60),
                          START, END)
    assert r.outcome is Outcome.DOWN


def test_exactly_flat_resolves_up_because_the_rule_is_greater_or_equal():
    r = reconstruct_basis("declared", TOPIC_TWAP_60, series(TOPIC_TWAP_60, 63000.0, 63000.0, 60),
                          START, END)
    assert r.outcome is Outcome.UP


def test_boundaries_are_selected_by_source_timestamp_not_receive_time():
    """The start reference is the last observation at or BEFORE the open;
    the end reference the first at or AFTER the close. Receive times here
    are deliberately 1.6s later than the observation times, so a receive-time
    selection would pick different points."""
    data = series(TOPIC_TWAP_60, 63000.0, 63100.0, 60)
    r = reconstruct_basis("declared", TOPIC_TWAP_60, data, START, END)
    assert r.start_obs_ts_ns == int(START * 1e9)
    assert r.end_obs_ts_ns == int(END * 1e9)
    assert r.start_offset_s == pytest.approx(0.0)
    assert r.end_offset_s == pytest.approx(0.0)


def test_missing_boundary_observation_refuses_to_reconstruct():
    """A gap at the boundary must produce no label rather than a guess from
    the nearest available point."""
    data = [o for o in series(TOPIC_TWAP_60, 63000.0, 63100.0, 60)
            if not (END - 30 <= o.source_ts <= END + 30)]
    r = reconstruct_basis("declared", TOPIC_TWAP_60, data, START, END, tolerance_s=5.0)
    assert r.outcome is None
    assert r.is_reconstructed is False
    assert "missing boundary observation" in r.reason


def test_no_observations_at_all_is_reported_not_crashed():
    r = reconstruct_basis("declared", TOPIC_TWAP_60, [], START, END)
    assert r.outcome is None
    assert "no crypto_prices_twap_sixty observations" in r.reason


# ------------------------------------------- both bases, and their conflict

def test_both_bases_are_reconstructed_for_every_round():
    rec = reconstruct_label(
        "r1", "chainlink_twap", 60,
        {
            TOPIC_TWAP_60: series(TOPIC_TWAP_60, 63000.0, 63100.0, 60),
            TOPIC_CHAINLINK: series(TOPIC_CHAINLINK, 63000.0, 63100.0),
        },
        START, END, reported_outcome=Outcome.UP, reported_source="gamma",
    )
    assert rec.declared.topic == TOPIC_TWAP_60
    assert rec.reference.topic == TOPIC_CHAINLINK
    assert rec.declared.outcome is Outcome.UP
    assert rec.reference.outcome is Outcome.UP
    assert rec.declared_agrees is True
    assert rec.reference_agrees is True
    assert rec.bases_agree is True


def test_a_round_where_the_two_bases_disagree_is_flagged_not_resolved():
    """This is the discriminating case for the brief-vs-live discrepancy:
    the TWAP rose over the window while the spot reference fell."""
    rec = reconstruct_label(
        "r1", "chainlink_twap", 60,
        {
            TOPIC_TWAP_60: series(TOPIC_TWAP_60, 63000.0, 63100.0, 60),
            TOPIC_CHAINLINK: series(TOPIC_CHAINLINK, 63000.0, 62900.0),
        },
        START, END, reported_outcome=Outcome.UP,
    )
    assert rec.declared.outcome is Outcome.UP
    assert rec.reference.outcome is Outcome.DOWN
    assert rec.bases_agree is False
    # only the declared basis matched the venue - an empirical finding,
    # recorded rather than assumed either way
    assert rec.declared_agrees is True
    assert rec.reference_agrees is False


def test_agreement_summary_reports_both_rates_and_the_reproducibility_gate():
    good = reconstruct_label(
        "r1", "chainlink_twap", 60,
        {TOPIC_TWAP_60: series(TOPIC_TWAP_60, 1.0, 2.0, 60),
         TOPIC_CHAINLINK: series(TOPIC_CHAINLINK, 1.0, 0.5)},
        START, END, reported_outcome=Outcome.UP,
    )
    s = summarize_agreement([good])
    assert s["declared_basis_agreement_rate"] == 1.0
    assert s["reference_basis_agreement_rate"] == 0.0
    assert s["labels_reproducible"] is True


def test_labels_not_reproducible_when_neither_basis_matches():
    bad = reconstruct_label(
        "r1", "chainlink_twap", 60,
        {TOPIC_TWAP_60: series(TOPIC_TWAP_60, 2.0, 1.0, 60),
         TOPIC_CHAINLINK: series(TOPIC_CHAINLINK, 2.0, 1.0)},
        START, END, reported_outcome=Outcome.UP,
    )
    s = summarize_agreement([bad])
    assert s["labels_reproducible"] is False


# ------------------------------------------------- the venue's own result

def test_open_market_outcome_prices_are_not_treated_as_a_resolution():
    """An open market's outcomePrices is the live mid; reading it as a
    resolution would fabricate a label for every round."""
    outcome, reason = reported_outcome_from_gamma({
        "closed": False, "outcomes": '["Up", "Down"]', "outcomePrices": '["0.505", "0.495"]',
    })
    assert outcome is None
    assert reason == "market not closed"


def test_closed_market_with_a_decisive_price_gives_the_outcome():
    up, _ = reported_outcome_from_gamma({
        "closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["1", "0"]',
    })
    down, _ = reported_outcome_from_gamma({
        "closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["0", "1"]',
    })
    assert up is Outcome.UP and down is Outcome.DOWN


def test_closed_but_ambiguous_prices_give_no_outcome():
    outcome, reason = reported_outcome_from_gamma({
        "closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["0.5", "0.5"]',
    })
    assert outcome is None
    assert "no unambiguous winning outcome" in reason


def test_market_resolved_event_is_read_when_captured():
    assert reported_outcome_from_market_resolved({"winning_outcome": "Down"})[0] is Outcome.DOWN
    assert reported_outcome_from_market_resolved({"outcome": "Up"})[0] is Outcome.UP
    assert reported_outcome_from_market_resolved({"nothing": 1})[0] is None
