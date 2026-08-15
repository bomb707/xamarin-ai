"""Phase 12C item 2: market discovery and metadata.

Fixtures below are VERBATIM excerpts of live responses captured on
2026-08-15, not invented shapes - including the traps (`startDate` a day
before the round, JSON-encoded `outcomes`/`clobTokenIds` strings, an
unpopulated CLOB `tokens` array for a not-yet-live market).
"""
from __future__ import annotations

import json

import pytest

from xamarinbot.realtime.discovery import (
    MarketDiscovery,
    MarketDiscoveryError,
    build_metadata,
    parse_iso8601,
    round_start_for,
    slug_for_round_start,
)

# --- live capture, 2026-08-15, round starting 2026-08-16T05:00:00Z --------
LIVE_GAMMA = {
    "id": "3598766",
    "question": "Bitcoin Up or Down - August 16, 1:00AM-1:05AM ET",
    "conditionId": "0xc0c6407cec278c626c1a129615d1b2a84bf263e833c90885c9a48753752942c6",
    "questionID": "0x46d58b5f52b31a223c7809aa97f1b4463602de674621b58ce6ae6ee784b99f7a",
    "slug": "btc-updown-5m-1786856400",
    "resolutionSource": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
    # NOTE: startDate is the ROW CREATION time, ~24h before the round.
    "startDate": "2026-08-15T05:08:42.831334Z",
    "endDate": "2026-08-16T05:05:00Z",
    "eventStartTime": "2026-08-16T05:00:00Z",
    "description": "This market will resolve to \"Up\" if the time-weighted average price (TWAP) of Bitcoin...",
    "outcomes": "[\"Up\", \"Down\"]",
    "outcomePrices": "[\"0.505\", \"0.495\"]",
    "clobTokenIds": "[\"447413276661762906288585216172399813197578048801541282311616544503234928995\", \"23220182791370624177894013008514261155772426308054044036371232485690905399131\"]",
    "orderPriceMinTickSize": 0.01,
    "orderMinSize": 5,
    "makerBaseFee": 1000,
    "takerBaseFee": 1000,
    "feesEnabled": True,
    "feeType": "crypto_fees_v2",
    "feeSchedule": {"exponent": 1, "rate": 0.07, "takerOnly": True, "rebateRate": 0.2},
    "makerRebatesFeeShareBps": 10000,
    "cryptoMarketConfigId": "btc-5m-twap-60",
    "cryptoMarketConfig": {
        "id": "btc-5m-twap-60", "asset": "btc", "duration": "5m",
        "twapEnabled": True, "twapLookbackSeconds": 60,
    },
    "closed": False,
    "events": [{"id": "852046", "startTime": "2026-08-16T05:00:00Z", "seriesSlug": "btc-up-or-down-5m"}],
}

# --- live capture, 2026-08-15, a round that WAS live ----------------------
LIVE_CLOB = {
    "enable_order_book": True,
    "active": True,
    "closed": False,
    "accepting_orders": True,
    "minimum_order_size": 5,
    "minimum_tick_size": 0.01,
    "condition_id": "0xc0c6407cec278c626c1a129615d1b2a84bf263e833c90885c9a48753752942c6",
    "question_id": "0x46d58b5f52b31a223c7809aa97f1b4463602de674621b58ce6ae6ee784b99f7a",
    "market_slug": "btc-updown-5m-1786856400",
    "seconds_delay": 0,
    "maker_base_fee": 1000,
    "taker_base_fee": 1000,
    "tokens": [
        {"token_id": "447413276661762906288585216172399813197578048801541282311616544503234928995",
         "outcome": "Up", "price": 0.505, "winner": False},
        {"token_id": "23220182791370624177894013008514261155772426308054044036371232485690905399131",
         "outcome": "Down", "price": 0.495, "winner": False},
    ],
}

# --- live capture: a market ~24h out, whose CLOB tokens are NOT populated -
LIVE_CLOB_NOT_YET_LIVE = {
    "enable_order_book": False,
    "active": False,
    "closed": False,
    "accepting_orders": False,
    "minimum_order_size": 5,
    "minimum_tick_size": 0.01,
    "condition_id": LIVE_GAMMA["conditionId"],
    "seconds_delay": 0,
    "tokens": [
        {"token_id": "", "outcome": "", "price": 0, "winner": False},
        {"token_id": "", "outcome": "", "price": 0, "winner": False},
    ],
}


def test_parse_iso8601_handles_the_forms_gamma_actually_emits():
    # Z suffix, whole seconds
    assert parse_iso8601("2026-08-16T05:05:00Z") == pytest.approx(1786856700.0)
    # Z suffix with six-digit fractional seconds
    assert parse_iso8601("2026-08-15T05:08:42.831334Z") == pytest.approx(1786770522.831334)
    # missing values stay missing rather than becoming the epoch
    assert parse_iso8601(None) is None
    assert parse_iso8601("") is None


def test_iso_timestamps_are_not_floats():
    """The previous adapter did `float(payload["endDate"])`, which raises on
    every real payload. This is the regression guard for that."""
    with pytest.raises(ValueError):
        float(LIVE_GAMMA["endDate"])
    assert parse_iso8601(LIVE_GAMMA["endDate"]) is not None


def test_round_window_uses_event_start_not_row_creation_start_date():
    meta = build_metadata(LIVE_GAMMA, LIVE_CLOB)
    # eventStartTime, NOT startDate (which is ~24h earlier)
    assert meta.start_ts == parse_iso8601("2026-08-16T05:00:00Z")
    assert meta.end_ts == parse_iso8601("2026-08-16T05:05:00Z")
    assert meta.duration_s == 300.0
    # and the slug's unix timestamp agrees with the chosen start
    assert meta.start_ts == 1786856400
    # the trap is reported rather than silently dodged
    assert any("row-creation time" in w for w in meta.warnings)


def test_slug_timestamp_round_trips_with_round_start():
    assert round_start_for(1786856400 + 137) == 1786856400
    assert slug_for_round_start(1786856400) == "btc-updown-5m-1786856400"


def test_up_down_comes_from_explicit_outcome_labels():
    meta = build_metadata(LIVE_GAMMA, LIVE_CLOB)
    assert meta.up_token_id == LIVE_CLOB["tokens"][0]["token_id"]
    assert meta.down_token_id == LIVE_CLOB["tokens"][1]["token_id"]
    assert meta.outcome_label_source == "clob_market_info+gamma_outcomes"
    assert meta.token_side(meta.up_token_id) == "UP"
    assert meta.token_side(meta.down_token_id) == "DOWN"


def test_reversed_outcome_order_is_followed_not_index_order():
    """The decisive test for "never infer UP/DOWN by token index ordering":
    with the labels reversed, the mapping must reverse too."""
    clob = json.loads(json.dumps(LIVE_CLOB))
    clob["tokens"][0]["outcome"] = "Down"
    clob["tokens"][1]["outcome"] = "Up"
    gamma = dict(LIVE_GAMMA, outcomes="[\"Down\", \"Up\"]")
    meta = build_metadata(gamma, clob)
    assert meta.up_token_id == LIVE_CLOB["tokens"][1]["token_id"]
    assert meta.down_token_id == LIVE_CLOB["tokens"][0]["token_id"]


def test_conflicting_mappings_raise_rather_than_silently_preferring_one():
    clob = json.loads(json.dumps(LIVE_CLOB))
    clob["tokens"][0]["outcome"] = "Down"
    clob["tokens"][1]["outcome"] = "Up"
    # Gamma still says index 0 is Up -> genuine disagreement
    with pytest.raises(MarketDiscoveryError, match="disagree on UP/DOWN"):
        build_metadata(LIVE_GAMMA, clob)


def test_no_explicit_labels_anywhere_refuses_to_guess():
    gamma = {k: v for k, v in LIVE_GAMMA.items() if k not in ("outcomes", "clobTokenIds")}
    with pytest.raises(MarketDiscoveryError, match="refusing to infer UP/DOWN"):
        build_metadata(gamma, LIVE_CLOB_NOT_YET_LIVE)


def test_not_yet_live_market_falls_back_to_gamma_labels_and_is_not_executable():
    meta = build_metadata(LIVE_GAMMA, LIVE_CLOB_NOT_YET_LIVE)
    assert meta.outcome_label_source == "gamma_outcomes"
    assert meta.up_token_id and meta.down_token_id
    assert meta.is_executable is False
    assert any("not live yet" in w for w in meta.warnings)


def test_executable_parameters_prefer_clob_market_info():
    gamma = dict(LIVE_GAMMA, orderPriceMinTickSize=0.001, orderMinSize=99)
    meta = build_metadata(gamma, LIVE_CLOB)
    assert meta.tick_size == 0.01       # CLOB's minimum_tick_size
    assert meta.min_order_size == 5.0   # CLOB's minimum_order_size


def test_fee_and_delay_configuration_is_captured_verbatim():
    meta = build_metadata(LIVE_GAMMA, LIVE_CLOB)
    assert meta.fees.effective_rate == 0.07
    assert meta.fees.schedule_taker_only is True
    assert meta.fees.schedule_rebate_rate == 0.2
    assert meta.fees.maker_base_fee == 1000.0   # verbatim, NOT interpreted as a rate
    assert meta.fees.taker_base_fee == 1000.0
    assert meta.fees.fee_type == "crypto_fees_v2"
    assert meta.taker_delay_ms == 0.0
    assert meta.fees.raw["feeSchedule"]["rate"] == 0.07


def test_settlement_kind_and_twap_window_come_from_market_metadata():
    meta = build_metadata(LIVE_GAMMA, LIVE_CLOB)
    assert meta.settlement_kind == "chainlink_twap"
    assert meta.twap_window_s == 60
    assert "twap-60s" in (meta.resolution_source or "")


def test_twap_window_is_not_assumed_when_the_market_does_not_declare_one():
    gamma = {k: v for k, v in LIVE_GAMMA.items() if k != "cryptoMarketConfig"}
    meta = build_metadata(gamma, LIVE_CLOB)
    assert meta.settlement_kind == "chainlink_reference"
    assert meta.twap_window_s is None  # never defaulted to 30


def test_every_item_2_field_is_persisted_in_the_round_row():
    meta = build_metadata(LIVE_GAMMA, LIVE_CLOB)
    row = meta.as_row("sess", "DISCOVERED")
    for field in (
        "round_id", "condition_id", "slug", "question", "description",
        "resolution_source", "start_ts_ns", "end_ts_ns", "up_token_id",
        "down_token_id", "tick_size", "min_order_size", "fee_config_json",
        "taker_delay_ms", "twap_window_s", "raw_metadata_json",
    ):
        assert row[field] is not None, f"{field} missing from persisted round row"
    # raw market metadata is retained in full, not just the parsed fields
    raw = json.loads(row["raw_metadata_json"])
    assert raw["gamma"]["conditionId"] == LIVE_GAMMA["conditionId"]
    assert raw["clob"]["condition_id"] == LIVE_CLOB["condition_id"]


def test_discovery_does_not_hardcode_a_market_and_uses_the_slug_for_the_asked_round():
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        if "/markets" in url and params:
            return [LIVE_GAMMA]
        return LIVE_CLOB

    d = MarketDiscovery(http_get=fake_get)
    meta = d.discover_round(1786856400)
    assert calls[0][1]["slug"] == "btc-updown-5m-1786856400"
    # the condition id used for CLOB market-info came from the Gamma payload
    assert LIVE_GAMMA["conditionId"] in calls[1][0]
    assert meta.condition_id == LIVE_GAMMA["conditionId"]
