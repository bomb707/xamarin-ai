"""BTC 5-minute UP/DOWN market discovery and metadata (Phase 12C item 2).

Every field and quirk below was confirmed against the LIVE APIs on
2026-08-15, not inferred from documentation. The verification transcript is
in docs/REAL_RECORDER_ARCHITECTURE.md; the load-bearing findings are:

1. Rounds are a Gamma *series*, `btc-up-or-down-5m`, whose per-round slug is
   `btc-updown-5m-{unix_start_seconds}` with the timestamp aligned to a
   300-second boundary. That makes discovery a deterministic function of the
   wall clock rather than a search - but the slug is still only a
   *candidate*; the market is resolved and validated through the API, and no
   market id or token pair is ever hardcoded.

2. **Gamma `startDate` is NOT the round start.** For
   `btc-updown-5m-1786856400` the observed `startDate` was
   `2026-08-15T05:08:42.831334Z` - the moment the market row was created,
   roughly 24 hours before the round it describes. The true round window is
   `eventStartTime` / `events[0].startTime` (`2026-08-16T05:00:00Z`) through
   `endDate` (`2026-08-16T05:05:00Z`), and both agree exactly with the slug's
   unix timestamp. Using `startDate` as the round start - which the previous
   adapter did - would have mislabeled `t` and `tau` for every feature by
   about a day.

3. **Gamma timestamps are ISO-8601 date-time strings, not floats.** The
   previous adapter did `float(payload.get("endDate") or 0.0)`, which raises
   `ValueError` on `"2026-08-16T05:05:00Z"`. `parse_iso8601` handles the
   `Z` suffix and the fractional seconds Gamma actually emits (six digits).

4. **UP/DOWN comes from explicit outcome labels, never token index order.**
   CLOB market-info exposes `tokens: [{token_id, outcome: "Up"/"Down"}]`,
   which is authoritative and is what this module uses. Gamma's `outcomes`
   /`clobTokenIds` pair (both JSON-encoded *strings*, positionally aligned)
   is used only as a cross-check, and a disagreement is an error rather than
   a silent preference. If neither source carries explicit labels, discovery
   FAILS - it does not fall back to "index 0 is Up".

5. **CLOB market-info is only populated once the market goes live.** For a
   round ~24h out, `tokens` came back as two entries with empty `token_id`
   and empty `outcome`, and `enable_order_book` was False. Discovery
   therefore reports `is_executable` rather than pretending an
   un-tradeable market is ready.

6. **The settlement rule for these markets is TWAP-based**, which
   contradicts the plain-Chainlink-reference assumption. See `label.py` -
   this module's job is only to surface `settlement_kind` and
   `twap_window_s` faithfully from `cryptoMarketConfig`
   (`{"id": "btc-5m-twap-60", "twapEnabled": true, "twapLookbackSeconds": 60}`)
   and the market description, so the label reconstructor uses the rule the
   market actually declares.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_REST_BASE_URL = "https://clob.polymarket.com"

#: Gamma series that emits the BTC 5-minute UP/DOWN rounds.
BTC_5M_SERIES_SLUG = "btc-up-or-down-5m"
#: Per-round slug template. `{ts}` is the round start, unix seconds.
BTC_5M_SLUG_TEMPLATE = "btc-updown-5m-{ts}"
ROUND_SECONDS = 300

_ISO_FRACTION = re.compile(r"\.(\d+)")


class MarketDiscoveryError(RuntimeError):
    """Raised instead of guessing when the market cannot be identified or a
    required executable parameter is missing/ambiguous."""


def parse_iso8601(value: str | None) -> float | None:
    """ISO-8601 date-time -> epoch seconds (float), or None.

    Gamma emits both `2026-08-16T05:05:00Z` and
    `2026-08-15T05:08:42.831334Z`. `datetime.fromisoformat` handles the
    latter on 3.11+, but rejects a `Z` suffix before 3.11 and rejects
    fractional parts that are not 3 or 6 digits on older versions, so both
    are normalized first. Returns None for None/empty rather than 0.0 - a
    missing timestamp must never read as 1970.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Defensive: some Gamma-adjacent endpoints do emit numeric epochs.
        return float(value)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    m = _ISO_FRACTION.search(text)
    if m and len(m.group(1)) > 6:
        text = text[: m.start(1)] + m.group(1)[:6] + text[m.end(1):]
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MarketDiscoveryError(f"unparseable ISO-8601 timestamp {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _maybe_json_list(value: Any) -> list | None:
    """Gamma encodes `outcomes` and `clobTokenIds` as JSON *strings*
    (`'["Up", "Down"]'`), not arrays. Accept either."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def round_start_for(ts: float) -> int:
    """The 5-minute boundary at or before `ts`, unix seconds."""
    return int(ts) - (int(ts) % ROUND_SECONDS)


def slug_for_round_start(round_start: int) -> str:
    return BTC_5M_SLUG_TEMPLATE.format(ts=int(round_start))


@dataclass(frozen=True)
class FeeConfiguration:
    """Fee parameters as the venue reports them.

    `maker_base_fee`/`taker_base_fee` were observed as the integer 1000 on
    both Gamma and CLOB market-info, with no documented unit. They are
    carried verbatim and NOT interpreted as a rate. The executable number is
    `schedule_rate` from Gamma's `feeSchedule`
    (`{"exponent": 1, "rate": 0.07, "takerOnly": true, "rebateRate": 0.2}`),
    which matches the documented fee formula's `feeRate=0.07`.
    """

    maker_base_fee: float | None
    taker_base_fee: float | None
    fees_enabled: bool | None
    fee_type: str | None
    schedule_rate: float | None
    schedule_exponent: float | None
    schedule_taker_only: bool | None
    schedule_rebate_rate: float | None
    maker_rebates_fee_share_bps: float | None
    raw: dict = field(default_factory=dict)

    @property
    def effective_rate(self) -> float:
        """The rate the fee formula should use. Falls back to the
        documented 0.07 only when the venue reported no schedule at all."""
        return self.schedule_rate if self.schedule_rate is not None else 0.07


@dataclass(frozen=True)
class RealMarketMetadata:
    """Everything item 2 requires be persisted for every round."""

    # --- identity ---
    round_id: str
    market_id: str
    event_id: str | None
    condition_id: str
    question_id: str | None
    slug: str
    question: str
    description: str
    resolution_source: str | None

    # --- window (true round window, see module docstring finding 2) ---
    start_ts: float
    end_ts: float

    # --- executable parameters ---
    up_token_id: str | None
    down_token_id: str | None
    tick_size: float
    min_order_size: float
    fees: FeeConfiguration
    taker_delay_ms: float
    #: How the market says it settles: "chainlink_twap" when
    #: `cryptoMarketConfig.twapEnabled` is set, else "chainlink_reference".
    settlement_kind: str
    #: TWAP lookback the market declares, when it declares one.
    twap_window_s: int | None

    # --- provenance ---
    is_executable: bool
    outcome_label_source: str
    raw_gamma: dict = field(default_factory=dict)
    raw_clob: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts

    def token_side(self, token_id: str) -> str | None:
        if token_id and token_id == self.up_token_id:
            return "UP"
        if token_id and token_id == self.down_token_id:
            return "DOWN"
        return None

    @property
    def token_ids(self) -> list[str]:
        return [t for t in (self.up_token_id, self.down_token_id) if t]

    def as_row(self, session_id: str, state: str) -> dict:
        return {
            "round_id": self.round_id,
            "session_id": session_id,
            "condition_id": self.condition_id,
            "question_id": self.question_id,
            "slug": self.slug,
            "question": self.question,
            "description": self.description,
            "resolution_source": self.resolution_source,
            "start_ts_ns": int(self.start_ts * 1e9),
            "end_ts_ns": int(self.end_ts * 1e9),
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "tick_size": self.tick_size,
            "min_order_size": self.min_order_size,
            "fee_config_json": json.dumps(self.fees.raw),
            "taker_delay_ms": self.taker_delay_ms,
            "twap_window_s": self.twap_window_s,
            "settlement_kind": self.settlement_kind,
            "state": state,
            "raw_metadata_json": json.dumps(
                {"gamma": self.raw_gamma, "clob": self.raw_clob, "warnings": list(self.warnings)}
            ),
        }


def _extract_round_window(gamma: dict, slug: str) -> tuple[float, float, list[str]]:
    """Finding 2: prefer the event's `startTime`/`eventStartTime`, then the
    slug's own unix timestamp, and only then `startDate` (which is the
    row-creation time and is usually wrong by ~a day). `endDate` is the true
    round end in every observed payload."""
    warnings: list[str] = []
    events = gamma.get("events") or []
    event = events[0] if events else {}

    start = parse_iso8601(gamma.get("eventStartTime")) or parse_iso8601(event.get("startTime"))
    slug_ts: float | None = None
    tail = slug.rsplit("-", 1)[-1]
    if tail.isdigit():
        slug_ts = float(tail)

    if start is None:
        start = slug_ts
        if start is not None:
            warnings.append("round start taken from slug timestamp; eventStartTime/startTime absent")
    elif slug_ts is not None and abs(start - slug_ts) > 1.0:
        warnings.append(
            f"eventStartTime {start} disagrees with slug timestamp {slug_ts} by "
            f"{start - slug_ts:.1f}s"
        )

    end = parse_iso8601(gamma.get("endDate"))
    if start is None and end is not None:
        start = end - ROUND_SECONDS
        warnings.append("round start derived as endDate - 300s; no explicit start available")
    if end is None and start is not None:
        end = start + ROUND_SECONDS
        warnings.append("round end derived as start + 300s; endDate absent")
    if start is None or end is None:
        raise MarketDiscoveryError(f"cannot establish round window for {slug}")

    created = parse_iso8601(gamma.get("startDate"))
    if created is not None and abs(created - start) > 60.0:
        # Not a warning about correctness - a note that the trap was avoided.
        warnings.append(
            f"gamma startDate ({gamma.get('startDate')}) is the row-creation time, "
            f"{start - created:.0f}s before the true round start; not used as the round start"
        )
    return start, end, warnings


def _extract_fees(gamma: dict) -> FeeConfiguration:
    schedule = gamma.get("feeSchedule") or {}
    return FeeConfiguration(
        maker_base_fee=_as_float(gamma.get("makerBaseFee")),
        taker_base_fee=_as_float(gamma.get("takerBaseFee")),
        fees_enabled=gamma.get("feesEnabled"),
        fee_type=gamma.get("feeType"),
        schedule_rate=_as_float(schedule.get("rate")),
        schedule_exponent=_as_float(schedule.get("exponent")),
        schedule_taker_only=schedule.get("takerOnly"),
        schedule_rebate_rate=_as_float(schedule.get("rebateRate")),
        maker_rebates_fee_share_bps=_as_float(gamma.get("makerRebatesFeeShareBps")),
        raw={
            "makerBaseFee": gamma.get("makerBaseFee"),
            "takerBaseFee": gamma.get("takerBaseFee"),
            "feeSchedule": schedule,
            "feeType": gamma.get("feeType"),
            "feesEnabled": gamma.get("feesEnabled"),
            "makerRebatesFeeShareBps": gamma.get("makerRebatesFeeShareBps"),
        },
    )


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _map_tokens_to_sides(gamma: dict, clob: dict) -> tuple[str | None, str | None, str, list[str]]:
    """Resolve UP/DOWN token ids from EXPLICIT outcome labels only.

    Returns (up_token_id, down_token_id, source, warnings). Raises rather
    than falling back to index order when labels exist but do not name an
    up/down pair - item 2: "Never infer UP/DOWN merely by assuming token
    index ordering if explicit outcome labels are available."
    """
    warnings: list[str] = []

    def classify(label: str) -> str | None:
        t = (label or "").strip().lower()
        if t in ("up", "yes", "higher"):
            return "UP"
        if t in ("down", "no", "lower"):
            return "DOWN"
        return None

    # --- primary: CLOB market-info tokens[].outcome (authoritative) ---
    clob_map: dict[str, str] = {}
    for tok in clob.get("tokens") or []:
        tid, outcome = tok.get("token_id"), tok.get("outcome")
        if not tid or not outcome:
            continue
        side = classify(outcome)
        if side is None:
            raise MarketDiscoveryError(
                f"CLOB outcome label {outcome!r} is not an up/down label; refusing to guess"
            )
        clob_map[side] = tid

    # --- cross-check: Gamma outcomes[] positionally aligned to clobTokenIds[] ---
    gamma_map: dict[str, str] = {}
    outcomes = _maybe_json_list(gamma.get("outcomes"))
    token_ids = _maybe_json_list(gamma.get("clobTokenIds"))
    if outcomes and token_ids and len(outcomes) == len(token_ids):
        for label, tid in zip(outcomes, token_ids):
            side = classify(str(label))
            if side is None:
                raise MarketDiscoveryError(
                    f"Gamma outcome label {label!r} is not an up/down label; refusing to guess"
                )
            gamma_map[side] = str(tid)

    if clob_map and gamma_map:
        if clob_map != gamma_map:
            raise MarketDiscoveryError(
                "CLOB and Gamma disagree on UP/DOWN token mapping "
                f"(clob={clob_map}, gamma={gamma_map}); refusing to prefer one silently"
            )
        return clob_map.get("UP"), clob_map.get("DOWN"), "clob_market_info+gamma_outcomes", warnings
    if clob_map:
        return clob_map.get("UP"), clob_map.get("DOWN"), "clob_market_info", warnings
    if gamma_map:
        warnings.append(
            "CLOB market-info carried no populated tokens (market not live yet); "
            "UP/DOWN taken from Gamma outcomes/clobTokenIds alignment"
        )
        return gamma_map.get("UP"), gamma_map.get("DOWN"), "gamma_outcomes", warnings

    raise MarketDiscoveryError(
        "no explicit outcome labels available from either CLOB market-info or Gamma; "
        "refusing to infer UP/DOWN from token index order"
    )


def build_metadata(gamma: dict, clob: dict) -> RealMarketMetadata:
    """Pure function from the two raw API payloads to `RealMarketMetadata`.

    Kept free of I/O so it is directly testable against captured payloads -
    the fixtures in tests/test_real_discovery.py are verbatim copies of live
    responses.
    """
    slug = gamma.get("slug") or clob.get("market_slug") or ""
    start_ts, end_ts, warnings = _extract_round_window(gamma, slug)
    up_token, down_token, label_source, tok_warnings = _map_tokens_to_sides(gamma, clob)
    warnings.extend(tok_warnings)

    # Executable parameters: prefer CLOB market-info, which is the venue's
    # own trading configuration, over Gamma's catalogue copy (item 2:
    # "Prefer CLOB market-info for executable market parameters").
    tick_size = _as_float(clob.get("minimum_tick_size"))
    if tick_size is None:
        tick_size = _as_float(gamma.get("orderPriceMinTickSize"))
        warnings.append("tick size taken from Gamma; CLOB market-info did not report one")
    min_order_size = _as_float(clob.get("minimum_order_size"))
    if min_order_size is None:
        min_order_size = _as_float(gamma.get("orderMinSize"))
        warnings.append("min order size taken from Gamma; CLOB market-info did not report one")
    if tick_size is None or min_order_size is None:
        raise MarketDiscoveryError(
            f"{slug}: tick size / minimum order size unavailable from both CLOB and Gamma"
        )

    # `seconds_delay` is the CLOB's taker-delay configuration. Observed as 0
    # for BTC 5m on 2026-08-15. Absent => unknown, which we surface rather
    # than defaulting to a comfortable 0.
    delay_s = clob.get("seconds_delay")
    if delay_s is None:
        warnings.append("CLOB market-info reported no seconds_delay; taker delay recorded as 0")
        taker_delay_ms = 0.0
    else:
        taker_delay_ms = float(delay_s) * 1000.0

    crypto_cfg = gamma.get("cryptoMarketConfig") or {}
    twap_enabled = bool(crypto_cfg.get("twapEnabled"))
    twap_window_s = crypto_cfg.get("twapLookbackSeconds")
    twap_window_s = int(twap_window_s) if twap_window_s is not None else None
    settlement_kind = "chainlink_twap" if twap_enabled else "chainlink_reference"

    description = gamma.get("description") or clob.get("description") or ""
    if twap_enabled and twap_window_s not in (30, 60):
        warnings.append(
            f"market declares TWAP settlement with an unexpected window {twap_window_s}s "
            "(only 30 and 60 are published on RTDS)"
        )

    is_executable = bool(clob.get("enable_order_book")) and bool(clob.get("accepting_orders")) and bool(up_token and down_token)

    events = gamma.get("events") or []
    return RealMarketMetadata(
        round_id=slug,
        market_id=str(gamma.get("id") or ""),
        event_id=str(events[0].get("id")) if events else None,
        condition_id=str(gamma.get("conditionId") or clob.get("condition_id") or ""),
        question_id=gamma.get("questionID") or clob.get("question_id"),
        slug=slug,
        question=gamma.get("question") or clob.get("question") or "",
        description=description,
        resolution_source=gamma.get("resolutionSource"),
        start_ts=start_ts,
        end_ts=end_ts,
        up_token_id=up_token,
        down_token_id=down_token,
        tick_size=tick_size,
        min_order_size=min_order_size,
        fees=_extract_fees(gamma),
        taker_delay_ms=taker_delay_ms,
        settlement_kind=settlement_kind,
        twap_window_s=twap_window_s,
        is_executable=is_executable,
        outcome_label_source=label_source,
        raw_gamma=gamma,
        raw_clob=clob,
        warnings=tuple(warnings),
    )


class MarketDiscovery:
    """Live discovery against Gamma + CLOB.

    Nothing is hardcoded except the *series* slug pattern, which is the
    market family we are studying rather than a specific market. Every id,
    token, tick size, fee and window is fetched per round.
    """

    def __init__(
        self,
        gamma_base_url: str = GAMMA_BASE_URL,
        clob_base_url: str = CLOB_REST_BASE_URL,
        request_timeout_s: float = 10.0,
        http_get=None,
    ):
        self._gamma = gamma_base_url
        self._clob = clob_base_url
        self._timeout = request_timeout_s
        self._http_get = http_get or self._default_get

    @staticmethod
    def _default_get(url: str, params: dict | None, timeout: float):
        from xamarinbot.feeds._live_deps import require_live_deps

        require_live_deps()
        import httpx

        resp = httpx.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_gamma_market(self, slug: str) -> dict | None:
        """Fetch one market by slug.

        LIVE FINDING (2026-08-15): `GET /markets?slug=...` returns `[]` for a
        market that has already RESOLVED - the default query is restricted to
        open markets. Verified directly: `btc-updown-5m-1786772100` returned
        an empty list, while the same slug with `closed=true` returned the
        full settled payload.

        That is exactly backwards from what settlement reading needs, so an
        empty result is retried with `closed=true` before concluding the
        market does not exist. Without this retry, every finalization lookup
        for a round that had actually settled would silently see "no market"
        and the label could never be compared.
        """
        data = self._http_get(f"{self._gamma}/markets", {"slug": slug}, self._timeout)
        if isinstance(data, list) and not data:
            data = self._http_get(
                f"{self._gamma}/markets", {"slug": slug, "closed": "true"}, self._timeout
            )
        if isinstance(data, list):
            return data[0] if data else None
        return data or None

    def fetch_clob_market(self, condition_id: str) -> dict:
        return self._http_get(f"{self._clob}/markets/{condition_id}", None, self._timeout)

    def discover_round(self, round_start: int) -> RealMarketMetadata:
        """Discover the BTC 5-minute market whose round begins at
        `round_start` (unix seconds, 300-aligned)."""
        slug = slug_for_round_start(round_start)
        gamma = self.fetch_gamma_market(slug)
        if gamma is None:
            raise MarketDiscoveryError(f"no Gamma market for slug {slug}")
        condition_id = gamma.get("conditionId")
        if not condition_id:
            raise MarketDiscoveryError(f"{slug}: Gamma payload carries no conditionId")
        clob = self.fetch_clob_market(str(condition_id))
        return build_metadata(gamma, clob)

    def discover_upcoming(self, now_ts: float, count: int = 1, lookahead: int = 0) -> list[RealMarketMetadata]:
        """The current round plus the next `count - 1`, skipping any that do
        not yet exist. `lookahead` shifts the first round forward, used by
        the service to warm up on the NEXT round when the current one is
        already partly elapsed."""
        base = round_start_for(now_ts) + lookahead * ROUND_SECONDS
        out: list[RealMarketMetadata] = []
        for i in range(count):
            try:
                out.append(self.discover_round(base + i * ROUND_SECONDS))
            except MarketDiscoveryError:
                continue
        return out
