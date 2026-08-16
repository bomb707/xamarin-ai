"""Real recorder service - the Phase 12C orchestrator.

Wires discovery, the CLOB market stream, the shared RTDS connection, the
raw recorder, the round lifecycle and the label reconstructor into one
process that captures N consecutive real BTC 5-minute rounds.

What it does NOT do, by design (item 14): authenticate, hold a private key,
or send a maker order, taker order, cancel or replacement. The only
"orders" that exist here are hypothetical quotes recorded for
counterfactual analysis, and they never leave the process.

Ordering of concerns per round (item 8):

    DISCOVERED   metadata fetched and persisted; tokens subscribed
    PRE_ROUND    reference/TWAP/Binance history recorded BEFORE the open
                 (item 7 - the reason PRE_ROUND exists at all)
    ACTIVE       full book + trade capture inside [start, end)
    ENDED        keep capturing through the settle window
    RESOLVED     venue published an outcome
    FINALIZED    final metadata, market_resolved event, winning outcome,
                 boundary Chainlink observations, buffers flushed, metrics
                 closed for the round
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace

from xamarinbot.realtime.attribution import (
    CONNECTION_CONTEXTS,
    RoundWindow,
    Stream,
    attribute_failure,
    attribute_gap,
)
from xamarinbot.realtime.clob_ws import PolymarketMarketStream
from xamarinbot.realtime.identity import RecorderIdentity
from xamarinbot.realtime.counterfactual import MakerCounterfactualTracker
from xamarinbot.realtime.discovery import (
    MarketDiscovery,
    MarketDiscoveryError,
    RealMarketMetadata,
    ROUND_SECONDS,
    round_start_for,
)
from xamarinbot.realtime.label import (
    LabelReconstruction,
    Outcome,
    reconstruct_label,
    reported_outcome_from_clob,
    reported_outcome_from_gamma,
    reported_outcome_from_market_resolved,
)
from xamarinbot.realtime.lifecycle import LifecycleConfig, RoundLifecycle, RoundState
from xamarinbot.realtime.metrics import RecorderMetrics
from xamarinbot.realtime.raw_events import RawEvent, RawEventBuilder, Topic
from xamarinbot.realtime.raw_store import RawEventStore
from xamarinbot.realtime.recorder import RawRecorder, RecorderConfig
from xamarinbot.realtime.rtds import (
    TOPIC_BINANCE,
    TOPIC_CHAINLINK,
    TOPIC_TWAP_30,
    TOPIC_TWAP_60,
    RTDSClient,
)

ALL_REFERENCE_TOPICS = (TOPIC_BINANCE, TOPIC_CHAINLINK, TOPIC_TWAP_30, TOPIC_TWAP_60)


@dataclass(frozen=True)
class ServiceConfig:
    #: How many consecutive rounds to capture.
    n_rounds: int = 3
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    #: How often to verify the in-memory book against a REST resnapshot.
    integrity_check_interval_s: float = 60.0
    #: Service loop tick.
    poll_interval_s: float = 0.5
    #: How long to keep asking Gamma for a resolution after a round ends.
    resolution_poll_interval_s: float = 10.0
    #: Boundary tolerance for label reconstruction: how far from the exact
    #: round boundary an observation may sit and still be used.
    label_tolerance_s: float = 5.0
    #: How long to keep polling for the venue's resolution AFTER every round
    #: has finalized. Measured settlement latency for BTC 5-minute markets is
    #: ~3-8 minutes past the round close, so a per-round tail can never cover
    #: it without idling the recorder; 15 minutes of post-capture sweep does.
    #: Set to 0 to skip the sweep entirely.
    resolution_sweep_s: float = 900.0


@dataclass
class RoundCapture:
    """Everything the service accumulates for one round."""

    metadata: RealMarketMetadata
    lifecycle: RoundLifecycle
    reconstruction: LabelReconstruction | None = None
    reported_outcome: Outcome | None = None
    reported_source: str | None = None
    integrity_results: list = field(default_factory=list)
    #: Reference observations attributed to this round, keyed by RTDS topic.
    observations: dict[str, list] = field(default_factory=dict)
    finalized_at: float | None = None
    notes: list[str] = field(default_factory=list)


class RealRecorderService:
    """Captures `cfg.n_rounds` consecutive real BTC 5-minute rounds."""

    def __init__(
        self,
        store: RawEventStore,
        cfg: ServiceConfig | None = None,
        discovery: MarketDiscovery | None = None,
        session_id: str | None = None,
        log=print,
        identity: RecorderIdentity | None = None,
    ):
        self.cfg = cfg or ServiceConfig()
        self.store = store
        self.session_id = session_id or f"cap-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.discovery = discovery or MarketDiscovery()
        self.metrics = RecorderMetrics()
        self.recorder = RawRecorder(store, self.metrics, self.cfg.recorder)
        self.quotes = MakerCounterfactualTracker()
        self._log = log

        # Separate builders per stream: each reader thread owns its own
        # sequence counter and reconnect generation, so a CLOB reconnect
        # does not renumber RTDS events and vice versa. They share the
        # session id, and ordering across streams is recovered from the
        # receive timestamps rather than from a contended global counter.
        self._clob_builder = RawEventBuilder(session_id=f"{self.session_id}-clob")
        self._rtds_builder = RawEventBuilder(session_id=f"{self.session_id}-rtds")

        # Item 6: stamp WHICH CODE is producing this capture, before a single
        # event is written. A capture with no identity row is a
        # LEGACY_RECORDER capture and is reported separately rather than
        # pooled with post-fix data.
        self.identity = identity or RecorderIdentity.capture()
        for stream_session in (f"{self.session_id}-clob", f"{self.session_id}-rtds"):
            self.store.upsert_session_meta(stream_session, self.identity)
        self.store.upsert_session_meta(self.session_id, self.identity)

        self.rtds = RTDSClient(
            builder=self._rtds_builder,
            on_raw_event=self._submit,
            on_parse_failure=self._on_rtds_parse_failure,
            on_data_gap=self._on_data_gap,
            on_reconnect=lambda gen: self.metrics.record_reconnect(),
            on_observation=self._on_observation,
        )
        self.stream: PolymarketMarketStream | None = None
        self.captures: dict[str, RoundCapture] = {}
        self._last_integrity_check = 0.0
        #: Readiness audit item 4: optional live observers, so a shadow
        #: strategy can consume the SAME event stream the recorder persists
        #: rather than waiting for the capture to be finished and replayed.
        #: Both are called on the recorder's own thread; neither may raise.
        self.on_live_event = None      # (RawEvent) -> None
        self.on_tick = None            # (list[RoundCapture], now: float) -> None
        self.on_round_finalized = None  # (RoundCapture) -> None
        #: Item E: the venue publishes minutes AFTER the shadow loop
        #: finalizes, so "shadow finalized" and "venue resolved" are
        #: different events and need different notifications.
        self.on_round_resolved = None   # (RoundCapture) -> None

    # ------------------------------------------------------------- hooks

    def round_windows(self) -> list[RoundWindow]:
        """Every round's REQUIRED recording interval, for attribution."""
        lc = self.cfg.lifecycle
        return [
            RoundWindow(
                round_id=c.metadata.round_id,
                condition_id=c.metadata.condition_id,
                up_token_id=c.metadata.up_token_id,
                down_token_id=c.metadata.down_token_id,
                start_ts_ns=int(c.metadata.start_ts * 1e9),
                end_ts_ns=int(c.metadata.end_ts * 1e9),
                pre_round_lead_ns=int(lc.pre_round_lead_s * 1e9),
                post_round_tail_ns=int(lc.post_round_tail_s * 1e9),
            )
            for c in self.captures.values()
        ]

    def _on_data_gap(self, gap) -> None:
        """Persist a completed feed outage as a structured attribution.

        Gate A.0.2 items 1-3. The previous `stream_stalled` control event was
        outside the eligibility gate entirely: preflight read only
        `parse_failure`, so a 30+ second RTDS blackout - the exact failure
        the watchdog exists to catch - could be recorded in full and still
        leave every overlapping round `data_valid=True`, because no frame had
        happened to fail parsing.

        A gap shorter than one publication interval is NOT recorded as
        damage: a clean reconnect that resubscribes in milliseconds loses no
        observation, and disqualifying rounds for it would make the whole
        signal noise.
        """
        if not gap.is_material:
            self._log(
                f"[data-gap] {gap.stream} {gap.failure_kind}: "
                f"{gap.duration_ns / 1e9:.3f}s - below one publication interval, "
                "no observation lost"
            )
            return
        self.metrics.record_data_gap()
        self._log(
            f"[data-gap] {gap.stream} {gap.failure_kind}: "
            f"{gap.duration_ns / 1e9:.1f}s of missing observations"
        )
        try:
            attribution = attribute_gap(gap, self.round_windows())
            self.recorder.submit(self._clob_builder.build(
                Topic.RECORDER_CONTROL, "data_gap",
                dict(attribution.as_payload(), duration_s=gap.duration_ns / 1e9),
                round_id=(attribution.affected_round_ids[0]
                          if len(attribution.affected_round_ids) == 1 else None),
            ))
        except Exception:
            # As with parse failures: the counter above already incremented,
            # so a record lost here becomes an UNRECORDED failure and forces
            # the conservative session-wide fallback rather than vanishing.
            pass

    def _on_clob_parse_failure(self, raw, exc) -> None:
        self._on_parse_failure(raw, exc, Stream.CLOB)

    def _on_rtds_parse_failure(self, raw, exc) -> None:
        self._on_parse_failure(raw, exc, Stream.RTDS)

    def _on_parse_failure(self, raw, exc, stream: Stream = Stream.UNKNOWN) -> None:
        """Count the failure AND persist WHICH MARKETS it damaged.

        Gate A.0 wrote a timestamped record but tagged it with whichever
        round was ACTIVE, which is a guess rather than attribution - and a
        guess that is confidently wrong for exactly the failures that matter
        most: a bootstrap for a FUTURE round's token, a connection-wide
        outage, or a gap in the global reference stream. See
        `realtime/attribution.py`; the rule is that a failure which cannot be
        placed is recorded as UNATTRIBUTED and widens rather than narrows.
        """
        self.metrics.record_parse_failure()
        self._log(f"[parse-failure] {stream.value}: {type(exc).__name__}: {exc}")
        try:
            attribution = attribute_failure(
                stream=stream,
                failure_kind=str(raw) if str(raw) in CONNECTION_CONTEXTS else type(exc).__name__,
                recv_timestamp_ns=time.time_ns(),
                raw=str(raw),
                windows=self.round_windows(),
            )
            payload = dict(
                attribution.as_payload(),
                error_type=type(exc).__name__,
                error=str(exc)[:400],
            )
            # `round_id` on the raw row stays a single column, so it carries
            # the sole affected round when there is exactly one and NULL
            # otherwise. `affected_round_ids` in the payload is the truth;
            # the column is an index convenience and is never read as the
            # attribution.
            sole = (attribution.affected_round_ids[0]
                    if len(attribution.affected_round_ids) == 1 else None)
            self.recorder.submit(self._clob_builder.build(
                Topic.RECORDER_CONTROL, "parse_failure", payload, round_id=sole,
            ))
        except Exception:
            # Recording the failure must never itself break the reader. The
            # session counter above already incremented, so a failure lost
            # here becomes an UNRECORDED failure and forces the conservative
            # session-wide fallback rather than disappearing.
            pass

    def _submit(self, event: RawEvent) -> None:
        """Persist, and hand the same event to any live observer.

        The observer sees it BEFORE it is batched to disk, which is what
        makes a live strategy causal: it acts on wire arrival, not on the
        writer's flush cadence.
        """
        self.recorder.submit(event)
        if self.on_live_event is not None:
            try:
                self.on_live_event(event)
            except Exception:
                # A shadow-side failure must never damage the capture - but a
                # swallowed error with no detail is exactly how a broken
                # observer looks healthy. Report it once, with the traceback.
                import traceback

                self._log("[live-observer] event hook failed:\n" + traceback.format_exc())
                self.on_live_event = None

    def _on_market_event(self, event: RawEvent) -> None:
        """Every CLOB event goes to the recorder AND to the maker
        counterfactual tracker (item 11).

        Feeding the tracker from the same stream that maintains the book is
        what makes its observations the ones a real resting order would
        actually have experienced - rather than a separate, subtly different
        view reconstructed afterwards.
        """
        self._submit(event)
        if not self.quotes.open_quotes or event.token_id is None:
            return
        ts = (
            event.source_timestamp_ns / 1e9
            if event.source_timestamp_ns is not None
            else event.recv_wall_timestamp_ns / 1e9
        )
        if event.event_type in ("book", "price_change", "book_snapshot_rest",
                                "book_snapshot_rest_reconcile"):
            book = self.stream.book(event.token_id) if self.stream else None
            if book is not None and not book.awaiting_snapshot:
                self.quotes.observe_book(event.token_id, ts, book.bids, book.asks)
        elif event.event_type == "last_trade_price":
            payload = event.payload
            try:
                self.quotes.observe_trade(
                    event.token_id, ts, float(payload["price"]), float(payload["size"]),
                    payload.get("side"),
                )
            except (KeyError, TypeError, ValueError):
                pass
        self.quotes.expire_due(ts)

    def record_paper_quote(
        self, round_id: str, side: str, price: float, qty: float, ttl_s: float,
        q: float | None = None, now: float | None = None,
    ):
        """Register a hypothetical maker quote for counterfactual capture.

        NOTHING is sent to Polymarket (item 14) - this only starts observing
        what would have happened around a quote at this price. The real
        book state at submit is captured here, which is what replaces the
        synthetic path's `distance_to_touch_ticks=0` / `queue_ahead_shares=0`
        placeholders.
        """
        capture = self.captures.get(round_id)
        if capture is None or self.stream is None:
            return None
        meta = capture.metadata
        token_id = meta.up_token_id if side == "UP" else meta.down_token_id
        if token_id is None:
            return None
        book = self.stream.book(token_id)
        if book is None or book.awaiting_snapshot:
            return None
        now = time.time() if now is None else now
        quote = self.quotes.register(
            round_id=round_id, token_id=token_id, side=side, price=price, qty=qty,
            ttl_s=ttl_s, submit_ts=now, bids=book.bids, asks=book.asks,
            tick_size=book.tick_size or meta.tick_size, q=q,
            fair_value=(q if (q is not None and side == "UP") else (1.0 - q) if q is not None else None),
        )
        self.recorder.submit(self._clob_builder.build(
            Topic.PAPER_QUOTE, "hypothetical_quote_submitted", quote.as_dict(),
            round_id=round_id, condition_id=meta.condition_id, token_id=token_id,
            normalized_side=side,
        ))
        return quote

    def _on_observation(self, obs) -> None:
        """Attribute every reference observation to every round currently
        recording. Reference feeds are global, so a pre-round observation
        legitimately belongs to more than one round's window (item 7's
        lookback overlaps the previous round)."""
        for capture in self.captures.values():
            if capture.lifecycle.is_recording or capture.lifecycle.state is RoundState.DISCOVERED:
                capture.observations.setdefault(obs.topic, []).append(obs)

    def _update_reference_round_tag(self, now: float) -> None:
        """Keep `rtds.current_round_id` pointed at the round a reference
        observation most directly belongs to.

        Found by a real capture on 2026-08-15: this was never assigned, so
        every one of the 2,647 persisted RTDS rows carried `round_id = NULL`
        and no reference data was attributable to any round in the raw log
        at all (the in-memory attribution used for label reconstruction
        still worked, which is exactly what made the gap easy to miss).

        A reference observation genuinely belongs to several rounds' windows
        at once, so this tag is the PRIMARY round only - the ACTIVE one if
        there is one, else the nearest round still recording. Multi-round
        attribution stays a timestamp-window query against `rounds`, which
        is the honest way to express a one-to-many relationship.
        """
        active = [c for c in self.captures.values() if c.lifecycle.state is RoundState.ACTIVE]
        if active:
            self.rtds.current_round_id = active[0].metadata.round_id
            return
        recording = [c for c in self.captures.values() if c.lifecycle.is_recording]
        if recording:
            self.rtds.current_round_id = min(
                recording, key=lambda c: abs(c.metadata.start_ts - now)
            ).metadata.round_id
            return
        upcoming = [c for c in self.captures.values() if not c.lifecycle.is_finished]
        self.rtds.current_round_id = (
            min(upcoming, key=lambda c: c.metadata.start_ts).metadata.round_id
            if upcoming else None
        )

    def _on_lifecycle_transition(self, round_id: str, old: RoundState, new: RoundState, ts: float) -> None:
        self._log(f"[{round_id}] {old.value} -> {new.value}")
        self.recorder.submit(self._clob_builder.build(
            Topic.RECORDER_CONTROL, "lifecycle_transition",
            {"round_id": round_id, "from": old.value, "to": new.value, "ts": ts},
            round_id=round_id,
        ))

    # --------------------------------------------------------- discovery

    def discover(self, now: float | None = None) -> list[RoundCapture]:
        """Discover the rounds to capture.

        Starts at the first round whose FULL pre-round window can still be
        covered, i.e. the first round opening at least `pre_round_lead_s`
        from now - not merely the next round.

        A real capture on 2026-08-15 showed why the weaker rule is not
        enough: starting at "the next round" gave round 1 only 94 seconds of
        pre-round reference history against a configured 420s lead, so its
        early-round momentum and TWAP context were structurally incomplete
        while rounds 2 and 3 were fine. Item 7 requires the lookback, and a
        round that cannot have it should not be in the dataset at all rather
        than be quietly short.
        """
        now = time.time() if now is None else now
        lead = self.cfg.lifecycle.pre_round_lead_s
        first_start = round_start_for(now) + ROUND_SECONDS
        while first_start - now < lead:
            first_start += ROUND_SECONDS
        out: list[RoundCapture] = []
        for i in range(self.cfg.n_rounds):
            start = first_start + i * ROUND_SECONDS
            try:
                meta = self.discovery.discover_round(start)
            except MarketDiscoveryError as exc:
                self._log(f"[discovery] round at {start} unavailable: {exc}")
                continue
            lifecycle = RoundLifecycle(
                round_id=meta.round_id, start_ts=meta.start_ts, end_ts=meta.end_ts,
                cfg=self.cfg.lifecycle, on_transition=self._on_lifecycle_transition,
            )
            capture = RoundCapture(metadata=meta, lifecycle=lifecycle)
            self.captures[meta.round_id] = capture
            out.append(capture)

            self.store.upsert_round(meta.as_row(self.session_id, RoundState.DISCOVERED.value))
            self.recorder.submit(self._clob_builder.build(
                Topic.MARKET_METADATA, "market_metadata_discovered",
                {"gamma": meta.raw_gamma, "clob": meta.raw_clob, "warnings": list(meta.warnings)},
                round_id=meta.round_id, condition_id=meta.condition_id,
            ))
            for w in meta.warnings:
                self._log(f"[{meta.round_id}] metadata warning: {w}")
        return out

    def _subscribe_tokens(self, captures: list[RoundCapture]) -> None:
        token_ids, side_map, round_map, cond_map = [], {}, {}, {}
        for c in captures:
            m = c.metadata
            for tid in m.token_ids:
                token_ids.append(tid)
                side_map[tid] = m.token_side(tid)
                round_map[tid] = m.round_id
                cond_map[tid] = m.condition_id
        if not token_ids:
            return
        if self.stream is None:
            self.stream = PolymarketMarketStream(
                token_ids,
                builder=self._clob_builder,
                on_raw_event=self._on_market_event,
                side_for_token=side_map,
                round_for_token=round_map,
                condition_for_token=cond_map,
                on_parse_failure=self._on_clob_parse_failure,
                on_data_gap=self._on_data_gap,
                on_reconnect=lambda gen: self.metrics.record_reconnect(),
                on_resnapshot=lambda tid: self.metrics.record_resnapshot(),
            )
            self.stream.start()
        else:
            self.stream.add_tokens(
                token_ids, side_for_token=side_map,
                round_for_token=round_map, condition_for_token=cond_map,
            )

    # -------------------------------------------------------------- loop

    def run(self) -> list[RoundCapture]:
        """Capture the configured rounds and return their results."""
        self.recorder.start()
        self.rtds.start()

        captures = self.discover()
        if not captures:
            self._log("[service] no rounds discovered; nothing to capture")
            self.shutdown()
            return []

        # Subscribe to every round's tokens up front, so PRE_ROUND capture
        # for round N+1 is already running while round N is ACTIVE - a
        # subscribe at the rollover instant would guarantee a gap exactly
        # where the next round's opening book matters most.
        self._subscribe_tokens(captures)
        # Tag reference observations from the very first one, so PRE_ROUND
        # history is attributable in the raw log and not only in memory.
        self._update_reference_round_tag(time.time())

        last_state_log = 0.0
        try:
            while not all(c.lifecycle.is_finished for c in captures):
                now = time.time()
                for capture in captures:
                    self._tick_round(capture, now)
                self._update_reference_round_tag(now)
                if self.on_tick is not None:
                    try:
                        self.on_tick(captures, now)
                    except Exception:
                        import traceback

                        self._log("[live-observer] tick hook failed:\n"
                                  + traceback.format_exc())
                        self.on_tick = None
                self._maybe_check_integrity(now)
                if now - last_state_log >= 30.0:
                    self._log_progress(captures, now)
                    last_state_log = now
                time.sleep(self.cfg.poll_interval_s)
        except KeyboardInterrupt:
            self._log("[service] interrupted; finalizing what has been captured")
            for capture in captures:
                if not capture.lifecycle.is_finished:
                    capture.notes.append("capture interrupted before normal finalization")
                    self._force_finalize(capture, time.time())

        # Every round is finalized, so the market subscription has nothing
        # left to capture. Stopping it here - BEFORE the resolution sweep -
        # avoids reconnecting into settled markets for the duration of the
        # sweep, which in a real capture produced seven spurious stall/
        # reconnect cycles and a stream of 404s on `/book` for tokens whose
        # markets no longer existed.
        # The reference streams are equally done: the sweep is REST-only.
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.rtds.stop()

        try:
            # Rounds finalize on schedule with their reconstructed label; the
            # venue's own outcome lands minutes later (measured 3-8 min), so
            # it is collected here rather than by idling the recorder through
            # the next round's PRE_ROUND window.
            if self.cfg.resolution_sweep_s > 0:
                self.resolve_pending(max_wait_s=self.cfg.resolution_sweep_s)
        except KeyboardInterrupt:
            self._log("[service] interrupted during the resolution sweep")
        finally:
            self.shutdown()
        return captures

    def _tick_round(self, capture: RoundCapture, now: float) -> None:
        lc = capture.lifecycle
        if lc.is_finished:
            return
        # Clock-driven transitions up to ENDED.
        if lc.state in (RoundState.DISCOVERED, RoundState.PRE_ROUND, RoundState.ACTIVE):
            lc.advance(now)
            self.store.upsert_round(capture.metadata.as_row(self.session_id, lc.state.value))
            return

        if lc.state is RoundState.ENDED:
            # Keep capturing through the tail window, then try to read the
            # venue's own resolution.
            if now < lc.finalize_after_ts:
                self._try_capture_resolution(capture, now)
                return
            self._try_capture_resolution(capture, now)
            if capture.reported_outcome is not None:
                lc.transition_to(RoundState.RESOLVED, now)
            else:
                capture.notes.append(
                    f"no venue resolution observed within {self.cfg.lifecycle.post_round_tail_s:.0f}s "
                    "of round end; finalized without a reported outcome"
                )
                self._finalize(capture, now)
            return

        if lc.state is RoundState.RESOLVED:
            self._finalize(capture, now)

    def resolve_pending(self, max_wait_s: float = 900.0, poll_interval_s: float = 20.0) -> int:
        """Post-capture sweep for rounds that finalized before the venue
        published a resolution.

        LIVE FINDING (2026-08-15): BTC 5-minute markets settle roughly 3-8
        minutes after the round closes. Measured directly: a round that had
        ended 159s earlier was still open, one at 459s was closed with
        `outcomePrices ["1","0"]`, one at 759s was closed with the CLOB
        `winner` flag also set.

        Holding the recorder open for that long per round is the wrong
        trade - it would idle the capture for longer than the round itself
        and delay the next round's PRE_ROUND window. Instead each round is
        finalized on schedule with its reconstructed label, and this sweep
        fills in the venue's own outcome afterwards, updating
        `round_results.reported_outcome` and `label_agreement` in place.

        Returns the number of rounds resolved.
        """
        pending = [
            c for c in self.captures.values()
            if c.reconstruction is not None and c.reported_outcome is None
        ]
        if not pending:
            return 0
        self._log(f"[resolve] {len(pending)} round(s) awaiting the venue's resolution")
        deadline = time.time() + max_wait_s
        resolved = 0
        while pending and time.time() < deadline:
            for capture in list(pending):
                outcome, source = self._fetch_reported_outcome(capture)
                if outcome is None:
                    continue
                capture.reported_outcome, capture.reported_source = outcome, source
                self._apply_resolution(capture)
                pending.remove(capture)
                resolved += 1
                self._log(
                    f"[resolve] {capture.metadata.round_id} -> {outcome.value} ({source}); "
                    f"declared_agrees={capture.reconstruction.declared_agrees} "
                    f"reference_agrees={capture.reconstruction.reference_agrees}"
                )
            if pending:
                time.sleep(poll_interval_s)
        for capture in pending:
            capture.notes.append(
                f"venue resolution still unavailable {max_wait_s:.0f}s after capture; "
                "label agreement could not be evaluated"
            )
        return resolved

    def _fetch_reported_outcome(self, capture: RoundCapture):
        """Gamma first (more timely), CLOB `tokens[].winner` as an
        independent cross-check."""
        m = capture.metadata
        try:
            gamma = self.discovery.fetch_gamma_market(m.slug)
        except Exception as exc:
            self._log(f"[resolve] {m.round_id}: gamma fetch failed: {exc}")
            gamma = None
        if gamma:
            self.recorder.submit(self._clob_builder.build(
                Topic.MARKET_METADATA, "market_metadata_resolution", gamma,
                round_id=m.round_id, condition_id=m.condition_id,
            ))
            outcome, source = reported_outcome_from_gamma(gamma)
            if outcome is not None:
                return outcome, source
        try:
            clob = self.discovery.fetch_clob_market(m.condition_id)
        except Exception as exc:
            self._log(f"[resolve] {m.round_id}: clob fetch failed: {exc}")
            return None, None
        self.recorder.submit(self._clob_builder.build(
            Topic.MARKET_METADATA, "clob_market_info_resolution", clob,
            round_id=m.round_id, condition_id=m.condition_id,
        ))
        return reported_outcome_from_clob(clob)

    def _notify_resolved(self, capture: RoundCapture) -> None:
        if self.on_round_resolved is None:
            return
        try:
            self.on_round_resolved(capture)
        except Exception:
            import traceback

            self._log("[live-observer] resolve hook failed:\n" + traceback.format_exc())

    def _apply_resolution(self, capture: RoundCapture) -> None:
        """Re-run the comparison now that the venue's outcome is known, and
        rewrite this round's result row."""
        m = capture.metadata
        rec = capture.reconstruction
        capture.reconstruction = LabelReconstruction(
            round_id=rec.round_id, declared=rec.declared, reference=rec.reference,
            reported_outcome=capture.reported_outcome,
            reported_source=capture.reported_source,
        )
        rec = capture.reconstruction
        self.recorder.submit(self._clob_builder.build(
            Topic.RECORDER_CONTROL, "label_reconstruction_resolved",
            _reconstruction_payload(rec), round_id=m.round_id, condition_id=m.condition_id,
        ))
        self.store.upsert_round_result({
            "round_id": m.round_id,
            "reported_outcome": capture.reported_outcome.value if capture.reported_outcome else None,
            "reconstructed_outcome": rec.declared.outcome.value if rec.declared.outcome else None,
            "reconstruction_basis": f"{rec.declared.basis}:{rec.declared.topic}",
            "label_agreement": None if rec.declared_agrees is None else int(rec.declared_agrees),
            "start_reference_value": rec.declared.start_value,
            "end_reference_value": rec.declared.end_value,
            "start_reference_ts_ns": rec.declared.start_obs_ts_ns,
            "end_reference_ts_ns": rec.declared.end_obs_ts_ns,
            "metrics_json": json.dumps(self.round_metrics(m.round_id)),
            "is_training_grade": int(self.metrics.is_training_grade()),
            "notes": "; ".join(capture.notes) or None,
        })
        self.recorder.flush()
        self._notify_resolved(capture)

    def _try_capture_resolution(self, capture: RoundCapture, now: float) -> None:
        if capture.reported_outcome is not None:
            return
        # Prefer the market channel's own market_resolved event when we
        # captured one - it is the venue announcing the result on the same
        # stream as the market data, so it needs no extra request.
        if self.stream is not None:
            payload = self.stream.resolution_for(capture.metadata.condition_id)
            if payload:
                outcome, source = reported_outcome_from_market_resolved(payload)
                if outcome is not None:
                    capture.reported_outcome, capture.reported_source = outcome, source
                    return
        # Fall back to polling Gamma, rate-limited.
        key = f"_last_res_poll_{capture.metadata.round_id}"
        last = getattr(self, key, 0.0)
        if now - last < self.cfg.resolution_poll_interval_s:
            return
        setattr(self, key, now)
        try:
            gamma = self.discovery.fetch_gamma_market(capture.metadata.slug)
        except Exception as exc:
            self._log(f"[{capture.metadata.round_id}] resolution poll failed: {exc}")
            return
        if not gamma:
            return
        self.recorder.submit(self._clob_builder.build(
            Topic.MARKET_METADATA, "market_metadata_poll", gamma,
            round_id=capture.metadata.round_id, condition_id=capture.metadata.condition_id,
        ))
        outcome, source = reported_outcome_from_gamma(gamma)
        if outcome is not None:
            capture.reported_outcome, capture.reported_source = outcome, source
        capture.metadata = replace(capture.metadata, raw_gamma=gamma)

    def _maybe_check_integrity(self, now: float) -> None:
        if self.stream is None or now - self._last_integrity_check < self.cfg.integrity_check_interval_s:
            return
        self._last_integrity_check = now
        for capture in self.captures.values():
            if capture.lifecycle.state is not RoundState.ACTIVE:
                continue
            for token_id in capture.metadata.token_ids:
                result = self.stream.check_integrity(token_id)
                self.metrics.record_integrity_check(result.matched)
                capture.integrity_results.append(result)
                if not result.matched:
                    # Item 15: mark suspect, resnapshot, record a
                    # reconciliation event - never silently continue. The
                    # resnapshot and the reconciliation event are done
                    # inside check_integrity; this is the marking.
                    capture.notes.append(f"book integrity mismatch on {token_id}: {result.detail}")
                    self._log(f"[{capture.metadata.round_id}] BOOK INTEGRITY MISMATCH: {result.detail}")

    # ---------------------------------------------------------- finalize

    def _finalize(self, capture: RoundCapture, now: float) -> None:
        """Item 8's finalization list, in order."""
        m = capture.metadata
        rid = m.round_id

        # 1. persist final market metadata
        try:
            gamma = self.discovery.fetch_gamma_market(m.slug)
            if gamma:
                self.recorder.submit(self._clob_builder.build(
                    Topic.MARKET_METADATA, "market_metadata_final", gamma,
                    round_id=rid, condition_id=m.condition_id,
                ))
                if capture.reported_outcome is None:
                    outcome, source = reported_outcome_from_gamma(gamma)
                    if outcome is not None:
                        capture.reported_outcome, capture.reported_source = outcome, source
        except Exception as exc:
            capture.notes.append(f"final metadata fetch failed: {exc}")

        # 2. persist the market_resolved event, if one was captured
        if self.stream is not None:
            payload = self.stream.resolution_for(m.condition_id)
            if payload:
                self.recorder.submit(self._clob_builder.build(
                    Topic.MARKET_METADATA, "market_resolved_final", payload,
                    round_id=rid, condition_id=m.condition_id,
                ))
            else:
                capture.notes.append("no market_resolved event captured on the market channel")

        # 3/4. record the winning outcome and the boundary Chainlink
        #      observations, via independent label reconstruction.
        capture.reconstruction = reconstruct_label(
            round_id=rid,
            settlement_kind=m.settlement_kind,
            twap_window_s=m.twap_window_s,
            observations_by_topic=capture.observations,
            start_ts=m.start_ts,
            end_ts=m.end_ts,
            reported_outcome=capture.reported_outcome,
            reported_source=capture.reported_source,
            tolerance_s=self.cfg.label_tolerance_s,
            # Gate A.0 item 3: `reconstruct_label` has always accepted the
            # market's own free text, but nothing passed it, so
            # `rule_text_agrees` was permanently None and could never
            # contribute to `LabelStatus`. A market whose structured
            # `cryptoMarketConfig` contradicts its own published rules is
            # exactly the round whose label should not be trusted.
            resolution_source=m.resolution_source,
            description=m.description,
        )
        self.recorder.submit(self._clob_builder.build(
            Topic.RECORDER_CONTROL, "label_reconstruction",
            _reconstruction_payload(capture.reconstruction),
            round_id=rid, condition_id=m.condition_id,
        ))

        # Close any still-open hypothetical quotes so their counterfactual
        # windows end with the round rather than dangling, and persist each
        # one's complete observation set (item 11).
        self.quotes.close_all(now, reason="round_finalized")
        for quote in self.quotes.all_quotes():
            if quote.round_id != rid:
                continue
            self.recorder.submit(self._clob_builder.build(
                Topic.PAPER_QUOTE, "hypothetical_quote_counterfactual", quote.as_dict(),
                round_id=rid, condition_id=m.condition_id, token_id=quote.token_id,
                normalized_side=quote.side,
            ))

        # 5. flush buffered events
        self.recorder.flush()

        # 6. close recorder metrics for the round
        declared = capture.reconstruction.declared
        self.store.upsert_round_result({
            "round_id": rid,
            "reported_outcome": capture.reported_outcome.value if capture.reported_outcome else None,
            "reconstructed_outcome": declared.outcome.value if declared.outcome else None,
            "reconstruction_basis": f"{declared.basis}:{declared.topic}",
            "label_agreement": (
                None if capture.reconstruction.declared_agrees is None
                else int(capture.reconstruction.declared_agrees)
            ),
            "start_reference_value": declared.start_value,
            "end_reference_value": declared.end_value,
            "start_reference_ts_ns": declared.start_obs_ts_ns,
            "end_reference_ts_ns": declared.end_obs_ts_ns,
            "metrics_json": json.dumps(self.round_metrics(rid)),
            "is_training_grade": int(self.metrics.is_training_grade()),
            "notes": "; ".join(capture.notes) or None,
        })
        capture.finalized_at = now
        capture.lifecycle.transition_to(RoundState.FINALIZED, now)
        self.store.upsert_round(m.as_row(self.session_id, RoundState.FINALIZED.value))
        if self.on_round_finalized is not None:
            try:
                self.on_round_finalized(capture)
            except Exception:
                import traceback

                self._log("[live-observer] finalize hook failed:\n"
                          + traceback.format_exc())

    def _force_finalize(self, capture: RoundCapture, now: float) -> None:
        """Finalize a round from whatever state it is in, for an
        interrupted capture. Walks the legal transitions rather than
        assigning FINALIZED directly, so the transition log stays truthful."""
        lc = capture.lifecycle
        while lc.state in (RoundState.DISCOVERED, RoundState.PRE_ROUND, RoundState.ACTIVE):
            order = [RoundState.DISCOVERED, RoundState.PRE_ROUND, RoundState.ACTIVE, RoundState.ENDED]
            lc.transition_to(order[order.index(lc.state) + 1], now)
        if lc.state in (RoundState.ENDED, RoundState.RESOLVED):
            self._finalize(capture, now)

    # ----------------------------------------------------------- reading

    def round_metrics(self, round_id: str) -> dict:
        """Per-round event counts, alongside the session-wide health
        counters. Counts come from the store (what was actually persisted),
        not from an in-memory tally that could disagree with it."""
        return {
            "event_counts": self.store.counts_by_topic(round_id),
            "total_events": self.store.count(round_id),
            "session_metrics": self.metrics.as_dict(),
        }

    def _log_progress(self, captures: list[RoundCapture], now: float) -> None:
        states = ", ".join(f"{c.metadata.round_id.split('-')[-1]}={c.lifecycle.state.value}" for c in captures)
        self._log(
            f"[service] {states} | events rx={self.metrics.events_received} "
            f"persisted={self.metrics.events_persisted} queue={self.recorder.queue_depth} "
            f"dropped={self.metrics.dropped_events} reconnects={self.metrics.reconnect_count}"
        )

    def shutdown(self) -> None:
        if self.stream is not None:
            self.stream.stop()
        self.rtds.stop()
        self.recorder.stop()


def resolve_from_store(
    store: RawEventStore,
    discovery: MarketDiscovery | None = None,
    max_wait_s: float = 900.0,
    poll_interval_s: float = 20.0,
    log=print,
) -> dict:
    """Fill in venue resolutions and label agreement for an ALREADY-CAPTURED
    session, without re-recording anything.

    Exists because settlement lands 3-8 minutes after a round closes, well
    after the recorder has finalized it (see `resolve_pending`). A capture
    can therefore be banked promptly and its labels completed afterwards -
    including for a capture whose process has already exited, which
    `resolve_pending` cannot do because it works off in-memory state.

    Reads each round's stored `label_reconstruction` raw event rather than
    re-deriving the reconstruction, so the label being compared is exactly
    the one the recorder produced at finalization.
    """
    discovery = discovery or MarketDiscovery()
    rounds = store.round_ids()
    stored: dict[str, dict] = {}
    for rid in rounds:
        for ev in store.events(round_id=rid, topics=[Topic.RECORDER_CONTROL]):
            if ev.event_type in ("label_reconstruction", "label_reconstruction_resolved"):
                stored[rid] = ev.payload  # last one wins
    pending = [rid for rid in rounds if rid in stored and not stored[rid].get("reported_outcome")]
    if not pending:
        log("[resolve] nothing pending")
        return {"resolved": 0, "pending": 0}

    log(f"[resolve] {len(pending)} round(s) awaiting the venue's resolution")
    deadline = time.time() + max_wait_s
    resolved = 0
    results: dict[str, dict] = {}
    while pending and time.time() < deadline:
        for rid in list(pending):
            row = store.get_round(rid) or {}
            slug = row.get("slug") or rid
            condition_id = row.get("condition_id")
            outcome = source = None
            try:
                gamma = discovery.fetch_gamma_market(slug)
                if gamma:
                    outcome, source = reported_outcome_from_gamma(gamma)
            except Exception as exc:
                log(f"[resolve] {rid}: gamma fetch failed: {exc}")
            if outcome is None and condition_id:
                try:
                    outcome, source = reported_outcome_from_clob(
                        discovery.fetch_clob_market(condition_id)
                    )
                except Exception as exc:
                    log(f"[resolve] {rid}: clob fetch failed: {exc}")
            if outcome is None:
                continue

            rec = stored[rid]
            declared = rec.get("declared", {})
            reference = rec.get("reference", {})
            declared_agrees = (
                None if declared.get("outcome") is None
                else declared["outcome"] == outcome.value
            )
            reference_agrees = (
                None if reference.get("outcome") is None
                else reference["outcome"] == outcome.value
            )
            store.upsert_round_result({
                "round_id": rid,
                "reported_outcome": outcome.value,
                "reconstructed_outcome": declared.get("outcome"),
                "reconstruction_basis": f"{declared.get('basis')}:{declared.get('topic')}",
                "label_agreement": None if declared_agrees is None else int(declared_agrees),
                "start_reference_value": declared.get("start_value"),
                "end_reference_value": declared.get("end_value"),
                "start_reference_ts_ns": declared.get("start_obs_ts_ns"),
                "end_reference_ts_ns": declared.get("end_obs_ts_ns"),
                "metrics_json": json.dumps({
                    "event_counts": store.counts_by_topic(rid),
                    "total_events": store.count(rid),
                }),
                "is_training_grade": None,
                "notes": f"resolved post-capture via {source}",
            })
            results[rid] = {
                "reported": outcome.value,
                "declared": declared.get("outcome"),
                "reference": reference.get("outcome"),
                "declared_agrees": declared_agrees,
                "reference_agrees": reference_agrees,
                "source": source,
            }
            log(f"[resolve] {rid} -> {outcome.value} ({source}) "
                f"declared={declared.get('outcome')}({declared_agrees}) "
                f"reference={reference.get('outcome')}({reference_agrees})")
            pending.remove(rid)
            resolved += 1
        if pending:
            time.sleep(poll_interval_s)

    for rid in pending:
        log(f"[resolve] {rid}: venue resolution still unavailable after {max_wait_s:.0f}s")
    return {"resolved": resolved, "pending": len(pending), "results": results}


def _reconstruction_payload(rec: LabelReconstruction) -> dict:
    def basis(b) -> dict:
        return {
            "basis": b.basis, "topic": b.topic,
            "start_value": b.start_value, "end_value": b.end_value,
            "start_obs_ts_ns": b.start_obs_ts_ns, "end_obs_ts_ns": b.end_obs_ts_ns,
            "start_offset_s": b.start_offset_s, "end_offset_s": b.end_offset_s,
            "outcome": b.outcome.value if b.outcome else None,
            "reason": b.reason,
        }

    return {
        "round_id": rec.round_id,
        "declared": basis(rec.declared),
        "reference": basis(rec.reference),
        "reported_outcome": rec.reported_outcome.value if rec.reported_outcome else None,
        "reported_source": rec.reported_source,
        "declared_agrees": rec.declared_agrees,
        "reference_agrees": rec.reference_agrees,
        "bases_agree": rec.bases_agree,
        # Gate A.0 items 2/3: the resolved status, including the rule-text
        # cross-check, so the projection can refuse a non-CONFIRMED label
        # without recomputing it.
        "status": rec.status.value,
        "rule_text_agrees": rec.rule_text_agrees,
        "is_trainable_label": rec.is_trainable,
    }
