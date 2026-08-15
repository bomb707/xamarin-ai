"""Captured-sample integrity report (Phase 12C item 15).

    "We are validating data integrity, not profitability."

So this report contains no PnL, no edge, no hit rate and no fill rate. It
answers exactly one question: is the captured dataset trustworthy enough to
proceed to real walk-forward model calibration? Every field item 15 names is
printed, including the ones that are bad news.
"""
from __future__ import annotations

from xamarinbot.realtime.label import summarize_agreement
from xamarinbot.realtime.raw_events import Topic
from xamarinbot.realtime.rtds import (
    TOPIC_BINANCE,
    TOPIC_CHAINLINK,
    TOPIC_TWAP_30,
    TOPIC_TWAP_60,
)

_REFERENCE_LABELS = {
    TOPIC_CHAINLINK: "Chainlink reference updates",
    TOPIC_TWAP_30: "TWAP-30 updates",
    TOPIC_TWAP_60: "TWAP-60 updates",
    TOPIC_BINANCE: "Binance updates",
}


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def format_capture_report(captures: list, metrics, store) -> str:
    """The full item 15 report for a capture session."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("PHASE 12C - REAL CAPTURE SAMPLE REPORT")
    add("Data integrity only. No profitability claim is made or implied.")
    add("=" * 78)

    # ---------------------------------------------- market IDs / windows
    add("")
    add("MARKET IDS / TIME WINDOWS")
    add("-" * 78)
    for c in captures:
        m = c.metadata
        add(f"  {m.round_id}")
        add(f"    condition_id     {m.condition_id}")
        add(f"    question         {m.question}")
        add(f"    window           {m.start_ts:.0f} -> {m.end_ts:.0f}  ({m.duration_s:.0f}s)")
        add(f"    settlement       {m.settlement_kind} (twap_window={m.twap_window_s})")
        add(f"    resolution src   {m.resolution_source}")
        add(f"    tick / min size  {m.tick_size} / {m.min_order_size}")
        add(f"    fee rate         {m.fees.effective_rate} (type={m.fees.fee_type})")
        add(f"    taker delay      {m.taker_delay_ms} ms")
        add(f"    lifecycle        {c.lifecycle.state.value}")
        for w in m.warnings:
            add(f"    ! warning        {w}")
        for n in c.notes:
            add(f"    ! note           {n}")

    # ------------------------------------------------- token/outcome map
    add("")
    add("TOKEN / OUTCOME MAPPING SUCCESS")
    add("-" * 78)
    mapped = 0
    for c in captures:
        m = c.metadata
        ok = bool(m.up_token_id and m.down_token_id)
        mapped += int(ok)
        add(f"  {m.round_id}: {'OK' if ok else 'FAILED'} via {m.outcome_label_source}")
        add(f"    UP   {m.up_token_id}")
        add(f"    DOWN {m.down_token_id}")
    add(f"  mapped {mapped}/{len(captures)} rounds from EXPLICIT outcome labels "
        f"(never token index order)")

    # ------------------------------------------------- events per stream
    add("")
    add("EVENTS PER STREAM")
    add("-" * 78)
    for c in captures:
        rid = c.metadata.round_id
        counts = store.counts_by_topic(rid)
        add(f"  {rid}  (total persisted {store.count(rid)})")
        if not counts:
            add("    (no events attributed to this round)")
        for key in sorted(counts):
            add(f"    {key:<48} {counts[key]}")

    # ----------------------------------------------- book snapshots etc.
    add("")
    add("BOOK SNAPSHOTS / DELTAS / TRADES")
    add("-" * 78)
    for c in captures:
        rid = c.metadata.round_id
        counts = store.counts_by_topic(rid)
        snaps = counts.get("clob_market:book", 0)
        rest_snaps = (
            counts.get("clob_rest:book_snapshot_rest", 0)
            + counts.get("clob_rest:book_snapshot_rest_reconcile", 0)
        )
        deltas = counts.get("clob_market:price_change", 0)
        trades = counts.get("clob_market:last_trade_price", 0)
        bba = counts.get("clob_market:best_bid_ask", 0)
        add(f"  {rid}: ws_snapshots={snaps} rest_snapshots={rest_snaps} "
            f"deltas={deltas} trades={trades} best_bid_ask={bba}")

    # --------------------------------------------- reference feed counts
    add("")
    add("REFERENCE FEED UPDATES")
    add("-" * 78)
    for c in captures:
        add(f"  {c.metadata.round_id}")
        for topic, label in _REFERENCE_LABELS.items():
            obs = c.observations.get(topic, [])
            if obs:
                span = (obs[-1].source_ts_ns - obs[0].source_ts_ns) / 1e9
                pre = sum(1 for o in obs if o.source_ts < c.metadata.start_ts)
                add(f"    {label:<28} {len(obs):>6}  span={span:8.1f}s  pre-round={pre}")
            else:
                add(f"    {label:<28} {0:>6}  (NONE CAPTURED)")

    # ------------------------------------------------ pre-round coverage
    add("")
    add("PRE-ROUND HISTORY COVERAGE (item 7)")
    add("-" * 78)
    for c in captures:
        m = c.metadata
        lead = c.lifecycle.cfg.pre_round_lead_s
        add(f"  {m.round_id}: configured lead {lead:.0f}s")
        for topic, label in _REFERENCE_LABELS.items():
            obs = [o for o in c.observations.get(topic, []) if o.source_ts < m.start_ts]
            if obs:
                earliest = m.start_ts - min(o.source_ts for o in obs)
                verdict = "OK" if earliest >= lead * 0.9 else "SHORT"
                add(f"    {label:<28} {len(obs):>6} pre-round obs, earliest {earliest:7.1f}s "
                    f"before open  [{verdict}]")
            else:
                add(f"    {label:<28} {0:>6} pre-round obs  [MISSING]")

    # --------------------------------------------------------- latencies
    add("")
    add("LATENCY DISTRIBUTIONS")
    add("-" * 78)
    for stats in (metrics.source_to_recv, metrics.publisher_to_recv):
        d = stats.as_dict()
        add(f"  {d['name']:<18} n={d['count']:<8} "
            f"p50={_fmt(d['p50_ms'],1)}ms p95={_fmt(d['p95_ms'],1)}ms "
            f"p99={_fmt(d['p99_ms'],1)}ms max={_fmt(d['max_ms'],1)}ms")
    add("  source->recv includes the oracle/venue's own publication delay;")
    add("  publisher->recv isolates the hop this system could influence.")

    # --------------------------------------------------- recorder health
    add("")
    add("RECORDER HEALTH")
    add("-" * 78)
    md = metrics.as_dict()
    for key in (
        "events_received", "events_persisted", "queue_high_water", "dropped_events",
        "parse_failures", "duplicate_events", "reconnect_count", "resnapshot_count",
        "book_integrity_checks", "book_integrity_mismatches", "freshness_failures",
        "suspect_intervals",
    ):
        add(f"  {key:<28} {md[key]}")

    # --------------------------------------------------- stream stalls
    add("")
    add("STREAM CONTINUITY")
    add("-" * 78)
    stalls = [
        e for e in store.events(topics=[Topic.RECORDER_CONTROL])
        if e.event_type == "stream_stalled"
    ]
    if stalls:
        for e in stalls:
            p = e.payload
            add(f"  ! STALL  {p.get('stream')} silent for {p.get('silent_for_s'):.0f}s "
                f"(threshold {p.get('stall_timeout_s')}s) -> forced reconnect")
        add("  A stalled socket that never raises is the worst failure mode for a")
        add("  recorder: it produces a clean-looking capture with missing data. Each")
        add("  stall above was detected by the watchdog and reconnected, and the")
        add("  affected interval should be checked for a reference-data gap.")
    else:
        add("  no stream stalls detected")
    for c in captures:
        for topic, label in _REFERENCE_LABELS.items():
            obs = c.observations.get(topic, [])
            if not obs:
                continue
            covered = max(o.source_ts for o in obs)
            if covered < c.metadata.end_ts:
                add(f"  ! GAP    {c.metadata.round_id} {label}: last observation "
                    f"{c.metadata.end_ts - covered:.0f}s BEFORE the round close - the "
                    f"settlement boundary is not covered")

    # ------------------------------------------------- book integrity
    add("")
    add("BOOK-INTEGRITY CHECKS (in-memory vs REST resnapshot)")
    add("-" * 78)
    any_checks = False
    for c in captures:
        for r in c.integrity_results:
            any_checks = True
            add(f"  {c.metadata.round_id} {r.token_id[:16]}... "
                f"{'MATCH' if r.matched else 'MISMATCH'}: {r.detail}")
    if not any_checks:
        add("  (no integrity checks ran - capture too short for the configured interval)")
    add("  A mismatch marks the interval suspect, triggers a resnapshot, and writes a")
    add("  reconciliation event; it never silently continues.")

    # ------------------------------------------------- label agreement
    add("")
    add("LABEL RECONSTRUCTION")
    add("-" * 78)
    add("  Reconstructed independently as  y_hat = UP if P_end >= P_start else DOWN,")
    add("  under BOTH the basis the market metadata declares AND the plain")
    add("  Chainlink-reference basis, then compared to Polymarket's own resolution.")
    add("")
    recs = [c.reconstruction for c in captures if c.reconstruction is not None]
    for c in captures:
        r = c.reconstruction
        if r is None:
            add(f"  {c.metadata.round_id}: not finalized; no reconstruction")
            continue
        add(f"  {r.round_id}")
        add(f"    reported by venue     {r.reported_outcome.value if r.reported_outcome else 'NONE'} "
            f"({r.reported_source or 'n/a'})")
        for b in (r.declared, r.reference):
            add(f"    [{b.basis:<9}] topic={b.topic}")
            add(f"        P_start={_fmt(b.start_value,4)} P_end={_fmt(b.end_value,4)} "
                f"-> {b.outcome.value if b.outcome else 'UNRECONSTRUCTED'}")
            add(f"        boundary offsets: start={_fmt(b.start_offset_s,2)}s end={_fmt(b.end_offset_s,2)}s")
            add(f"        {b.reason}")
        add(f"    declared agrees={r.declared_agrees}  reference agrees={r.reference_agrees}  "
            f"bases agree with each other={r.bases_agree}")

    if recs:
        summary = summarize_agreement(recs)
        add("")
        add("  AGREEMENT SUMMARY")
        for k in (
            "rounds", "declared_basis_agreed", "declared_basis_comparable",
            "declared_basis_agreement_rate", "reference_basis_agreed",
            "reference_basis_comparable", "reference_basis_agreement_rate",
            "bases_agreed_with_each_other", "bases_comparable", "bases_agreement_rate",
            "labels_reproducible",
        ):
            add(f"    {k:<36} {_fmt(summary[k])}")

    # ------------------------------------------------------ the verdict
    add("")
    add("VERDICT")
    add("-" * 78)
    grade = metrics.is_training_grade()
    add(f"  training-grade interval: {grade}")
    if not grade:
        add(f"  disqualified by: {', '.join(metrics.disqualifiers())}")
        add("  Per item 6, a dropped-event/parse-failure/integrity-mismatch count above")
        add("  zero makes this interval unsuitable for high-fidelity model training")
        add("  unless explicitly repaired.")
    labels_ok = summarize_agreement(recs)["labels_reproducible"] if recs else False
    add(f"  labels reproducible:     {labels_ok}")
    if not labels_ok:
        add("  Per item 8, the q model must NOT be trained while labels cannot be")
        add("  reproduced reliably.")
    add("")
    add("  NOT ASSESSED IN THIS PHASE (deliberately): fill probability, q_fill,")
    add("  strategy parameters, profitability. No model was fit and no parameter")
    add("  was tuned from this capture.")
    add("=" * 78)
    return "\n".join(lines)


def format_counterfactual_report(tracker, round_id: str | None = None) -> str:
    """Item 11's counterfactual summary - evidence, not fills."""
    s = tracker.summary(round_id)
    lines = ["MAKER-QUOTE COUNTERFACTUAL CAPTURE (item 11)", "-" * 78]
    for k in ("n_quotes", "n_touch_observed", "n_cross_observed",
              "mean_queue_ahead_at_submit", "mean_distance_to_touch_ticks",
              "mean_traded_at_or_through", "cross_observed_fraction"):
        lines.append(f"  {k:<32} {_fmt(s[k])}")
    lines.append(f"  note: {s['note']}")
    return "\n".join(lines)
