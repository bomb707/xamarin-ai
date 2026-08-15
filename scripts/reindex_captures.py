#!/usr/bin/env python3
"""Rebuild `captures/continuous/INDEX.jsonl` from the captures themselves.

Gate A.0 item 1: "Existing captures must be reindexed/revalidated; do not
discard them."

The index written before Gate A.0 lacks the eligibility breakdown, and its
`label_status` predates the rule-text cross-check being wired in (item 3), so
every existing record has to be re-derived. The captures themselves are the
source of truth and are untouched - only the index is rewritten.

Safe to run while continuous capture is going: it writes a new index
atomically and leaves the old one as `INDEX.jsonl.bak`. Rounds captured
during the rebuild are picked up by the next run.

    python scripts/reindex_captures.py
    python scripts/reindex_captures.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xamarinbot.eligibility import summarize  # noqa: E402
from xamarinbot.realtime.preflight import (  # noqa: E402
    attribution_summary,
    evaluate_round,
    label_fields,
)
from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402

CAPTURE_DIR = _REPO_ROOT / "captures" / "continuous"
INDEX = CAPTURE_DIR / "INDEX.jsonl"


def rebuild(
    dbs: list[Path],
    include_in_flight: bool = False,
    verify_projection: bool = True,
) -> tuple[list[dict], int]:
    """Rebuild index rows, plus a count of rounds deliberately left out.

    A round that has not reached FINALIZED is still being captured - the
    batch that owns it has not run its resolution sweep, so it has no label,
    no session metrics and no end-boundary reference. Indexing it would add
    a row that is guaranteed to fail three gates for reasons that say nothing
    about the round, inflating `captured` and every disqualification count
    while continuous capture is running. `append_index` never writes such a
    round (it only sees rounds a finished batch returned), so the rebuild
    matches it. The skipped count is printed, never silently dropped.
    """
    rows: list[dict] = []
    in_flight = 0
    for db in dbs:
        store = RawEventStore(str(db))
        identity = store.recorder_identity()
        # One attribution pass per capture file (item 1) - the verdict is a
        # property of the whole session, not of each round independently.
        summary = attribution_summary(store)
        try:
            for round_id in store.round_ids():
                meta = store.get_round(round_id) or {}
                if not include_in_flight and meta.get("state") != "FINALIZED":
                    in_flight += 1
                    continue
                labels = label_fields(store, round_id)
                elig = evaluate_round(
                    store, round_id,
                    verify_projection_run=verify_projection,
                    attribution=summary,
                )
                rows.append({
                    "recorder_code_sha": identity.recorder_code_sha,
                    "recorder_code_dirty": identity.recorder_code_dirty,
                    "recorder_process_pid": identity.process_pid,
                    "recorder_process_started_at": identity.process_started_at,
                    "recorder_schema_version": identity.recorder_schema_version,
                    "round_id": round_id,
                    "condition_id": meta.get("condition_id"),
                    "start_ts": (meta.get("start_ts_ns") or 0) / 1e9,
                    "end_ts": (meta.get("end_ts_ns") or 0) / 1e9,
                    "db": str(db.relative_to(_REPO_ROOT)),
                    "state": meta.get("state"),
                    "settlement_kind": meta.get("settlement_kind"),
                    "twap_window_s": meta.get("twap_window_s"),
                    "min_order_size": meta.get("min_order_size"),
                    "tick_size": meta.get("tick_size"),
                    "reported_outcome": labels["reported_outcome"],
                    "reconstructed_outcome": labels["reconstructed_outcome"],
                    "label_status": labels["label_status"],
                    "declared_agrees": labels["declared_agrees"],
                    "rule_text_agrees": labels["rule_text_agrees"],
                    "notes": None,
                    "indexed_at": time.time(),
                    "reindexed": True,
                    **elig.as_index_fields(),
                })
        finally:
            store.close()
    rows.sort(key=lambda r: (r["start_ts"], r["round_id"]))
    return rows, in_flight


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--dir", default=str(CAPTURE_DIR))
    ap.add_argument("--include-in-flight", action="store_true",
                    help="also index rounds the running batch has not finalized yet")
    ap.add_argument("--no-verify-projection", action="store_true",
                    help="skip item 3's real projection (fast, DIAGNOSTIC ONLY - "
                         "every round is then marked projection_verified=false and "
                         "no round is training-eligible)")
    args = ap.parse_args()

    capture_dir = Path(args.dir)
    dbs = sorted(capture_dir.glob("*.db"))
    if not dbs:
        print(f"[reindex] no captures under {capture_dir}", file=sys.stderr)
        return 1

    print(f"[reindex] rebuilding from {len(dbs)} capture file(s)")
    rows, in_flight = rebuild(
        dbs,
        include_in_flight=args.include_in_flight,
        verify_projection=not args.no_verify_projection,
    )
    if in_flight:
        print(f"[reindex] {in_flight} round(s) still in flight (not FINALIZED) - "
              "left out; the next run picks them up")

    records = [evaluate_round_from_row(r) for r in rows]
    report_groups(rows, records)
    report_attribution(dbs)
    counts = summarize(records)

    if args.dry_run:
        print("[reindex] --dry-run: index not written")
        return 0

    index = capture_dir / "INDEX.jsonl"
    if index.exists():
        backup = index.with_suffix(".jsonl.bak")
        backup.write_text(index.read_text())
        print(f"[reindex] previous index preserved at {backup.name}")
    tmp = index.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(index)
    print(f"[reindex] wrote {len(rows)} records to {index}")
    return 0


def report_groups(rows: list[dict], records: list) -> None:
    """Item 7: LEGACY_RECORDER and POST_A0_1_RECORDER reported SEPARATELY.

    Pooling them would let a pre-fix round's verdict - computed by code that
    guessed at parse-failure attribution and never checked rule text - stand
    beside a post-fix one as if the two carried the same evidence.
    """
    by_generation: dict[str, list] = {}
    for row, rec in zip(rows, records):
        by_generation.setdefault(row.get("recorder_generation") or "UNKNOWN", []).append(rec)

    for generation in sorted(by_generation):
        group = by_generation[generation]
        c = summarize(group)
        print()
        print(f"  {generation}   ({len(group)} finalized round(s))")
        print(f"    finalized rounds           {c['captured']}")
        print(f"    label valid                {c['label_valid']}")
        print(f"    rule-text VERIFIED_TRUE    {c['rule_text_verified']}")
        print(f"    data-quality valid         {c['data_training_grade']}")
        print(f"    projection preconditions   {c['projection_preconditions_valid']}")
        print(f"    ACTUAL projection valid    {c['projection_valid']}")
        print(f"    FINAL training eligible    {c['training_eligible']}")
        if c["rule_text_by_status"]:
            print("    rule-text status breakdown:")
            for status, n in c["rule_text_by_status"].items():
                print(f"      {status:<38} {n:>5}")
        if c["disqualifiers_by_reason"]:
            print("    exclusions by category:")
            for reason, n in c["disqualifiers_by_reason"].items():
                print(f"      {reason:<38} {n:>5}")


def report_attribution(dbs: list[Path]) -> None:
    """Item 7: parse failures broken down by HOW they were attributed."""
    totals: dict[str, int] = {}
    complete, incomplete = [], []
    for db in dbs:
        store = RawEventStore(str(db))
        try:
            summary = attribution_summary(store)
        finally:
            store.close()
        for status, n in summary.by_status().items():
            if n:
                totals[status] = totals.get(status, 0) + n
        (complete if summary.is_complete else incomplete).append(db.name)

    print()
    print("  PARSE-FAILURE ATTRIBUTION")
    if not totals:
        print("    (no parse failures in any capture)")
    for status, n in sorted(totals.items()):
        print(f"    {status:<40} {n:>5}")
    print(f"    captures with COMPLETE attribution     {len(complete):>5}")
    print(f"    captures on session-wide fallback      {len(incomplete):>5}")
    for name in incomplete:
        print(f"      ! {name}")


def evaluate_round_from_row(row: dict):
    """Rebuild a `RoundEligibility` from an index row, so the summary uses
    exactly the values that were written rather than recomputing them."""
    from xamarinbot.eligibility import Disqualifier, RoundEligibility

    return RoundEligibility(
        round_id=row["round_id"],
        label_valid=bool(row.get("label_valid")),
        data_valid=bool(row.get("data_training_grade")),
        projection_valid=bool(row.get("projection_valid")),
        disqualifiers=tuple(Disqualifier(d) for d in row.get("disqualifiers", [])),
        projection_preconditions_valid=bool(row.get("projection_preconditions_valid")),
        projection_error=row.get("projection_error"),
        projection_verified=bool(row.get("projection_verified")),
        rule_text_status=row.get("rule_text_status"),
        recorder_generation=row.get("recorder_generation"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
