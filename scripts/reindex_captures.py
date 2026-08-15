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
from xamarinbot.realtime.preflight import evaluate_round, label_fields  # noqa: E402
from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402

CAPTURE_DIR = _REPO_ROOT / "captures" / "continuous"
INDEX = CAPTURE_DIR / "INDEX.jsonl"


def rebuild(dbs: list[Path], include_in_flight: bool = False) -> tuple[list[dict], int]:
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
        try:
            for round_id in store.round_ids():
                meta = store.get_round(round_id) or {}
                if not include_in_flight and meta.get("state") != "FINALIZED":
                    in_flight += 1
                    continue
                labels = label_fields(store, round_id)
                elig = evaluate_round(store, round_id)
                rows.append({
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
    args = ap.parse_args()

    capture_dir = Path(args.dir)
    dbs = sorted(capture_dir.glob("*.db"))
    if not dbs:
        print(f"[reindex] no captures under {capture_dir}", file=sys.stderr)
        return 1

    print(f"[reindex] rebuilding from {len(dbs)} capture file(s)")
    rows, in_flight = rebuild(dbs, include_in_flight=args.include_in_flight)
    if in_flight:
        print(f"[reindex] {in_flight} round(s) still in flight (not FINALIZED) - "
              "left out; the next run picks them up")

    records = [evaluate_round_from_row(r) for r in rows]
    counts = summarize(records)
    print(f"  captured                     {counts['captured']}")
    print(f"  label CONFIRMED (valid)      {counts['label_valid']}")
    print(f"  data-quality clean           {counts['data_training_grade']}")
    print(f"  projection valid             {counts['projection_valid']}")
    print(f"  FINAL training eligible      {counts['training_eligible']}")
    if counts["disqualifiers_by_reason"]:
        print("  disqualification reasons by category:")
        for reason, n in counts["disqualifiers_by_reason"].items():
            print(f"    {reason:<40} {n:>5}")

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
    )


if __name__ == "__main__":
    raise SystemExit(main())
