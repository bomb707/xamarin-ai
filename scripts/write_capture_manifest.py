#!/usr/bin/env python3
"""Write the canonical capture verification manifest (Phase 12C.1 item 17).

The capture databases are gitignored - a three-round capture is ~500MB - so
the manifest IS the committed evidence that a given capture existed, what it
contained, and whether it passed its data-quality gate. Without it, "we have
a training-grade capture" is an unverifiable claim.

    python scripts/write_capture_manifest.py
    python scripts/write_capture_manifest.py --db captures/phase12c_verify.db

Read-only with respect to the captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402

CAPTURES = _REPO_ROOT / "captures"
MANIFEST = CAPTURES / "VERIFICATION_MANIFEST.json"


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def describe_capture(db: Path, *, hash_db: bool) -> dict:
    store = RawEventStore(str(db))
    rounds = []
    for rid in store.round_ids():
        row = store.get_round(rid) or {}
        rounds.append({
            "round_id": rid,
            "condition_id": row.get("condition_id"),
            "start_ts_ns": row.get("start_ts_ns"),
            "end_ts_ns": row.get("end_ts_ns"),
            "min_order_size": row.get("min_order_size"),
            "tick_size": row.get("tick_size"),
            "settlement_kind": row.get("settlement_kind"),
            "twap_window_s": row.get("twap_window_s"),
            "state": row.get("state"),
        })
    results = [
        {
            "round_id": r["round_id"],
            "reported_outcome": r["reported_outcome"],
            "reconstructed_outcome": r["reconstructed_outcome"],
            "reconstruction_basis": r["reconstruction_basis"],
            "label_agreement": r["label_agreement"],
            "is_training_grade": r["is_training_grade"],
            "notes": r["notes"],
        }
        for r in store.round_results()
    ]
    counts = store.counts_by_topic()
    total = store.count()
    store.close()

    metrics_path = db.with_name(db.stem + "_metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    return {
        "db": str(db.relative_to(_REPO_ROOT)),
        "db_bytes": db.stat().st_size,
        "db_sha256": sha256(db) if hash_db else None,
        "db_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(db.stat().st_mtime)),
        "raw_event_count": total,
        "event_counts_by_topic": counts,
        "rounds": rounds,
        "round_results": results,
        "recorder_metrics": metrics,
        "data_quality_verdict": _verdict(metrics, results, rounds),
        "known_gaps": _known_gaps(metrics, counts, rounds, results),
    }


def _verdict(metrics: dict | None, results: list, rounds: list) -> str:
    """The recorder's own health counters are necessary but NOT sufficient.

    A capture is only training-grade if it is BOTH clean by the recorder's
    accounting AND every round's settlement label was actually reconstructed.
    Keeping these separate is what let the pre-teardown-fix `sample` capture
    look healthy while two of its three rounds had no usable label.
    """
    if metrics is None:
        return "UNKNOWN - no metrics file alongside the capture"
    if not metrics.get("is_training_grade"):
        return "NOT_TRAINING_GRADE"
    labelled = {r["round_id"] for r in results if r.get("reconstructed_outcome")}
    if len(labelled) < len(rounds):
        return "NOT_TRAINING_GRADE - unlabellable rounds"
    return "TRAINING_GRADE"


def _known_gaps(metrics: dict | None, counts: dict, rounds: list, results: list) -> list[str]:
    """Everything about this capture a reader must know before trusting it."""
    gaps: list[str] = []
    if metrics is None:
        return ["no recorder metrics recorded alongside this capture"]
    for key, msg in (
        ("dropped_events", "events were dropped by the bounded ingestion queue"),
        ("parse_failures", "wire frames failed to parse"),
        ("book_integrity_mismatches", "in-memory book disagreed with a REST resnapshot"),
    ):
        if metrics.get(key):
            gaps.append(f"{metrics[key]} {msg}")
    if metrics.get("reconnect_count"):
        gaps.append(
            f"{metrics['reconnect_count']} stream reconnect(s); each is a short "
            "data gap even though the watchdog recovered"
        )
    if metrics.get("duplicate_events"):
        gaps.append(
            f"{metrics['duplicate_events']} duplicate events suppressed (cause not "
            "investigated)"
        )
    lat = metrics.get("source_to_recv") or {}
    if (lat.get("p99_ms") or 0) > 1000:
        gaps.append(
            f"source->recv p99 {lat['p99_ms']:.0f}ms / max {lat.get('max_ms', 0):.0f}ms - "
            "dominated by the RTDS reference feeds; bears on canonical_horizon_s margin"
        )
    if not any(k.startswith("rtds_") for k in counts):
        gaps.append("no RTDS reference data in this capture - labels cannot be reconstructed")

    # The decisive data-quality question is not "did the recorder report
    # itself healthy" but "can each round's label actually be reconstructed".
    # The pre-teardown-fix `sample` capture reported dropped=0 /
    # parse_failures=0 and would otherwise read as TRAINING_GRADE, while a
    # silent RTDS stall had left two of its three rounds with no reference
    # data covering their settlement boundary. Metrics alone cannot see that;
    # the reconstruction outcome can.
    unlabelled = [
        r["round_id"] for r in results if not r.get("reconstructed_outcome")
    ]
    missing_results = [
        r["round_id"] for r in rounds
        if r["round_id"] not in {x["round_id"] for x in results}
    ]
    if unlabelled:
        gaps.append(
            f"{len(unlabelled)}/{len(rounds)} round(s) could NOT have their settlement "
            f"label reconstructed ({', '.join(unlabelled)}) - reference data does not "
            "cover the round boundary; NOT usable for training regardless of the "
            "recorder's own health counters"
        )
    if missing_results:
        gaps.append(
            f"{len(missing_results)} round(s) have no result row at all "
            f"({', '.join(missing_results)})"
        )
    disagreed = [
        r["round_id"] for r in results
        if r.get("label_agreement") == 0
    ]
    if disagreed:
        gaps.append(
            f"{len(disagreed)} round(s) reconstructed a DIFFERENT outcome from the "
            f"venue ({', '.join(disagreed)}) - LABEL_AMBIGUOUS, exclude from training"
        )
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", action="append", default=None,
                    help="capture db (repeatable); default: every *.db under captures/ incl. archive/")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip SHA-256 (hashing a 500MB capture takes a few seconds)")
    ap.add_argument("--out", default=str(MANIFEST))
    args = ap.parse_args()

    dbs = [Path(d) for d in args.db] if args.db else sorted(CAPTURES.rglob("*.db"))
    dbs = [d for d in dbs if d.exists()]
    if not dbs:
        print("[manifest] no capture databases found", file=sys.stderr)
        return 1

    manifest = {
        "schema": "xamarinbot.capture_manifest/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recorder_commit": git_commit(),
        "note": (
            "Capture databases are gitignored (~500MB each). This manifest is the "
            "committed evidence of what was captured and whether it passed its "
            "data-quality gate. Regenerate with scripts/write_capture_manifest.py."
        ),
        "captures": [describe_capture(d, hash_db=not args.no_hash) for d in dbs],
        "archived": (
            "captures/archive/phase12c_pre_teardown_fix_* are preserved as historical "
            "evidence, NOT as usable datasets: `sample` lost all reference data to a "
            "silent RTDS stall, and `final` reported parse_failures=49 / "
            "training_grade=false caused entirely by its own end-of-round teardown "
            "(both fixed in Phase 12C). The canonical verified capture is "
            "captures/phase12c_verify.db."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[manifest] wrote {out}")
    for cap in manifest["captures"]:
        print(f"  {cap['db']:<48} {cap['raw_event_count']:>8} events  "
              f"{cap['data_quality_verdict']}")
        for gap in cap["known_gaps"]:
            print(f"      ! {gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
