#!/usr/bin/env python3
"""Fill in venue resolutions and label agreement for an already-captured
Phase 12C session.

BTC 5-minute markets settle roughly 3-8 minutes after the round closes
(measured 2026-08-15), which is long after the recorder has finalized the
round. Rather than idle the recorder through that window - which would eat
into the next round's PRE_ROUND capture - a capture is banked promptly and
its labels are completed afterwards by this script.

Read-only with respect to `raw_events`; it only updates `round_results`.

    python scripts/resolve_capture_labels.py captures/run1.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.realtime.label import Outcome  # noqa: E402
from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402
from xamarinbot.realtime.service import resolve_from_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", help="capture database written by run_real_recorder.py")
    ap.add_argument("--max-wait", type=float, default=900.0,
                    help="seconds to keep polling for outstanding resolutions")
    ap.add_argument("--poll-interval", type=float, default=20.0)
    args = ap.parse_args()

    store = RawEventStore(args.db)
    out = resolve_from_store(store, max_wait_s=args.max_wait, poll_interval_s=args.poll_interval)

    print()
    print("=" * 74)
    print("LABEL AGREEMENT (post-capture resolution)")
    print("=" * 74)
    rows = store.round_results()
    declared_ok = declared_n = 0
    for r in rows:
        agree = r["label_agreement"]
        if agree is not None:
            declared_n += 1
            declared_ok += int(agree)
        print(f"  {r['round_id']}")
        print(f"    reported by venue     {r['reported_outcome']}")
        print(f"    reconstructed         {r['reconstructed_outcome']}  "
              f"({r['reconstruction_basis']})")
        print(f"    P_start / P_end       {r['start_reference_value']} / {r['end_reference_value']}")
        print(f"    agreement             {agree}")
        if r["notes"]:
            print(f"    notes                 {r['notes']}")
    if declared_n:
        print(f"\n  declared-basis agreement: {declared_ok}/{declared_n} "
              f"({declared_ok / declared_n:.0%})")
        print(f"  labels reproducible:      {declared_ok == declared_n}")
    else:
        print("\n  no round could be compared to a venue resolution")
    print(f"\n  resolved this pass: {out['resolved']}, still pending: {out['pending']}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
