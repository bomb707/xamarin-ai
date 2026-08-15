#!/usr/bin/env python3
"""Phase 12C: capture N consecutive real BTC 5-minute rounds.

RECORDER ONLY. This script authenticates nothing, holds no private key, and
never sends a maker order, taker order, cancel or replacement to Polymarket
(item 14). It subscribes to public data, writes an immutable raw-event log,
reconstructs the settlement label independently, and prints the integrity
report.

    python scripts/run_real_recorder.py --rounds 3 --db captures/run1.db

Each round takes 5 minutes plus a pre-round lead (default 7 minutes) and a
post-round tail, so `--rounds 3` runs for roughly 25 minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.realtime.lifecycle import LifecycleConfig  # noqa: E402
from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402
from xamarinbot.realtime.recorder import RecorderConfig  # noqa: E402
from xamarinbot.realtime.report import format_capture_report  # noqa: E402
from xamarinbot.realtime.service import RealRecorderService, ServiceConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3, help="consecutive rounds to capture")
    ap.add_argument("--db", default="captures/real_capture.db", help="SQLite path for the raw log")
    ap.add_argument("--pre-round-lead", type=float, default=420.0,
                    help="seconds of reference history to record before each round opens")
    ap.add_argument("--post-round-tail", type=float, default=90.0,
                    help="seconds to keep recording after each round closes")
    ap.add_argument("--integrity-interval", type=float, default=60.0,
                    help="seconds between in-memory-vs-REST book verifications")
    ap.add_argument("--queue-size", type=int, default=50_000)
    ap.add_argument("--resolution-sweep", type=float, default=900.0,
                    help="seconds to keep polling for the venue's resolution after all rounds "
                         "finalize (measured settlement latency is ~3-8 min past round close; "
                         "0 disables the sweep)")
    ap.add_argument("--report", default=None, help="write the capture report here as well as stdout")
    ap.add_argument("--metrics-json", default=None, help="write session metrics as JSON here")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ServiceConfig(
        n_rounds=args.rounds,
        lifecycle=LifecycleConfig(
            pre_round_lead_s=args.pre_round_lead,
            post_round_tail_s=args.post_round_tail,
        ),
        recorder=RecorderConfig(queue_maxsize=args.queue_size),
        integrity_check_interval_s=args.integrity_interval,
        resolution_sweep_s=args.resolution_sweep,
    )

    store = RawEventStore(str(db_path))
    service = RealRecorderService(store, cfg)
    print(f"[recorder] session {service.session_id}")
    print(f"[recorder] writing to {db_path}")
    print(f"[recorder] capturing {args.rounds} round(s); NO ORDERS WILL BE PLACED")

    started = time.time()
    captures = service.run()
    elapsed = time.time() - started

    report = format_capture_report(captures, service.metrics, store)
    print()
    print(report)
    print(f"\n[recorder] wall-clock {elapsed / 60.0:.1f} min, "
          f"{store.count()} raw events in {db_path}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)
        print(f"[recorder] report written to {args.report}")
    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(json.dumps(service.metrics.as_dict(), indent=1))
        print(f"[recorder] metrics written to {args.metrics_json}")

    store.close()
    # Exit non-zero when the capture is not usable for training, so a CI or
    # cron invocation surfaces it rather than silently banking a bad dataset.
    return 0 if service.metrics.is_training_grade() else 2


if __name__ == "__main__":
    raise SystemExit(main())
