#!/usr/bin/env python3
"""Continuous real BTC five-minute capture (post-12C.2, profitability phase).

The only question that matters now is

    E[Pi_round | unseen real BTC5m rounds] > 0 ?

and it cannot be asked until enough CHRONOLOGICALLY INDEPENDENT real rounds
exist. One round is a verification sample, not a dataset: fitting `q` on it
would be fitting to a single five-minute path.

This driver accumulates rounds by running `RealRecorderService` in
back-to-back batches, each into its own database, resolving labels after each
batch and appending one line per round to a rolling index so progress toward
Gate A is visible without opening any capture.

RECORDER ONLY. No private key, no authenticated session, no order, cancel or
replacement is ever sent - the same guarantee `scripts/run_real_recorder.py`
carries, enforced by `tests/test_import_boundaries.py`.

    # accumulate until stopped (nohup/systemd friendly)
    python scripts/run_continuous_capture.py

    # bounded run
    python scripts/run_continuous_capture.py --hours 12

    # progress toward Gate A, without capturing anything
    python scripts/run_continuous_capture.py --status

Why batches rather than one endless session
-------------------------------------------
The recorder subscribes every round's tokens up front so each round gets its
full PRE_ROUND lookback (item 7). Subscribing hundreds of rounds at once
would balloon both the live event rate and a single database. Batching keeps
each file to roughly `--rounds x 170MB`, bounds the blast radius of any single
failure, and gives the resolution sweep a natural place to run.

The cost is the per-batch PRE_ROUND lead: the first round of a batch must
open at least `pre_round_lead_s` (420s) from now, so a batch spends ~7 idle
minutes before its first round. Larger batches amortize that - 8 rounds is
~40 minutes of rounds per ~60 minutes of wall clock.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xamarinbot.realtime.lifecycle import LifecycleConfig  # noqa: E402
from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402
from xamarinbot.realtime.service import RealRecorderService, ServiceConfig  # noqa: E402

CAPTURE_DIR = _REPO_ROOT / "captures" / "continuous"
INDEX = CAPTURE_DIR / "INDEX.jsonl"

_stop = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    print(f"\n[continuous] signal {signum} received - finishing the current batch, "
          "then stopping. Send again to abort immediately.", flush=True)
    signal.signal(signum, signal.SIG_DFL)


def utc(ts: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts))


def append_index(db: Path, captures) -> int:
    """One JSONL line per finalized round. This is the progress ledger; it is
    small, append-only, and readable without touching a capture."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with INDEX.open("a") as fh:
        for cap in captures:
            m = cap.metadata
            rec = cap.reconstruction
            fh.write(json.dumps({
                "round_id": m.round_id,
                "condition_id": m.condition_id,
                "start_ts": m.start_ts,
                "end_ts": m.end_ts,
                "db": str(db.relative_to(_REPO_ROOT)),
                "state": cap.lifecycle.state.value,
                "settlement_kind": m.settlement_kind,
                "twap_window_s": m.twap_window_s,
                "min_order_size": m.min_order_size,
                "tick_size": m.tick_size,
                "reported_outcome": (
                    cap.reported_outcome.value if cap.reported_outcome else None
                ),
                "reconstructed_outcome": (
                    rec.declared.outcome.value
                    if rec is not None and rec.declared.outcome else None
                ),
                "label_status": rec.status.value if rec is not None else None,
                "declared_agrees": rec.declared_agrees if rec is not None else None,
                "notes": cap.notes,
                "indexed_at": time.time(),
            }) + "\n")
            n += 1
    return n


def read_index() -> list[dict]:
    if not INDEX.exists():
        return []
    return [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]


def print_status() -> int:
    """Progress toward Gate A, computed only from the index."""
    rows = read_index()
    print("=" * 78)
    print("CONTINUOUS CAPTURE STATUS")
    print("=" * 78)
    if not rows:
        print("  no rounds captured yet")
        print(f"  index: {INDEX} (absent)")
        return 0

    labelled = [r for r in rows if r["reconstructed_outcome"]]
    confirmed = [r for r in rows if r.get("label_status") == "CONFIRMED"]
    ambiguous = [r for r in rows if r.get("label_status") == "LABEL_AMBIGUOUS"]
    unresolved = [r for r in rows if r.get("label_status") in (None, "UNRESOLVED")]
    agreed = [r for r in rows if r.get("declared_agrees") is True]
    comparable = [r for r in rows if r.get("declared_agrees") is not None]

    span_start = min(r["start_ts"] for r in rows)
    span_end = max(r["end_ts"] for r in rows)
    dbs = sorted({r["db"] for r in rows})
    disk = sum((_REPO_ROOT / d).stat().st_size for d in dbs if (_REPO_ROOT / d).exists())

    print(f"  rounds captured        {len(rows)}")
    print(f"  labelled               {len(labelled)}")
    print(f"  CONFIRMED (trainable)  {len(confirmed)}")
    print(f"  LABEL_AMBIGUOUS        {len(ambiguous)}  (excluded from training)")
    print(f"  unresolved             {len(unresolved)}")
    if comparable:
        print(f"  label agreement        {len(agreed)}/{len(comparable)} "
              f"({len(agreed) / len(comparable):.1%})")
    print(f"  chronological span     {utc(span_start)} -> {utc(span_end)} "
          f"({(span_end - span_start) / 3600:.1f}h)")
    print(f"  capture files          {len(dbs)}  ({disk / 1e9:.1f} GB)")
    print(f"  index                  {INDEX.relative_to(_REPO_ROOT)}")

    # Rounds are 5 minutes apart, so chronological independence is a property
    # of the market, not of the sampling - but a train/calibrate/test split
    # still needs enough rounds in each slice to say anything.
    print()
    print("  GATE A READINESS (chronological TRAIN -> CALIBRATE -> FREEZE -> TEST)")
    for target in (100, 200, 500):
        have = len(confirmed)
        bar = "#" * min(30, int(30 * have / target))
        print(f"    {target:>4} trainable rounds  [{bar:<30}] {have}/{target}"
              f"{'  READY' if have >= target else ''}")
    print()
    print("  No q model may be fitted from this data until the approval gate.")
    print("=" * 78)
    return 0


def run_batch(rounds: int, sweep_s: float, log) -> tuple[Path | None, list]:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    db = CAPTURE_DIR / f"btc5m_{utc()}.db"
    cfg = ServiceConfig(
        n_rounds=rounds,
        lifecycle=LifecycleConfig(),
        resolution_sweep_s=sweep_s,
    )
    store = RawEventStore(str(db))
    service = RealRecorderService(store, cfg, log=log)
    try:
        captures = service.run()
    finally:
        store.close()
    return db, captures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=8,
                    help="rounds per batch (default 8; ~%s)" % "1.4GB, ~60min wall clock")
    ap.add_argument("--hours", type=float, default=None,
                    help="stop after this many hours (default: run until interrupted)")
    ap.add_argument("--sweep", type=float, default=720.0,
                    help="seconds to poll for venue resolutions after each batch")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--quiet", action="store_true", help="only print per-batch summaries")
    args = ap.parse_args()

    if args.status:
        return print_status()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    deadline = time.time() + args.hours * 3600 if args.hours else None
    log = (lambda *a: None) if args.quiet else print

    print("=" * 78)
    print("CONTINUOUS REAL BTC 5-MINUTE CAPTURE")
    print("  RECORDER ONLY - no orders, no credentials, no private key.")
    print(f"  batch size   {args.rounds} rounds")
    print(f"  output       {CAPTURE_DIR.relative_to(_REPO_ROOT)}/btc5m_<utc>.db")
    print(f"  index        {INDEX.relative_to(_REPO_ROOT)}")
    print(f"  until        {'interrupted' if deadline is None else utc(deadline)}")
    print("=" * 78, flush=True)

    batch = 0
    total_rounds = 0
    while not _stop and (deadline is None or time.time() < deadline):
        batch += 1
        started = time.time()
        print(f"\n[continuous] batch {batch} starting at {utc()}", flush=True)
        try:
            db, captures = run_batch(args.rounds, args.sweep, log)
        except Exception:
            # A batch failure must not end the accumulation. Record it and
            # move on - an hour of missed capture is recoverable, a stopped
            # collector overnight is not.
            print(f"[continuous] batch {batch} FAILED:\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
            time.sleep(60)
            continue

        n = append_index(db, captures)
        total_rounds += n
        size_gb = db.stat().st_size / 1e9 if db.exists() else 0.0
        labelled = sum(
            1 for c in captures
            if c.reconstruction is not None and c.reconstruction.declared.outcome
        )
        print(
            f"[continuous] batch {batch} done in {(time.time() - started) / 60:.1f} min: "
            f"{n} rounds ({labelled} labelled), {size_gb:.2f} GB -> {db.name}; "
            f"{total_rounds} rounds accumulated",
            flush=True,
        )

    print(f"\n[continuous] stopped after {batch} batch(es), "
          f"{total_rounds} rounds accumulated this session.", flush=True)
    print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
