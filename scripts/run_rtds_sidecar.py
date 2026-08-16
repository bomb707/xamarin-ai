#!/usr/bin/env python3
"""DIAGNOSTIC RTDS sidecar - a second, independent connection.

Purpose (Gate A yield recovery, items 6-8): determine whether the ~30-second
RTDS outages that dominate DATA_GAP exclusions are CONNECTION-LOCAL or
SERVER/SOURCE-WIDE. A single connection cannot answer that: when it goes
silent there is nothing to compare it against.

This process opens its own RTDS socket to the same four required BTC topics,
with the same symbol validation and the same parsing, and records every valid
observation plus every gap. It NEVER feeds a strategy decision, NEVER affects
training eligibility, and NEVER places an order.

Started deliberately at a different moment from the production stream, so a
failure at a fixed CONNECTION AGE shows up as two outages at different wall
clocks rather than one simultaneous one.

    python scripts/run_rtds_sidecar.py --hours 4
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xamarinbot.realtime.raw_events import RawEventBuilder  # noqa: E402
from xamarinbot.realtime.rtds import RTDSClient  # noqa: E402

OUT_DIR = _REPO_ROOT / "captures" / "diagnostics"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS obs (
    wire_topic TEXT, symbol TEXT, source_ts_ns INTEGER,
    value REAL, full_value TEXT, recv_ns INTEGER, generation INTEGER
);
CREATE INDEX IF NOT EXISTS idx_obs ON obs(wire_topic, source_ts_ns);
CREATE TABLE IF NOT EXISTS gap (
    wire_topic TEXT, failure_kind TEXT,
    last_data_ns INTEGER, detected_ns INTEGER, recovered_ns INTEGER
);
CREATE TABLE IF NOT EXISTS conn (
    generation INTEGER, event TEXT, at_ns INTEGER
);
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--label", default="sidecar")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    db = OUT_DIR / f"rtds_{args.label}_{stamp}.db"
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()

    builder = RawEventBuilder(session_id=f"{args.label}-{stamp}")

    def on_obs(obs) -> None:
        conn.execute(
            "INSERT INTO obs VALUES (?,?,?,?,?,?,?)",
            (obs.topic, obs.symbol, obs.source_ts_ns, obs.value,
             str(obs.full_accuracy_value), obs.recv_wall_ns,
             builder.reconnect_generation),
        )

    def on_gap(gap) -> None:
        conn.execute("INSERT INTO gap VALUES (?,?,?,?,?)",
                     (gap.wire_topic, gap.failure_kind, gap.last_data_ns,
                      gap.detected_ns, gap.recovered_ns))
        conn.commit()
        print(f"[{args.label}] GAP {gap.wire_topic} {gap.failure_kind} "
              f"{gap.duration_ns/1e9:.1f}s", flush=True)

    def on_reconnect(generation: int) -> None:
        conn.execute("INSERT INTO conn VALUES (?,?,?)",
                     (generation, "reconnect", time.time_ns()))
        conn.commit()
        print(f"[{args.label}] RECONNECT generation={generation}", flush=True)

    client = RTDSClient(
        builder=builder,
        on_raw_event=lambda e: None,     # diagnostics only; no raw persistence
        on_observation=on_obs,
        on_data_gap=on_gap,
        on_reconnect=on_reconnect,
    )
    conn.execute("INSERT INTO conn VALUES (?,?,?)", (0, "start", time.time_ns()))
    conn.commit()

    print(f"[{args.label}] DIAGNOSTIC ONLY - no strategy, no orders, no "
          f"eligibility impact\n[{args.label}] -> {db}", flush=True)
    client.start()
    deadline = time.time() + args.hours * 3600
    try:
        last_commit = time.time()
        while time.time() < deadline:
            time.sleep(5.0)
            if time.time() - last_commit > 30:
                conn.commit()
                last_commit = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
        g = conn.execute("SELECT COUNT(*) FROM gap").fetchone()[0]
        print(f"[{args.label}] stopped: {n} observations, {g} gaps -> {db}",
              flush=True)
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
