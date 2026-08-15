"""WAL-backed, batched raw-event store (Phase 12C item 6).

Contrast with `events/store.py`'s `EventStore`, which is kept as-is for
deterministic replay of normalized events:

  EventStore                          RawEventStore
  ----------------------------------  ----------------------------------
  one COMMIT per append               one COMMIT per BATCH
  journal_mode=delete (default)       journal_mode=WAL
  timestamps REAL (seconds)           timestamps INTEGER (nanoseconds)
  source/recv collapsed by event_time source/publisher/recv/monotonic all
                                      stored separately
  payload = normalized dict           payload = verbatim wire text, plus
                                      normalized projections alongside

Rows are append-only and are never updated or deleted. `recorder_sequence`
is assigned by the ingesting process (not by SQLite), so the ordering
survives the batching - two events in the same batch keep the order they
were read off the wire.
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Iterable, Sequence

from xamarinbot.realtime.raw_events import RawEvent, Topic

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_events (
    session_id             TEXT    NOT NULL,
    recorder_sequence      INTEGER NOT NULL,
    topic                  TEXT    NOT NULL,
    event_type             TEXT    NOT NULL,
    round_id               TEXT,
    condition_id           TEXT,
    token_id               TEXT,
    source_timestamp_ns    INTEGER,
    publisher_timestamp_ns INTEGER,
    recv_wall_timestamp_ns INTEGER NOT NULL,
    recv_monotonic_ns      INTEGER NOT NULL,
    payload_json           TEXT    NOT NULL,
    normalized_side        TEXT,
    reconnect_generation   INTEGER NOT NULL,
    PRIMARY KEY (session_id, recorder_sequence)
);
CREATE INDEX IF NOT EXISTS idx_raw_round  ON raw_events(round_id);
CREATE INDEX IF NOT EXISTS idx_raw_topic  ON raw_events(round_id, topic);
CREATE INDEX IF NOT EXISTS idx_raw_token  ON raw_events(token_id, source_timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_raw_recv   ON raw_events(recv_wall_timestamp_ns);

-- Round-level metadata and outcome, one row per round, written at
-- DISCOVERED and rewritten (same primary key) at FINALIZED. This is the
-- one deliberately-mutable table: it is a projection of the immutable
-- raw_events log, not a substitute for it.
CREATE TABLE IF NOT EXISTS rounds (
    round_id           TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    condition_id       TEXT,
    question_id        TEXT,
    slug               TEXT,
    question           TEXT,
    description        TEXT,
    resolution_source  TEXT,
    start_ts_ns        INTEGER,
    end_ts_ns          INTEGER,
    up_token_id        TEXT,
    down_token_id      TEXT,
    tick_size          REAL,
    min_order_size     REAL,
    fee_config_json    TEXT,
    taker_delay_ms     REAL,
    twap_window_s      INTEGER,
    settlement_kind    TEXT,
    state              TEXT,
    raw_metadata_json  TEXT
);

-- Round-level capture outcome: reconstructed vs reported label, metrics,
-- and the training-grade verdict for that round's interval.
CREATE TABLE IF NOT EXISTS round_results (
    round_id                  TEXT PRIMARY KEY,
    reported_outcome          TEXT,
    reconstructed_outcome     TEXT,
    reconstruction_basis      TEXT,
    label_agreement           INTEGER,
    start_reference_value     REAL,
    end_reference_value       REAL,
    start_reference_ts_ns     INTEGER,
    end_reference_ts_ns       INTEGER,
    metrics_json              TEXT,
    is_training_grade         INTEGER,
    notes                     TEXT
);
"""

_INSERT = """
INSERT OR IGNORE INTO raw_events (
    session_id, recorder_sequence, topic, event_type, round_id, condition_id,
    token_id, source_timestamp_ns, publisher_timestamp_ns,
    recv_wall_timestamp_ns, recv_monotonic_ns, payload_json,
    normalized_side, reconnect_generation
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_COLUMNS = (
    "session_id, recorder_sequence, topic, event_type, round_id, condition_id, "
    "token_id, source_timestamp_ns, publisher_timestamp_ns, "
    "recv_wall_timestamp_ns, recv_monotonic_ns, payload_json, "
    "normalized_side, reconnect_generation"
)


def _row_to_event(row: Sequence) -> RawEvent:
    return RawEvent(
        session_id=row[0],
        recorder_sequence=row[1],
        topic=Topic(row[2]),
        event_type=row[3],
        round_id=row[4],
        condition_id=row[5],
        token_id=row[6],
        source_timestamp_ns=row[7],
        publisher_timestamp_ns=row[8],
        recv_wall_timestamp_ns=row[9],
        recv_monotonic_ns=row[10],
        payload_json=row[11],
        normalized_side=row[12],
        reconnect_generation=row[13],
    )


class RawEventStore:
    """Append-only raw-event log.

    `write_batch` is the only ingestion path, and it is what makes the
    architecture work: one transaction amortizes the fsync across the whole
    batch, so the writer thread can keep up with a feed the reader thread
    must never be blocked by.
    """

    def __init__(self, db_path: str = ":memory:", *, synchronous: str = "NORMAL"):
        self.db_path = db_path
        # check_same_thread=False: the writer thread owns the connection but
        # tests and the reporting layer read from the main thread. All
        # access is serialized through `_lock`.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL lets readers proceed during a write transaction, which is what
        # allows the periodic integrity check to query the log without
        # stalling ingestion. It is unavailable for :memory: databases,
        # where the PRAGMA is simply a no-op returning "memory".
        self._conn.execute(f"PRAGMA journal_mode={'WAL' if db_path != ':memory:' else 'MEMORY'}")
        # NORMAL (not FULL) is the deliberate durability trade: it survives
        # process crash, only risking the last transactions on OS/power
        # failure. FULL would put an fsync per batch in the writer's path
        # for a recorder whose data can simply be recaptured.
        self._conn.execute(f"PRAGMA synchronous={synchronous}")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- raw

    def write_batch(self, events: Sequence[RawEvent]) -> int:
        """Persists a whole batch in ONE transaction. Returns the number of
        rows actually inserted - lower than `len(events)` if the batch
        contained a (session_id, recorder_sequence) already present, which
        can only happen on a replayed batch after a crash."""
        if not events:
            return 0
        rows = [
            (
                e.session_id, e.recorder_sequence, e.topic.value, e.event_type,
                e.round_id, e.condition_id, e.token_id, e.source_timestamp_ns,
                e.publisher_timestamp_ns, e.recv_wall_timestamp_ns,
                e.recv_monotonic_ns, e.payload_json, e.normalized_side,
                e.reconnect_generation,
            )
            for e in events
        ]
        with self._lock:
            cur = self._conn.executemany(_INSERT, rows)
            self._conn.commit()
            return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(rows)

    def events(
        self,
        round_id: str | None = None,
        topics: Iterable[Topic] | None = None,
        token_id: str | None = None,
    ) -> list[RawEvent]:
        """Raw events in recorder order (session, recorder_sequence) - the
        order they were read off the wire, which is the only ordering the
        raw layer claims. Causal/normalized ordering is a downstream
        concern, deliberately not applied here."""
        clauses, params = [], []
        if round_id is not None:
            clauses.append("round_id = ?")
            params.append(round_id)
        if token_id is not None:
            clauses.append("token_id = ?")
            params.append(token_id)
        if topics is not None:
            topics = list(topics)
            clauses.append(f"topic IN ({','.join('?' * len(topics))})")
            params.extend(t.value for t in topics)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT {_COLUMNS} FROM raw_events{where} ORDER BY session_id, recorder_sequence"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]

    def count(self, round_id: str | None = None) -> int:
        with self._lock:
            if round_id is None:
                return self._conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
            return self._conn.execute(
                "SELECT COUNT(*) FROM raw_events WHERE round_id = ?", (round_id,)
            ).fetchone()[0]

    def counts_by_topic(self, round_id: str | None = None) -> dict[str, int]:
        sql = "SELECT topic, event_type, COUNT(*) FROM raw_events"
        params: list = []
        if round_id is not None:
            sql += " WHERE round_id = ?"
            params.append(round_id)
        sql += " GROUP BY topic, event_type"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return {f"{t}:{et}": n for t, et, n in rows}

    def round_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT round_id FROM raw_events WHERE round_id IS NOT NULL ORDER BY round_id"
            ).fetchall()
        return [r[0] for r in rows]

    # ----------------------------------------------------------- metadata

    def upsert_round(self, row: dict) -> None:
        keys = [
            "round_id", "session_id", "condition_id", "question_id", "slug",
            "question", "description", "resolution_source", "start_ts_ns",
            "end_ts_ns", "up_token_id", "down_token_id", "tick_size",
            "min_order_size", "fee_config_json", "taker_delay_ms",
            "twap_window_s", "settlement_kind", "state", "raw_metadata_json",
        ]
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO rounds ({','.join(keys)}) "
                f"VALUES ({','.join('?' * len(keys))})",
                [row.get(k) for k in keys],
            )
            self._conn.commit()

    def get_round(self, round_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM rounds WHERE round_id = ?", (round_id,))
            names = [d[0] for d in cur.description]
            row = cur.fetchone()
        return dict(zip(names, row)) if row else None

    def upsert_round_result(self, row: dict) -> None:
        keys = [
            "round_id", "reported_outcome", "reconstructed_outcome",
            "reconstruction_basis", "label_agreement", "start_reference_value",
            "end_reference_value", "start_reference_ts_ns", "end_reference_ts_ns",
            "metrics_json", "is_training_grade", "notes",
        ]
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO round_results ({','.join(keys)}) "
                f"VALUES ({','.join('?' * len(keys))})",
                [row.get(k) for k in keys],
            )
            self._conn.commit()

    def round_results(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM round_results ORDER BY round_id")
            names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(names, r)) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
