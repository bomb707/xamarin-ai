"""Append-only, SQLite-backed causal event store (Roadmap Phase 2).

Phase 12C.1 item 2 added `provenance`: a store now records whether the
observations inside it were fabricated, replayed from a real capture, or
observed live. It defaults to `SYNTHETIC_TEST` so an unlabelled store is
treated as fabricated until something proves otherwise - forgetting to set
provenance downgrades a run to "test only" rather than silently promoting
fabricated data to production evidence.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Sequence

from xamarinbot.events.types import Event, EventType, causal_sort
from xamarinbot.provenance import DataProvenance

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    round_id TEXT NOT NULL,
    recv_ts REAL NOT NULL,
    source_ts REAL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_round ON events(round_id);

-- Store-level metadata. Currently just provenance, kept in a table rather
-- than only in memory so a store reopened from disk still knows what it
-- holds - a projected real capture must not become "unlabelled" (and
-- therefore refused) simply by being closed and reopened.
CREATE TABLE IF NOT EXISTS store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_INSERT = (
    "INSERT INTO events (event_type, round_id, recv_ts, source_ts, payload) "
    "VALUES (?, ?, ?, ?, ?)"
)


class EventStore:
    """Durable, append-only event log. Events are never mutated or deleted;
    `sequence` is assigned monotonically by SQLite AUTOINCREMENT and is the
    final deterministic tie-break key (Event.sort_key)."""

    def __init__(
        self,
        db_path: str = ":memory:",
        provenance: DataProvenance | None = None,
    ):
        """`provenance` defaults to whatever an existing store already
        recorded, then to `SYNTHETIC_TEST` for a fresh one (fail closed).
        Passing it explicitly on an existing store with a DIFFERENT
        provenance is an error rather than a silent relabel - a store does
        not become real because a caller said so."""
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        stored = self._read_meta("provenance")
        if stored is None:
            self.provenance = provenance or DataProvenance.SYNTHETIC_TEST
            self._write_meta("provenance", self.provenance.value)
        else:
            existing = DataProvenance(stored)
            if provenance is not None and provenance is not existing:
                raise ValueError(
                    f"{db_path} already holds {existing.value} data; refusing to "
                    f"relabel it as {provenance.value}. Project into a new store "
                    "instead of relabelling an existing one."
                )
            self.provenance = existing

    # ----------------------------------------------------------- metadata

    def _read_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM store_meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _write_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    @property
    def is_real(self) -> bool:
        return self.provenance.is_real

    # ------------------------------------------------------------- writes

    def append(
        self,
        event_type: EventType,
        round_id: str,
        recv_ts: float,
        source_ts: float | None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        payload = payload or {}
        cur = self._conn.execute(
            _INSERT,
            (event_type.value, round_id, recv_ts, source_ts, json.dumps(payload)),
        )
        self._conn.commit()
        return Event(
            sequence=cur.lastrowid,
            event_type=event_type,
            round_id=round_id,
            recv_ts=recv_ts,
            source_ts=source_ts,
            payload=payload,
        )

    def append_many(
        self,
        rows: Iterable[tuple[EventType, str, float, float | None, dict[str, Any]]],
    ) -> int:
        """Append a batch in ONE transaction. Returns the number inserted.

        Phase 12C.1 item 8: projecting a single real captured round produces
        roughly 180,000 normalized events, and `append()` commits per call -
        an fsync per event. This is the bulk path the projection uses; the
        per-event `append()` stays for interactive/incremental writers.

        Ordering within the batch is preserved by SQLite's AUTOINCREMENT, so
        `sequence` remains a valid final tie-break key.
        """
        payload_rows: Sequence[tuple] = [
            (et.value, round_id, recv_ts, source_ts, json.dumps(payload or {}))
            for et, round_id, recv_ts, source_ts, payload in rows
        ]
        if not payload_rows:
            return 0
        self._conn.executemany(_INSERT, payload_rows)
        self._conn.commit()
        return len(payload_rows)

    # -------------------------------------------------------------- reads

    def all_events(self, round_id: str | None = None) -> list[Event]:
        """All events, deterministically ordered per `causal_sort` -
        `Event.sort_key` plus Phase 12C item 13's per-order_id guarantee
        that no FILL/CANCEL/ORDER_STATUS precedes its own ORDER_SUBMIT."""
        if round_id is None:
            rows = self._conn.execute(
                "SELECT sequence, event_type, round_id, recv_ts, source_ts, payload FROM events"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT sequence, event_type, round_id, recv_ts, source_ts, payload "
                "FROM events WHERE round_id = ?",
                (round_id,),
            ).fetchall()
        return causal_sort([self._row_to_event(row) for row in rows])

    def round_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT round_id FROM events ORDER BY round_id"
        ).fetchall()
        return [r[0] for r in rows]

    def count(self, round_id: str | None = None) -> int:
        if round_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE round_id = ?", (round_id,)
        ).fetchone()[0]

    def counts_by_type(self, round_id: str | None = None) -> dict[str, int]:
        sql = "SELECT event_type, COUNT(*) FROM events"
        params: list = []
        if round_id is not None:
            sql += " WHERE round_id = ?"
            params.append(round_id)
        sql += " GROUP BY event_type"
        return {t: n for t, n in self._conn.execute(sql, params).fetchall()}

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_event(row: tuple) -> Event:
        sequence, event_type, round_id, recv_ts, source_ts, payload = row
        return Event(
            sequence=sequence,
            event_type=EventType(event_type),
            round_id=round_id,
            recv_ts=recv_ts,
            source_ts=source_ts,
            payload=json.loads(payload),
        )
