"""Append-only, SQLite-backed causal event store (Roadmap Phase 2)."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from xamarinbot.events.types import Event, EventType

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
"""


class EventStore:
    """Durable, append-only event log. Events are never mutated or deleted;
    `sequence` is assigned monotonically by SQLite AUTOINCREMENT and is the
    final deterministic tie-break key (Event.sort_key)."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

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
            "INSERT INTO events (event_type, round_id, recv_ts, source_ts, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_type.value, round_id, recv_ts, source_ts, json.dumps(payload)),
        )
        self._conn.commit()
        sequence = cur.lastrowid
        return Event(
            sequence=sequence,
            event_type=event_type,
            round_id=round_id,
            recv_ts=recv_ts,
            source_ts=source_ts,
            payload=payload,
        )

    def all_events(self, round_id: str | None = None) -> list[Event]:
        """All events, deterministically ordered per Event.sort_key."""
        if round_id is None:
            rows = self._conn.execute("SELECT * FROM events").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE round_id = ?", (round_id,)
            ).fetchall()
        events = [self._row_to_event(row) for row in rows]
        return sorted(events, key=lambda e: e.sort_key)

    def round_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT round_id FROM events ORDER BY round_id"
        ).fetchall()
        return [r[0] for r in rows]

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
