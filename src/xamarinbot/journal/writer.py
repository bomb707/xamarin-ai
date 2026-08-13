"""SQLite-backed journal writer/reader for the SS20 schema.

Every record kind is stored as a JSON payload under its dataclass name, kept
generic rather than one hand-written table per entity, so adding the
Phase-4/5/8 record kinds later needs no migration - the schema itself
(journal/schema.py) is still the typed, exact-field-name source of truth.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    round_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_journal_kind_round ON journal(kind, round_id);
"""


class JournalWriter:
    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def write(self, record) -> None:
        if not is_dataclass(record):
            raise TypeError("journal records must be dataclasses (see journal/schema.py)")
        kind = type(record).__name__
        round_id = getattr(record, "round_id")
        self._conn.execute(
            "INSERT INTO journal (kind, round_id, payload) VALUES (?, ?, ?)",
            (kind, round_id, json.dumps(asdict(record), default=str)),
        )
        self._conn.commit()

    def read(self, record_cls, round_id: str | None = None) -> list:
        kind = record_cls.__name__
        if round_id is None:
            rows = self._conn.execute(
                "SELECT payload FROM journal WHERE kind = ? ORDER BY id", (kind,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM journal WHERE kind = ? AND round_id = ? ORDER BY id",
                (kind, round_id),
            ).fetchall()
        return [record_cls(**json.loads(r[0])) for r in rows]

    def round_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT round_id FROM journal ORDER BY round_id").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
