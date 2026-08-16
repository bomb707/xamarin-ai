"""Permanent shadow journal (readiness audit items 12 and 13).

What was missing
----------------
`journal/schema.py` declares `FeatureStateRecord`, `ModelOutputRecord` and
`CandidateActionRecord` with the comment "declared, not yet populated" -
and they genuinely were not: nothing in `ShadowRunner` or the recorder ever
wrote one. `ShadowRoundResult` kept decision records in memory and returned
them; when the process exited they were gone.

For a strategy-research loop that is fatal. The question "why did the bot
trade here, and what did it know?" cannot be answered from raw market data
alone - it needs the feature vector, the probability, the regime, EVERY
candidate that was considered and rejected, and the chosen action.

This module writes all of it, plus the provenance needed to map a decision
back to the exact capture, event range, code SHA and strategy config that
produced it.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, is_dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_journal (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    round_id     TEXT,
    decision_ts  REAL,
    written_at   REAL NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sj_round ON shadow_journal(round_id, kind);
CREATE INDEX IF NOT EXISTS idx_sj_kind  ON shadow_journal(kind);
"""


def _jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value
    return str(obj)


class ShadowJournal:
    """Append-only record of every shadow decision and its provenance."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _write(self, kind: str, round_id: str | None, decision_ts: float | None,
               payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO shadow_journal (kind, round_id, decision_ts, written_at, payload) "
            "VALUES (?,?,?,?,?)",
            (kind, round_id, decision_ts, time.time(), json.dumps(_jsonable(payload))),
        )
        self._conn.commit()

    # ------------------------------------------------------------ writers

    def write_manifest(self, manifest) -> None:
        self._write("strategy_manifest", None, None, manifest.as_dict())

    def write_round_opened(self, shadow, manifest, capture_db: str) -> None:
        """Audit item 13: the linkage row. One losing decision weeks later
        must be traceable to the exact capture, code and config."""
        from xamarinbot.realtime.identity import RecorderIdentity

        identity = RecorderIdentity.capture()
        m = shadow.metadata
        self._write("round_opened", shadow.round_id, None, {
            "round_id": shadow.round_id,
            "condition_id": m.condition_id,
            "up_token_id": m.up_token_id,
            "down_token_id": m.down_token_id,
            "start_ts": m.start_ts,
            "end_ts": m.end_ts,
            "settlement_kind": m.settlement_kind,
            "twap_window_s": m.twap_window_s,
            "tick_size": m.tick_size,
            "min_order_size": m.min_order_size,
            "taker_delay_ms": m.taker_delay_ms,
            # provenance linkage
            "raw_capture_db": capture_db,
            "code_sha": identity.recorder_code_sha,
            "code_dirty": identity.recorder_code_dirty,
            "process_pid": identity.process_pid,
            "recorder_schema_version": identity.recorder_schema_version,
            "strategy_version": manifest.strategy_version,
            "config_hash": manifest.config_hash,
            "model_version": manifest.model_version,
        })

    def write_decision(
        self, shadow, decision_ts: float, t: float, fv, q, regime, decision,
        *, chosen=None, blocked_reason: str | None = None,
        invalid_reason: str | None = None, elapsed_ms: float = 0.0,
        lateness_ms: float = 0.0, missed_deadline: bool = False,
        manifest=None, constraints=None,
    ) -> None:
        """One complete decision record.

        Everything the audit's item-12 list names that exists at this point
        is written. Absences are recorded as nulls WITH a reason rather than
        omitted, so a gap in the journal is always explained.
        """
        p = shadow.session.portfolio
        payload = {
            # identity
            "round_id": shadow.round_id,
            "condition_id": shadow.metadata.condition_id,
            "strategy_version": manifest.strategy_version if manifest else None,
            "config_hash": manifest.config_hash if manifest else None,
            "model_version": manifest.model_version if manifest else None,
            "feature_version": getattr(fv, "feature_version", None) if fv else None,
            # clock
            "decision_ts": decision_ts,
            "elapsed_t_s": t,
            "tau": getattr(fv, "tau", None) if fv else None,
            "decide_elapsed_ms": elapsed_ms,
            "lateness_ms": lateness_ms,
            "missed_deadline": missed_deadline,
            # market constraints in force
            "tick_size": getattr(constraints, "tick_size", None),
            "min_order_shares": getattr(constraints, "min_order_shares", None),
            "taker_delay_ms": getattr(constraints, "taker_delay_ms", None),
            # features / model / regime
            "features": _jsonable(fv) if fv is not None else None,
            "invalid_reason": invalid_reason,
            "q": q,
            "regime_state": getattr(getattr(regime, "state", None), "value", None),
            "regime_seed_action": getattr(getattr(regime, "seed_action", None), "value", None),
            # portfolio BEFORE the action is applied
            "portfolio_before": {
                "U": p.U, "D": p.D, "C": p.C,
                "Pi_U": p.U - p.C, "Pi_D": p.D - p.C,
            },
            # every candidate considered, not just the winner
            "candidates": [_jsonable(c) for c in (getattr(decision, "candidates", None) or [])],
            "chosen": _jsonable(chosen) if chosen is not None else None,
            "blocked_reason": blocked_reason,
            # provenance linkage to the raw log
            "raw_seq_first": shadow.raw_seq_first,
            "raw_seq_last": shadow.raw_seq_last,
        }
        self._write("decision", shadow.round_id, decision_ts, payload)

    def write_decision_error(self, shadow, decision_ts: float, t: float, exc: Exception) -> None:
        self._write("decision_error", shadow.round_id, decision_ts, {
            "elapsed_t_s": t,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        })

    def write_settlement(self, shadow, capture, manifest) -> None:
        """Final outcome and paper PnL.

        Maker fills on REAL data are deliberately NOT adjudicated - see
        `execution/session.py`. When any maker quote is unresolved the round
        PnL is reported as NOT IDENTIFIABLE rather than silently counting
        the maker as unfilled, which would be an assumption dressed as a
        measurement.
        """
        p = shadow.session.portfolio
        rec = capture.reconstruction
        reported = capture.reported_outcome.value if capture.reported_outcome else None
        reconstructed = (rec.declared.outcome.value
                         if rec is not None and rec.declared.outcome else None)
        unresolved = int(getattr(shadow.session, "n_maker_expired_unresolved", 0) or 0)
        open_makers = list(shadow.session.supervisor.open_order_ids())

        pnl = None
        pnl_status = "UNKNOWN"
        if reported == "UP":
            pnl, pnl_status = p.U - p.C, "IDENTIFIED"
        elif reported == "DOWN":
            pnl, pnl_status = p.D - p.C, "IDENTIFIED"
        if unresolved or open_makers:
            pnl_status = "NOT_IDENTIFIABLE_UNRESOLVED_MAKER"

        self._write("settlement", shadow.round_id, None, {
            "round_id": shadow.round_id,
            "reported_outcome": reported,
            "reconstructed_outcome": reconstructed,
            "label_status": rec.status.value if rec is not None else None,
            "settlement_kind": shadow.metadata.settlement_kind,
            "p0": shadow.p0,
            "U": p.U, "D": p.D, "C": p.C,
            "paper_pnl": pnl,
            "paper_pnl_status": pnl_status,
            "unresolved_makers": unresolved,
            "open_makers_at_settlement": len(open_makers),
            "decisions": shadow.decisions,
            "grid_points_fired": len(shadow.fired),
            "blocked": dict(shadow.blocked),
            "missed_deadlines": shadow.missed_deadlines,
            "projected_events": dict(shadow.projector.counts),
            "projection_skipped": dict(shadow.projector.skipped),
            "raw_seq_first": shadow.raw_seq_first,
            "raw_seq_last": shadow.raw_seq_last,
            "strategy_version": manifest.strategy_version,
            "config_hash": manifest.config_hash,
            "model_version": manifest.model_version,
        })

    # ------------------------------------------------------------ readers

    def read(self, kind: str | None = None, round_id: str | None = None) -> list[dict]:
        sql = "SELECT kind, round_id, decision_ts, payload FROM shadow_journal"
        clauses, args = [], []
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        if round_id:
            clauses.append("round_id = ?")
            args.append(round_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        rows = self._conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            # The payload owns `round_id`; the column is an index. Merge with
            # the payload winning, so a reader never sees two of a key.
            record = {"kind": r[0], "round_id": r[1], "decision_ts": r[2]}
            record.update(json.loads(r[3]))
            out.append(record)
        return out

    def counts(self) -> dict:
        rows = self._conn.execute(
            "SELECT kind, COUNT(*) FROM shadow_journal GROUP BY kind"
        ).fetchall()
        return {k: n for k, n in rows}

    def close(self) -> None:
        self._conn.close()
