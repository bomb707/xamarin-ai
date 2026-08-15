"""Bounded ingestion queue + batched writer thread (Phase 12C item 6).

    "Use an asynchronous bounded ingestion queue and a batched/WAL-backed
     writer or equivalent architecture. The WebSocket reader must not block
     on one SQLite commit per incoming event."

The measured live rate for one BTC 5-minute round's two tokens is ~130
`price_change` messages/second (docs/REAL_RECORDER_ARCHITECTURE.md), so the
reader path here does exactly three things per event: build the immutable
record, bump counters, and `put_nowait`. Everything else - serialization
into a transaction, the commit, duplicate detection - happens on the writer
thread.

The queue is BOUNDED on purpose. An unbounded queue does not actually solve
back-pressure; it converts a persistence stall into unbounded memory growth
and eventually loses the whole capture instead of a countable prefix of it.
A bounded queue that drops and COUNTS is strictly more honest, and the
count is exactly what `RecorderMetrics.is_training_grade()` keys off - a
dropped event is not a warning, it disqualifies the interval.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from xamarinbot.realtime.metrics import RecorderMetrics
from xamarinbot.realtime.raw_events import RawEvent
from xamarinbot.realtime.raw_store import RawEventStore


@dataclass(frozen=True)
class RecorderConfig:
    #: Maximum events buffered between the reader threads and the writer.
    #: 50k at the measured ~130 events/s is roughly six minutes of slack -
    #: longer than a whole 5-minute round, so a transient disk stall cannot
    #: cost a round, while still bounding memory at a few tens of MB.
    queue_maxsize: int = 50_000
    #: Rows per transaction. Large enough to amortize the commit, small
    #: enough that a crash loses well under a second of data.
    batch_size: int = 500
    #: Force a commit even on a partially-filled batch after this long, so a
    #: quiet feed still lands on disk promptly.
    batch_timeout_s: float = 0.5
    #: Remember this many recent dedupe keys. A duplicate can only
    #: realistically arrive within a resubscribe window, so an unbounded
    #: set would grow forever to catch nothing.
    dedupe_window: int = 20_000


class RawRecorder:
    """Owns the queue, the writer thread, and the health metrics.

    Usage is deliberately explicit rather than implicit-on-import:

        recorder = RawRecorder(store, metrics)
        recorder.start()
        ...  # feed threads call recorder.submit(event)
        recorder.flush()     # at round finalization
        recorder.stop()
    """

    def __init__(
        self,
        store: RawEventStore,
        metrics: RecorderMetrics | None = None,
        cfg: RecorderConfig | None = None,
    ):
        self.store = store
        self.metrics = metrics or RecorderMetrics()
        self.cfg = cfg or RecorderConfig()
        self._queue: queue.Queue = queue.Queue(maxsize=self.cfg.queue_maxsize)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dedupe: dict[tuple, None] = {}
        self._flush_lock = threading.Lock()
        self._drained = threading.Event()
        self._drained.set()

    # --------------------------------------------------------- reader side

    def submit(self, event: RawEvent) -> bool:
        """Called from a feed reader thread. NEVER blocks and never raises:
        a full queue drops the event and increments `dropped_events`, which
        is what disqualifies the interval for training. Returns True if the
        event was enqueued."""
        self.metrics.record_received(
            event.topic.value,
            event.event_type,
            event.source_to_recv_latency_ns,
            event.publisher_to_recv_latency_ns,
        )
        self._drained.clear()
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.metrics.record_dropped()
            return False
        self.metrics.observe_queue_depth(self._queue.qsize())
        return True

    # --------------------------------------------------------- writer side

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="raw-recorder-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            batch = self._collect_batch()
            if batch:
                self._persist(batch)
            elif self._queue.empty():
                self._drained.set()

    def _collect_batch(self) -> list[RawEvent]:
        """Blocks up to `batch_timeout_s` for the first event, then drains
        without blocking up to `batch_size`. This is what turns N commits
        into one without adding latency on a quiet feed."""
        batch: list[RawEvent] = []
        try:
            first = self._queue.get(timeout=self.cfg.batch_timeout_s)
        except queue.Empty:
            return batch
        batch.append(first)
        while len(batch) < self.cfg.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _persist(self, batch: list[RawEvent]) -> None:
        deduped = []
        for e in batch:
            key = e.dedupe_key
            if key in self._dedupe:
                self.metrics.record_duplicate()
                continue
            self._dedupe[key] = None
            deduped.append(e)
        # Bounded LRU-ish trim: dict preserves insertion order, so dropping
        # from the front drops the oldest keys.
        excess = len(self._dedupe) - self.cfg.dedupe_window
        if excess > 0:
            for key in list(self._dedupe)[:excess]:
                del self._dedupe[key]

        with self._flush_lock:
            n = self.store.write_batch(deduped)
        self.metrics.record_persisted(n)
        if self._queue.empty():
            self._drained.set()

    # ------------------------------------------------------------ control

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Blocks until the queue has been drained to the store, or
        `timeout_s` elapses. Called at round finalization (item 8: "flush
        buffered events") so a round's data is durable before its result is
        written. Returns True if fully drained."""
        if self._thread is None:
            # No writer thread: drain synchronously so a test or a
            # single-threaded caller still gets correct behavior rather
            # than a silent no-op.
            while True:
                batch = []
                try:
                    while len(batch) < self.cfg.batch_size:
                        batch.append(self._queue.get_nowait())
                except queue.Empty:
                    pass
                if not batch:
                    break
                self._persist(batch)
            return True
        return self._drained.wait(timeout=timeout_s) and self._queue.empty()

    def stop(self, timeout_s: float = 10.0) -> None:
        self.flush(timeout_s)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
