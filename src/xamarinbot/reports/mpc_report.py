"""Controller latency benchmark (Roadmap Phase 10 deliverable)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyBenchmarkReport:
    n_calls: int
    n_fallback: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @property
    def fallback_rate(self) -> float:
        return self.n_fallback / self.n_calls if self.n_calls else 0.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, math.ceil(pct / 100.0 * len(sorted_values)) - 1)
    return sorted_values[max(0, idx)]


def build_latency_benchmark(elapsed_ms_samples: list[float], fallback_flags: list[bool]) -> LatencyBenchmarkReport:
    ordered = sorted(elapsed_ms_samples)
    return LatencyBenchmarkReport(
        n_calls=len(elapsed_ms_samples),
        n_fallback=sum(fallback_flags),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        p99_ms=_percentile(ordered, 99),
        max_ms=ordered[-1] if ordered else 0.0,
    )


def format_latency_benchmark(report: LatencyBenchmarkReport) -> str:
    return (
        "=== MPCController Latency Benchmark (Phase 10) ===\n"
        f"Calls: {report.n_calls}  |  Fallback to one-step: {report.n_fallback} ({report.fallback_rate:.1%})\n"
        f"p50: {report.p50_ms:.3f}ms  |  p95: {report.p95_ms:.3f}ms  |  p99: {report.p99_ms:.3f}ms  |  max: {report.max_ms:.3f}ms"
    )
