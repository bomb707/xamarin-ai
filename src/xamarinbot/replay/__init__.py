"""Deterministic replay of a recorded `EventStore` (Phase 12C.1 item 3).

This package was `replay/feeds.py`. The rename is not cosmetic: the module
contains no random number generator and fabricates no values - it reads an
`EventStore` that something else populated and reconstructs feed-interface
views of it, gated by a causal cursor. Calling that "Mock" while it was the
component replaying **real captured market data** into the shadow runner was
actively misleading, and it is what made the synthetic-vs-real boundary hard
to see at a glance.

    feeds.py       replay adapters over a normalized EventStore
    projection.py  REAL raw capture -> normalized EventStore (item 8)

The genuine data fabricator now lives outside the production namespace
entirely, in `devtools/synthetic/`, and `tests/test_import_boundaries.py`
asserts that nothing under `src/xamarinbot/**` can import it.
"""
