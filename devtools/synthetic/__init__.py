"""SYNTHETIC market-data generation. Fabricated, not observed.

Quarantined out of `src/xamarinbot/` by Phase 12C.1 item 4. It previously
lived at `src/xamarinbot/synthetic/`, inside the production runtime
namespace, where two genuinely production modules
(`walkforward/pipeline.py`, `model/dataset.py`) had imported from it.

Anything this package produces is stamped
`DataProvenance.SYNTHETIC_TEST`, and every economic-evaluation entry point
refuses that provenance unless the caller passes `allow_synthetic=True`.
"""
