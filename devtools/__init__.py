"""Development-only tooling. NOT part of the shipped `xamarinbot` package.

Everything under here fabricates data or drives demos. It exists so unit and
property tests have deterministic fixtures and so the pipeline can be
exercised end to end without a live venue.

`tests/test_import_boundaries.py` asserts that **nothing** under
`src/xamarinbot/**` imports this package. That guard is the mechanism behind
Phase 12C.1's invariant:

    Production/live code can never accidentally consume synthetic data.
"""
