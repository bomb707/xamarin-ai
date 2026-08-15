"""Executable market constraints, read from the market itself.

Phase 12C.1's second invariant:

    All executable market constraints come from the current Polymarket
    market, not static guesses.

This package deliberately depends on nothing under `realtime/`, so the
strategy and execution layers can consume `MarketConstraints` without
dragging in a live-market client. The real adapter direction is one-way:
`realtime/discovery.py` knows how to build a `MarketConstraints`; this
package does not know `realtime` exists.
"""
