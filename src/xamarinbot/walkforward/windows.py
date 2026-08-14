"""Rolling walk-forward windows (Roadmap Phase 11: "Run rolling train/
validate/test windows across many rounds and volatility regimes." "Lock
parameters before each test segment.")

Each window's train/validate rounds strictly precede its own test rounds
chronologically (round_ids are assumed pre-ordered chronologically, as
`synthetic/rounds.py` generates them) - the walk-forward analogue of Phase
2's causal-replay invariant: a test segment is only ever evaluated with
parameters fit on data that came before it, never after.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardWindow:
    window_index: int
    train_round_ids: tuple[str, ...]
    validate_round_ids: tuple[str, ...]
    test_round_ids: tuple[str, ...]

    @property
    def all_round_ids(self) -> tuple[str, ...]:
        return self.train_round_ids + self.validate_round_ids + self.test_round_ids


def rolling_windows(
    round_ids_chronological: list[str], n_train: int, n_validate: int, n_test: int, step: int | None = None
) -> list[WalkForwardWindow]:
    """`step` controls how far each window advances - defaults to
    `n_test`, giving strictly non-overlapping test segments that together
    walk forward across the whole dataset, the standard walk-forward
    pattern. A smaller step reuses some rounds' test data as a later
    window's train/validate data (fine - "future" data only ever appears
    in a *later* window's train, never a given window's own test)."""
    if n_train <= 0 or n_validate <= 0 or n_test <= 0:
        raise ValueError("n_train, n_validate, n_test must all be positive")
    step = step if step is not None else n_test
    if step <= 0:
        raise ValueError("step must be positive")

    windows: list[WalkForwardWindow] = []
    n = len(round_ids_chronological)
    start = 0
    idx = 0
    window_size = n_train + n_validate + n_test
    while start + window_size <= n:
        train = tuple(round_ids_chronological[start : start + n_train])
        validate = tuple(round_ids_chronological[start + n_train : start + n_train + n_validate])
        test = tuple(round_ids_chronological[start + n_train + n_validate : start + window_size])
        windows.append(WalkForwardWindow(idx, train, validate, test))
        idx += 1
        start += step
    return windows
