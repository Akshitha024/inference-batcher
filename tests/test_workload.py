"""Workload tests."""

from __future__ import annotations

import pytest

from ibatch.workload.generator import generate


@pytest.mark.parametrize("seed", [11, 17, 23])
def test_deterministic(seed: int) -> None:
    a = generate(n=200, seed=seed)
    b = generate(n=200, seed=seed)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]


def test_arrival_times_monotonic() -> None:
    reqs = generate(n=500, seed=1)
    arrivals = [r.arrival_step for r in reqs]
    assert all(arrivals[i] <= arrivals[i + 1] for i in range(len(arrivals) - 1))


def test_has_long_tail() -> None:
    reqs = generate(n=1000, seed=2)
    long = [r for r in reqs if r.prompt_tokens >= 4000]
    assert len(long) >= 200  # roughly 30%
