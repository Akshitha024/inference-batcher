"""Scheduler tests."""

from __future__ import annotations

import pytest

from ibatch.scheduler.sim import simulate
from ibatch.types import SchedulerConfig, Strategy
from ibatch.workload.generator import generate


@pytest.mark.parametrize("strat", list(Strategy))
def test_completes_small_workload(strat: Strategy) -> None:
    reqs = generate(n=50, seed=3)
    cfg = SchedulerConfig(kv_budget_tokens=200_000)
    r = simulate(reqs, strat, cfg)
    assert r.n_completed > 0


def test_continuous_batch_higher_throughput_than_static() -> None:
    reqs = generate(n=200, seed=5)
    cfg = SchedulerConfig(kv_budget_tokens=400_000)
    s = simulate(reqs, Strategy.STATIC_BATCH, cfg)
    c = simulate(reqs, Strategy.CONTINUOUS_BATCH, cfg)
    assert c.throughput_tokens_per_step >= s.throughput_tokens_per_step


def test_kv_budget_causes_rejections() -> None:
    reqs = generate(n=200, seed=7)
    cfg = SchedulerConfig(kv_budget_tokens=10_000)  # tiny budget
    r = simulate(reqs, Strategy.CONTINUOUS_BATCH, cfg)
    assert r.n_rejected > 0
