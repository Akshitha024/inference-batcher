"""End-to-end runner."""

from __future__ import annotations

import json
from pathlib import Path

from ibatch.scheduler.sim import simulate
from ibatch.types import RunResult, SchedulerConfig, Strategy
from ibatch.viz.charts import (
    completion_pie,
    kv_util_bars,
    latency_bars,
    peak_batch_bar,
    rejected_bars,
    throughput_bars,
)
from ibatch.workload.generator import generate


def run(out_dir: Path, n_requests: int = 10_000, seed: int = 17) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = Path("results/figures")
    reqs = generate(n=n_requests, seed=seed)
    cfg = SchedulerConfig()
    rows: list[RunResult] = []
    for s in [Strategy.STATIC_BATCH, Strategy.CONTINUOUS_BATCH, Strategy.CHUNKED_PREFILL]:
        rows.append(simulate(reqs, s, cfg))
    throughput_bars(rows, figs / "throughput.png")
    latency_bars(rows, figs / "latency.png")
    kv_util_bars(rows, figs / "kv_util.png")
    rejected_bars(rows, figs / "rejected.png")
    peak_batch_bar(rows, figs / "peak_batch.png")
    completion_pie(rows, figs / "completion.png")
    summary: dict[str, object] = {
        "n_requests": n_requests,
        "config": cfg.model_dump(),
        "results": [r.model_dump() for r in rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
