"""Discrete-step continuous-batching simulator.

Three strategies are modeled:

  - static_batch: collect requests until a max_batch_size is reached, then
    process the batch in lock-step. Wastes throughput when requests have
    different output lengths.
  - continuous_batch: admit each request as soon as KV budget allows; decode
    one token per active session per step; retire as outputs complete.
    Matches vLLM's iteration-level scheduling.
  - chunked_prefill: like continuous_batch but interleaves prefill chunks with
    decode rather than blocking the whole batch on a prefill. Matches Sarathi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ibatch.types import Request, RunResult, SchedulerConfig, Strategy


@dataclass
class _Session:
    rid: int
    kv_tokens: int
    prefill_remaining: int
    output_remaining: int
    arrival_step: int


def simulate(reqs: list[Request], strategy: Strategy, cfg: SchedulerConfig) -> RunResult:
    pending = sorted(reqs, key=lambda r: r.arrival_step)
    cursor = 0
    active: dict[int, _Session] = {}
    completed_step: dict[int, int] = {}
    rejected = 0
    tokens_produced = 0
    kv_utilization_samples: list[float] = []
    peak_batch = 0

    for step in range(cfg.max_steps):
        # Admit new arrivals.
        while cursor < len(pending) and pending[cursor].arrival_step <= step:
            r = pending[cursor]
            cursor += 1
            need = r.prompt_tokens * cfg.bytes_per_token
            current_use = sum(s.kv_tokens for s in active.values()) * cfg.bytes_per_token
            if current_use + need > cfg.kv_budget_tokens * cfg.bytes_per_token:
                rejected += 1
                continue
            if len(active) >= cfg.max_batch_size:
                rejected += 1
                continue
            active[r.rid] = _Session(
                rid=r.rid,
                kv_tokens=r.prompt_tokens,
                prefill_remaining=r.prompt_tokens,
                output_remaining=r.output_tokens,
                arrival_step=r.arrival_step,
            )

        peak_batch = max(peak_batch, len(active))

        # Per-strategy step.
        if strategy == Strategy.STATIC_BATCH:
            if len(active) >= cfg.max_batch_size // 2 or (cursor >= len(pending) and active):
                _step_decode_all(active, completed_step, step, cfg)
        elif strategy == Strategy.CONTINUOUS_BATCH:
            _step_decode_all(active, completed_step, step, cfg)
        else:  # chunked_prefill
            _step_chunked_prefill(active, completed_step, step, cfg)

        # Count tokens generated this step.
        tokens_produced += sum(1 for s in active.values() if s.prefill_remaining == 0)

        # KV utilization sample.
        used = sum(s.kv_tokens for s in active.values())
        kv_utilization_samples.append(used / max(1, cfg.kv_budget_tokens))

        # Retire completed sessions.
        finished = [sid for sid, s in active.items() if s.output_remaining <= 0]
        for sid in finished:
            completed_step[sid] = step
            del active[sid]

        if not active and cursor >= len(pending):
            break

    arr = np.array(
        [completed_step[r.rid] - r.arrival_step for r in reqs if r.rid in completed_step]
    )
    if arr.size == 0:
        arr = np.array([0.0])
    return RunResult(
        strategy=strategy,
        n_requests=len(reqs),
        n_completed=len(completed_step),
        n_rejected=rejected,
        n_steps=step + 1,
        throughput_tokens_per_step=tokens_produced / max(1, step + 1),
        p50_latency_steps=float(np.percentile(arr, 50)),
        p95_latency_steps=float(np.percentile(arr, 95)),
        p99_latency_steps=float(np.percentile(arr, 99)),
        mean_kv_utilization=float(np.mean(kv_utilization_samples)),
        peak_active_batch=peak_batch,
    )


def _step_decode_all(
    active: dict[int, _Session], _completed: dict[int, int], _step: int, cfg: SchedulerConfig
) -> None:
    """One decode token per session whose prefill is complete; advance prefill otherwise."""
    for s in active.values():
        if s.prefill_remaining > 0:
            advance = min(s.prefill_remaining, cfg.prefill_tokens_per_step)
            s.prefill_remaining -= advance
        else:
            s.output_remaining -= cfg.decode_tokens_per_step
            s.kv_tokens += cfg.decode_tokens_per_step


def _step_chunked_prefill(
    active: dict[int, _Session], _completed: dict[int, int], _step: int, cfg: SchedulerConfig
) -> None:
    """Interleave a prefill chunk with the decode step.

    Each session: if prefill remaining, advance by `chunk_size_tokens` and ALSO
    emit a decode token (mimicking Sarathi-style piggybacking).
    """
    for s in active.values():
        if s.prefill_remaining > 0:
            advance = min(s.prefill_remaining, cfg.chunk_size_tokens)
            s.prefill_remaining -= advance
            if s.prefill_remaining == 0:
                # Piggyback the first decode token in the same step.
                s.output_remaining -= cfg.decode_tokens_per_step
                s.kv_tokens += cfg.decode_tokens_per_step
        else:
            s.output_remaining -= cfg.decode_tokens_per_step
            s.kv_tokens += cfg.decode_tokens_per_step
