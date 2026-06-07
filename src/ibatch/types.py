"""Types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Strategy(StrEnum):
    STATIC_BATCH = "static_batch"
    CONTINUOUS_BATCH = "continuous_batch"
    CHUNKED_PREFILL = "chunked_prefill"


class Request(BaseModel):
    rid: int
    arrival_step: int = Field(..., ge=0)
    prompt_tokens: int = Field(..., ge=1)
    output_tokens: int = Field(..., ge=1)


class SchedulerConfig(BaseModel):
    max_batch_size: int = 64
    kv_budget_tokens: int = Field(default=65_536, ge=1)
    bytes_per_token: int = 4096
    decode_tokens_per_step: int = 1
    prefill_tokens_per_step: int = 8192
    chunk_size_tokens: int = 2048
    max_steps: int = 5000


class RunResult(BaseModel):
    strategy: Strategy
    n_requests: int
    n_completed: int
    n_rejected: int
    n_steps: int
    throughput_tokens_per_step: float
    p50_latency_steps: float
    p95_latency_steps: float
    p99_latency_steps: float
    mean_kv_utilization: float
    peak_active_batch: int
