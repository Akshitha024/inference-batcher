"""Synthetic request workload calibrated to chat-serving distributions."""

from __future__ import annotations

import numpy as np

from ibatch.types import Request


def generate(n: int = 10_000, arrival_rate_per_step: float = 4.0, seed: int = 17) -> list[Request]:
    rng = np.random.default_rng(seed)
    # Prompt lengths log-normal with a 30% long-tail above 4000 tokens.
    prompts = np.where(
        rng.random(n) < 0.3,
        rng.uniform(4000, 16000, size=n),
        np.maximum(64, rng.lognormal(6.6, 0.35, size=n)),
    ).astype(int)
    outputs = np.maximum(8, rng.lognormal(5.0, 0.4, size=n)).astype(int)
    # Poisson arrivals at the given rate.
    inter = rng.exponential(1.0 / arrival_rate_per_step, size=n)
    arrival = np.cumsum(inter).astype(int)
    return [
        Request(
            rid=i,
            arrival_step=int(arrival[i]),
            prompt_tokens=int(prompts[i]),
            output_tokens=int(outputs[i]),
        )
        for i in range(n)
    ]
