"""Six chart families."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ibatch.types import RunResult


def _save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def throughput_bars(rows: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    tps = [r.throughput_tokens_per_step for r in rows]
    ax.bar(strats, tps, color=["#3b6fa1", "#5b8d4a", "#c25a4f"])
    ax.set_ylabel("tokens / step")
    ax.set_title("Throughput by scheduling strategy")
    for i, v in enumerate(tps):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    return _save(fig, out)


def latency_bars(rows: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    strats = [r.strategy.value for r in rows]
    x = np.arange(len(strats))
    w = 0.27
    ax.bar(x - w, [r.p50_latency_steps for r in rows], w, label="p50")
    ax.bar(x, [r.p95_latency_steps for r in rows], w, label="p95")
    ax.bar(x + w, [r.p99_latency_steps for r in rows], w, label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(strats)
    ax.set_ylabel("latency (steps)")
    ax.set_title("Per-request latency percentiles")
    ax.legend()
    return _save(fig, out)


def kv_util_bars(rows: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    util = [r.mean_kv_utilization for r in rows]
    ax.bar(strats, util, color="#5b8d4a")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean KV utilization")
    ax.set_title("Mean KV-budget utilization")
    return _save(fig, out)


def rejected_bars(rows: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    rej = [r.n_rejected for r in rows]
    ax.bar(strats, rej, color="#c25a4f")
    ax.set_ylabel("rejected requests")
    ax.set_title("Admission-control rejections")
    return _save(fig, out)


def peak_batch_bar(rows: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    strats = [r.strategy.value for r in rows]
    peak = [r.peak_active_batch for r in rows]
    ax.bar(strats, peak, color="#3b6fa1")
    ax.set_ylabel("peak active batch")
    ax.set_title("Peak concurrent sessions")
    return _save(fig, out)


def completion_pie(rows: list[RunResult], out: Path) -> Path:
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4))
    if len(rows) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows, strict=True):
        ax.pie(
            [r.n_completed, r.n_rejected],
            labels=["completed", "rejected"],
            autopct="%1.0f%%",
            colors=["#5b8d4a", "#c25a4f"],
        )
        ax.set_title(r.strategy.value)
    return _save(fig, out)
