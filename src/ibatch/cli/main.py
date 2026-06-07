"""Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from ibatch.runner import run

app = typer.Typer(no_args_is_help=True, help="Continuous-batching inference scheduler simulator.")
console = Console()


@app.command()
def info() -> None:
    console.print("inference-batcher: see `ibatch bench --help`.")


@app.command()
def bench(
    out_dir: Path = typer.Option(Path("runs/latest")),
    n: int = typer.Option(10_000),
    seed: int = typer.Option(17),
) -> None:
    res = run(out_dir, n_requests=n, seed=seed)
    results_any = res["results"]
    assert isinstance(results_any, list)
    headline = [
        {
            "strategy": r["strategy"],
            "throughput": r["throughput_tokens_per_step"],
            "p99": r["p99_latency_steps"],
            "rejected": r["n_rejected"],
        }
        for r in results_any
    ]
    console.print_json(json.dumps(headline, default=str))


if __name__ == "__main__":
    app()
