"""End-to-end runner smoke test."""

from __future__ import annotations

from pathlib import Path

from ibatch.runner import run


def test_runner_smoke(tmp_path: Path) -> None:
    s = run(tmp_path / "out", n_requests=100, seed=1)
    assert len(s["results"]) == 3
    assert (tmp_path / "out" / "summary.json").exists()
