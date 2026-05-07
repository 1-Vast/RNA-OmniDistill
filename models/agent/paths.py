"""Run discovery helper: find recent experiment directories under outputs/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_RUN_MARKERS = (
    "trainlog.jsonl",
    "best.pt",
    "last.pt",
    "benchmark.json",
    "predictions.jsonl",
)

_SKIP_DIRS = {".git", "__pycache__", "dataset", "checkpoints", "tuning", "llm_shell"}


def discover_runs(root: Path = Path("outputs"), max_items: int = 20) -> list[dict[str, Any]]:
    """Find experiment run directories under *root* by checking for marker files.

    Args:
        root: Root directory to search (default: outputs/).
        max_items: Maximum number of results to return.

    Returns:
        List of dicts with keys: path, has_trainlog, has_best, has_last,
        has_benchmark, has_predictions, modified_time. Sorted by modified_time
        descending (most recent first).
    """
    if not root.exists():
        return []

    results: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name in _SKIP_DIRS:
            continue

        info: dict[str, Any] = {
            "path": str(entry),
            "name": entry.name,
            "has_trainlog": (entry / "trainlog.jsonl").exists(),
            "has_best": (entry / "best.pt").exists(),
            "has_last": (entry / "last.pt").exists(),
            "has_benchmark": (entry / "benchmark.json").exists(),
            "has_predictions": (entry / "predictions.jsonl").exists(),
        }

        # Get latest modification time from any marker or directory itself
        mtimes = [entry.stat().st_mtime]
        for marker in _RUN_MARKERS:
            mp = entry / marker
            if mp.exists():
                mtimes.append(mp.stat().st_mtime)
        info["modified_time"] = max(mtimes)

        # Only include if at least one marker exists
        if any(info.get(f"has_{m.replace('.jsonl', '').replace('.json', '').replace('.pt', '')}")
               for m in _RUN_MARKERS):
            results.append(info)

    results.sort(key=lambda r: r["modified_time"], reverse=True)
    return results[:max_items]


def discover_latest_run(root: Path = Path("outputs")) -> dict[str, Any] | None:
    """Return the most recent run directory, or None."""
    runs = discover_runs(root, max_items=1)
    return runs[0] if runs else None
