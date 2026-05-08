from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SENSITIVE_PATTERNS = (
    re.compile(r"(password|api[_-]?key|llm_api_key|token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(\u5bc6\u7801)\s*[:=\uff1a]\s*\S+", re.IGNORECASE),
)


def sanitize_text(text: str) -> str:
    cleaned = text
    for pattern in SENSITIVE_PATTERNS:
        cleaned = pattern.sub(lambda m: m.group(0).split(m.group(1))[0] + m.group(1) + "=[REDACTED]", cleaned)
    return cleaned


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_payload(item) for key, item in value.items() if "password" not in str(key).lower() and "api_key" not in str(key).lower()}
    return value


def append_memory(memory_path: Path, kind: str, summary: str, data: dict[str, Any] | None = None) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "summary": sanitize_text(summary),
        "data": sanitize_payload(data or {}),
    }
    with memory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_recent_memory(memory_path: Path, limit: int = 10) -> list[dict[str, Any]]:
    if not memory_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in memory_path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"kind": "corrupt", "summary": "Unreadable memory row."})
    return rows


def compact_memory(
    memory_path: Path,
    keep_recent: int = 50,
    keep_errors: int = 10,
    max_bytes: int = 512_000,
) -> dict[str, Any]:
    """File-level memory compaction: truncate old entries, keep recent + error records.

    Args:
        memory_path: Path to memory.jsonl.
        keep_recent: Number of most recent records to keep.
        keep_errors: Extra error/blocked/loop/failed records to keep (beyond recent).
        max_bytes: If file is under this size and lines <= keep_recent*2, skip compaction.

    Returns:
        dict with keys: status, before_lines, after_lines, before_bytes, after_bytes, kept_recent, kept_errors.
    """
    report: dict[str, Any] = {
        "status": "skipped",
        "path": str(memory_path),
        "before_lines": 0,
        "after_lines": 0,
        "before_bytes": 0,
        "after_bytes": 0,
        "kept_recent": 0,
        "kept_errors": 0,
    }

    if not memory_path.exists():
        report["status"] = "missing"
        return report

    raw = memory_path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    report["before_lines"] = len(lines)
    report["before_bytes"] = len(raw.encode("utf-8"))

    # Skip if under thresholds
    if report["before_bytes"] <= max_bytes and report["before_lines"] <= keep_recent * 2:
        return report

    # Parse all lines
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"kind": "corrupt", "summary": "corrupt_line", "data": {}})

    if not records:
        return report

    # Identify error records by kind
    error_kinds = {"error", "blocked", "loop", "loop_stopped", "failed", "timeout"}
    error_indices = {i for i, r in enumerate(records) if r.get("kind", "") in error_kinds}

    # Start with most recent keep_recent records
    recent_start = max(0, len(records) - keep_recent)
    kept_indices: set[int] = set(range(recent_start, len(records)))

    # Add up to keep_errors error records not already in recent set
    error_candidates = sorted(error_indices - kept_indices, reverse=True)
    kept_indices.update(error_candidates[:keep_errors])

    # Rebuild: keep records in original order
    kept = [records[i] for i in sorted(kept_indices)]

    # Sanitize and deduplicate
    seen = set()
    deduped: list[dict[str, Any]] = []
    for r in kept:
        key = json.dumps(r, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sanitize_payload(r))

    # Write back
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in deduped) + "\n"
    memory_path.write_text(content, encoding="utf-8")
    report["status"] = "compacted"
    report["after_lines"] = len(deduped)
    report["after_bytes"] = len(content.encode("utf-8"))
    report["kept_recent"] = min(keep_recent, len(deduped))
    report["kept_errors"] = len([r for r in deduped if r.get("kind", "") in error_kinds])

    # Write compact report
    report_path = memory_path.parent / "memory_compact_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return report


def auto_compact(memory_path: Path, max_bytes: int = 512_000) -> dict[str, Any]:
    """Call after append_memory; only compacts if file exceeds max_bytes."""
    if not memory_path.exists():
        return {"status": "missing"}
    size = memory_path.stat().st_size
    if size <= max_bytes:
        return {"status": "skipped", "before_bytes": size}
    return compact_memory(memory_path, max_bytes=max_bytes)
