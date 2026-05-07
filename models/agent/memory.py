from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SENSITIVE_PATTERNS = (
    re.compile(r"(password|api[_-]?key|llm_api_key|token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(密码)\s*[:=：]\s*\S+", re.IGNORECASE),
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
