from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent.analyzer import write_json, write_markdown


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class AgentRuntimeGuard:
    def __init__(
        self,
        out: Path,
        max_api_calls: int = 20,
        max_tokens_total: int = 20000,
        max_same_prompt: int = 2,
        max_consecutive_errors: int = 3,
        max_consecutive_blocked: int = 5,
        mode: str = "dry_run",
    ) -> None:
        self.out = out
        self.max_api_calls = int(max_api_calls)
        self.max_tokens_total = int(max_tokens_total)
        self.max_same_prompt = int(max_same_prompt)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.max_consecutive_blocked = int(max_consecutive_blocked)
        self.mode = mode
        self.api_calls = 0
        self.estimated_tokens = 0
        self.prompt_hash_counts: dict[str, int] = {}
        self.consecutive_errors = 0
        self.consecutive_blocked = 0
        self.hard_stop: dict[str, Any] | None = None

    def estimate_tokens(self, prompt: str) -> int:
        return max(1, len(prompt) // 4)

    def check_before_call(self, prompt: str, last_command: str | None = None) -> tuple[bool, dict[str, Any]]:
        ph = prompt_hash(prompt)
        repeated = self.prompt_hash_counts.get(ph, 0)
        estimated = self.estimate_tokens(prompt)
        reason = None
        if self.api_calls >= self.max_api_calls:
            reason = "max_api_calls"
        elif self.estimated_tokens + estimated > self.max_tokens_total:
            reason = "max_tokens_total"
        elif repeated >= self.max_same_prompt:
            reason = "same_prompt_repeated"
        data = {
            "stop_reason": reason,
            "api_calls": self.api_calls,
            "estimated_tokens": self.estimated_tokens,
            "prompt_hash": ph,
            "repeated_count": repeated,
            "last_command": last_command,
            "recommended_action": "Switch to /dry, reduce input files, or start a new shell.",
        }
        return reason is None, data

    def record_call(self, prompt: str, response_usage: dict[str, Any] | None = None) -> None:
        if self.mode == "dry_run":
            return
        ph = prompt_hash(prompt)
        self.prompt_hash_counts[ph] = self.prompt_hash_counts.get(ph, 0) + 1
        usage_total = None
        if response_usage:
            try:
                usage_total = int(response_usage.get("total_tokens"))
            except (TypeError, ValueError):
                usage_total = None
        self.api_calls += 1
        self.estimated_tokens += usage_total if usage_total is not None else self.estimate_tokens(prompt)

    def write_limit_stop(self, target: Path, data: dict[str, Any]) -> None:
        write_json(target / "limit_stop.json", data)
        write_markdown(target / "limit_stop.md", ["# Agent Limit Stop", "", *[f"- {key}: {value}" for key, value in data.items()]])

    def record_result(self, status: str) -> tuple[bool, dict[str, Any]]:
        """Record a command result and check consecutive error/blocked thresholds.

        Args:
            status: One of ok, error, blocked, loop_stopped, timeout.

        Returns:
            (should_continue, data) where should_continue is False if hard stop triggered.
        """
        error_statuses = {"error", "timeout", "loop_stopped"}

        if status == "ok":
            self.consecutive_errors = 0
            self.consecutive_blocked = 0
        elif status in error_statuses:
            self.consecutive_errors += 1
        elif status == "blocked":
            self.consecutive_blocked += 1

        stop_reason = None
        if self.consecutive_errors >= self.max_consecutive_errors:
            stop_reason = "max_consecutive_errors"
        elif self.consecutive_blocked >= self.max_consecutive_blocked:
            stop_reason = "max_consecutive_blocked"

        data = {
            "stop_reason": stop_reason,
            "status": status,
            "consecutive_errors": self.consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "consecutive_blocked": self.consecutive_blocked,
            "max_consecutive_blocked": self.max_consecutive_blocked,
        }

        if stop_reason:
            self.hard_stop = data
            self.write_limit_stop(self.out, {
                "type": "hard_stop",
                "reason": stop_reason,
                "message": "Agent stopped after repeated errors. Human review required.",
                "consecutive_errors": self.consecutive_errors,
                "consecutive_blocked": self.consecutive_blocked,
            })
            return False, data
        return True, data

    def clear_hard_stop(self) -> None:
        """Clear hard stop state and reset counters."""
        self.hard_stop = None
        self.consecutive_errors = 0
        self.consecutive_blocked = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "max_api_calls": self.max_api_calls,
            "estimated_tokens": self.estimated_tokens,
            "max_tokens_total": self.max_tokens_total,
            "repeated_prompt_count": max(self.prompt_hash_counts.values()) if self.prompt_hash_counts else 0,
            "consecutive_errors": self.consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "consecutive_blocked": self.consecutive_blocked,
            "max_consecutive_blocked": self.max_consecutive_blocked,
            "hard_stop": self.hard_stop is not None,
        }


def sync_guard_state(state: dict[str, Any], guard: AgentRuntimeGuard) -> None:
    snap = guard.snapshot()
    state["api_calls"] = snap["api_calls"]
    state["estimated_tokens"] = snap["estimated_tokens"]
    state["prompt_hash_counts"] = dict(guard.prompt_hash_counts)
