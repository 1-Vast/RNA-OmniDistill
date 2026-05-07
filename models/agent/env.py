"""Runtime environment detection for Agent context awareness.

Detects platform, hostname, working directory, and CUDA availability
to guide the Agent's training/location suggestions.
"""

from __future__ import annotations

import os
import platform
import socket
from pathlib import Path
from typing import Any


def detect_runtime_env() -> dict[str, Any]:
    """Return a dict describing the current runtime environment.

    Keys:
        hostname, platform, os_name, cwd, is_windows, is_remote_guess,
        has_cuda (if torch is available), torch_available.
    Does NOT read passwords, .env, or API keys.
    """
    info: dict[str, Any] = {
        "hostname": _safe_hostname(),
        "platform": platform.system(),
        "os_name": os.name,
        "cwd": str(Path.cwd()),
        "is_windows": os.name == "nt",
        "is_remote_guess": False,
        "torch_available": False,
        "has_cuda": False,
    }

    # Heuristic remote detection
    hostname_lower = info["hostname"].lower()
    cwd_lower = info["cwd"].lower()
    remote_signals = any(
        keyword in hostname_lower + cwd_lower
        for keyword in ("seetacloud", "connect", "/root/", "auto-dl", "autodl")
    )
    if remote_signals and os.name != "nt":
        info["is_remote_guess"] = True

    # Optional: CUDA detection
    try:
        import torch
        info["torch_available"] = True
        info["has_cuda"] = torch.cuda.is_available()
    except ImportError:
        pass

    return info


def _safe_hostname() -> str:
    """Return hostname without raising on network errors."""
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def env_summary() -> str:
    """One-line summary for Agent display."""
    info = detect_runtime_env()
    location = "remote" if info["is_remote_guess"] else "local"
    cuda = "cuda" if info["has_cuda"] else "no_cuda"
    return f"{info['platform']} | {location} | {cuda} | {info['cwd']}"
