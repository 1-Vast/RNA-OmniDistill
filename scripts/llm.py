"""Compatibility wrapper for the Agent CLI.

The implementation lives in the top-level ``agent`` package so it can be
packaged independently from the model framework.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.cli import *  # noqa: F403
from agent.cli import main


if __name__ == "__main__":
    main()
