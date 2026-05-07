"""Minimal launcher for the RNA-OmniDiffusion Agent Shell.

Usage:
    python agent.py              # dry-run mode
    python agent.py --live       # live API mode
    python agent.py --quiet      # quiet UI
    python agent.py --verbose    # verbose UI
    python agent.py --help       # agent CLI help

On Windows CMD:  agent
On PowerShell:   python agent.py   or   .\agent.cmd
On Linux/macOS:  python agent.py   or   bash agent
"""

from __future__ import annotations

import sys

try:
    from scripts.llm import main as _llm_main
except ImportError:
    # Add repo root to path if running from repo root
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parent
    sys.path.insert(0, str(_ROOT))
    from scripts.llm import main as _llm_main


def main() -> None:
    argv = ["agent"]
    if len(sys.argv) == 1:
        argv.append("--dry_run")
        argv.append("--no_api")
    else:
        argv.extend(sys.argv[1:])
    _llm_main(argv)


if __name__ == "__main__":
    main()
