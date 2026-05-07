"""Thin compatibility wrapper -- delegates to main.py.

Kept for any external tooling that may still import from models.cli.
New code should import from main directly.
"""

from main import build_parser

__all__ = ["build_parser"]
