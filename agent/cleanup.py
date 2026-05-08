from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.analyzer import ROOT, write_json, write_markdown


def normalize_root(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def safe_root(root: Path) -> bool:
    resolved = normalize_root(root)
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return False
    allowed = [
        ROOT / "outputs" / "llm",
        ROOT / "outputs" / "llm_shell",
        ROOT / "outputs" / "llm_shell_test",
        ROOT / "outputs" / "llm_test",
    ]
    server_root = ROOT / "outputs"
    if resolved.name.startswith("llm_server_") and resolved.parent == server_root.resolve():
        allowed.append(resolved)
    blocked = [ROOT / item for item in ["dataset", "config", "models", "scripts", "release", "docs", ".git"]]
    return any(resolved == item.resolve() or item.resolve() in resolved.parents for item in allowed) and not any(
        resolved == item.resolve() or item.resolve() in resolved.parents for item in blocked
    )


def validate_cleanup_request(root: Path, keep: int = 10) -> dict[str, Any]:
    original_keep = keep
    warnings = []
    if keep < 1:
        keep = 10
        warnings.append("keep < 1 normalized to 10")
    safe = safe_root(root)
    return {
        "root": str(root),
        "keep": original_keep,
        "normalized_keep": keep,
        "safe": safe,
        "status": "PASS" if safe else "blocked",
        "warnings": warnings,
        "errors": [] if safe else ["unsafe root"],
    }


def cleanup_reports(root: Path, keep: int = 10, dry_run: bool = False) -> dict[str, Any]:
    validation = validate_cleanup_request(root, keep=keep)
    keep = int(validation["normalized_keep"])
    if validation["status"] == "blocked":
        return {**validation, "dry_run": dry_run, "kept_dirs": [], "removed_dirs": []}
    root.mkdir(parents=True, exist_ok=True)
    base_dirs = list((root / "turns").iterdir()) if (root / "turns").exists() else []
    if not base_dirs:
        base_dirs = [path for path in root.iterdir() if path.is_dir()]
    candidates = [
        path for path in base_dirs
        if any((path / name).exists() for name in ["prompt.md", "response.md", "inspect.md", "doctor.md", "case_report.md", "request.json", "blocked.md", "error.md"])
    ]
    ordered = sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)
    kept = ordered[: max(0, keep)]
    removed = ordered[max(0, keep):]
    errors = []
    if not dry_run:
        for directory in sorted(removed, key=lambda item: len(item.parts), reverse=True):
            try:
                for item in sorted(directory.rglob("*"), key=lambda child: len(child.parts), reverse=True):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        item.rmdir()
                directory.rmdir()
            except OSError as exc:
                errors.append(f"{directory}: {exc}")
    report = {
        "root": str(root),
        "keep": validation["keep"],
        "normalized_keep": validation["normalized_keep"],
        "total_dirs": len(ordered),
        "kept_dirs": [str(item) for item in kept],
        "removed_dirs": [str(item) for item in removed],
        "dry_run": dry_run,
        "errors": errors,
        "warnings": validation["warnings"],
        "status": "PASS" if not errors else "WARN",
    }
    write_json(root / "cleanup_report.json", report)
    write_markdown(root / "cleanup_report.md", [
        "# Cleanup Report",
        "",
        f"- root: {root}",
        f"- keep: {validation['keep']}",
        f"- normalized_keep: {keep}",
        f"- total_dirs: {len(ordered)}",
        f"- kept: {len(kept)}",
        f"- removed: {len(removed)}",
        f"- dry_run: {dry_run}",
        f"- errors: {len(errors)}",
        f"- warnings: {len(validation['warnings'])}",
    ])
    return report
