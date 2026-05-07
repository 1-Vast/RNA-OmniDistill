from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.agent.analyzer import (
    RNAAnalysisAgent,
    case_analysis,
    case_markdown,
    compare_markdown,
    compare_runs,
    inspect_markdown,
    inspect_run_artifacts,
    load_env,
    trace_markdown,
    trace_provenance,
    write_json,
    write_markdown,
    write_report,
)


HELP_TEXT = """Built-in commands:
  diagnose <run_dir>
  inspect <run_dir>
  trace <config_path> <ckpt_path> <benchmark_json>
  compare <run_a> <run_b>
  case <predictions_jsonl>
  doctor <run_dir> <config_path>
  schedule <run_dir> <config_path>
  report <file1> <file2> ...
  auditdata <jsonl1> <jsonl2> ...
  dry <command...>
  live <command...>

Slash commands:
  /help
  /status
  /usage
  /cleanup [keep]
  /mode
  /dry
  /live
  /quiet
  /normal
  /verbose
  /history
  /clear
  /exit

Safety:
  Agent is read-only by default.
  It never runs benchmark by default.
  Candidate training requires exact user confirmation."""

CONFIRM_TRAIN = {
    "\u8fdb\u884c\u8bad\u7ec3",
    "\u786e\u8ba4\u8bad\u7ec3",
    "\u6267\u884c\u8bad\u7ec3",
    "yes train",
    "run train",
    "execute train",
}
CONFIRM_BENCH = {
    "\u8fdb\u884c benchmark",
    "\u786e\u8ba4 benchmark",
    "\u6267\u884c benchmark",
    "yes benchmark",
    "run benchmark",
    "execute benchmark",
}
TRAIN_COMMAND = ["python", "main.py", "train", "--config", "config/candidate.yaml", "--device", "cuda"]
SAFE_COMPILE = ["python", "-m", "py_compile", "models/agent/analyzer.py", "scripts/llm.py"]
SAFE_SMOKE = ["python", "main.py", "smoke"]
SAFE_AUDIT = ["python", "scripts/audit.py", "clean", "--out", "outputs/clean"]


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Input directory not found: {path}")
    return path


def write_rule_outputs(
    out: Path,
    stem: str,
    result: dict[str, Any],
    lines: list[str],
    data: dict[str, Any],
    prompt: str,
    dry_run: bool,
    agent: RNAAnalysisAgent,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{stem}.json", result)
    write_markdown(out / f"{stem}.md", lines)
    write_json(out / "request.json", {**data, "prompt": prompt})
    write_markdown(out / "prompt.md", [prompt])
    if not dry_run:
        write_markdown(out / "response.md", [agent.call(prompt)])
    print(f"Wrote {out}")


def run_diagnose(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    data, markdown = agent.diagnose(require_dir(Path(args.run)))
    write_report(Path(args.out), "diagnose", data, markdown)


def run_schedule(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    config = require_file(Path(args.config)) if args.config else None
    data, markdown = agent.schedule(require_dir(Path(args.run)), config)
    write_report(Path(args.out), "schedule", data, markdown)


def run_report(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    data, markdown = agent.report([require_file(Path(path)) for path in args.inputs], max_chars=args.max_chars)
    write_report(Path(args.out), "report", data, markdown)


def run_auditdata(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    data, markdown = agent.auditdata([require_file(Path(path)) for path in args.inputs], max_rows=args.max_rows)
    write_report(Path(args.out), "dataaudit", data, markdown)


def run_inspect(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    result = inspect_run_artifacts(Path(args.run))
    data, prompt = agent.build_inspect_prompt(result)
    write_rule_outputs(Path(args.out), "inspect", result, inspect_markdown(result), data, prompt, args.dry_run, agent)


def run_trace(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    result = trace_provenance(Path(args.config), Path(args.ckpt), Path(args.benchmark))
    data, prompt = agent.build_trace_prompt(result)
    write_rule_outputs(Path(args.out), "trace", result, trace_markdown(result), data, prompt, args.dry_run, agent)


def run_compare(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    result = compare_runs(Path(args.a), Path(args.b))
    data, prompt = agent.build_compare_prompt(result)
    write_rule_outputs(Path(args.out), "compare", result, compare_markdown(result), data, prompt, args.dry_run, agent)


def run_case(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    result = case_analysis(Path(args.pred), top_bad=args.top_bad, top_good=args.top_good)
    out = Path(args.out)
    data, prompt = agent.build_case_prompt(result)
    write_rule_outputs(out, "case_summary", result, case_markdown(result), data, prompt, args.dry_run, agent)
    with (out / "bad_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in result.get("bad_cases", []):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "good_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in result.get("good_cases", []):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def doctor_result(run_dir: Path, config: Path) -> dict[str, Any]:
    inspect_result = inspect_run_artifacts(run_dir)
    trace_result = trace_provenance(config, run_dir / "best.pt", run_dir / "benchmark.json")
    cases_result = case_analysis(run_dir / "predictions.jsonl")
    ratio = inspect_result.get("pair_count_ratio")
    return {
        "inspect": inspect_result,
        "trace": trace_result,
        "cases": {
            "rows": cases_result.get("rows"),
            "missing_fields": cases_result.get("missing_fields"),
            "reason_counts": dict(cases_result.get("reason_counts", {})),
            "avg_pair_f1": cases_result.get("avg_pair_f1"),
        },
        "should_run_more_experiments": "yes" if inspect_result.get("warnings") or trace_result.get("warnings") else "no",
        "should_change_model": "yes" if ratio is not None and not (0.5 <= float(ratio) <= 1.5) else "no",
        "should_update_paper_table": "no" if trace_result.get("provenance_mismatch_risk") else "yes",
    }


def run_doctor(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run, model=getattr(args, "model", None), timeout=getattr(args, "timeout", 60), max_retries=getattr(args, "max_retries", 2))
    out = Path(args.out)
    result = doctor_result(Path(args.run), Path(args.config))
    write_json(out / "inspect" / "inspect.json", result["inspect"])
    write_markdown(out / "inspect" / "inspect.md", inspect_markdown(result["inspect"]))
    write_json(out / "trace" / "trace.json", result["trace"])
    write_markdown(out / "trace" / "trace.md", trace_markdown(result["trace"]))
    write_json(out / "cases" / "case_summary.json", result["cases"])
    data, prompt = agent.build_doctor_prompt(result)
    lines = [
        "# Doctor Report",
        "",
        f"- run health: {result['inspect'].get('status')}",
        f"- inference provenance: {result['trace'].get('status')}",
        f"- sample failure modes: {result['cases'].get('reason_counts')}",
        f"- should_run_more_experiments: {result['should_run_more_experiments']}",
        f"- should_change_model: {result['should_change_model']}",
        f"- should_update_paper_table: {result['should_update_paper_table']}",
    ]
    write_rule_outputs(out, "doctor", result, lines, data, prompt, args.dry_run, agent)


def normalize_root(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def safe_root(root: Path) -> bool:
    resolved = normalize_root(root)
    allowed = [
        ROOT / "outputs" / "llm",
        ROOT / "outputs" / "llm_shell",
        ROOT / "outputs" / "llm_shell_test",
        ROOT / "outputs" / "llm_test",
    ]
    blocked = [ROOT / item for item in ["dataset", "config", "models", "scripts", "release", "docs", ".git"]]
    return any(resolved == item.resolve() or item.resolve() in resolved.parents for item in allowed) and not any(
        resolved == item.resolve() or item.resolve() in resolved.parents for item in blocked
    )


def cleanup_reports(root: Path, keep: int = 10, dry_run: bool = False) -> dict[str, Any]:
    if not safe_root(root):
        report = {"root": str(root), "keep": keep, "dry_run": dry_run, "status": "blocked", "errors": ["unsafe root"], "kept_dirs": [], "removed_dirs": []}
        return report
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
        "keep": keep,
        "total_dirs": len(ordered),
        "kept_dirs": [str(item) for item in kept],
        "removed_dirs": [str(item) for item in removed],
        "dry_run": dry_run,
        "errors": errors,
        "status": "PASS" if not errors else "WARN",
    }
    write_json(root / "cleanup_report.json", report)
    write_markdown(root / "cleanup_report.md", [
        "# Cleanup Report",
        "",
        f"- root: {root}",
        f"- keep: {keep}",
        f"- total_dirs: {len(ordered)}",
        f"- kept: {len(kept)}",
        f"- removed: {len(removed)}",
        f"- dry_run: {dry_run}",
        f"- errors: {len(errors)}",
    ])
    return report


def run_cleanup(args: argparse.Namespace) -> None:
    report = cleanup_reports(Path(args.root), keep=args.keep, dry_run=args.dry_run)
    if report["status"] == "blocked":
        print(f"cleanup blocked -> {args.root}")
        raise SystemExit("Cleanup root is unsafe.")
    print(f"cleanup {report['status']} -> {Path(args.root) / 'cleanup_report.md'}")


def make_usage(state: dict[str, Any] | None, out: Path) -> dict[str, Any]:
    if state is None:
        env = load_env()
        data = {
            "mode": "standalone",
            "api_calls": 0,
            "estimated_tokens": 0,
            "max_api_calls": None,
            "max_tokens_total": None,
            "repeated_prompt_count": 0,
            "blocked_count": 0,
            "error_count": 0,
            "live_api_enabled": bool(env.get("LLM_API_KEY")),
            "env_loaded": (ROOT / ".env").exists(),
            "model": env.get("LLM_MODEL"),
        }
    else:
        data = {
            **shell_status(state),
            "repeated_prompt_count": max(state["prompt_hash_counts"].values()) if state["prompt_hash_counts"] else 0,
            "blocked_count": state["blocked_count"],
            "error_count": state["error_count"],
            "live_api_enabled": state["mode"] == "live" and not state["no_api"],
            "env_loaded": (ROOT / ".env").exists(),
            "model": state.get("model"),
        }
    write_json(out / "usage.json", data)
    write_markdown(out / "usage.md", ["# Agent Usage", "", *[f"- {key}: {value}" for key, value in data.items()]])
    return data


def run_usage(args: argparse.Namespace) -> None:
    out = Path(args.out)
    make_usage(None, out)
    print(f"usage -> {out / 'usage.md'}")


def path_tokens(text: str) -> list[str]:
    return re.findall(r"(?:outputs|config|release|dataset|docs)[A-Za-z0-9_./\\-]*(?:\.yaml|\.yml|\.md|\.jsonl|\.json|\.pt)?", text)


def parse_agent_command(raw: str) -> tuple[str, list[str], str | None]:
    text = raw.strip()
    if not text:
        return "empty", [], None
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    prefix_mode = None
    if parts and parts[0].lower() in {"dry", "live"}:
        prefix_mode = parts[0].lower()
        parts = parts[1:]
    if not parts:
        return "empty", [], prefix_mode
    command = parts[0].lower()
    direct = {"diagnose", "inspect", "trace", "compare", "case", "doctor", "schedule", "report", "auditdata"}
    if command in direct:
        return command, parts[1:], prefix_mode
    paths = path_tokens(text)
    lower = text.lower()
    if "smoke" in lower or "\u8fd0\u884c smoke" in lower or "\u6d4b\u8bd5\u57fa\u672c\u6d41\u7a0b" in lower:
        return "safe_smoke", [], prefix_mode
    if "clean audit" in lower or "\u8fd0\u884c audit" in lower or "\u8fd0\u884c\u5ba1\u8ba1" in lower or "\u6e05\u7406\u68c0\u67e5" in lower:
        return "safe_audit", [], prefix_mode
    if "\u7f16\u8bd1 agent" in lower or "\u68c0\u67e5 llm.py" in lower or "py_compile" in lower:
        return "safe_compile", [], prefix_mode
    if "cleanup" in lower or "\u6e05\u7406\u65e7\u62a5\u544a" in lower or "\u4fdd\u7559" in lower:
        keep = next((int(item) for item in re.findall(r"\d+", text)), 10)
        return "cleanup", [str(keep)], prefix_mode
    if "benchmark" in lower or "\u8dd1 benchmark" in lower:
        return "benchmark_candidate", [], prefix_mode
    if "train" in lower or "\u8bad\u7ec3" in lower:
        return "train_candidate", [], prefix_mode
    if "doctor" in lower or "\u7efc\u5408\u8bca\u65ad" in lower or "\u4e00\u952e\u8bca\u65ad" in lower:
        return "doctor", paths[:2] or ["outputs/candidate", "config/candidate.yaml"], prefix_mode
    if "inspect" in lower or "\u68c0\u67e5 candidate" in lower or "\u4f53\u68c0 candidate" in lower or "\u68c0\u67e5\u8bad\u7ec3" in lower:
        return "inspect", paths[:1] or ["outputs/candidate"], prefix_mode
    if "trace" in lower or "\u8ffd\u8e2a" in lower or "\u63a8\u7406\u94fe\u8def" in lower:
        return "trace", paths[:3], prefix_mode
    if "compare" in lower or "\u5bf9\u6bd4" in lower:
        return "compare", paths[:2] or ["outputs/candidate", "outputs/oldbase"], prefix_mode
    if "case" in lower or "\u9519\u8bef\u6837\u672c" in lower or "\u6837\u672c\u5206\u6790" in lower:
        return "case", paths[:1] or ["outputs/candidate/predictions.jsonl"], prefix_mode
    if "diagnose" in lower or "\u8bca\u65ad" in lower:
        return "diagnose", paths[:1] or ["outputs/candidate"], prefix_mode
    if "schedule" in lower or "\u8ba1\u5212" in lower or "\u8c03\u5ea6" in lower or "\u4e0b\u4e00\u6b65" in lower:
        return "schedule", paths[:2] or ["outputs/candidate", "config/candidate.yaml"], prefix_mode
    if "report" in lower or "\u62a5\u544a" in lower or "\u603b\u7ed3" in lower or "\u6574\u7406" in lower:
        return "report", paths or ["release/model_card.md", "release/results_summary.md", "release/limitations.md"], prefix_mode
    if "auditdata" in lower or "\u6570\u636e\u5ba1\u8ba1" in lower or "\u5ba1\u8ba1" in lower:
        return "auditdata", paths or ["dataset/archive/train.jsonl", "dataset/archive/test.jsonl"], prefix_mode
    return "unknown", [], prefix_mode


def block_reason(raw: str, command: str) -> str | None:
    lower = raw.lower()
    if command in {"train_candidate", "benchmark_candidate"}:
        return None
    blocked = [
        "git push", "git commit", "git reset", "git checkout", "git clean",
        "rm ", "del ", "remove ", "mv ", "cp ", "overwrite",
        "\u4fee\u6539\u914d\u7f6e", "\u5220\u9664", "\u8986\u76d6",
        ".env", "api_key", "llm_api_key", "cuda_visible_devices",
        "pip install", "conda install", "curl ", "wget ",
    ]
    if command not in {"trace", "case", "doctor", "inspect", "compare", "diagnose", "report"}:
        blocked.extend(["benchmark.json", "predictions.jsonl", "best.pt", "checkpoint"])
    for item in blocked:
        if item in lower:
            return item
    return None


def run_subprocess_summary(cmd: list[str], timeout: int) -> tuple[str, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        return "FAIL", "\n".join(output.splitlines()[-20:])
    if "smoke_ok" in output:
        return "PASS", "smoke_ok"
    if "clean PASS" in output:
        return "PASS", "clean PASS"
    return "PASS", "\n".join(output.splitlines()[-5:])


def concise_print(state: dict[str, Any], plan: list[str], result: list[str], suggestion: str, extra: list[str] | None = None) -> None:
    ui = state.get("ui_mode", "normal")
    if ui == "verbose" and extra:
        print("Debug:")
        for item in extra:
            print(f"- {item}")
    if ui == "normal":
        print("Plan:")
        for item in plan[:3]:
            print(f"- {item}")
        print("\nRunning...")
    print("Result:")
    for item in result[:5]:
        print(f"- {item}")
    print("\nSuggestion:")
    print(f"- {suggestion}")


def write_history(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def shell_prompt(agent: RNAAnalysisAgent, command: str, args: list[str]) -> tuple[dict[str, Any], str]:
    if command == "diagnose":
        return agent.build_diagnose_prompt(Path(args[0]))
    if command == "inspect":
        return agent.build_inspect_prompt(inspect_run_artifacts(Path(args[0])))
    if command == "trace":
        return agent.build_trace_prompt(trace_provenance(Path(args[0]), Path(args[1]), Path(args[2])))
    if command == "compare":
        return agent.build_compare_prompt(compare_runs(Path(args[0]), Path(args[1])))
    if command == "case":
        return agent.build_case_prompt(case_analysis(Path(args[0])))
    if command == "doctor":
        return agent.build_doctor_prompt(doctor_result(Path(args[0]), Path(args[1])))
    if command == "schedule":
        return agent.build_schedule_prompt(Path(args[0]), Path(args[1]))
    if command == "report":
        return agent.build_report_prompt([Path(item) for item in args])
    if command == "auditdata":
        return agent.build_auditdata_prompt([Path(item) for item in args])
    raise ValueError("Could not parse command. Type /help for examples.")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def guard_live_call(state: dict[str, Any], prompt: str, turn_dir: Path) -> tuple[bool, str | None]:
    ph = prompt_hash(prompt)
    repeated = state["prompt_hash_counts"].get(ph, 0)
    estimated = max(1, len(prompt) // 4)
    reason = None
    if state["api_calls"] >= state["max_api_calls"]:
        reason = "max_api_calls"
    elif state["estimated_tokens"] + estimated > state["max_tokens_total"]:
        reason = "max_tokens_total"
    elif repeated >= state["max_same_prompt"]:
        reason = "same_prompt_repeated"
    elif state["turn"] >= state["max_turns"]:
        reason = "max_turns"
    if reason:
        data = {
            "stop_reason": reason,
            "api_calls": state["api_calls"],
            "estimated_tokens": state["estimated_tokens"],
            "prompt_hash": ph,
            "repeated_count": repeated,
            "last_command": state.get("last_command"),
            "recommended_action": "Switch to /dry, reduce input files, or start a new shell.",
        }
        write_json(turn_dir / "limit_stop.json", data)
        write_markdown(turn_dir / "limit_stop.md", ["# Agent Limit Stop", "", *[f"- {k}: {v}" for k, v in data.items()]])
        return False, reason
    state["prompt_hash_counts"][ph] = repeated + 1
    state["api_calls"] += 1
    state["estimated_tokens"] += estimated
    return True, None


def shell_status(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": state["mode"],
        "safety": "read-only",
        "api_calls": state["api_calls"],
        "max_api_calls": state["max_api_calls"],
        "estimated_tokens": state["estimated_tokens"],
        "max_tokens_total": state["max_tokens_total"],
        "turns": state["turn"],
        "max_turns": state["max_turns"],
        "out": str(state["out"]),
        "history": str(state["history"]),
        "last_command": state.get("last_command"),
        "last_status": state.get("last_status"),
        "ui_mode": state["ui_mode"],
        "pending_confirmation": state.get("pending_confirmation"),
    }


def execute_confirmed_train(state: dict[str, Any]) -> tuple[str, str]:
    if state["mode"] == "dry_run":
        return "DRY_RUN", "planned: python main.py train --config config/candidate.yaml --device cuda"
    return run_subprocess_summary(TRAIN_COMMAND, timeout=max(60, int(state["timeout"]) * 20))


def default_state(args: argparse.Namespace, out: Path, history: Path) -> dict[str, Any]:
    return {
        "out": out,
        "history": history,
        "mode": "dry_run" if args.dry_run or args.no_api else "live",
        "no_api": bool(args.no_api),
        "turn": 0,
        "max_turns": int(args.max_turns),
        "api_calls": 0,
        "max_api_calls": int(args.max_api_calls),
        "estimated_tokens": 0,
        "max_tokens_total": int(args.max_tokens_total),
        "prompt_hash_counts": {},
        "max_same_prompt": int(args.max_same_prompt),
        "timeout": int(args.timeout),
        "max_retries": int(args.max_retries),
        "blocked_count": 0,
        "error_count": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "last_command": None,
        "last_status": None,
        "pending_confirmation": None,
        "confirmed_commands": [],
        "ui_mode": "normal",
        "model": args.model,
    }


def execute_input(raw: str, state: dict[str, Any], agent: RNAAnalysisAgent, echo: bool = True) -> bool:
    raw = raw.strip()
    if not raw:
        return True
    state["turn"] += 1
    turn = state["turn"]
    turn_dir = state["out"] / "turns" / f"{turn:04d}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    status = "ok"
    command = None
    args: list[str] = []

    def record() -> None:
        state["last_command"] = command or raw
        state["last_status"] = status
        if status == "blocked":
            state["blocked_count"] += 1
        if status == "error":
            state["error_count"] += 1
        write_history(state["history"], {
            "turn": turn,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": state["mode"],
            "raw_input": raw,
            "parsed_command": command,
            "args": args,
            "status": status,
            "out_dir": str(turn_dir),
            "pending_confirmation": state.get("pending_confirmation"),
        })

    pending = state.get("pending_confirmation")
    if pending:
        lowered = raw.lower()
        if pending["intent"] == "train_candidate" and lowered in CONFIRM_TRAIN:
            command = "confirmed_train_candidate"
            status_text, summary = execute_confirmed_train(state)
            state["confirmed_commands"].append({"intent": pending["intent"], "command": pending["command"], "confirmed_by_user": True})
            state["pending_confirmation"] = None
            concise_print(state, ["Confirmed training command.", "Config: config/candidate.yaml"], [f"{status_text}: {summary}"], "Inspect the run directory before any benchmark.")
            record()
            return True
        if pending["intent"] == "benchmark_candidate" and lowered in CONFIRM_BENCH:
            command = "blocked_benchmark_candidate"
            status = "blocked"
            state["pending_confirmation"] = None
            message = "No safe benchmark command was found. Please run the documented benchmark command manually after confirming checkpoint and split."
            write_markdown(turn_dir / "blocked.md", [message])
            concise_print(state, ["Benchmark confirmation received."], [message], "Use trace before manual benchmark.")
            record()
            return True
        state["pending_confirmation"] = None

    slash = raw.lower()
    if slash in {"/exit", "/quit"}:
        command = "exit"
        record()
        return False
    if slash == "/help":
        command = "help"
        if echo:
            print(HELP_TEXT)
        record()
        return True
    if slash == "/status":
        command = "status"
        print(json.dumps(shell_status(state), indent=2, ensure_ascii=False))
        record()
        return True
    if slash == "/usage":
        command = "usage"
        make_usage(state, state["out"])
        concise_print(state, ["Write usage report."], [f"usage -> {state['out'] / 'usage.md'}"], "Switch to /dry if API usage is high.")
        record()
        return True
    if slash.startswith("/cleanup"):
        command = "cleanup"
        keep = int(re.findall(r"\d+", raw)[0]) if re.findall(r"\d+", raw) else 10
        report = cleanup_reports(state["out"], keep=keep, dry_run=False)
        concise_print(state, [f"Cleanup reports under {state['out']}."], [f"{report['status']}: kept {len(report['kept_dirs'])}, removed {len(report['removed_dirs'])}"], "Run /usage to review remaining session state.")
        record()
        return True
    if slash == "/mode":
        command = "mode"
        print(state["mode"])
        record()
        return True
    if slash == "/dry":
        command = "dry"
        state["mode"] = "dry_run"
        print("Mode: dry-run")
        record()
        return True
    if slash == "/live":
        command = "live"
        env = load_env()
        if state["no_api"] or not (ROOT / ".env").exists() or not env.get("LLM_API_KEY"):
            status = "blocked"
            print("Cannot switch to live mode: .env or LLM_API_KEY is missing.")
        else:
            state["mode"] = "live"
            print("Mode: live")
        record()
        return True
    if slash == "/quiet":
        command = "quiet"
        state["ui_mode"] = "quiet"
        print("UI: quiet")
        record()
        return True
    if slash == "/normal":
        command = "normal"
        state["ui_mode"] = "normal"
        print("UI: normal")
        record()
        return True
    if slash == "/verbose":
        command = "verbose"
        state["ui_mode"] = "verbose"
        print("UI: verbose")
        record()
        return True
    if slash == "/history":
        command = "history"
        if state["history"].exists():
            print("\n".join(state["history"].read_text(encoding="utf-8", errors="replace").splitlines()[-20:]))
        record()
        return True
    if slash == "/clear":
        command = "clear"
        state["last_command"] = None
        state["last_status"] = None
        state["pending_confirmation"] = None
        print("Cleared in-memory shell state.")
        record()
        return True

    try:
        command, args, prefix_mode = parse_agent_command(raw)
        if prefix_mode:
            state["mode"] = "dry_run" if prefix_mode == "dry" else "live"
        reason = block_reason(raw, command)
        if reason:
            status = "blocked"
            message = "Blocked by Agent safety policy. This shell is read-only and cannot run training, benchmark, git, deletion, or config-modifying commands."
            write_markdown(turn_dir / "blocked.md", [message, f"Matched: {reason}"])
            concise_print(state, ["Safety check."], [message], "Use a supported read-only diagnostic command.")
        elif command == "train_candidate":
            status = "blocked"
            state["pending_confirmation"] = {
                "intent": "train_candidate",
                "command": "python main.py train --config config/candidate.yaml --device cuda",
                "risk": "writes_outputs_and_checkpoints",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "expires_after_turns": 1,
            }
            message = "Training is blocked by default. To proceed, reply exactly: \u8fdb\u884c\u8bad\u7ec3"
            write_markdown(turn_dir / "blocked.md", [message, "Planned command: python main.py train --config config/candidate.yaml --device cuda"])
            concise_print(state, ["Training request detected."], [message], "Use inspect before training if unsure.")
        elif command == "benchmark_candidate":
            status = "blocked"
            state["pending_confirmation"] = {
                "intent": "benchmark_candidate",
                "command": "python scripts/eval.py bench --config config/candidate.yaml",
                "risk": "updates_metrics",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "expires_after_turns": 1,
            }
            message = "Benchmark execution is blocked by default. To proceed, reply exactly: \u8fdb\u884c benchmark"
            write_markdown(turn_dir / "blocked.md", [message])
            concise_print(state, ["Benchmark request detected."], [message], "Run trace before manual benchmark.")
        elif command == "safe_smoke":
            code, summary = run_subprocess_summary(SAFE_SMOKE, timeout=120)
            concise_print(state, ["Run safe command: python main.py smoke"], [f"{code}: {summary}"], "Run clean audit before committing.")
        elif command == "safe_audit":
            code, summary = run_subprocess_summary(SAFE_AUDIT, timeout=120)
            concise_print(state, ["Run safe command: python scripts/audit.py clean --out outputs/clean"], [f"{code}: {summary}"], "Repository is clean enough for documentation or commit." if code == "PASS" else "Fix audit warnings before committing.")
        elif command == "safe_compile":
            code, summary = run_subprocess_summary(SAFE_COMPILE, timeout=120)
            concise_print(state, ["Compile Agent files."], [f"{code}: {summary}"], "Run agent_test next.")
        elif command == "cleanup":
            keep = int(args[0]) if args else 10
            report = cleanup_reports(state["out"], keep=keep, dry_run=False)
            concise_print(state, [f"Cleanup reports under {state['out']}."], [f"{report['status']}: removed {len(report['removed_dirs'])}"], "Run /usage to verify state.")
        elif command in {"diagnose", "inspect", "trace", "compare", "case", "doctor", "schedule", "report", "auditdata"}:
            data, prompt = shell_prompt(agent, command, args)
            write_json(turn_dir / "request.json", {**data, "prompt": prompt})
            write_markdown(turn_dir / "prompt.md", [prompt])
            if state["mode"] == "live":
                ok, stop = guard_live_call(state, prompt, turn_dir)
                if not ok:
                    status = "blocked"
                    concise_print(state, ["Check API/cycle guard."], [f"Agent stopped this request because API/cycle guard was triggered. Reason: {stop}", f"See: {turn_dir / 'limit_stop.md'}"], "Switch to /dry or reduce input files.")
                else:
                    response = agent.call(prompt)
                    write_markdown(turn_dir / "response.md", [response])
                    concise_print(state, [f"Run {command} analysis."], [f"response -> {turn_dir / 'response.md'}"], "Review the report before changing experiments.")
            else:
                concise_print(state, [f"Build {command} prompt."], [f"prompt -> {turn_dir / 'prompt.md'}"], "Switch to /live only if you need an API summary.")
        else:
            status = "error"
            message = "Could not parse command. Type /help for examples."
            write_markdown(turn_dir / "error.md", [message])
            print(message)
    except subprocess.TimeoutExpired:
        status = "error"
        message = "Detected possible loop or stalled request. Reason: command timeout."
        write_markdown(turn_dir / "loop_detected.md", [message, f"Last command: {raw}"])
        concise_print(state, ["Run command with timeout guard."], [message], "Use /status and retry a smaller command.")
    except Exception as exc:
        status = "error"
        write_markdown(turn_dir / "error.md", [str(exc)])
        concise_print(state, ["Parse and execute request."], [str(exc)], "Check command arguments or use /help.")
    if state["api_calls"] >= 0.8 * state["max_api_calls"]:
        print("API call usage is high.")
    if state["estimated_tokens"] >= 0.8 * state["max_tokens_total"]:
        print("Estimated token usage is high.")
    record()
    return True


def run_agent(args: argparse.Namespace) -> None:
    out = Path(args.out)
    history = Path(args.history) if args.history else out / "history.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    state = default_state(args, out, history)
    agent = RNAAnalysisAgent(dry_run=False, model=args.model, timeout=args.timeout, max_retries=args.max_retries)
    print("RNA-OmniDiffusion Agent Shell")
    print(f"Mode: {'dry-run' if state['mode'] == 'dry_run' else 'live'}")
    print("Safety: read-only")
    print("Type /help for commands, /exit to quit.")
    while True:
        try:
            raw = input("agent> ")
        except EOFError:
            print()
            break
        if not execute_input(raw, state, agent):
            break


def run_agent_test(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists():
        for path in sorted(out.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        dry_run=True,
        no_api=True,
        model=None,
        max_turns=100,
        max_api_calls=20,
        max_tokens_total=20000,
        max_same_prompt=2,
        timeout=60,
        max_retries=2,
    )
    state = default_state(ns, out, out / "history.jsonl")
    agent = RNAAnalysisAgent(dry_run=False)
    commands = [
        "/status",
        "/usage",
        "\u8fd0\u884c smoke",
        "\u8fd0\u884c audit",
        "\u7f16\u8bd1 agent",
        "\u68c0\u67e5 candidate",
        "\u7efc\u5408\u8bca\u65ad",
        "\u6e05\u7406\u65e7\u62a5\u544a\uff0c\u53ea\u4fdd\u755910\u6b21",
        "\u8bad\u7ec3 candidate",
        "\u8fdb\u884c\u8bad\u7ec3",
        "\u8dd1 benchmark",
        "\u8fdb\u884c benchmark",
        "git push origin main",
        "/live",
        "/dry",
        "/quiet",
        "/status",
        "/normal",
        "/cleanup 10",
        "/exit",
    ]
    for item in commands:
        if not execute_input(item, state, agent, echo=False):
            break
    history = [json.loads(line) for line in (out / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if not (out / "usage.json").exists():
        raise SystemExit("agent_test failed: usage.json missing.")
    if not (out / "cleanup_report.md").exists():
        raise SystemExit("agent_test failed: cleanup_report.md missing.")
    if not any(item.get("parsed_command") == "safe_smoke" and item.get("status") == "ok" for item in history):
        raise SystemExit("agent_test failed: smoke was not executed.")
    if not any(item.get("parsed_command") == "confirmed_train_candidate" for item in history):
        raise SystemExit("agent_test failed: confirmed train gate was not exercised.")
    if not any(item.get("status") == "blocked" for item in history):
        raise SystemExit("agent_test failed: no blocked command recorded.")
    print(f"agent_test PASS -> {out}")


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional read-only LLM analysis utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--run", required=True)
    diagnose.add_argument("--out", default="outputs/llm/diagnose")
    add_common_llm_args(diagnose)
    diagnose.set_defaults(func=run_diagnose)

    schedule = sub.add_parser("schedule")
    schedule.add_argument("--run", required=True)
    schedule.add_argument("--config", required=True)
    schedule.add_argument("--out", default="outputs/llm/schedule")
    add_common_llm_args(schedule)
    schedule.set_defaults(func=run_schedule)

    report = sub.add_parser("report")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--out", default="outputs/llm/report")
    report.add_argument("--max_chars", type=int, default=20000)
    add_common_llm_args(report)
    report.set_defaults(func=run_report)

    auditdata = sub.add_parser("auditdata")
    auditdata.add_argument("--inputs", nargs="+", required=True)
    auditdata.add_argument("--out", default="outputs/llm/data")
    auditdata.add_argument("--max_rows", type=int, default=None)
    add_common_llm_args(auditdata)
    auditdata.set_defaults(func=run_auditdata)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--run", required=True)
    inspect.add_argument("--out", default="outputs/llm/inspect")
    add_common_llm_args(inspect)
    inspect.set_defaults(func=run_inspect)

    trace = sub.add_parser("trace")
    trace.add_argument("--config", required=True)
    trace.add_argument("--ckpt", required=True)
    trace.add_argument("--benchmark", required=True)
    trace.add_argument("--out", default="outputs/llm/trace")
    add_common_llm_args(trace)
    trace.set_defaults(func=run_trace)

    compare = sub.add_parser("compare")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--out", default="outputs/llm/compare")
    add_common_llm_args(compare)
    compare.set_defaults(func=run_compare)

    case = sub.add_parser("case")
    case.add_argument("--pred", required=True)
    case.add_argument("--out", default="outputs/llm/cases")
    case.add_argument("--top_bad", type=int, default=20)
    case.add_argument("--top_good", type=int, default=20)
    add_common_llm_args(case)
    case.set_defaults(func=run_case)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--run", required=True)
    doctor.add_argument("--config", required=True)
    doctor.add_argument("--out", default="outputs/llm/doctor")
    add_common_llm_args(doctor)
    doctor.set_defaults(func=run_doctor)

    usage = sub.add_parser("usage")
    usage.add_argument("--out", default="outputs/llm/usage")
    usage.set_defaults(func=run_usage)

    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--root", default="outputs/llm_shell")
    cleanup.add_argument("--keep", type=int, default=10)
    cleanup.add_argument("--dry_run", action="store_true")
    cleanup.set_defaults(func=run_cleanup)

    agent = sub.add_parser("agent")
    agent.add_argument("--dry_run", action="store_true")
    agent.add_argument("--out", default="outputs/llm_shell")
    agent.add_argument("--history", default=None)
    agent.add_argument("--no_api", action="store_true")
    agent.add_argument("--model", default=None)
    agent.add_argument("--max_api_calls", type=int, default=int(os.environ.get("LLM_MAX_API_CALLS", "20")))
    agent.add_argument("--max_tokens_total", type=int, default=int(os.environ.get("LLM_MAX_TOKENS_TOTAL", "20000")))
    agent.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "60")))
    agent.add_argument("--max_turns", type=int, default=int(os.environ.get("LLM_MAX_TURNS", "100")))
    agent.add_argument("--max_retries", type=int, default=int(os.environ.get("LLM_MAX_RETRIES", "2")))
    agent.add_argument("--max_same_prompt", type=int, default=int(os.environ.get("LLM_MAX_SAME_PROMPT", "2")))
    agent.add_argument("--max_idle_seconds", type=int, default=int(os.environ.get("LLM_MAX_IDLE_SECONDS", "120")))
    agent.set_defaults(func=run_agent)

    agent_test = sub.add_parser("agent_test")
    agent_test.add_argument("--out", default="outputs/llm_shell_test")
    agent_test.set_defaults(func=run_agent_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
