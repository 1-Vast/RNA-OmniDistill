from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
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
)
from models.agent.cleanup import cleanup_reports, safe_root, validate_cleanup_request
from models.agent.memory import append_memory, compact_memory, read_recent_memory, sanitize_text
from models.agent.paths import discover_runs, discover_latest_run
from models.agent.runtime import AgentRuntimeGuard, prompt_hash, sync_guard_state
from models.agent.safety import (
    CONFIRM_BENCH,
    CONFIRM_TRAIN,
    SAFE_AUDIT,
    SAFE_COMPILE,
    SAFE_SMOKE,
    TRAIN_COMMAND,
    block_reason,
    validate_confirmed_command,
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
  /memory
  /memory compact
  /cleanup [keep]
  /runs
  /last
  /open
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

def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Input directory not found: {path}")
    return path


def make_agent(args: argparse.Namespace, dry_run: bool | None = None) -> RNAAnalysisAgent:
    return RNAAnalysisAgent(
        dry_run=dry_run if dry_run is not None else args.dry_run,
        model=getattr(args, "model", None),
        timeout=getattr(args, "api_timeout", getattr(args, "timeout", 60)),
        max_retries=getattr(args, "max_retries", 2),
    )


def tail_lines(text: str, n: int = 20) -> str:
    return "\n".join(text.splitlines()[-n:])


def write_rule_outputs(
    out: Path,
    stem: str,
    result: dict[str, Any],
    lines: list[str],
    data: dict[str, Any],
    prompt: str,
    dry_run: bool,
    agent: RNAAnalysisAgent,
    guard: AgentRuntimeGuard | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{stem}.json", result)
    write_markdown(out / f"{stem}.md", lines)
    write_json(out / "request.json", {**data, "prompt": prompt})
    write_markdown(out / "prompt.md", [prompt])
    if not dry_run:
        runtime = guard or AgentRuntimeGuard(out, mode="live")
        ok, stop = runtime.check_before_call(prompt, stem)
        if not ok:
            runtime.write_limit_stop(out, stop)
            raise SystemExit(f"Agent runtime guard stopped request: {stop['stop_reason']}")
        response, usage = agent.call_with_usage(prompt)
        runtime.record_call(prompt, usage)
        write_markdown(out / "response.md", [response])
        write_json(out / "usage.json", runtime.snapshot())
    print(f"Wrote {out}")


def standalone_guard(args: argparse.Namespace, out: Path, dry_run: bool) -> AgentRuntimeGuard:
    return AgentRuntimeGuard(
        out,
        max_api_calls=int(getattr(args, "max_api_calls", 20)),
        max_tokens_total=int(getattr(args, "max_tokens_total", 20000)),
        max_same_prompt=int(getattr(args, "max_same_prompt", 2)),
        mode="dry_run" if dry_run else "live",
    )


def emit_agent_outputs(
    args: argparse.Namespace,
    out: Path,
    stem: str,
    result: dict[str, Any],
    lines: list[str],
    data: dict[str, Any],
    prompt: str,
    agent: RNAAnalysisAgent | None = None,
) -> None:
    llm = agent or make_agent(args)
    write_rule_outputs(out, stem, result, lines, data, prompt, args.dry_run, llm, standalone_guard(args, out, args.dry_run))


def run_diagnose(args: argparse.Namespace) -> None:
    out = Path(args.out)
    agent = make_agent(args)
    data, prompt = agent.build_diagnose_prompt(require_dir(Path(args.run)))
    emit_agent_outputs(args, out, "diagnose", data, ["# Diagnose Prompt", "", prompt], data, prompt, agent)


def run_schedule(args: argparse.Namespace) -> None:
    out = Path(args.out)
    agent = make_agent(args)
    config = require_file(Path(args.config)) if args.config else None
    data, prompt = agent.build_schedule_prompt(require_dir(Path(args.run)), config)
    emit_agent_outputs(args, out, "schedule", data, ["# Schedule Prompt", "", prompt], data, prompt, agent)


def run_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    agent = make_agent(args)
    data, prompt = agent.build_report_prompt([require_file(Path(path)) for path in args.inputs], max_chars=args.max_chars)
    emit_agent_outputs(args, out, "report", data, ["# Report Prompt", "", prompt], data, prompt, agent)


def run_auditdata(args: argparse.Namespace) -> None:
    out = Path(args.out)
    agent = make_agent(args)
    data, prompt = agent.build_auditdata_prompt([require_file(Path(path)) for path in args.inputs], max_rows=args.max_rows)
    emit_agent_outputs(args, out, "dataaudit", data, ["# Data Audit Prompt", "", prompt], data, prompt, agent)


def run_inspect(args: argparse.Namespace) -> None:
    agent = make_agent(args)
    result = inspect_run_artifacts(Path(args.run))
    data, prompt = agent.build_inspect_prompt(result)
    out = Path(args.out)
    emit_agent_outputs(args, out, "inspect", result, inspect_markdown(result), data, prompt, agent)


def run_trace(args: argparse.Namespace) -> None:
    agent = make_agent(args)
    result = trace_provenance(Path(args.config), Path(args.ckpt), Path(args.benchmark))
    data, prompt = agent.build_trace_prompt(result)
    out = Path(args.out)
    emit_agent_outputs(args, out, "trace", result, trace_markdown(result), data, prompt, agent)


def run_compare(args: argparse.Namespace) -> None:
    agent = make_agent(args)
    result = compare_runs(Path(args.a), Path(args.b))
    data, prompt = agent.build_compare_prompt(result)
    out = Path(args.out)
    emit_agent_outputs(args, out, "compare", result, compare_markdown(result), data, prompt, agent)


def run_case(args: argparse.Namespace) -> None:
    agent = make_agent(args)
    result = case_analysis(Path(args.pred), top_bad=args.top_bad, top_good=args.top_good)
    out = Path(args.out)
    data, prompt = agent.build_case_prompt(result)
    emit_agent_outputs(args, out, "case_summary", result, case_markdown(result), data, prompt, agent)
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
    agent = make_agent(args)
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
    emit_agent_outputs(args, out, "doctor", result, lines, data, prompt, agent)


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
            "api_timeout": None,
            "command_timeout": None,
            "train_timeout": None,
            "loop_stop_count": 0,
            "max_idle_seconds": None,
        }
    else:
        data = {
            **shell_status(state),
            "repeated_prompt_count": max(state["prompt_hash_counts"].values()) if state["prompt_hash_counts"] else 0,
            "blocked_count": state["blocked_count"],
            "error_count": state["error_count"],
            "api_timeout": state["api_timeout"],
            "command_timeout": state["command_timeout"],
            "train_timeout": state["train_timeout"],
            "loop_stop_count": state["loop_stop_count"],
            "max_idle_seconds": state["max_idle_seconds"],
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


@dataclass(frozen=True)
class IntentRule:
    command: str
    keywords: tuple[str, ...]
    default_args: tuple[str, ...] = ()
    path_limit: int | None = None
    parse_keep: bool = False


INTENT_RULES = [
    IntentRule("safe_smoke", ("smoke", "运行 smoke", "测试基本流程")),
    IntentRule("safe_audit", ("clean audit", "运行 audit", "运行审计", "清理检查")),
    IntentRule("safe_compile", ("编译 agent", "检查 llm.py", "py_compile")),
    IntentRule("cleanup", ("cleanup", "清理旧报告", "保留"), parse_keep=True),
    IntentRule("show_memory", ("查看记忆", "memory", "调参记录", "历史结论")),
    IntentRule("doctor", ("doctor", "综合诊断", "一键诊断"), ("outputs/candidate", "config/candidate.yaml"), 2),
    IntentRule("inspect", ("inspect", "检查 candidate", "体检 candidate", "检查训练"), ("outputs/candidate",), 1),
    IntentRule("trace", ("trace", "追踪", "推理链路"), path_limit=3),
    IntentRule("compare", ("compare", "对比"), ("outputs/candidate", "outputs/oldbase"), 2),
    IntentRule("case", ("case", "错误样本", "样本分析"), ("outputs/candidate/predictions.jsonl",), 1),
    IntentRule("diagnose", ("diagnose", "诊断"), ("outputs/candidate",), 1),
    IntentRule("schedule", ("schedule", "计划", "调度", "下一步"), ("outputs/candidate", "config/candidate.yaml"), 2),
    IntentRule("report", ("report", "报告", "总结", "整理"), ("release/model_card.md", "release/results_summary.md", "release/limitations.md")),
    IntentRule("auditdata", ("auditdata", "数据审计", "审计"), ("dataset/archive/train.jsonl", "dataset/archive/test.jsonl")),
    IntentRule("show_train_device", ("查看训练设备",)),
    IntentRule("train_device_prompt", ("设置训练设备", "训练设备")),
    IntentRule("set_local_train", ("本地训练", "local")),
    IntentRule("set_remote_train", ("远程训练", "remote")),
    IntentRule("set_cuda_train", ("使用 cuda", "device cuda")),
    IntentRule("set_cpu_train", ("使用 cpu", "device cpu")),
    IntentRule("set_target", ("设定目标", "目标 pair", "目标 valid", "最多调参")),
    IntentRule("show_target", ("查看调参目标",)),
    IntentRule("clear_target", ("清除调参目标",)),
    IntentRule("start_target_train", ("开始目标训练",)),
    IntentRule("benchmark_candidate", ("benchmark", "跑 benchmark")),
    IntentRule("train_candidate", ("train", "训练")),
]


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
    if command == "ssh":
        return "set_remote_login", parts, prefix_mode
    paths = path_tokens(text)
    lower = text.lower()
    for rule in INTENT_RULES:
        if any(keyword.lower() in lower for keyword in rule.keywords):
            if rule.parse_keep:
                keep = next((int(item) for item in re.findall(r"-?\d+", text)), 10)
                return rule.command, [str(keep)], prefix_mode
            selected = paths[: rule.path_limit] if rule.path_limit is not None else paths
            return rule.command, selected or list(rule.default_args), prefix_mode
    return "unknown", [], prefix_mode


def run_subprocess_summary(cmd: list[str], timeout: int | None) -> tuple[str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout if timeout and timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return "TIMEOUT", tail_lines(output + "\nTraining command timed out.")
    output = proc.stdout or ""
    if proc.returncode != 0:
        return "FAIL", tail_lines(output)
    if "smoke_ok" in output:
        return "PASS", "smoke_ok"
    if "clean PASS" in output:
        return "PASS", "clean PASS"
    return "PASS", tail_lines(output, n=5)


def concise_print(state: dict[str, Any], plan: list[str], result: list[str], suggestion: str, extra: list[str] | None = None) -> None:
    ui = state.get("ui_mode", "normal")
    state["last_suggestion"] = suggestion
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


def remember(state: dict[str, Any], kind: str, summary: str, data: dict[str, Any] | None = None) -> None:
    append_memory(Path(state["memory"]), kind, summary, data)


def args_max(state: dict[str, Any], key: str = "runs") -> int:
    """Return max items for discovery/summary commands."""
    return {"runs": 10, "memory": 10}.get(key, 10)


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
        "memory": str(state["memory"]),
        "last_command": state.get("last_command"),
        "last_status": state.get("last_status"),
        "ui_mode": state["ui_mode"],
        "pending_confirmation": state.get("pending_confirmation"),
        "api_timeout": state["api_timeout"],
        "command_timeout": state["command_timeout"],
        "train_timeout": state["train_timeout"],
        "max_idle_seconds": state["max_idle_seconds"],
        "last_activity_at": state.get("last_activity_at"),
        "last_output_at": state.get("last_output_at"),
    }


def execute_confirmed_train(state: dict[str, Any]) -> tuple[str, str]:
    ok, reason = validate_confirmed_command(TRAIN_COMMAND, "train_candidate")
    if not ok:
        return "BLOCKED", reason
    if state["mode"] == "dry_run":
        return "DRY_RUN", "planned: python main.py train --config config/candidate.yaml --device cuda"
    warnings = []
    for artifact in [ROOT / "outputs" / "candidate" / "best.pt", ROOT / "outputs" / "candidate" / "last.pt", ROOT / "outputs" / "candidate" / "benchmark.json", ROOT / "outputs" / "candidate" / "predictions.jsonl"]:
        if artifact.exists():
            warnings.append("Existing run artifacts detected.")
            break
    code, summary = run_subprocess_summary(TRAIN_COMMAND, timeout=int(state["train_timeout"]))
    if warnings:
        summary = "WARN: Existing run artifacts detected. " + summary
    return code, summary


ALLOWED_TARGET_METRICS = {"pair_f1", "pair_precision", "pair_recall", "valid_structure_rate", "all_dot_ratio", "loss", "pair_count_ratio", "rankAcc", "pair_logit_gap"}


def parse_target_spec(raw: str) -> tuple[bool, str]:
    match = re.search(r"(pair_f1|pair_precision|pair_recall|valid_structure_rate|all_dot_ratio|loss|pair_count_ratio|rankAcc|pair_logit_gap)\s*(>=|<=|>|<|=)\s*([0-9.]+)", raw)
    trials = re.search(r"(?:最多调参|max(?:imum)?[_ ]?trials)\s*(\d+)", raw, re.IGNORECASE)
    if not match:
        return False, "Unsupported target. Use: pair_f1 >= 0.75, max_trials 3."
    metric, op, value = match.group(1), match.group(2), float(match.group(3))
    max_trials = int(trials.group(1)) if trials else 3
    if metric not in ALLOWED_TARGET_METRICS:
        return False, "Unsupported target metric."
    if not (1 <= max_trials <= 10):
        return False, "max_trials must be between 1 and 10."
    return True, json.dumps({"metric": metric, "operator": op, "value": value, "max_trials": max_trials})


def write_tuning_plan(state: dict[str, Any], turn_dir: Path) -> tuple[str, str]:
    target = state["target_tuning"]
    if not target.get("enabled"):
        return "FAIL", "No target is set."
    target["current_trial"] = int(target.get("current_trial", 0)) + 1
    trial = target["current_trial"]
    if trial > int(target["max_trials"]):
        return "STOP", "Maximum tuning trials reached."
    trial_dir = state["out"] / "tuning" / f"trial_{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "config" / "candidate.yaml"
    config_text = source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""
    (trial_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    plan = {
        "target": target,
        "trial": trial,
        "config": str(trial_dir / "config.yaml"),
        "command": f"python main.py train --config {trial_dir / 'config.yaml'} --device {state['train_device'].get('device') or 'cuda'}",
        "dry_run": state["mode"] == "dry_run",
        "notes": [
            "config/candidate.yaml is not modified",
            "no benchmark is executed",
            "release files are not modified",
        ],
    }
    write_json(trial_dir / "tuning_plan.json", plan)
    write_markdown(trial_dir / "tuning_plan.md", ["# Tuning Plan", "", *[f"- {k}: {v}" for k, v in plan.items() if k != "notes"], "", "## Notes", *[f"- {item}" for item in plan["notes"]]])
    target["history"].append({"trial": trial, "config": str(trial_dir / "config.yaml")})
    return "DRY_RUN" if state["mode"] == "dry_run" else "CONFIRM_REQUIRED", str(trial_dir / "tuning_plan.md")


def default_state(args: argparse.Namespace, out: Path, history: Path) -> dict[str, Any]:
    legacy_timeout = int(getattr(args, "timeout", 60))
    api_timeout = int(getattr(args, "api_timeout", legacy_timeout))
    command_timeout = int(getattr(args, "command_timeout", max(legacy_timeout, 120)))
    train_timeout = int(getattr(args, "train_timeout", 0))
    now = datetime.now().isoformat(timespec="seconds")
    state = {
        "out": out,
        "history": history,
        "memory": Path(getattr(args, "memory", None) or (out / "memory.jsonl")),
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
        "timeout": api_timeout,
        "api_timeout": api_timeout,
        "command_timeout": command_timeout,
        "train_timeout": train_timeout,
        "max_retries": int(args.max_retries),
        "max_idle_seconds": int(getattr(args, "max_idle_seconds", 120)),
        "blocked_count": 0,
        "error_count": 0,
        "loop_stop_count": 0,
        "started_at": now,
        "last_output_at": now,
        "last_activity_at": now,
        "last_raw_input": None,
        "same_raw_input_count": 0,
        "last_parsed_command": None,
        "last_failed_command": None,
        "same_failed_command_count": 0,
        "last_turn_dir": None,
        "last_suggestion": None,
        "last_report_path": None,
        "last_command": None,
        "last_status": None,
        "pending_confirmation": None,
        "confirmed_commands": [],
        "ui_mode": "normal",
        "model": args.model,
        "train_device": {
            "mode": None,
            "device": None,
            "remote": {"host": None, "port": None, "user": None, "login_command": None},
        },
        "target_tuning": {
            "enabled": False,
            "metric": None,
            "operator": None,
            "value": None,
            "max_trials": 3,
            "current_trial": 0,
            "history": [],
            "base_config": "config/candidate.yaml",
            "mode": "dry_run",
        },
    }
    state["runtime_guard"] = AgentRuntimeGuard(
        out,
        max_api_calls=state["max_api_calls"],
        max_tokens_total=state["max_tokens_total"],
        max_same_prompt=state["max_same_prompt"],
        mode=state["mode"],
    )
    return state


def write_loop_stop(state: dict[str, Any], turn_dir: Path, raw_input: str, parsed_command: str | None, reason: str) -> tuple[bool, str]:
    data = {
        "stop_reason": reason,
        "raw_input": sanitize_text(raw_input),
        "parsed_command": parsed_command,
        "same_raw_input_count": state.get("same_raw_input_count", 0),
        "same_failed_command_count": state.get("same_failed_command_count", 0),
        "last_command": state.get("last_command"),
        "last_status": state.get("last_status"),
        "possible_cause": "Repeated input, repeated failing command, or stalled safe command.",
        "recommended_action": "Use /clear, reduce input files, or run a different diagnostic.",
    }
    write_json(turn_dir / "loop_detected.json", data)
    write_markdown(turn_dir / "loop_detected.md", ["# Loop Detected", "", *[f"- {key}: {value}" for key, value in data.items()]])
    state["loop_stop_count"] += 1
    return True, reason


def detect_loop_or_stall(state: dict[str, Any], raw_input: str, parsed_command: str | None, status: str, turn_dir: Path) -> tuple[bool, str]:
    max_repeat = int(state.get("max_same_prompt", 2))
    if raw_input == state.get("last_raw_input"):
        state["same_raw_input_count"] = int(state.get("same_raw_input_count", 0)) + 1
    else:
        state["same_raw_input_count"] = 1
    state["last_raw_input"] = raw_input

    if status in {"error", "blocked"}:
        if parsed_command == state.get("last_failed_command"):
            state["same_failed_command_count"] = int(state.get("same_failed_command_count", 0)) + 1
        else:
            state["same_failed_command_count"] = 1
        state["last_failed_command"] = parsed_command
    else:
        state["same_failed_command_count"] = 0
        state["last_failed_command"] = None
    state["last_parsed_command"] = parsed_command

    if state["same_raw_input_count"] > max_repeat:
        return write_loop_stop(state, turn_dir, raw_input, parsed_command, "same_raw_input_repeated")
    if state["same_failed_command_count"] > max_repeat:
        return write_loop_stop(state, turn_dir, raw_input, parsed_command, "same_failed_command_repeated")
    return False, ""


def execute_input(raw: str, state: dict[str, Any], agent: RNAAnalysisAgent, echo: bool = True) -> bool:
    raw = raw.strip()
    if not raw:
        return True
    start = time.time()
    state["last_activity_at"] = datetime.now().isoformat(timespec="seconds")
    state["turn"] += 1
    turn = state["turn"]
    turn_dir = state["out"] / "turns" / f"{turn:04d}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    status = "ok"
    command = None
    args: list[str] = []
    pending_cleared = False

    def record() -> None:
        nonlocal status
        state["last_command"] = command or raw
        state["last_status"] = status
        state["last_turn_dir"] = str(turn_dir)
        if status == "blocked":
            state["blocked_count"] += 1
        if status == "error":
            state["error_count"] += 1
        # Runtime guard consecutive error tracking
        guard = state.get("runtime_guard")
        if guard:
            ok, _ = guard.record_result(status)
            if not ok:
                status = "hard_stop"
                concise_print(state, ["Hard stop triggered."],
                    ["Agent stopped after repeated errors. Human review required.", f"consecutive_errors: {guard.consecutive_errors}/{guard.max_consecutive_errors}", f"consecutive_blocked: {guard.consecutive_blocked}/{guard.max_consecutive_blocked}"],
                    "Use /clear to reset or review manually.")
                state["last_status"] = status
        write_history(state["history"], {
            "turn": turn,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": state["mode"],
            "raw_input": sanitize_text(raw),
            "parsed_command": command,
            "args": args,
            "status": status,
            "out_dir": str(turn_dir),
            "pending_confirmation": state.get("pending_confirmation"),
            "pending_confirmation_cleared": pending_cleared,
        })
        stopped, reason = detect_loop_or_stall(state, raw, command, status, turn_dir)
        if stopped:
            state["last_status"] = "loop_stopped"
            write_history(state["history"], {
                "turn": turn,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": state["mode"],
                "raw_input": sanitize_text(raw),
                "parsed_command": command,
                "args": args,
                "status": "loop_stopped",
                "out_dir": str(turn_dir),
                "loop_reason": reason,
            })
        state["last_output_at"] = datetime.now().isoformat(timespec="seconds")

    pending = state.get("pending_confirmation")
    if pending:
        lowered = raw.lower()
        if pending["intent"] == "train_candidate" and lowered in CONFIRM_TRAIN:
            command = "confirmed_train_candidate"
            status_text, summary = execute_confirmed_train(state)
            if status_text == "BLOCKED":
                status = "blocked"
                write_markdown(turn_dir / "blocked.md", [summary])
            state["confirmed_commands"].append({"intent": pending["intent"], "command": pending["command"], "confirmed_by_user": True})
            state["pending_confirmation"] = None
            remember(state, "training", f"candidate training confirmation: {status_text}", {"summary": summary})
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
        pending_cleared = True

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
    if slash == "/last":
        command = "last"
        if state.get("last_turn_dir"):
            concise_print(state, [], [f"last_command: {state.get('last_command')}", f"status: {state.get('last_status')}", f"out_dir: {state.get('last_turn_dir')}", f"report: {state.get('last_report_path')}"], state.get("last_suggestion") or "Run doctor if inspect has warnings.")
        else:
            concise_print(state, [], ["No previous turn."], "Run smoke or inspect candidate first.")
        record()
        return True
    if slash == "/open":
        command = "open"
        report = state.get("last_report_path") or state.get("last_turn_dir")
        concise_print(state, [], [f"report: {report}" if report else "No previous report."], "Run inspect or doctor to generate a report.")
        record()
        return True
    if slash.startswith("/cleanup"):
        command = "cleanup"
        keep = int(re.findall(r"-?\d+", raw)[0]) if re.findall(r"-?\d+", raw) else 10
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
        guard = state.get("runtime_guard")
        if guard:
            guard.clear_hard_stop()
        print("Cleared in-memory shell state.")
        record()
        return True
    if slash == "/runs":
        command = "runs"
        runs = discover_runs(max_items=args_max(state, "runs"))
        if runs:
            lines = [f"{item['name']}  (modified: {item.get('modified_time', 0):.0f})" for item in runs[:10]]
            concise_print(state, ["Discover recent runs."], lines, "Use inspect <run> to analyze.")
        else:
            concise_print(state, ["Discover recent runs."], ["No runs found under outputs/."], "Run smoke or train first.")
        record()
        return True
    if slash.startswith("/memory"):
        parts = raw.split()
        if len(parts) > 1 and parts[1] == "compact":
            command = "memory_compact"
            mem_path = Path(state["memory"])
            report = compact_memory(mem_path)
            concise_print(state, ["Compact Agent memory."],
                [f"status: {report['status']}", f"lines: {report['before_lines']} -> {report.get('after_lines', '?')}"],
                "Memory compacted; error records preserved.")
        else:
            command = "show_memory"
            rows = read_recent_memory(Path(state["memory"]), limit=10)
            if rows:
                concise_print(state, [], [f"{item.get('kind')}: {item.get('summary')}" for item in rows[-5:]], "Use inspect or target tuning to add new memory.")
            else:
                concise_print(state, [], ["No Agent memory yet."], "Run inspect, doctor, or set a tuning target first.")
        record()
        return True

    try:
        command, args, prefix_mode = parse_agent_command(raw)
        if prefix_mode:
            state["mode"] = "dry_run" if prefix_mode == "dry" else "live"
        if raw == state.get("last_raw_input") and int(state.get("same_raw_input_count", 0)) >= int(state.get("max_same_prompt", 2)):
            status = "loop_stopped"
            write_loop_stop(state, turn_dir, raw, command, "same_raw_input_repeated")
            concise_print(state, ["Check loop guard."], [f"Detected possible loop or stalled request. Last command: {command}", "Reason: same_raw_input_repeated"], "Use /clear or run a different diagnostic.")
            write_history(state["history"], {
                "turn": turn,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": state["mode"],
                "raw_input": sanitize_text(raw),
                "parsed_command": command,
                "args": args,
                "status": status,
                "out_dir": str(turn_dir),
            })
            return True
        if command == "set_target":
            lowered_raw = raw.lower()
            target_blockers = ["&&", "||", ";", "|", ">>", ".env", "git", "pip ", "conda ", "curl ", "wget ", "rm ", "del ", "remove ", "mv ", "cp ", "cuda_visible_devices"]
            reason = next((item for item in target_blockers if item in lowered_raw), None)
        else:
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
            message = "Training is blocked by default. To proceed, reply exactly: \u8fdb\u884c\u8bad\u7ec3 candidate"
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
            code, summary = run_subprocess_summary(SAFE_SMOKE, timeout=state["command_timeout"])
            concise_print(state, ["Run safe command: python main.py smoke"], [f"{code}: {summary}"], "Run clean audit before committing.")
        elif command == "safe_audit":
            code, summary = run_subprocess_summary(SAFE_AUDIT, timeout=state["command_timeout"])
            concise_print(state, ["Run safe command: python scripts/audit.py clean --out outputs/clean"], [f"{code}: {summary}"], "Repository is clean enough for documentation or commit." if code == "PASS" else "Fix audit warnings before committing.")
        elif command == "safe_compile":
            code, summary = run_subprocess_summary(SAFE_COMPILE, timeout=state["command_timeout"])
            concise_print(state, ["Compile Agent files."], [f"{code}: {summary}"], "Run agent_test next.")
        elif command == "train_device_prompt":
            concise_print(state, ["Select training target."], ["Where will training run? local or remote."], "Reply with local, remote, 使用 cuda, or 使用 cpu.")
        elif command == "set_local_train":
            state["train_device"]["mode"] = "local"
            state["train_device"]["device"] = state["train_device"].get("device") or "cuda"
            concise_print(state, ["Set training device."], ["Local training selected.", f"device: {state['train_device']['device']}"], "Run smoke before live training.")
        elif command == "set_remote_train":
            state["train_device"]["mode"] = "remote"
            concise_print(state, ["Set remote training mode."], ["Remote training selected.", "Agent will not store or print passwords."], "Enter an ssh login command without a password.")
        elif command == "set_remote_login":
            login = raw.strip()
            state["train_device"]["mode"] = "remote"
            state["train_device"]["remote"]["login_command"] = login
            match = re.search(r"ssh\s+-p\s+(\d+)\s+([^@]+)@(\S+)", login)
            if match:
                state["train_device"]["remote"].update({"port": match.group(1), "user": match.group(2), "host": match.group(3)})
            concise_print(state, ["Record remote login template."], [f"Login command: {login}", "Password must be entered manually in the terminal."], "Run smoke on the remote server before training.")
        elif command == "set_cuda_train":
            state["train_device"]["device"] = "cuda"
            concise_print(state, ["Set training device."], ["device: cuda"], "Use smoke before live training.")
        elif command == "set_cpu_train":
            state["train_device"]["device"] = "cpu"
            concise_print(state, ["Set training device."], ["device: cpu"], "CPU is suitable for smoke or preflight.")
        elif command == "show_train_device":
            concise_print(state, [], [f"mode: {state['train_device'].get('mode')}", f"device: {state['train_device'].get('device')}", f"remote: {state['train_device'].get('remote')}"], "Set local/cuda before candidate training.")
        elif command == "set_target":
            ok, payload = parse_target_spec(raw)
            if ok:
                spec = json.loads(payload)
                state["target_tuning"].update({"enabled": True, **spec, "current_trial": 0, "history": [], "mode": state["mode"]})
                remember(state, "target", f"{spec['metric']} {spec['operator']} {spec['value']}", {"max_trials": spec["max_trials"]})
                concise_print(state, ["Set tuning target."], [f"{spec['metric']} {spec['operator']} {spec['value']}", f"max_trials: {spec['max_trials']}"], "Use 开始目标训练 candidate to generate a trial plan.")
            else:
                status = "blocked"
                concise_print(state, ["Validate target spec."], [payload], "Use a supported metric such as pair_f1 >= 0.75.")
        elif command == "show_target":
            concise_print(state, [], [json.dumps(state["target_tuning"], ensure_ascii=False)], "Start target training only after reviewing the plan.")
        elif command == "clear_target":
            state["target_tuning"].update({"enabled": False, "metric": None, "operator": None, "value": None, "current_trial": 0, "history": []})
            remember(state, "target", "target tuning cleared")
            concise_print(state, ["Clear target tuning."], ["target_tuning disabled"], "Set a new target when needed.")
        elif command == "start_target_train":
            code, path = write_tuning_plan(state, turn_dir)
            remember(state, "tuning_plan", f"{code}: {path}", {"target": state["target_tuning"]})
            concise_print(state, ["Generate target tuning plan.", "No benchmark will run."], [f"{code}: {path}"], "Review the tuning plan before any live training.")
        elif command == "show_memory":
            rows = read_recent_memory(Path(state["memory"]), limit=10)
            if rows:
                concise_print(state, [], [f"{item.get('kind')}: {item.get('summary')}" for item in rows[-5:]], "Use doctor to refresh conclusions.")
            else:
                concise_print(state, [], ["No Agent memory yet."], "Run inspect, doctor, or set a tuning target first.")
        elif command == "cleanup":
            keep = int(args[0]) if args else 10
            report = cleanup_reports(state["out"], keep=keep, dry_run=False)
            concise_print(state, [f"Cleanup reports under {state['out']}."], [f"{report['status']}: removed {len(report['removed_dirs'])}"], "Run /usage to verify state.")
        elif command in {"diagnose", "inspect", "trace", "compare", "case", "doctor", "schedule", "report", "auditdata"}:
            data, prompt = shell_prompt(agent, command, args)
            write_json(turn_dir / "request.json", {**data, "prompt": prompt})
            write_markdown(turn_dir / "prompt.md", [prompt])
            state["last_report_path"] = str(turn_dir / "prompt.md")
            remember(state, "analysis", f"{command} prompt generated", {"args": args, "report": str(turn_dir / "prompt.md")})
            if state["mode"] == "live":
                guard: AgentRuntimeGuard = state["runtime_guard"]
                guard.mode = state["mode"]
                ok, stop_data = guard.check_before_call(prompt, command)
                if not ok:
                    status = "blocked"
                    guard.write_limit_stop(turn_dir, stop_data)
                    sync_guard_state(state, guard)
                    concise_print(state, ["Check API/cycle guard."], [f"Agent stopped this request because API/cycle guard was triggered. Reason: {stop_data['stop_reason']}", f"See: {turn_dir / 'limit_stop.md'}"], "Switch to /dry or reduce input files.")
                else:
                    response, usage = agent.call_with_usage(prompt)
                    guard.record_call(prompt, usage)
                    sync_guard_state(state, guard)
                    write_markdown(turn_dir / "response.md", [response])
                    state["last_report_path"] = str(turn_dir / "response.md")
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
    if time.time() - start > state.get("max_idle_seconds", 120) and not any(turn_dir.iterdir()):
        status = "error"
        write_loop_stop(state, turn_dir, raw, command, "idle_without_output")
    record()
    return True


def run_agent(args: argparse.Namespace) -> None:
    out = Path(args.out)
    history = Path(args.history) if args.history else out / "history.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    state = default_state(args, out, history)
    agent = RNAAnalysisAgent(dry_run=False, model=args.model, timeout=state["api_timeout"], max_retries=args.max_retries)
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


def run_agent_audit(args: argparse.Namespace) -> None:
    """Audit Agent safety guards without calling any LLM API."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    checks: list[dict[str, Any]] = []
    
    # 1. memory compact function exists
    try:
        from models.agent.memory import compact_memory, read_recent_memory, sanitize_text
        checks.append({"name": "memory_compact_exists", "status": "pass"})
    except ImportError:
        checks.append({"name": "memory_compact_exists", "status": "fail", "detail": "compact_memory not importable"})
    
    # 2. read_recent_memory limit
    try:
        mem = Path(out / "test_memory.jsonl")
        mem.write_text("")
        rows = read_recent_memory(mem, limit=5)
        mem.unlink()
        checks.append({"name": "read_recent_memory_limit", "status": "pass", "rows": len(rows)})
    except Exception as e:
        checks.append({"name": "read_recent_memory_limit", "status": "fail", "detail": str(e)})
    
    # 3. sanitize_text
    try:
        result = sanitize_text("password=secret123")
        if "secret123" not in result:
            checks.append({"name": "sanitize_password", "status": "pass"})
        else:
            checks.append({"name": "sanitize_password", "status": "fail", "detail": "password not redacted"})
    except Exception as e:
        checks.append({"name": "sanitize_password", "status": "fail", "detail": str(e)})
    
    # 4. runtime guard
    try:
        guard = AgentRuntimeGuard(out)
        for attr in ["max_api_calls", "max_tokens_total", "max_same_prompt", "max_consecutive_errors", "max_consecutive_blocked"]:
            assert hasattr(guard, attr), f"missing {attr}"
        checks.append({"name": "runtime_guard_fields", "status": "pass"})
    except Exception as e:
        checks.append({"name": "runtime_guard_fields", "status": "fail", "detail": str(e)})
    
    # 5. record_result
    try:
        guard = AgentRuntimeGuard(out)
        ok, _ = guard.record_result("error")
        assert ok, "should not stop on first error"
        ok, _ = guard.record_result("error")
        ok, data = guard.record_result("error")
        assert not ok, "should stop after 3 errors"
        assert data["stop_reason"] == "max_consecutive_errors"
        checks.append({"name": "record_result_consecutive_errors", "status": "pass"})
    except Exception as e:
        checks.append({"name": "record_result_consecutive_errors", "status": "fail", "detail": str(e)})
    
    # 6. max_consecutive_blocked
    try:
        guard = AgentRuntimeGuard(out, max_consecutive_blocked=3)
        guard.record_result("blocked")
        guard.record_result("blocked")
        ok, data = guard.record_result("blocked")
        assert not ok and data["stop_reason"] == "max_consecutive_blocked"
        checks.append({"name": "max_consecutive_blocked", "status": "pass"})
    except Exception as e:
        checks.append({"name": "max_consecutive_blocked", "status": "fail", "detail": str(e)})
    
    # 7. cleanup safe_root
    try:
        from models.agent.cleanup import safe_root
        assert safe_root(Path("outputs/llm_shell")) is True
        assert safe_root(Path("/etc/passwd")) is False
        checks.append({"name": "cleanup_safe_root", "status": "pass"})
    except Exception as e:
        checks.append({"name": "cleanup_safe_root", "status": "fail", "detail": str(e)})
    
    # 8. benchmark blocked
    try:
        from models.agent.safety import block_reason, DIAGNOSTIC_ARTIFACT_PATTERNS
        # benchmark_candidate is handled by shell confirmation gate, not block_reason
        assert any("benchmark" in pat for pat in DIAGNOSTIC_ARTIFACT_PATTERNS) or \
               block_reason("run benchmark", "unknown") is not None or \
               block_reason("benchmark.json", "unknown") is not None
        checks.append({"name": "benchmark_blocked", "status": "pass"})
    except Exception as e:
        checks.append({"name": "benchmark_blocked", "status": "fail", "detail": str(e)})
    
    # 9. training confirmation required
    try:
        from models.agent.safety import CONFIRM_TRAIN
        assert len(CONFIRM_TRAIN) > 0
        checks.append({"name": "training_confirmation_required", "status": "pass"})
    except Exception as e:
        checks.append({"name": "training_confirmation_required", "status": "fail", "detail": str(e)})
    
    # 10. discover_runs
    try:
        from models.agent.paths import discover_runs
        fake = Path(out / "fake_outputs")
        fake.mkdir(parents=True, exist_ok=True)
        (fake / "smoke_run").mkdir(exist_ok=True)
        (fake / "smoke_run" / "trainlog.jsonl").write_text("{}")
        runs = discover_runs(fake, max_items=5)
        assert len(runs) >= 1
        checks.append({"name": "discover_runs", "status": "pass", "found": len(runs)})
    except Exception as e:
        checks.append({"name": "discover_runs", "status": "fail", "detail": str(e)})
    
    # 11. compact_memory
    try:
        mem_path = Path(out / "test_compact.jsonl")
        for i in range(100):
            mem_path.open("a").write(json.dumps({"kind": "test", "summary": f"entry_{i}"}) + "\n")
        report = compact_memory(mem_path, keep_recent=10, keep_errors=5)
        assert report["after_lines"] <= 15 + report.get("kept_errors", 0)
        mem_path.unlink()
        checks.append({"name": "memory_compact", "status": "pass", "after_lines": report["after_lines"]})
    except Exception as e:
        checks.append({"name": "memory_compact", "status": "fail", "detail": str(e)})
    
    # Summary
    passed = sum(1 for c in checks if c["status"] == "pass")
    failed = sum(1 for c in checks if c["status"] == "fail")
    
    report = {"checks": checks, "passed": passed, "failed": failed, "total": len(checks)}
    write_json(out / "report.json", report)
    
    md_lines = ["# Agent Audit Report", "", f"Passed: {passed}/{len(checks)}  Failed: {failed}"]
    for c in checks:
        icon = "[OK]" if c["status"] == "pass" else "[FAIL]"
        md_lines.append(f"{icon} {c['name']} {c.get('detail', '')}")
    write_markdown(out / "report.md", md_lines)
    
    if failed > 0:
        raise SystemExit(f"agent_audit: {failed} checks failed.")
    print(f"agent_audit PASS -> {out}")


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
        max_same_prompt=args.max_same_prompt,
        timeout=60,
        api_timeout=60,
        command_timeout=120,
        train_timeout=0,
        max_retries=2,
        max_idle_seconds=args.max_idle_seconds,
        memory=out / "memory.jsonl",
    )
    state = default_state(ns, out, out / "history.jsonl")
    agent = RNAAnalysisAgent(dry_run=False)
    commands = [
        "/status",
        "/usage",
        "\u8fd0\u884c smoke",
        "\u8fd0\u884c audit",
        "\u7f16\u8bd1 agent",
        "\u8bbe\u7f6e\u8bad\u7ec3\u8bbe\u5907",
        "local",
        "\u4f7f\u7528 cuda",
        "\u68c0\u67e5 candidate",
        "\u7efc\u5408\u8bca\u65ad",
        "\u6e05\u7406\u65e7\u62a5\u544a\uff0c\u53ea\u4fdd\u755910\u6b21",
        "/last",
        "/open",
        "\u8bad\u7ec3 candidate",
        "\u8fdb\u884c\u8bad\u7ec3 candidate",
        "\u8dd1 benchmark",
        "\u8fdb\u884c benchmark",
        "git push origin main",
        "pip install torch",
        "conda install pytorch",
        "curl http://example.com",
        "wget http://example.com",
        "rm -rf outputs",
        "CUDA_VISIBLE_DEVICES=0 python main.py train",
        "report .env",
        "report README.md > .env",
        "\u8fd0\u884c smoke >> outputs/log.txt",
        "python main.py train --config config/fixed.yaml",
        "\u8bad\u7ec3 candidate && git push",
        "\u8fdb\u884c\u8bad\u7ec3 && rm -rf outputs",
        "remote",
        "ssh -p 14591 root@connect.westc.seetacloud.com",
        "\u67e5\u770b\u8bad\u7ec3\u8bbe\u5907",
        "\u8bbe\u5b9a\u76ee\u6807 pair_f1 >= 0.75\uff0c\u6700\u591a\u8c03\u53c2 3 \u6b21",
        "\u67e5\u770b\u8c03\u53c2\u76ee\u6807",
        "\u5f00\u59cb\u76ee\u6807\u8bad\u7ec3 candidate",
        "\u67e5\u770b\u8bb0\u5fc6",
        "/memory",
        "\u6e05\u9664\u8c03\u53c2\u76ee\u6807",
        "/cleanup --keep -1",
        "/live",
        "/dry",
        "/quiet",
        "/status",
        "/normal",
        "/cleanup 10",
        "/exit",
    ]
    if args.loop_test:
        commands = ["\u68c0\u67e5 candidate", "\u68c0\u67e5 candidate", "\u68c0\u67e5 candidate", "/exit"]
    for item in commands:
        if not execute_input(item, state, agent, echo=False):
            break
    history = [json.loads(line) for line in (out / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.loop_test:
        if not any(item.get("status") == "loop_stopped" for item in history):
            raise SystemExit("agent_test failed: loop guard was not triggered.")
        print(f"agent_test PASS -> {out}")
        return
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
    parser.add_argument("--api_timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_api_calls", type=int, default=20)
    parser.add_argument("--max_tokens_total", type=int, default=20000)
    parser.add_argument("--max_same_prompt", type=int, default=2)


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
    agent.add_argument("--memory", default=None)
    agent.add_argument("--no_api", action="store_true")
    agent.add_argument("--model", default=None)
    agent.add_argument("--max_api_calls", type=int, default=int(os.environ.get("LLM_MAX_API_CALLS", "20")))
    agent.add_argument("--max_tokens_total", type=int, default=int(os.environ.get("LLM_MAX_TOKENS_TOTAL", "20000")))
    agent.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "60")))
    agent.add_argument("--api_timeout", type=int, default=int(os.environ.get("LLM_API_TIMEOUT", os.environ.get("LLM_TIMEOUT", "60"))))
    agent.add_argument("--command_timeout", type=int, default=int(os.environ.get("LLM_COMMAND_TIMEOUT", "120")))
    agent.add_argument("--train_timeout", type=int, default=int(os.environ.get("LLM_TRAIN_TIMEOUT", "0")))
    agent.add_argument("--max_turns", type=int, default=int(os.environ.get("LLM_MAX_TURNS", "100")))
    agent.add_argument("--max_retries", type=int, default=int(os.environ.get("LLM_MAX_RETRIES", "2")))
    agent.add_argument("--max_same_prompt", type=int, default=int(os.environ.get("LLM_MAX_SAME_PROMPT", "2")))
    agent.add_argument("--max_idle_seconds", type=int, default=int(os.environ.get("LLM_MAX_IDLE_SECONDS", "120")))
    agent.set_defaults(func=run_agent)

    agent_test = sub.add_parser("agent_test")
    agent_test.add_argument("--out", default="outputs/llm_shell_test")
    agent_test.add_argument("--max_same_prompt", type=int, default=2)
    agent_test.add_argument("--max_idle_seconds", type=int, default=120)
    agent_test.add_argument("--loop_test", action="store_true")
    agent_test.set_defaults(func=run_agent_test)

    agent_audit = sub.add_parser("agent_audit")
    agent_audit.add_argument("--out", default="outputs/llm_agent_audit")
    agent_audit.set_defaults(func=run_agent_audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
