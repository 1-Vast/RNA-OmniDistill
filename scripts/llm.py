from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.agent.analyzer import RNAAnalysisAgent, write_report


HELP_TEXT = """Built-in commands:
  diagnose <run_dir>
  schedule <run_dir> <config_path>
  report <file1> <file2> ...
  auditdata <jsonl1> <jsonl2> ...
  dry <command...>
  live <command...>

Slash commands:
  /help
  /status
  /mode
  /dry
  /live
  /history
  /clear
  /exit

Safety:
  Agent is read-only.
  It never runs training or benchmark.
  It never modifies labels, predictions, metrics, checkpoints, or configs."""


ACTION_BLOCK_PATTERNS = [
    "main.py train",
    "run.py external",
    "run.py ablate",
    "eval.py bench",
    "cuda_visible_devices",
    "git commit",
    "git push",
    "git reset",
    "git checkout",
    "git clean",
    "write config",
    "修改配置",
]

TOKEN_BLOCK_PATTERNS = [
    " train ",
    " benchmark ",
    " rm ",
    " del ",
    " remove ",
    " mv ",
    " cp ",
    " overwrite ",
    " 删除",
    " 覆盖",
    " .env",
    " api_key",
    " llm_api_key",
    " checkpoint",
    " best.pt",
]


def safety_block_reason(raw: str, parsed_command: str | None = None) -> str | None:
    lowered = " " + raw.lower().replace("\\", "/") + " "
    for pattern in ACTION_BLOCK_PATTERNS:
        if pattern in lowered:
            return pattern.strip()
    if parsed_command not in {"diagnose", "report"}:
        for pattern in [" predictions.jsonl", " benchmark.json"]:
            if pattern in lowered:
                return pattern.strip()
    for pattern in TOKEN_BLOCK_PATTERNS:
        if pattern in lowered:
            return pattern.strip()
    return None


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    return path


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Input directory not found: {path}")
    return path


def require_dir_for_shell(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Input directory not found: {path}")
    return path


def require_file_for_shell(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    return path


def run_diagnose(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    data, markdown = agent.diagnose(require_dir(Path(args.run)))
    write_report(Path(args.out), "diagnose", data, markdown)


def run_schedule(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    config = require_file(Path(args.config)) if args.config else None
    data, markdown = agent.schedule(require_dir(Path(args.run)), config)
    write_report(Path(args.out), "schedule", data, markdown)


def run_report(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    data, markdown = agent.report([require_file(Path(path)) for path in args.inputs], max_chars=args.max_chars)
    write_report(Path(args.out), "report", data, markdown)


def run_auditdata(args: argparse.Namespace) -> None:
    agent = RNAAnalysisAgent(dry_run=args.dry_run)
    data, markdown = agent.auditdata([require_file(Path(path)) for path in args.inputs], max_rows=args.max_rows)
    write_report(Path(args.out), "dataaudit", data, markdown)


def path_tokens(text: str) -> list[str]:
    return re.findall(r"(?:outputs|config|release|dataset|docs)[A-Za-z0-9_./\\-]*(?:\.yaml|\.yml|\.md|\.jsonl|/candidate|\\candidate)?", text)


def parse_agent_command(raw: str) -> tuple[str, list[str], str | None]:
    text = raw.strip()
    if not text:
        return "empty", [], None
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    if not parts:
        return "empty", [], None
    prefix_mode = None
    if parts[0] in {"dry", "live"}:
        prefix_mode = parts[0]
        parts = parts[1:]
        if not parts:
            return "empty", [], prefix_mode
    command = parts[0].lower()
    if command in {"diagnose", "schedule", "report", "auditdata"}:
        return command, parts[1:], prefix_mode

    paths = path_tokens(text)
    lowered = text.lower()
    if any(key in lowered for key in ["诊断", "diagnose"]):
        return "diagnose", paths[:1], prefix_mode
    if any(key in lowered for key in ["计划", "调度", "schedule", "下一步"]):
        return "schedule", paths[:2], prefix_mode
    if any(key in lowered for key in ["报告", "总结", "report", "整理"]):
        return "report", paths, prefix_mode
    if any(key in lowered for key in ["数据审计", "审计", "auditdata"]):
        return "auditdata", paths, prefix_mode
    return "unknown", [], prefix_mode


def write_history(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_turn_request(turn_dir: Path, data: dict, prompt: str) -> None:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "request.json").write_text(json.dumps({**data, "prompt": prompt}, indent=2, ensure_ascii=False), encoding="utf-8")
    (turn_dir / "prompt.md").write_text(prompt.rstrip() + "\n", encoding="utf-8")


def build_shell_prompt(agent: RNAAnalysisAgent, command: str, args: list[str]) -> tuple[dict, str]:
    if command == "diagnose":
        if len(args) != 1:
            raise ValueError("Usage: diagnose <run_dir>")
        return agent.build_diagnose_prompt(require_dir_for_shell(Path(args[0])))
    if command == "schedule":
        if len(args) != 2:
            raise ValueError("Usage: schedule <run_dir> <config_path>")
        return agent.build_schedule_prompt(require_dir_for_shell(Path(args[0])), require_file_for_shell(Path(args[1])))
    if command == "report":
        if not args:
            raise ValueError("Usage: report <file1> <file2> ...")
        return agent.build_report_prompt([require_file_for_shell(Path(item)) for item in args])
    if command == "auditdata":
        if not args:
            raise ValueError("Usage: auditdata <jsonl1> <jsonl2> ...")
        return agent.build_auditdata_prompt([require_file_for_shell(Path(item)) for item in args])
    raise ValueError("Could not parse command. Type /help for examples.")


def shell_status(mode: str, out: Path, history: Path, no_api: bool) -> str:
    return (
        f"Mode: {mode}\n"
        "Safety: read-only\n"
        f"API disabled: {str(no_api).lower()}\n"
        f"Output: {out}\n"
        f"History: {history}"
    )


def execute_shell_input(
    raw: str,
    state: dict,
    agent: RNAAnalysisAgent,
    echo: bool = True,
) -> bool:
    raw = raw.strip()
    if not raw:
        return True
    out = Path(state["out"])
    history = Path(state["history"])
    turn = int(state["turn"]) + 1
    state["turn"] = turn
    timestamp = datetime.now().isoformat(timespec="seconds")
    mode = str(state["mode"])
    parsed_command = None
    parsed_args: list[str] = []
    status = "ok"
    turn_dir: Path | None = None

    def finish(record: dict) -> bool:
        write_history(history, record)
        return True

    if raw in {"/exit", "/quit"}:
        write_history(history, {
            "turn": turn, "timestamp": timestamp, "mode": mode, "raw_input": raw,
            "parsed_command": "exit", "args": [], "status": "ok", "out_dir": None,
        })
        return False
    if raw == "/help":
        if echo:
            print(HELP_TEXT)
        return finish({"turn": turn, "timestamp": timestamp, "mode": mode, "raw_input": raw, "parsed_command": "help", "args": [], "status": "ok", "out_dir": None})
    if raw in {"/status", "/mode"}:
        if echo:
            print(shell_status(mode, out, history, bool(state["no_api"])))
        return finish({"turn": turn, "timestamp": timestamp, "mode": mode, "raw_input": raw, "parsed_command": raw[1:], "args": [], "status": "ok", "out_dir": None})
    if raw == "/dry":
        state["mode"] = "dry_run"
        if echo:
            print("Mode: dry-run")
        return finish({"turn": turn, "timestamp": timestamp, "mode": "dry_run", "raw_input": raw, "parsed_command": "dry", "args": [], "status": "ok", "out_dir": None})
    if raw == "/live":
        if state["no_api"]:
            if echo:
                print("Live mode is disabled by --no_api.")
            status = "blocked"
        else:
            state["mode"] = "live"
            if echo:
                print("Mode: live")
            status = "ok"
        return finish({"turn": turn, "timestamp": timestamp, "mode": str(state["mode"]), "raw_input": raw, "parsed_command": "live", "args": [], "status": status, "out_dir": None})
    if raw == "/clear":
        if echo:
            print("Screen clear is not supported in non-interactive logs.")
        return finish({"turn": turn, "timestamp": timestamp, "mode": mode, "raw_input": raw, "parsed_command": "clear", "args": [], "status": "ok", "out_dir": None})
    if raw == "/history":
        if history.exists() and echo:
            print(history.read_text(encoding="utf-8", errors="replace"))
        return finish({"turn": turn, "timestamp": timestamp, "mode": mode, "raw_input": raw, "parsed_command": "history", "args": [], "status": "ok", "out_dir": None})

    try:
        parsed_command, parsed_args, prefix_mode = parse_agent_command(raw)
        if prefix_mode:
            mode = "dry_run" if prefix_mode == "dry" else "live"
        if state["no_api"] and mode == "live":
            mode = "dry_run"
        reason = safety_block_reason(raw, parsed_command)
        if reason:
            status = "blocked"
            turn_dir = out / "turns" / f"{turn:04d}_blocked"
            turn_dir.mkdir(parents=True, exist_ok=True)
            message = (
                "Blocked by Agent safety policy.\n"
                "This shell is read-only and cannot run training, benchmark, git, deletion, or config-modifying commands.\n"
                f"Matched: {reason}\n"
            )
            (turn_dir / "blocked.md").write_text(message, encoding="utf-8")
            if echo:
                print(message.rstrip())
        elif parsed_command == "unknown":
            status = "error"
            turn_dir = out / "turns" / f"{turn:04d}_error"
            turn_dir.mkdir(parents=True, exist_ok=True)
            message = "Could not parse command. Type /help for examples.\n"
            (turn_dir / "error.md").write_text(message, encoding="utf-8")
            if echo:
                print(message.rstrip())
        elif parsed_command == "empty":
            status = "ok"
        else:
            turn_dir = out / "turns" / f"{turn:04d}_{parsed_command}"
            data, prompt = build_shell_prompt(agent, parsed_command, parsed_args)
            data["dry_run"] = mode == "dry_run"
            data["raw_input"] = raw
            data["parsed_command"] = parsed_command
            write_turn_request(turn_dir, data, prompt)
            if mode == "live":
                response = agent.call(prompt)
                (turn_dir / "response.md").write_text(response.rstrip() + "\n", encoding="utf-8")
                if echo:
                    print(response)
            else:
                if echo:
                    print(f"Dry-run prompt written to {turn_dir / 'prompt.md'}")
    except Exception as exc:
        status = "error"
        turn_dir = out / "turns" / f"{turn:04d}_error"
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "error.md").write_text(str(exc) + "\n", encoding="utf-8")
        if echo:
            print(str(exc))

    write_history(history, {
        "turn": turn,
        "timestamp": timestamp,
        "mode": mode,
        "raw_input": raw,
        "parsed_command": parsed_command,
        "args": parsed_args,
        "status": status,
        "out_dir": str(turn_dir) if turn_dir else None,
    })
    return True


def run_agent(args: argparse.Namespace) -> None:
    out = Path(args.out)
    history = Path(args.history) if args.history else out / "history.jsonl"
    no_api = bool(args.no_api)
    mode = "dry_run" if args.dry_run or no_api else "live"
    out.mkdir(parents=True, exist_ok=True)
    agent = RNAAnalysisAgent(dry_run=False, model=args.model)
    state = {"out": out, "history": history, "mode": mode, "no_api": no_api, "turn": 0}
    print("RNA-OmniDiffusion Agent Shell")
    print(f"Mode: {'dry-run' if mode == 'dry_run' else 'live'}")
    print("Safety: read-only")
    print("Type /help for commands, /exit to quit.")
    while True:
        try:
            raw = input("agent> ")
        except EOFError:
            print()
            break
        if not execute_shell_input(raw, state, agent):
            break


def ensure_agent_test_fixtures() -> None:
    run = Path("outputs/candidate")
    run.mkdir(parents=True, exist_ok=True)


def run_agent_test(args: argparse.Namespace) -> None:
    ensure_agent_test_fixtures()
    out = Path(args.out)
    if out.exists():
        # Keep this scoped to the requested ignored output directory.
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    agent = RNAAnalysisAgent(dry_run=False)
    state = {"out": out, "history": out / "history.jsonl", "mode": "dry_run", "no_api": True, "turn": 0}
    commands = [
        "/help",
        "/status",
        "diagnose outputs/candidate",
        "schedule outputs/candidate config/candidate.yaml",
        "report release/model_card.md release/results_summary.md release/limitations.md",
        "auditdata dataset/archive/train.jsonl dataset/archive/test.jsonl",
        "train config/candidate.yaml",
        "git push origin main",
        "/exit",
    ]
    for item in commands:
        keep_going = execute_shell_input(item, state, agent, echo=False)
        if not keep_going:
            break
    history = read_history(out / "history.jsonl")
    blocked = [item for item in history if item.get("status") == "blocked"]
    prompt_dirs = [p for p in (out / "turns").glob("*") if (p / "prompt.md").exists()]
    if len(blocked) < 2:
        raise SystemExit("agent_test failed: dangerous commands were not blocked.")
    if len(prompt_dirs) < 4:
        raise SystemExit("agent_test failed: expected at least 4 prompt turns.")
    print(f"agent_test PASS -> {out}")


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LLM analysis agent for RNA-OmniDiffusion. "
            "It only reads artifacts and writes reports; it never runs benchmark inference."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True)
    common.add_argument("--dry_run", action="store_true")

    diagnose = sub.add_parser("diagnose", parents=[common], help="Diagnose one run directory.")
    diagnose.add_argument("--run", required=True)
    diagnose.set_defaults(func=run_diagnose)

    schedule = sub.add_parser("schedule", parents=[common], help="Suggest safe next experiments.")
    schedule.add_argument("--run", required=True)
    schedule.add_argument("--config")
    schedule.set_defaults(func=run_schedule)

    report = sub.add_parser("report", parents=[common], help="Generate a paper report draft from existing files.")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--max_chars", type=int, default=20000)
    report.set_defaults(func=run_report)

    auditdata = sub.add_parser("auditdata", parents=[common], help="Audit RNA JSONL dataset summaries.")
    auditdata.add_argument("--inputs", nargs="+", required=True)
    auditdata.add_argument("--max_rows", type=int)
    auditdata.set_defaults(func=run_auditdata)

    agent = sub.add_parser("agent", help="Open the read-only interactive analysis shell.")
    agent.add_argument("--dry_run", action="store_true")
    agent.add_argument("--out", default="outputs/llm_shell")
    agent.add_argument("--history")
    agent.add_argument("--no_api", action="store_true")
    agent.add_argument("--model")
    agent.set_defaults(func=run_agent)

    agent_test = sub.add_parser("agent_test", help="Run non-interactive shell safety test.")
    agent_test.add_argument("--out", default="outputs/llm_shell_test")
    agent_test.set_defaults(func=run_agent_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
