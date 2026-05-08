from __future__ import annotations

import sys

CONFIRM_TRAIN = {
    "杩涜璁粌",
    "杩涜璁粌 candidate",
    "纭璁粌",
    "纭璁粌 candidate",
    "鎵ц璁粌",
    "鎵ц璁粌 candidate",
    "yes train",
    "yes train candidate",
    "run train",
    "run train candidate",
    "execute train",
}
CONFIRM_BENCH = {
    "杩涜 benchmark",
    "纭 benchmark",
    "鎵ц benchmark",
    "yes benchmark",
    "run benchmark",
    "execute benchmark",
}
TRAIN_COMMAND = [sys.executable, "main.py", "train", "--config", "config/candidate.yaml", "--device", "cuda"]
SAFE_COMPILE = [sys.executable, "-m", "py_compile", "agent/analyzer.py", "scripts/llm.py"]
SAFE_SMOKE = [sys.executable, "main.py", "smoke"]
SAFE_AUDIT = [sys.executable, "scripts/audit.py", "clean", "--out", "outputs/clean"]

DANGEROUS_ALWAYS = ["git push", "git commit", "git reset", "git checkout", "git clean", "&&", "||", ";", "|", ">", ">>", " rm ", " del ", "remove ", " mv ", " cp ", ".env", "api_key", "llm_api_key", "cuda_visible_devices", "pip install", "conda install", "curl ", "wget ", "release/best_config.yaml"]
DANGEROUS_WRITE = ["rm ", "del ", "remove ", "mv ", "cp ", "overwrite", "淇敼閰嶇疆", "鍒犻櫎", "瑕嗙洊", ".env", "api_key", "llm_api_key", "cuda_visible_devices", "pip install", "conda install", "curl ", "wget "]
DIAGNOSTIC_ARTIFACT_PATTERNS = ["benchmark.json", "predictions.jsonl", "best.pt", "checkpoint"]
DIAGNOSTIC_COMMANDS = {"trace", "case", "doctor", "inspect", "compare", "diagnose", "report"}


def block_reason(raw: str, command: str) -> str | None:
    lower = raw.lower()
    for item in DANGEROUS_ALWAYS:
        if item in f" {lower} ":
            return item
    if command in {"train_candidate", "benchmark_candidate"}:
        return None
    blocked = list(DANGEROUS_WRITE)
    if command not in DIAGNOSTIC_COMMANDS:
        blocked.extend(DIAGNOSTIC_ARTIFACT_PATTERNS)
    for item in blocked:
        if item in lower:
            return item
    return None


def validate_confirmed_command(cmd: list[str], intent: str) -> tuple[bool, str]:
    joined = " ".join(cmd[1:]).lower()
    forbidden = [";", "&&", "||", "|", ">", ">>", " rm ", " del ", " remove ", " mv ", " cp ", "git", "pip", "conda", "curl", "wget", "release/best_config.yaml", "dataset", "benchmark", "eval.py", "scripts/run.py", "cuda_visible_devices"]
    for item in forbidden:
        if item in f" {joined} ":
            return False, f"forbidden token: {item.strip()}"
    if intent == "train_candidate" and cmd != TRAIN_COMMAND:
        return False, "only fixed candidate training command is allowed"
    if "config/candidate.yaml" not in joined:
        return False, "only config/candidate.yaml is allowed"
    return True, "ok"
