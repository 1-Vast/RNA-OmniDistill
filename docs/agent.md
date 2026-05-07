# Agent Shell Safety Architecture

The optional LLM analysis agent is an experiment assistant, not a structure
predictor. It reads existing artifacts and writes Markdown/JSON reports.

This document describes the safety architecture as of the final freeze.

## Core Safety Rules

- **Read-only by default**: The Agent never runs training, benchmark, git,
  data modification, or config-editing commands without explicit user
  confirmation.
- **Training requires confirmation**: `进行训练 candidate` or `train candidate`
  triggers a confirmation gate.
- **Benchmark is blocked**: The Agent never executes benchmarks automatically.
- **No passwords**: Remote passwords are never stored, printed, or written
  to code, docs, .env, or Agent memory.
- **No config mutation**: The Agent never modifies `config/candidate.yaml`.
  Trial configs are written only to `outputs/llm_shell/tuning/`.

## Runtime Guards

### API Limits
- `max_api_calls` (default 20): stops when exceeded
- `max_tokens_total` (default 20000): estimated token budget
- `max_same_prompt` (default 2): blocks repeated identical prompts

### Consecutive Error Circuit Breaker
- `max_consecutive_errors` (default 3): stops after repeated errors
  (error / timeout / loop_stopped)
- `max_consecutive_blocked` (default 5): stops after repeated blocked commands
- Both trigger a **hard stop**: writes `limit_stop.md`, requires human review
- Use `/clear` to reset counters

### Loop / Stall Detection
- Detects repeated raw input and repeated failing commands
- Writes `loop_detected.json` and stops when threshold exceeded

## Memory Management

### File-Level Compaction
- `compact_memory()` truncates old entries while preserving recent records
  and error/blocked/loop entries
- Default: keep 50 recent + 10 error records, max 512 KB
- Auto-compact on append when file exceeds `max_bytes`
- Writes `memory_compact_report.json` for audit

### Sanitization
- `sanitize_text()` redacts password, API key, token patterns
- JSONL lines without valid JSON are marked corrupt
- Password fields are stripped from payloads

## Run Discovery

- `discover_runs()` finds experiment directories under `outputs/` by checking
  for marker files (`trainlog.jsonl`, `best.pt`, `last.pt`, `benchmark.json`,
  `predictions.jsonl`)
- Use `/runs` or `查找实验` / `最近实验` to list recent runs
- Agent prefers latest discovered run when no run_dir is specified

## Slash Commands

| Command | Description |
|---|---|
| `/help` | Show built-in and slash commands |
| `/status` | JSON status snapshot |
| `/usage` | Write usage report to disk |
| `/memory` | Show recent memory entries |
| `/memory compact` | Compact memory file |
| `/cleanup [keep]` | Cleanup old turns, keep N most recent |
| `/runs` | Discover recent experiment runs |
| `/last` | Show last command result |
| `/open` | Print last report path |
| `/mode` / `/dry` / `/live` | Switch runtime mode |
| `/quiet` / `/normal` / `/verbose` | UI verbosity |
| `/history` | Show recent history lines |
| `/clear` | Reset in-memory state + hard stop |
| `/exit` | Quit |

## Diagnostics

| Command | Description |
|---|---|
| `diagnose <run>` | Full diagnostic prompt |
| `inspect <run>` | Inspect run artifacts |
| `trace <config> <ckpt> <bench>` | Provenance trace |
| `compare <run_a> <run_b>` | Compare two runs |
| `case <predictions>` | Case-level analysis |
| `doctor <run> <config>` | Doctor-style conclusion refresh |
| `schedule <run> <config>` | Training schedule plan |
| `report <file...>` | Generalized report from inputs |
| `auditdata <jsonl...>` | Data audit prompt |

## Target Tuning

- Set target: `设定目标 pair_f1 >= 0.75, 最多调参 3 次`
- Start: `开始目标训练 candidate`
- Generates tuning plan + trial config in `outputs/llm_shell/tuning/`
- Never modifies `config/candidate.yaml`
- Max 10 trials; no benchmark execution

## Agent Audit

```bash
python scripts/llm.py agent_audit --out outputs/llm_agent_audit
```

Checks all safety guards: memory compact, sanitize, runtime limits,
consecutive error tracking, cleanup safety, benchmark blocking,
training confirmation, run discovery.

## Environment

```text
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```
