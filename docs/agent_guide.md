# RNA-OmniDiffusion Agent Guide

## 1. What the Agent Is

The Agent is an **optional analysis assistant** — not part of the model pipeline.
- Reads existing artifacts (train logs, checkpoints, benchmarks).
- Writes diagnostic reports (Markdown/JSON).
- Does NOT change predictions, benchmark metrics, or release files.
- **Read-only** by default.

## 2. Starting the Agent

```bash
# Universal
python agent.py

# Windows CMD
agent

# Windows PowerShell
python agent.py
# or
.\agent.cmd

# Linux / macOS
python agent.py
# or
bash agent.sh

# Original entry point
python scripts/llm.py agent --dry_run
```

## 3. Interaction Style

Type natural language — Chinese, English, or mixed:

```
agent> 运行 smoke
agent> 检查 candidate
agent> 查找实验
agent> 查看记忆
agent> 压缩记忆
agent> 设置训练设备为远程
agent> 设定目标 pair_f1 >= 0.75, 最多调参 3 次
agent> 扫描解码参数
agent> 审计 collator
agent> 查看运行环境
agent> /exit
```

## 4. Local vs Remote

| Environment | Purpose |
|---|---|
| **Local** (your PC) | Edit code, run smoke, dry-run, generate configs, inspect outputs. Do NOT run full training. |
| **Remote** (server) | Run full training, decoding sweeps, collator audits, performance validation. |

## 5. Remote Login

```bash
ssh -p 49018 root@connect.nmb1.seetacloud.com
```

**Password**: enter manually in terminal. Do **not** save in code, docs, `.env`, or Agent memory.

## 6. Recommended Remote Workflow

After SSH login:

```bash
cd /root/RNA-OmniDiffusion
git pull origin main

# Environment check
python main.py overview
python main.py params --config config/candidate.yaml
python main.py smoke

# Dataset check
python scripts/check_datasets.py check --root dataset/raw --out outputs/dataset_check/remote

# Collator audit
python scripts/audit_collator.py --config config/candidate.yaml --split train --samples 512 --out outputs/audit/collator_candidate

# Full training
python main.py train --config config/candidate.yaml --device cuda

# Decoding sweep (after training)
python scripts/sweep_decoding.py --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split val --out outputs/sweeps/decoding_candidate --device cuda --max_samples 512

# View best decoding params
cat outputs/sweeps/decoding_candidate/best.json
```

## 7. Slash Commands

| Command | Description |
|---|---|
| `/help` | Show all built-in commands |
| `/examples` | Show natural language examples |
| `/runs` | Discover recent experiment runs |
| `/memory` | Show recent Agent memory |
| `/memory compact` | Compact the memory file |
| `/usage` | Write usage report to disk |
| `/cleanup [keep]` | Remove old turns, keep N most recent |
| `/status` | JSON status snapshot |
| `/clear` | Reset in-memory state |
| `/quiet` / `/normal` / `/verbose` | UI verbosity |
| `/dry` / `/live` | Switch runtime mode |
| `/exit` | Quit |

## 8. Natural Language Examples

```
运行 smoke              →    run smoke test
检查 candidate          →    inspect outputs/candidate
查找实验                →    list recent runs
查看记忆                →    show memory entries
压缩记忆                →    compact memory
综合诊断                →    doctor analysis
清理旧报告              →    cleanup old turns
设置训练设备为远程        →    set remote training mode
设定目标 pair_f1 >= 0.75  →    target tuning
查看远程登录命令          →    show SSH login template
服务器训练               →    show remote training workflow
扫描解码参数             →    sweep decoding plan
审计 collator            →    collator audit plan
生成临时配置             →    trial config plan
查看本地数据集            →    dataset check plan
对比本地和服务器数据集     →    dataset comparison plan
查看运行环境             →    detect runtime environment
```

## 9. Safety Architecture

| Guard | Description |
|---|---|
| Read-only default | All write/execute commands blocked unless explicitly confirmed |
| Training gate | Requires `进行训练 candidate` confirmation |
| Benchmark blocked | Never executes benchmarks automatically |
| API limits | `max_api_calls`, `max_tokens_total`, `max_same_prompt` |
| Consecutive error stop | 3 errors or 5 blocked → hard stop, human review required |
| Loop detection | Repeated input/failed commands → stop with report |
| Memory compact | Truncates old entries, preserves recent + error records |
| Password sanitization | All password/token patterns redacted |
| Config protection | `config/candidate.yaml` never modified by Agent |
| Path constraint | Trial configs only under `outputs/llm_shell/tuning/` |

## 10. Target Tuning

```
agent> 设定目标 pair_f1 >= 0.75, 最多调参 3 次
agent> 开始目标训练 candidate
```

The Agent generates a tuning plan and trial config in `outputs/llm_shell/tuning/`.
`config/candidate.yaml` is never modified. Max trials = 10. No benchmark execution.

## 11. Environment Variables

```text
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

The Agent works in dry-run mode without these. Set them for live LLM API calls.
