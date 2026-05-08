# RNA-OmniDiffusion Agent Mode 使用指南

Agent 是 RNA-OmniDiffusion 的实验分析助手。它可以读取日志、配置、checkpoint 元信息、benchmark JSON 和预测文件，生成诊断报告、对比报告、调参计划和数据审计提示。它不是模型的一部分，不参与 forward，不生成结构标签，不修改 benchmark 指标。

## 1. 核心边界

- Agent 默认是 `dry_run` + `read-only`。
- Agent 不参与训练、推理、解码或 benchmark inference。
- Agent 不修改 `config/candidate.yaml`。
- Agent 不写 `dataset/teacher_emb/`、`checkpoints/`、`.pt/.pth/.ckpt`、benchmark 结果或预测文件。
- Agent 不保存远程密码、API key 或 `.env` 内容。
- 训练命令有确认门；benchmark 命令即使确认也不会自动执行，只给人工命令。

## 2. 启动方式

在仓库根目录运行：

```powershell
conda activate DL
agent
```

等价入口：

```powershell
python agent.py
python -m agent agent
python scripts/llm.py agent
```

Linux/macOS：

```bash
python agent.py
bash agent.sh
```

显式 dry-run：

```bash
python scripts/llm.py agent --dry_run
```

自定义输出目录：

```bash
python scripts/llm.py agent --out outputs/llm_shell_my_run
```

## 3. 启动画面含义

```text
RNA-OmniDiffusion Agent
mode: dry_run | safety: read-only | Windows | local | cuda | D:\RNA-OmniDiffusion
```

| 字段 | 含义 |
|---|---|
| `mode: dry_run` | 只生成计划、报告和提示，不调用真实 LLM API，不执行危险操作 |
| `safety: read-only` | 禁止训练、benchmark、git、删除、复制、移动、改配置等高风险命令 |
| `Windows/local/cuda` | 当前运行环境推断 |
| 路径 | 当前仓库根目录 |

## 4. 最常用命令

可以输入英文、中文或中英混合：

```text
agent> run smoke
agent> inspect candidate
agent> list runs
agent> diagnose outputs/candidate
agent> compare outputs/candidate outputs/candidate_from_rnafm_pretrain
agent> trace config/candidate.yaml outputs/candidate/best.pt outputs/candidate/benchmark.json
agent> set target pair_f1 >= 0.75 max_trials 3
agent> show env
agent> remote login
agent> server training
agent> audit collator
agent> sweep decoding
agent> trial config
```

中文示例：

```text
agent> 运行 smoke
agent> 检查 candidate
agent> 查找实验
agent> 查看记忆
agent> 压缩记忆
agent> 综合诊断 outputs/candidate
agent> 对比 outputs/candidate outputs/candidate_from_rnafm_pretrain
agent> 设置训练设备为远程
agent> 设定目标 pair_f1 >= 0.75 最大调参 3 次
agent> 查看远程登录命令
agent> 服务器训练
agent> 扫描解码参数
agent> 审计 collator
agent> 生成 trial config
```

## 5. Slash Commands

| 命令 | 作用 |
|---|---|
| `/help` | 显示内置命令和安全规则 |
| `/examples` | 显示自然语言示例 |
| `/status` | 输出当前状态 JSON，包括 mode、API 计数、输出目录 |
| `/usage` | 生成使用报告到 `outputs/llm_shell/usage.md` |
| `/memory` | 查看最近 Agent memory |
| `/memory compact` | 压缩 memory，保留最近记录和错误记录 |
| `/cleanup [keep]` | 清理旧 turn 目录，默认保留最近 10 个 |
| `/runs` | 扫描 `outputs/` 下已有实验 |
| `/last` | 显示上一条命令结果 |
| `/open` | 打印上一份报告路径 |
| `/mode` | 查看当前 dry/live 模式 |
| `/dry` | 切到 dry-run |
| `/live` | 切到 live API 模式，但安全边界仍然存在 |
| `/quiet` | 极简输出 |
| `/normal` | 普通输出 |
| `/verbose` | 详细输出 |
| `/history` | 查看最近交互历史 |
| `/clear` | 清除内存状态和硬停止状态 |
| `/exit` | 退出 |

## 6. 内置分析命令

这些命令既可以在交互 shell 里输入，也可以通过 `scripts/llm.py` 直接运行。

| 命令 | 作用 | 输出 |
|---|---|---|
| `inspect <run_dir>` | 检查一个 run 的日志、checkpoint、benchmark、预测文件 | `inspect.json/md` |
| `diagnose <run_dir>` | 生成训练问题诊断 prompt/report | `diagnose.json/md` |
| `compare <run_a> <run_b>` | 对比两个实验目录 | `compare.json/md` |
| `trace <config> <ckpt> <benchmark>` | 追踪配置、checkpoint、benchmark 是否匹配 | `trace.json/md` |
| `case <predictions.jsonl>` | 分析预测样本，抽取好/坏案例 | `case_summary.json/md` |
| `doctor <run_dir> <config>` | 组合 inspect/trace/case 的综合结论 | `doctor.json/md` |
| `schedule <run_dir> <config>` | 给出后续实验计划 | `schedule.json/md` |
| `report <file...>` | 对任意文档/日志生成报告 | `report.json/md` |
| `auditdata <jsonl...>` | 生成数据审计报告 | `dataaudit.json/md` |

直接命令示例：

```bash
python scripts/llm.py inspect --run outputs/candidate --out outputs/llm/inspect_candidate --dry_run
python scripts/llm.py compare --a outputs/candidate --b outputs/candidate_from_rnafm_pretrain --out outputs/llm/compare --dry_run
python scripts/llm.py diagnose --run outputs/candidate_from_rnafm_pretrain --out outputs/llm/diagnose --dry_run
python scripts/llm.py agent_audit --out outputs/llm_agent_audit
```

## 7. Safe Commands

Agent 允许少量低风险命令在 shell 内触发：

```text
agent> run smoke
agent> run audit
agent> 编译 agent
```

对应实际命令：

```bash
python main.py smoke
python scripts/audit.py clean --out outputs/clean
python -m compileall -q agent scripts/llm.py agent.py
```

这些命令只用于快速健康检查。完整训练和 benchmark 仍不应由 Agent 自动执行。

## 8. 训练确认门

输入：

```text
agent> train candidate
```

Agent 不会立刻训练，而是提示：

```text
Training is blocked by default. To proceed, reply exactly: 进行训练 candidate
```

只有下一轮输入完全匹配：

```text
agent> 进行训练 candidate
```

才会进入训练确认路径。即使如此：

- 在 `dry_run` 模式下，只会输出计划命令。
- 在 live 模式下，也只允许固定命令：`python main.py train --config config/candidate.yaml --device cuda`。
- 训练会写 `outputs/candidate/`，所以正式训练建议在远程 GPU 服务器手动执行。

## 9. Benchmark 安全策略

输入：

```text
agent> benchmark candidate
```

Agent 会阻止自动 benchmark。即使回复确认，也只会提示手动运行命令。原因是 benchmark 会产生正式指标文件，必须由用户确认 checkpoint、split、decode 参数后手动执行。

推荐人工命令：

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```

## 10. RNA-FM 蒸馏相关用法

Agent 只做实验分析，不参与 RNA-FM embedding 提取、训练或推理。

常用分析：

```text
agent> compare outputs/candidate outputs/candidate_from_rnafm_pretrain
agent> diagnose outputs/candidate_from_rnafm_pretrain
agent> report outputs/seq_pretrain_rnafm/trainlog.jsonl outputs/candidate_from_rnafm_pretrain/trainlog.jsonl
```

对应 dry-run 直接命令：

```bash
python scripts/llm.py compare --a outputs/candidate --b outputs/candidate_from_rnafm_pretrain --out outputs/llm/compare_candidate_vs_rnafm --dry_run
python scripts/llm.py diagnose --run outputs/candidate_from_rnafm_pretrain --out outputs/llm/diagnose_rnafm --dry_run
```

注意：

- RNA-FM 只是 frozen representation teacher。
- Agent 不使用 RNA-FM 生成结构。
- Agent 不生成 pseudo labels。
- Agent 不修改 labels、predictions、metrics。

## 11. 输出目录结构

默认交互输出：

```text
outputs/llm_shell/
  history.jsonl
  memory.jsonl
  turns/
    0001/
      request.json
      prompt.md
      result.md
      blocked.md
```

| 文件 | 含义 |
|---|---|
| `history.jsonl` | 每轮输入、解析命令、状态摘要 |
| `memory.jsonl` | Agent 简短记忆，自动脱敏 |
| `turns/<id>/prompt.md` | 本轮生成给 LLM 的 prompt |
| `turns/<id>/request.json` | 本轮结构化输入 |
| `turns/<id>/blocked.md` | 被安全策略阻止的原因 |
| `usage.md/json` | `/usage` 生成的使用统计 |
| `cleanup_report.md/json` | `/cleanup` 生成的清理报告 |

这些都在 `outputs/` 下，默认不提交。

## 12. Live LLM 模式

默认 dry-run 不需要 API key。需要真实 LLM 分析时，设置：

```text
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

然后：

```bash
python scripts/llm.py agent --out outputs/llm_shell_live
```

或在 shell 里：

```text
agent> /live
```

live 模式只影响是否调用 LLM API，不解除安全策略。

可选限制：

```bash
python scripts/llm.py agent --max_api_calls 20 --max_tokens_total 20000 --timeout 60
```

环境变量等价项：

```text
LLM_MAX_API_CALLS=20
LLM_MAX_TOKENS_TOTAL=20000
LLM_TIMEOUT=60
LLM_COMMAND_TIMEOUT=120
LLM_TRAIN_TIMEOUT=0
LLM_MAX_TURNS=100
LLM_MAX_RETRIES=2
LLM_MAX_SAME_PROMPT=2
LLM_MAX_IDLE_SECONDS=120
```

## 13. 远程服务器工作流

Agent 可以给出远程训练计划，但不会保存密码，也不会自动 SSH。

```text
agent> remote login
agent> server training
```

推荐手动流程：

```bash
ssh -p 49018 root@connect.nmb1.seetacloud.com
cd /root/autodl-tmp/RNA-OmniDiffusion
git pull origin main
python main.py smoke
python main.py train --config config/candidate.yaml --device cuda
```

RNA-FM 预训练流程：

```bash
python scripts/extract_rnafm_embeddings.py \
  --input dataset/archive/train.jsonl \
  --output_jsonl dataset/unlabeled/train_seq_rnafm.jsonl \
  --output_npy dataset/teacher_emb/rnafm/train_embeddings.npy \
  --model_dir external \
  --batch_size 8 \
  --max_length 512 \
  --pool mean \
  --dtype float16 \
  --device cuda \
  --overwrite

python main.py train --config config/seq_pretrain_rnafm.yaml --device cuda
python main.py train --config config/candidate_from_rnafm_pretrain.yaml --device cuda
```

## 14. 调参目标

设置目标：

```text
agent> set target pair_f1 >= 0.75 max_trials 3
```

查看目标：

```text
agent> show target
```

清除目标：

```text
agent> clear target
```

生成试验计划：

```text
agent> start target training candidate
```

Agent 只会生成计划和 trial config 路径，不会修改 `config/candidate.yaml`。试验配置应写入 `outputs/llm_shell/tuning/` 或 `outputs/trials/`。

## 15. 常见问题

**Q: 为什么输入 benchmark 被阻止？**  
A: Benchmark 会写正式指标，Agent 只负责提示和诊断，不自动生成或覆盖指标。

**Q: 为什么输入 git push 被阻止？**  
A: Agent shell 是实验助手，不做版本控制操作。提交和推送应由开发者在普通 shell 中执行。

**Q: 为什么训练默认不执行？**  
A: 训练会写 checkpoint 和 outputs，需要显式确认；完整训练建议在远程 GPU 手动运行。

**Q: dry-run 和 read-only 有什么区别？**  
A: dry-run 控制是否调用 LLM API/是否执行确认后的低风险命令；read-only 是安全策略，禁止高风险工作流。

**Q: Agent 会不会影响模型结果？**  
A: 不会。Agent 不进入 `models/training.py`、`models/omni.py` forward、eval decode 或 benchmark inference。

## 16. 推荐日常流程

本地：

```text
agent> run smoke
agent> run audit
agent> inspect candidate
agent> compare outputs/candidate outputs/candidate_from_rnafm_pretrain
```

远程训练后：

```text
agent> diagnose outputs/candidate_from_rnafm_pretrain
agent> report outputs/candidate_from_rnafm_pretrain/trainlog.jsonl outputs/candidate_from_rnafm_pretrain/benchmark.json
agent> schedule outputs/candidate_from_rnafm_pretrain config/candidate_from_rnafm_pretrain.yaml
```

提交前：

```bash
python scripts/llm.py agent_audit --out outputs/llm_agent_audit
python scripts/audit.py clean --out outputs/clean
```
