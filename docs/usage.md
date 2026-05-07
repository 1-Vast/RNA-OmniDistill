# RNA-OmniDiffusion Usage Guide

## Quick Checks

```bash
python -m py_compile main.py models/agent/analyzer.py scripts/llm.py scripts/audit.py
python main.py smoke
python scripts/audit.py clean --out outputs/clean
```

## Direct CLI Training

CPU preflight:

```bash
python main.py train --config config/candidate.yaml --device cpu
```

GPU training:

```bash
python main.py train --config config/candidate.yaml --device cuda
```

Do not edit `config/candidate.yaml` for Agent-driven default training. Check the configured output directory before trusting a new run.

## Agent-Assisted Training

Start the shell:

```bash
python scripts/llm.py agent --dry_run
```

Example:

```text
agent> train candidate
agent> run train candidate
```

In dry-run mode the Agent only prints the planned command. In live mode, candidate training still requires explicit confirmation and is limited to:

```bash
python main.py train --config config/candidate.yaml --device cuda
```

Benchmark execution remains blocked by default.

## Local Training Device

```text
agent> set training device
agent> local
agent> device cuda
```

The Agent reports the local command and suggests running smoke before live training.

## Remote Training Device

Use a login template only:

```text
agent> remote
agent> ssh -p <PORT> <USER>@<HOST>
```

Passwords must be entered manually in the terminal. Do not store passwords in `.env`, Agent history, reports, prompts, or documentation.

After login, run manually:

```bash
cd /path/to/RNA-OmniDiffusion
python main.py smoke
python main.py train --config config/candidate.yaml --device cuda
```

## Agent Memory

The Agent records lightweight tuning history and conclusions in:

```text
outputs/llm_shell/memory.jsonl
```

Memory entries include target metrics, generated tuning plans, confirmed training status, and analysis prompt/report pointers. They do not store passwords, API keys, labels, predictions, checkpoints, or benchmark metric rewrites. Use:

```text
agent> /memory
agent> view memory
```

## Target Tuning Plans

```text
agent> set target pair_f1 >= 0.75 max_trials 3
agent> show target
agent> start target training candidate
```

The Agent writes dry-run tuning plans and config copies only under:

```text
outputs/llm_shell/tuning/
```

It does not modify `config/candidate.yaml`, does not write release files, and does not run full benchmark automatically.

Allowed target metrics:

- `pair_f1`
- `pair_precision`
- `pair_recall`
- `valid_structure_rate`
- `all_dot_ratio`
- `loss`
- `pair_count_ratio`
- `rankAcc`
- `pair_logit_gap`

`max_trials` must be between 1 and 10.
