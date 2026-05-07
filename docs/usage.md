# RNA-OmniDiffusion Usage Guide

## Framework Overview

RNA-OmniDiffusion is a pair-refined, constraint-guided masked diffusion model for RNA secondary structure prediction.

### Core Model
- **RNAOmniDiffusion** — Transformer encoder with task, segment, time, and position embeddings.

### Heads
- **Token heads**: sequence head, structure head, and general fallback head for token prediction.
- **Pair head**: MLP that predicts base-pair logits over sequence positions.
- **Pair refine**: Optional 2D convolutional refinement over the pair-logit map.

### Training Tasks
- **seq2struct**: Input sequence → predict dot-bracket structure.
- **invfold**: Input structure → predict RNA sequence.
- **inpaint**: Mask spans, recover sequence/structure tokens.
- **motif_control**: Condition on motif/family tokens when enabled.

### Decoding
- **Strict Nussinov**: Converts pair logits into valid non-crossing dot-bracket structures.
- **Token decoding**: Iterative unmasking over structure tokens.
- **Hybrid decoding**: Combines token compatibility with pair logits.

### Commands
| Command    | Description |
|------------|-------------|
| `overview` | Show this framework overview |
| `models`   | Alias for overview |
| `train`    | Train from a YAML config |
| `eval`     | Evaluate validation split from checkpoint |
| `infer`    | Run single-sample inference |
| `smoke`    | Run tiny CPU sanity test |
| `params`   | Inspect adjustable config parameters |
| `agent`    | Show optional Agent shell usage |

## Quick Start

```bash
# Show framework overview
python main.py overview

# Run smoke test (no dataset needed)
python main.py smoke

# Inspect adjustable parameters
python main.py params --config config/candidate.yaml
```

## Direct Training

### Smoke Test
```bash
python main.py smoke
```
Creates a tiny synthetic dataset, runs 1 epoch with 2 steps, and prints a basic structure prediction check.

### Full Training
```bash
python main.py train --config config/candidate.yaml --device cuda
```

Options:
- `--config`: YAML config file (default: `config/candidate.yaml`)
- `--device`: `auto`, `cuda`, or `cpu` (default: `auto`)
- `--resume`: Path to checkpoint for resuming training
- `--max_steps`: Cap steps for sanity checks

Training output:
- Console: startup summary, per-epoch compact line, checkpoint notifications.
- `outputs/{name}/trainlog.jsonl`: Machine-readable JSONL log (one line per epoch).
- `outputs/{name}/best.pt`: Best checkpoint (by `save_best_by` metric).
- `outputs/{name}/last.pt`: Most recent checkpoint.

Do not edit `config/candidate.yaml` for default training. The file is used as the reference configuration.

## Direct Evaluation

```bash
python main.py eval --config config/candidate.yaml --ckpt outputs/candidate/best.pt --device cuda
```

Options:
- `--config`: Evaluation config (default: `config/base.yaml`)
- `--ckpt`: Checkpoint path (required)
- `--device`: `auto`, `cuda`, or `cpu`

## Direct Inference

### Sequence-to-Structure (seq2struct)
```bash
python main.py infer \
  --config config/candidate.yaml \
  --ckpt outputs/candidate/best.pt \
  --task seq2struct \
  --seq GCAUAGC \
  --device cuda
```

### Inverse Folding (invfold)
```bash
python main.py infer \
  --config config/candidate.yaml \
  --ckpt outputs/candidate/best.pt \
  --task invfold \
  --struct "((...))" \
  --device cuda
```

Options:
- `--config`: Inference config (default: `config/base.yaml`)
- `--ckpt`: Checkpoint path (required)
- `--task`: `seq2struct` or `invfold` (required)
- `--seq`: RNA sequence (required for seq2struct)
- `--struct`: Dot-bracket structure (required for invfold)
- `--device`: `auto`, `cuda`, or `cpu`

## Agent-Assisted Training

The optional LLM analysis agent is an experiment assistant, not a structure predictor. It reads existing artifacts and writes reports. In dry-run mode it only prints planned commands without calling any API.

### Start the Agent Shell
```bash
python scripts/llm.py agent --dry_run
```

### Common Commands
```text
agent> run smoke
agent> inspect candidate
agent> diagnose
agent> train candidate
agent> do train candidate
agent> set training device
agent> set target pair_f1 >= 0.75 max_trials 3
agent> /memory
agent> /usage
agent> /exit
```

### Training Safety
- Agent is read-only by default.
- Candidate training requires explicit confirmation (`do train candidate`).
- Benchmark execution remains blocked.
- Remote passwords are never stored or printed.

### Target Tuning
```text
agent> set target pair_f1 >= 0.75 max_trials 3
agent> show target
agent> start target training candidate
```

Allowed target metrics: `pair_f1`, `pair_precision`, `pair_recall`, `valid_structure_rate`, `all_dot_ratio`, `loss`, `pair_count_ratio`, `rankAcc`, `pair_logit_gap`. `max_trials` must be between 1 and 10.

The Agent writes tuning plans and config copies only under `outputs/llm_shell/tuning/`. It never modifies `config/candidate.yaml`.

## Remote Training

### SSH Login
```bash
ssh -p 49018 root@connect.nmb1.seetacloud.com
```

**Password**: Enter manually in the terminal when prompted. Do not store in code, docs, `.env`, shell scripts, or Agent history.

### After Login
```bash
cd /root/RNA-OmniDiffusion
python main.py smoke
python main.py train --config config/candidate.yaml --device cuda
```

### Syncing Code
Preferred method — remote `git pull`:
```bash
ssh -p 49018 root@connect.nmb1.seetacloud.com
cd /root/RNA-OmniDiffusion
git pull origin main
python main.py smoke
```

Alternative — local `rsync` (excludes outputs, datasets, checkpoints):
```bash
rsync -avz \
  -e "ssh -p 49018" \
  --exclude ".git" --exclude ".env" \
  --exclude "outputs" --exclude "dataset" --exclude "checkpoints" \
  --exclude "__pycache__" --exclude "*.pyc" \
  ./ root@connect.nmb1.seetacloud.com:/root/RNA-OmniDiffusion/
```

Do not upload `outputs/`, `dataset/`, `checkpoints/`, or `.env`. Passwords remain terminal-entry only.

## Suggested Workflow

1. **Smoke test**: `python main.py smoke` — verify code and environment.
2. **Inspect config**: `python main.py params --config config/candidate.yaml`
3. **Train candidate**: `python main.py train --config config/candidate.yaml --device cuda`
4. **Inspect output**: Check `outputs/candidate/trainlog.jsonl` and `best.pt`.
5. **Inference test**: `python main.py infer --config config/candidate.yaml --ckpt outputs/candidate/best.pt --task seq2struct --seq GCAUAGC`
6. **Benchmark** (only if needed):
   ```bash
   python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
   ```
7. **Agent diagnosis** (optional):
   ```bash
   python scripts/llm.py diagnose --run outputs/candidate --out outputs/llm/diagnose
   ```

Never update release metrics without provenance verification.

## Parameter Reference

```bash
python main.py params --config config/candidate.yaml
python main.py params --config config/candidate.yaml --section training
python main.py params --config config/candidate.yaml --json
```

Sections: `training`, `model`, `decoding`, `ablation`, `agent`.
