# RNA-OmniDiffusion Candidate — Reproduction Guide

## Environment

```bash
# Python 3.12+, PyTorch 2.x, CUDA
pip install torch pyyaml
```

## Smoke Test

```bash
python main.py smoke
# Expected: smoke_ok, bp_f1_vs_toy=1.0000
```

## Train Candidate Model

```bash
python main.py train --config config/candidate.yaml --device cuda --max_steps 2000
# Output: outputs/candidate/best.pt
```

## Benchmark (ArchiveII, strict Nussinov)

```bash
python scripts/eval.py bench \
  --config config/candidate.yaml \
  --ckpt outputs/candidate/best.pt \
  --split test \
  --device cuda \
  --decode nussinov \
  --stage_logits \
  --workers 8 \
  --chunksize 2 \
  --profile
# Expected: Pair F1 ≈ 0.57, Valid = 1.0
```

## External Generalization (bpRNA)

```bash
# Download and prepare bpRNA
python scripts/dataset.py download --name bprna --out dataset/raw/bprna
python scripts/dataset.py prepare --input dataset/raw/bprna --out dataset/processed/bprna/clean.jsonl
python scripts/dataset.py split --input dataset/processed/bprna/clean.jsonl --out dataset/processed/bprna_random --mode random

# Train on bpRNA
python main.py train --config config/external_bprna_candidate.yaml --device cuda --max_steps 2000

# Benchmark on bpRNA
python scripts/eval.py bench \
  --config config/external_bprna_candidate.yaml \
  --ckpt outputs/external_bprna_candidate/best.pt \
  --split test \
  --device cuda \
  --decode nussinov \
  --stage_logits \
  --workers 8 --chunksize 2 --profile
# Expected: Pair F1 ≈ 0.53, Valid = 1.0
```

## Run External Controls

```bash
python scripts/run.py external \
  --configs config/external_bprna_candidate.yaml config/external_bprna_norefine.yaml config/external_bprna_oldbase.yaml \
  --dataset bprna \
  --split random \
  --device cuda \
  --decode nussinov \
  --bench_workers 8 \
  --tag external_bprna
```

## Component Ablation

```bash
# No pairrefine control
python main.py train --config config/candidate_norefine.yaml --device cuda --max_steps 2000

# Old masking control
python main.py train --config config/candidate_oldmask.yaml --device cuda --max_steps 2000

# Historical baseline
python main.py train --config config/oldbase.yaml --device cuda --max_steps 2000
```
