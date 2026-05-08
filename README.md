# RNA-OmniDistill

**Relational Masked Diffusion with Frozen RNA-FM Distillation for Constraint-Guided RNA Folding**

[![Framework](https://img.shields.io/badge/method-Relational_Masked_Diffusion-blue)]()
[![Stage](https://img.shields.io/badge/stage-research-orange)]()
[![Domain](https://img.shields.io/badge/domain-RNA_secondary_structure-green)]()

RNA-OmniDistill is a **relational masked diffusion framework** for RNA secondary structure prediction. It formulates RNA folding as **joint discrete denoising over sequence tokens, structure tokens, and pair-relation tokens**. A compact student encoder is pretrained on sequence-only RNA data using masked nucleotide denoising, with optional **frozen RNA-FM sequence-level representation distillation**. The pretrained encoder is then adapted to **supervised pair-relation adaptation** via a pair-relation head and lightweight 2D relation refinement. Final structures are obtained through **strict Nussinov constraint projection** into valid non-crossing dot-bracket space.

## Core Innovation

RNA-OmniDistill introduces three key ideas:

1. **Relational Masked Diffusion**: RNA folding is cast as joint denoising over sequence, structure, and pair-relation tokens — explicitly modeling base-pair interactions as relational variables in the diffusion process.

2. **Frozen RNA-FM Sequence-Level Distillation**: A frozen RNA-FM model provides global sequence representation guidance during unsupervised pretraining, without predicting structures, generating pseudo labels, or participating in inference.

3. **Constraint-Guided Relation Projection**: Predicted soft pair-relation fields are projected into valid structures via strict Nussinov dynamic programming, satisfying non-crossing, minimum loop length, and canonical/wobble pairing constraints.

## Overview

The model architecture:

- **nucleotide tokens** for RNA sequence denoising
- **structure tokens** for task-conditioned folding paths
- **pair-relation tokens** for explicit base-pair supervision
- **pair-relation field** refined by lightweight 2D convolution
- **strict Nussinov projection** for guaranteed valid structures

## Method Stages

### Stage 1: Sequence-Only Pretraining

- input: unlabeled RNA sequence
- objective: masked nucleotide denoising
- optional teacher: frozen RNA-FM sequence-level representation distillation
- no structural labels
- no pseudo pairs

The pretraining loss is:

```text
L_pretrain = L_denoise + lambda_d * L_distill
```

### Stage 2: Supervised Pair-Relation Adaptation

- encoder-only initialization from Stage 1
- pair-logit relation head
- pair-relation BCE / weighted pair loss
- lightweight 2D relation refinement
- optional token auxiliary objective when enabled by the config

The fine-tuning loss is:

```text
L_finetune = L_pair + optional token auxiliary
```

### Stage 3: Strict Nussinov Decoding

- canonical base-pair constraints
- minimum loop length
- non-crossing dynamic programming projection
- final dot-bracket structure output

## RNA-FM Teacher Boundary

RNA-FM does:

- serve as a frozen sequence-level representation teacher
- provide a continuous prior for sequence-only pretraining
- produce one mean-pooled embedding vector per sequence

RNA-FM does not:

- predict dot-bracket structures
- generate pair labels
- participate in benchmark inference
- replace RNA-OmniDistill

Offline teacher extraction writes mean-pooled embeddings to ignored local files:

```bash
python scripts/extract_rnafm_embeddings.py --input dataset/archive/train.jsonl --output_jsonl dataset/unlabeled/train_seq_rnafm.jsonl --output_npy dataset/teacher_emb/rnafm/train_embeddings.npy --dummy --limit 256 --embedding_dim 640 --overwrite
python scripts/extract_rnafm_embeddings.py --input dataset/archive/val.jsonl --output_jsonl dataset/unlabeled/val_seq_rnafm.jsonl --output_npy dataset/teacher_emb/rnafm/val_embeddings.npy --dummy --limit 64 --embedding_dim 640 --overwrite
```

Use `--dummy` only for pipeline smoke tests. Real RNA-FM loading is isolated in `models/teacher/rnafm_teacher.py`; the core student model does not import external RNA-FM code.

## Preliminary Results (2026-05, Seed 42)

All results on ArchiveII test split, strict Nussinov constraint projection:

| Pretrain Source | Teacher | Pair F1 | Precision | Recall | Δ Baseline |
|---|---:|---:|---:|---:|---:|
| None (supervised) | — | 0.5762 | 0.5324 | 0.6302 | — |
| Rfam 50k | — | 0.5925 | 0.5499 | 0.6463 | +1.63pp |
| Rfam 50k | RNA-FM | 0.5969 | 0.5504 | 0.6556 | +2.07pp |
| bpRNA 50k | RNA-FM | 0.5998 | 0.5561 | 0.6546 | +2.36pp |
| **RNAcentral 50k** | **RNA-FM** | **0.6171** | **0.5794** | **0.6640** | **+4.09pp** |

These are single-run results. Seed repeats and external benchmarks are needed before claiming stable improvements.

## Experiment Paths

- baseline supervised candidate
- D-only sequence pretraining (no teacher)
- D-RNAFM frozen RNA-FM distillation
- Cross-source pretraining: Rfam / bpRNA / RNAcentral
- Seed repeat (42, 43, 44, 45, 46, 47)
- External benchmark (bpRNA-1m(90))
- Relation refinement ablation
- Low-label adaptation

See [docs/experiment_plan.md](docs/experiment_plan.md) for the full experiment matrix.

## Quick Start

```bash
python main.py overview
python main.py smoke
python main.py params --config config/candidate.yaml
python main.py train --config config/candidate.yaml --device cuda
python main.py train --config config/seq_pretrain.yaml --device cuda
python main.py train --config config/candidate_from_seq_pretrain.yaml --device cuda
python main.py train --config config/seq_pretrain_rnafm.yaml --device cuda
python main.py train --config config/candidate_from_rnafm_pretrain.yaml --device cuda
```

Strict Nussinov benchmark:

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```

External bpRNA comparison:

```bash
python scripts/run.py external --configs config/candidate.yaml --dataset bprna --split random --device cuda --decode nussinov --bench_workers 8 --tag external_bprna
```

## Core Structure

```text
main.py
models/
  omni.py
  training.py
  dataset.py
  collator.py
  decode.py
  teacher/rnafm_teacher.py
scripts/
  extract_rnafm_embeddings.py
  data.py
  eval.py
  run.py
  audit.py
config/
  candidate.yaml
  seq_pretrain.yaml
  candidate_from_seq_pretrain.yaml
  seq_pretrain_rnafm.yaml
  candidate_from_rnafm_pretrain.yaml
docs/
  rna_omnidistill.md
  dataset_processing_and_splits.md
  usage.md
```

## Data Paths

Data processing and experiment split notes are in [docs/dataset_processing_and_splits.md](docs/dataset_processing_and_splits.md).

Important local artifacts are ignored by git:

- `outputs/`
- `dataset/unlabeled/`
- `dataset/teacher_emb/`
- large `dataset/processed/` files
- checkpoints and tensor dumps (`*.pt`, `*.pth`, `*.ckpt`, `*.npy`, `*.safetensors`)
- `external/model.safetensors`
- `.env`

## What Not To Claim

- Not an LLM-powered predictor.
- Does not use LLM semantic tokens in the mainline.
- Does not use token-level RNA-FM distillation.
- Does not use RNA-FM pair priors.
- Does not use RNA-FM or LLM pseudo-structure labels.
- Does not use an Agent during training or inference.
- Not a foundation model.
- Not RNA-FM structure prediction.

## Negative Results

See [docs/negative_results.md](docs/negative_results.md) for excluded experimental branches, including LLM semantic conditioning, token-only decode, greedy decode as final metric, conflict loss, and masking variants.
- no pseudo pair labels
- no RNA-FM structural prior

## Current Limitations

- Precision is still lower than recall in the current candidate path.
- Family-disjoint generalization is not established as a final claim.
- RNA-FM distillation requires seed repeats and larger unlabeled pretraining before being treated as stable evidence.
- Strict Nussinov projection is part of the system, not an optional post-hoc decoration.
