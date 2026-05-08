# RNA-OmniDistill

Relational Masked Diffusion with Frozen RNA-FM Distillation for Constraint-Guided RNA Folding.

RNA-OmniDistill is a relation-aware masked diffusion framework for RNA secondary structure prediction. It formulates RNA folding as a discrete denoising problem over nucleotide tokens, structure tokens, and pair-relation variables. A compact student encoder is pretrained on sequence-only RNA data using masked nucleotide denoising and frozen RNA-FM sequence-level representation distillation. The pretrained encoder is then adapted to supervised pair-relation prediction with a pair-logit relation head and lightweight 2D relation refinement. Final structures are obtained through strict Nussinov constraint projection.

RNA-OmniDistill 是一个关系感知的掩码离散扩散 RNA 折叠框架。它将 RNA 二级结构预测表述为核苷酸 token、结构 token 与碱基对关系变量的联合离散去噪问题。模型先在仅含序列的 RNA 数据上通过 masked nucleotide denoising 和冻结 RNA-FM sequence-level 表征蒸馏预训练紧凑 Student encoder，随后在真实结构标签上微调 pair-logit relation head 与轻量 2D relation refinement，最终通过严格 Nussinov 约束投影得到合法 RNA 二级结构。

## Overview

The model uses:

- nucleotide tokens for RNA sequence denoising
- structure tokens for task-conditioned folding/inverse-folding paths
- pair-relation variables for base-pair supervision
- a pair-relation field refined by lightweight 2D convolution
- strict Nussinov constraint projection for valid non-crossing dot-bracket output

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

## Experiments

Core experiment paths:

- baseline supervised candidate
- D-only sequence pretraining
- D-RNAFM frozen teacher distillation
- Rfam / bpRNA / RNAcentral data processing
- seed repeat experiments
- external benchmark experiments
- low-label experiments
- scale-up pretraining experiments

Common commands:

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

- not RNA-FM structure prediction
- not token-level distillation
- not pair prior
- not language-model semantic conditioning
- not a foundation model
- not a language-model-powered system
- no pseudo pair labels
- no RNA-FM structural prior

## Current Limitations

- Precision is still lower than recall in the current candidate path.
- Family-disjoint generalization is not established as a final claim.
- RNA-FM distillation requires seed repeats and larger unlabeled pretraining before being treated as stable evidence.
- Strict Nussinov projection is part of the system, not an optional post-hoc decoration.
