# RNA-OmniDistill Usage Guide

`main.py` is the local CLI entry point. Run `python main.py overview` for the framework map and `python main.py params --config config/candidate.yaml` for tunable parameters.

## Framework Overview

RNA-OmniDistill is a relation-aware masked diffusion model for constraint-guided RNA folding.

### Core Model

- `RNAOmniDiffusion`: Transformer student encoder with task, segment, time, and position embeddings.
- Token heads: sequence, structure, and general fallback prediction heads.
- Pair-relation head: predicts pair logits over RNA sequence positions.
- 2D relation refinement: lightweight convolutional refinement over the pair-relation field.
- Nussinov decoder: strict projection into valid non-crossing dot-bracket structures.

### Commands

| Command | Description |
| --- | --- |
| `overview` | Show the model framework map |
| `models` | Alias for overview |
| `train` | Train from a YAML config |
| `eval` | Evaluate validation split from checkpoint |
| `infer` | Run single-sample inference |
| `smoke` | Run tiny CPU/GPU sanity test |
| `params` | Inspect tunable parameters |

## Quick Start

```bash
python main.py overview
python main.py smoke
python main.py params --config config/candidate.yaml
```

## Direct Training

Smoke test:

```bash
python main.py smoke
```

Canonical supervised training:

```bash
python main.py train --config config/candidate.yaml --device cuda
```

Sequence-only masked denoising pretraining:

```bash
python main.py train --config config/seq_pretrain.yaml --device cuda
python main.py train --config config/candidate_from_seq_pretrain.yaml --device cuda
```

Frozen RNA-FM sequence-level distillation path:

```bash
python main.py train --config config/seq_pretrain_rnafm.yaml --device cuda
python main.py train --config config/candidate_from_rnafm_pretrain.yaml --device cuda
```

Do not edit `config/candidate.yaml` for default supervised training. It is the reference configuration.

## Direct Evaluation

```bash
python main.py eval --config config/candidate.yaml --ckpt outputs/candidate/best.pt --device cuda
```

Strict Nussinov benchmark:

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```

Decode-only scan from staged logits:

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --decode_only --workers 8 --chunksize 2 --profile --scan config/scan.json
```

## Direct Inference

Sequence-to-structure:

```bash
python main.py infer --config config/candidate.yaml --ckpt outputs/candidate/best.pt --task seq2struct --seq GCAUAGC --device cuda
```

Inverse folding:

```bash
python main.py infer --config config/candidate.yaml --ckpt outputs/candidate/best.pt --task invfold --struct "((...))" --device cuda
```

## RNA-FM Frozen Teacher Distillation

RNA-FM is an optional frozen representation teacher for sequence-only pretraining. It provides sequence-level mean-pooled embeddings. It does not generate structures, pseudo pairs, labels, predictions, benchmark metrics, or inference outputs.

Generate smoke-test dummy embeddings:

```bash
python scripts/extract_rnafm_embeddings.py --input dataset/archive/train.jsonl --output_jsonl dataset/unlabeled/train_seq_rnafm.jsonl --output_npy dataset/teacher_emb/rnafm/train_embeddings.npy --dummy --limit 256 --embedding_dim 640 --overwrite
python scripts/extract_rnafm_embeddings.py --input dataset/archive/val.jsonl --output_jsonl dataset/unlabeled/val_seq_rnafm.jsonl --output_npy dataset/teacher_emb/rnafm/val_embeddings.npy --dummy --limit 64 --embedding_dim 640 --overwrite
```

Generated `.npy` embeddings and checkpoints are ignored by git.

## Data Processing

Data preparation details are in `docs/dataset_processing_and_splits.md`.

Useful help commands:

```bash
python scripts/data.py prep_rfam_fasta --help
python scripts/extract_rnafm_embeddings.py --help
python scripts/run.py summarize --help
```

## Local Artifact Boundaries

Do not commit:

- `outputs/`
- `dataset/teacher_emb/`
- `dataset/unlabeled/`
- large `dataset/processed/` files
- `external/model.safetensors`
- `bpRNA_1m_90.zip`
- `rnacentral_active.fasta.gz`
- `*.npy`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `.env`

## Suggested Workflow

1. `python main.py smoke`
2. `python main.py params --config config/candidate.yaml`
3. `python main.py train --config config/candidate.yaml --device cuda`
4. Inspect `outputs/candidate/trainlog.jsonl` and `best.pt` locally.
5. Run benchmark only after provenance checks.

Never update release metrics without provenance verification.
