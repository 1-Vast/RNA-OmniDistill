# Dataset Processing And Splits

This document describes the RNA-OmniPrefold data pipeline. The mainline separates sequence-only pretraining data from supervised structure data.

## Dataset Roles

| Dataset | Role | Structure labels | Notes |
|---|---|---:|---|
| Rfam | Sequence-only pretraining | No | FASTA/region sources can be converted to unlabeled JSONL. |
| RNAcentral | Sequence-only pretraining | No | Large unlabeled sequence source. |
| bpRNA | External benchmark or supervised splits | Yes | Keep benchmark/test splits out of pretraining. |
| ArchiveII | Supervised training and benchmark | Yes | Local small split is used for smoke and compact tests. |

## JSONL Formats

Supervised JSONL:

```json
{"id":"sample_1","seq":"GGGAAACCC","struct":"(((...)))","family":"example"}
```

Sequence-only JSONL:

```json
{"id":"seq_1","seq":"GGGAAACCC","family":"example"}
```

Use `allow_unlabeled: true` for sequence-only pretraining configs.

## Split Principles

- Pretraining uses sequence-only data.
- Supervised training and benchmark use structure-labeled data.
- External benchmark splits must not be used in pretraining or fine-tuning.
- Generated JSONL, processed datasets, and checkpoints remain ignored by git.

## Commands

Inspect data-processing help:

```bash
python scripts/data.py --help
python scripts/data.py prep_rfam_fasta --help
python scripts/check_datasets.py --help
```

Train sequence-only pretraining:

```bash
python main.py train --config config/seq_pretrain.yaml --device cuda
```

Fine-tune from sequence pretraining:

```bash
python main.py train --config config/candidate_from_seq_pretrain.yaml --device cuda
```

## Repository Boundaries

Do not commit generated data:

- `dataset/`
- `outputs/`
- `checkpoints/`
- `*.npy`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.safetensors`
