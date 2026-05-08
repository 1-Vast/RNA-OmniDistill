# RNA-OmniDistill Candidate Model Card

## Model Name

RNA-OmniDistill Candidate

## Method Family

Relational Masked Diffusion

## Intended Use

RNA secondary structure prediction from RNA sequence input. The validated inference path is sequence-to-structure prediction with strict Nussinov decoding.

This is a research prototype for RNA secondary structure prediction only. It is not intended for clinical use, structural biology discovery, or any production deployment.

## Core Design

- Masked discrete diffusion over RNA sequence and structure tokens.
- Bidirectional Transformer encoder: 8 layers, 512 hidden size, 8 heads.
- MLP pair head for base-pair logits.
- 2D Conv2d residual pair refinement over the pair-logit matrix.
- Strict Nussinov constrained decoding for legal non-crossing structures.
- Staged-logit benchmark path for efficient strict evaluation.

## Current Results (2026-05)

| Pretrain Source | Teacher | Pair F1 | Precision | Recall |
|---|---|---|---|---|
| None (supervised baseline) | none | 0.5762 | 0.5324 | 0.6302 |
| Rfam 50k | none | 0.5925 | 0.5499 | 0.6463 |
| Rfam 50k | RNA-FM | 0.5969 | 0.5504 | 0.6556 |
| bpRNA 50k | RNA-FM | 0.5998 | 0.5561 | 0.6546 |
| RNAcentral 50k | RNA-FM | 0.6171 | 0.5794 | 0.6640 |

All results on ArchiveII test split, strict Nussinov decoding, seed 42.
RNAcentral 50k D-RNAFM is the current best configuration.

RNA-FM is a frozen sequence-level representation teacher. It is not used for structure prediction, token-level distillation, or benchmark inference.

DeepSeek Agent is a read-only experiment assistant only.

## Archived Results

### ArchiveII (previous candidate)

| Metric | Value |
|---|---:|
| Pair F1 | 0.5689 |
| Precision | 0.5090 |
| Recall | 0.6517 |
| MCC | 0.5729 |
| Valid Rate | 1.0000 |
| Pair Ratio | 1.3808 |
| Test Samples | 338 |

### ArchiveII 3-Seed Stability (previous candidate)

| Metric | Value |
|---|---:|
| Mean F1 | 0.5813 |
| Std F1 | 0.0078 |
| Seeds | 42, 43, 44 |

### bpRNA External Random Split (previous candidate)

| Metric | Value |
|---|---:|
| Pair F1 | 0.5285 |
| Precision | 0.4877 |
| Recall | 0.6070 |
| MCC | 0.5344 |
| Valid Rate | 1.0000 |
| Pair Ratio | 1.38 |
| Test Samples | 12,732 |
| External Drop vs ArchiveII | -7.1% |

## Supported Claims

- Pair refinement improves strict secondary structure prediction.
- Strict Nussinov decoding is required for valid non-crossing structures.
- The candidate generalizes from ArchiveII to a larger bpRNA random split with moderate degradation.
- Sequence-only pretraining (RNAcentral 50k) provides the largest gain over supervised baseline.

## Unsupported Claims

- Token-only decoding is not structurally valid.
- Greedy decoding is not a final benchmark metric.
- Masking variants are not a main contribution on ArchiveII.
- Conflict loss is not a positive result in current experiments.
- No language-model-based structure prediction or semantic conditioning is used.

## Limitations

See [limits.md](limits.md).

## Paper Framework

See [paper.md](paper.md).
