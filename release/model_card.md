# RNA-OmniDiffusion Candidate Model Card

## Model Name

RNA-OmniDiffusion Candidate

## Intended Use

RNA secondary structure prediction from RNA sequence input. The validated inference path is sequence-to-structure prediction with strict Nussinov decoding.

## Core Design

- Masked discrete diffusion over RNA sequence and structure tokens.
- Bidirectional Transformer encoder: 8 layers, 512 hidden size, 8 heads.
- MLP pair head for base-pair logits.
- 2D Conv2d residual pair refinement over the pair-logit matrix.
- Strict Nussinov constrained decoding for legal non-crossing structures.
- Staged-logit benchmark path for efficient strict evaluation.

## Current Best Results

### ArchiveII

| Metric | Value |
|---|---:|
| Pair F1 | 0.5689 |
| Precision | 0.5090 |
| Recall | 0.6517 |
| MCC | 0.5729 |
| Valid Rate | 1.0000 |
| Pair Ratio | 1.3808 |
| Test Samples | 338 |

### ArchiveII 3-Seed Stability

| Metric | Value |
|---|---:|
| Mean F1 | 0.5813 |
| Std F1 | 0.0078 |
| Seeds | 42, 43, 44 |

### bpRNA External Random Split

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

## Unsupported Claims

- Token-only decoding is not structurally valid.
- Greedy decoding is not a final benchmark metric.
- Masking variants are not a main contribution on ArchiveII.
- Conflict loss is not a positive result in current experiments.
- LLM semantic conditioning is not a positive result.

## Limitations

See [limits.md](limits.md).

## Paper Framework

See [paper.md](paper.md).
