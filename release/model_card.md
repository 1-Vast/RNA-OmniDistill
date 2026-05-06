# RNA-OmniDiffusion Candidate — Model Card

## Model Name
RNA-OmniDiffusion Candidate

## Core Design
- Discrete masked diffusion for RNA sequence-structure modeling
- Transformer backbone (8 layers, 512 hidden, 8 heads)
- MLP-based pair head for base-pair logit prediction
- **2D Conv2d pairrefine** residual refinement on pair logits
- **Strict Nussinov constrained decoding** for legal non-crossing structures
- Masking variants disabled (harmful on ArchiveII)
- Conflict loss rejected (harmful at all tested magnitudes)

## Current Best Results

### ArchiveII (in-domain)
| Metric | Value |
|---|---|
| Pair F1 | 0.5689 |
| Precision | 0.5090 |
| Recall | 0.6517 |
| MCC | 0.5729 |
| Valid Rate | 1.0000 |
| Pair Ratio | 1.3808 |
| Test Samples | 338 |

### ArchiveII 3-Seed Stability
| Metric | Value |
|---|---|
| Mean F1 | 0.5813 |
| Std F1 | 0.0078 |
| Seeds | 42, 43, 44 |

### bpRNA External Generalization (random split)
| Metric | Value |
|---|---|
| Pair F1 | 0.5285 |
| Precision | 0.4877 |
| Recall | 0.6070 |
| MCC | 0.5344 |
| Valid Rate | 1.0000 |
| Pair Ratio | 1.38 |
| Test Samples | 12,732 |
| External Drop vs ArchiveII | -7.1% |

### Component Contributions (ArchiveII)
| Comparison | Delta F1 | Interpretation |
|---|---|---|
| candidate − norefine | +0.0723 | pairrefine contribution |
| candidate − oldmask | +0.1716 | disabling masking contribution |
| candidate − oldbase | +0.1843 | full candidate improvement |

## Limitations
See `release/limitations.md`.

## Reproducibility
See `release/reproduce.md`.
