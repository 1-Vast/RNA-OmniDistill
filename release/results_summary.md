# RNA-OmniPrefold Results Summary

## Current Mainline

The maintained mainline is relation-aware masked denoising with pair-relation field prediction, lightweight 2D refinement, and strict Nussinov decoding.

## Archived Candidate Results

### ArchiveII Main Results

| Model | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio |
|---|---:|---:|---:|---:|---:|---:|
| oldbase | 0.3846 | 0.3398 | 0.4465 | 0.3864 | 1.0000 | 1.4213 |
| norefine | 0.4966 | 0.4470 | 0.5630 | 0.4485 | 1.0000 | 1.3913 |
| candidate | 0.5689 | 0.5090 | 0.6517 | 0.5729 | 1.0000 | 1.3808 |

### bpRNA External Random Split

| Model | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| oldbase | 0.4234 | 0.4019 | 0.4741 | 0.4335 | 1.0000 | 1.40 | 12,732 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 0.4451 | 1.0000 | 1.42 | 12,732 |
| candidate | 0.5285 | 0.4877 | 0.6070 | 0.5344 | 1.0000 | 1.38 | 12,732 |

## Key Findings

1. Pair refinement is an effective module.
2. Strict Nussinov decoding guarantees valid non-crossing structures.
3. External bpRNA generalization is positive but not a family-disjoint claim.
4. Negative results and removed branches are documented in `docs/negative.md`.
