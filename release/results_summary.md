# RNA-OmniDiffusion Candidate — Results Summary

## Table 1: ArchiveII Main Results

| Model | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio |
|---|---:|---:|---:|---:|---:|---:|
| oldbase | 0.3846 | 0.3398 | 0.4465 | 0.3864 | 1.0000 | 1.4213 |
| norefine | 0.4966 | 0.4470 | 0.5630 | 0.4485 | 1.0000 | 1.3913 |
| **candidate** | **0.5689** | **0.5090** | **0.6517** | **0.5729** | **1.0000** | **1.3808** |

## Table 2: bpRNA External Random

| Model | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| oldbase | 0.4234 | 0.4019 | 0.4741 | 0.4335 | 1.0000 | 1.40 | 12,732 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 0.4451 | 1.0000 | 1.42 | 12,732 |
| **candidate** | **0.5285** | **0.4877** | **0.6070** | **0.5344** | **1.0000** | **1.38** | 12,732 |

## Table 3: Component Contribution

| Comparison | Delta F1 | Interpretation |
|---|---:|---|
| candidate − norefine (ArchiveII) | +0.0723 | pairrefine contribution |
| candidate − oldmask (ArchiveII) | +0.1716 | disabling masking contribution |
| candidate − oldbase (ArchiveII) | +0.1843 | full candidate improvement |
| candidate − norefine (bpRNA) | +0.0886 | external pairrefine contribution |
| candidate − oldbase (bpRNA) | +0.1051 | external full improvement |

## Table 4: 3-Seed Stability (ArchiveII)

| Seed | Pair F1 | Precision | Recall | Valid | Pair Ratio |
|---|---:|---:|---:|---:|---:|
| 42 | 0.5749 | 0.5145 | 0.6585 | 1.0 | 1.3805 |
| 43 | 0.5900 | 0.5326 | 0.6685 | 1.0 | 1.3663 |
| 44 | 0.5789 | 0.5195 | 0.6612 | 1.0 | 1.3846 |
| **Mean ± Std** | **0.5813 ± 0.0078** | — | — | — | — |

## Key Findings

1. **pairrefine** is the primary effective module (+7.2% F1 on ArchiveII, +20.1% on bpRNA)
2. **Disabling masking variants** is critical (-17.2% F1 penalty when enabled)
3. **Strict Nussinov decoding** guarantees valid non-crossing structures
4. **External generalization** holds: only 7.1% F1 drop on 37x larger bpRNA dataset
5. **Seed stability** confirmed: std F1 = 0.0078 across 3 seeds
