# RNA-OmniDistill Results Summary

## Pretraining Source Comparison

| Pretrain Source | Teacher | Pair F1 | Precision | Recall |
|---|---|---|---|---|
| None (supervised baseline) | none | 0.5762 | 0.5324 | 0.6302 |
| Rfam 50k | none | 0.5925 | 0.5499 | 0.6463 |
| Rfam 50k | RNA-FM | 0.5969 | 0.5504 | 0.6556 |
| bpRNA 50k | RNA-FM | 0.5998 | 0.5561 | 0.6546 |
| RNAcentral 50k | RNA-FM | 0.6171 | 0.5794 | 0.6640 |

All results on ArchiveII test split, strict Nussinov decoding, seed 42.
RNAcentral 50k D-RNAFM is the current best configuration.

RNAcentral 50k sequence-only pretraining provides the largest gain (+4.09pp Pair F1 over supervised baseline).

## Archived Results

### Table A1: ArchiveII Main Results (previous candidate)

| Model | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio |
|---|---:|---:|---:|---:|---:|---:|
| oldbase | 0.3846 | 0.3398 | 0.4465 | 0.3864 | 1.0000 | 1.4213 |
| norefine | 0.4966 | 0.4470 | 0.5630 | 0.4485 | 1.0000 | 1.3913 |
| candidate | 0.5689 | 0.5090 | 0.6517 | 0.5729 | 1.0000 | 1.3808 |

### Table A2: bpRNA External Random Split (previous candidate)

| Model | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oldbase | 0.4234 | 0.4019 | 0.4741 | 0.4335 | 1.0000 | 1.40 | 12,732 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 0.4451 | 1.0000 | 1.42 | 12,732 |
| candidate | 0.5285 | 0.4877 | 0.6070 | 0.5344 | 1.0000 | 1.38 | 12,732 |

### Table A3: Component Contribution (previous candidate)

| Comparison | Delta F1 | Interpretation |
|---|---:|---|
| candidate - norefine (ArchiveII) | +0.0723 | pair refinement contribution |
| candidate - oldmask (ArchiveII) | +0.1716 | disabling harmful masking variants |
| candidate - oldbase (ArchiveII) | +0.1843 | full candidate improvement |
| candidate - norefine (bpRNA) | +0.0886 | external pair refinement contribution |
| candidate - oldbase (bpRNA) | +0.1051 | external full improvement |

### Table A4: 3-Seed Stability on ArchiveII (previous candidate)

| Seed | Pair F1 | Precision | Recall | Valid | Pair Ratio |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.5749 | 0.5145 | 0.6585 | 1.0000 | 1.3805 |
| 43 | 0.5900 | 0.5326 | 0.6685 | 1.0000 | 1.3663 |
| 44 | 0.5789 | 0.5195 | 0.6612 | 1.0000 | 1.3846 |
| Mean +/- Std | 0.5813 +/- 0.0078 | - | - | - | - |

## Key Findings

1. Sequence-only pretraining (RNAcentral 50k) provides the largest gain over supervised baseline.
2. Pair refinement is the primary effective module.
3. Masking variants are not a supported main contribution on ArchiveII.
4. Strict Nussinov decoding guarantees valid non-crossing structures.
5. External bpRNA generalization is positive but still not a family-disjoint claim.
6. Seed stability is reasonable across seeds 42, 43, and 44.

## Negative Results (unchanged)

- Conflict loss did not improve results in current experiments.
- Token-only decoding produces structurally invalid outputs.
- Greedy decoding is not a valid benchmark metric.
- Masking variants do not provide a measurable benefit on ArchiveII.
