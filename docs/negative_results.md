# Negative Results — RNA-OmniDistill

This document records experimental branches that were tested but excluded from the RNA-OmniDistill mainline.

---

## LLM Semantic Conditioning

**Date**: 2026-05

LLM semantic token conditioning was tested as an auxiliary conditioning path, where a general-purpose LLM (DeepSeek-chat) performed 3-hop biological reasoning on RNA sequences to generate structured semantic tokens (GC level, family hints, structural tendency priors). These tokens were injected as condition tokens (segment_id=3) into the relational masked diffusion model.

### Experiment Setup

- Model: RNA-OmniDistill, 10 epochs
- Dataset: ArchiveII test split
- LLM: DeepSeek-chat, temperature=0
- Semantic tokens: 200 real API-generated annotations + 2892 `<SEM_UNKNOWN>` fallback tokens
- Baseline: supervised training without semantic tokens

### Results

| Model | Pair F1 | Precision | Recall | Valid |
|-------|---------|-----------|--------|-------|
| Baseline (supervised) | 0.5723 | 0.5246 | 0.6322 | 1.0000 |
| LLM-Semantic | 0.3851 | 0.3499 | 0.4299 | 1.0000 |
| Delta | -18.7pp | -17.5pp | -20.2pp | — |

### Root Cause Analysis

1. **Sparse coverage**: Only 200/3092 (6.5%) training samples had real API-generated semantic tokens. The remaining 2892 samples used `<SEM_UNKNOWN>` fallback, which acted as noise rather than signal.
2. **Vocabulary mismatch**: LLM-generated family/motif hints did not align well with the model's learned token distribution.
3. **Information redundancy**: The LLM's biological reasoning (GC content, length, complementarity) provides information the model can already derive from the RNA sequence itself.
4. **Input distribution shift**: The extra segment_id=3 condition tokens altered the input distribution, interfering with the trained pair-relation head.

### Conclusion

LLM semantic token conditioning is excluded from the RNA-OmniDistill mainline. The deprecated code is preserved in `models/deprecated/`, `scripts/deprecated/`, and `config/deprecated/` for reproducibility.

---

## Token-Only Decode

Token-only decoding (iterative unmasking without Nussinov constraint projection) produces zero valid structures on ArchiveII. The valid structure rate was 0.0 in diagnostic runs.

**Conclusion**: Strict Nussinov projection is an essential part of the inference path. Token-only decoding is excluded as a valid benchmark metric.

---

## Greedy Decode as Final Metric

Greedy pair decoding without Nussinov constraint projection produces inflated Pair F1 at the cost of invalid structures. It is retained only as a fast pair-head probe for debugging, not as a paper metric.

**Conclusion**: The final paper metric must use strict Nussinov decoding.

---

## Conflict Loss

Conflict loss (penalizing nucleotides that participate in more than one pair) was tested and did not produce a reliable improvement over the baseline pair loss.

**Conclusion**: Conflict loss is excluded from the main training objective.

---

## Masking Variants

Pair-aware masking and motif-span masking variants were tested in ablation studies. A seed repeat found the effect inconclusive or negligible on ArchiveII.

**Conclusion**: Standard random token masking is used for the mainline. Masking variants are not claimed as contributions.

---

## LLM Semantic Conditioning (Prompt-Based)

LLM semantic conditioning through prompt templates (as opposed to the LLM token injection described above) was also tested. The model did not benefit from prompt-based semantic priors.

**Conclusion**: Prompt-based LLM conditioning is excluded from the mainline.

---

*Last updated: 2026-05-08*
