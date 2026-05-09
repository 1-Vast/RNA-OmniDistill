# Negative Results

This document records experimental branches that were tested but are excluded from the RNA-OmniPrefold mainline.

## Sequence-Level Representation Distillation (RNA-FM)

Sequence-level teacher distillation (RNA-FM) was tested as an optional sequence-pretraining signal and has been removed from the mainline. All RNA-FM related code (models/teacher, scripts/extract_rnafm_embeddings.py, RNA-FM configs) has been deleted from the main branch.

Evidence:

- Local toy comparison on 2026-05-09: D-only and teacher-distilled pretraining both reached the same strict Nussinov test Pair F1 of 0.8333 on the 2-sample toy test split.
- In that toy run, fine-tune best validation loss was slightly worse with teacher distillation: 1.0245 vs 0.9643 for D-only.
- In the earlier Rfam 50k single-seed comparison, the isolated teacher contribution was small: Pair F1 0.5969 vs 0.5925 for D-only, a +0.44pp difference.

Conclusion: RNA-FM sequence-level representation distillation is no longer part of this repository. Deleted on 2026-05-09 (commit 6e9e468).

## LLM Semantic Conditioning

LLM semantic token conditioning was tested as an auxiliary conditioning path, where a general-purpose model generated sample-level biological hints that were injected as condition tokens.

Results:

| Model | Pair F1 | Precision | Recall | Valid |
|---|---:|---:|---:|---:|
| Baseline supervised | 0.5723 | 0.5246 | 0.6322 | 1.0000 |
| LLM semantic tokens | 0.3851 | 0.3499 | 0.4299 | 1.0000 |

Likely causes:

- Low coverage: most samples used fallback unknown semantic tokens.
- Fallback token noise changed the input distribution.
- Sample-level semantic hints were poorly aligned with pair-level structure prediction.
- The model can already derive many simple sequence properties directly from sequence tokens.

Conclusion: LLM semantic tokens are not part of the mainline.

## Token-Only Decode

Token-only decoding without strict constraint projection produced invalid structures in diagnostic runs.

Conclusion: strict Nussinov projection remains part of the default inference path.

## Greedy Decode As Final Metric

Greedy pair decoding can be useful as a pair-head probe but may produce invalid or over-paired outputs.

Conclusion: final benchmark metrics should use strict Nussinov decoding.

## Conflict Loss

Conflict loss was tested as a precision-oriented regularizer and did not produce a reliable improvement over the baseline pair loss.

Conclusion: conflict loss is excluded from the main training objective.

## Masking Variants

Pair-aware masking and motif-span masking variants were tested in ablations. Current evidence is inconclusive or negligible on ArchiveII.

Conclusion: they are not claimed as main contributions.
