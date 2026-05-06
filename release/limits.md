# RNA-OmniDiffusion Candidate — Limitations

## Structural Validity
- **Valid = 1.0000 comes from strict Nussinov constrained decoding**, not from free token generation.
- The model does not freely generate legal RNA structures via token-only decoding.
- Nussinov DP guarantees non-crossing, one-pair-per-position structures.
- Removing Nussinov (token-only decode) results in valid_rate = 0.0000.

## Decoding Methods
- **Greedy decoding is a probe only** and must not be used as a final benchmark metric.
- Greedy decode can produce crossing pairs before dot-bracket conversion.

## Family-Disjoint Generalization
- **Family-disjoint generalization is not yet established.**
- bpRNA split is random (all families labeled "OTHER"), not family-disjoint.
- Rfam seed data has consensus structures (SS_cons) unsuitable as individual ground-truth.

## Multi-Task Foundation Modeling
- **Multi-task sequence-structure modeling is preliminary.**
- seq2struct works as the primary task.
- invfold and inpaint require further validation.
- Full foundation model evidence is not yet available.

## LLM Semantic Tokens
- **LLM semantic token pipeline is implemented but not currently effective.**
- ArchiveII and bpRNA metadata contain no useful family/motif descriptions.
- LLM outputs are 100% UNKNOWN on current data.
- Semantic tokens must not be treated as a current contribution.
- LLM is never used during benchmark inference.

## Conflict Loss
- **Conflict loss was tested at multiple magnitudes and consistently harmed performance.**
- It is disabled in the candidate configuration.

## Masking Variants
- **Pair-aware, motif-span, motif-condition, and family-condition masking all hurt performance.**
- All masking variants are disabled in the candidate configuration.
- Re-enabling masking reduces F1 by ~17%.

## Data Scope
- ArchiveII: 2,704 train / 338 val / 338 test samples.
- bpRNA: 59,413 train / 12,731 val / 12,732 test samples (random split).
- No RNA 3D, ligand, or protein tasks.
- No pseudoknot decoding by default.
