# RNA-OmniDiffusion Candidate — Limitations

## Structural Validity
- Valid = 1.0000 comes from strict Nussinov constrained decoding, not from free token generation.
- Removing Nussinov (token-only decode) results in valid_rate = 0.0000.
- Nussinov DP guarantees non-crossing, one-pair-per-position structures.

## Decoding Methods
- Greedy decoding is a speed probe only and must not be used as a final benchmark metric.
- Greedy decode can produce crossing pairs before dot-bracket conversion.

## Family-Disjoint Generalization
- Not yet established. bpRNA split is external random (all families labeled "OTHER").
- Rfam seed consensus structures (SS_cons) are unsuitable as individual ground-truth.

## Multi-Task
- Not established. seq2struct is the primary validated task.
- invfold and inpaint are preliminary.

## LLM Semantic Conditioning
- Tested on Rfam metadata-rich data. Semantic prefix, constraint program, motif repair, and low-data adaptation all failed to produce model-level gains at current scale.
- Excluded from final candidate release.
- See `docs/llm_negative_result.md`.

## Conflict Loss
- Tested at multiple magnitudes. Consistently harmful. Disabled in candidate.

## Masking Variants
- Pair-aware, motif-span, motif-condition, and family-condition masking all reduce performance.
- All masking variants disabled in candidate.

## Data Scope
- ArchiveII: 2,704 train / 338 val / 338 test
- bpRNA: 59,413 train / 12,731 val / 12,732 test (random split)
- No RNA 3D, ligand, or protein tasks
- No pseudoknot decoding by default
