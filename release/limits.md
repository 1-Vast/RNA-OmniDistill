# RNA-OmniDiffusion Candidate Limitations

## Structural Validity

- Valid rate of 1.0000 comes from strict Nussinov constrained decoding, not free token generation.
- Token-only decoding produced valid rate 0.0000 in diagnostics.
- Strict Nussinov enforces non-crossing structures and one pair per position.

## Decoding Methods

- Greedy decoding is a speed and pair-head probe only.
- Greedy decoding must not be reported as a final strict benchmark metric.
- Greedy candidates can contain crossing pairs before dot-bracket conversion.

## Family-Disjoint Generalization

- Family-disjoint generalization is not established as a final claim.
- The current bpRNA result is an external random split.
- Rfam seed consensus structures are not treated as individual ground-truth labels.

## Multi-Task Scope

- Seq2struct is the primary validated task.
- Inverse folding and inpainting exist in the pipeline but remain preliminary.

## LLM Semantic Conditioning

- Semantic prefixing, constraint-program prompting, motif repair, and low-data adaptation were tested and did not produce model-level gains at current scale.
- LLM semantic conditioning is excluded from the candidate model.
- See [../docs/llm_negative_result.md](../docs/llm_negative_result.md).

## Conflict Loss

- Conflict loss was tested as a precision-oriented regularizer.
- It was harmful at tested magnitudes and is disabled in the candidate model.

## Masking Variants

- Pair-aware masking, motif-span masking, motif condition, and family condition did not provide a stable main contribution on ArchiveII.
- These masking variants are disabled in the candidate model.

## Data Scope

- ArchiveII: 2,704 train, 338 validation, 338 test.
- bpRNA: 59,413 train, 12,731 validation, 12,732 test under random split.
- No RNA 3D, ligand, or protein tasks.
- No pseudoknot decoding by default.
