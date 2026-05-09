# RNA-OmniPrefold: 2026 Paper Framework

## Working Title

**RNA-OmniPrefold: Relation-aware Masked Denoising for Constraint-Guided RNA Folding**

## One-Sentence Claim

RNA-OmniPrefold combines masked discrete denoising, explicit pair-relation prediction, lightweight 2D pair refinement, and strict Nussinov projection to produce valid RNA secondary structures.

## Method

The model receives task, sequence, and structure tokens in a unified sequence format. A Transformer encoder predicts masked tokens and pair logits over RNA positions. A small residual 2D refiner improves local continuity in the pair-relation field. Final dot-bracket structures are decoded with strict Nussinov dynamic programming under canonical/wobble and minimum-loop constraints.

## Supported Contributions

1. Relation-aware masked denoising for RNA folding tasks.
2. Explicit pair-relation field prediction.
3. Lightweight pair-logit refinement.
4. Strict constraint-guided decoding.
5. Optional preference optimization is planned but must be validated before it is reported as a result.

## Claim Boundaries

Supported:

- Pair-relation prediction and strict decoding are central to the model.
- Strict Nussinov decoding is required for valid structures.

Not supported:

- Semantic-token conditioning improves the model.
- Token-only decoding is a valid final benchmark.
- Greedy decoding is a final benchmark metric.
- The model solves RNA 3D, ligand, protein, or pseudoknot tasks.
