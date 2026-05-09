# RNA-OmniPrefold Candidate Model Card

## Summary

RNA-OmniPrefold is a compact relation-aware masked denoising model for RNA secondary structure prediction. It combines token denoising, pair-relation field prediction, lightweight 2D pair refinement, and strict Nussinov decoding.

## Intended Use

- RNA secondary structure prediction under canonical/wobble and non-crossing constraints.
- Research experiments on pair-relation modeling and constrained decoding.

## Not Intended For

- RNA 3D structure prediction.
- Pseudoknot prediction in the strict default path.
- Clinical or production biological decision-making without independent validation.
- Pseudo-label generation.

## Training Components

- Sequence-only masked denoising pretraining.
- Supervised pair-relation adaptation.
- Strict constraint projection at inference.

## Limitations

- Precision remains lower than recall in the current candidate path.
- Family-disjoint generalization is not established.
- Validity relies on strict decoding.
