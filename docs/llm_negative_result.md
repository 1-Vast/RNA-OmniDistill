# LLM Semantic Conditioning — Negative Result

## Summary

LLM-based semantic conditioning (prefix tokens, constraint programs, motif repair, low-data adaptation) was tested on Rfam metadata-rich data but did not yield model-level gains.

## What Was Tested

| Route | Description | Result |
|---|---|---|
| L0: Semantic Prefix | Prepend semantic tokens to input | No gain (delta = -0.002) |
| L1: Constraint Program | LLM-derived constraint for pair refinement | No gain (delta = -0.002) |
| L2: Motif Repair | LLM-guided mask region selection | No gain |
| L3: Low-data Adaptation | Semantic-conditioned few-shot | Insufficient data |

## What Worked

- Rfam metadata parsing (97,913 records, 4,227 families)
- LLM API annotation pipeline (DeepSeek, 128 samples, 0% unknown)
- Ontology normalization (invalid_after_count = 0)
- Semantic token injection (Gate 0 PASS)
- Shape-safe training with semantic tokens (Gate 1 PASS)

## Why It Failed

- 100 training steps insufficient for model convergence
- 128 samples too few for condition differentiation
- Ordinary seq2struct is sequence-dominated; semantic conditions provide marginal signal
- All variants converged to same F1 ≈ 0.12 (random performance level)

## Decision

**LLM semantic conditioning is excluded from the final candidate release.**

The LLM annotation pipeline is preserved as experimental infrastructure (`scripts/semantic.py` in archive), but is not part of the current mainline contribution.

## Future Work

To revisit LLM conditioning:
1. Scale to 1000+ metadata-rich samples
2. Train for 2000+ steps with proper convergence
3. Design tasks where semantic knowledge provides irreducible signal (e.g., family-disjoint generalization, motif repair with ground-truth masks)
4. Use stronger conditioning mechanisms (FiLM, cross-attention) rather than prefix tokens

## Related Files

- `config/archive_failed/` — semantic, constraint, and LLM route configs
- `scripts/semantic.py` — preserved for future reference, not in mainline
- `docs/llm_semantic_plan.md` — superseded by this document
