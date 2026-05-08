# LLM Curriculum — Merge/Delete Decision

## Merge into mainline only if ALL of:

1. Smoke tests pass (build, apply, stats generation)
2. Default candidate behavior unchanged
3. No-curriculum baseline not degraded
4. Rule curriculum has non-negative effect (Pair F1 >= baseline)
5. LLM curriculum shows stable gain over rule curriculum (>0.3pp Pair F1)
6. Shuffled policy cannot match real LLM policy
7. Code complexity is controllable (default off, minimal changes)

## Delete experiment if ANY of:

1. Pipeline does not run end-to-end
2. Requires major main model changes
3. Introduces input semantic tokens (repeating failed approach)
4. Training unstable or performance degrades
5. Small benefit (<0.3pp) with significant complexity increase

## Current Verdict

Pending smoke test completion.
