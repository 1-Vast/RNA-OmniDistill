# Next LLM Strategy Design

## Background

After systematic evaluation of multiple LLM integration routes, all tested approaches failed to outperform deterministic rule-based baselines. The validated mainline is:

**MS-MPRM + PairRefine + pair-aware masking + rule hard replay + Nussinov decode**

Hard replay (+0.0067 at 300 steps) is the only training-side intervention showing clear benefit.

## Why Previous LLM Routes Failed

| Mechanism | Failure |
|---|---|
| Forward injection (semantic tokens) | Sample-level signal at position-level input |
| Pair BCE gradient competition (preference) | BCE dominates, pref signal too weak |
| Model-biased candidates (reranker) | LLM sees no independent evidence |
| Weak weight delta (curriculum) | 0.001 F1 change, control surface too small |
| LLM = rule (hard replay selection) | Rule already uses F1/recall/long-range statistics |
| No real generalization (synthetic) | Synthetic-only 0.1749 vs real 0.2009 |
| Random > LLM (decode policy) | Policy search space too simple for LLM |

## Proposed New Strategies

### Strategy A: LLM Failure Slice Hypothesis → Programmatic Experiment (Level A)

**Role**: LLM reads slice evaluation metrics and proposes testable hypotheses as programmatic experiment configs.

**Why different**: LLM doesn't touch model, data, or training. It only proposes what to test next.

**Dataflow**:
```
slice evaluation report → LLM generates hypothesis JSON → program converts to experiment configs → 300-step validation
```

**LLM Output**:
```json
{
  "dominant_failure": "long_range_underpairing",
  "proposed_change": "increase long_range_replay_ratio",
  "search_space": {"long_range_replay_ratio": [0.1,0.2,0.3]},
  "expected_metric": "long_range_recall"
}
```

**Baselines**: Hand-designed hypothesis, random hypothesis
**Success**: LLM hypothesis achieves better metric than hand hypothesis
**Risk**: Low — LLM is experiment planner, not model component

### Strategy B: LLM Rule Discovery for Replay (Level A)

**Role**: LLM proposes new replay selection rules (not per-sample decisions).

**Why different**: LLM contributes "rule structure", not duplicate computation. Rule replay already works; LLM could discover better rules.

**LLM Output**:
```json
{
  "rule_name": "long_range_fragmented_underpair",
  "condition": {"long_range_recall_max": 0.4, "stem_fragmentation_min": 0.3},
  "replay_priority": 2.0
}
```

**Baselines**: Current hand rule, random generated rule
**Success**: LLM-discovered rule > current hand rule
**Risk**: Moderate — requires replay sampler to support dynamic rules

### Strategy C: LLM Multi-Stage Training Protocol (Level B)

**Role**: LLM designs stage-by-stage training protocol (stage 1: BCE, stage 2: hard replay, stage 3: fine-tune).

**LLM Output**: Protocol JSON with stage definitions, replay ratios, warmup steps.

**Baselines**: Hand protocol, random protocol
**Success**: LLM protocol > hand protocol
**Risk**: Higher — requires multi-stage training infrastructure

### Strategy D: LLM Data Subset Contrast Design (Level B)

**Role**: LLM designs data subset splits for controlled comparison experiments.

**LLM Output**: Subset definitions based on structural features (long-range, high-GC, multi-stem).

**Baselines**: Hand-defined subsets, random subsets
**Success**: LLM subsets reveal clearer performance patterns
**Risk**: Moderate — analysis-only, no model change

## Priority

| Level | Strategy | Reason |
|---|---|---|
| **A** | **A: Hypothesis Generator** | Safest, no model integration, clear success criteria |
| **A** | **B: Rule Discovery** | Builds on proven hard replay, LLM adds rule structure |
| B | C: Multi-Stage Protocol | More complex infrastructure needed |
| B | D: Subset Contrast | Analysis tool, not performance module |
