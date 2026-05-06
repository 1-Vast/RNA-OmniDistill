# LLM Semantic Token — Plan and Status

## Status: Pipeline Working, Rfam Pilot Positive

## Root Cause of Previous Failure
- **ArchiveII/bpRNA metadata is sparse**: family=OTHER and empty description caused unknown_ratio=100%.
- LLM cannot infer biological function from bare sequence alone.

## Solution: Rfam Metadata Distillation
- Parse Rfam Stockholm GF metadata: DE (definition), CC (comment), WK (wikipedia), TP (type), CL (clan)
- Extract structural statistics: pairing ratio, stem/hairpin counts, length bins
- Use metadata-rich prompt with controlled ontology tokens
- Rule baseline vs LLM comparison

## Rfam Pilot Results (128 samples)

| Metric | Rule Baseline | LLM (DeepSeek) |
|---|---|---|
| unknown_ratio (family_type) | **0%** | **0%** |
| valid_json_rate | 100% | 100% |
| LLM adds value over rule | — | Richer diversity (function_tag: gene regulation, translational repressor, microRNA precursor) |
| Agreement (family_type) | 73% | — |

### Distribution Comparison

| Field | Rule | LLM |
|---|---|---|
| family_type | miRNA(111), sRNA(11), cis(6) | miRNA(93), sRNA(11), microRNA(7) |
| structure_bias | **unknown(128)** | hairpin(67), stem-loop(44) |
| function_tag | **unknown(119)** | gene regulation(100), regulatory(8) |

**LLM dramatically outperforms rules on structure_bias and function_tag** (128→0 unknown).

## Rules (Must NOT Violate)
- Do NOT call LLM during benchmark inference.
- Do NOT use true structure in val/test prompts.
- Do NOT treat LLM semantic as a current main contribution.
- Do NOT commit `.env` or API keys.

## Next Steps
1. Run full Rfam semantic annotation (128→1000+ samples)
2. Create semantic-conditioned model config
3. Evaluate: no_semantic vs rule_semantic vs llm_semantic vs shuffled
4. Test on family-disjoint split
5. Low-data regime evaluation
6. Motif-conditioned generation

## Provider
See `.env.example`. Currently using DeepSeek (deepseek-chat).
