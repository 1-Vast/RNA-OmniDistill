# LLM Semantic Token — Plan and Status

## Status: Rfam Pilot Successful, Ready for Model Smoke

## Root Cause of Previous Failure
- ArchiveII/bpRNA metadata sparse (family=OTHER, empty description) → **unknown_ratio=100%**.
- LLM cannot infer biological function from bare sequence alone.

## Solution: Rfam Metadata Distillation ✅
- Parse Rfam Stockholm GF metadata: DE, CC, WK, TP, CL
- Extract structural statistics: pairing ratio, stem/hairpin counts
- Metadata-rich prompt with controlled ontology
- Rule baseline vs LLM comparison

## Rfam Pilot Results (128 samples, DeepSeek)

| Metric | Rule Baseline | LLM | Winner |
|---|---|---|---|
| unknown_ratio (family_type) | 0% | 0% | Tie |
| unknown_ratio (structure_bias) | 100% | 34% | **LLM** |
| unknown_ratio (function_tag) | 93% | 8% | **LLM** |
| valid_json_rate | 100% | 100% | Tie |
| evidence_nonempty_rate | N/A | 100% | LLM |
| constraint_hint | 100% UNKNOWN | 100% UNKNOWN | Tie |

### Distribution (LLM, 128 records)
| family_type | miRNA(111), bacterial_sRNA(11), cis_regulatory(6) |
| structure_bias | loop_rich(67), unknown(44), low_structure(17) |
| function_tag | regulatory(114), unknown(10), processing(4) |

### Rule vs LLM Comparison
| Field | Agreement | LLM Knows, Rule Unknown |
|---|---|---|
| family_type | 100% | 0 |
| structure_bias | 34% | **84/128 (66%)** |
| function_tag | 6% | **112/128 (88%)** |

**LLM dramatically outperforms rules on structure_bias and function_tag.**

## Next Steps
1. Scale annotation to 1000+ Rfam samples
2. Run semantic-conditioned model smoke:
   ```bash
   python scripts/run.py semantic \
     --base_config config/candidate.yaml \
     --semantic_jsonl dataset/processed/rfam_semantic/semantic.api_pilot.normalized.jsonl \
     --rule_jsonl dataset/processed/rfam_semantic/semantic.rule.normalized.jsonl \
     --device cuda --decode nussinov --tag semantic_rfam_pilot --quick
   ```
3. Compare: no_semantic vs rule_semantic vs llm_semantic vs shuffled
4. Test on family-disjoint split
5. Low-data evaluation

## Rules
- LLM is **never called during benchmark inference**.
- True structure is never in val/test prompts.
- Semantic tokens are experimental, **not a current main contribution**.
- `.env` is gitignored.

## Provider
DeepSeek (deepseek-chat). See `.env.example`.
