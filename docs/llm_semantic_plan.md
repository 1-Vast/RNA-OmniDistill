# LLM Semantic Token — Plan and Status

## Status: Experimental / Not Currently Effective

## Problem
- ArchiveII and current bpRNA metadata lack informative family/motif/function descriptions.
- LLM annotation outputs mostly UNKNOWN semantic tokens.
- Therefore, semantic tokens do not currently improve model performance.

## What Exists
- `scripts/semantic.py`: supports offline annotation with providers (deepseek, openai, gemini, none).
- Tokenizer extension: 40+ semantic condition tokens in `models/token.py`.
- `config/archive_failed/`: semantic ablation configs preserved for reproducibility.
- API connectivity verified: DeepSeek API calls succeed.

## Rules (Must NOT Violate)
- Do NOT call LLM during benchmark inference.
- Do NOT use true structure in val/test prompts.
- Do NOT treat LLM semantic as a current contribution.
- Do NOT commit `.env` or API keys.
- Do NOT print API keys in logs or outputs.

## Next Valid Path
To make semantic tokens effective:
1. Use datasets with rich family descriptions: **Rfam full**, **RNAcentral**, **bpRNA-1m with source annotations**.
2. Build or adopt a manual ontology for:
   - `family_type`: tRNA, riboswitch, ribozyme, miRNA, snRNA, rRNA, lncRNA, ...
   - `primary_motif`: hairpin, stem_loop, bulge, internal_loop, multiloop, pseudoknot, ...
   - `structure_bias`: stem_rich, loop_rich, cloverleaf, long_range_pairing, ...
   - `function_tag`: translation, catalytic, regulatory, splicing, ...
3. Run LLM annotation on metadata-rich data splits.
4. Evaluate in controlled settings:
   - Family-disjoint generalization
   - Low-data regime
   - Motif-conditioned generation ("generate a tRNA-like structure")

## Required Future Experiments
| Experiment | Baseline | Variant | Metric |
|---|---|---|---|
| no_semantic vs llm_semantic | candidate | candidate + llm tokens | Pair F1 |
| shuffled control | llm_semantic | shuffled semantic | Pair F1 |
| family-disjoint | candidate | candidate_semantic | Family-disjoint F1 |
| low-data | candidate | candidate_semantic | F1 @ 10% data |

## Provider Configuration
See `.env.example`:

```
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=YOUR_KEY
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512
```
