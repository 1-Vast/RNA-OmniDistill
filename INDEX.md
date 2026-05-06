# RNA-OmniDiffusion — Index

## Core

- `main.py` — train, eval, infer, smoke entrypoint
- `models/omni.py` — Transformer model, pair head, pairrefine, loss
- `models/decode.py` — iterative decoding, strict Nussinov decoding
- `models/token.py` — RNA tokenizer
- `models/dataset.py` — JSONL dataset
- `models/collator.py` — task sampling, masking, pair labels
- `models/mask.py` — masking helpers

## Mainline Configs

- `config/candidate.yaml` — **best candidate model** (pairrefine=true, masking=false)
- `config/fixed.yaml` — alias of candidate
- `config/oldbase.yaml` — historical baseline (pairrefine=false, masking=true)
- `config/candidate_norefine.yaml` — no pairrefine control
- `config/candidate_oldmask.yaml` — old masking control

## External Configs

- `config/external_bprna_candidate.yaml` — bpRNA candidate
- `config/external_bprna_norefine.yaml` — bpRNA no pairrefine
- `config/external_bprna_oldbase.yaml` — bpRNA historical baseline

## Scripts

- `scripts/eval.py` — benchmark, analysis, paper artifact generation
- `scripts/run.py` — external, multitask, foundation workflows
- `scripts/dataset.py` — download, prepare, split external datasets
- `scripts/audit.py` — clean audit, config integrity checks
- `scripts/semantic.py` — **[EXPERIMENTAL]** offline LLM semantic annotation

## Experimental (Not Mainline)

- `config/multitask_candidate.yaml` — multi-task config (preliminary)
- `config/archive_failed/` — failed/diagnostic probes (precision, conflict-loss, semantic ablation)
- `docs/llm_semantic_plan.md` — LLM semantic token future roadmap

## Release

- `release/best_config.yaml` — best candidate configuration
- `release/model_card.md` — model description and results
- `release/results_summary.md` — paper-ready tables
- `release/reproduce.md` — reproduction commands
- `release/limitations.md` — known limitations

## Security

- `.env` is gitignored — API keys must NOT be committed
- LLM is NEVER called during benchmark inference
- LLM API is optional and only used for offline annotation
