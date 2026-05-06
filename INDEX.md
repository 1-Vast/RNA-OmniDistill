# RNA-OmniDiffusion — Index

## Core

- `main.py` — train, eval, infer, smoke entrypoint
- `models/omni.py` — Transformer model, pair head, pairrefine, loss
- `models/decode.py` — iterative decoding, strict Nussinov
- `models/mask.py` — masking helpers
- `models/token.py` — RNA tokenizer
- `models/dataset.py` — JSONL dataset
- `models/collator.py` — task sampling, masking, pair labels

## Mainline Configs

- `config/candidate.yaml` — best candidate (pairrefine=true, masking=false)
- `config/fixed.yaml` — alias of candidate
- `config/oldbase.yaml` — historical baseline (pairrefine=false, masking=true)
- `config/candidate_norefine.yaml` — no pairrefine control
- `config/candidate_oldmask.yaml` — old masking control

## External Configs

- `config/external_bprna_candidate.yaml`
- `config/external_bprna_norefine.yaml`
- `config/external_bprna_oldbase.yaml`

## Scripts

- `scripts/eval.py` — benchmark, paper artifact generation
- `scripts/run.py` — external, ablation workflows
- `scripts/dataset.py` — download, prepare, split external datasets
- `scripts/audit.py` — clean audit, config integrity

## Release

- `release/best_config.yaml`
- `release/model_card.md`
- `release/results_summary.md`
- `release/reproduce.md`
- `release/limitations.md`

## Archived / Failed Probes

- `config/archive_failed/` — failed/diagnostic configs (precision, conflict-loss, semantic, constraint, LLM routes)
- `docs/llm_negative_result.md` — LLM semantic conditioning negative result
