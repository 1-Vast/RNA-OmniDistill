# RNA-OmniDiffusion

Pair-refined, constraint-guided masked diffusion for RNA secondary structure prediction.

This repository is organized as a compact, reproducible research codebase. The current model is not an LLM-powered system: it deliberately excludes external LLM calls, RNA-FM, RNA 3D, ligand, and protein tasks. The main contribution is a trainable pair-logit model with lightweight 2D refinement and strict Nussinov decoding.

## Current Claim

RNA-OmniDiffusion is a candidate 2026-style model paper framework for RNA secondary structure prediction:

- masked discrete diffusion backbone over sequence and structure tokens
- MLP pair head for base-pair logits
- local 2D pair refinement on the pair-logit map
- strict Nussinov decoding for valid non-crossing structures
- staged-logit benchmark path for fast strict evaluation
- documented negative results for token-only decode, greedy decode as final metric, conflict loss, and masking variants

The paper framing and final experimental narrative are in [release/paper.md](release/paper.md).

## Main Results

### ArchiveII

| Model | Pair F1 | Precision | Recall | Valid | Pair Ratio |
|---|---:|---:|---:|---:|---:|
| oldbase | 0.3846 | 0.3398 | 0.4465 | 1.0000 | 1.4213 |
| norefine | 0.4966 | 0.4470 | 0.5630 | 1.0000 | 1.3913 |
| candidate | 0.5689 | 0.5090 | 0.6517 | 1.0000 | 1.3808 |

ArchiveII 3-seed stability: mean Pair F1 = 0.5813, std = 0.0078.

### bpRNA External

| Model | Pair F1 | Precision | Recall | Valid | Pair Ratio | N |
|---|---:|---:|---:|---:|---:|---:|
| oldbase | 0.4234 | 0.4019 | 0.4741 | 1.0000 | 1.40 | 12,732 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 1.0000 | 1.42 | 12,732 |
| candidate | 0.5285 | 0.4877 | 0.6070 | 1.0000 | 1.38 | 12,732 |

External drop from ArchiveII candidate to bpRNA candidate is about 7.1%.

## Core Structure

```text
main.py
models/
  omni.py        # Transformer, pair head, pair refinement, loss
  decode.py      # strict Nussinov, greedy probe, staged decode helpers
  mask.py        # masking helpers
  dataset.py     # JSONL dataset
  collator.py    # task sampling and labels
  token.py       # RNA tokenizer
  pairprior.py   # optional diagnostic pair-prior probe, disabled by default
  agent/         # LLM analysis agent; not used for inference
scripts/
  data.py        # data preparation CLI
  eval.py        # benchmark, export, analysis, diagnosis
  run.py         # potential, sweep, external, ablation workflows
  audit.py       # clean/name/config audits
  probe.py       # smoke and small sanity checks
  llm.py         # analysis agent CLI for diagnostics and reports
release/
  paper.md       # 2026 paper framework
  results_summary.md
  reproduce.md
  model_card.md
```

## Reproduce

Smoke:

```bash
python main.py smoke
```

Train candidate:

```bash
python main.py train --config config/candidate.yaml --device cuda
```

Strict Nussinov benchmark:

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```

Decode-only scan from staged logits:

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --decode_only --workers 8 --chunksize 2 --profile --scan config/scan.json
```

External bpRNA comparison:

```bash
python scripts/run.py external --configs config/external_bprna_candidate.yaml config/external_bprna_norefine.yaml config/external_bprna_oldbase.yaml --dataset bprna --split random --device cuda --decode nussinov --bench_workers 8 --tag external_bprna
```

Full reproduction details are in [release/reproduce.md](release/reproduce.md).

## LLM Analysis Agent

The optional LLM agent is an experiment assistant, not a structure predictor. It only reads existing artifacts and writes Markdown/JSON reports. It never modifies labels, predictions, benchmark metrics, or test data.

Environment variables are read from `.env`:

```text
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

Four analysis functions are available:

```bash
python scripts/llm.py diagnose --run outputs/candidate --out outputs/llm/diagnose
python scripts/llm.py schedule --run outputs/candidate --config config/candidate.yaml --out outputs/llm/schedule
python scripts/llm.py report --inputs release/paper.md release/results_summary.md release/limits.md --out outputs/llm/report
python scripts/llm.py auditdata --inputs dataset/archive/train.jsonl dataset/archive/test.jsonl --out outputs/llm/data
```

Use `--dry_run` to inspect the exact prompt without calling the API.

### Interactive Agent Shell

```bash
python scripts/llm.py agent --dry_run
```

Example shell session:

```text
RNA-OmniDiffusion Agent Shell
Mode: dry-run
Safety: read-only
Type /help for commands, /exit to quit.

agent> diagnose outputs/candidate
agent> schedule outputs/candidate config/candidate.yaml
agent> report release/model_card.md release/results_summary.md release/limitations.md
agent> auditdata dataset/archive/train.jsonl dataset/archive/test.jsonl
agent> /exit
```

The shell is read-only. It does not run training, does not run benchmark inference, and does not modify labels, predictions, metrics, checkpoints, or configs. Use `--dry_run` or `--no_api` to generate prompts without API calls.

## What Not To Claim

- Do not claim LLM semantic conditioning as a positive result. It was tested and did not improve the model-level result.
- Do not use greedy decode as the final paper metric. It is only a fast pair-head probe.
- Do not use token-only decode as a strict structural result. Its valid structure rate was 0 in diagnostics.
- Do not present masking as a main contribution on ArchiveII. The seed repeat found the effect inconclusive or negligible.
- Do not describe this as RNA 3D, ligand, protein, RNA-FM, or Agent work.
- Do not present pair-prior as a candidate-model contribution. It is an optional weak diagnostic probe only; see [docs/pairprior_probe.md](docs/pairprior_probe.md).

## Current Limitations

- Precision is still lower than recall, and pair ratio remains above 1.0.
- Family-disjoint generalization is not established as a final claim.
- The model is moderate-scale and from scratch; it is not a foundation model trained on massive unlabeled RNA corpora.
- Strict Nussinov is essential for valid structures, so decoding is part of the model system.
