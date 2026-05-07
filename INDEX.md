# RNA-OmniDiffusion Index

## Paper Entry

- [release/paper.md](release/paper.md): 2026 paper framework, method narrative, experiment table plan, limitations, and claim boundaries.

## Core Code

- `main.py`: Research-style CLI entry window with overview, mode registry, tunable parameter registry, and lazy training/inference dispatch.
- `models/omni.py`: Transformer encoder, token heads, pair head, pair refinement, token/pair/conflict loss plumbing.
- `models/decode.py`: strict Nussinov decoding, greedy probe, staged decode utilities.
- `models/mask.py`: random, pair-aware, and motif-span masking helpers.
- `models/dataset.py`: JSONL RNA dataset.
- `models/collator.py`: task sampling, masking, segment ids, pair labels.
- `models/token.py`: RNA tokenizer.
- `models/pairprior.py`: optional diagnostic pair-prior probe, disabled by default.
- `models/agent/analyzer.py`: LLM analysis agent for diagnostics, scheduling, paper reporting, and data audit.

## Scripts

- `scripts/data.py`: fetch, prepare, check, and split datasets.
- `scripts/eval.py`: strict benchmark, export, analyze, diagnose, scan.
- `scripts/run.py`: potential, sweep, external benchmark, and ablation workflows.
- `scripts/audit.py`: clean audit, naming audit, config integrity.
- `scripts/probe.py`: smoke, overfit, and small sanity probes.
- `scripts/llm.py`: CLI and concise interactive shell for the optional LLM analysis agent, including inspect, trace, compare, case, doctor, usage, memory, cleanup, API runtime guard, loop/stall guard, cleanup guard, `/last`, `/open`, and confirmation-gated candidate training controls.

## Main Configs

- `config/candidate.yaml`: current candidate model.
- `config/fixed.yaml`: alias-style candidate configuration used by prior runs.
- `config/oldbase.yaml`: historical baseline.
- `config/candidate_norefine.yaml`: no pair-refinement control.
- `config/candidate_oldmask.yaml`: old masking control.
- `config/precision*.yaml`: rejected precision-oriented probes; useful only for diagnostics.
- `config/archive_failed/`: failed or diagnostic configurations.

## Release Artifacts

- `release/paper.md`: paper framework.
- `release/results_summary.md`: benchmark and component tables.
- `release/reproduce.md`: reproducibility commands.
- `release/model_card.md`: candidate model card.
- `release/limits.md`: limitations.
- `docs/pairprior_probe.md`: pair-prior negative/weak-probe note.
- `docs/usage.md`: direct CLI training, Agent-assisted training, local/remote device setup, Agent memory, and target tuning plan guide.

## Current Mainline Commands

```bash
python main.py overview
python main.py smoke
python main.py train --config config/candidate.yaml --device cuda
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --decode_only --workers 8 --chunksize 2 --profile --scan config/scan.json
python scripts/llm.py diagnose --run outputs/candidate --out outputs/llm/diagnose
```
