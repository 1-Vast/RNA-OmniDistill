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
- `agent.py`: Thin root-level launcher for the Agent shell. Run `python agent.py` to start.
- `agent.cmd`: Windows CMD launcher. Run `agent` in cmd.exe.
- `agent`: Unix shell launcher. Run `bash agent` on Linux/macOS.
- `models/agent/analyzer.py`: LLM analysis agent for diagnostics, scheduling, paper reporting, and data audit.
- `models/agent/runtime.py`: Runtime guard with API/token limits, repeated-prompt detection, and consecutive error/blocked circuit breaker.
- `models/agent/memory.py`: Memory persistence with sanitization, file-level compact, and CORRUPT/error preservation.
- `models/agent/env.py`: Runtime environment detection (platform, hostname, CUDA, local/remote guess).
- `models/agent/paths.py`: Run discovery helper for recent experiment directories.
- `models/agent/cleanup.py`: Safe cleanup of old report directories.
- `models/agent/safety.py`: Command blocking, confirmation gates, safe-command whitelists.

## Scripts

- `scripts/data.py`: fetch, prepare, check, and split datasets.
- `scripts/eval.py`: strict benchmark, export, analyze, diagnose, scan.
- `scripts/run.py`: potential, sweep, external benchmark, and ablation workflows.
- `scripts/audit.py`: clean audit, naming audit, config integrity.
- `scripts/probe.py`: smoke, overfit, and small sanity probes.
- `scripts/llm.py`: CLI and interactive shell for the optional LLM analysis agent. Supports /runs, /memory compact, agent_audit safety checks. Includes inspect, trace, compare, case, doctor, target tuning, and confirmation-gated training.
- `scripts/download_datasets.py`: Download RNA structure datasets to `dataset/raw/`. Standard library only, no credentials.
- `scripts/upload_datasets.py`: SFTP upload of raw datasets to remote server. Default dry-run, password prompted at terminal, never hardcoded.
- `scripts/make_trial_config.py`: Generate temporary trial configs from base YAML without modifying originals.
- `scripts/audit_collator.py`: Audit collator/masking statistics (task distribution, mask ratio, pair balance, length distribution).
- `scripts/sweep_decoding.py`: Sweep decoding hyperparameters on existing checkpoints without retraining.
- `scripts/check_datasets.py`: Dataset metadata checker (local scan + report comparison).

## Docs

- `docs/usage.md`: direct CLI training, Agent-assisted training, local/remote device setup, Agent memory, and target tuning plan guide.
- `docs/agent.md`: Agent safety architecture reference.
- `docs/agent_guide.md`: Comprehensive Agent usage guide with natural language examples and remote workflow.

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
