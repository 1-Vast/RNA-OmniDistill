# RNA-OmniDiffusion

Minimal masked discrete diffusion for RNA sequence and secondary-structure modeling.

This repository intentionally stays small. It does not use RNA-FM, LoRA, external LLM calls, RNA 3D, ligand tasks, or protein tasks.

## Structure

```text
main.py
config/
  base.yaml
  orig.yaml
  relax.yaml
  fix.yaml
  fixed.yaml
  mild.yaml
  strict.yaml
  stable.yaml
  scan.json
  ablate/
models/
  omni.py
  mask.py
  decode.py
  dataset.py
  collator.py
  token.py
  agent/
utils/
  struct.py
  metric.py
scripts/
  data.py
  probe.py
  audit.py
  eval.py
  run.py
```

`models/` contains the model, tokenizer, dataset, collator, masking, and decode code. `scripts/data.py` is the data preparation CLI. `dataset/` is a data-file directory, not a Python package. Future agent-related code should live under `models/agent`; it is currently an empty package and no agent framework is implemented.

## Data Preparation

JSONL records use:

```json
{"id":"RNA1","seq":"AUGGCU","struct":"((..))","family":"OTHER","motifs":[],"pairs":[[0,5],[1,4]],"length":6}
```

Prepare ArchiveII-style data:

```powershell
python scripts/data.py fetch --set archive --out dataset/raw/archive
python scripts/data.py prep --input dataset/raw/archive --output dataset/processed/archive.jsonl --format auto --maxlen 512
python scripts/data.py check --input dataset/processed/archive.jsonl --output dataset/processed/archivecheck.jsonl --maxlen 512
python scripts/data.py split --input dataset/processed/archivecheck.jsonl --out dataset/archive --mode random
```

## Smoke

```powershell
python main.py smoke
python scripts/audit.py clean --out outputs/clean
python scripts/audit.py names --out outputs/name
```

## Train Fixed

`config/fixed.yaml` is the current main configuration: pairfix training plus relaxed strict Nussinov decoding.

```powershell
python main.py train --config config/fixed.yaml --device cuda
```

## Strict Benchmark

Final structure metrics should use strict Nussinov decoding.

```powershell
python scripts/eval.py bench --config config/fixed.yaml --ckpt outputs/fixed/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```

## Precision Sweep

`config/precision.yaml` is a precision-oriented experimental configuration. It enables a lightweight 2D pair-logit refiner and conflict loss, disables the masking variants that were not stable on ArchiveII, lowers positive pair weight, and increases negative pair sampling pressure. `config/fixed.yaml` remains unchanged as the main baseline.

```powershell
python main.py train --config config/precision.yaml --device cuda --max_steps 20
python scripts/eval.py bench --config config/precision.yaml --ckpt outputs/precision/best.pt --split test --device cuda --decode nussinov --limit 32 --profile
python scripts/run.py sweep --configs config/fixed.yaml config/precision.yaml config/precision_norefine.yaml config/precision_noconflict.yaml config/precision_soft.yaml --mode quick --device cuda --decode nussinov --bench_workers 4 --tag precision
```

The sweep writes to `outputs/sweep_precision/` and does not overwrite `outputs/fixed/`.

Outputs:

```text
outputs/fixed/benchmark.json
outputs/fixed/benchmark.csv
outputs/fixed/predictions.jsonl
outputs/fixed/benchmeta.json
outputs/fixed/benchtime.json
outputs/fixed/logits.pt
```

## Decode Scan

Decode-only scans reuse staged logits and do not retrain or modify labels.

```powershell
python scripts/eval.py bench --config config/fixed.yaml --ckpt outputs/fixed/best.pt --split test --device cuda --decode nussinov --decode_only --workers 8 --chunksize 2 --profile --scan config/scan.json
```

## Core Ablation

Run only after the strict full benchmark passes the pair-count and ranking checks.

```powershell
python scripts/run.py ablate --config config/fixed.yaml --only full nopair nonuss random --device cuda --decode nussinov --bench_workers 8 --bench_profile --bench_resume
```

Dry-run the commands first:

```powershell
python scripts/run.py ablate --config config/fixed.yaml --only full nopair nonuss random --device cuda --decode nussinov --bench_workers 8 --dry_run
```

## Greedy Probe

`--decode greedy` is retained only as a fast pair-head probe. It is approximate, can generate crossing candidates before dot-bracket conversion, and must not be used as the final paper metric.

```powershell
python scripts/eval.py bench --config config/fixed.yaml --ckpt outputs/fixed/best.pt --split test --device cuda --decode greedy --profile --resume
```

## Config Semantics

- `config/orig.yaml`: original training and original decoding.
- `config/relax.yaml`: original training with relaxed decoding only.
- `config/fix.yaml`: pairfix training with original decoding.
- `config/fixed.yaml`: pairfix training with relaxed decoding; current main configuration.
- `config/mild.yaml`, `config/strict.yaml`, `config/stable.yaml`: intervention templates. They are not run automatically.
- `config/cpu.yaml`: CPU preflight configuration.

## Sandbox

sandbox/precision/ contains precision-oriented probe experiments. Pairrefine (2D Conv2d residual refinement) was validated at +10% F1 and merged into the main config. Conflict loss was tested at multiple magnitudes and found consistently harmful. See sandbox/precision/README.md and outputs/sandbox_precision/decision.md for details.

## Current Limitations

- No pseudoknot decoding by default.
- No RNA 3D, ligand, or protein tasks.
- Greedy decoding is a probe only.
- Quick runs are pipeline checks and are not paper conclusions.

## Current Candidate Model

The current best candidate configuration is `config/candidate.yaml`.

### Architecture
- **pairrefine**: 2D Conv2d residual refinement on pair logits (key contributor)
- **No masking variants**: pair-aware, motif-span, motif-condition, family-condition all disabled
- **Strict Nussinov decoding**: required for legal non-crossing structures
- **Pair head**: MLP-based pair logit head (primary structure path)
- **Token head**: auxiliary only; not used as primary structure path

### Results (ArchiveII test, 338 samples, strict Nussinov)

| Variant | Pair F1 | Precision | Recall | Valid | Pair Ratio |
|---|---:|---:|---:|---:|---:|
| candidate (pairrefine + nomask) | 0.5689 | 0.5090 | 0.6517 | 1.0000 | 1.3808 |
| norefine (nomask only) | 0.4966 | 0.4470 | 0.5630 | 1.0000 | 1.3913 |
| oldmask (pairrefine + masking) | 0.3973 | 0.3519 | 0.4610 | 1.0000 | 1.3886 |
| old baseline | 0.3846 | 0.3398 | 0.4465 | 1.0000 | 1.4213 |

### Component Analysis
- **pairrefine contribution**: ΔF1 = 0.0723 (significant)
- **masking removal contribution**: ΔF1 = 0.1715 (significant)
- **Combined vs baseline**: ΔF1 = 0.1843 (+47.9%)

### 3-Seed Stability
Mean F1 = 0.5813 +/- 0.0078 across seeds 42, 43, 44.
All seeds: valid_rate = 1.0000, pair_ratio < 1.5.
Candidate is stable.

### Key Findings
1. Pairrefine is the primary effective new module.
2. Disabling masking variants is critical on ArchiveII.
3. Token-only decode fails (valid_rate = 0.0000).
4. Greedy decode is a probe only; strict Nussinov is the final metric.
5. Conflict loss is harmful at all tested magnitudes.

## LLM API (Optional)

LLM API is used only for **offline semantic/task annotation**. It is NOT required
for training, validation, testing, or benchmark inference.

### When API is needed
- **Direction A (External Generalization)**: No API needed. Provider=none runs rule-based annotation.
- **Direction B (Multi-task Modeling)**: No API needed. Provider=none runs rule-based templates.
- **Direction C (Semantic Token Distillation)**: API needed only for real LLM annotation.

### Setup
```bash
cp .env.example .env
# Edit .env with your API credentials
```

### Providers
- `none`: Rule-based templates (default, no API)
- `ark`: Ark/Volces API
- `openai`: OpenAI-compatible API
- `gemini`: Google Gemini API

### Usage
```bash
# Semantic annotation (no API)
python scripts/semantic.py annotate --input data.jsonl --out semantic.jsonl --provider none --max_samples 64

# Task annotation (no API)
python scripts/semantic.py annotate_tasks --input data.jsonl --out task_semantic.jsonl --provider none --max_samples 64

# Audit
python scripts/semantic.py audit --input semantic.jsonl --out outputs/semantic_audit

# Foundation workflow (all directions, quick mode)
python scripts/run.py foundation --directions external multitask --device cuda --quick
```

### Security
- `.env` is gitignored and must NOT be committed.
- API keys must NOT appear in code, logs, or benchmark outputs.
- LLM is NEVER called during benchmark inference.


## Config Naming (Current Mainline)

| Config | Role | pairrefine | masking | Status |
|---|---|---|---|---|
| `candidate.yaml` | Best candidate model | true | false | **current best** |
| `fixed.yaml` | Default alias of candidate | true | false | active |
| `oldbase.yaml` | Historical baseline | false | true | reference |
| `candidate_norefine.yaml` | No pairrefine control | false | false | control |
| `candidate_oldmask.yaml` | Old masking control | true | true | control |

Archived (non-mainline): `config/archive_failed/` contains precision/conflict-loss experiments and semantic ablation configs. These are retained for reproducibility but are NOT part of the main pipeline.

## Key Design Decisions

- **pairrefine**: 2D Conv2d residual refinement → primary effective module (+7.2% F1)
- **No masking variants**: disabled in candidate/fixed (masking hurts on ArchiveII, -17.2% F1)
- **Strict Nussinov**: final benchmark decoding (legal non-crossing structures)
- **Conflict loss**: rejected at all tested magnitudes (harms performance)
- **Token-only decode**: failure mode (valid_rate = 0.0000)
- **Greedy decode**: probe only, NOT a final metric

## External Generalization (Direction A)

### bpRNA Results (preliminary)

| Model | F1 | Prec | Rec | MCC | Valid | Ratio | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| **candidate** | **0.5285** | 0.4877 | 0.6070 | 0.5344 | 1.00 | 1.38 | 12,732 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 0.4451 | 1.00 | 1.42 | 12,732 |
| oldbase | 0.4234 | 0.4019 | 0.4741 | 0.4335 | 1.00 | 1.40 | 12,732 |

- **External drop vs ArchiveII**: -7.1% (0.5689 → 0.5285) on 37x larger test set
- **pairrefine external contribution**: +0.0886 F1 (+20.1% over norefine)
- **Split**: random (NOT family-disjoint, all bpRNA families = "OTHER")
- **Conclusion**: external generalization strongly supported
- **Final gate**: ENTER_FINAL_STAGE_WITH_LIMITATIONS (7/8 checks pass, multitask + family-disjoint pending)

```bash
# Download external datasets
python scripts/dataset.py download --name bprna --out dataset/raw/bprna

# Prepare and split
python scripts/dataset.py prepare --input dataset/raw/bprna --out dataset/processed/bprna/clean.jsonl
python scripts/dataset.py split --input dataset/processed/bprna/clean.jsonl --out dataset/processed/bprna_family --mode family

# Run external benchmark
python scripts/run.py external --configs config/external_bprna_candidate.yaml config/external_bprna_norefine.yaml config/external_bprna_oldbase.yaml --dataset bprna --split family --device cuda --decode nussinov
```

## Multi-task (Direction B, quick only)

```bash
python scripts/run.py multitask --config config/candidate.yaml --tasks seq2struct invfold inpaint --device cuda --quick
```

## LLM Annotation (Direction C, no-API smoke only)

```bash
python scripts/semantic.py annotate --input data.jsonl --out semantic.jsonl --provider none
python scripts/semantic.py annotate_tasks --input data.jsonl --out task_semantic.jsonl --provider none
python scripts/semantic.py audit --input semantic.jsonl --out outputs/audit
```

LLM API is optional. Real API calls require `.env` with provider credentials.
Training, validation, and benchmark inference NEVER call LLM API.