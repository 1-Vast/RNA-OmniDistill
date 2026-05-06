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
