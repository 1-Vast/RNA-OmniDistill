# RNA-OmniDiffusion

Pair-refined, constraint-guided RNA secondary structure prediction.

**Current release**: candidate model validated on ArchiveII (F1=0.5689) and bpRNA external (F1=0.5285, 12,732 samples).

## Project Status

| Component | Status |
|---|---|
| Pairrefine + strict Nussinov | Main contribution |
| ArchiveII benchmark | F1 = 0.5689 |
| bpRNA external (12,732 samples) | F1 = 0.5285 (-7.1% drop) |
| 3-seed stability | Mean = 0.5813 +/- 0.0078 |
| Component ablation | Complete |
| Multi-task | Preliminary |
| Family-disjoint | Not established |
| LLM semantic | Excluded (no gain) |
| Conflict loss | Rejected |
| Token-only decode | Failure mode |

## Main Model

Configuration: `config/candidate.yaml` (aliased as `config/fixed.yaml`)

- **pairrefine**: 2D Conv2d residual refinement on pair logits
- **masking variants**: disabled
- **conflict loss**: disabled
- **decoding**: strict Nussinov
- **pair head**: MLP-based

## Main Results

### ArchiveII (338 test samples)

| Model | Pair F1 | Precision | Recall | Valid | Ratio |
|---|---:|---:|---:|---:|---:|
| oldbase | 0.3846 | 0.3398 | 0.4465 | 1.0 | 1.42 |
| norefine | 0.4966 | 0.4470 | 0.5630 | 1.0 | 1.39 |
| **candidate** | **0.5689** | **0.5090** | **0.6517** | **1.0** | **1.38** |

3-Seed: mean F1 = 0.5813 +/- 0.0078

### bpRNA External (12,732 test samples, random split)

| Model | Pair F1 | Precision | Recall | Valid | Ratio |
|---|---:|---:|---:|---:|---:|
| oldbase | 0.4234 | 0.4019 | 0.4741 | 1.0 | 1.40 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 1.0 | 1.42 |
| **candidate** | **0.5285** | **0.4877** | **0.6070** | **1.0** | **1.38** |

External drop: -7.1% on 37x larger test set.

### Component Contribution

| Comparison | Delta F1 |
|---|---|
| candidate - norefine (ArchiveII) | +0.0723 |
| candidate - oldmask (ArchiveII) | +0.1716 |
| candidate - oldbase (ArchiveII) | +0.1843 |
| candidate - norefine (bpRNA) | +0.0886 |
| candidate - oldbase (bpRNA) | +0.1051 |

## Reproduce

```bash
# Smoke test
python main.py smoke

# Train candidate
python main.py train --config config/candidate.yaml --device cuda

# Benchmark (strict Nussinov)
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov

# External benchmark
python scripts/run.py external --configs config/external_bprna_candidate.yaml config/external_bprna_norefine.yaml config/external_bprna_oldbase.yaml --dataset bprna --split random --device cuda --decode nussinov --bench_workers 8 --tag external_bprna
```

Full reproduction: `release/reproduce.md`.

## Configs

| Config | Role |
|---|---|
| `config/candidate.yaml` | Best candidate model |
| `config/fixed.yaml` | Alias of candidate |
| `config/oldbase.yaml` | Historical baseline |
| `config/candidate_norefine.yaml` | No pairrefine control |
| `config/candidate_oldmask.yaml` | Old masking control |
| `config/archive_failed/` | Failed/diagnostic probes |

## Release

`release/` contains best config, model card, results summary, reproduction guide, and limitations.

## What Is Not Included

- LLM semantic conditioning (tested, no model-level gain → `docs/llm_negative_result.md`)
- Token-only decode (failure mode)
- Greedy decode as final metric (probe only)
- Conflict loss (harmful)
- Masking variants (harmful on ArchiveII)
- Agent / RNA-FM / 3D / ligand / protein
