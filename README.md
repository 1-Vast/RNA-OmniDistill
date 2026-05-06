# RNA-OmniDiffusion

Pair-refined, constraint-guided RNA secondary structure prediction model with foundation-style extensions.

**Current stage**: Best candidate validated on ArchiveII and bpRNA external benchmark. Multi-task and family-disjoint extensions are preliminary.

## Project Status

| Component | Status |
|---|---|
| Pairrefine + strict Nussinov | ✅ Main contribution |
| ArchiveII benchmark | ✅ F1 = 0.5689 |
| bpRNA external (12,732 samples) | ✅ F1 = 0.5285 (-7.1% drop) |
| 3-seed stability | ✅ Mean = 0.5813 ± 0.0078 |
| Component ablation | ✅ pairrefine +7.2%, masking -17.2% |
| Multi-task (invfold/inpaint) | ⚠️ Preliminary |
| Family-disjoint | ❌ Not yet established |
| LLM semantic tokens | 🔬 Experimental (not effective on current data) |
| Conflict loss | ❌ Rejected |
| Token-only decode | ❌ Failure mode |

## Main Model

Configuration: `config/candidate.yaml` (also aliased as `config/fixed.yaml`)

| Component | Setting |
|---|---|
| pairrefine | **true** (2D Conv2d residual refinement) |
| masking variants | **false** (harmful on ArchiveII) |
| conflict loss | **disabled** (harmful) |
| decoding | strict Nussinov |
| pair head | MLP-based |

Historical baseline: `config/oldbase.yaml` (pairrefine=false, masking=true)

## Main Results

### ArchiveII (in-domain, 338 test samples)

| Model | Pair F1 | Precision | Recall | Valid | Ratio |
|---|---:|---:|---:|---:|---:|
| oldbase | 0.3846 | 0.3398 | 0.4465 | 1.0 | 1.42 |
| norefine | 0.4966 | 0.4470 | 0.5630 | 1.0 | 1.39 |
| **candidate** | **0.5689** | **0.5090** | **0.6517** | **1.0** | **1.38** |

3-Seed: mean F1 = 0.5813 ± 0.0078

### bpRNA External (12,732 test samples, random split)

| Model | Pair F1 | Precision | Recall | Valid | Ratio |
|---|---:|---:|---:|---:|---:|
| oldbase | 0.4234 | 0.4019 | 0.4741 | 1.0 | 1.40 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 1.0 | 1.42 |
| **candidate** | **0.5285** | **0.4877** | **0.6070** | **1.0** | **1.38** |

External drop: -7.1% on 37x larger test set.

### Component Contribution

| Comparison | Delta F1 | Interpretation |
|---|---:|---|
| candidate − norefine (ArchiveII) | +0.0723 | pairrefine |
| candidate − oldmask (ArchiveII) | +0.1716 | disabling masking |
| candidate − oldbase (ArchiveII) | +0.1843 | full improvement |
| candidate − norefine (bpRNA) | +0.0886 | external pairrefine |
| candidate − oldbase (bpRNA) | +0.1051 | external full improvement |

## What Works

- **pairrefine**: 2D Conv2d residual refinement on pair logits.
- **strict Nussinov decoding**: guarantees legal non-crossing structures (valid=1.0).
- **external generalization**: candidate transfers to bpRNA with only -7.1% F1 drop.
- **seed stability**: std F1 = 0.0078 across 3 seeds.

## What Does Not Work

- **token-only decode**: valid_rate = 0.0000.
- **greedy as final metric**: approximate, produces crossing candidates.
- **conflict loss**: harmful at all tested magnitudes.
- **masking variants**: -17.2% F1 when enabled.
- **LLM semantic tokens**: metadata lacks informative descriptions (100% unknown).

## Reproduce

```bash
# Smoke
python main.py smoke

# Train
python main.py train --config config/candidate.yaml --device cuda --max_steps 2000

# Benchmark (ArchiveII)
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov

# External benchmark
python scripts/run.py external --configs config/external_bprna_candidate.yaml config/external_bprna_norefine.yaml config/external_bprna_oldbase.yaml --dataset bprna --split random --device cuda --decode nussinov --bench_workers 8 --tag external_bprna
```

Full reproduction: see `release/reproduce.md`.

## Configs

| Config | Role |
|---|---|
| `config/candidate.yaml` | Best candidate model |
| `config/fixed.yaml` | Alias of candidate |
| `config/oldbase.yaml` | Historical baseline |
| `config/candidate_norefine.yaml` | No pairrefine control |
| `config/candidate_oldmask.yaml` | Old masking control |
| `config/archive_failed/` | Failed/diagnostic probes |

## Experimental Extensions

### Multi-task (`config/multitask_candidate.yaml`)
seq2struct / invfold / inpaint — preliminary, not yet validated.

### LLM Semantic (`scripts/semantic.py`)
Offline semantic annotation pipeline. Not currently effective due to metadata limitations. See `docs/llm_semantic_plan.md` for future roadmap. **LLM is never called during benchmark inference.**

### External Datasets (`scripts/dataset.py`)
Download and prepare bpRNA, Rfam seed, RNAstralign.

## Release

See `release/` for best config, model card, results summary, reproduction guide, and limitations.
