# RNA-OmniDiffusion-v2

Minimal, runnable masked discrete diffusion for unified RNA sequence, secondary-structure, motif, and family modeling.

The first target is a smoke-testable training pipeline, not SOTA performance. This version uses a small Transformer encoder trained from scratch and does not include RNA-FM, LoRA, semantic cache, external LLM calls, ligand/protein tasks, or RNA 3D tasks.

See `INDEX.md` for a file-by-file repository guide.

## Data Format

Use JSONL files under `dataset/processed/`. Each line should contain:

```json
{
  "id": "RNA_000001",
  "seq": "GGGAAACCC",
  "struct": "(((...)))",
  "family": "miRNA",
  "motifs": [{"type": "STEM", "start": 0, "end": 8}],
  "pairs": [[0, 8], [1, 7], [2, 6]],
  "length": 9
}
```

`motifs` and `pairs` are optional. If absent, pairs are parsed from dot-bracket structure and simple motifs are inferred. Sequence and structure lengths must match; malformed samples are skipped with a clear warning.

## Commands

Smoke test:

```bash
python main.py smoke
```

Train:

```bash
python main.py train --config config/config.yaml
```

Evaluate:

```bash
python main.py eval --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt
```

Seq2struct inference:

```bash
python main.py infer --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt --task seq2struct --seq AUGGCUACGU
```

Inverse folding inference:

```bash
python main.py infer --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt --task invfold --struct "((...))"
```

## Real-data Training Workflow

1. Check and clean raw JSONL data:

```bash
python scripts/check_dataset.py --input dataset/processed/raw.jsonl --output dataset/processed/clean.jsonl --max_length 512
```

2. Create splits:

```bash
python scripts/make_splits.py --input dataset/processed/clean.jsonl --out_dir dataset/processed --mode random
```

For family-disjoint evaluation:

```bash
python scripts/make_splits.py --input dataset/processed/clean.jsonl --out_dir dataset/processed --mode family_disjoint
```

3. Run a tiny overfit sanity check:

```bash
python scripts/overfit_tiny.py --config config/config.yaml --num_samples 16 --steps 200
```

4. Train:

```bash
python main.py train --config config/config.yaml
```

5. Evaluate:

```bash
python main.py eval --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt
```

6. Infer seq2struct:

```bash
python main.py infer --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt --task seq2struct --seq AUGGCUACGU
```

Training writes `best.pt`, `last.pt`, `train_log.jsonl`, and `config_used.yaml` under the configured output directory. `best.pt` is selected by validation pair F1.

## Real-data Benchmark Workflow

1. Prepare data from JSONL, FASTA-like dot-bracket, CT, or BPSEQ:

```bash
python scripts/prepare_rna_dataset.py --input dataset/raw --output dataset/processed/clean.jsonl --format auto --max_length 512
```

2. Check cleaned data:

```bash
python scripts/check_dataset.py --input dataset/processed/clean.jsonl --output dataset/processed/clean.checked.jsonl --max_length 512
```

3. Create random splits:

```bash
python scripts/make_splits.py --input dataset/processed/clean.checked.jsonl --out_dir dataset/processed --mode random
```

4. Create family-disjoint splits:

```bash
python scripts/make_splits.py --input dataset/processed/clean.checked.jsonl --out_dir dataset/processed_family --mode family_disjoint
```

5. Run real-data smoke:

```bash
python scripts/run_realdata_smoke.py --config config/config.yaml --num_train 128 --num_val 32 --steps 100
```

6. Train:

```bash
python main.py train --config config/config.yaml
```

Resume:

```bash
python main.py train --config config/config.yaml --resume outputs/rna_omnidiffusion_v2/last.pt
```

7. Benchmark:

```bash
python scripts/run_benchmark.py --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt --split test
```

8. Export predictions:

```bash
python scripts/export_predictions.py --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt --input dataset/processed/test.jsonl --output outputs/rna_omnidiffusion_v2/predictions.jsonl
```

## Core Real-data Experiments

Download raw datasets:

```bash
python scripts/download_rna_datasets.py --dataset archiveii --out dataset/raw/archiveii
python scripts/download_rna_datasets.py --dataset bprna90 --out dataset/raw/bprna90
python scripts/download_rna_datasets.py --dataset rfam_seed --out dataset/raw/rfam_seed
```

Run ArchiveII full protocol:

```bash
python scripts/run_core_experiments.py --experiment archiveii_full
```

Dry-run core workflows:

```bash
python scripts/run_core_experiments.py --experiment archiveii_full --quick --dry_run
python scripts/run_core_experiments.py --experiment archiveii_core_ablation --quick --dry_run
```

Run the core ArchiveII ablations:

```bash
python scripts/run_core_experiments.py --experiment archiveii_core_ablation
```

Run external ArchiveII evaluation with an RNAStrAlign.512 checkpoint:

```bash
python scripts/run_core_experiments.py --experiment rnastralign_to_archiveii
```

Compare core benchmark results:

```bash
python scripts/compare_core_results.py --inputs outputs/archiveii_full/benchmark_test.json outputs/rnastralign512_full/benchmark_test.json outputs/rnastralign512_full/archiveii_external_benchmark.json --names archiveii_random rnastralign_random rnastralign_to_archiveii --out outputs/core_results
```

## Model Potential Evaluation

Dry-run the ArchiveII potential suite:

```bash
python scripts/run_potential_suite.py --dataset archiveii --mode quick --dry_run
```

Run a quick pipeline check:

```bash
python scripts/run_potential_suite.py --dataset archiveii --mode quick
```

Run the full ArchiveII potential evaluation:

```bash
python scripts/run_potential_suite.py --dataset archiveii --mode full
```

Outputs:

- `outputs/potential/archiveii_quick/model_potential.md`
- `outputs/potential/archiveii_quick/agent_potential.md`
- `outputs/potential/archiveii_full/model_potential.md`
- `outputs/potential/archiveii_full/agent_potential.md`

`quick` mode validates the workflow only and is not a paper conclusion. `full` mode is the first valid setting for judging model potential. `agent_potential` is only a readiness assessment; no Agent framework is implemented or selected.

## Diagnosis And Benchmark Comparison

Analyze training dynamics:

```bash
python scripts/analyze_training.py --log outputs/rna_omnidiffusion_v2/train_log.jsonl --out outputs/rna_omnidiffusion_v2/training_analysis.json
```

Diagnose prediction failures:

```bash
python scripts/diagnose_predictions.py --pred outputs/rna_omnidiffusion_v2/predictions_test.jsonl --out outputs/rna_omnidiffusion_v2/prediction_diagnosis.json
```

Compare random and family-disjoint benchmark summaries:

```bash
python scripts/compare_benchmarks.py --inputs outputs/random/benchmark_test.json outputs/family_disjoint/benchmark_test.json --names random family_disjoint --out outputs/benchmark_comparison
```

Run the full protocol from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_protocol.ps1
```

Run the full protocol from Linux shell:

```bash
bash scripts/run_full_protocol.sh
```

## Ablation Workflow

The ablation configs live under `config/ablations/` and are shallow overrides on top of `config/config.yaml`.

Quick dry-run check:

```bash
python scripts/run_ablations.py --base_config config/config.yaml --quick --dry_run
```

Run selected variants:

```bash
python scripts/run_ablations.py --base_config config/config.yaml --only full no_pair_head no_nussinov
```

Quick execution for script validation:

```bash
python scripts/run_ablations.py --base_config config/config.yaml --quick --only full no_nussinov
```

Compare ablation benchmarks:

```bash
python scripts/compare_ablations.py --inputs outputs/ablations/full/benchmark_test.json outputs/ablations/no_pair_head/benchmark_test.json --out outputs/ablations/summary
```

Ablation variants:

- `full`: all RNA-OmniDiffusion-v2 components enabled.
- `no_pair_head`: disables pair head, pair loss, and pair-based decoding.
- `no_nussinov`: decodes directly from the structure token head.
- `random_mask_only`: disables pair-aware masking and motif-span masking.
- `no_pair_aware_masking`: disables paired-position mask expansion.
- `no_motif_span_masking`: disables motif-span inpainting masks.
- `no_motif_family_condition`: removes motif and family condition tokens.
- `token_decode`: token-only structure decoding.
- `pair_decode`: pair-probability Nussinov decoding.
- `hybrid_decode`: pair decoding with token-pair compatibility.

## Tasks

The collator samples one task per sample:

- `seq2struct`: sequence visible, structure tokens masked.
- `invfold`: structure visible, sequence tokens masked.
- `inpaint`: random or motif span masked across sequence and structure, with pair-aware expansion.
- `motif_control`: task, motif, and family tokens visible; sequence and structure masked.

Training uses mask-based discrete diffusion: sample `t` in `[0, 1]`, derive a mask ratio, replace selected target tokens with `<MASK>`, and predict clean tokens only at masked positions. Pair loss is binary cross entropy over positive pairs plus sampled negatives.

## Current Limitations

- No pseudoknot decoding by default.
- No RNA 3D, ligand, or protein tasks.
- Minimal Transformer trained from scratch.
- Nussinov decoding is intentionally simple and CPU-friendly.
