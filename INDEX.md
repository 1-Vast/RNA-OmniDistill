# Repository Index

This repository is intentionally small. Use this index to find the core training, data, model, and inference paths quickly.

## Start Here

- `README.md`: purpose, data format, commands, and current limitations.
- `main.py`: unified CLI for `train`, `eval`, `infer`, and `smoke`.
- `config/config.yaml`: default data paths, task sampling ratios, model size, training settings, and decoding options.

## Data Pipeline

- `data/tokenizer.py`: `RNAOmniTokenizer`, including sequence, structure, task, motif, family, and segment-related tokens.
- `data/dataset.py`: `RNAOmniDataset`, JSONL loading, validation, pair parsing, motif fallback inference, and length checks.
- `data/collator.py`: `RNAOmniCollator`, task sampling and construction of masked diffusion batches.
- `dataset/processed/*.jsonl`: tiny synthetic smoke-test data.

## Masking And Diffusion Targets

- `models/masking.py`: random token masking, pair-aware expansion, motif-span masking, and contiguous span masking.
- `data/collator.py`: applies the task-specific corruption policy and builds labels only for masked target tokens.

## Model And Loss

- `models/rna_omnidiffusion.py`: Transformer encoder backbone, token heads, pair head, and `compute_omni_loss`.

## Decoding

- `models/decoding.py`: entropy-based iterative unmasking, seq2struct generation, inverse folding generation, and Nussinov-style constrained decoding.

## Structure Utilities And Metrics

- `utils/structure.py`: dot-bracket parsing, pair conversion, structure validation, canonical/wobble pairing, and simple motif inference.
- `utils/metrics.py`: base-pair precision, recall, F1, MCC, token accuracy, valid structure rate, and canonical pair ratio.

## Smoke Test

- `scripts/check_dataset.py`: clean and summarize raw RNA JSONL data before training.
- `scripts/make_splits.py`: create random or family-disjoint train/val/test splits.
- `scripts/overfit_tiny.py`: tiny overfit sanity check for label/mask/pair-loss alignment.
- `scripts/prepare_rna_dataset.py`: convert JSONL, FASTA-like dot-bracket, CT, and BPSEQ inputs to project JSONL.
- `scripts/download_rna_datasets.py`: download or document acquisition for ArchiveII, bpRNA, bpRNA-90, Rfam seed, and optional RNAStrAlign sources.
- `scripts/run_realdata_smoke.py`: quick real-data subset training and diagnostic prediction export.
- `scripts/run_benchmark.py`: checkpoint evaluation on a split with model and simple baselines.
- `scripts/export_predictions.py`: seq2struct prediction export with optional top-k pair probabilities.
- `scripts/analyze_training.py`: train log analysis and automatic failure diagnosis.
- `scripts/diagnose_predictions.py`: per-sample prediction failure mining and bad/good case export.
- `scripts/compare_benchmarks.py`: random vs family-disjoint benchmark table export.
- `scripts/run_ablations.py`: train, benchmark, analyze, and diagnose ablation variants.
- `scripts/compare_ablations.py`: summarize ablation benchmark JSON files into paper-style tables.
- `scripts/run_core_experiments.py`: orchestrate ArchiveII, RNAStrAlign.512, external benchmark, and core ablation workflows.
- `scripts/compare_core_results.py`: summarize core real-data benchmark JSON files.
- `scripts/run_potential_suite.py`: ArchiveII model-potential suite with quick/full modes and no Agent implementation.
- `scripts/summarize_model_potential.py`: combine benchmark, diagnosis, and ablation summaries into model-potential reports.
- `scripts/evaluate_agent_potential.py`: evaluate future Agent-readiness from existing diagnostics without calling an LLM.
- `scripts/run_full_protocol.ps1` / `scripts/run_full_protocol.sh`: full real-data protocol wrappers.
- `scripts/train_realdata.ps1` / `scripts/train_realdata.sh`: end-to-end real-data command wrappers.
- `scripts/smoke_test_rna_omni.py`: script wrapper around `python main.py smoke`.
- `python main.py smoke`: creates tiny JSONL data if needed, runs two training steps, performs seq2struct inference, and prints `smoke_ok`.

## Recommended Reading Order

1. `README.md`
2. `config/config.yaml`
3. `data/tokenizer.py`
4. `data/dataset.py`
5. `data/collator.py`
6. `models/rna_omnidiffusion.py`
7. `models/decoding.py`
8. `utils/structure.py`
9. `utils/metrics.py`
