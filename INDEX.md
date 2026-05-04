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

