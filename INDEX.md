# RNA-OmniPrefold Index

## Core

- `main.py`: CLI entry point for overview, training, evaluation, inference, smoke tests, and parameter inspection.
- `models/omni.py`: student encoder, token heads, pair-logit relation head, and lightweight 2D relation refinement.
- `models/training.py`: config loading, model construction, loss computation, device helpers, checkpoint loading.
- `models/dataset.py`: JSONL RNA dataset.
- `models/collator.py`: task sampling, masking, segment ids, pair labels, and batch construction.
- `models/decode.py`: strict Nussinov decoding, greedy probe, and staged decode utilities.
- `scripts/data.py`: data preparation, FASTA/JSONL conversion, split, and validation CLI.
- `scripts/eval.py`: benchmark, export, analysis, diagnosis, and decoding scan CLI.
- `scripts/run.py`: experiment orchestration for sweeps, external benchmarks, ablations, and summaries.

## Configs

- `config/candidate.yaml`: canonical supervised training configuration.
- `config/seq_pretrain.yaml`: sequence-only masked denoising pretraining.
- `config/candidate_from_seq_pretrain.yaml`: supervised fine-tuning initialized from sequence pretraining.

## Docs

- `docs/usage.md`: local CLI usage for training, evaluation, inference, data processing, and audits.
- `docs/dataset_processing_and_splits.md`: data pipeline and split documentation.
- `docs/negative.md`: excluded experimental branches and negative results.

## Mainline Commands

```bash
python main.py overview
python main.py smoke
python main.py params --config config/candidate.yaml
python main.py train --config config/candidate.yaml --device cuda
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```
