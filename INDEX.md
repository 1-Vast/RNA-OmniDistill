# RNA-OmniDistill Index

## Core

- `main.py`: CLI entry point for overview, training, evaluation, inference, smoke tests, and parameter inspection.
- `models/omni.py`: student encoder, token heads, pair-logit relation head, and lightweight 2D relation refinement.
- `models/training.py`: config loading, model construction, loss computation, device helpers, checkpoint loading.
- `models/dataset.py`: JSONL RNA dataset.
- `models/collator.py`: task sampling, masking, segment ids, pair labels, and batch construction.
- `models/decode.py`: strict Nussinov decoding, greedy probe, and staged decode utilities.
- `models/teacher/rnafm_teacher.py`: frozen RNA-FM representation teacher adapter for offline sequence embedding extraction.
- `scripts/extract_rnafm_embeddings.py`: offline sequence-level RNA-FM embedding extraction.
- `scripts/data.py`: data preparation, FASTA/JSONL conversion, split, and validation CLI.
- `scripts/eval.py`: benchmark, export, analysis, diagnosis, and decoding scan CLI.
- `scripts/run.py`: experiment orchestration for sweeps, external benchmarks, ablations, and summaries.
- `scripts/experiments.py`: experiment manager (plan, make_configs, summarize, export_table).
- `scripts/process_all_data.py`: one-shot bpRNA + RNAcentral data processing.
- `scripts/make_experiment_configs.py`: parameterized experiment config generator.
- `scripts/run_experiment_matrix.sh`: server-side experiment runner template.

## Configs

- `config/candidate.yaml`: canonical supervised training configuration.
- `config/seq_pretrain.yaml`: sequence-only masked denoising pretraining.
- `config/candidate_from_seq_pretrain.yaml`: supervised fine-tuning initialized from sequence pretraining.
- `config/seq_pretrain_rnafm.yaml`: sequence-only masked denoising with frozen RNA-FM sequence-level distillation.
- `config/candidate_from_rnafm_pretrain.yaml`: supervised fine-tuning initialized from RNA-FM-distilled pretraining.

## Docs

- `docs/rna_omnidistill.md`: main method formulation and framework architecture.
- `docs/dataset_processing_and_splits.md`: Rfam, bpRNA, RNAcentral data pipeline and split documentation.
- `docs/negative_results.md`: excluded experimental branches (LLM semantic, token-only decode, etc.).
- `docs/experiment_plan.md`: reproducibility, ablation, external benchmark, and calibration experiment matrix.
- `docs/usage.md`: local CLI usage for training, evaluation, inference, data processing, and audits.

## Mainline Commands

```bash
python main.py overview
python main.py smoke
python main.py params --config config/candidate.yaml
python main.py train --config config/candidate.yaml --device cuda
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
python scripts/run.py external --configs config/candidate.yaml --dataset bprna --split random --device cuda --decode nussinov --bench_workers 8 --tag external_bprna
```
