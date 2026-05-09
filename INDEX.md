# RNA-OmniPrefold Index

## Core

- `main.py`: CLI entry point for overview, training, evaluation, inference, smoke tests, and parameter inspection.
- `models/omni.py`: student encoder, token heads, pair-logit relation head, and lightweight 2D relation refinement.
- `models/training.py`: config loading, model construction, loss computation, device helpers, checkpoint loading, and preference loss integration.
- `models/pref.py`: preference-ranking loss and buffer utilities.
- `models/dataset.py`: JSONL RNA dataset.
- `models/collator.py`: task sampling, masking, segment ids, pair labels, sample IDs, and batch construction.
- `models/decode.py`: strict Nussinov decoding, greedy probe, and staged decode utilities.
- `utils/reward.py`: dot-bracket/pair conversion helpers and multi-factor structure scoring.
- `scripts/data.py`: data preparation, FASTA/JSONL conversion, split, and validation CLI.
- `scripts/eval.py`: benchmark, export, analysis, diagnosis, and decoding scan CLI.
- `scripts/run.py`: experiment orchestration for sweeps, external benchmarks, ablations, and summaries.
- `scripts/cand.py`: candidate-structure generator for preference optimization.
- `scripts/judge.py`: preference judge (rule, LLM mock, LLM real).
- `scripts/audit.py`: cleanup audit tool.

## Configs

- `config/candidate.yaml`: canonical supervised training configuration (do not edit).
- `config/seq_pretrain.yaml`: sequence-only masked denoising pretraining.
- `config/candidate_from_seq_pretrain.yaml`: supervised fine-tuning initialized from sequence pretraining.
- `config/rulepref.yaml`: training with rule-based preference optimization.
- `config/pref.yaml`: training with LLM-based preference optimization.

## Docs

- `docs/usage.md`: local CLI usage for training, evaluation, inference, data processing, and audits.
- `docs/dataset_processing_and_splits.md`: data pipeline and split documentation.
- `docs/pref.md`: LLM-guided preference optimization documentation.
- `docs/negative.md`: excluded experimental branches and negative results.

## Mainline Commands

```bash
python main.py overview
python main.py smoke
python main.py params --config config/candidate.yaml
python main.py train --config config/candidate.yaml --device cuda
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```

## Preference Commands

```bash
python scripts/cand.py --config config/candidate.yaml --ckpt outputs/candidate/best.pt --input dataset/archive/val.jsonl --out outputs/pref/cand.jsonl --device cuda
python scripts/judge.py --input outputs/pref/cand.jsonl --out outputs/pref/rulebuf.jsonl --mode rule
python scripts/judge.py --input outputs/pref/cand.jsonl --out outputs/pref/llmbuf.mock.jsonl --mode llm --provider mock
python main.py train --config config/rulepref.yaml --device cuda
```
