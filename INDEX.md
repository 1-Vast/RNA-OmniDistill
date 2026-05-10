# RNA-OmniPrefold Index

## Core

- `main.py`: CLI entry point.
- `models/omni.py`: RNA encoder, pair head, PairRefine blocks.
- `models/training.py`: Config loading, training loop, loss computation.
- `models/dataset.py`: JSONL dataset loader.
- `models/collator.py`: Batch construction, pair-aware masking.
- `models/decode.py`: Strict Nussinov decoder.
- `models/mask.py`: Masking utilities.
- `utils/metric.py`: Pair F1, structure evaluation.
- `utils/struct.py`: Dot-bracket parsing.

## Configs

- `config/msmprm.yaml`: Recommended mainline (MS-MPRM + PairRefine + pair-aware masking).
- `config/candidate.yaml`: Canonical config (do not edit).

## Scripts

- `scripts/eval.py`: Benchmark / evaluation.
- `scripts/data.py`: Data preparation.
- `scripts/run.py`: Experiment orchestration.
- `scripts/rerank.py`: Top-K candidate reranking (experimental TTO).

## Docs

- `docs/negative.md`: Deprecated / negative routes.
- `docs/architecture.md`: Architecture and validated components.

## Mainline Commands

```bash
python main.py overview
python main.py smoke
python main.py train --config config/msmprm.yaml --device cuda
python scripts/eval.py bench --config config/msmprm.yaml --ckpt outputs/msmprm/best.pt --split test --device cuda --decode nussinov --stage_logits
```
