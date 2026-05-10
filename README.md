# RNA-OmniPrefold

Relation-aware Masked Denoising for Constraint-Guided RNA Folding.

RNA-OmniPrefold predicts RNA secondary structures using relation-aware masked modeling, pair refinement, and strict Nussinov decoding.

## Validated Mainline

```
RNA-OmniPrefold =
  MS-MPRM
  + PairRefine (2D convolutional refinement)
  + pair-aware masking
  + Pair BCE loss
  + strict Nussinov decode
```

## Key Results

| Method | Steps | Pair F1 | Notes |
|---|---|---|---|
| MS-MPRM | 500 | 0.2527 | current best |
| no-pref | 500 | 0.2440 | BCE only |
| MS-MPRM | 300 | 0.2009 | pair-aware + refine |
| baseline | 300 | 0.1852 | base |

- PairRefine contributes +0.0130 F1 (refine-off drops to 0.1879)
- More training provides +0.0431 F1 (300→500 steps)
- Valid structure rate: 1.0 (guaranteed by strict Nussinov)

## Quick Start

```bash
python main.py overview
python main.py smoke
python main.py train --config config/msmprm.yaml --device cuda
python scripts/eval.py bench --config config/msmprm.yaml --ckpt outputs/msmprm/best.pt --split test --device cuda --decode nussinov --stage_logits
```

## Deprecated / Negative Routes

Several experimental routes were tested but did not yield reliable improvements. See [docs/negative.md](docs/negative.md) for details:

- Representation distillation (weak)
- Language-model semantic conditioning (negative)
- Pairwise preference optimization (inconclusive at full budget)
- PairLossPolicy / weighted BCE (negative)
- LLM reranker (does not exceed Rule)
- Structural tag auxiliary (no improvement)

## Repository Structure

```
main.py
models/
  omni.py        -- RNA encoder + pair head + pairrefine
  training.py    -- config, training loop, loss
  dataset.py     -- JSONL dataset loader
  collator.py    -- batch construction, masking
  decode.py      -- Nussinov decoder
  mask.py        -- masking utilities
scripts/
  eval.py        -- benchmark / evaluation
  data.py        -- data preparation
  run.py         -- experiment orchestration
  rerank.py      -- Top-K candidate reranking (experimental TTO)
config/
  msmprm.yaml    -- recommended mainline config
  candidate.yaml -- canonical config (do not edit)
docs/
  negative.md    -- negative / inconclusive routes
  architecture.md
utils/
  metric.py      -- Pair F1 / structure evaluation
  struct.py      -- dot-bracket parsing
```

## What Not To Claim

- Not powered by language models.
- Does not use language-model-generated semantic tokens.
- Does not use pseudo-structure labels.
- Not a general RNA foundation model.
