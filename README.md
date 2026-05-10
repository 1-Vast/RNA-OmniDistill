# RNA-OmniPrefold

Relation-aware Masked Denoising for Constraint-Guided RNA Folding.

## Validated Mainline

```
RNA-OmniPrefold =
  MS-MPRM
  + PairRefine
  + pair-aware masking
  + rule hard replay
  + strict Nussinov decode
```

## Key Results

| Method | Steps | F1 | Notes |
|---|---|---|---|
| MS-MPRM + hard replay | 300 | 0.2076 | current best 300-step |
| MS-MPRM baseline | 300 | 0.2009 | no replay |
| MS-MPRM | 500 | 0.2527 | current overall best |
| no-pref | 500 | 0.2440 | BCE only |

- Hard replay adds +0.0067 at 300 steps by duplicating selected hard samples
- PairRefine is essential (refine-off drops to 0.1879)
- More training provides major gains (300→500: +0.043)

## Quick Start

```bash
python main.py overview
python main.py smoke
python main.py train --config config/msmprm.yaml --device cuda
python scripts/eval.py bench --config config/msmprm.yaml --ckpt outputs/msmprm/best.pt --split test --device cuda --decode nussinov --stage_logits
```

## LLM Status

LLM-assisted strategies (semantic tokens, preference optimization, reranker, curriculum, hard replay curation, decode policy search) were systematically evaluated but did not outperform deterministic rule-based baselines under controlled comparisons.

**LLM modules are not part of the validated mainline.** See [docs/negative.md](docs/negative.md) for full experimental record.

## Repository

```
main.py
models/  omni.py training.py dataset.py collator.py decode.py mask.py
scripts/ eval.py data.py run.py curate.py replay.py audiflow.py
config/  msmprm.yaml candidate.yaml
docs/    negative.md architecture.md
utils/   metric.py struct.py
```
