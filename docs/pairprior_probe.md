# Pair-Prior Probe

Pair-prior is retained only as an optional diagnostic probe. It is not part of the candidate model and should not be reported as a main contribution.

## Status

- Candidate path: disabled.
- Official benchmark path: equivalent to candidate when `--pair_prior none`, which is the default.
- Optional probe path: enabled only with `--pair_prior auto --pair_prior_alpha <value>`.

## Evidence

- Sandbox L6 showed weak positive signal.
- Official alpha sweep on the converged candidate was essentially flat.
- Best official-path delta was about +0.0009 Pair F1.
- This indicates that pair-prior is mostly redundant once the candidate pair head and pairrefine module are trained.

## Interpretation

The pair-prior can be useful as a diagnostic check for whether decoding is sensitive to simple biological pair compatibility. It should not be used to claim model improvement.

## Reproducibility

Default strict benchmark, no pair-prior:

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile --pair_prior none
```

Optional diagnostic probe:

```bash
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile --pair_prior auto --pair_prior_alpha 0.1
```

## Claim Boundary

Do not include pair-prior in the final candidate definition, main result table, or model contribution list.
