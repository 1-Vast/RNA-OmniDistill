# RNA-OmniPrefold Candidate Reproduction Guide

## Environment

```bash
pip install torch pyyaml numpy tqdm
```

CUDA is recommended for training. CPU is supported for smoke tests and small probes.

## Smoke Test

```bash
python main.py smoke
```

Expected terminal marker:

```text
smoke_ok
```

## Train Candidate Model

```bash
python main.py train --config config/candidate.yaml --device cuda
```

Output:

```text
outputs/candidate/best.pt
outputs/candidate/last.pt
outputs/candidate/trainlog.jsonl
```

## Benchmark ArchiveII with Strict Nussinov

```bash
python scripts/eval.py bench \
  --config config/candidate.yaml \
  --ckpt outputs/candidate/best.pt \
  --split test \
  --device cuda \
  --decode nussinov \
  --stage_logits \
  --workers 8 \
  --chunksize 2 \
  --profile
```

Expected candidate range:

```text
Pair F1 around 0.57
Valid rate = 1.0
```

## Decode-Only Scan

```bash
python scripts/eval.py bench \
  --config config/candidate.yaml \
  --ckpt outputs/candidate/best.pt \
  --split test \
  --device cuda \
  --decode nussinov \
  --decode_only \
  --workers 8 \
  --chunksize 2 \
  --profile \
  --scan config/scan.json
```

This reuses staged logits and scans decoding parameters without retraining.

## External bpRNA Comparison

```bash
python scripts/run.py external \
--configs config/candidate.yaml \
  --dataset bprna \
  --split random \
  --device cuda \
  --decode nussinov \
  --bench_workers 8 \
  --tag external_bprna
```

Expected candidate range:

```text
Pair F1 around 0.53
Valid rate = 1.0
```

## Component Controls

```bash
python scripts/run.py ablate --config config/candidate.yaml --only full nopair nonuss random --device cuda
```

Use strict Nussinov benchmark for all final comparisons.
