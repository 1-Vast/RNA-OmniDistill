# Dataset Processing and Splits

This document describes the data pipeline for RNA-OmniDistill experiments. It covers all datasets, their processing steps, split strategies, and output locations.

---

## Overview

Three datasets are used across the RNA-OmniDistill framework:

| Dataset | Use | Structure labels | Size |
|---------|-----|-----------------|------|
| Rfam (seed) | Sequence-only pretraining | No | ~10M+ raw sequences |
| bpRNA-1m(90) | Sequence-only pretraining + structure fine-tune | Yes (dot-bracket) | 28,370 sequences |
| RNAcentral active | Sequence-only pretraining | No | 8.4 GB FASTA |
| ArchiveII | Supervised fine-tune + benchmark | Yes (dot-bracket) | 3,865 samples |

**Pipeline principle**: Pretraining uses sequence-only data (no structure labels). Fine-tuning and benchmark use data with structure labels. bpRNA test split is never used in pretraining or fine-tuning.

---

## 1. Rfam

### Source

Rfam seed alignments in Stockholm format.

- **Input**: `dataset/Rfam/Rfam.seed.gz` (5.9 MB gzipped)
- **Format**: Stockholm alignment blocks with family accessions
- **Download**: `python scripts/download_datasets.py --datasets Rfam`

### Processing

The `prep_rfam_seed` subcommand in `scripts/data.py` handles Rfam seed processing:

1. **Parse Stockholm blocks**: Each `# STOCKHOLM ... //` block is one alignment.
2. **Extract family accession**: From `#=GF AC` lines.
3. **Remove gap characters**: Strip `-`, `.`, `~` from aligned sequences.
4. **T to U conversion**: All thymine converted to uracil.
5. **Base normalization**: Non-AUCGN bases mapped to `N`.
6. **Length filter**: Keep sequences with length 20-512.
7. **Deduplication** (optional): Remove duplicate sequences via `--dedup`.

### Output

- **Path**: `dataset/processed/rfam/rfam_seed_seq.jsonl`
- **Format**: Sequence-only JSONL (no structure field)
- **Fields**: `id`, `seq`, `family`, `source`

### Command

```bash
python scripts/data.py prep_rfam_seed \
  --input dataset/Rfam/Rfam.seed.gz \
  --output dataset/processed/rfam/rfam_seed_seq.jsonl \
  --min_length 20 --max_length 512 --dedup
```

### Split for Pretraining

The sequence-only JSONL is split into train/val for pretraining:

```bash
python scripts/data.py split_seq_jsonl \
  --input dataset/processed/rfam/rfam_seed_seq.jsonl \
  --out dataset/unlabeled/ \
  --train_ratio 0.9 --val_ratio 0.1 --seed 42
```

Output:
- `dataset/unlabeled/train_seq.jsonl` (90%)
- `dataset/unlabeled/val_seq.jsonl` (10%)

### Rfam FASTA Files (Alternative)

Individual Rfam family FASTA files can also be used via `prep_rfam_fasta`:

```bash
python scripts/data.py prep_rfam_fasta \
  --input dataset/Rfam/fasta_files/ \
  --output dataset/processed/rfam/rfam_fasta_seq.jsonl \
  --min_length 20 --max_length 512 --dedup
```

This processes `RF*.fa.gz` files, one per Rfam family, and produces the same sequence-only JSONL format.

---

## 2. bpRNA-1m(90)

### Source

bpRNA-1m(90) is a non-redundant subset of bpRNA-1m with 90% sequence identity clustering.

- **Input**: `dataset/bpRNA/bpRNA_1m_90.zip` (4.0 GB)
- **Contents**: FASTA (28,370 sequences) + DBN files (28,370 structures)
- **Download**: Manual from http://bprna.cgrb.oregonstate.edu/download.php

### Processing

The `process_bprna()` function in `scripts/process_all_data.py` handles bpRNA processing:

1. **Extract zip**: The zip contains a FASTA file and a directory of `.dbn` files.
2. **Parse DBN files**: Each `.dbn` file has a `#Name:` header, a sequence line, and a dot-bracket structure line.
3. **Match FASTA with DBN**: Sequences in the FASTA are matched to DBN files by sequence ID.
4. **Dot-bracket to pairs**: The dot-bracket string is converted to a list of `(i, j)` base pairs.
5. **Clean sequence**: Uppercase, T to U, non-AUCGN to N.
6. **Length filter**: 20-512 nucleotides.
7. **Unmatched sequences**: Sequences in FASTA without a matching DBN file are skipped.

### Output

Two output types are produced:

**Structure JSONL** (with dot-bracket labels):
- `dataset/processed/bprna/bprna_1m90_train.jsonl`
- `dataset/processed/bprna/bprna_1m90_val.jsonl`
- `dataset/processed/bprna/bprna_1m90_test.jsonl`

**Sequence-only JSONL** (no structure, for pretraining):
- `dataset/processed/bprna/bprna_1m90_seq.jsonl` (all matched sequences)
- `dataset/processed/bprna/bprna_1m90_seq_10k.jsonl` (10k subset)
- `dataset/processed/bprna/bprna_1m90_seq_50k.jsonl` (50k subset)

### Split

80/10/10 random split with fixed seed 42:
- Train: 80% (~22,696)
- Validation: 10% (~2,837)
- Test: 10% (~2,837)

### Command

```bash
python scripts/process_all_data.py
```

This processes both bpRNA and RNAcentral in one run.

---

## 3. RNAcentral Active

### Source

RNAcentral is a comprehensive database of non-coding RNA sequences.

- **Input**: `dataset/RNAcentral/rnacentral_active.fasta.gz` (8.4 GB gzipped)
- **Format**: FASTA with sequence IDs from RNAcentral
- **Download**: Manual from https://rnacentral.org/

### Processing

The `process_rnacentral()` function in `scripts/process_all_data.py` handles RNAcentral processing:

1. **Stream FASTA**: The 8.4 GB gzipped FASTA is streamed line by line (no full decompression).
2. **Clean sequence**: Uppercase, T to U, non-AUCGN to N.
3. **Length filter**: 20-512 nucleotides.
4. **Deduplication**: In-memory set of seen sequences.
5. **Candidate collection**: Up to 3x the target sample size is collected.
6. **Reservoir sampling**: Random sample of the target size is drawn with fixed seed 42.

### Output

- `dataset/processed/rnacentral/rnacentral_active_50k.jsonl` (50,000 sequences)
- `dataset/processed/rnacentral/rnacentral_active_100k.jsonl` (100,000 sequences)

### Command

```bash
python scripts/process_all_data.py
```

Or individually:

```bash
python -c "
from scripts.process_all_data import process_rnacentral
process_rnacentral(limit=50000)
process_rnacentral(limit=100000)
"
```

---

## 4. ArchiveII

### Source

ArchiveII is a standard RNA secondary structure benchmark.

- **Input**: `dataset/archive/train.jsonl`, `dataset/archive/val.jsonl`, `dataset/archive/test.jsonl`
- **Format**: JSONL with `id`, `seq`, `struct`, `family`, `motifs`, `pairs`
- **Total**: 3,865 samples (3092 train / 386 val / 387 test)
- **Download**: `python scripts/data.py fetch --set archive --out dataset/archive`

### Pre-split

ArchiveII comes pre-split from the original benchmark. The splits are:
- Train: 3,092 samples
- Validation: 386 samples
- Test: 387 samples

These splits are used directly for supervised fine-tuning and benchmark evaluation. No additional splitting is needed.

### Usage in Configs

ArchiveII is the primary supervised fine-tuning dataset:

```yaml
# config/candidate_from_rnafm_pretrain.yaml
data:
  train_jsonl: dataset/archive/train.jsonl
  val_jsonl: dataset/archive/val.jsonl
  test_jsonl: dataset/archive/test.jsonl
```

---

## 5. Experiment Split Principles

### Pretraining Data (Sequence-Only)

| Dataset | Split | Used in Stage 1? | Notes |
|---------|-------|-------------------|-------|
| Rfam seed | train_seq.jsonl (90%) | Yes | Sequence-only MLM + distillation |
| Rfam seed | val_seq.jsonl (10%) | Yes | Validation for pretraining |
| bpRNA-1m(90) | bprna_1m90_seq.jsonl | Yes | Sequence-only, no structure used |
| RNAcentral | rnacentral_active_50k.jsonl | Yes | Sequence-only |
| RNAcentral | rnacentral_active_100k.jsonl | Yes | Sequence-only |
| bpRNA-1m(90) test | (excluded) | **No** | Never used in pretraining |

### Fine-Tuning Data (With Structure)

| Dataset | Split | Used in Stage 2? | Notes |
|---------|-------|-------------------|-------|
| ArchiveII | train.jsonl | Yes | Supervised fine-tuning |
| ArchiveII | val.jsonl | Yes | Validation for fine-tuning |
| ArchiveII | test.jsonl | Yes | Benchmark evaluation |
| bpRNA-1m(90) | bprna_1m90_train.jsonl | Yes | Additional supervised data |
| bpRNA-1m(90) | bprna_1m90_val.jsonl | Yes | Validation |
| bpRNA-1m(90) | bprna_1m90_test.jsonl | **No** | Held out for external benchmark |

### Key Rules

1. **bpRNA test split is never used** in pretraining or fine-tuning. It is reserved as an external benchmark.
2. **All random splits use fixed seed 42** for reproducibility.
3. **Actual counts are recorded** in processing scripts and output stats.
4. **Sequence-only data** (Rfam, RNAcentral, bpRNA seq) is used only for Stage 1 pretraining.
5. **Structure-labeled data** (ArchiveII, bpRNA struct) is used only for Stage 2 fine-tuning.

---

## 6. Data Leakage

### Known Leakage Risks

| Source | Risk | Mitigation |
|--------|------|------------|
| bpRNA test in pretrain | bpRNA test split is excluded from all pretraining data | Explicit exclusion in processing |
| Rfam / RNAcentral overlap with ArchiveII | Rfam and RNAcentral may contain sequences similar to ArchiveII test sequences | Weak constraint (no sequence identity filtering) |
| RNAcentral overlap with bpRNA | RNAcentral is a comprehensive database that may include bpRNA sequences | Not currently filtered |

### Future Improvements

- **Sequence identity filtering**: Use CD-HIT or MMseqs2 to remove sequences with >80% identity to ArchiveII test and bpRNA test sets.
- **Family-aware splitting**: For Rfam, split by family accession rather than randomly to prevent family-level leakage.
- **Deduplication across datasets**: Cross-dataset deduplication during the merge step.

---

## 7. Output Directory Structure

```
dataset/
  archive/
    train.jsonl          # ArchiveII train (3,092 samples)
    val.jsonl            # ArchiveII val (386 samples)
    test.jsonl           # ArchiveII test (387 samples)
  bpRNA/
    bpRNA_1m_90.zip      # Raw zip (4.0 GB)
  Rfam/
    Rfam.seed.gz         # Raw Stockholm (5.9 MB)
    Rfam.full_region.gz  # Raw full region (125 MB)
    fasta_files/         # Individual family FASTA files
  RNAcentral/
    rnacentral_active.fasta.gz  # Raw FASTA (8.4 GB)
  processed/
    rfam/
      rfam_seed_seq.jsonl       # Sequence-only JSONL
    bprna/
      bprna_1m90_train.jsonl    # Structure JSONL (train)
      bprna_1m90_val.jsonl      # Structure JSONL (val)
      bprna_1m90_test.jsonl     # Structure JSONL (test)
      bprna_1m90_seq.jsonl      # Sequence-only JSONL (all)
      bprna_1m90_seq_10k.jsonl  # Sequence-only JSONL (10k)
      bprna_1m90_seq_50k.jsonl  # Sequence-only JSONL (50k)
    rnacentral/
      rnacentral_active_50k.jsonl   # Sequence-only JSONL (50k)
      rnacentral_active_100k.jsonl  # Sequence-only JSONL (100k)
  unlabeled/
    train_seq.jsonl       # Rfam seed train split (90%)
    val_seq.jsonl         # Rfam seed val split (10%)
    train_seq_rnafm.jsonl # Rfam seed train + RNA-FM embedding paths
    val_seq_rnafm.jsonl   # Rfam seed val + RNA-FM embedding paths
  teacher_emb/
    rnafm/
      train_embeddings.npy  # RNA-FM embeddings for train
      val_embeddings.npy    # RNA-FM embeddings for val
```

---

## 8. Commands Reference

### Full Pipeline (One Shot)

```bash
# Process bpRNA + RNAcentral
python scripts/process_all_data.py
```

### Rfam Seed Processing

```bash
# Extract sequences from Stockholm format
python scripts/data.py prep_rfam_seed \
  --input dataset/Rfam/Rfam.seed.gz \
  --output dataset/processed/rfam/rfam_seed_seq.jsonl \
  --min_length 20 --max_length 512 --dedup

# Split into train/val for pretraining
python scripts/data.py split_seq_jsonl \
  --input dataset/processed/rfam/rfam_seed_seq.jsonl \
  --out dataset/unlabeled/ \
  --train_ratio 0.9 --val_ratio 0.1 --seed 42
```

### Rfam FASTA Processing (Alternative)

```bash
python scripts/data.py prep_rfam_fasta \
  --input dataset/Rfam/fasta_files/ \
  --output dataset/processed/rfam/rfam_fasta_seq.jsonl \
  --min_length 20 --max_length 512 --dedup
```

### bpRNA Processing (Standalone)

```bash
# Extract bpRNA zip first
# Then run process_all_data.py which handles bpRNA + RNAcentral
python scripts/process_all_data.py
```

### RNAcentral Processing (Standalone)

```bash
python -c "
from scripts.process_all_data import process_rnacentral
process_rnacentral(limit=50000)
process_rnacentral(limit=100000)
"
```

### ArchiveII Download

```bash
python scripts/data.py fetch --set archive --out dataset/archive
```

### RNA-FM Teacher Embedding Extraction

```bash
# Extract sequence-level mean-pooled embeddings for pretraining
python scripts/extract_rnafm_embeddings.py \
  --input dataset/unlabeled/train_seq.jsonl \
  --output_jsonl dataset/unlabeled/train_seq_rnafm.jsonl \
  --output_npy dataset/teacher_emb/rnafm/train_embeddings.npy \
  --dummy --limit 256 --embedding_dim 640 --overwrite

python scripts/extract_rnafm_embeddings.py \
  --input dataset/unlabeled/val_seq.jsonl \
  --output_jsonl dataset/unlabeled/val_seq_rnafm.jsonl \
  --output_npy dataset/teacher_emb/rnafm/val_embeddings.npy \
  --dummy --limit 64 --embedding_dim 640 --overwrite
```

### Dataset Inspection

```bash
# Inspect Rfam file format
python scripts/data.py inspect_rfam --input dataset/Rfam/Rfam.seed.gz

# Check dataset metadata
python scripts/check_datasets.py check --root dataset/ --out outputs/dataset_check/local
```

---

## 9. Processing Scripts Summary

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `scripts/data.py` | Rfam processing, splitting, ArchiveII fetch | `prep_rfam_seed`, `prep_rfam_fasta`, `split_seq_jsonl`, `fetch`, `split` |
| `scripts/process_all_data.py` | bpRNA + RNAcentral one-shot processing | `process_bprna()`, `process_rnacentral()` |
| `scripts/extract_rnafm_embeddings.py` | RNA-FM teacher embedding extraction | Main entry point |
| `scripts/download_datasets.py` | Download all raw datasets | Multi-dataset downloader |
| `scripts/check_datasets.py` | Dataset metadata checker | `check`, `compare` |
| `scripts/dataset.py` | Legacy dataset utilities | `download`, `prepare`, `split` |
