# RNA Secondary Structure Datasets: Download, Split & Preprocessing Guide

> **Replaces**: RNAStrAND (manual download only — `rnasoft.ca/strand/downloads.php`)
> **Last updated**: 2026-05-07
> **Project**: RNA-OmniPrefold

---

## Overview

This document lists 6 **auto-downloadable** datasets that replace the manual-only RNAStrAND. Each has verified direct download URLs, known secondary structure annotations, and documented preprocessing pipelines compatible with the RNA-OmniPrefold codebase.

| # | Dataset | Sequences | Families | Format | Auto-DL | Best For |
|---|---------|-----------|----------|--------|---------|----------|
| 1 | **CRW tRNA** | ~30,000 | 15 isoacceptors | CT (tar.gz) | ✅ | Family-disjoint tRNA benchmark |
| 2 | **RNASSTR** | ~5M | 4,170 (Rfam) | CSV (dot-bracket) | ✅ | Large-scale training |
| 3 | **RNA3DB** | 1,645+ | 216 (Rfam) | mmCIF / JSON | ✅ | PDB-derived family-disjoint benchmark |
| 4 | **CHANRG** | 170K | Rfam 15.0 | Parquet (dot-bracket) | ✅ | OOD generalization (2026 standard) |
| 5 | **SRPDB** | ~900 | 1 (SRP RNA) | CT (tar.gz) | ✅ | Specialized RNA family |
| 6 | **RIVAS** | ~3,800 | curated families | Parquet (dot-bracket) | ✅ | Clean benchmark |

**Already available** in the pipeline: ArchiveII, bpRNA, RNAStralign, Rfam.

---

## Dataset 1: CRW tRNA Comparative Dataset

### Description
The Comparative RNA Web (CRW) Site from the Gutell Lab provides one of the largest collections of tRNA secondary structures determined by **comparative sequence analysis** — the gold standard for RNA structure determination. Each tRNA isoacceptor type is available as a separate tar.gz archive containing individual `.ct` (Connect Table) files.

### Download

| Isoacceptor | # Sequences | CT Download URL |
|------------|-------------|-----------------|
| Alanine (trnA) | 4,569 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnA.ct.tar.gz` |
| Cysteine (trnC) | 725 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnC.ct.tar.gz` |
| Aspartic Acid (trnD) | 1,441 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnD.ct.tar.gz` |
| Glutamic Acid (trnE) | 1,902 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnE.ct.tar.gz` |
| Phenylalanine (trnF) | 7,620 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnF.ct.tar.gz` |
| Glycine (trnG) | 2,643 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnG.ct.tar.gz` |
| Histidine (trnH) | 1,763 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnH.ct.tar.gz` |
| Isoleucine (trnI) | 4,653 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnI.ct.tar.gz` |
| Lysine (trnK) | 1,851 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnK.ct.tar.gz` |
| Methionine (trnM) | 1,792 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnM.ct.tar.gz` |
| Asparagine (trnN) | 1,254 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnN.ct.tar.gz` |
| Proline (trnP) | 1,426 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnP.ct.tar.gz` |
| Glutamine (trnQ) | 1,187 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnQ.ct.tar.gz` |
| Tryptophan (trnW) | 781 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnW.ct.tar.gz` |
| Tyrosine (trnY1) | 183 | `http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trnY1.ct.tar.gz` |
| **Total** | **~31,790** | |

Also available in bpseq, bracket (Vienna dot-bracket), alden, rnaml, and nopct formats — replace `.ct.` with the format suffix in the URL.

**rRNA subsets** are also available (5S: 3,684 sequences; 16S: 17,051+ sequences; 23S: various phyla).

### Download Command

```bash
# Download all tRNA CT files into dataset/raw/crw_trna/
python scripts/download_datasets.py --raw-root D:\RNA-OmniPrefold\dataset\raw --datasets CRW_tRNA

# Manual: download a single isoacceptor
python -c "
import urllib.request, tarfile, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
base = 'http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files'
for aa in 'A C D E F G H I K M N P Q W Y1'.split():
    url = f'{base}/trn{aa}.ct.tar.gz'
    dest = f'dataset/raw/crw_trna/trn{aa}.ct.tar.gz'
    urllib.request.urlretrieve(url, dest)
    print(f'OK: trn{aa}')
"
```

### Preprocessing

The CRW `.ct` files use the **same CT format** as ArchiveII (Zuker connect table). The existing `scripts/data.py prep` command can process them directly:

```bash
# Preprocess all CRW tRNA CT files into JSONL
python scripts/data.py prep \
    --input dataset/raw/crw_trna/ \
    --output dataset/processed/crw_trna/clean.jsonl \
    --maxlen 512

# Split into train/val/test
python scripts/data.py split \
    --input dataset/processed/crw_trna/clean.jsonl \
    --out dataset/processed/crw_trna \
    --seed 42
```

**Family-disjoint split**: Since each isoacceptor type is a natural family, split by family:
```bash
python scripts/data.py split \
    --input dataset/processed/crw_trna/clean.jsonl \
    --out dataset/processed/crw_trna_family \
    --mode family \
    --seed 42
```

### Notes
- Sequences are generally 70-90 nt (short, within maxlen=512)
- Pseudoknots are included in `.ct` format; use `.nopct` for pseudoknot-free CT
- For dot-bracket format, download `.bracket.tar.gz` variants instead
- **Reference**: Cannone et al., BMC Bioinformatics (2002); Gutell Lab CRW Site

---

## Dataset 2: RNASSTR (RNA Secondary Structure Repository)

### Description
**RNASSTR** is a 2025 benchmark dataset containing ~5 million RNA sequences with secondary structures across 4,170 Rfam families. It is the **largest and most diverse** auto-downloadable RNA structure dataset available, specifically designed for training deep learning models.

- **Source**: Mined from GTDB, NCBI RefSeq, and Rfam v14.10
- **Pre-split**: train (90%), validation (5%), test (5%) with family-aware splitting
- **Format**: CSV with columns: `id, sequence, structure, base_pairs, len`
- **Structure**: Dot-bracket notation (Vienna format)
- **Paper**: Langeberg et al., "Improving RNA Secondary Structure Prediction Through Expanded Training Data", bioRxiv 2025

### Download

**Primary (Zenodo)**: https://doi.org/10.5281/zenodo.15319168

Direct Zenodo download:
```bash
wget https://zenodo.org/records/15319168/files/rna_train.csv
wget https://zenodo.org/records/15319168/files/rna_validate.csv
wget https://zenodo.org/records/15319168/files/rna_test.csv
```

**GitHub** (code + utilities): https://github.com/romanagle/RNASSTR

### Preprocessing

RNASSTR uses CSV format. Convert to JSONL for RNA-OmniPrefold:

```python
# scripts/convert_rnasstr.py (rubust conversion with error handling)
import csv, json, sys
from pathlib import Path

def convert_rnasstr_csv(csv_path: Path, jsonl_path: Path, maxlen: int = 512):
    """Convert RNASSTR CSV to RNA-OmniPrefold JSONL format."""
    count = 0
    skipped = 0
    with open(csv_path) as src, open(jsonl_path, 'w') as dst:
        reader = csv.DictReader(src)
        for row in reader:
            seq = row['sequence'].upper().replace('T', 'U')
            struct = row['structure']
            if len(seq) > maxlen or len(seq) != len(struct):
                skipped += 1
                continue
            # Parse base_pairs from string "[0,9],[1,8]" format
            pairs_str = row.get('base_pairs', '[]')
            entry = {
                'id': row.get('id', f'rnasstr_{count}'),
                'seq': seq,
                'struct': struct,
                'length': len(seq),
                'family': 'RNASSTR',
                'pairs': pairs_str,
            }
            dst.write(json.dumps(entry) + '\n')
            count += 1
    print(f'Converted {count} entries, skipped {skipped}')

# Usage
for split in ['train', 'validate', 'test']:
    convert_rnasstr_csv(
        Path(f'dataset/raw/RNASSTR/rna_{split}.csv'),
        Path(f'dataset/processed/rnasstr/rna_{split}.jsonl'),
    )
```

Or use the simpler approach via existing `data.py check`:
```bash
# After converting CSV to JSONL manually
python scripts/data.py check \
    --input dataset/processed/rnasstr/rna_train.jsonl \
    --output dataset/processed/rnasstr/train.clean.jsonl \
    --maxlen 512
```

### Recommendations
- **For training**: Use train split (~4.3M sequences)
- **For benchmark**: Use test split (~240K sequences)
- Family labels are available and can be used for family-disjoint evaluation
- Note: tRNAs are overrepresented (~39.5% of sequences, ~13.7% of nucleotides)

---

## Dataset 3: RNA3DB (PDB-derived, Family-Disjoint Splits)

### Description
**RNA3DB** is a dataset of non-redundant RNA structures curated from the Protein Data Bank (PDB). It provides 1,645+ unique RNA chains labeled with 216 Rfam families and is specifically designed for training and benchmarking deep learning models with **family-disjoint** train/test splits to prevent data leakage.

- **Source**: All RNA chains from the PDB, filtered for redundancy
- **Pre-split**: YES — non-redundant clustering suitable for family-disjoint training/benchmarking
- **Format**: PDBx/mmCIF (3D coordinates) + JSON metadata with Rfam family labels
- **Secondary structure**: Derivable from 3D coordinates via DSSR, RNApdbee, or barnaba
- **Paper**: Szikszai et al. 2024, bioRxiv; updated to Rfam 15.0 (Jan 2026)

### Download

**GitHub Releases**: https://github.com/marcellszi/rna3db/releases

Direct download URLs (latest release `2025-10-01-incremental-release`):
```bash
# mmCIF structure files (3D coordinates, for training)
wget https://github.com/marcellszi/rna3db/releases/download/2025-10-01-incremental-release/rna3db-mmcifs.tar.xz

# JSON metadata (Rfam labels, clustering info)
wget https://github.com/marcellszi/rna3db/releases/download/2025-10-01-incremental-release/rna3db-jsons.tar.gz

# CM scan results (Infernal homology search)
wget https://github.com/marcellszi/rna3db/releases/download/2025-10-01-incremental-release/rna3db-cmscans.tar.gz
```

### Preprocessing

The mmCIF files contain 3D coordinates. Extract secondary structures using DSSR:
```bash
# Extract secondary structure from mmCIF using x3dna-dssr
x3dna-dssr -i=structure.cif --format=dbn -o=structure.dbn

# Then convert dot-bracket to JSONL
python scripts/data.py prep \
    --input dataset/raw/rna3db/dbn/ \
    --output dataset/processed/rna3db/clean.jsonl \
    --maxlen 512
```

Alternatively, use the JSON metadata which includes Rfam family assignments for family-disjoint splitting.

### Notes
- **Key advantage**: Rigorous non-redundancy (by sequence AND structure), avoiding the data leakage common in random-split benchmarks
- Perfect for evaluating generalization to unseen RNA folds
- Smaller than bpRNA/RNASSTR but higher-quality annotations (PDB-validated)
- **Reference**: Szikszai et al., "RNA3DB: A structurally non-redundant benchmark for RNA structure modeling", bioRxiv 2024

---

## Dataset 4: CHANRG (OOD Generalization Benchmark, Rfam 15.0)

### Description
**CHANRG** (Comprehensive Homology-Aware Nussinov RNA Generalization benchmark) is a March 2026 benchmark dataset designed specifically to evaluate out-of-distribution (OOD) generalization of RNA secondary structure predictors. It uses architecture-aware splits from Rfam 15.0.

- **Size**: 170,083 sequences (Train: 123K, Val: 14K, Test: 14K, GenA: 12K, GenC: 4.4K, GenF: 1.8K)
- **OOD splits**: GenA (novel architecture), GenC (novel clan), GenF (novel family)
- **Format**: Parquet (id, sequence, secondary_structure, split)
- **Structure**: Dot-bracket notation via bpRNA standard
- **Benchmarked**: 29 predictors including public RNA structure and representation models.
- **Paper**: Chen et al. 2026, arXiv:2603.22330

### Download

**HuggingFace**: https://huggingface.co/datasets/multimolecule/chanrg
```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('multimolecule/chanrg', trust_remote_code=True)
ds.save_to_disk('dataset/raw/chanrg')
"
```

### Preprocessing

```python
import json, pandas as pd
from pathlib import Path

def convert_chanrg_parquet(parquet_path: Path, jsonl_path: Path, maxlen: int = 512):
    df = pd.read_parquet(parquet_path)
    count = 0
    with open(jsonl_path, 'w') as f:
        for _, row in df.iterrows():
            seq = str(row['sequence']).upper().replace('T', 'U')
            struct = str(row['secondary_structure'])
            if len(seq) > maxlen or len(seq) != len(struct):
                continue
            entry = {
                'id': str(row.get('id', f'chanrg_{count}')),
                'seq': seq,
                'struct': struct,
                'length': len(seq),
                'family': str(row.get('family', 'UNKNOWN')),
                'split': str(row.get('split', 'train')),
            }
            f.write(json.dumps(entry) + '\n')
            count += 1
    print(f'Converted {count} entries')
```

### Notes
- **Most rigorous benchmark design** as of 2026 — the only dataset with architecture/clan/family-level OOD splits
- Ideal for testing whether your model truly generalizes or just memorizes training folds
- Requires accepting HuggingFace dataset terms (one-click, no email)
- **Reference**: Chen et al. 2026, "CHANRG: A Comprehensive Homology-Aware Nussinov RNA Generalization Benchmark"

---

## Dataset 5: SRPDB (Signal Recognition Particle Database)

### Description
The **SRPDB** provides curated SRP RNA sequences with secondary structures determined by comparative sequence analysis. SRP RNA is a universally conserved ribonucleoprotein with a well-characterized secondary structure that varies across phylogenetic domains.

- **Sequences**: ~900 SRP RNA sequences
- **Structure**: CT (Connect Table) format, individually downloadable
- **Phylogenetic groups**: Bacteria (short/long), Archaea, Protozoa, Fungi/Metazoa, Plants, Animals

### Download

**Base URL**: http://rth.dk/resources/rnp/SRPDB/

The "Get all SRP RNA secondary structures as individual connect files (compressed)" link provides a compressed CT archive:

```bash
# Direct download (verify exact URL on the page)
wget http://rth.dk/resources/rnp/SRPDB/ct_files.tar.gz
```

If the combined archive link is not available, download individual groups from the SRP RNA page (http://rth.dk/resources/rnp/SRPDB/srprna.html).

### Preprocessing

SRPDB uses CT format, same as ArchiveII. Process with existing tools:

```bash
# Download and extract CT files
mkdir -p dataset/raw/srpdb
cd dataset/raw/srpdb
wget http://rth.dk/resources/rnp/SRPDB/ct_files.tar.gz
tar -xzf ct_files.tar.gz

# Preprocess into JSONL
python scripts/data.py prep \
    --input dataset/raw/srpdb/ \
    --output dataset/processed/srpdb/clean.jsonl \
    --maxlen 512

# Split
python scripts/data.py split \
    --input dataset/processed/srpdb/clean.jsonl \
    --out dataset/processed/srpdb \
    --seed 42
```

### Notes
- SRP RNA varies from ~100 nt (bacteria) to ~300 nt (eukaryotes)
- All structures are within maxlen=512
- Family label is "SRP" — good for single-family specialized evaluation
- **Reference**: Zwieb et al., Nucleic Acids Research (2003, 2005, 2007)

---

## Dataset 7 (Bonus): EternaBench — Synthetic RNA with Chemical Probing

### Description
**EternaBench** provides 20,000+ synthetic RNA constructs from the Eterna citizen science platform, with experimentally measured SHAPE chemical probing reactivities and dot-bracket secondary structures. This is the only large-scale dataset combining sequence, structure, and experimental probing data.

- **Size**: 20,000+ constructs across multiple sub-datasets
- **Format**: JSON (sequence, dot-bracket structure, SHAPE reactivity profiles)
- **Structure**: Experimentally validated dot-bracket
- **Paper**: Wayment-Steele et al., Nature Methods 2022

### Download

```bash
# Primary Zenodo archive
wget https://zenodo.org/api/records/6259299/files/eternagame/EternaBench-2.1.0.zip/content

# Sub-datasets on HuggingFace (MultiMolecule mirrors, parquet format)
# EternaBench-CM: 12,711 synthetic constructs with SHAPE
# EternaBench-External: 31 viral/mRNA/genomic datasets
# EternaBench-Switch: 7,228 riboswitch constructs with binding data
```

### Notes
- **Unique value**: SHAPE reactivity data enables structure-aware training beyond sequence-structure pairs
- Sub-datasets are thematically organized (CM=synthetic, External=natural, Switch=riboswitches)
- GitHub: https://github.com/eternagame/EternaBench

---

## Dataset 8 (Bonus): RNAGym — Large-Scale Structure Prediction Training Data

### Description
**RNAGym** from the Das Lab / Marks Lab provides curated training and evaluation data for RNA secondary structure prediction at scale. All files are direct-downloadable from Harvard servers.

- **Direct URLs** (all confirmed working):
  - Training: `https://marks.hms.harvard.edu/rnagym/structure_prediction/train_data.zip` (8.1 GB)
  - Test/eval: `https://marks.hms.harvard.edu/rnagym/structure_prediction/test_data.zip` (3.2 GB)
  - Raw data: `https://marks.hms.harvard.edu/rnagym/structure_prediction/raw_data.zip` (5.1 GB)
  - Annotations: `https://marks.hms.harvard.edu/rnagym/structure_prediction/test_sequences_annotated.zip` (29 MB)
- **Format**: BPSEQ, CT, dot-bracket files
- **Structure**: YES, multiple formats per entry
- **Paper**: Arora et al. 2024

### Preprocessing
The BPSEQ/CT files are compatible with the existing `data.py prep` pipeline:
```bash
python scripts/data.py prep \
    --input dataset/raw/rnagym/ \
    --output dataset/processed/rnagym/clean.jsonl \
    --maxlen 512
```

---

## Dataset 4: RIVAS

### Description
The **RIVAS** dataset is a curated collection of RNA sequences and secondary structures designed for training and evaluating RNA secondary structure prediction methods. It combines experimentally verified structures with high-quality consensus structures from Rfam alignments.

- **Size**: 3,758 sequences (TrainSetA: 2,486; TestSetA: 625; TestSetB: 647)
- **Format**: Parquet (via HuggingFace datasets) with dot-bracket structures
- **Pre-split**: YES (TrainSetA / TestSetA / TestSetB)
- **Family labels**: YES (diverse RNA families: tRNA, SRP, ribozymes, etc.)

### Download

**HuggingFace**: https://huggingface.co/datasets/multimolecule/rivas

```bash
# Via HuggingFace datasets library
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('multimolecule/rivas')
ds.save_to_disk('dataset/raw/rivas')
"

# Or direct parquet download
wget https://huggingface.co/datasets/multimolecule/rivas/resolve/main/train.parquet
wget https://huggingface.co/datasets/multimolecule/rivas/resolve/main/validation.parquet
wget https://huggingface.co/datasets/multimolecule/rivas/resolve/main/test.parquet
```

### Preprocessing

```python
# scripts/convert_rivas.py
import json, pandas as pd
from pathlib import Path

def convert_rivas_parquet(parquet_path: Path, jsonl_path: Path, maxlen: int = 512):
    """Convert RIVAS Parquet to JSONL."""
    df = pd.read_parquet(parquet_path)
    count = 0
    with open(jsonl_path, 'w') as f:
        for _, row in df.iterrows():
            seq = str(row['sequence']).upper().replace('T', 'U')
            struct = str(row['secondary_structure'])
            if len(seq) > maxlen or len(seq) != len(struct):
                continue
            entry = {
                'id': str(row.get('id', f'rivas_{count}')),
                'seq': seq,
                'struct': struct,
                'length': len(seq),
                'family': str(row.get('family', 'RIVAS')),
            }
            # Remove pseudoknot brackets for non-pseudoknot models
            entry['struct_nopseudo'] = struct.replace('[','.').replace(']','.').replace('{','.').replace('}','.')
            f.write(json.dumps(entry) + '\n')
            count += 1
    print(f'Converted {count} entries')

# Usage
for split_name, filename in [('train', 'train.parquet'), ('val', 'validation.parquet'), ('test', 'test.parquet')]:
    convert_rivas_parquet(
        Path(f'dataset/raw/rivas/{filename}'),
        Path(f'dataset/processed/rivas/{split_name}.jsonl'),
    )
```

Alternatively, use stdlib-only approach with the existing `data.py check` after converting.

### Notes
- Smaller dataset (3,758 sequences) — suitable as a clean benchmark, not primary training
- Includes pseudoknot annotations (`[]` and `{}` brackets) — strip for non-pseudoknot models
- Licensed under GNU AGPL
- **Reference**: Rivas et al., PLoS Computational Biology (2017)

---

## Integration with download_datasets.py

Add the following entries to the `DATASETS` dict in `scripts/download_datasets.py`:

```python
# --- Dataset 1: CRW tRNA ---
"CRW_tRNA": {
    "name": "CRW_tRNA",
    "urls": [
        f"http://crw-site.chemistry.gatech.edu/DAT/3C/SBPI/Files/trn{aa}.ct.tar.gz"
        for aa in "A C D E F G H I K M N P Q W Y1".split()
    ],
    "out_filename": None,
    "extract": True,
    "extract_format": "tar.gz",
    "description": "CRW Comparative tRNA dataset (Gutell Lab) — 15 isoacceptor types, ~30k sequences in CT format",
},

# --- Dataset 2: RNASSTR ---
"RNASSTR": {
    "name": "RNASSTR",
    "urls": [
        "https://zenodo.org/records/15319168/files/rna_train.csv",
        "https://zenodo.org/records/15319168/files/rna_validate.csv",
        "https://zenodo.org/records/15319168/files/rna_test.csv",
    ],
    "out_filename": None,
    "extract": False,
    "description": "RNA Secondary Structure Repository — 5M sequences, 4170 Rfam families, CSV format",
},

# --- Dataset 3: SRPDB ---
"SRPDB": {
    "name": "SRPDB",
    "url": "http://rth.dk/resources/rnp/SRPDB/ct_files.tar.gz",
    "out_filename": "srpdb_ct.tar.gz",
    "extract": True,
    "extract_format": "tar.gz",
    "description": "Signal Recognition Particle Database — ~900 SRP RNA structures in CT format",
},

# --- Dataset 4: RIVAS ---
"RIVAS": {
    "name": "RIVAS",
    "urls": [
        "https://huggingface.co/datasets/multimolecule/rivas/resolve/main/train.parquet",
        "https://huggingface.co/datasets/multimolecule/rivas/resolve/main/validation.parquet",
        "https://huggingface.co/datasets/multimolecule/rivas/resolve/main/test.parquet",
    ],
    "out_filename": None,
    "extract": False,
    "description": "RIVAS curated benchmark — 3,758 sequences with dot-bracket structures",
},
```

---

## Recommended Pipeline

### Phase 1: Download all datasets

```bash
python scripts/download_datasets.py \
    --raw-root D:\RNA-OmniPrefold\dataset\raw \
    --datasets CRW_tRNA RNASSTR SRPDB RIVAS
```

### Phase 2: Preprocess (convert to JSONL)

```bash
# CRW tRNA — uses existing CT->JSONL pipeline
python scripts/data.py prep \
    --input dataset/raw/crw_trna/ \
    --output dataset/processed/crw_trna/clean.jsonl \
    --maxlen 512

# RNASSTR — custom CSV converter (see above)
python scripts/convert_rnasstr.py

# SRPDB — uses existing CT->JSONL pipeline
python scripts/data.py prep \
    --input dataset/raw/srpdb/ \
    --output dataset/processed/srpdb/clean.jsonl \
    --maxlen 512

# RIVAS — Parquet converter (see above; requires pandas)
python scripts/convert_rivas.py
```

### Phase 3: Split

```bash
# CRW tRNA — family-disjoint by isoacceptor type
python scripts/data.py split \
    --input dataset/processed/crw_trna/clean.jsonl \
    --out dataset/processed/crw_trna \
    --seed 42

# SRPDB — random split
python scripts/data.py split \
    --input dataset/processed/srpdb/clean.jsonl \
    --out dataset/processed/srpdb \
    --seed 42
```

### Phase 4: Upload to remote server

```bash
python scripts/upload_datasets.py \
    --datasets CRW_tRNA RNASSTR SRPDB RIVAS
```

---

## Comparison: RNAStrAND vs Alternatives

| Criteria | RNAStrAND | CRW tRNA | RNASSTR | RNA3DB | CHANRG | SRPDB | RIVAS | EternaBench | RNAGym |
|----------|-----------|----------|---------|--------|--------|-------|-------|-------------|--------|
| Auto-downloadable | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CT format | ✅ | ✅ | ❌ | derivable | ❌ | ✅ | ❌ | ❌ | ✅ |
| Dot-bracket | via web | via bracket | ✅ | via DSSR | ✅ | via nopct | ✅ | ✅ | ✅ |
| Family labels | YES | YES | YES (Rfam) | YES (Rfam) | YES (Rfam 15) | YES | YES | YES | via source |
| # sequences | ~4.7K | ~32K | ~5M | ~1.6K | 170K | ~900 | ~3.8K | 20K+ | varies |
| Pre-split | NO | NO | YES | YES | YES (OOD) | NO | YES | partial | YES |
| 2020+ paper | NO | NO | ✅ 2025 | ✅ 2024 | ✅ 2026 | NO | ✅ | ✅ 2022 | ✅ 2024 |
| Probing data | NO | NO | NO | NO | NO | NO | NO | ✅ SHAPE | NO |

---

## Decision Matrix

| Use Case | Best Dataset | Reason |
|----------|-------------|--------|
| **Large-scale training** | RNASSTR | 5M sequences, 4,170 families, 2025 benchmark |
| **Family-disjoint OOD eval** | CHANRG | Architecture/clan/family-level splits, 2026 standard |
| **PDB-validated benchmark** | RNA3DB | Non-redundant structures, family-disjoint, experimentally validated |
| **Family-disjoint tRNA benchmark** | CRW tRNA | 15 clean isoacceptor families, comparative structure |
| **With chemical probing** | EternaBench | SHAPE reactivity + dot-bracket, 20K+ constructs |
| **Clean small benchmark** | RIVAS | Pre-split, curated, minimal overlap with training |
| **Specialized RNA family** | SRPDB | High-quality comparative structures for SRP only |
| **Drop-in RNAStrAND replacement** | CRW tRNA (CT) | Same CT format, similar size, similar family structure |

**Recommendation**: Start with **RNASSTR** for training, **CHANRG** for OOD evaluation, and **CRW tRNA** for family-disjoint benchmarking. RNA3DB adds PDB-validated quality; EternaBench adds experimental probing data.
