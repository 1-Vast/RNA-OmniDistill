# RNA-OmniDistill: Relational Masked Diffusion for Constraint-Guided RNA Folding

## Motivation

RNA secondary structure prediction needs both local sequence modeling and global validity constraints. RNA-OmniDistill frames folding as relation-aware masked denoising: the student learns sequence representations from unlabeled RNA, adapts those representations to supervised pair-relation prediction, and projects final pair scores into valid dot-bracket structures with strict Nussinov decoding.

## Relation-Aware Formulation

For an RNA sequence of length `L`, the model uses:

- `x_i`: nucleotide token at position `i`
- `y_i`: structure token at position `i`
- `r_ij`: pair-relation variable for positions `i, j`
- `S_ij`: pair-relation logit predicted by the relation head
- `R`: pair-relation field over all position pairs

The encoder produces contextual sequence states. A pair-logit relation head maps pairs of hidden states into `S_ij`, and lightweight 2D relation refinement operates over the full relation field `R`.

## Stage 1: Sequence-Only Pretraining

Stage 1 uses unlabeled RNA sequences only.

- objective: masked nucleotide denoising
- optional teacher: frozen RNA-FM sequence-level representation distillation
- teacher signal: one mean-pooled embedding per sequence
- no structural labels
- no pair pseudo-labels

The pretraining loss is:

```text
L_pretrain = L_denoise + lambda_d * L_distill
```

RNA-FM is frozen throughout this stage and is not used for token-level targets, pair priors, or structural prediction.

## Stage 2: Supervised Pair-Relation Adaptation

Stage 2 adapts the pretrained encoder on supervised RNA secondary structure labels.

- encoder-only initialization
- pair-logit relation head
- weighted pair-relation BCE / pair loss
- lightweight 2D relation refinement
- optional token auxiliary objective when enabled by the config

The fine-tuning loss is:

```text
L_finetune = L_pair + optional token auxiliary
```

The pair objective is the main supervised structure signal. Token objectives remain auxiliary and do not replace pair-relation learning.

## Stage 3: Strict Nussinov Projection

Final structures are decoded from pair logits with strict Nussinov constraint projection.

- canonical pair constraints
- minimum loop length
- valid non-crossing secondary structure
- final dot-bracket output

This projection is part of the model system because it converts a dense pair-relation field into a legal RNA secondary structure.

## Dataset Roles

- Rfam: sequence-only pretraining and family-aware split experiments
- RNAcentral: large-scale sequence-only pretraining
- bpRNA: external benchmark and supervised data processing
- ArchiveII: historical benchmark and supervised candidate path

## Experimental Matrix

- baseline: supervised candidate from `config/candidate.yaml`
- D-only: sequence-only masked denoising pretraining
- D-RNAFM: masked denoising plus frozen RNA-FM sequence-level distillation
- seed repeat: repeat candidate and distillation paths across seeds
- external: bpRNA/Rfam external benchmark evaluation
- low-label: supervised fine-tuning with reduced label fractions
- scale-up: larger Rfam/RNAcentral sequence-only pretraining

## Excluded Components

- no language-model-based assistance component
- no language-model semantic conditioning
- no token-level distillation
- no pair prior
- no pseudo-labels
- no RNA-FM structural prior
- no language-model-based structure prediction

## Configuration Reference

Stage 1 sequence-only pretraining:

```bash
python main.py train --config config/seq_pretrain.yaml --device cuda
```

Stage 1 sequence-only pretraining with frozen RNA-FM distillation:

```bash
python main.py train --config config/seq_pretrain_rnafm.yaml --device cuda
```

Stage 2 supervised fine-tuning from sequence pretraining:

```bash
python main.py train --config config/candidate_from_seq_pretrain.yaml --device cuda
```

Stage 2 supervised fine-tuning from RNA-FM-distilled pretraining:

```bash
python main.py train --config config/candidate_from_rnafm_pretrain.yaml --device cuda
```

Canonical supervised baseline:

```bash
python main.py train --config config/candidate.yaml --device cuda
```
