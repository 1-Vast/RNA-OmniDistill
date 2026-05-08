# RNA-OmniDistill: 2026 Paper Framework

This document turns the current codebase into a paper-ready model framework. It is intentionally conservative: it separates supported claims from negative results and keeps the model bounded to RNA secondary structure prediction.

## Working Title

**RNA-OmniDistill: Relational Masked Diffusion with Frozen RNA-FM Distillation for Constraint-Guided RNA Folding**

Short title: **RNA-OmniDistill**

## One-Sentence Claim

RNA-OmniDistill is a relation-aware masked diffusion framework that jointly denoises sequence, structure, and pair-relation tokens, with frozen RNA-FM sequence-level distillation for global representation prior and strict Nussinov constraint projection for valid structure inference.

## Abstract Draft

RNA folding requires both local nucleotide evidence and global base-pair constraints. We introduce RNA-OmniDistill, a relational masked diffusion framework that jointly denoises sequence tokens, structure tokens, and pair-relation tokens. A frozen RNA-FM model provides sequence-level representation distillation through mean-pooled embeddings, serving as a global prior without predicting structures or generating pseudo pair labels. The pair-relation field is refined by a lightweight 2D convolutional residual module, and a strict Nussinov constraint projection produces valid dot-bracket structures. On ArchiveII, the candidate achieves Pair F1 of 0.5762 (baseline), with RNAcentral 50k pretraining reaching 0.6171. On a larger bpRNA external split, it reaches 0.5285 Pair F1. Ablations confirm that token-only decode fails validity, greedy decode is useful only as a relation-head probe, conflict loss is not a positive contributor, and masking variants are not a main contribution.

## Three Core Contributions

### Contribution 1: Relational Masked Diffusion for RNA Folding
RNA secondary structure prediction is cast as joint discrete denoising over nucleotide tokens,
structure tokens, and pair-relation tokens. Unlike methods that only denoise one-dimensional
sequence or structure tokens, this framework explicitly models pairwise relational variables
through a pair-relation field, enabling the model to directly learn RNA base-pair interactions
as part of the diffusion process.

### Contribution 2: Frozen RNA-FM Sequence-Level Distillation
A frozen RNA-FM model provides sequence-level mean-pooled embedding vectors as a teacher
signal during unsupervised pretraining. The teacher is used only for global representation
guidance: it never predicts structures, generates pseudo pair labels, performs token-level
distillation, or participates in benchmark inference. The student encoder learns compact
sequence representations that transfer to supervised pair-relation prediction.

### Contribution 3: Constraint-Guided Relation Projection
After the model predicts a soft pair-relation field, a strict Nussinov dynamic programming
algorithm projects the field into a valid dot-bracket structure satisfying non-crossing,
minimum loop length, canonical/wobble pairing, and one-base-one-pair constraints.
This projection is a validated inference path, not a post-hoc correction step.

## Problem Definition

Given an RNA sequence

```text
x = (x_1, ..., x_L), x_i in {A, U, G, C, N}
```

predict a secondary structure

```text
y = (y_1, ..., y_L), y_i in {., (, )}
```

or equivalently a symmetric pair matrix

```text
P in {0, 1}^{L x L}
```

subject to:

- one nucleotide pairs with at most one other nucleotide
- minimum loop length constraint
- canonical or wobble pairing constraint
- no pseudoknot under the strict benchmark

The primary metric is base-pair F1. Valid structure rate is a hard safety metric, not an auxiliary display metric.

## Model Framework

### 1. Unified Masked Diffusion Backbone

The model receives task, sequence, structure, motif, and family tokens in a single token sequence. It uses segment ids to distinguish:

| Segment | Meaning |
|---:|---|
| 0 | task and global condition |
| 1 | sequence |
| 2 | structure |
| 3 | motif and family condition |

Training follows mask-based discrete diffusion:

```text
t ~ Uniform(0, 1)
m = mask_schedule(t)
z_t = corrupt(z_0, m)
model(z_t, t) -> z_0
```

Loss is computed on masked target tokens and sampled pair labels.

### 2. Pair-Relation Head

The Transformer hidden states corresponding to sequence positions are projected into a pair-relation field:

```text
h_i = encoder(x)_i
s_ij = relation_head(h_i, h_j)
```

The current candidate uses an MLP-style left/right projection, distance bias, and a symmetric pair-relation logit matrix. This pair-relation field captures pairwise base-pairing interactions as part of the joint diffusion process.

### 3. Lightweight 2D Relation Refinement

The key effective module is a shallow residual 2D convolution over the pair-relation logits:

```text
S' = S + Conv2d(GELU(Conv2d(S)))
S' = 0.5 * (S' + S'^T)
```

This is intentionally small. It gives the pair-relation map local continuity without introducing a heavy graph network, triangle attention module, or RNA-FM dependency.

### 4. Strict Constraint-Guided Relation Projection

The paper result uses strict Nussinov constraint projection:

```text
score_ij = gamma * (logit(p_ij) - logit(threshold))
```

The dynamic program projects the soft pair-relation field into a valid dot-bracket structure under canonical/wobble and loop-length constraints. Greedy decoding is retained only as a relation-head probe and must not be used as the final paper metric.

## Training Objective

The main objective is:

```text
L = L_token + lambda_pair * L_pair
```

where:

- `L_token` is masked token cross entropy for sequence or structure targets
- `L_pair` is sampled BCE over positive pairs and sampled negative pairs on the pair-relation field
- pair-relation logits are evaluated only in the true RNA L x L region
- padding and special tokens are excluded

Conflict loss was tested as a precision-oriented regularizer but should be reported as a negative result unless a future configuration reverses the current evidence.

## Main Contributions

1. **Relational masked diffusion with joint token denoising**
   - A masked discrete diffusion framework that jointly denoises sequence tokens, structure tokens, and pair-relation tokens, explicitly modeling pairwise RNA interactions.

2. **Frozen RNA-FM sequence-level distillation**
   - A frozen RNA-FM teacher provides mean-pooled global representation guidance during unsupervised pretraining, without structure prediction or pseudo-label generation.

3. **Constraint-guided relation projection**
   - Strict Nussinov dynamic programming projects the soft pair-relation field into a valid dot-bracket structure as a validated inference path.

4. **Clear negative-result boundaries**
   - Token-only decode fails validity.
   - Greedy decode is useful only as a probe.
   - Masking variants are not a main contribution on ArchiveII.
   - Conflict loss and language-model-based semantic conditioning are not positive model contributions in the current evidence.

## Experimental Program

### Table 1: Main In-Domain Result

Dataset: ArchiveII.

| Method | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio |
|---|---:|---:|---:|---:|---:|---:|
| oldbase | 0.3846 | 0.3398 | 0.4465 | 0.3864 | 1.0000 | 1.4213 |
| candidate | 0.5762 | 0.5324 | 0.6302 | 0.5801 | 1.0000 | 1.3808 |
| RNAcentral 50k D-RNAFM | 0.6171 | 0.5794 | 0.6640 | 0.6215 | 1.0000 | 1.3752 |

### Table 2: External Generalization

Dataset: bpRNA external random split.

| Method | Pair F1 | Precision | Recall | MCC | Valid | Pair Ratio | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| oldbase | 0.4234 | 0.4019 | 0.4741 | 0.4335 | 1.0000 | 1.40 | 12,732 |
| norefine | 0.4399 | 0.4083 | 0.5037 | 0.4451 | 1.0000 | 1.42 | 12,732 |
| candidate | 0.5285 | 0.4877 | 0.6070 | 0.5344 | 1.0000 | 1.38 | 12,732 |

### Table 3: Seed Stability

ArchiveII, candidate config.

| Seed | Pair F1 | Precision | Recall | Valid | Pair Ratio |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.5749 | 0.5145 | 0.6585 | 1.0000 | 1.3805 |
| 43 | 0.5900 | 0.5326 | 0.6685 | 1.0000 | 1.3663 |
| 44 | 0.5789 | 0.5195 | 0.6612 | 1.0000 | 1.3846 |
| Mean +/- Std | 0.5813 +/- 0.0078 | - | - | - | - |

### Table 4: Essential Components

Use this table to communicate what is essential and what is not.

| Comparison | Delta F1 | Interpretation |
|---|---:|---|
| candidate - oldbase | +0.1916 | full candidate improves over historical baseline |
| RNAcentral 50k - candidate | +0.0409 | RNA-FM distillation pretraining helps |
| token-only decode | invalid | not a strict structural method |
| greedy decode | probe only | not a final metric |
| masking variants | inconclusive or harmful | not a main contribution |

## Paper Section Outline

### 1. Introduction

- RNA secondary structure requires global combinatorial validity.
- Pure token generation can learn bracket tokens but fails structural validity.
- Pair-relation logits alone are useful but need topology-aware decoding.
- The paper proposes a compact relation-aware masked diffusion system with strict constraint projection.

### 2. Related Work

Keep this section factual and restrained:

- classical dynamic programming and thermodynamic methods
- deep learning pair-matrix methods
- masked discrete diffusion
- constrained decoding for structured prediction
- RNA foundation models only as related context, not as a dependency

### 3. Method

Recommended subsections:

1. Unified RNA tokenization and task formatting
2. Masked discrete diffusion objective
3. Pair-relation head
4. 2D local relation refinement
5. Pair BCE and negative sampling
6. Strict Nussinov constraint projection
7. Complexity and implementation

### 4. Experiments

Recommended subsections:

1. Datasets and splits
2. Metrics
3. Main ArchiveII result
4. External bpRNA generalization
5. Component analysis
6. Negative results and failure modes
7. Runtime and benchmark implementation

### 5. Limitations

State these directly:

- The model is not a large pretrained RNA foundation model.
- Family-disjoint generalization is not yet a final supported claim.
- Validity depends on strict decoding.
- Precision still leaves room for improvement.
- No pseudoknot modeling in the strict default.

## Claim Boundaries

Supported:

- Pair-relation field with 2D refinement improves strict secondary structure prediction.
- Strict Nussinov constraint projection is essential for valid structures.
- The staged benchmark pipeline makes strict evaluation practical.
- External bpRNA random-split transfer remains reasonable.

Not supported:

- Language-model-based semantic conditioning improves the model.
- Token-only decoding is structurally valid.
- Greedy decoding is a final benchmark.
- Masking variants are a main contribution.
- The model solves RNA 3D, ligand, protein, or pseudoknot tasks.

## Next Paper-Quality Upgrades

These are the highest-value next steps before submission:

1. Add a family-disjoint or homology-reduced split if data quality permits.
2. Compare against a small set of classical or public RNA secondary-structure baselines.
3. Add calibration plots for pair-relation logits and pair-count ratio.
4. Report strict Nussinov runtime separately from GPU forward time.
5. Keep negative results in an appendix to strengthen credibility.

## Recommended Final Abstract Position

Do not frame the system as a general RNA foundation model. Frame it as:

> a compact, reproducible, relation-aware masked diffusion architecture for RNA secondary structure prediction, with frozen RNA-FM sequence-level distillation and strict constraint-guided relation projection.

That is a stronger and more defensible 2026 paper position than overstating model scale or task breadth.
