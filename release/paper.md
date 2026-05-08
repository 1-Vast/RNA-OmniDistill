# RNA-OmniDiffusion: 2026 Paper Framework

This document turns the current codebase into a paper-ready model framework. It is intentionally conservative: it separates supported claims from negative results and keeps the model bounded to RNA secondary structure prediction.

## Working Title

**RNA-OmniDiffusion: Pair-Refined Constraint-Guided Masked Diffusion for RNA Secondary Structure Prediction**

Short title: **RNA-OmniDiffusion**

## One-Sentence Claim

RNA-OmniDiffusion combines masked discrete diffusion, a trainable pair-logit head, lightweight 2D pair refinement, and strict Nussinov decoding to produce valid RNA secondary structures with stable gains over a historical pair-head baseline.

## Abstract Draft

RNA secondary structure prediction requires both local sequence evidence and global pairing constraints. We introduce RNA-OmniDiffusion, a compact masked discrete diffusion model that learns sequence, structure, and base-pair representations jointly while preserving strict non-crossing structural validity at inference. The model uses a bidirectional Transformer encoder with task-aware tokenization, an MLP pair head for base-pair logits, and a lightweight 2D convolutional residual refiner over the pair-logit map. Final structures are decoded with a strict Nussinov dynamic program rather than unconstrained token generation. On ArchiveII, the candidate model improves Pair F1 from 0.3846 to 0.5689 over the historical baseline while maintaining 100% valid structures. On a larger bpRNA external split, it reaches 0.5285 Pair F1 with only a 7.1% relative drop. Ablations show that the pair head and strict decoding are essential, while token-only decoding, greedy decoding as a final metric, conflict loss, and masking variants are not reliable contributors.

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

### 2. Pair-Logit Structural Head

The Transformer hidden states corresponding to sequence positions are projected into pair features:

```text
h_i = encoder(x)_i
s_ij = pair_head(h_i, h_j)
```

The current candidate uses an MLP-style left/right projection, distance bias, and a symmetric pair-logit matrix.

### 3. Lightweight 2D Pair Refinement

The key effective module is a shallow residual 2D convolution over pair logits:

```text
S' = S + Conv2d(GELU(Conv2d(S)))
S' = 0.5 * (S' + S'^T)
```

This is intentionally small. It gives the pair map local continuity without introducing a heavy graph network, triangle attention module, or RNA-FM dependency.

### 4. Strict Constraint-Guided Decoding

The paper result uses strict Nussinov decoding:

```text
score_ij = gamma * (logit(p_ij) - logit(threshold))
```

The dynamic program chooses a maximum-score non-crossing set of pairs under canonical/wobble and loop-length constraints. Greedy decoding is retained only as a pair-head probe and must not be used as the final paper metric.

## Training Objective

The main objective is:

```text
L = L_token + lambda_pair * L_pair
```

where:

- `L_token` is masked token cross entropy for sequence or structure targets
- `L_pair` is sampled BCE over positive pairs and sampled negative pairs
- pair logits are evaluated only in the true RNA L x L region
- padding and special tokens are excluded

Conflict loss was tested as a precision-oriented regularizer but should be reported as a negative result unless a future configuration reverses the current evidence.

## Main Contributions

1. **Pair-refined masked diffusion architecture**
   - A compact masked discrete diffusion model with explicit pair-logit prediction and 2D local refinement.

2. **Constraint-guided structure generation**
   - Strict Nussinov decoding is treated as part of the model system, not a post-hoc visualization step.

3. **Reproducible staged benchmark pipeline**
   - GPU staged logits plus multiprocessing strict decoding makes full strict benchmarks practical.

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
| norefine | 0.4966 | 0.4470 | 0.5630 | 0.4485 | 1.0000 | 1.3913 |
| candidate | 0.5689 | 0.5090 | 0.6517 | 0.5729 | 1.0000 | 1.3808 |

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
| candidate - norefine | +0.0723 | 2D pair refinement helps |
| candidate - oldbase | +0.1843 | full candidate improves over historical baseline |
| token-only decode | invalid | not a strict structural method |
| greedy decode | probe only | not a final metric |
| masking variants | inconclusive or harmful | not a main contribution |

## Paper Section Outline

### 1. Introduction

- RNA secondary structure requires global combinatorial validity.
- Pure token generation can learn bracket tokens but fails structural validity.
- Pair logits alone are useful but need topology-aware decoding.
- The paper proposes a compact pair-refined masked diffusion system with strict decoding.

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
3. Pair-logit head
4. 2D local pair refinement
5. Pair BCE and negative sampling
6. Strict Nussinov decoding
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

- Pair-refined pair logits improve strict secondary structure prediction.
- Strict Nussinov decoding is essential for valid structures.
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
3. Add calibration plots for pair logits and pair-count ratio.
4. Report strict Nussinov runtime separately from GPU forward time.
5. Keep negative results in an appendix to strengthen credibility.

## Recommended Final Abstract Position

Do not frame the system as a general RNA foundation model. Frame it as:

> a compact, reproducible, constraint-guided masked diffusion architecture for RNA secondary structure prediction, with a validated pair-refinement module and strict decoding.

That is a stronger and more defensible 2026 paper position than overstating model scale or task breadth.
