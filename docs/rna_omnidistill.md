# RNA-OmniDistill: Method Framework

RNA-OmniDistill is a two-stage framework for RNA secondary structure prediction. It combines frozen RNA-FM representation distillation with a trainable pair-logit model, lightweight 2D pair refinement, and strict Nussinov decoding.

---

## Two-Stage Architecture

### Stage 1: Sequence-only Teacher-guided Pretraining

**Goal**: Learn sequence representations from unlabeled RNA data via masked denoising, guided by a frozen RNA-FM teacher.

| Component | Detail |
|-----------|--------|
| Input | Unlabeled RNA sequences |
| Task | Masked nucleotide denoising |
| Teacher | Frozen RNA-FM (representation teacher only) |
| Teacher signal | Sequence-level mean-pooled embedding (one vector per sequence) |
| Student | Transformer encoder (to be used in Stage 2) |
| Loss | L_pretrain = L_denoise + lambda_distill * L_distill |

**Key properties**:
- Teacher is frozen and never updated.
- Distillation is sequence-level only (mean-pooled embedding), not token-level.
- Teacher does NOT predict structures, generate pseudo pair labels, or participate in inference.
- Student encoder weights are saved for Stage 2 initialization.

### Stage 2: Supervised Pairwise Structure Adaptation

**Goal**: Predict base-pair matrices from labeled RNA secondary structures using the pretrained encoder.

| Component | Detail |
|-----------|--------|
| Input | Labeled RNA secondary structures (dot-bracket) |
| Loading | Encoder-only from Stage 1 checkpoint |
| Task | Base-pair matrix prediction |
| Modules | Pair-logit head, lightweight 2D pair refinement, strict Nussinov decoding |
| Loss | L_finetune = L_token + lambda_pair * L_pair |

**Key properties**:
- L_pair is the primary objective; L_token is auxiliary.
- Pair-logit head produces base-pair logits from encoder hidden states.
- Lightweight 2D refinement (convolutional) smooths the pair-logit map.
- Strict Nussinov decoding enforces valid non-crossing structures.
- No teacher involvement during this stage.

---

## DeepSeek Agent (Read-Only Experiment Assistant)

The optional LLM agent is positioned as a **read-only experiment assistant**, not a structure predictor.

### What the Agent does:
- Experiment auditing and diagnosis
- Training schedule planning
- Paper report drafting from existing artifacts
- Data audit and comparison
- Dry-run prompt generation (no API call)

### What the Agent does NOT do:
- Participate in forward pass or training loss computation
- Execute benchmark inference
- Modify labels, predictions, or metrics
- Serve as a structure predictor
- Generate pseudo pair labels or structural priors

No agent output is used to claim model performance improvements.

---

## Configuration Reference

### Stage 1 Config: `config/seq_pretrain_rnafm.yaml`

Sequence-only RNA-FM teacher distillation pretraining.

```bash
python main.py train --config config/seq_pretrain_rnafm.yaml --device cuda --max_steps 20
```

### Stage 2 Config: `config/candidate_from_rnafm_pretrain.yaml`

Supervised fine-tune initialized from the distilled student encoder.

```bash
python main.py train --config config/candidate_from_rnafm_pretrain.yaml --device cuda --max_steps 20
```

### Baseline Config: `config/candidate.yaml`

Canonical supervised ArchiveII candidate (no distillation).

```bash
python main.py train --config config/candidate.yaml --device cuda
```

---

## Command Examples

### Offline Teacher Embedding Extraction

```bash
python scripts/extract_rnafm_embeddings.py \
  --input dataset/archive/train.jsonl \
  --output_jsonl dataset/unlabeled/train_seq_rnafm.jsonl \
  --output_npy dataset/teacher_emb/rnafm/train_embeddings.npy \
  --dummy --limit 256 --embedding_dim 640 --overwrite
```

### Stage 1: Distillation Pretrain

```bash
python main.py train --config config/seq_pretrain_rnafm.yaml --device cuda --max_steps 20
```

### Stage 2: Supervised Fine-tune

```bash
python main.py train --config config/candidate_from_rnafm_pretrain.yaml --device cuda --max_steps 20
```

### Benchmark

```bash
python scripts/eval.py bench \
  --config config/candidate_from_rnafm_pretrain.yaml \
  --ckpt outputs/candidate_from_rnafm_pretrain/best.pt \
  --split test --device cuda \
  --decode nussinov --stage_logits --workers 8 --chunksize 2 --profile
```

### Agent Diagnosis

```bash
python scripts/llm.py diagnose --run outputs/candidate --out outputs/llm/diagnose
```

---

## Claim Boundaries

### What RNA-OmniDistill IS:

- A two-stage framework: sequence-only teacher-guided pretraining + supervised pairwise structure adaptation.
- A trainable pair-logit model with lightweight 2D refinement and strict Nussinov decoding.
- An optional frozen RNA-FM representation teacher for sequence-level distillation only.
- An optional read-only LLM agent for experiment assistance.

### What RNA-OmniDistill IS NOT:

- NOT an LLM-powered structure predictor.
- NOT a token-level distillation method.
- NOT an RNA-FM structure prediction system.
- NOT a method where the agent improves model performance.
- NOT a method using RNA-FM structural priors or pseudo pair labels.
- NOT a foundation model trained on massive unlabeled RNA corpora.

### What NOT to claim:

- Do not claim LLM semantic conditioning as a positive result.
- Do not use greedy decode as the final paper metric.
- Do not use token-only decode as a strict structural result.
- Do not present masking as a main contribution on ArchiveII.
- Do not describe this as RNA 3D, ligand, protein, RNA-FM, or Agent work.
- Do not present pair-prior as a candidate-model contribution.
- Do not claim RNA-FM distillation as a stable result without seed repeats.
- Do not describe DeepSeek Agent as improving model performance.
- Do not claim token-level distillation (only sequence-level).
- Do not claim RNA-FM structural prior or pseudo pair labels.

---

## Preliminary Result

A single preliminary run on ArchiveII:
- Baseline candidate Pair F1 = 0.5849
- RNA-FM distilled pretraining (actual-2704) Pair F1 = 0.5959

**This is a preliminary single-run positive signal and should not be interpreted as a stable improvement** before seed repeats, D-only ablation, external benchmarks, and larger unlabeled pretraining.

Do not cite this as the main result. Do not claim statistical significance.

---

## Current Limitations

- Precision is still lower than recall, and pair ratio remains above 1.0.
- Family-disjoint generalization is not established as a final claim.
- The model is moderate-scale and from scratch; it is not a foundation model trained on massive unlabeled RNA corpora.
- Strict Nussinov is essential for valid structures, so decoding is part of the model system.
- RNA-FM distillation result is preliminary (single run, no seed repeats).
