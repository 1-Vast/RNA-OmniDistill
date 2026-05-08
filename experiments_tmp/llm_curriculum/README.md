# LLM-Guided Relational Curriculum Pilot

**Purpose**: Test whether a general-purpose LLM can act as a training-policy compiler for RNA-OmniDistill.

**Status**: Sandbox experiment. Not part of mainline.

## Difference from Failed LLM Semantic Tokens

| Aspect | Semantic Tokens (FAILED) | Curriculum Policy |
|--------|--------------------------|-------------------|
| Enters model input | Yes (segment_id=3) | No |
| Affects training | No | Yes |
| Result | -18.7pp Pair F1 | TBD |
| Risk of distribution shift | High | Low |

## What the LLM Does Here

The LLM acts as a **training-policy compiler**:
- Reads RNA family metadata (family, length, GC content, source)
- Produces constrained training-policy JSON
- Controls: mask policy (span/ratio), negative sampling, curriculum stage

The LLM does NOT:
- Enter model forward pass
- Produce structure labels or pair labels
- Generate semantic tokens as model input
- Participate in benchmark inference
- Modify evaluation metrics

**Core concept**: LLM tells the training system *how to learn*, not *what the structure is*.

## Usage

```bash
# Build rule policy
python experiments_tmp/llm_curriculum/build_policy.py \
  --input dataset/processed/rfam/rfam_fasta_seq_50k.jsonl \
  --output experiments_tmp/llm_curriculum/outputs/rule_policy.jsonl \
  --mode rule

# Build LLM policy (dry-run)
python experiments_tmp/llm_curriculum/build_policy.py \
  --input dataset/processed/rfam/rfam_fasta_seq_50k.jsonl \
  --output experiments_tmp/llm_curriculum/outputs/llm_policy.jsonl \
  --mode llm

# Apply policy to sequences
python experiments_tmp/llm_curriculum/apply_policy.py \
  --input dataset/processed/rfam/rfam_fasta_seq_50k.jsonl \
  --policy experiments_tmp/llm_curriculum/outputs/rule_policy.jsonl \
  --output dataset/processed_tmp/llm_curriculum/rfam_50k_policy_seq.jsonl

# Apply shuffled policy (control)
python experiments_tmp/llm_curriculum/apply_policy.py \
  --input dataset/processed/rfam/rfam_fasta_seq_50k.jsonl \
  --policy experiments_tmp/llm_curriculum/outputs/rule_policy.jsonl \
  --output dataset/processed_tmp/llm_curriculum/rfam_50k_policy_shuffled_seq.jsonl \
  --shuffle_policy
```

## Merge Criteria

See `notes/decision.md` for merge/delete decision rules.
