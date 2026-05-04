# RNA-OmniDiffusion-v2

Minimal, runnable masked discrete diffusion for unified RNA sequence, secondary-structure, motif, and family modeling.

The first target is a smoke-testable training pipeline, not SOTA performance. This version uses a small Transformer encoder trained from scratch and does not include RNA-FM, LoRA, semantic cache, external LLM calls, ligand/protein tasks, or RNA 3D tasks.

See `INDEX.md` for a file-by-file repository guide.

## Data Format

Use JSONL files under `dataset/processed/`. Each line should contain:

```json
{
  "id": "RNA_000001",
  "seq": "GGGAAACCC",
  "struct": "(((...)))",
  "family": "miRNA",
  "motifs": [{"type": "STEM", "start": 0, "end": 8}],
  "pairs": [[0, 8], [1, 7], [2, 6]],
  "length": 9
}
```

`motifs` and `pairs` are optional. If absent, pairs are parsed from dot-bracket structure and simple motifs are inferred. Sequence and structure lengths must match; malformed samples are skipped with a clear warning.

## Commands

Smoke test:

```bash
python main.py smoke
```

Train:

```bash
python main.py train --config config/config.yaml
```

Evaluate:

```bash
python main.py eval --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt
```

Seq2struct inference:

```bash
python main.py infer --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt --task seq2struct --seq AUGGCUACGU
```

Inverse folding inference:

```bash
python main.py infer --config config/config.yaml --ckpt outputs/rna_omnidiffusion_v2/best.pt --task invfold --struct "((...))"
```

## Tasks

The collator samples one task per sample:

- `seq2struct`: sequence visible, structure tokens masked.
- `invfold`: structure visible, sequence tokens masked.
- `inpaint`: random or motif span masked across sequence and structure, with pair-aware expansion.
- `motif_control`: task, motif, and family tokens visible; sequence and structure masked.

Training uses mask-based discrete diffusion: sample `t` in `[0, 1]`, derive a mask ratio, replace selected target tokens with `<MASK>`, and predict clean tokens only at masked positions. Pair loss is binary cross entropy over positive pairs plus sampled negatives.

## Current Limitations

- No pseudoknot decoding by default.
- No RNA 3D, ligand, or protein tasks.
- Minimal Transformer trained from scratch.
- Nussinov decoding is intentionally simple and CPU-friendly.
