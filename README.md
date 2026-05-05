# RNA-OmniDiffusion

Minimal masked discrete diffusion for RNA sequence, structure, motif, and family-conditioned modeling.

This repository intentionally stays small. It does not use RNA-FM, LoRA, external LLM calls, RNA 3D, ligand tasks, or protein tasks.

## Structure

```text
main.py
config/
  base.yaml
  archive.yaml
  orig.yaml
  relax.yaml
  fix.yaml
  fixed.yaml
  mild.yaml
  strict.yaml
  stable.yaml
  cpu.yaml
  ablate/
data/
  dataset.py
  collator.py
  token.py
models/
  omni.py
  mask.py
  decode.py
utils/
  struct.py
  metric.py
scripts/
  data.py
  probe.py
  audit.py
  eval.py
  run.py
```

## Data

JSONL records use:

```json
{"id":"RNA1","seq":"AUGGCU","struct":"((..))","family":"OTHER","motifs":[],"pairs":[[0,5],[1,4]],"length":6}
```

Prepare ArchiveII:

```powershell
python scripts/data.py fetch --set archive --out dataset/raw/archive
python scripts/data.py prep --input dataset/raw/archive --output dataset/processed/archive.jsonl --format auto --maxlen 512
python scripts/data.py check --input dataset/processed/archive.jsonl --output dataset/processed/archivecheck.jsonl --maxlen 512
python scripts/data.py split --input dataset/processed/archivecheck.jsonl --out dataset/archive --mode random
```

## Train

```powershell
python main.py smoke
python main.py train --config config/archive.yaml --device cuda
python main.py eval --config config/archive.yaml --ckpt outputs/archive/best.pt --device cuda
python main.py infer --config config/archive.yaml --ckpt outputs/archive/best.pt --task seq2struct --seq AUGGCUACGU --device cuda
```

## Audit

```powershell
python scripts/audit.py names --out outputs/name
python scripts/audit.py align --config config/archive.yaml --batches 3 --device cuda --out outputs/align
python scripts/audit.py profile --config config/archive.yaml --steps 10 --device cuda --out outputs/profile
```

## Evaluate

```powershell
python scripts/eval.py bench --config config/archive.yaml --ckpt outputs/archive/best.pt --split test --device cuda
python scripts/eval.py export --config config/archive.yaml --ckpt outputs/archive/best.pt --input dataset/archive/test.jsonl --out outputs/archive/pred.jsonl --device cuda
python scripts/eval.py analyze --log outputs/archive/trainlog.jsonl --out outputs/archive/analysis.json
python scripts/eval.py diagnose --pred outputs/archive/predictions.jsonl --out outputs/archive/diagnosis.json
python scripts/eval.py compare --inputs outputs/archive/benchmark.json outputs/pairfix/benchmark.json --names archive pairfix --out outputs/compare
```

## Experiments

Quick sweep, for direction only:

```powershell
python scripts/run.py sweep --mode quick --device cuda
```

Full sweep:

```powershell
python scripts/run.py sweep --mode full --device cuda
```

ArchiveII full decision workflow:

```powershell
python scripts/run.py potential --config config/fixed.yaml --mode full --device cuda
```

Configuration semantics:

- `config/orig.yaml`: original training and original decoding. This is the strict baseline.
- `config/relax.yaml`: original training with relaxed Nussinov decoding only.
- `config/fix.yaml`: pairfix training with original decoding.
- `config/fixed.yaml`: pairfix training with relaxed decoding. This is the current main full-run candidate.
- `config/archive.yaml`: legacy alias kept for compatibility; do not describe it as the strict baseline in reports.
- `config/mild.yaml`, `config/strict.yaml`, and `config/stable.yaml`: intervention templates generated for post-full diagnosis. They are not run automatically.

After `fixed` full completes, inspect:

```powershell
type outputs\fixed\full.md
```

Run core ablation only if `full.md` says:

```text
Decision: Run core ablation next.
```

Then use:

```powershell
conda run -n DL python scripts\run.py ablate --config config/fixed.yaml --only full nopair nonuss random --device cuda
```

Agent integration remains deferred. These scripts only train, evaluate, diagnose, and recommend safe next configurations.

## Current Limitations

- No pseudoknot decoding by default.
- No RNA 3D, ligand, or protein tasks.
- Pair head diagnostics are still required before treating benchmark gains as reliable structure learning.
- Quick runs are pipeline checks and are not paper conclusions.
