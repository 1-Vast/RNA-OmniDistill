param(
  [string]$Config = "config/config.yaml",
  [string]$RawInput = "dataset/raw",
  [string]$Clean = "dataset/processed/clean.jsonl",
  [string]$Checked = "dataset/processed/clean.checked.jsonl",
  [string]$Mode = "random"
)

python scripts/prepare_rna_dataset.py --input $RawInput --output $Clean --format auto --max_length 512
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/check_dataset.py --input $Clean --output $Checked --max_length 512
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/make_splits.py --input $Checked --out_dir dataset/processed --mode $Mode
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/run_realdata_smoke.py --config $Config --num_train 128 --num_val 32 --steps 100
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python main.py train --config $Config

