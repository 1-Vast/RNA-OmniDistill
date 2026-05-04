param(
  [string]$Config = "config/config.yaml",
  [string]$RawInput = "dataset/raw",
  [int]$MaxLength = 512,
  [int]$Epochs = 20,
  [switch]$SkipFamilyDisjoint
)

$ErrorActionPreference = "Stop"

function Run-Step {
  param([string]$Name, [scriptblock]$Command)
  Write-Host "==> $Name"
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name"
  }
}

New-Item -ItemType Directory -Force -Path outputs/protocol | Out-Null
$RandomConfig = "outputs/protocol/random_config.yaml"
$FamilyConfig = "outputs/protocol/family_disjoint_config.yaml"

Run-Step "prepare data" {
  python scripts/prepare_rna_dataset.py --input $RawInput --output dataset/processed/clean.jsonl --format auto --max_length $MaxLength
}

Run-Step "check data" {
  python scripts/check_dataset.py --input dataset/processed/clean.jsonl --output dataset/processed/clean.checked.jsonl --max_length $MaxLength
}

Run-Step "make random splits" {
  python scripts/make_splits.py --input dataset/processed/clean.checked.jsonl --out_dir dataset/processed --mode random
}

Run-Step "write random config" {
  python -c "import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); cfg['training']['epochs']=int(sys.argv[3]); cfg['training']['output_dir']='outputs/random'; yaml.safe_dump(cfg, open(sys.argv[2],'w',encoding='utf-8'), sort_keys=False)" $Config $RandomConfig $Epochs
}

Run-Step "realdata smoke" {
  python scripts/run_realdata_smoke.py --config $RandomConfig --num_train 128 --num_val 32 --steps 100
}

Run-Step "train random" {
  python main.py train --config $RandomConfig
}

Run-Step "benchmark random" {
  python scripts/run_benchmark.py --config $RandomConfig --ckpt outputs/random/best.pt --split test
}

if (-not $SkipFamilyDisjoint) {
  Run-Step "make family-disjoint splits" {
    python scripts/make_splits.py --input dataset/processed/clean.checked.jsonl --out_dir dataset/processed_family --mode family_disjoint
  }
  Run-Step "write family-disjoint config" {
    python -c "import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); cfg['data']['train_jsonl']='dataset/processed_family/train.jsonl'; cfg['data']['val_jsonl']='dataset/processed_family/val.jsonl'; cfg['data']['test_jsonl']='dataset/processed_family/test.jsonl'; cfg['training']['epochs']=int(sys.argv[3]); cfg['training']['output_dir']='outputs/family_disjoint'; yaml.safe_dump(cfg, open(sys.argv[2],'w',encoding='utf-8'), sort_keys=False)" $Config $FamilyConfig $Epochs
  }
  Run-Step "train family-disjoint" {
    python main.py train --config $FamilyConfig
  }
  Run-Step "benchmark family-disjoint" {
    python scripts/run_benchmark.py --config $FamilyConfig --ckpt outputs/family_disjoint/best.pt --split test
  }
  Run-Step "compare benchmarks" {
    python scripts/compare_benchmarks.py --inputs outputs/random/benchmark_test.json outputs/family_disjoint/benchmark_test.json --names random family_disjoint --out outputs/benchmark_comparison
  }
} else {
  Write-Host "Skipping family-disjoint branch."
}

Write-Host "Protocol random branch complete."
