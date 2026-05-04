#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config/config.yaml}"
RAW_INPUT="${RAW_INPUT:-dataset/raw}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-20}"
SKIP_FAMILY_DISJOINT="${SKIP_FAMILY_DISJOINT:-false}"

run_step() {
  echo "==> $1"
  shift
  "$@"
}

mkdir -p outputs/protocol
RANDOM_CONFIG="outputs/protocol/random_config.yaml"
FAMILY_CONFIG="outputs/protocol/family_disjoint_config.yaml"

run_step "prepare data" python scripts/prepare_rna_dataset.py --input "$RAW_INPUT" --output dataset/processed/clean.jsonl --format auto --max_length "$MAX_LENGTH"
run_step "check data" python scripts/check_dataset.py --input dataset/processed/clean.jsonl --output dataset/processed/clean.checked.jsonl --max_length "$MAX_LENGTH"
run_step "make random splits" python scripts/make_splits.py --input dataset/processed/clean.checked.jsonl --out_dir dataset/processed --mode random
run_step "write random config" python -c "import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); cfg['training']['epochs']=int(sys.argv[3]); cfg['training']['output_dir']='outputs/random'; yaml.safe_dump(cfg, open(sys.argv[2],'w',encoding='utf-8'), sort_keys=False)" "$CONFIG" "$RANDOM_CONFIG" "$EPOCHS"
run_step "realdata smoke" python scripts/run_realdata_smoke.py --config "$RANDOM_CONFIG" --num_train 128 --num_val 32 --steps 100
run_step "train random" python main.py train --config "$RANDOM_CONFIG"
run_step "benchmark random" python scripts/run_benchmark.py --config "$RANDOM_CONFIG" --ckpt outputs/random/best.pt --split test

if [[ "$SKIP_FAMILY_DISJOINT" != "true" ]]; then
  run_step "make family-disjoint splits" python scripts/make_splits.py --input dataset/processed/clean.checked.jsonl --out_dir dataset/processed_family --mode family_disjoint
  run_step "write family-disjoint config" python -c "import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); cfg['data']['train_jsonl']='dataset/processed_family/train.jsonl'; cfg['data']['val_jsonl']='dataset/processed_family/val.jsonl'; cfg['data']['test_jsonl']='dataset/processed_family/test.jsonl'; cfg['training']['epochs']=int(sys.argv[3]); cfg['training']['output_dir']='outputs/family_disjoint'; yaml.safe_dump(cfg, open(sys.argv[2],'w',encoding='utf-8'), sort_keys=False)" "$CONFIG" "$FAMILY_CONFIG" "$EPOCHS"
  run_step "train family-disjoint" python main.py train --config "$FAMILY_CONFIG"
  run_step "benchmark family-disjoint" python scripts/run_benchmark.py --config "$FAMILY_CONFIG" --ckpt outputs/family_disjoint/best.pt --split test
  run_step "compare benchmarks" python scripts/compare_benchmarks.py --inputs outputs/random/benchmark_test.json outputs/family_disjoint/benchmark_test.json --names random family_disjoint --out outputs/benchmark_comparison
else
  echo "Skipping family-disjoint branch."
fi

echo "Protocol random branch complete."
