#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config/config.yaml}"
RAW_INPUT="${RAW_INPUT:-dataset/raw}"
CLEAN="${CLEAN:-dataset/processed/clean.jsonl}"
CHECKED="${CHECKED:-dataset/processed/clean.checked.jsonl}"
MODE="${MODE:-random}"

python scripts/prepare_rna_dataset.py --input "$RAW_INPUT" --output "$CLEAN" --format auto --max_length 512
python scripts/check_dataset.py --input "$CLEAN" --output "$CHECKED" --max_length 512
python scripts/make_splits.py --input "$CHECKED" --out_dir dataset/processed --mode "$MODE"
python scripts/run_realdata_smoke.py --config "$CONFIG" --num_train 128 --num_val 32 --steps 100
python main.py train --config "$CONFIG"

