"""Candidate-structure generator for preference optimization.

Loads a checkpoint and generates multiple candidate dot-bracket
structures per RNA sequence by sweeping decode hyperparameters
(Nussinov gamma and pair-logit threshold).

Usage
-----
conda run -n DL python scripts/cand.py \
  --config config/candidate.yaml \
  --ckpt outputs/candidate/best.pt \
  --input dataset/archive/val.jsonl \
  --out outputs/pref/cand.jsonl \
  --limit 16 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from models.decode import generate_structure_seq2struct
from models.training import (
    build_model,
    load_checkpoint,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate RNA structures for preference optimization"
    )
    parser.add_argument("--config", required=True, help="YAML training config")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--input", required=True, help="JSONL file with RNA sequences")
    parser.add_argument("--out", required=True, help="Output candidate JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 4.0, 8.0],
        help="Nussinov gamma values to sweep",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.15, 0.25, 0.4, 0.6],
        help="Pair-logit thresholds to sweep",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=8,
        help="Max candidates per sample (random subset if sweep produces more)",
    )
    return parser.parse_args()


def load_samples(path: str, limit: int) -> list[dict]:
    samples: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def main() -> None:
    args = parse_args()

    if not Path(args.ckpt).exists():
        print(f"Error: checkpoint not found: {args.ckpt}", file=sys.stderr)
        raise SystemExit(1)

    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    except RuntimeError as exc:
        print(f"Error loading checkpoint: {exc}", file=sys.stderr)
        raise SystemExit(1)
    model.eval()

    # Merge user decode config via config path
    from models.training import load_config as _load_config
    user_config = _load_config(args.config)
    decode_cfg = {**config.get("decoding", {}), **user_config.get("decoding", {})}

    samples = load_samples(args.input, args.limit)
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(42)

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            seq = sample["seq"]
            candidates: list[dict] = []

            for gamma in args.gammas:
                for thresh in args.thresholds:
                    if len(candidates) >= args.max_candidates * 2:
                        break
                    local_cfg = dict(decode_cfg)
                    local_cfg["nussinov_gamma"] = gamma
                    local_cfg["pair_threshold"] = thresh
                    with torch.no_grad():
                        try:
                            struct = generate_structure_seq2struct(
                                model, tokenizer, seq, local_cfg, device
                            )
                        except Exception:
                            continue
                    if struct is None:
                        continue
                    from utils.reward import dotbracket_to_pairs, score_struct
                    pairs = dotbracket_to_pairs(struct)
                    feats = score_struct(seq, struct)
                    candidates.append({
                        "cid": f"c{len(candidates)}",
                        "struct": struct,
                        "pairs": [[int(i), int(j)] for i, j in pairs],
                        "features": feats,
                    })
                if len(candidates) >= args.max_candidates * 2:
                    break

            # Deduplicate by struct string
            seen = set()
            unique = []
            for c in candidates:
                if c["struct"] not in seen:
                    seen.add(c["struct"])
                    unique.append(c)

            # Sub-sample to max_candidates
            if len(unique) > args.max_candidates:
                unique = rng.sample(unique, args.max_candidates)

            # Re-index cid
            for idx, c in enumerate(unique):
                c["cid"] = f"c{idx}"

            entry = {
                "id": sample.get("id", ""),
                "seq": seq,
                "candidates": unique,
            }
            handle.write(json.dumps(entry) + "\n")

    print(f"Candidates written to {out_path}")
    print(f"  samples: {len(samples)}")
    print(f"  total candidates: {sum(1 for _ in out_path.open())}")


if __name__ == "__main__":
    main()
