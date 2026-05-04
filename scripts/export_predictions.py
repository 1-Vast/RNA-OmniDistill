from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import RNAOmniDataset  # noqa: E402
from main import build_model, load_checkpoint, load_config  # noqa: E402
from models.decoding import _build_inference_batch, _forward_model, generate_structure_seq2struct  # noqa: E402
from utils.metrics import base_pair_f1, base_pair_precision, base_pair_recall, canonical_pair_ratio  # noqa: E402
from utils.structure import parse_dot_bracket, validate_structure  # noqa: E402


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


@torch.no_grad()
def topk_pair_probs(model, tokenizer, seq: str, struct: str, config: dict, device, topk: int) -> list[dict]:
    batch, _, _ = _build_inference_batch(tokenizer, "seq2struct", seq, struct, device=device)
    outputs = _forward_model(model, batch)
    pair_logits = outputs.get("pair_logits")
    if pair_logits is None or len(seq) < 2:
        return []
    probs = torch.sigmoid(pair_logits[0, : len(seq), : len(seq)]).detach().cpu()
    rows = []
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            rows.append((float(probs[i, j]), i, j))
    rows.sort(reverse=True)
    return [{"i": i, "j": j, "prob": prob} for prob, i, j in rows[:topk]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export seq2struct predictions to JSONL.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--save_pair_probs", default="false")
    parser.add_argument("--topk_pairs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt_config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}")
    user_config = load_config(args.config)
    ckpt_config["decoding"] = {**ckpt_config.get("decoding", {}), **user_config.get("decoding", {})}
    dataset_path = Path(args.input)
    if not dataset_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {dataset_path}")
    dataset = RNAOmniDataset(dataset_path, max_length=int(user_config["data"]["max_length"]))
    samples = dataset.samples[: args.num_samples] if args.num_samples else dataset.samples
    model = build_model(ckpt_config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            pred_struct = generate_structure_seq2struct(model, tokenizer, sample["seq"], ckpt_config["decoding"], device)
            try:
                pred_pairs = parse_dot_bracket(pred_struct)
            except ValueError:
                pred_pairs = []
            row = {
                "id": sample["id"],
                "seq": sample["seq"],
                "true_struct": sample["struct"],
                "pred_struct": pred_struct,
                "true_pairs": sample.get("pairs", []),
                "pred_pairs": pred_pairs,
                "pair_precision": base_pair_precision(pred_struct, sample["struct"]),
                "pair_recall": base_pair_recall(pred_struct, sample["struct"]),
                "pair_f1": base_pair_f1(pred_struct, sample["struct"]),
                "valid": validate_structure(sample["seq"], pred_struct, bool(ckpt_config["decoding"].get("allow_wobble", True))),
                "canonical_pair_ratio": canonical_pair_ratio(sample["seq"], pred_struct),
                "family": sample.get("family", "OTHER"),
                "length": sample["length"],
            }
            if str_to_bool(args.save_pair_probs):
                row["pair_prob_topk"] = topk_pair_probs(
                    model, tokenizer, sample["seq"], pred_struct, ckpt_config, device, args.topk_pairs
                )
            handle.write(json.dumps(row) + "\n")
    print(f"predictions saved to {output_path}")


if __name__ == "__main__":
    main()
