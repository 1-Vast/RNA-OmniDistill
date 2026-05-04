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
from main import deep_update, load_config, train_model  # noqa: E402
from models.decoding import generate_structure_seq2struct  # noqa: E402
from utils.metrics import evaluate_structures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick real-data training smoke test.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--num_train", type=int, default=128)
    parser.add_argument("--num_val", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = Path(config["data"]["train_jsonl"])
    val_path = Path(config["data"]["val_jsonl"])
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            "train/val JSONL files are missing. Please run prepare_rna_dataset.py and make_splits.py first."
        )
    config = deep_update(
        config,
        {
            "training": {
                "output_dir": "outputs/realdata_smoke",
                "epochs": max(1, int(config["training"].get("epochs", 1))),
                "log_every": 1,
                "warmup_steps": 1,
            }
        },
    )
    result = train_model(config, max_steps=args.steps, train_subset=args.num_train)
    history = result["history"]
    initial_loss = history[0]["train_loss"] if history else 0.0
    final_loss = history[-1]["train_loss"] if history else 0.0

    val_dataset = RNAOmniDataset(val_path, max_length=int(config["data"]["max_length"]))
    samples = val_dataset.samples[: args.num_val]
    model = result["model"]
    tokenizer = result["tokenizer"]
    model.eval()
    device = next(model.parameters()).device
    pred_rows = []
    for sample in samples:
        pred_struct = generate_structure_seq2struct(model, tokenizer, sample["seq"], config["decoding"], device)
        pred_rows.append(
            {
                "id": sample["id"],
                "seq": sample["seq"],
                "true_struct": sample["struct"],
                "pred_struct": pred_struct,
                "family": sample.get("family", "OTHER"),
                "length": sample["length"],
            }
        )
    metrics = evaluate_structures(
        [row["pred_struct"] for row in pred_rows],
        [row["true_struct"] for row in pred_rows],
        [row["seq"] for row in pred_rows],
        allow_wobble=bool(config["decoding"].get("allow_wobble", True)),
    )
    metrics["initial_loss"] = initial_loss
    metrics["final_loss"] = final_loss
    metrics["loss_decreased"] = final_loss < initial_loss

    output_dir = Path("outputs/realdata_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pred_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in pred_rows[:5]:
            handle.write(json.dumps(row) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"loss_decreased={metrics['loss_decreased']} initial_loss={initial_loss:.4f} final_loss={final_loss:.4f}")
    for key in [
        "pair_precision",
        "pair_recall",
        "pair_f1",
        "valid_structure_rate",
        "all_dot_ratio",
        "avg_pred_pair_count",
        "avg_true_pair_count",
    ]:
        print(f"{key}={metrics.get(key, 0.0):.4f}")
    print("prediction examples:")
    for row in pred_rows[:5]:
        print(f"{row['id']} true={row['true_struct']} pred={row['pred_struct']}")
    print(f"saved examples -> {output_dir / 'pred_examples.jsonl'}")
    print(f"saved metrics -> {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
