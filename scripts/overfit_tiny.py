from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.collator import RNAOmniCollator  # noqa: E402
from data.dataset import RNAOmniDataset  # noqa: E402
from data.tokenizer import RNAOmniTokenizer  # noqa: E402
from main import build_model, create_tiny_jsonl_dataset, load_config, move_batch_to_device, set_seed  # noqa: E402
from models.decoding import generate_structure_seq2struct  # noqa: E402
from models.rna_omnidiffusion import compute_omni_loss  # noqa: E402
from utils.metrics import evaluate_structures  # noqa: E402


def forward_model(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        segment_ids=batch["segment_ids"],
        task_ids=batch["task_ids"],
        time_steps=batch["time_steps"],
        seq_positions=batch["seq_positions"],
    )


def batch_token_accuracy(outputs: dict, batch: dict) -> float:
    labels = batch["labels"]
    mask = labels != -100
    if not mask.any():
        return 0.0
    preds = torch.argmax(outputs["token_logits"], dim=-1)
    return float((preds[mask] == labels[mask]).sum().detach().cpu()) / int(mask.sum().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny overfit sanity test.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config)
    config["training"]["amp"] = False
    config["training"]["batch_size"] = min(int(config["training"]["batch_size"]), max(1, args.num_samples))
    config["decoding"]["num_steps"] = min(int(config["decoding"].get("num_steps", 32)), 4)
    set_seed(int(config["training"].get("seed", 42)))

    train_path = Path(config["data"]["train_jsonl"])
    if not train_path.exists():
        print("train.jsonl not found; creating toy data with the smoke-test generator.")
        create_tiny_jsonl_dataset(config, overwrite=False)

    dataset = RNAOmniDataset(train_path, max_length=int(config["data"]["max_length"]))
    subset_size = min(args.num_samples, len(dataset))
    subset = Subset(dataset, list(range(subset_size)))
    tokenizer = RNAOmniTokenizer.from_samples([dataset.samples[idx] for idx in range(subset_size)])
    collator = RNAOmniCollator(
        tokenizer,
        config["tasks"],
        pair_negative_ratio=int(config["training"].get("pair_negative_ratio", 3)),
        seed=int(config["training"].get("seed", 42)),
    )
    loader = DataLoader(subset, batch_size=int(config["training"]["batch_size"]), shuffle=True, collate_fn=collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, tokenizer, device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=0.0)
    lambda_pair = float(config["training"].get("lambda_pair", 0.5))

    first_batch = move_batch_to_device(next(iter(loader)), device)
    model.eval()
    with torch.no_grad():
        initial_outputs = forward_model(model, first_batch)
        initial_loss_dict = compute_omni_loss(initial_outputs, first_batch, lambda_pair)
        initial_loss = float(initial_loss_dict["loss"].detach().cpu())

    model.train()
    final_loss = initial_loss
    final_token_loss = float(initial_loss_dict["token_loss"].detach().cpu())
    final_pair_loss = float(initial_loss_dict["pair_loss"].detach().cpu())
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_model(model, batch)
            loss_dict = compute_omni_loss(outputs, batch, lambda_pair)
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise SystemExit("Loss became non-finite during overfit_tiny.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
            optimizer.step()
            step += 1
            final_loss = float(loss.detach().cpu())
            final_token_loss = float(loss_dict["token_loss"].detach().cpu())
            final_pair_loss = float(loss_dict["pair_loss"].detach().cpu())
            if step == 1 or step % 20 == 0 or step == args.steps:
                print(
                    f"step={step} loss={final_loss:.4f} "
                    f"token_loss={final_token_loss:.4f} pair_loss={final_pair_loss:.4f}"
                )
            if step >= args.steps:
                break

    model.eval()
    eval_batch = move_batch_to_device(next(iter(loader)), device)
    with torch.no_grad():
        outputs = forward_model(model, eval_batch)
        token_acc = batch_token_accuracy(outputs, eval_batch)

    samples = [dataset.samples[idx] for idx in range(subset_size)]
    pred_structs = [
        generate_structure_seq2struct(model, tokenizer, sample["seq"], config["decoding"], device)
        for sample in samples
    ]
    metrics = evaluate_structures(
        pred_structs,
        [sample["struct"] for sample in samples],
        [sample["seq"] for sample in samples],
        allow_wobble=bool(config["decoding"].get("allow_wobble", True)),
    )

    print(f"initial_loss={initial_loss:.4f}")
    print(f"final_loss={final_loss:.4f}")
    print(f"token_accuracy={token_acc:.4f}")
    print(f"pair_f1={metrics['pair_f1']:.4f}")
    print(f"final_token_loss={final_token_loss:.4f}")
    print(f"final_pair_loss={final_pair_loss:.4f}")
    if math.isnan(final_token_loss) or math.isnan(final_pair_loss):
        raise SystemExit("NaN detected in overfit_tiny losses.")
    if final_loss >= initial_loss:
        print("Warning: loss did not decrease. Possible causes:")
        print("  1. labels are not aligned")
        print("  2. mask positions are not entering loss")
        print("  3. pair_labels do not match seq positions")
        print("  4. learning rate is too low")
        print("  5. pair loss weight is too high or too low")


if __name__ == "__main__":
    main()

