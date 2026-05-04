from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.collator import RNAOmniCollator
from data.dataset import RNAOmniDataset
from data.tokenizer import RNAOmniTokenizer
from models.decoding import generate_sequence_invfold, generate_structure_seq2struct
from models.rna_omnidiffusion import RNAOmniDiffusion, compute_omni_loss
from utils.metrics import base_pair_f1, token_accuracy


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def synthetic_samples() -> list[dict]:
    return [
        {
            "id": "toy_0001",
            "seq": "GCAUAGC",
            "struct": "((...))",
            "family": "miRNA",
            "motifs": [{"type": "STEM", "start": 0, "end": 6}, {"type": "HAIRPIN", "start": 2, "end": 4}],
        },
        {
            "id": "toy_0002",
            "seq": "GGGAAACCC",
            "struct": "(((...)))",
            "family": "riboswitch",
            "motifs": [{"type": "STEM", "start": 0, "end": 8}, {"type": "HAIRPIN", "start": 3, "end": 5}],
        },
        {
            "id": "toy_0003",
            "seq": "AUGCAU",
            "struct": "(....)",
            "family": "tRNA",
            "motifs": [{"type": "HAIRPIN", "start": 1, "end": 4}],
        },
        {
            "id": "toy_0004",
            "seq": "GCUAAGC",
            "struct": "((...))",
            "family": "miRNA",
        },
    ]


def create_tiny_jsonl_dataset(config: Dict[str, Any], overwrite: bool = False) -> None:
    samples = synthetic_samples()
    data_cfg = config["data"]
    splits = {
        "train_jsonl": samples,
        "val_jsonl": samples[:2],
        "test_jsonl": samples[2:],
    }
    for key, rows in splits.items():
        path = Path(data_cfg[key])
        if path.exists() and not overwrite:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def ensure_dataset_paths(config: Dict[str, Any], create_if_missing: bool) -> None:
    required = [Path(config["data"]["train_jsonl"]), Path(config["data"]["val_jsonl"])]
    if all(path.exists() for path in required):
        return
    if create_if_missing:
        print("Dataset JSONL files not found; creating a tiny synthetic dataset under dataset/processed/.")
        create_tiny_jsonl_dataset(config, overwrite=False)
        return
    missing = ", ".join(str(path) for path in required if not path.exists())
    raise FileNotFoundError(f"Missing required dataset file(s): {missing}")


def build_datasets_and_tokenizer(config: Dict[str, Any]) -> tuple[RNAOmniDataset, RNAOmniDataset, RNAOmniTokenizer]:
    max_length = int(config["data"]["max_length"])
    train_dataset = RNAOmniDataset(config["data"]["train_jsonl"], max_length=max_length)
    val_dataset = RNAOmniDataset(config["data"]["val_jsonl"], max_length=max_length)
    tokenizer = RNAOmniTokenizer.from_samples(train_dataset.samples + val_dataset.samples)
    return train_dataset, val_dataset, tokenizer


def build_model(config: Dict[str, Any], tokenizer: RNAOmniTokenizer, device: torch.device) -> RNAOmniDiffusion:
    model_cfg = config["model"]
    model = RNAOmniDiffusion(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        max_position_embeddings=int(model_cfg["max_position_embeddings"]),
        num_tasks=len(tokenizer.task_to_id),
    )
    return model.to(device)


def make_loader(dataset: RNAOmniDataset, tokenizer: RNAOmniTokenizer, config: Dict[str, Any], shuffle: bool) -> DataLoader:
    training_cfg = config["training"]
    collator = RNAOmniCollator(
        tokenizer=tokenizer,
        task_ratios=config["tasks"],
        pair_negative_ratio=int(training_cfg.get("pair_negative_ratio", 3)),
        seed=int(training_cfg.get("seed", 42)),
    )
    return DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=shuffle,
        collate_fn=collator,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def save_checkpoint(
    path: Path,
    model: RNAOmniDiffusion,
    tokenizer: RNAOmniTokenizer,
    config: Dict[str, Any],
    epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[Dict[str, Any], RNAOmniTokenizer, dict]:
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    tokenizer = RNAOmniTokenizer.from_dict(checkpoint["tokenizer"])
    return config, tokenizer, checkpoint


def evaluate_model(
    model: RNAOmniDiffusion,
    loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                segment_ids=batch["segment_ids"],
                task_ids=batch["task_ids"],
                time_steps=batch["time_steps"],
                seq_positions=batch["seq_positions"],
            )
            loss_dict = compute_omni_loss(outputs, batch, float(config["training"].get("lambda_pair", 0.5)))
            losses.append(float(loss_dict["loss"].detach().cpu()))
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    return {"loss": sum(losses) / max(1, len(losses))}


def train_model(config: Dict[str, Any], max_steps: int | None = None) -> dict:
    set_seed(int(config["training"].get("seed", 42)))
    ensure_dataset_paths(config, create_if_missing=True)
    train_dataset, val_dataset, tokenizer = build_datasets_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, tokenizer, device)
    train_loader = make_loader(train_dataset, tokenizer, config, shuffle=True)
    val_loader = make_loader(val_dataset, tokenizer, config, shuffle=False)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    total_steps = max(1, int(config["training"]["epochs"]) * max(1, len(train_loader)))
    warmup_steps = int(config["training"].get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(config["training"]["output_dir"])
    best_path = output_dir / "best.pt"
    best_val = math.inf
    history = []
    global_step = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    segment_ids=batch["segment_ids"],
                    task_ids=batch["task_ids"],
                    time_steps=batch["time_steps"],
                    seq_positions=batch["seq_positions"],
                )
                loss_dict = compute_omni_loss(
                    outputs,
                    batch,
                    lambda_pair=float(config["training"].get("lambda_pair", 0.5)),
                )
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            record = {
                "step": global_step,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "token_loss": float(loss_dict["token_loss"].cpu()),
                "pair_loss": float(loss_dict["pair_loss"].cpu()),
            }
            history.append(record)
            print(
                f"step={global_step} epoch={epoch} "
                f"loss={record['loss']:.4f} token={record['token_loss']:.4f} pair={record['pair_loss']:.4f}"
            )
            if max_steps is not None and global_step >= max_steps:
                break

        val_metrics = evaluate_model(model, val_loader, config, device, max_batches=2 if max_steps else None)
        print(f"epoch={epoch} val_loss={val_metrics['loss']:.4f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(best_path, model, tokenizer, config, epoch, best_val)
        if max_steps is not None and global_step >= max_steps:
            break

    return {"model": model, "tokenizer": tokenizer, "config": config, "best_path": best_path, "history": history}


def run_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = train_model(config)
    print(f"Best checkpoint: {result['best_path']}")


def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    config = load_config(args.config)
    config = deep_update(ckpt_config, config)
    val_dataset = RNAOmniDataset(config["data"]["val_jsonl"], max_length=int(config["data"]["max_length"]))
    model = build_model(config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    loader = make_loader(val_dataset, tokenizer, config, shuffle=False)
    metrics = evaluate_model(model, loader, config, device)
    print(f"eval_loss={metrics['loss']:.4f}")


def run_infer(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    model = build_model(config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if args.task == "seq2struct":
        if not args.seq:
            raise ValueError("--seq is required for seq2struct inference.")
        struct = generate_structure_seq2struct(model, tokenizer, args.seq, config["decoding"], device)
        print(struct)
    elif args.task == "invfold":
        if not args.struct:
            raise ValueError("--struct is required for invfold inference.")
        seq = generate_sequence_invfold(model, tokenizer, args.struct, config["decoding"], device)
        print(seq)
    else:
        raise ValueError("Minimal CLI inference supports --task seq2struct or --task invfold.")


def run_smoke(args: argparse.Namespace) -> None:
    config = load_config("config/config.yaml")
    config = deep_update(
        config,
        {
            "data": {"max_length": 64},
            "model": {
                "hidden_size": 64,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "max_position_embeddings": 256,
            },
            "training": {
                "batch_size": 2,
                "epochs": 1,
                "lr": 0.001,
                "warmup_steps": 1,
                "amp": False,
                "output_dir": "outputs/smoke",
                "seed": 42,
            },
            "decoding": {"num_steps": 4, "use_nussinov": True},
        },
    )
    create_tiny_jsonl_dataset(config, overwrite=False)
    result = train_model(config, max_steps=2)
    model = result["model"]
    tokenizer = result["tokenizer"]
    model.eval()
    device = next(model.parameters()).device
    seq = synthetic_samples()[0]["seq"]
    pred_struct = generate_structure_seq2struct(model, tokenizer, seq, config["decoding"], device)
    true_struct = synthetic_samples()[0]["struct"]
    print(f"smoke_losses={[round(item['loss'], 4) for item in result['history']]}")
    print(f"seq={seq}")
    print(f"pred_struct={pred_struct}")
    print(f"bp_f1_vs_toy={base_pair_f1(pred_struct, true_struct):.4f}")
    print("smoke_ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RNA-OmniDiffusion-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="config/config.yaml")
    train.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", default="config/config.yaml")
    eval_parser.add_argument("--ckpt", required=True)
    eval_parser.set_defaults(func=run_eval)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--config", default="config/config.yaml")
    infer.add_argument("--ckpt", required=True)
    infer.add_argument("--task", required=True, choices=["seq2struct", "invfold"])
    infer.add_argument("--seq")
    infer.add_argument("--struct")
    infer.set_defaults(func=run_infer)

    smoke = subparsers.add_parser("smoke")
    smoke.set_defaults(func=run_smoke)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
