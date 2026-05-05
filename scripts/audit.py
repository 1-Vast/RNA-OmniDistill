from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.collator import RNAOmniCollator
from data.dataset import RNAOmniDataset
from data.token import RNAOmniTokenizer
from main import build_model, load_config, loss_from_batch, move_batch_to_device, resolve_device
from models.omni import _pair_loss_mask
from utils.struct import parse_dot_bracket


def ensure_archive_paths(config: dict) -> None:
    data = config["data"]
    target = Path(data["train_jsonl"]).parent
    fallback = Path("dataset/processed_archiveii")
    if Path(data["train_jsonl"]).exists():
        return
    if fallback.exists():
        target.mkdir(parents=True, exist_ok=True)
        for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            src = fallback / name
            if src.exists():
                (target / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return
    raise SystemExit("ArchiveII split files are missing. Run `python scripts/data.py split --input dataset/processed/archivecheck.jsonl --out dataset/archive --mode random` first.")


def make_batch(config: dict, batches: int) -> tuple[dict, RNAOmniTokenizer]:
    ensure_archive_paths(config)
    dataset = RNAOmniDataset(config["data"]["train_jsonl"], max_length=int(config["data"]["max_length"]))
    samples = dataset.samples[: max(1, int(config["training"].get("batch_size", 8)) * batches)]
    tokenizer = RNAOmniTokenizer.from_samples(samples)
    collator = RNAOmniCollator(
        tokenizer,
        config["tasks"],
        pair_negative_ratio=int(config["training"].get("pair_negative_ratio", config["training"].get("pairRatio", 3))),
        seed=int(config["training"].get("seed", 42)),
        ablation=config.get("ablation", {}),
    )
    loader = DataLoader(samples, batch_size=int(config["training"].get("batch_size", 8)), collate_fn=collator)
    return next(iter(loader)), tokenizer


def run_align(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    batch, tokenizer = make_batch(config, args.batches)
    device = resolve_device(args.device)
    model = build_model(config, tokenizer, device)
    batch_device = move_batch_to_device(batch, device)
    outputs = model(
        input_ids=batch_device["input_ids"],
        attention_mask=batch_device["attention_mask"],
        segment_ids=batch_device["segment_ids"],
        task_ids=batch_device["task_ids"],
        time_steps=batch_device["time_steps"],
        seq_positions=batch_device["seq_positions"],
    )
    loss = loss_from_batch(outputs, batch_device, {
        "lambda_pair": float(config["training"].get("lambda_pair", config["training"].get("lambdaPair", 0.5))),
        "lambda_seq": float(config["training"].get("lambda_seq", 1.0)),
        "lambda_struct": float(config["training"].get("lambda_struct", 1.0)),
        "token_id_weights": None,
        "pair_pos_weight": config["training"].get("pair_positive_weight", config["training"].get("pairWeight", "auto")),
        "use_pair_loss": True,
        "pair_options": {
            "pairWeight": config["training"].get("pair_positive_weight", config["training"].get("pairWeight", "auto")),
            "pairRatio": int(config["training"].get("pair_negative_ratio", config["training"].get("pairRatio", 3))),
            "pairUpper": bool(config["training"].get("pairUpper", True)),
            "pairLoop": int(config["training"].get("pairLoop", 3)),
            "pairDiag": bool(config["training"].get("pairDiag", False)),
            "pairFloat": bool(config["training"].get("pairFloat", True)),
            "sampleNegOnGpu": bool(config["training"].get("sampleNegOnGpu", True)),
        },
    })
    pair_mask = _pair_loss_mask(
        batch_device["pair_mask"],
        batch_device["lengths"],
        {
            "pairUpper": bool(config["training"].get("pairUpper", True)),
            "pairLoop": int(config["training"].get("pairLoop", 3)),
            "pairDiag": bool(config["training"].get("pairDiag", False)),
        },
    ).detach().cpu()
    rows = []
    fail = []
    for i, seq in enumerate(batch["raw_seq"]):
        length = int(batch["lengths"][i])
        seq_pos = batch["seq_positions"][i, :length]
        struct_pos = batch["struct_positions"][i, :length]
        seq_ok = bool((batch["segment_ids"][i, seq_pos] == 1).all())
        struct_ok = bool((batch["segment_ids"][i, struct_pos] == 2).all())
        parsed = set(tuple(sorted(pair)) for pair in parse_dot_bracket(batch["raw_struct"][i]))
        labels = set()
        for a, b in torch.nonzero(batch["pair_labels"][i] > 0.5, as_tuple=False).tolist():
            labels.add((a, b))
        label_ok = parsed == labels
        lower_polluted = bool(torch.tril(pair_mask[i], diagonal=0).any())
        row = {
            "id": i,
            "length": length,
            "seqPositions": int((seq_pos >= 0).sum()),
            "structPositions": int((struct_pos >= 0).sum()),
            "seqSegmentOk": seq_ok,
            "structSegmentOk": struct_ok,
            "labelOk": label_ok,
            "lossMaskLowerPolluted": lower_polluted,
            "pos": float(loss.get("pos", torch.tensor(0.0))),
            "neg": float(loss.get("neg", torch.tensor(0.0))),
        }
        rows.append(row)
        if not seq_ok:
            fail.append("sequence positions enter non-sequence segment")
        if not struct_ok:
            fail.append("structure positions enter non-structure segment")
        if not label_ok:
            fail.append("pair labels differ from dot-bracket parse")
        if lower_polluted:
            fail.append("pair loss mask includes lower triangle or diagonal")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS" if not fail else "FAIL",
        "failures": sorted(set(fail)),
        "pos": float(loss.get("pos", torch.tensor(0.0))),
        "neg": float(loss.get("neg", torch.tensor(0.0))),
        "gap": float(loss.get("gap", torch.tensor(0.0))),
        "rankAcc": None if loss.get("rankAcc") is None else float(loss["rankAcc"]),
        "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "batch.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    lines = [
        "# Alignment Audit",
        "",
        f"Status: **{report['status']}**",
        f"Device: {report['device']} {report['gpu']}",
        f"Positive pairs: {report['pos']:.0f}",
        f"Negative pairs: {report['neg']:.0f}",
        f"Gap: {report['gap']:.4f}",
        f"RankAcc: {report['rankAcc']}",
        "",
        "## Checks",
        f"- position offset: {'FAIL' if any('positions' in item for item in fail) else 'PASS'}",
        f"- label offset: {'FAIL' if any('labels' in item for item in fail) else 'PASS'}",
        f"- mask pollution: {'FAIL' if any('mask' in item for item in fail) else 'PASS'}",
        f"- continue full training: {'yes' if not fail else 'no'}",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"align {report['status']} -> {out / 'report.md'}")


def run_profile(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    batch, tokenizer = make_batch(config, 1)
    device = resolve_device(args.device)
    model = build_model(config, tokenizer, device)
    batch = move_batch_to_device(batch, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []
    for _ in range(args.steps):
        start = time.time()
        opt.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            task_ids=batch["task_ids"],
            time_steps=batch["time_steps"],
            seq_positions=batch["seq_positions"],
        )
        forward = time.time()
        loss = torch.nn.functional.cross_entropy(outputs["token_logits"].view(-1, outputs["token_logits"].size(-1)), batch["labels"].view(-1), ignore_index=-100)
        loss.backward()
        opt.step()
        end = time.time()
        times.append({"forward": forward - start, "step": end - start})
    avg_forward = sum(t["forward"] for t in times) / max(1, len(times))
    avg_step = sum(t["step"] for t in times) / max(1, len(times))
    report = {
        "device": device.type,
        "cuda": device.type == "cuda",
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "batch": int(batch["input_ids"].size(0)),
        "maxlen": int(config["data"]["max_length"]),
        "hidden": int(config["model"]["hidden_size"]),
        "layers": int(config["model"]["num_layers"]),
        "dataTime": 0.0,
        "forwardTime": avg_forward,
        "backwardTime": max(0.0, avg_step - avg_forward),
        "stepTime": avg_step,
        "decodeTime": 0.0,
        "epochTime": 0.0,
        "fullTime": avg_step * args.steps,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text("# Runtime Profile\n\n" + "\n".join(f"- {k}: {v}" for k, v in report.items()) + "\n", encoding="utf-8")
    print(f"profile -> {out / 'report.md'}")


def run_names(args: argparse.Namespace) -> None:
    roots = [Path("main.py"), Path("config"), Path("data"), Path("models"), Path("utils"), Path("scripts"), Path("README.md"), Path("INDEX.md")]
    bad = []
    for root in roots:
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            if path.name == "__init__.py":
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            stem = path.stem
            if "_" in stem:
                bad.append(str(path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = "PASS" if not bad else "FAIL"
    report = ["# Name Audit", "", f"Status: **{status}**", ""]
    if bad:
        report.extend(["## Files", *[f"- {item}" for item in bad]])
    else:
        report.append("No project-defined filenames with underscores were found.")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"status": status, "bad": bad}, indent=2) + "\n", encoding="utf-8")
    print(f"names {status} -> {out / 'report.md'}")
    if bad:
        raise SystemExit("Name audit failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RNA-OmniDiffusion runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    align = sub.add_parser("align")
    align.add_argument("--config", default="config/archive.yaml")
    align.add_argument("--batches", type=int, default=1)
    align.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    align.add_argument("--out", default="outputs/align")
    align.set_defaults(func=run_align)
    profile = sub.add_parser("profile")
    profile.add_argument("--config", default="config/archive.yaml")
    profile.add_argument("--steps", type=int, default=2)
    profile.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    profile.add_argument("--out", default="outputs/profile")
    profile.set_defaults(func=run_profile)
    names = sub.add_parser("names")
    names.add_argument("--out", default="outputs/name")
    names.set_defaults(func=run_names)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

