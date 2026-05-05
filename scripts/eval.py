from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import RNAOmniDataset
from main import build_model, load_checkpoint, load_config, resolve_device
from models.decode import generate_structure_seq2struct
from utils.metric import base_pair_f1, base_pair_precision, base_pair_recall, canonical_pair_ratio, evaluate_structures
from utils.struct import canonical_pair, parse_dot_bracket, pairs_to_dot_bracket, validate_structure


def random_valid(seq: str, loop: int = 3) -> str:
    rng = random.Random(42 + len(seq))
    candidates = [(i, j) for i in range(len(seq)) for j in range(i + loop, len(seq)) if canonical_pair(seq[i], seq[j])]
    rng.shuffle(candidates)
    pairs = []
    used = set()
    for i, j in candidates:
        if i in used or j in used:
            continue
        if any(i < a < j < b or a < i < b < j for a, b in pairs):
            continue
        pairs.append((i, j))
        used.update([i, j])
        if len(pairs) >= max(1, len(seq) // 12):
            break
    return pairs_to_dot_bracket(pairs, len(seq))


def row(method: str, sample: dict, pred: str) -> dict:
    try:
        pred_pairs = parse_dot_bracket(pred)
    except ValueError:
        pred_pairs = []
    return {
        "method": method,
        "id": sample.get("id", ""),
        "seq": sample["seq"],
        "true_struct": sample["struct"],
        "pred_struct": pred,
        "true_pairs": sample.get("pairs", []),
        "pred_pairs": pred_pairs,
        "pair_precision": base_pair_precision(pred, sample["struct"]),
        "pair_recall": base_pair_recall(pred, sample["struct"]),
        "pair_f1": base_pair_f1(pred, sample["struct"]),
        "valid": validate_structure(sample["seq"], pred),
        "canonical_pair_ratio": canonical_pair_ratio(sample["seq"], pred),
        "family": sample.get("family", "OTHER"),
        "length": sample["length"],
    }


def summarize(rows: list[dict]) -> dict:
    return evaluate_structures([r["pred_struct"] for r in rows], [r["true_struct"] for r in rows], [r["seq"] for r in rows]) if rows else {}


def run_bench(args: argparse.Namespace) -> None:
    user_config = load_config(args.config)
    split_path = Path(args.input) if args.input else Path(user_config["data"][f"{args.split}_jsonl"])
    if not split_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {split_path}")
    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    config["decoding"] = {**config.get("decoding", {}), **user_config.get("decoding", {})}
    dataset = RNAOmniDataset(split_path, max_length=int(user_config["data"]["max_length"]))
    if args.samples:
        dataset.samples = dataset.samples[: args.samples]
    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint is not compatible with the current model structure. "
            "The pair head was changed to an MLP; retrain the checkpoint or use a matching config."
        ) from exc
    model.eval()
    methods = {"model": [], "all": [], "random": []}
    for sample in dataset.samples:
        pred = generate_structure_seq2struct(model, tokenizer, sample["seq"], config["decoding"], device)
        methods["model"].append(row("model", sample, pred))
        methods["all"].append(row("all", sample, "." * sample["length"]))
        methods["random"].append(row("random", sample, random_valid(sample["seq"], int(config["decoding"].get("min_loop_length", 3)))))
    summary = {"overall": {name: summarize(rows) for name, rows in methods.items()}}
    out_dir = Path(args.out).parent if args.out else Path(config["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.out) if args.out else out_dir / "benchmark"
    bench_path = prefix if prefix.suffix == ".json" else prefix.with_suffix(".json")
    pred_path = out_dir / "predictions.jsonl"
    bench_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with pred_path.open("w", encoding="utf-8") as handle:
        for item in methods["model"]:
            handle.write(json.dumps(item) + "\n")
    print(json.dumps(summary["overall"], indent=2))
    print(f"benchmark -> {bench_path}")
    print(f"predictions -> {pred_path}")


def run_export(args: argparse.Namespace) -> None:
    args.split = "test"
    args.samples = args.samples
    args.out = args.out
    run_bench(args)


def run_analyze(args: argparse.Namespace) -> None:
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"Training log does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows and any("val_pair_f1" in row for row in rows):
        best = max(rows, key=lambda r: float(r.get("val_pair_f1", 0.0)))
    elif rows:
        best = min(rows, key=lambda r: float(r.get("val_loss", float("inf"))))
    else:
        best = {}
    result = {
        "epochs": len(rows),
        "best": best,
        "gap": float(best.get("gap", best.get("positive_pair_logit_mean", 0.0) - best.get("negative_pair_logit_mean", 0.0))),
        "rankAcc": best.get("rankAcc"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"analysis -> {out}")


def run_diagnose(args: argparse.Namespace) -> None:
    path = Path(args.pred)
    if not path.exists():
        raise SystemExit(f"Prediction file does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    worst = sorted(rows, key=lambda r: float(r.get("pair_f1", 0.0)))[:20]
    result = {"count": len(rows), "worst": worst, "allDot": sum(1 for r in rows if set(r.get("pred_struct", "")) <= {"."})}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"diagnosis -> {out}")


def run_compare(args: argparse.Namespace) -> None:
    rows = []
    for name, file in zip(args.names, args.inputs):
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        metrics = data.get("overall", {}).get("model", {})
        rows.append({"name": name, **metrics})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Variant | Pair F1 | All-dot | Valid | Pair Ratio |", "|---|---:|---:|---:|---:|"]
    for item in rows:
        ratio = float(item.get("avg_pred_pair_count", 0.0)) / max(1e-8, float(item.get("avg_true_pair_count", 0.0)))
        lines.append(f"| {item['name']} | {float(item.get('pair_f1', 0.0)):.4f} | {float(item.get('all_dot_ratio', 0.0)):.4f} | {float(item.get('valid_structure_rate', 0.0)):.4f} | {ratio:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"compare -> {out / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RNA-OmniDiffusion.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("bench")
    bench.add_argument("--config", default="config/archive.yaml")
    bench.add_argument("--ckpt", required=True)
    bench.add_argument("--split", default="test", choices=["train", "val", "test"])
    bench.add_argument("--input")
    bench.add_argument("--out")
    bench.add_argument("--samples", type=int)
    bench.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    bench.set_defaults(func=run_bench)
    export = sub.add_parser("export")
    export.add_argument("--config", default="config/archive.yaml")
    export.add_argument("--ckpt", required=True)
    export.add_argument("--input", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--samples", type=int)
    export.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    export.set_defaults(func=run_export)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--log", required=True)
    analyze.add_argument("--out", required=True)
    analyze.set_defaults(func=run_analyze)
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--pred", required=True)
    diagnose.add_argument("--out", required=True)
    diagnose.set_defaults(func=run_diagnose)
    compare = sub.add_parser("compare")
    compare.add_argument("--inputs", nargs="+", required=True)
    compare.add_argument("--names", nargs="+", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=run_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

