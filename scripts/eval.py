from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import RNAOmniDataset
from main import build_model, load_checkpoint, load_config, resolve_device
from models.decode import (
    GREEDY_DECODE_WARNING,
    batched_greedy_decode_gpu,
    generate_structure_seq2struct,
    pairs_matrix_to_dotbracket_batch_with_stats,
)
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


def row(method: str, sample: dict, pred: str, sample_index: int | None = None) -> dict:
    try:
        pred_pairs = parse_dot_bracket(pred)
    except ValueError:
        pred_pairs = []
    return {
        "method": method,
        "sample_index": sample_index,
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
    if not rows:
        return {}
    return evaluate_structures(
        [item["pred_struct"] for item in rows],
        [item["true_struct"] for item in rows],
        [item["seq"] for item in rows],
    )


def build_seq2struct_batch(tokenizer, samples: list[dict], device: torch.device) -> dict:
    token_rows, segment_rows, seq_positions = [], [], []
    max_tokens = 0
    max_len = max(int(sample["length"]) for sample in samples)
    for sample in samples:
        tokens = [tokenizer.task_token("seq2struct"), "<SEQ>"]
        segments = [0, 1]
        seq_pos = []
        for base in sample["seq"]:
            seq_pos.append(len(tokens))
            tokens.append(base)
            segments.append(1)
        tokens += ["</SEQ>", "<STRUCT>"]
        segments += [1, 2]
        for _ in sample["seq"]:
            tokens.append("<MASK>")
            segments.append(2)
        tokens.append("</STRUCT>")
        segments.append(2)
        token_rows.append(tokenizer.encode(tokens))
        segment_rows.append(segments)
        seq_positions.append(seq_pos)
        max_tokens = max(max_tokens, len(tokens))

    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_tokens), tokenizer.pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    segment_ids = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    seq_pos_tensor = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    for idx, ids in enumerate(token_rows):
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, : len(ids)] = 1
        segment_ids[idx, : len(ids)] = torch.tensor(segment_rows[idx], dtype=torch.long, device=device)
        seq_pos_tensor[idx, : len(seq_positions[idx])] = torch.tensor(seq_positions[idx], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "task_ids": torch.full((batch_size,), tokenizer.task_to_id["seq2struct"], dtype=torch.long, device=device),
        "time_steps": torch.ones(batch_size, dtype=torch.float32, device=device),
        "seq_positions": seq_pos_tensor,
    }


def forward_pair_logits(model, tokenizer, samples: list[dict], device: torch.device) -> torch.Tensor:
    batch = build_seq2struct_batch(tokenizer, samples, device)
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            task_ids=batch["task_ids"],
            time_steps=batch["time_steps"],
            seq_positions=batch["seq_positions"],
        )
    pair_logits = outputs.get("pair_logits")
    if pair_logits is None:
        raise SystemExit("Model did not return pair_logits; greedy decode requires use_pair_head=true.")
    max_len = max(int(sample["length"]) for sample in samples)
    if pair_logits.ndim != 3 or pair_logits.size(1) < max_len or pair_logits.size(2) < max_len:
        raise SystemExit(f"Unexpected pair_logits shape {tuple(pair_logits.shape)} for max RNA length {max_len}.")
    return pair_logits[:, :max_len, :max_len]


def write_benchmark_csv(path: Path, summary: dict) -> None:
    fieldnames = [
        "method",
        "pair_precision",
        "pair_recall",
        "pair_f1",
        "valid_structure_rate",
        "canonical_pair_ratio",
        "all_dot_ratio",
        "avg_pred_pair_count",
        "avg_true_pair_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in summary.get("overall", {}).items():
            writer.writerow({"method": method, **{key: metrics.get(key, 0.0) for key in fieldnames if key != "method"}})


def run_bench(args: argparse.Namespace) -> None:
    start_time = time.time()
    user_config = load_config(args.config)
    split_path = Path(args.input) if args.input else Path(user_config["data"][f"{args.split}_jsonl"])
    if not split_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {split_path}")
    if not Path(args.ckpt).exists():
        raise SystemExit(
            "Checkpoint not found. Run: conda run -n DL python scripts\\run.py potential "
            "--config config/fixed.yaml --mode full --device cuda"
        )
    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    config["decoding"] = {**config.get("decoding", {}), **user_config.get("decoding", {})}
    dataset = RNAOmniDataset(split_path, max_length=int(user_config["data"]["max_length"]))
    limit = args.limit if args.limit is not None else args.samples
    if limit:
        dataset.samples = dataset.samples[: int(limit)]

    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint is not compatible with the current model structure. "
            "The pair head was changed to an MLP; retrain the checkpoint or use a matching config."
        ) from exc
    model.eval()

    decode_method = "greedy" if args.fast else args.decode
    batch_size = int(args.batch or config.get("training", {}).get("batch_size", 8))
    out_dir = Path(args.out).parent if args.out else Path(config["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.out) if args.out else out_dir / "benchmark"
    bench_path = prefix if prefix.suffix == ".json" else prefix.with_suffix(".json")
    pred_path = out_dir / "predictions.jsonl"
    tmp_path = out_dir / "predictions.tmp.jsonl"

    completed_indices = set()
    model_rows: list[dict] = []
    resumed = False
    meta_path = out_dir / "benchmeta.json"
    if args.resume and pred_path.exists() and meta_path.exists():
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("decode_method") != decode_method:
            print(
                f"Resume ignored because existing decode_method={old_meta.get('decode_method')} "
                f"does not match requested decode_method={decode_method}.",
                file=sys.stderr,
            )
        else:
            resumed = True
    elif args.resume and pred_path.exists():
        print("Resume ignored because benchmeta.json is missing.", file=sys.stderr)
    if resumed:
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if "sample_index" in item and item["sample_index"] is not None:
                    completed_indices.add(int(item["sample_index"]))
                    model_rows.append(item)
                else:
                    resumed = False
                    completed_indices.clear()
                    model_rows.clear()
                    print("Resume ignored because existing predictions do not contain sample_index.", file=sys.stderr)
                    break
    indexed_samples = list(enumerate(dataset.samples))
    remaining = [(idx, sample) for idx, sample in indexed_samples if idx not in completed_indices]

    forward_seconds = decode_seconds = metric_seconds = write_seconds = 0.0
    skipped_crossing_counts: list[int] = []
    handle = tmp_path.open("a" if resumed else "w", encoding="utf-8")
    if resumed:
        for item in model_rows:
            handle.write(json.dumps(item) + "\n")
    try:
        for start in range(0, len(remaining), batch_size):
            batch_items = remaining[start : start + batch_size]
            batch_indices = [idx for idx, _ in batch_items]
            batch_samples = [sample for _, sample in batch_items]
            if decode_method == "greedy":
                t0 = time.time()
                pair_logits = forward_pair_logits(model, tokenizer, batch_samples, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                forward_seconds += time.time() - t0
                t0 = time.time()
                pair_matrix = batched_greedy_decode_gpu(
                    pair_logits,
                    seqs=[sample["seq"] for sample in batch_samples],
                    min_loop_length=int(config["decoding"].get("min_loop_length", 3)),
                    pair_threshold=float(config["decoding"].get("pair_threshold", 0.25)),
                    allow_wobble=bool(config["decoding"].get("allow_wobble", True)),
                    canonical_only=True,
                    prevent_crossing=False,
                )
                structs, skipped = pairs_matrix_to_dotbracket_batch_with_stats(
                    pair_matrix,
                    [int(sample["length"]) for sample in batch_samples],
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                decode_seconds += time.time() - t0
                skipped_crossing_counts.extend(skipped)
            else:
                structs = []
                for sample in batch_samples:
                    t0 = time.time()
                    structs.append(generate_structure_seq2struct(model, tokenizer, sample["seq"], config["decoding"], device))
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    decode_seconds += time.time() - t0
                    skipped_crossing_counts.append(0)

            t0 = time.time()
            for sample_index, sample, pred in zip(batch_indices, batch_samples, structs):
                item = row("model", sample, pred, sample_index)
                model_rows.append(item)
                handle.write(json.dumps(item) + "\n")
            metric_seconds += time.time() - t0
            if len(model_rows) % max(1, int(args.save_every)) == 0:
                t0 = time.time()
                handle.flush()
                write_seconds += time.time() - t0
    finally:
        handle.close()
    tmp_path.replace(pred_path)

    all_rows = [row("all", sample, "." * sample["length"], idx) for idx, sample in indexed_samples]
    random_rows = [
        row("random", sample, random_valid(sample["seq"], int(config["decoding"].get("min_loop_length", 3))), idx)
        for idx, sample in indexed_samples
    ]
    total_seconds = time.time() - start_time
    summary = {
        "decode_method": decode_method,
        "partial": bool(limit),
        "samples": len(model_rows),
        "decode_warning": GREEDY_DECODE_WARNING if decode_method == "greedy" else "",
        "skipped_crossing_pairs_total": int(sum(skipped_crossing_counts)),
        "skipped_crossing_pairs_avg": float(sum(skipped_crossing_counts) / max(1, len(skipped_crossing_counts))),
        "crossing_sample_ratio": float(sum(1 for value in skipped_crossing_counts if value > 0) / max(1, len(skipped_crossing_counts))),
        "device": device.type,
        "cuda": bool(device.type == "cuda"),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "benchmark_seconds": total_seconds,
        "samples_per_sec": len(model_rows) / max(1e-8, total_seconds),
        "overall": {"model": summarize(model_rows), "all": summarize(all_rows), "random": summarize(random_rows)},
    }
    benchmeta = {
        "config": args.config,
        "ckpt": args.ckpt,
        "split": args.split,
        "input_jsonl": str(split_path),
        "decode_method": decode_method,
        "total_samples": len(dataset.samples),
        "completed_samples": len(model_rows),
        "partial": bool(limit),
        "resumed": resumed,
        "device": summary["device"],
        "cuda": summary["cuda"],
        "gpu": summary["gpu"],
        "batch": batch_size,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
    }
    batch_count = max(1, (len(remaining) + batch_size - 1) // batch_size)
    benchtime = {
        "total_seconds": total_seconds,
        "forward_seconds": forward_seconds,
        "decode_seconds": decode_seconds,
        "metric_seconds": metric_seconds,
        "write_seconds": write_seconds,
        "samples_per_sec": summary["samples_per_sec"],
        "avg_forward_seconds_per_batch": forward_seconds / batch_count,
        "avg_decode_seconds_per_batch": decode_seconds / batch_count,
        "length_bucket_speed": {},
    }
    bench_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_benchmark_csv(bench_path.with_suffix(".csv"), summary)
    (out_dir / "benchmeta.json").write_text(json.dumps(benchmeta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "benchtime.json").write_text(json.dumps(benchtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"benchmark -> {bench_path}")
    print(f"predictions -> {pred_path}")
    if args.profile:
        print(json.dumps(benchtime, indent=2))


def run_export(args: argparse.Namespace) -> None:
    args.split = "test"
    args.limit = args.samples
    args.decode = "greedy"
    args.fast = True
    args.profile = False
    args.resume = False
    args.save_every = 100
    args.batch = None
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
        keys = sorted({k for item in rows for k in item})
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
    bench.add_argument("--limit", type=int)
    bench.add_argument("--decode", choices=["nussinov", "greedy"], default="nussinov")
    bench.add_argument("--batch", type=int)
    bench.add_argument("--fast", action="store_true")
    bench.add_argument("--profile", action="store_true")
    bench.add_argument("--resume", action="store_true")
    bench.add_argument("--save_every", type=int, default=100)
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
