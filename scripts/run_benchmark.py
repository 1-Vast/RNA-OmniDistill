from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import RNAOmniDataset  # noqa: E402
from main import build_model, load_checkpoint, load_config  # noqa: E402
from models.decoding import generate_structure_seq2struct  # noqa: E402
from utils.metrics import (  # noqa: E402
    base_pair_f1,
    base_pair_precision,
    base_pair_recall,
    canonical_pair_ratio,
    evaluate_structures,
)
from utils.structure import canonical_pair, parse_dot_bracket, pairs_to_dot_bracket, validate_structure  # noqa: E402


def length_bucket(length: int) -> str:
    if length <= 100:
        return "0-100"
    if length <= 200:
        return "100-200"
    if length <= 300:
        return "200-300"
    return "300-512"


def pair_count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 15:
        return "6-15"
    return "16+"


def noncrossing_with(pair: tuple[int, int], pairs: list[tuple[int, int]]) -> bool:
    i, j = pair
    for k, l in pairs:
        if i in (k, l) or j in (k, l):
            return False
        if i < k < j < l or k < i < l < j:
            return False
    return True


def random_valid_pair_baseline(seq: str, min_loop_length: int, allow_wobble: bool, seed: int = 42) -> str:
    rng = random.Random(seed + len(seq))
    candidates = []
    for i in range(len(seq)):
        for j in range(i + min_loop_length, len(seq)):
            if canonical_pair(seq[i], seq[j], allow_wobble):
                candidates.append((i, j))
    rng.shuffle(candidates)
    pairs = []
    target = max(1, len(seq) // 12) if candidates else 0
    for pair in candidates:
        if len(pairs) >= target:
            break
        if noncrossing_with(pair, pairs):
            pairs.append(pair)
    return pairs_to_dot_bracket(pairs, len(seq))


def rnafold_predict(seq: str, rnafold_bin: str) -> str | None:
    if shutil.which(rnafold_bin) is None:
        return None
    try:
        proc = subprocess.run(
            [rnafold_bin, "--noPS"],
            input=seq + "\n",
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    return lines[1].split()[0]


def prediction_row(method: str, sample: dict, pred_struct: str) -> dict:
    try:
        pred_pairs = parse_dot_bracket(pred_struct)
    except ValueError:
        pred_pairs = []
    return {
        "method": method,
        "id": sample["id"],
        "seq": sample["seq"],
        "true_struct": sample["struct"],
        "pred_struct": pred_struct,
        "true_pairs": sample.get("pairs", []),
        "pred_pairs": pred_pairs,
        "pair_precision": base_pair_precision(pred_struct, sample["struct"]),
        "pair_recall": base_pair_recall(pred_struct, sample["struct"]),
        "pair_f1": base_pair_f1(pred_struct, sample["struct"]),
        "valid": validate_structure(sample["seq"], pred_struct),
        "canonical_pair_ratio": canonical_pair_ratio(sample["seq"], pred_struct),
        "family": sample.get("family", "OTHER"),
        "length": sample["length"],
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    seqs = [row["seq"] for row in rows]
    preds = [row["pred_struct"] for row in rows]
    trues = [row["true_struct"] for row in rows]
    return evaluate_structures(preds, trues, seqs)


def grouped_metrics(rows: list[dict], key_fn) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return {str(key): summarize(group_rows) for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0]))}


def write_csv(path: Path, summary: dict) -> None:
    rows = []
    for method, values in summary["overall"].items():
        row = {"group": "overall", "name": "all", "method": method}
        row.update(values)
        rows.append(row)
    for section in ["by_length", "by_family", "by_pair_count"]:
        for method, groups in summary[section].items():
            for name, values in groups.items():
                row = {"group": section, "name": name, "method": method}
                row.update(values)
                rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark seq2struct predictions on a split.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--input_jsonl")
    parser.add_argument("--output_prefix")
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    user_config = load_config(args.config)
    split_path = Path(args.input_jsonl) if args.input_jsonl else Path(user_config["data"][f"{args.split}_jsonl"])
    if not split_path.exists():
        label = "input_jsonl" if args.input_jsonl else args.split
        raise SystemExit(f"{label} JSONL does not exist: {split_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt_config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}")
    ckpt_config["decoding"] = {**ckpt_config.get("decoding", {}), **user_config.get("decoding", {})}
    dataset = RNAOmniDataset(split_path, max_length=int(user_config["data"]["max_length"]))
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: max(0, args.max_samples)]
    model = build_model(ckpt_config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    method_rows: dict[str, list[dict]] = defaultdict(list)
    for sample in dataset.samples:
        pred = generate_structure_seq2struct(model, tokenizer, sample["seq"], ckpt_config["decoding"], device)
        method_rows["model"].append(prediction_row("model", sample, pred))
        if user_config.get("baselines", {}).get("run_all_dot", True):
            method_rows["all_dot"].append(prediction_row("all_dot", sample, "." * sample["length"]))
        if user_config.get("baselines", {}).get("run_random_pair", True):
            rand_struct = random_valid_pair_baseline(
                sample["seq"],
                int(ckpt_config["decoding"].get("min_loop_length", 3)),
                bool(ckpt_config["decoding"].get("allow_wobble", True)),
            )
            method_rows["random_pair"].append(prediction_row("random_pair", sample, rand_struct))

    if user_config.get("baselines", {}).get("run_rnafold", False):
        rnafold_bin = user_config.get("baselines", {}).get("rnafold_bin", "RNAfold")
        if shutil.which(rnafold_bin) is None:
            print(f"Warning: RNAfold binary not found ({rnafold_bin}); skipping RNAfold baseline.")
        else:
            for sample in dataset.samples:
                pred = rnafold_predict(sample["seq"], rnafold_bin)
                if pred is not None:
                    method_rows["rnafold"].append(prediction_row("rnafold", sample, pred))

    if args.output_prefix:
        output_prefix = Path(args.output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        predictions_path = output_prefix.with_name(output_prefix.name + "_predictions.jsonl")
        json_path = output_prefix.with_name(output_prefix.name + "_benchmark.json")
        csv_path = output_prefix.with_name(output_prefix.name + "_benchmark.csv")
    else:
        output_dir = Path(ckpt_config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / f"predictions_{args.split}.jsonl"
        json_path = output_dir / f"benchmark_{args.split}.json"
        csv_path = output_dir / f"benchmark_{args.split}.csv"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in method_rows["model"]:
            handle.write(json.dumps(row) + "\n")

    top_families = {family for family, _ in sorted(
        defaultdict(int, {sample.get("family", "OTHER"): 0 for sample in dataset.samples}).items()
    )}
    summary = {"overall": {}, "by_length": {}, "by_family": {}, "by_pair_count": {}}
    family_counts = defaultdict(int)
    for sample in dataset.samples:
        family_counts[sample.get("family", "OTHER")] += 1
    top20 = {family for family, _ in sorted(family_counts.items(), key=lambda item: item[1], reverse=True)[:20]}
    for method, rows in method_rows.items():
        summary["overall"][method] = summarize(rows)
        summary["by_length"][method] = grouped_metrics(rows, lambda row: length_bucket(row["length"]))
        summary["by_family"][method] = grouped_metrics(
            [row for row in rows if row["family"] in top20],
            lambda row: row["family"],
        )
        summary["by_pair_count"][method] = grouped_metrics(
            rows,
            lambda row: pair_count_bucket(len(row["true_pairs"])),
        )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(csv_path, summary)
    print(json.dumps(summary["overall"], indent=2))
    print(f"benchmark json -> {json_path}")
    print(f"benchmark csv -> {csv_path}")
    print(f"model predictions -> {predictions_path}")


if __name__ == "__main__":
    main()
