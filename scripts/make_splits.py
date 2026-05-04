from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    train_count = min(train_count, total)
    val_count = min(val_count, total - train_count)
    return train_count, val_count


def random_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    train_count, val_count = split_counts(len(rows), train_ratio, val_ratio)
    train = rows[:train_count]
    val = rows[train_count : train_count + val_count]
    test = rows[train_count + val_count :]
    return train, val, test


def family_disjoint_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    by_family = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family", "OTHER") or "OTHER")].append(row)
    families = list(by_family)
    if len(families) < 3:
        print("Warning: fewer than 3 families; falling back to random split.")
        return None

    rng = random.Random(seed)
    rng.shuffle(families)
    family_count = len(families)
    val_family_count = max(1, int(round(family_count * val_ratio))) if val_ratio > 0 else 0
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    test_family_count = max(1, int(round(family_count * test_ratio))) if test_ratio > 0 else 0
    if val_family_count + test_family_count >= family_count:
        val_family_count = 1 if val_ratio > 0 else 0
        test_family_count = 1 if test_ratio > 0 else 0
    train_family_count = family_count - val_family_count - test_family_count
    if train_family_count <= 0:
        print("Warning: family ratios leave no train families; falling back to random split.")
        return None

    train_families = set(families[:train_family_count])
    val_families = set(families[train_family_count : train_family_count + val_family_count])
    test_families = set(families[train_family_count + val_family_count :])
    buckets = {"train": [], "val": [], "test": []}

    for family in families:
        family_rows = by_family[family]
        if family in train_families:
            bucket = "train"
        elif family in val_families:
            bucket = "val"
        else:
            bucket = "test"
        buckets[bucket].extend(family_rows)

    for name in buckets:
        rng.shuffle(buckets[name])
    return buckets["train"], buckets["val"], buckets["test"]


def check_family_disjoint(train: list[dict], val: list[dict], test: list[dict]) -> None:
    train_f = {row.get("family", "OTHER") for row in train}
    val_f = {row.get("family", "OTHER") for row in val}
    test_f = {row.get("family", "OTHER") for row in test}
    assert train_f.isdisjoint(val_f)
    assert train_f.isdisjoint(test_f)
    assert val_f.isdisjoint(test_f)


def length_report(rows: list[dict[str, Any]]) -> dict:
    lengths = [int(row.get("length", len(row.get("seq", "")))) for row in rows]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(lengths, dtype=np.float32)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def split_report(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    splits = {"train": train, "val": val, "test": test}
    families = {name: {row.get("family", "OTHER") for row in rows} for name, rows in splits.items()}
    seqs = {name: {row.get("seq", "") for row in rows} for name, rows in splits.items()}
    report = {"splits": {}, "overlap": {}}
    for name, rows in splits.items():
        report["splits"][name] = {
            "samples": len(rows),
            "families": len(families[name]),
            "length_distribution": length_report(rows),
        }
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        report["overlap"][f"{left}_{right}_family_overlap"] = sorted(families[left] & families[right])
        report["overlap"][f"{left}_{right}_sequence_overlap_count"] = len(seqs[left] & seqs[right])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test JSONL splits.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["random", "family_disjoint"], default="random")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {input_path}")
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit(f"Input JSONL is empty: {input_path}")

    if args.mode == "family_disjoint":
        split = family_disjoint_split(rows, args.train_ratio, args.val_ratio, args.seed)
        if split is None:
            train, val, test = random_split(rows, args.train_ratio, args.val_ratio, args.seed)
        else:
            train, val, test = split
            check_family_disjoint(train, val, test)
    else:
        train, val, test = random_split(rows, args.train_ratio, args.val_ratio, args.seed)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    report = split_report(train, val, test)
    with (out_dir / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"train: {len(train)} samples -> {out_dir / 'train.jsonl'}")
    print(f"val: {len(val)} samples -> {out_dir / 'val.jsonl'}")
    print(f"test: {len(test)} samples -> {out_dir / 'test.jsonl'}")
    if args.mode == "family_disjoint":
        print("family-disjoint split verified" if split is not None else "used random fallback")
    leakage = report["overlap"]["train_val_sequence_overlap_count"] + report["overlap"]["train_test_sequence_overlap_count"]
    if args.mode == "random" and leakage > max(10, 0.05 * max(1, len(train))):
        print("Warning: random split has substantial exact sequence overlap. Consider deduplication or family_disjoint mode.")
    print(f"split report -> {out_dir / 'split_report.json'}")


if __name__ == "__main__":
    main()
