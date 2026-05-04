from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.structure import (  # noqa: E402
    ALLOWED_STRUCT_CHARS,
    canonical_pair,
    has_pseudoknot,
    infer_simple_motifs,
    parse_dot_bracket,
)


ALLOWED_SEQ_CHARS = set("AUGCNT")


def normalize_pairs(raw_pairs: Iterable[Iterable[int]] | None, length: int) -> list[tuple[int, int]]:
    if raw_pairs is None:
        return []
    pairs = []
    used = set()
    for raw_pair in raw_pairs:
        pair = list(raw_pair)
        if len(pair) != 2:
            continue
        i, j = int(pair[0]), int(pair[1])
        if i > j:
            i, j = j, i
        if i < 0 or j >= length or i == j:
            continue
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j))
    return sorted(pairs)


def clean_sample(raw: dict[str, Any], line_no: int, max_length: int) -> tuple[dict[str, Any] | None, list[str]]:
    warnings = []
    if "seq" not in raw or "struct" not in raw:
        return None, [f"line {line_no}: missing seq or struct"]

    seq_raw = str(raw["seq"]).upper()
    invalid_seq = sorted(set(seq_raw) - ALLOWED_SEQ_CHARS)
    if invalid_seq:
        return None, [f"line {line_no}: invalid sequence characters {invalid_seq}"]
    seq = seq_raw.replace("T", "U")
    struct = str(raw["struct"])
    invalid_struct = sorted(set(struct) - ALLOWED_STRUCT_CHARS)
    if invalid_struct:
        return None, [f"line {line_no}: invalid structure characters {invalid_struct}"]
    if len(seq) != len(struct):
        return None, [f"line {line_no}: len(seq)={len(seq)} len(struct)={len(struct)} mismatch"]
    if len(seq) > max_length:
        return None, [f"line {line_no}: length {len(seq)} exceeds max_length {max_length}"]

    try:
        parsed_pairs = parse_dot_bracket(struct)
    except ValueError as exc:
        return None, [f"line {line_no}: invalid dot-bracket: {exc}"]

    raw_pairs = normalize_pairs(raw.get("pairs"), len(seq)) if raw.get("pairs") is not None else None
    if raw_pairs is None:
        pairs = parsed_pairs
    elif set(raw_pairs) != set(parsed_pairs):
        warnings.append(f"line {line_no}: pairs do not match struct; using parsed struct pairs")
        pairs = parsed_pairs
    else:
        pairs = raw_pairs

    if raw.get("length") != len(seq):
        warnings.append(f"line {line_no}: length field corrected from {raw.get('length')} to {len(seq)}")

    family = str(raw.get("family", "OTHER") or "OTHER")
    motifs = raw.get("motifs")
    if not motifs:
        try:
            motifs = infer_simple_motifs(seq=seq, struct=struct, pairs=pairs)
        except Exception:
            motifs = []

    cleaned = {
        "id": str(raw.get("id", f"RNA_{line_no:06d}")),
        "seq": seq,
        "struct": struct,
        "family": family,
        "motifs": motifs,
        "pairs": [[i, j] for i, j in pairs],
        "length": len(seq),
    }
    return cleaned, warnings


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def print_stats(cleaned: list[dict[str, Any]], total: int, invalid_count: int, warning_count: int) -> None:
    lengths = [sample["length"] for sample in cleaned]
    pair_counts = [len(sample["pairs"]) for sample in cleaned]
    family_counts = Counter(sample["family"] for sample in cleaned)
    struct_counts = Counter(char for sample in cleaned for char in sample["struct"])
    all_dot_count = sum(1 for sample in cleaned if set(sample["struct"]) <= {"."})
    pseudoknot_count = sum(1 for sample in cleaned if has_pseudoknot(sample["pairs"]))
    canonical = 0
    pair_total = 0
    for sample in cleaned:
        for i, j in sample["pairs"]:
            pair_total += 1
            canonical += int(canonical_pair(sample["seq"][i], sample["seq"][j]))

    print(f"total samples: {total}")
    print(f"valid samples: {len(cleaned)}")
    print(f"invalid samples: {invalid_count}")
    print(f"warnings: {warning_count}")
    if lengths:
        print(
            "length distribution: "
            f"min={min(lengths)} max={max(lengths)} mean={float(np.mean(lengths)):.2f} "
            f"p50={percentile(lengths, 50):.2f} p90={percentile(lengths, 90):.2f} "
            f"p95={percentile(lengths, 95):.2f}"
        )
    else:
        print("length distribution: empty")
    print(f"family count top 20: {family_counts.most_common(20)}")
    print(f"structure token distribution: {dict(struct_counts)}")
    if pair_counts:
        print(
            "pair count distribution: "
            f"min={min(pair_counts)} max={max(pair_counts)} mean={float(np.mean(pair_counts)):.2f} "
            f"p50={percentile(pair_counts, 50):.2f} p90={percentile(pair_counts, 90):.2f}"
        )
    else:
        print("pair count distribution: empty")
    print(f"all-dot structure ratio: {all_dot_count / max(1, len(cleaned)):.4f}")
    print(f"pseudoknot ratio: {pseudoknot_count / max(1, len(cleaned)):.4f}")
    print(f"canonical pair ratio: {canonical / max(1, pair_total):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and clean RNA JSONL data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {input_path}")

    cleaned = []
    invalid = []
    warnings = []
    total = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid.append(f"line {line_no}: malformed JSON: {exc}")
                continue
            sample, sample_warnings = clean_sample(raw, line_no, args.max_length)
            warnings.extend(sample_warnings)
            if sample is None:
                invalid.extend(sample_warnings)
            else:
                cleaned.append(sample)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in cleaned:
            handle.write(json.dumps(sample) + "\n")

    print_stats(cleaned, total, len(invalid), len(warnings))
    if invalid[:10]:
        print("invalid examples:")
        for message in invalid[:10]:
            print(f"  {message}")
    if warnings[:10]:
        print("warning examples:")
        for message in warnings[:10]:
            print(f"  {message}")
    print(f"cleaned JSONL saved to: {output_path}")


if __name__ == "__main__":
    main()

