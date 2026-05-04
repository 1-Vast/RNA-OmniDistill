from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import evaluate_structures
from utils.structure import parse_dot_bracket, validate_structure


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Malformed JSON at {path}:{line_no}: {exc}")
    return rows


def pair_sets(row: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    true_pairs = {tuple(pair) for pair in row.get("true_pairs", [])}
    pred_pairs = {tuple(pair) for pair in row.get("pred_pairs", [])}
    if not true_pairs:
        true_pairs = set(parse_dot_bracket(row.get("true_struct", "")))
    if not pred_pairs:
        try:
            pred_pairs = set(parse_dot_bracket(row.get("pred_struct", "")))
        except ValueError:
            pred_pairs = set()
    return true_pairs, pred_pairs


def reason_for(row: dict) -> str:
    true_pairs, pred_pairs = pair_sets(row)
    pred_struct = row.get("pred_struct", "")
    valid = bool(row.get("valid", validate_structure(row.get("seq", ""), pred_struct)))
    if not valid:
        return "invalid"
    if pred_struct and set(pred_struct) <= {"."}:
        return "all_dot"
    if len(pred_pairs) < 0.2 * max(1, len(true_pairs)):
        return "under_pairing"
    if len(pred_pairs) > 2.0 * max(1, len(true_pairs)):
        return "over_pairing"
    if int(row.get("length", len(row.get("seq", "")))) > 300 and float(row.get("pair_f1", 0.0)) < 0.2:
        return "long_sequence_failure"
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    if fp > fn and fp >= 3:
        return "false_positive_pairs"
    if fn >= 3:
        return "false_negative_pairs"
    return "low_pair_f1"


def bucket_length(length: int) -> str:
    if length <= 100:
        return "0-100"
    if length <= 200:
        return "100-200"
    if length <= 300:
        return "200-300"
    return "300+"


def bucket_pair_count(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 15:
        return "6-15"
    return "16+"


def summarize_group(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "pair_f1": 0.0}
    seqs = [row["seq"] for row in rows]
    preds = [row["pred_struct"] for row in rows]
    trues = [row["true_struct"] for row in rows]
    metrics = evaluate_structures(preds, trues, seqs)
    metrics["count"] = len(rows)
    return metrics


def write_cases(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            true_pairs, pred_pairs = pair_sets(row)
            case = {
                "id": row.get("id"),
                "seq": row.get("seq"),
                "true_struct": row.get("true_struct"),
                "pred_struct": row.get("pred_struct"),
                "family": row.get("family", "OTHER"),
                "length": row.get("length"),
                "pair_f1": row.get("pair_f1", 0.0),
                "reason": row.get("reason", reason_for(row)),
                "false_positive_pairs": sorted(pred_pairs - true_pairs),
                "false_negative_pairs": sorted(true_pairs - pred_pairs),
            }
            handle.write(json.dumps(case) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose prediction JSONL failure modes.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise SystemExit(f"Prediction file not found: {pred_path}. Run run_benchmark.py or export_predictions.py first.")
    rows = read_jsonl(pred_path)
    if not rows:
        raise SystemExit(f"Prediction file is empty: {pred_path}")
    for row in rows:
        row["reason"] = reason_for(row)

    sorted_rows = sorted(rows, key=lambda row: float(row.get("pair_f1", 0.0)))
    bad = sorted_rows[:20]
    good = list(reversed(sorted_rows[-20:]))
    all_dot = [row for row in rows if set(row.get("pred_struct", "")) <= {"."}]
    fp_many = []
    fn_many = []
    for row in rows:
        true_pairs, pred_pairs = pair_sets(row)
        if len(pred_pairs - true_pairs) >= 3:
            fp_many.append(row)
        if len(true_pairs - pred_pairs) >= 3:
            fn_many.append(row)
    long_failures = [row for row in rows if int(row.get("length", 0)) > 300 and float(row.get("pair_f1", 0.0)) < 0.2]

    by_family = defaultdict(list)
    by_length = defaultdict(list)
    by_pair_count = defaultdict(list)
    for row in rows:
        true_pairs, _ = pair_sets(row)
        by_family[row.get("family", "OTHER")].append(row)
        by_length[bucket_length(int(row.get("length", 0)))].append(row)
        by_pair_count[bucket_pair_count(len(true_pairs))].append(row)

    diagnosis = {
        "total": len(rows),
        "worst_20": [{key: row.get(key) for key in ["id", "family", "length", "pair_f1", "reason"]} for row in bad],
        "best_20": [{key: row.get(key) for key in ["id", "family", "length", "pair_f1", "reason"]} for row in good],
        "all_dot_count": len(all_dot),
        "false_positive_many_count": len(fp_many),
        "false_negative_many_count": len(fn_many),
        "long_sequence_failure_count": len(long_failures),
        "family_wise_failure": {key: summarize_group(value) for key, value in sorted(by_family.items())},
        "length_bucket_failure": {key: summarize_group(value) for key, value in sorted(by_length.items())},
        "pair_count_bucket_failure": {key: summarize_group(value) for key, value in sorted(by_pair_count.items())},
        "suggestions": [],
    }
    if len(all_dot) / max(1, len(rows)) > 0.8:
        diagnosis["suggestions"].extend(["increase lambda_pair", "increase structure_positive_weight", "check pair labels"])
    if diagnosis["false_positive_many_count"] > diagnosis["false_negative_many_count"]:
        diagnosis["suggestions"].extend(["decrease lambda_pair", "increase negative sampling ratio", "check pair_positive_weight"])
    if diagnosis["false_negative_many_count"] >= diagnosis["false_positive_many_count"]:
        diagnosis["suggestions"].extend(["increase lambda_pair", "check Nussinov score source", "inspect pair head sequence positions"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")
    txt_path = out_path.with_suffix(".txt")
    txt_lines = [
        "RNA-OmniDiffusion Prediction Diagnosis",
        "",
        f"total: {diagnosis['total']}",
        f"all_dot_count: {diagnosis['all_dot_count']}",
        f"false_positive_many_count: {diagnosis['false_positive_many_count']}",
        f"false_negative_many_count: {diagnosis['false_negative_many_count']}",
        f"long_sequence_failure_count: {diagnosis['long_sequence_failure_count']}",
        "",
        "Suggestions:",
        *[f"- {item}" for item in sorted(set(diagnosis["suggestions"]))],
    ]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    write_cases(out_path.parent / "bad_cases.jsonl", bad)
    write_cases(out_path.parent / "good_cases.jsonl", good)
    print(f"diagnosis json -> {out_path}")
    print(f"diagnosis txt -> {txt_path}")
    print(f"bad cases -> {out_path.parent / 'bad_cases.jsonl'}")
    print(f"good cases -> {out_path.parent / 'good_cases.jsonl'}")


if __name__ == "__main__":
    main()
