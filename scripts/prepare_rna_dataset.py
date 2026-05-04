from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_dataset import clean_sample  # noqa: E402
from utils.structure import has_pseudoknot, infer_simple_motifs, pairs_to_dot_bracket  # noqa: E402


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [file for file in path.rglob("*") if file.is_file()]


def detect_format(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return "jsonl"
    if suffix in {".ct"}:
        return "ct"
    if suffix in {".bpseq"}:
        return "bpseq"
    if suffix in {".fa", ".fasta", ".dbn"} or text.lstrip().startswith(">"):
        return "fasta"
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    fields = first.split()
    if len(fields) >= 6 and fields[0].isdigit():
        return "ct"
    if len(fields) >= 3 and fields[0].isdigit():
        return "bpseq"
    if first.startswith("{"):
        return "jsonl"
    return "unknown"


def normalize_pairs(raw_pairs: Iterable[Iterable[int]], length: int, drop_pseudoknot: bool) -> tuple[list[tuple[int, int]], list[str]]:
    warnings = []
    pairs = []
    used = set()
    for raw_i, raw_j in raw_pairs:
        i, j = int(raw_i), int(raw_j)
        if i > j:
            i, j = j, i
        if i < 0 or j >= length or i == j:
            warnings.append(f"skipped invalid pair {(i, j)}")
            continue
        if i in used or j in used:
            warnings.append(f"skipped conflicting pair {(i, j)}")
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j))
    pairs = sorted(pairs)
    if drop_pseudoknot and has_pseudoknot(pairs):
        kept = []
        for pair in pairs:
            if has_pseudoknot(kept + [pair]):
                warnings.append(f"skipped pseudoknot-conflicting pair {pair}")
            else:
                kept.append(pair)
        pairs = kept
    return pairs, warnings


def make_sample(sample_id: str, seq: str, pairs: list[tuple[int, int]], family: str) -> dict[str, Any]:
    struct = pairs_to_dot_bracket(pairs, len(seq))
    return {
        "id": sample_id,
        "seq": seq.upper().replace("T", "U"),
        "struct": struct,
        "family": family or "OTHER",
        "motifs": infer_simple_motifs(seq=seq, struct=struct, pairs=pairs),
        "pairs": [[i, j] for i, j in pairs],
        "length": len(seq),
    }


def parse_jsonl(path: Path, max_length: int, drop_pseudoknot: bool) -> tuple[list[dict], list[dict]]:
    samples = []
    errors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                sample, warnings = clean_sample(raw, line_no, max_length)
                if sample is None:
                    errors.append({"file": str(path), "line": line_no, "reason": "; ".join(warnings)})
                    continue
                if drop_pseudoknot and has_pseudoknot(sample["pairs"]):
                    pairs, pair_warnings = normalize_pairs(sample["pairs"], sample["length"], True)
                    sample = make_sample(sample["id"], sample["seq"], pairs, sample.get("family", "OTHER"))
                    warnings.extend(pair_warnings)
                samples.append(sample)
                for warning in warnings:
                    errors.append({"file": str(path), "line": line_no, "warning": warning})
            except Exception as exc:
                errors.append({"file": str(path), "line": line_no, "reason": str(exc)})
    return samples, errors


def parse_fasta_dot(path: Path, max_length: int, drop_pseudoknot: bool) -> tuple[list[dict], list[dict]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    errors = []
    idx = 0
    while idx < len(lines):
        if not lines[idx].startswith(">"):
            errors.append({"file": str(path), "line": idx + 1, "reason": "expected FASTA header"})
            idx += 1
            continue
        header = lines[idx][1:].strip()
        seq = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        struct = lines[idx + 2].strip() if idx + 2 < len(lines) else ""
        idx += 3
        sample_id = header.split()[0] if header else f"{path.stem}_{len(samples) + 1}"
        family = "OTHER"
        for part in header.split():
            if part.startswith("family="):
                family = part.split("=", 1)[1] or "OTHER"
        sample, warnings = clean_sample({"id": sample_id, "seq": seq, "struct": struct, "family": family}, idx, max_length)
        if sample is None:
            errors.append({"file": str(path), "id": sample_id, "reason": "; ".join(warnings)})
        elif drop_pseudoknot and has_pseudoknot(sample["pairs"]):
            pairs, pair_warnings = normalize_pairs(sample["pairs"], sample["length"], True)
            samples.append(make_sample(sample_id, sample["seq"], pairs, family))
            errors.extend({"file": str(path), "id": sample_id, "warning": warning} for warning in pair_warnings)
        else:
            samples.append(sample)
    return samples, errors


def parse_bpseq(path: Path, max_length: int, drop_pseudoknot: bool) -> tuple[list[dict], list[dict]]:
    rows = []
    errors = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            errors.append({"file": str(path), "line": line_no, "reason": "invalid BPSEQ row"})
            continue
        rows.append((int(parts[0]), parts[1].upper(), int(parts[2])))
    if not rows:
        return [], errors or [{"file": str(path), "reason": "empty BPSEQ"}]
    rows.sort()
    seq = "".join(base for _, base, _ in rows)
    raw_pairs = []
    for idx, _, pair_idx in rows:
        if pair_idx > 0 and idx < pair_idx:
            raw_pairs.append((idx - 1, pair_idx - 1))
    pairs, warnings = normalize_pairs(raw_pairs, len(seq), drop_pseudoknot)
    try:
        sample = make_sample(path.stem, seq, pairs, "OTHER")
        checked, check_warnings = clean_sample(sample, 1, max_length)
        errors.extend({"file": str(path), "warning": warning} for warning in warnings + check_warnings)
        return ([checked] if checked else []), errors
    except Exception as exc:
        errors.append({"file": str(path), "reason": str(exc)})
        return [], errors


def parse_ct(path: Path, max_length: int, drop_pseudoknot: bool) -> tuple[list[dict], list[dict]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    errors = []
    idx = 0
    record = 0
    while idx < len(lines):
        header = lines[idx].split()
        if not header or not header[0].isdigit():
            errors.append({"file": str(path), "line": idx + 1, "reason": "invalid CT header"})
            idx += 1
            continue
        length = int(header[0])
        name = header[1] if len(header) > 1 else f"{path.stem}_{record + 1}"
        idx += 1
        rows = lines[idx : idx + length]
        idx += length
        record += 1
        seq_chars = []
        raw_pairs = []
        for row in rows:
            parts = row.split()
            if len(parts) < 5:
                errors.append({"file": str(path), "id": name, "reason": f"invalid CT row: {row}"})
                continue
            pos = int(parts[0])
            seq_chars.append(parts[1].upper())
            pair_pos = int(parts[4])
            if pair_pos > 0 and pos < pair_pos:
                raw_pairs.append((pos - 1, pair_pos - 1))
        seq = "".join(seq_chars)
        pairs, warnings = normalize_pairs(raw_pairs, len(seq), drop_pseudoknot)
        try:
            sample = make_sample(name, seq, pairs, "OTHER")
            checked, check_warnings = clean_sample(sample, record, max_length)
            if checked:
                samples.append(checked)
            else:
                errors.append({"file": str(path), "id": name, "reason": "; ".join(check_warnings)})
            errors.extend({"file": str(path), "id": name, "warning": warning} for warning in warnings + check_warnings)
        except Exception as exc:
            errors.append({"file": str(path), "id": name, "reason": str(exc)})
    return samples, errors


def deduplicate(samples: list[dict], dedup_by: str) -> tuple[list[dict], int]:
    seen = set()
    kept = []
    for sample in samples:
        if dedup_by == "seq":
            key = sample["seq"]
        elif dedup_by == "struct":
            key = sample["struct"]
        else:
            key = (sample["seq"], sample["struct"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(sample)
    return kept, len(samples) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert common RNA secondary-structure files to JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["auto", "jsonl", "fasta", "ct", "bpseq"], default="auto")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--drop_pseudoknot", default="true")
    parser.add_argument("--deduplicate", default="true")
    parser.add_argument("--dedup_by", choices=["seq", "seq_struct", "struct"], default="seq_struct")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")
    output_path = Path(args.output)
    error_path = output_path.parent / "prepare_errors.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsers = {"jsonl": parse_jsonl, "fasta": parse_fasta_dot, "ct": parse_ct, "bpseq": parse_bpseq}
    all_samples = []
    all_errors = []
    stats = Counter()
    for file in iter_input_files(input_path):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            file_format = detect_format(file, text) if args.format == "auto" else args.format
            if file_format not in parsers:
                all_errors.append({"file": str(file), "reason": "unknown format"})
                stats["unknown_failed"] += 1
                continue
            samples, errors = parsers[file_format](file, args.max_length, str_to_bool(args.drop_pseudoknot))
            all_samples.extend(samples)
            all_errors.extend(errors)
            stats[f"{file_format}_success"] += len(samples)
            stats[f"{file_format}_failed"] += sum(1 for item in errors if "reason" in item)
        except Exception as exc:
            all_errors.append({"file": str(file), "reason": str(exc)})
            stats["file_failed"] += 1

    before = len(all_samples)
    removed = 0
    if str_to_bool(args.deduplicate):
        all_samples, removed = deduplicate(all_samples, args.dedup_by)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in all_samples:
            handle.write(json.dumps(sample) + "\n")
    with error_path.open("w", encoding="utf-8") as handle:
        for error in all_errors:
            handle.write(json.dumps(error) + "\n")

    print(f"converted samples before_dedup={before}")
    print(f"converted samples after_dedup={len(all_samples)}")
    print(f"removed_duplicates={removed}")
    print(f"format stats={dict(stats)}")
    print(f"output={output_path}")
    print(f"errors={error_path}")


if __name__ == "__main__":
    main()

