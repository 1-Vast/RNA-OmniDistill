from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.struct import infer_simple_motifs, pairs_to_dot_bracket, parse_dot_bracket


ARCHIVE_URL = "https://rna.urmc.rochester.edu/pub/archiveII.tar.gz"


def fetch(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.set != "archive":
        raise SystemExit("Only --set archive is supported by the compact fetch command.")
    target = out / "archive.tar.gz"
    if not target.exists():
        try:
            urllib.request.urlretrieve(ARCHIVE_URL, target)
        except Exception as exc:
            raise SystemExit(f"Download failed: {exc}")
    try:
        with tarfile.open(target, "r:gz") as tar:
            tar.extractall(out)
    except Exception as exc:
        raise SystemExit(f"Extract failed: {exc}")
    (out / "report.json").write_text(json.dumps({"set": args.set, "file": str(target)}, indent=2) + "\n", encoding="utf-8")
    print(f"fetched -> {out}")


def read_ct(path: Path) -> dict | None:
    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    if not lines:
        return None
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        rows.append(parts)
    if not rows:
        return None
    seq = "".join(row[1].upper().replace("T", "U") for row in rows)
    pairs = []
    for row in rows:
        i = int(row[0]) - 1
        j = int(row[4]) - 1
        if j > i:
            pairs.append((i, j))
    return {"id": path.stem, "seq": seq, "struct": pairs_to_dot_bracket(pairs, len(seq)), "family": path.parent.name, "pairs": pairs, "length": len(seq)}


def prep(args: argparse.Namespace) -> None:
    source = Path(args.input)
    paths = [source] if source.is_file() else list(source.rglob("*"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for path in paths:
            row = None
            if path.suffix.lower() == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    seq = raw["seq"].upper().replace("T", "U")
                    struct = raw["struct"]
                    if len(seq) > args.maxlen or len(seq) != len(struct):
                        continue
                    pairs = raw.get("pairs") or parse_dot_bracket(struct)
                    raw.update({"seq": seq, "pairs": pairs, "length": len(seq), "family": raw.get("family", "OTHER"), "motifs": raw.get("motifs") or infer_simple_motifs(struct=struct)})
                    handle.write(json.dumps(raw) + "\n")
                    count += 1
                continue
            if path.suffix.lower() == ".ct":
                row = read_ct(path)
            if row and row["length"] <= args.maxlen:
                row["motifs"] = infer_simple_motifs(struct=row["struct"])
                handle.write(json.dumps(row) + "\n")
                count += 1
    print(f"prepared={count} -> {out}")


def check(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    total = 0
    with inp.open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            try:
                row = json.loads(line)
                row["seq"] = row["seq"].upper().replace("T", "U")
                if len(row["seq"]) != len(row["struct"]) or len(row["seq"]) > args.maxlen:
                    continue
                row["pairs"] = row.get("pairs") or parse_dot_bracket(row["struct"])
                row["length"] = len(row["seq"])
                row["family"] = row.get("family", "OTHER")
                row["motifs"] = row.get("motifs") or infer_simple_motifs(struct=row["struct"])
                dst.write(json.dumps(row) + "\n")
                valid += 1
            except Exception:
                continue
    print(json.dumps({"total": total, "valid": valid, "output": str(out)}, indent=2))


def split(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(args.seed).shuffle(rows)
    n = len(rows)
    train = int(n * 0.8)
    val = int(n * 0.1)
    parts = {"train.jsonl": rows[:train], "val.jsonl": rows[train:train + val], "test.jsonl": rows[train + val:]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with (out / name).open("w", encoding="utf-8") as handle:
            for row in part:
                handle.write(json.dumps(row) + "\n")
    print(f"split -> {out}")


def prep_rfam_fasta(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_path = Path(args.output)
    min_length = int(getattr(args, "min_length", 20))
    max_length = int(getattr(args, "max_length", 512))
    dedup = bool(getattr(args, "dedup", False))
    limit_val = getattr(args, "limit", None)
    seed = int(getattr(args, "seed", 42))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(
        p for p in input_dir.glob("RF*.fa.gz") if p.is_file()
    )
    if not fasta_files:
        print(f"No RF*.fa.gz files found in {input_dir}", file=sys.stderr)
        return

    rng = random.Random(seed)
    seen = set()
    count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for fa_path in fasta_files:
            stem = fa_path.name.replace(".fa.gz", "").replace(".fa", "")
            family = stem

            try:
                with gzip.open(fa_path, "rt", encoding="utf-8", errors="replace") as fh:
                    seq_id = None
                    seq_parts = []
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(">"):
                            if seq_id is not None and seq_parts:
                                seq = "".join(seq_parts).upper().replace("T", "U")
                                seq = "".join(
                                    base if base in {"A", "U", "G", "C", "N"} else "N"
                                    for base in seq
                                )
                                length = len(seq)
                                if min_length <= length <= max_length:
                                    if not dedup or seq not in seen:
                                        if dedup:
                                            seen.add(seq)
                                        sample_id = f"{family}_{seq_id or 'seq'}"
                                        row = {
                                            "id": sample_id,
                                            "seq": seq,
                                            "family": family,
                                            "source": "rfam_fasta",
                                        }
                                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                        count += 1
                                        if limit_val is not None and count >= int(limit_val):
                                            break
                            header = line[1:].split(None, 1)
                            seq_id = header[0] if header else None
                            seq_parts = []
                        else:
                            seq_parts.append(line)

                    if seq_id is not None and seq_parts and (limit_val is None or count < int(limit_val)):
                        seq = "".join(seq_parts).upper().replace("T", "U")
                        seq = "".join(
                            base if base in {"A", "U", "G", "C", "N"} else "N"
                            for base in seq
                        )
                        length = len(seq)
                        if min_length <= length <= max_length:
                            if not dedup or seq not in seen:
                                if dedup:
                                    seen.add(seq)
                                sample_id = f"{family}_{seq_id or 'seq'}"
                                row = {
                                    "id": sample_id,
                                    "seq": seq,
                                    "family": family,
                                    "source": "rfam_fasta",
                                }
                                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                count += 1
            except Exception as exc:
                print(f"Warning: failed to process {fa_path}: {exc}", file=sys.stderr)
                continue

            if limit_val is not None and count >= int(limit_val):
                break

    print(json.dumps({
        "command": "prep_rfam_fasta",
        "input": str(input_dir),
        "output": str(output_path),
        "families_scanned": len(fasta_files),
        "sequences_written": count,
        "min_length": min_length,
        "max_length": max_length,
        "dedup": dedup,
        "limit": limit_val,
        "seed": seed,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and prepare RNA datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--set", default="archive")
    f.add_argument("--out", default="dataset/raw/archive")
    f.set_defaults(func=fetch)
    p = sub.add_parser("prep")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="auto")
    p.add_argument("--maxlen", type=int, default=512)
    p.set_defaults(func=prep)
    c = sub.add_parser("check")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--maxlen", type=int, default=512)
    c.set_defaults(func=check)
    s = sub.add_parser("split")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", default="random")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=split)
    rf = sub.add_parser("prep_rfam_fasta")
    rf.add_argument("--input", required=True, help="Directory containing RF*.fa.gz files")
    rf.add_argument("--output", required=True, help="Output sequence-only JSONL path")
    rf.add_argument("--min_length", type=int, default=20)
    rf.add_argument("--max_length", type=int, default=512)
    rf.add_argument("--dedup", action="store_true")
    rf.add_argument("--limit", type=int, default=None)
    rf.add_argument("--seed", type=int, default=42)
    rf.set_defaults(func=prep_rfam_fasta)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


