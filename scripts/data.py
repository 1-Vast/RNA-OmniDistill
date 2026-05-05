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
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


