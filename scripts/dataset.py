"""External RNA dataset download, prepare, and split utilities.

Usage:
  python scripts/dataset.py download --name bprna --out dataset/raw/bprna
  python scripts/dataset.py prepare --input dataset/raw/bprna --format auto --out dataset/processed/bprna/clean.jsonl --max_length 512
  python scripts/dataset.py split --input dataset/processed/bprna/clean.jsonl --out dataset/processed/bprna --mode random
"""

from __future__ import annotations
import argparse, json, os, random, sys, gzip, hashlib
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError

KNOWN_DATASETS = {
    "bprna": {
        "name": "bpRNA",
        "url": "http://bprna.cgrb.oregonstate.edu/download/bpRNA.tar.gz",
        "description": "bpRNA-1m: over 100,000 annotated RNA secondary structures",
        "format": "stk_gz",
        "paper": "Danaee et al. 2018",
    },
    "rnastralign": {
        "name": "RNAstralign",
        "url": None,
        "description": "RNA structural alignment benchmark. Manual download required.",
        "format": "fasta_stockholm",
        "paper": "Wilm et al. 2012",
        "manual_url": "https://rna.urmc.rochester.edu/RNAstralign.html",
    },
    "rfam_seed": {
        "name": "Rfam Seed",
        "url": "ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz",
        "description": "Rfam seed alignments in Stockholm format",
        "format": "stockholm_gz",
        "paper": "Kalvari et al. 2020",
    },
}

def cmd_download(args):
    name = args.name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if name not in KNOWN_DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Known: {list(KNOWN_DATASETS.keys())}")
        sys.exit(1)

    info = KNOWN_DATASETS[name]
    url = info.get("url")
    report = {"dataset": name, "status": "unknown"}

    if url is None:
        print(f"Dataset '{name}' requires manual download.")
        print(f"  Paper: {info['paper']}")
        print(f"  URL: {info.get('manual_url', 'N/A')}")
        instruction_path = out_dir / "download_instruction.md"
        with open(instruction_path, "w") as f:
            f.write(f"# {info['name']} Download Instructions\n\n")
            f.write(f"Manual download required.\n")
            f.write(f"Paper: {info['paper']}\n")
            if info.get('manual_url'):
                f.write(f"URL: {info['manual_url']}\n")
            f.write(f"\nPlace downloaded files in: {out_dir}\n")
        report["status"] = "manual_required"
        report["instruction_file"] = str(instruction_path)
        print(f"  Instruction saved: {instruction_path}")
    else:
        try:
            filename = url.split("/")[-1]
            dest = out_dir / filename
            print(f"Downloading {url} -> {dest} ...")
            urlretrieve(url, dest)
            report["status"] = "downloaded"
            report["file"] = str(dest)
            report["size"] = dest.stat().st_size
            print(f"  Downloaded: {dest} ({report['size']} bytes)")
        except URLError as e:
            print(f"Download failed: {e}")
            report["status"] = "download_failed"
            report["error"] = str(e)
            instruction_path = out_dir / "download_instruction.md"
            with open(instruction_path, "w") as f:
                f.write(f"# {info['name']} Download Failed\n\n")
                f.write(f"URL: {url}\nError: {e}\n")
                f.write(f"\nManual download: {info.get('manual_url', url)}\n")
            report["instruction_file"] = str(instruction_path)

    report_path = out_dir / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_path}")


def cmd_prepare(args):
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = args.format or "auto"
    max_len = int(args.max_length or 512)

    records = []
    if input_path.is_dir():
        records = _prepare_from_dir(input_path, fmt, max_len)
    elif input_path.suffix == ".jsonl":
        records = _prepare_from_jsonl(input_path, max_len)
    elif input_path.suffix == ".gz":
        records = _prepare_from_gz(input_path, fmt, max_len)
    else:
        records = _prepare_from_file(input_path, fmt, max_len)

    if not records:
        print(f"WARNING: No valid records extracted from {input_path}")
        with open(out_path, "w") as f:
            pass
        return

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Prepared {len(records)} records -> {out_path}")


def _prepare_from_dir(dir_path, fmt, max_len):
    records = []
    for fpath in sorted(dir_path.rglob("*")):
        if fpath.suffix in [".jsonl", ".json"]:
            records.extend(_prepare_from_jsonl(fpath, max_len))
        elif fpath.suffix in [".ct"]:
            records.extend(_prepare_ct(fpath, max_len))
        elif fpath.suffix in [".bpseq"]:
            records.extend(_prepare_bpseq(fpath, max_len))
        elif fpath.suffix in [".stk", ".stockholm"]:
            records.extend(_prepare_stockholm(fpath, max_len))
        elif fpath.suffix in [".fasta", ".fa", ".fna"]:
            records.extend(_prepare_fasta_like(fpath, max_len))
    return records


def _prepare_from_jsonl(path, max_len):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                if "seq" in rec and "struct" in rec:
                    if len(rec["seq"]) <= max_len:
                        rec.setdefault("id", rec.get("id", path.stem))
                        rec.setdefault("family", rec.get("family", "OTHER"))
                        rec.setdefault("source", rec.get("source", path.stem))
                        rec.setdefault("motifs", [])
                        rec.setdefault("pairs", [])
                        rec["length"] = len(rec["seq"])
                        records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def _prepare_ct(path, max_len):
    records = []
    try:
        with open(path) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    total_bases = int(parts[0])
                    seq = []
                    struct_map = {}
                    pairs = []
                    for j in range(total_bases):
                        if i + j + 1 >= len(lines): break
                        ct_line = lines[i + j + 1].strip().split()
                        if len(ct_line) >= 5:
                            seq.append(ct_line[1])
                            pair_idx = int(ct_line[4]) - 1
                            if pair_idx >= 0 and pair_idx > j:
                                pairs.append((j, pair_idx))
                    seq_str = "".join(seq)
                    struct_str = "." * len(seq_str)
                    for a, b in pairs:
                        if a < b:
                            struct_str = struct_str[:a] + "(" + struct_str[a+1:]
                            struct_str = struct_str[:b] + ")" + struct_str[b+1:]
                    if len(seq_str) <= max_len:
                        records.append({
                            "id": f"{path.stem}_{i}",
                            "seq": seq_str,
                            "struct": struct_str,
                            "family": "OTHER",
                            "source": path.stem,
                            "motifs": [],
                            "pairs": pairs,
                            "length": len(seq_str),
                        })
                    i += total_bases + 1
                except (ValueError, IndexError):
                    i += 1
            else:
                i += 1
    except Exception as e:
        print(f"WARNING: CT parse error {path}: {e}")
    return records


def _prepare_bpseq(path, max_len):
    records = []
    try:
        with open(path) as f:
            lines = f.readlines()
        seq = []
        pairs = []
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 3:
                seq.append(parts[1])
                pair_idx = int(parts[2]) - 1
                if pair_idx >= 0:
                    pairs.append((len(seq) - 1, pair_idx))
        seq_str = "".join(seq)
        struct_str = _pairs_to_dotbracket(pairs, len(seq_str))
        if 0 < len(seq_str) <= max_len:
            records.append({
                "id": path.stem,
                "seq": seq_str,
                "struct": struct_str,
                "family": "OTHER",
                "source": path.stem,
                "motifs": [],
                "pairs": [(a, b) for a, b in pairs if a < b],
                "length": len(seq_str),
            })
    except Exception as e:
        print(f"WARNING: BPSEQ parse error {path}: {e}")
    return records


def _prepare_stockholm(path, max_len):
    records = []
    try:
        with open(path) as f:
            content = f.read()
        blocks = content.split("//")
        for block in blocks:
            block = block.strip()
            if not block or "#=GF" not in block: continue
            seqs = {}
            structs = {}
            family = "OTHER"
            for line in block.split("\n"):
                line = line.strip()
                if line.startswith("#=GF AC"):
                    family = line.split("AC")[-1].strip() or family
                elif not line.startswith("#") and line:
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        data = "".join(parts[1:]).upper().replace("T", "U")
                        if name.endswith("_SS"):
                            structs[name.replace("_SS", "")] = data
                        else:
                            seqs[name] = data
            for name, seq in seqs.items():
                struct = structs.get(name, "." * len(seq))
                if 0 < len(seq) <= max_len:
                    records.append({
                        "id": name,
                        "seq": seq.replace("T", "U"),
                        "struct": struct,
                        "family": family,
                        "source": path.stem,
                        "motifs": [],
                        "pairs": [],
                        "length": len(seq),
                    })
    except Exception as e:
        print(f"WARNING: Stockholm parse error {path}: {e}")
    return records


def _prepare_fasta_like(path, max_len):
    records = []
    try:
        with open(path) as f:
            seq_id = None
            seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq_id and seq:
                        seq_str = "".join(seq)
                        if len(seq_str) <= max_len:
                            records.append({
                                "id": seq_id,
                                "seq": seq_str,
                                "struct": "." * len(seq_str),
                                "family": "OTHER",
                                "source": path.stem,
                                "motifs": [],
                                "pairs": [],
                                "length": len(seq_str),
                            })
                    seq_id = line[1:].split()[0]
                    seq = []
                else:
                    seq.append(line.upper().replace("T", "U"))
            if seq_id and seq:
                seq_str = "".join(seq)
                if len(seq_str) <= max_len:
                    records.append({"id": seq_id, "seq": seq_str, "struct": "."*len(seq_str), "family": "OTHER", "source": path.stem, "motifs": [], "pairs": [], "length": len(seq_str)})
    except Exception as e:
        print(f"WARNING: FASTA parse error {path}: {e}")
    return records


def _prepare_from_gz(path, fmt, max_len):
    records = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if fmt in ["auto", "stockholm", "stk"] and "# STOCKHOLM" in content[:1000]:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".stk", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            records = _prepare_stockholm(Path(tmp_path), max_len)
            os.unlink(tmp_path)
        elif fmt in ["auto", "jsonl"]:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                for line in content.split("\n"):
                    if line.strip():
                        tmp.write(line + "\n")
                tmp_path = tmp.name
            records = _prepare_from_jsonl(Path(tmp_path), max_len)
            os.unlink(tmp_path)
    except Exception as e:
        print(f"WARNING: GZ parse error {path}: {e}")
    return records


def _prepare_from_file(path, fmt, max_len):
    if path.suffix in [".ct"]:
        return _prepare_ct(path, max_len)
    elif path.suffix in [".bpseq"]:
        return _prepare_bpseq(path, max_len)
    elif path.suffix in [".stk", ".stockholm"]:
        return _prepare_stockholm(path, max_len)
    elif path.suffix in [".fasta", ".fa", ".fna"]:
        return _prepare_fasta_like(path, max_len)
    return []


def _pairs_to_dotbracket(pairs, length):
    chars = ["."] * length
    for i, j in pairs:
        i, j = int(i), int(j)
        if i > j: i, j = j, i
        if 0 <= i < j < length:
            chars[i] = "("
            chars[j] = ")"
    return "".join(chars)


def cmd_split(args):
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode or "random"

    records = _prepare_from_jsonl(input_path, 999999)
    if not records:
        print(f"No records in {input_path}")
        sys.exit(1)

    if mode == "family":
        families = {}
        for r in records:
            fam = r.get("family", "OTHER") or "OTHER"
            families.setdefault(fam, []).append(r)
        no_family_count = len(families.get("OTHER", []))
        if no_family_count > len(records) * 0.3:
            print(f"WARNING: {no_family_count}/{len(records)} records have no family label.")
            print("Family-disjoint split may be unreliable. Consider using --mode random.")

        fam_list = list(families.keys())
        random.shuffle(fam_list)
        n_train = max(1, int(len(fam_list) * 0.7))
        n_val = max(1, int(len(fam_list) * 0.15))
        train_fams = set(fam_list[:n_train])
        val_fams = set(fam_list[n_train:n_train + n_val])
        test_fams = set(fam_list[n_train + n_val:])

        splits = {"train": [], "val": [], "test": []}
        for fam, recs in families.items():
            if fam in train_fams:
                splits["train"].extend(recs)
            elif fam in val_fams:
                splits["val"].extend(recs)
            else:
                splits["test"].extend(recs)
    else:
        random.shuffle(records)
        n_train = int(len(records) * 0.7)
        n_val = int(len(records) * 0.15)
        splits = {
            "train": records[:n_train],
            "val": records[n_train:n_train + n_val],
            "test": records[n_train + n_val:],
        }

    for split_name, split_records in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for r in split_records:
                f.write(json.dumps(r) + "\n")
        print(f"  {split_name}: {len(split_records)} records -> {path}")

    print(f"Split complete ({mode}): train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


def main():
    parser = argparse.ArgumentParser(description="External RNA dataset utilities")
    sub = parser.add_subparsers(dest="command")

    p_down = sub.add_parser("download")
    p_down.add_argument("--name", required=True, choices=list(KNOWN_DATASETS.keys()))
    p_down.add_argument("--out", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--input", required=True)
    p_prep.add_argument("--format", default="auto")
    p_prep.add_argument("--out", required=True)
    p_prep.add_argument("--max_length", type=int, default=512)

    p_split = sub.add_parser("split")
    p_split.add_argument("--input", required=True)
    p_split.add_argument("--out", required=True)
    p_split.add_argument("--mode", choices=["random", "family"], default="random")

    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "split":
        cmd_split(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
