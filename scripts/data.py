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


def inspect_rfam(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    max_lines = int(getattr(args, "max_lines", 100))
    filename = input_path.name

    inspect_dir = Path("outputs/rfam_inspect")
    inspect_dir.mkdir(parents=True, exist_ok=True)

    # Read first N lines from gzip
    lines: list[str] = []
    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            lines.append(line.rstrip("\n\r"))

    # -- Detect format --
    has_stockholm = any("# STOCKHOLM" in line.upper() for line in lines)
    has_stockholm_end = any(line.strip() == "//" for line in lines)
    fasta_count = sum(1 for line in lines if line.startswith(">"))
    tabular_kw = ["accession", "rfam_acc", "start", "end", "chromosome", "strand", "rfam"]
    tabular_score = sum(1 for line in lines for kw in tabular_kw if kw in line.lower())

    if has_stockholm or has_stockholm_end:
        format_type = "stockholm"
        has_sequence_data = True
        recommended_action = "Use prep_rfam_seed to extract sequences from Stockholm alignment"
    elif fasta_count >= 2:
        format_type = "fasta"
        has_sequence_data = True
        recommended_action = "Use prep_rfam_fasta subcommand or generic FASTA parser"
    elif tabular_score >= 3:
        format_type = "tabular_metadata"
        has_sequence_data = False
        recommended_action = "Metadata-only file; use prep_rfam_full_region for further inspection"
    else:
        format_type = "unknown"
        has_sequence_data = None
        recommended_action = "Manual inspection required"

    # Estimate total lines from file size and average line length
    file_size = input_path.stat().st_size
    avg_line_len = sum(len(l) + 1 for l in lines) / len(lines) if lines else 80
    estimated_total_lines = int(file_size * 3.5 / avg_line_len) if lines else None

    report = {
        "file_path": str(input_path.resolve()),
        "format": format_type,
        "first_n_lines": len(lines),
        "line_count_estimate": estimated_total_lines,
        "has_sequence_data": has_sequence_data,
        "recommended_action": recommended_action,
    }

    # Write JSON report
    json_path = inspect_dir / f"{filename}_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Append to / create summary markdown
    summary_entry = (
        f"## {filename}\n"
        f"- **Format**: {format_type}\n"
        f"- **Lines inspected**: {len(lines)}\n"
        f"- **Has sequence data**: {has_sequence_data}\n"
        f"- **Recommended action**: {recommended_action}\n\n"
    )
    summary_path = inspect_dir / "summary.md"
    if summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8") + summary_entry
    else:
        summary = "# Rfam Inspection Summary\n\n" + summary_entry
    summary_path.write_text(summary, encoding="utf-8")

    print(json.dumps(report, indent=2))


def prep_rfam_seed(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    min_length = int(getattr(args, "min_length", 20))
    max_length = int(getattr(args, "max_length", 512))
    dedup = bool(getattr(args, "dedup", False))
    seed = int(getattr(args, "seed", 42))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen: set[str] = set()

    total_raw = 0
    kept = 0
    dropped_short = 0
    dropped_long = 0
    duplicate_count = 0
    lengths: list[int] = []
    families: set[str] = set()

    with (
        gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as fh,
        output_path.open("w", encoding="utf-8") as out_fh,
    ):
        current_family = "unknown"
        in_block = False
        seq_buf: list[tuple[str, str]] = []  # (seq_id, seq_data)

        for line in fh:
            line = line.rstrip("\n\r")

            # End of alignment block
            if line.strip() == "//":
                in_block = False
                for seq_id, raw_seq in seq_buf:
                    total_raw += 1
                    seq = raw_seq.upper().replace("T", "U")
                    # Strip gap characters
                    seq = "".join(c for c in seq if c not in "-.~")
                    # Canonical bases + N; everything else → N
                    seq = "".join(c if c in "AUCGN" else "N" for c in seq)

                    length = len(seq)
                    if length < min_length:
                        dropped_short += 1
                        continue
                    if length > max_length:
                        dropped_long += 1
                        continue
                    if dedup:
                        if seq in seen:
                            duplicate_count += 1
                            continue
                        seen.add(seq)

                    row = {
                        "id": seq_id,
                        "seq": seq,
                        "family": current_family,
                        "source": "rfam_seed",
                    }
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                    lengths.append(length)
                    families.add(current_family)

                seq_buf = []
                current_family = "unknown"
                continue

            # Start of alignment block
            if "STOCKHOLM" in line.upper():
                in_block = True
                continue

            if not in_block:
                continue

            # Parse family accession
            if line.startswith("#=GF AC"):
                parts = line.split()
                if len(parts) >= 3:
                    current_family = parts[2].strip()
                continue

            # Skip other comment / markup lines
            if line.startswith("#") or not line.strip():
                continue

            # Sequence line: name <whitespace> sequence
            parts = line.split(None, 1)
            if len(parts) >= 2:
                sid = parts[0].strip()
                sdata = parts[1].strip().replace(" ", "")
                if sdata:
                    seq_buf.append((sid, sdata))

    # -- Statistics --
    n_lengths = len(lengths)
    if n_lengths:
        mean_length = sum(lengths) / n_lengths
        sorted_l = sorted(lengths)
        p50 = sorted_l[n_lengths // 2]
        p95 = sorted_l[int(n_lengths * 0.95)]
    else:
        mean_length = p50 = p95 = 0

    stats = {
        "command": "prep_rfam_seed",
        "input": str(input_path),
        "output": str(output_path),
        "total_raw": total_raw,
        "kept": kept,
        "dropped_short": dropped_short,
        "dropped_long": dropped_long,
        "duplicate_count": duplicate_count,
        "mean_length": round(mean_length, 1),
        "p50_length": p50,
        "p95_length": p95,
        "family_count": len(families),
    }

    inspect_dir = Path("outputs/rfam_inspect")
    inspect_dir.mkdir(parents=True, exist_ok=True)
    (inspect_dir / "rfam_seed_prep_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


def prep_rfam_full_region(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    min_length = int(getattr(args, "min_length", 20))
    max_length = int(getattr(args, "max_length", 512))
    dedup = bool(getattr(args, "dedup", False))
    limit_val = getattr(args, "limit", None)
    seed = int(getattr(args, "seed", 42))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Probe file format --
    probe_lines: list[str] = []
    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 30:
                break
            probe_lines.append(line.rstrip("\n\r"))

    # Detect separator
    header_fields: list[str] = []
    if probe_lines:
        first = probe_lines[0]
        if "\t" in first:
            header_fields = [f.strip().lower() for f in first.split("\t")]
        else:
            header_fields = [f.strip().lower() for f in first.split(",")]

    seq_keywords = {"sequence", "seq", "rna_seq", "rna_sequence", "aln_seq", "alignment"}
    meta_keywords = {"rfam_acc", "chromosome", "start", "end", "strand", "genomic", "accession"}

    has_seq_field = any(
        any(kw in field for kw in seq_keywords) for field in header_fields
    )
    is_metadata = any(
        any(kw in field for kw in meta_keywords) for field in header_fields
    )

    # Also check for actual sequence content in data rows
    data_lines = [l for l in probe_lines[1:] if l.strip() and not l.startswith("#")]
    seq_chars_total = sum(
        1 for line in data_lines for c in line.upper() if c in "AUCGN"
    )
    # A data line with genuine sequence should have >50% AUCGN content
    has_sequence_content = (
        seq_chars_total > len(data_lines) * 10 if data_lines else False
    )

    # Metadata-only (expected for Rfam.full_region)
    if is_metadata and not has_sequence_content and not has_seq_field:
        inspect_dir = Path("outputs/rfam_inspect")
        inspect_dir.mkdir(parents=True, exist_ok=True)
        md_content = (
            f"# Rfam.full_region Analysis\n\n"
            f"**File**: {input_path}\n"
            f"**Format**: Tabular metadata (no RNA sequence column found)\n\n"
            f"This file contains genomic coordinates and metadata for Rfam family annotations, "
            f"not RNA sequences. Fields detected: {', '.join(header_fields[:10])}\n\n"
            f"The full_region file is a tab-separated file with per-region annotation metadata "
            f"(rfam_acc, chromosome, start, end, strand, etc.) but does NOT contain RNA sequences.\n\n"
            f"**Cannot be used for sequence-only pretraining.**\n"
        )
        (inspect_dir / "full_region_metadata_only.md").write_text(
            md_content, encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "command": "prep_rfam_full_region",
                    "input": str(input_path),
                    "status": "metadata_only",
                    "message": "File contains tabular metadata without RNA sequences.",
                },
                indent=2,
            )
        )
        return

    # Has sequence data – process like prep_rfam_seed
    if header_fields and has_seq_field:
        # Find the sequence column index
        seq_col = None
        for i, field in enumerate(header_fields):
            if any(kw in field for kw in seq_keywords):
                seq_col = i
                break
    else:
        seq_col = None

    rng = random.Random(seed)
    seen: set[str] = set()
    total_raw = 0
    kept = 0
    dropped_short = 0
    dropped_long = 0
    duplicate_count = 0
    lengths: list[int] = []
    families: set[str] = set()
    count = 0

    with (
        gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as fh,
        output_path.open("w", encoding="utf-8") as out_fh,
    ):
        # Read header if tabular
        first = fh.readline().rstrip("\n\r")
        if seq_col is None and "\t" in first:
            cols = [c.strip().lower() for c in first.split("\t")]
            for i, col in enumerate(cols):
                if any(kw in col for kw in seq_keywords):
                    seq_col = i
                    break
            # Re-probe rest of lines as data
        elif seq_col is None:
            # FASTA-like or other format – assume each line is a sequence
            pass

        # Determine family column index for tabular
        family_col = None
        if seq_col is not None:
            cols = first.split("\t") if "\t" in first else first.split(",")
            for i, col in enumerate(cols):
                c = col.strip().lower()
                if c in ("family", "rfam_acc", "accession", "rfam_family"):
                    family_col = i
                    break

        for line in fh:
            line = line.rstrip("\n\r")
            if not line.strip() or line.startswith("#"):
                continue

            if limit_val is not None and count >= int(limit_val):
                break

            if seq_col is not None:
                # Tabular: extract column
                parts = line.split("\t") if "\t" in first else line.split(",")
                if seq_col >= len(parts):
                    continue
                raw_seq = parts[seq_col].strip()
                family = parts[family_col].strip() if family_col is not None and family_col < len(parts) else "unknown"
            else:
                # Each line is a sequence
                raw_seq = line.strip()
                family = "unknown"

            if not raw_seq:
                continue

            total_raw += 1
            seq = raw_seq.upper().replace("T", "U")
            seq = "".join(c for c in seq if c not in "-.~")
            seq = "".join(c if c in "AUCGN" else "N" for c in seq)

            length = len(seq)
            if length < min_length:
                dropped_short += 1
                continue
            if length > max_length:
                dropped_long += 1
                continue
            if dedup:
                if seq in seen:
                    duplicate_count += 1
                    continue
                seen.add(seq)

            row = {
                "id": f"full_region_{count}",
                "seq": seq,
                "family": family,
                "source": "rfam_full_region",
            }
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            lengths.append(length)
            families.add(family)
            count += 1

    n_lengths = len(lengths)
    if n_lengths:
        mean_length = sum(lengths) / n_lengths
        sorted_l = sorted(lengths)
        p50 = sorted_l[n_lengths // 2]
        p95 = sorted_l[int(n_lengths * 0.95)]
    else:
        mean_length = p50 = p95 = 0

    stats = {
        "command": "prep_rfam_full_region",
        "input": str(input_path),
        "output": str(output_path),
        "total_raw": total_raw,
        "kept": kept,
        "dropped_short": dropped_short,
        "dropped_long": dropped_long,
        "duplicate_count": duplicate_count,
        "mean_length": round(mean_length, 1),
        "p50_length": p50,
        "p95_length": p95,
        "family_count": len(families),
    }

    inspect_dir = Path("outputs/rfam_inspect")
    inspect_dir.mkdir(parents=True, exist_ok=True)
    (inspect_dir / "rfam_full_region_prep_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


def split_seq_jsonl(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.out)
    train_ratio = float(getattr(args, "train_ratio", 0.9))
    val_ratio = float(getattr(args, "val_ratio", 0.1))
    limit_val = getattr(args, "limit", None)
    seed = int(getattr(args, "seed", 42))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read sequence-only JSONL
    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if "id" in row and "seq" in row:
                    rows.append(row)

    total = len(rows)

    # Apply optional cap
    if limit_val is not None and limit_val < total:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit_val]
        actual_limit_applied = limit_val
    else:
        actual_limit_applied = total

    n = len(rows)

    # Shuffle (or re-shuffle after limit)
    rng = random.Random(seed)
    rng.shuffle(rows)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if val_end > n:
        val_end = n

    train = rows[:train_end]
    val = rows[train_end:val_end]

    train_path = output_dir / "train_seq.jsonl"
    val_path = output_dir / "val_seq.jsonl"

    with train_path.open("w", encoding="utf-8") as fh:
        for row in train:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as fh:
        for row in val:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "command": "split_seq_jsonl",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total": total,
        "train_count": len(train),
        "val_count": len(val),
        "actual_limit_applied": actual_limit_applied,
    }

    inspect_dir = Path("outputs/rfam_inspect")
    inspect_dir.mkdir(parents=True, exist_ok=True)
    (inspect_dir / "rfam_seed_split_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


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

    # --- inspect_rfam ---
    ir = sub.add_parser("inspect_rfam", help="Probe Rfam file format (Stockholm/FASTA/metadata)")
    ir.add_argument("--input", required=True, help="Path to .gz file to inspect")
    ir.add_argument("--max_lines", type=int, default=100, help="Number of lines to sample (default: 100)")
    ir.set_defaults(func=inspect_rfam)

    # --- prep_rfam_seed ---
    prs = sub.add_parser("prep_rfam_seed", help="Parse Rfam.seed.gz (Stockholm) into sequence-only JSONL")
    prs.add_argument("--input", required=True, help="Path to Rfam.seed.gz")
    prs.add_argument("--output", required=True, help="Output sequence-only JSONL path")
    prs.add_argument("--min_length", type=int, default=20)
    prs.add_argument("--max_length", type=int, default=512)
    prs.add_argument("--dedup", action="store_true", help="Deduplicate by sequence")
    prs.add_argument("--seed", type=int, default=42)
    prs.set_defaults(func=prep_rfam_seed)

    # --- prep_rfam_full_region ---
    prfr = sub.add_parser("prep_rfam_full_region", help="Parse Rfam.full_region.gz (tabular metadata / sequences)")
    prfr.add_argument("--input", required=True, help="Path to Rfam.full_region.gz")
    prfr.add_argument("--output", required=True, help="Output sequence-only JSONL path")
    prfr.add_argument("--min_length", type=int, default=20)
    prfr.add_argument("--max_length", type=int, default=512)
    prfr.add_argument("--dedup", action="store_true", help="Deduplicate by sequence")
    prfr.add_argument("--limit", type=int, default=None, help="Max sequences to process")
    prfr.add_argument("--seed", type=int, default=42)
    prfr.set_defaults(func=prep_rfam_full_region)

    # --- split_seq_jsonl ---
    ssj = sub.add_parser("split_seq_jsonl", help="Split sequence-only JSONL into train/val")
    ssj.add_argument("--input", required=True, help="Input sequence-only JSONL")
    ssj.add_argument("--out", required=True, help="Output directory for train_seq.jsonl / val_seq.jsonl")
    ssj.add_argument("--train_ratio", type=float, default=0.9)
    ssj.add_argument("--val_ratio", type=float, default=0.1)
    ssj.add_argument("--limit", type=int, default=None, help="Cap total samples before split")
    ssj.add_argument("--seed", type=int, default=42)
    ssj.set_defaults(func=split_seq_jsonl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


