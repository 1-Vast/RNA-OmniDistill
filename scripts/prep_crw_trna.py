"""Preprocess CRW tRNA CT files into JSONL for RNA-OmniDiffusion."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.struct import pairs_to_dot_bracket, infer_simple_motifs

VALID_BASES = set("ACGUTN")

def read_ct_robust(path: Path) -> dict | None:
    """Read CT file with CRW and ArchiveII format compatibility."""
    text = path.read_text(errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    rows = []
    for line in lines[1:]:  # skip first line (length + filename or "Filename: ...")
        parts = line.split()
        if len(parts) < 5:
            continue
        if not parts[0].isdigit():
            continue
        # Check column 2 is a valid nucleotide base
        if len(parts) >= 2 and parts[1].upper() not in VALID_BASES:
            continue
        rows.append(parts)

    if not rows:
        return None

    seq = "".join(row[1].upper().replace("T", "U") for row in rows)
    pairs = []
    for row in rows:
        i = int(row[0]) - 1
        try:
            j = int(row[4]) - 1
        except ValueError:
            continue
        if j > i:
            pairs.append((i, j))

    struct = pairs_to_dot_bracket(pairs, len(seq))
    family = path.parent.name if path.parent.name != "CRW_tRNA" else path.stem.split(".")[0] if "." in path.stem else path.stem

    return {
        "id": path.stem,
        "seq": seq,
        "struct": struct,
        "family": family,
        "pairs": pairs,
        "length": len(seq),
        "motifs": infer_simple_motifs(struct=struct),
    }


def main():
    raw_dir = Path(r"D:\RNA-OmniDiffusion\dataset\raw\CRW_tRNA")
    out_dir = Path(r"D:\RNA-OmniDiffusion\dataset\processed\crw_trna")
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(raw_dir.rglob("*.ct"))
    print(f"Found {len(ct_files)} .ct files")

    out_path = out_dir / "clean.jsonl"
    count = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for path in ct_files:
            try:
                row = read_ct_robust(path)
                if row is None or row["length"] > 512:
                    skipped += 1
                    continue
                handle.write(json.dumps(row) + "\n")
                count += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  SKIP {path.name}: {e}")

    print(f"Prepared: {count} sequences")
    print(f"Skipped:  {skipped}")
    print(f"Output:   {out_path}")

    # Quick stats
    lens = []
    families = set()
    with out_path.open() as f:
        for line in f:
            row = json.loads(line)
            lens.append(row["length"])
            families.add(row["family"])
    print(f"Length range: {min(lens)}-{max(lens)}")
    print(f"Families: {len(families)} unique")
    print(f"Families sample: {sorted(list(families))[:10]}...")


if __name__ == "__main__":
    main()
