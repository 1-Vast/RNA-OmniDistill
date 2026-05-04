from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    ("pair_precision", "Pair Precision"),
    ("pair_recall", "Pair Recall"),
    ("pair_f1", "Pair F1"),
    ("valid_structure_rate", "Valid Rate"),
    ("canonical_pair_ratio", "Canonical Ratio"),
    ("all_dot_ratio", "All-dot Ratio"),
]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Benchmark JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length.")

    rows = []
    comparison = {}
    for name, input_path in zip(args.names, args.inputs):
        data = read_json(Path(input_path))
        comparison[name] = data
        for method, metrics in data.get("overall", {}).items():
            row = {"setting": name, "method": method}
            for key, _ in METRICS:
                row[key] = float(metrics.get(key, 0.0))
            rows.append(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps({"rows": rows, "raw": comparison}, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["setting", "method"] + [key for key, _ in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Setting | Method | Pair Precision | Pair Recall | Pair F1 | Valid Rate | Canonical Ratio | All-dot Ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["setting"],
                    row["method"],
                    fmt(row["pair_precision"]),
                    fmt(row["pair_recall"]),
                    fmt(row["pair_f1"]),
                    fmt(row["valid_structure_rate"]),
                    fmt(row["canonical_pair_ratio"]),
                    fmt(row["all_dot_ratio"]),
                ]
            )
            + " |"
        )
    (out_dir / "comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"comparison json -> {out_dir / 'comparison.json'}")
    print(f"comparison csv -> {out_dir / 'comparison.csv'}")
    print(f"comparison md -> {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()

