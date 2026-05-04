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
    ("avg_pred_pair_count", "Avg Pred Pairs"),
]


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare core RNA-OmniDiffusion benchmark results.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length.")

    rows = []
    for name, input_path in zip(args.names, args.inputs):
        path = Path(input_path)
        if not path.exists():
            raise SystemExit(f"Benchmark JSON not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("overall", {}).get("model", {})
        row = {"setting": name}
        for key, _ in METRICS:
            row[key] = float(model.get(key, 0.0))
        rows.append(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "core_results.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "core_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["setting"] + [key for key, _ in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Setting | Pair Precision | Pair Recall | Pair F1 | Valid Rate | Canonical Ratio | All-dot Ratio | Avg Pred Pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["setting"],
                    fmt(row["pair_precision"]),
                    fmt(row["pair_recall"]),
                    fmt(row["pair_f1"]),
                    fmt(row["valid_structure_rate"]),
                    fmt(row["canonical_pair_ratio"]),
                    fmt(row["all_dot_ratio"]),
                    fmt(row["avg_pred_pair_count"]),
                ]
            )
            + " |"
        )
    (out_dir / "core_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"core results json -> {out_dir / 'core_results.json'}")
    print(f"core results csv -> {out_dir / 'core_results.csv'}")
    print(f"core results md -> {out_dir / 'core_results.md'}")


if __name__ == "__main__":
    main()

