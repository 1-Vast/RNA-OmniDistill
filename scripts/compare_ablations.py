from __future__ import annotations

import argparse
import csv
import glob
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


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def variant_name(path: Path) -> str:
    if path.parent.name:
        return path.parent.name
    return path.stem


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ablation benchmark JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    missing = []
    for path in expand_inputs(args.inputs):
        if not path.exists():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("overall", {}).get("model", {})
        row = {"variant": variant_name(path), "path": str(path)}
        for key, _ in METRICS:
            row[key] = float(metrics.get(key, 0.0))
        rows.append(row)
    if not rows:
        raise SystemExit("No benchmark inputs found. Missing: " + ", ".join(missing))

    best_f1 = max(row["pair_f1"] for row in rows)
    lowest_all_dot = min(row["all_dot_ratio"] for row in rows)
    highest_valid = max(row["valid_structure_rate"] for row in rows)
    for row in rows:
        row["best_pair_f1"] = row["pair_f1"] == best_f1
        row["lowest_all_dot_ratio"] = row["all_dot_ratio"] == lowest_all_dot
        row["highest_valid_structure_rate"] = row["valid_structure_rate"] == highest_valid

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "missing": missing}, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["variant"] + [key for key, _ in METRICS] + [
            "best_pair_f1",
            "lowest_all_dot_ratio",
            "highest_valid_structure_rate",
            "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Variant | Pair Precision | Pair Recall | Pair F1 | Valid Rate | Canonical Ratio | All-dot Ratio | Avg Pred Pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["variant"]):
        name = row["variant"]
        if row["best_pair_f1"]:
            name = f"**{name}**"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
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
    lines.append("")
    lines.append(f"Best Pair F1: {fmt(best_f1)}")
    lines.append(f"Lowest all-dot ratio: {fmt(lowest_all_dot)}")
    lines.append(f"Highest valid structure rate: {fmt(highest_valid)}")
    if missing:
        lines.append("")
        lines.append("Missing inputs:")
        lines.extend(f"- {item}" for item in missing)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary json -> {out_dir / 'summary.json'}")
    print(f"summary csv -> {out_dir / 'summary.csv'}")
    print(f"summary md -> {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

