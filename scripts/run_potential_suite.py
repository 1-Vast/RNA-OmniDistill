from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def write_potential_from_sweep() -> None:
    summary_path = Path("outputs/sweep/summary.json")
    out = Path("outputs/potential/archiveii_quick")
    out.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        raise SystemExit("Sweep summary missing; cannot write potential report.")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    archive = next((row for row in rows if row.get("variant") == "archive"), rows[0])
    random_f1 = 0.0121
    metrics = {
        "test_pair_f1": archive.get("pair_f1", 0.0),
        "random_valid_pair_baseline_f1": random_f1,
        "all_dot_ratio": archive.get("all_dot", 0.0),
        "valid_structure_rate": archive.get("valid", 0.0),
        "pair_count_ratio": archive.get("pair_ratio", 0.0),
        "pair_logit_gap": archive.get("gap", 0.0),
        "pair_ranking_accuracy_sampled": archive.get("rankAcc"),
    }
    potential = "Medium" if metrics["test_pair_f1"] > random_f1 and metrics["all_dot_ratio"] < 0.3 else "Low"
    result = {"potential": potential, "metrics": metrics, "source": str(summary_path)}
    (out / "model_potential.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Model Potential Summary",
        "",
        f"Conclusion: **{potential}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        if value is None:
            lines.append(f"| {key} |  |")
        else:
            lines.append(f"| {key} | {float(value):.4f} |")
    (out / "model_potential.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for compact potential workflow.")
    parser.add_argument("--dataset", choices=["archiveii"], default="archiveii")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if args.mode != "quick":
        raise SystemExit("This compatibility wrapper only runs quick mode. Use `python scripts/run.py potential --set archive --mode full --device cuda` for full.")
    if not Path("dataset/archive/train.jsonl").exists() and Path("dataset/processed_archiveii").exists():
        Path("dataset/archive").mkdir(parents=True, exist_ok=True)
        for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            src = Path("dataset/processed_archiveii") / name
            if src.exists():
                (Path("dataset/archive") / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    if not Path("dataset/archive/train.jsonl").exists():
        raise SystemExit("ArchiveII split files are missing. Prepare data before running quick potential.")
    run([sys.executable, "scripts/run.py", "sweep", "--mode", "quick", "--device", args.device])
    write_potential_from_sweep()
    print("potential -> outputs/potential/archiveii_quick/model_potential.md")


if __name__ == "__main__":
    main()
